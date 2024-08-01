import gc
import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, List

import comfy.cldm.cldm
import comfy.gligen
import comfy.ldm.modules.diffusionmodules.openaimodel
import comfy.model_management
import comfy.model_patcher
import numpy
import safetensors
import safetensors.torch
import tensorrt
import torch
import torch.version
from torch.cuda import nvtx

from .comfy_trace_utilities import hash_arg
from .onnx_module_refit import (
    make_constant_params_dict_by_onnx_model,
    make_module_onnx_tensor_gen_map_by_params_dict,
    make_params_dict_by_module,
)
from .tensorrt_utilities import Engine

_logger = logging.getLogger(__name__)


@dataclass
class TensorRTEngineConfig:
    enable_cuda_graph: bool
    keep_width: int = 768
    keep_height: int = 768
    keep_batch_size: int = 2
    keep_embedding_block: int = 2
    use_dedicated_engine: bool = False


class CallableTensorRTEngineWrapper:
    def __init__(self, tensorrt_context, identification) -> None:
        self.tensorrt_context: TensorRTEngineContext = tensorrt_context
        self.identification = identification + self.__class__.__name__

        self.engine: Engine = None
        self.onnx_cache_dir = None
        self.onnx_cache = None
        self.onnx_refit_info = None

        self.module_identification = None
        self.input_shape_info = None
        self.input_profile_info = None

        self.engine_comfy_model_patcher_wrapper = None

        self.engine_cache_map = {}

    def gen_onnx_args(self, kwargs, module=None):
        args = []
        args_name = []
        for arg_name, arg in kwargs.items():
            args.append(arg)
            if arg is not None:
                args_name.append(arg_name)

        return args, args_name, None

    def gen_onnx_outputs(self, module):
        return ["output"]

    def gen_tensorrt_args(self, kwargs):
        input_shape_info = {}
        feed_dict = {}
        for arg_name, arg in kwargs.items():
            if arg is not None:
                feed_dict[arg_name] = arg
                input_shape_info[arg_name] = tuple(arg.shape)

        return feed_dict, input_shape_info

    def gen_tensorrt_args_profile(self, input_shape_info):
        return {k: [v, v, v] for k, v in input_shape_info.items()}

    def gen_tensorrt_outputs(self, output):
        return output["output"]

    def is_profile_compatible(self, input_profile_info, input_shape_info):
        if input_profile_info is None:
            return False
        if len(input_profile_info) != len(input_shape_info):
            return False
        for arg_name, shape in input_shape_info.items():
            profile = input_profile_info.get(arg_name, None)
            if profile is None:
                return False
            if len(profile[0]) != len(shape):
                return False
            for d, mind, maxd in zip(shape, profile[0], profile[2]):
                if d < mind or d > maxd:
                    return False
        return True

    def __call__(self, module: torch.nn.Module, /, **kwargs: Any) -> Any:
        feed_dict, input_shape_info = self.gen_tensorrt_args(kwargs)

        if self.engine is None or not self.is_profile_compatible(
            self.input_profile_info, input_shape_info
        ):
            self.input_shape_info = input_shape_info
            input_profile_info = self.gen_tensorrt_args_profile(input_shape_info)

            if self.tensorrt_context.identify_weight_hash:
                if self.module_identification is None:
                    self.module_identification = sha256sum_state_dict(
                        module.state_dict()
                    )

            engine_cache_key = (
                hash_arg(torch.version.__version__),
                hash_arg(tensorrt.__version__),
                hash_arg(self.tensorrt_context.unet_config),
                hash_arg(self.identification),
                hash_arg(input_profile_info),
                hash_arg(self.tensorrt_context.enable_weight_streaming),
                hash_arg(str(self.tensorrt_context.model_sampling_type)),
                hash_arg(str(self.module_identification)),
            )

            if engine_cache_key in self.engine_cache_map:
                (
                    self.engine,
                    self.engine_comfy_model_patcher_wrapper,
                ) = self.engine_cache_map[engine_cache_key]
                self.input_profile_info = input_profile_info
            else:
                engine = get_engine_with_cache(engine_cache_key)

                args, args_name, dynamic_axes = self.gen_onnx_args(
                    kwargs, module=module
                )

                onnx_cache_key = (
                    hash_arg(torch.version.__version__),
                    hash_arg(self.tensorrt_context.unet_config),
                    hash_arg(self.identification),
                    hash_arg((args_name, dynamic_axes)),
                    hash_arg(str(self.tensorrt_context.model_sampling_type)),
                    hash_arg(str(self.module_identification)),
                )
                self.onnx_refit_info = get_refit_info_cache(onnx_cache_key)

                if (
                    (engine is None)
                    or (self.onnx_refit_info is None)
                    or (not self.tensorrt_context.enable_fast_refit)
                ) and self.onnx_cache is None:
                    module.to(device=self.tensorrt_context.cuda_device)
                    self.onnx_cache_dir = tempfile.TemporaryDirectory(
                        suffix="onnx_cache_dir"
                    )
                    self.onnx_cache = os.path.join(
                        self.onnx_cache_dir.name, "onnx_cache.onnx"
                    )
                    try:
                        use_patched_export = False
                        # only change is just make its export funtion return onnx params_dict
                        if torch.version.__version__ == "2.4.0":
                            from .patched_onnx_export.utils_2_4_0 import (
                                export as patched_export,
                            )

                            use_patched_export = True
                        if use_patched_export:
                            torch_out, params_dict = patched_export(
                                module,
                                tuple(args),
                                self.onnx_cache,
                                export_params=True,
                                verbose=False,
                                do_constant_folding=False,
                                input_names=args_name,
                                output_names=self.gen_onnx_outputs(module),
                                dynamic_axes=dynamic_axes,
                                # dynamo=True
                            )
                            if self.tensorrt_context.enable_fast_refit:
                                self.onnx_refit_info = gen_refit_info(onnx_cache_key)
                                self.onnx_refit_info.tensor_gen_map = (
                                    make_module_onnx_tensor_gen_map_by_params_dict(
                                        module, params_dict
                                    )
                                )
                                self.onnx_refit_info.constant_params_dict = (
                                    make_constant_params_dict_by_onnx_model(
                                        self.onnx_cache
                                    )
                                )
                                self.onnx_refit_info.save()
                            del params_dict
                        else:
                            torch.onnx.export(
                                module,
                                tuple(args),
                                self.onnx_cache,
                                export_params=True,
                                verbose=False,
                                do_constant_folding=False,
                                input_names=args_name,
                                output_names=self.gen_onnx_outputs(module),
                                dynamic_axes=dynamic_axes,
                            )
                    except Exception as e:
                        self.onnx_cache_dir.cleanup()
                        self.onnx_cache_dir = None
                        self.onnx_cache = None
                        self.onnx_refit_info = None
                        raise e

                nvtx.range_push("offload origin model")
                module.to(device="cpu")
                gc.collect()
                comfy.model_management.soft_empty_cache()
                nvtx.range_pop()

                additional_keep_models = []
                if engine is None:
                    additional_keep_models = get_additional_keep_models()
                    comfy.model_management.free_memory(
                        6 * 1024 * 1024 * 1024,
                        self.tensorrt_context.cuda_device,
                    )
                    comfy.model_management.soft_empty_cache()
                    engine = gen_engine(
                        engine_cache_key,
                        self.onnx_cache,
                        [input_profile_info],
                        self.tensorrt_context.dtype,
                        enable_weight_streaming=self.tensorrt_context.enable_weight_streaming,
                    )
                    engine.save_engine()

                self.engine = engine
                try:
                    nvtx.range_push("load engine")
                    if self.engine.engine is None:
                        self.engine.load()

                    # reserve some memory for pytorch
                    memory_limit_size = int(comfy.model_management.get_total_memory() - (
                        1024 * 1024 * 1024 * 2
                    ))

                    self.engine.activate(
                        True,
                        self.tensorrt_context.lowvram_model_memory
                        if memory_limit_size
                        > self.tensorrt_context.lowvram_model_memory
                        else memory_limit_size,
                    )
                    nvtx.range_push("refit engine")
                    if (
                        self.tensorrt_context.enable_fast_refit
                        and self.onnx_refit_info is not None
                    ):
                        _logger.info("using fast refit")
                        self.engine.refit_from_dict(
                            make_params_dict_by_module(
                                module, self.onnx_refit_info.tensor_gen_map
                            ),
                            self.onnx_refit_info.constant_params_dict,
                        )
                    else:
                        self.engine.refit_simple(self.onnx_cache)
                    nvtx.range_pop()
                    self.engine_comfy_model_patcher_wrapper = (
                        TensorRTEngineComfyModelPatcherWrapper(
                            engine,
                            load_device=self.tensorrt_context.cuda_device,
                            offload_device="cpu",
                            size=self.engine.get_device_memory_size(),
                        )
                    )
                    comfy.model_management.load_models_gpu(
                        [
                            *self.tensorrt_context.keep_models,
                            self.engine_comfy_model_patcher_wrapper,
                            *get_additional_keep_models(),
                            *additional_keep_models,
                        ],
                        self.engine.get_device_memory_size(),
                    )
                    self.input_profile_info = input_profile_info
                    self.engine_cache_map[engine_cache_key] = (
                        self.engine,
                        self.engine_comfy_model_patcher_wrapper,
                    )
                    nvtx.range_pop()
                except Exception as e:
                    self.engine = None
                    gc.collect()
                    raise e

        if self.engine.context is None:
            comfy.model_management.load_models_gpu(
                [
                    *self.tensorrt_context.keep_models,
                    self.engine_comfy_model_patcher_wrapper,
                    *get_additional_keep_models(),
                ],
                self.engine.get_device_memory_size(),
            )

        self.engine.allocate_buffers(
            feed_dict,
            device=self.tensorrt_context.cuda_device,
            allocate_input_buffers=False,
        )

        output = self.engine.infer(
            feed_dict,
            self.tensorrt_context.cuda_stream,
            self.tensorrt_context.infer_cuda_stream_sync,
        )
        output = self.gen_tensorrt_outputs(output)
        self.engine.release_buffers()

        return output


class TensorRTEngineComfyModelPatcherWrapper(comfy.model_patcher.ModelPatcher):
    def patch_model_lowvram(self, device_to=None, *arg, **kwargs):
        self.patch_model(device_to, patch_weights=False)

    def patch_model(self, device_to=None, *arg, **kwargs):
        if device_to is not None:
            if self.model.engine is None:
                self.model.load()
            if self.model.context is None:
                self.model.activate(True, self.model.last_device_memory_size)
            self.current_device = device_to

        return self.model

    def unpatch_model(self, device_to=None, *arg, **kwargs):
        if device_to is not None:
            self.model.offload()
            self.current_device = device_to


def get_additional_keep_models():
    models = []
    for model in comfy.model_management.current_loaded_models:
        if isinstance(
            model.real_model, (comfy.cldm.cldm.ControlNet, comfy.gligen.Gligen)
        ):
            models.append(model.model)
    return models


@dataclass
class TensorRTEngineContext:
    cuda_device = None
    shared_device_memory = None
    cuda_stream = None
    unet_config: dict = None
    model_sampling_type = None
    model_type: str = ""
    keep_models: List = field(default_factory=lambda: [])
    dtype: object = torch.float16
    enable_weight_streaming: bool = False
    enable_fast_refit: bool = True
    infer_cuda_stream_sync: bool = False
    identify_weight_hash: bool = False
    lowvram_model_memory = 0


TIMING_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "tensorrt_engine_cache",
    "timing_cache.cache",
)
if not os.path.exists(TIMING_CACHE_PATH):
    os.makedirs(os.path.dirname(TIMING_CACHE_PATH), exist_ok=True)
    with open(TIMING_CACHE_PATH, "wb") as f:
        pass


def get_key_hash(key):
    return hashlib.sha256(str(key).encode()).hexdigest()


def get_cache_path(key, dir_name):
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), dir_name)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    basename = get_key_hash(key)
    return os.path.join(cache_dir, basename)


def get_engine_path(key):
    return get_cache_path(key, "tensorrt_engine_cache") + ".trt"


def get_engine_with_cache(key):
    engine_path = get_engine_path(key)
    if os.path.exists(engine_path):
        return Engine(engine_path)
    return None


def gen_engine(key, onnx_model, input_profile, dtype, enable_weight_streaming=False):
    engine = Engine(get_engine_path(key))
    s = time.time()
    engine.build(
        onnx_model,
        dtype=dtype,
        enable_refit=True,
        timing_cache=TIMING_CACHE_PATH,
        input_profile=input_profile,
        enable_weight_streaming=enable_weight_streaming,
    )
    e = time.time()
    _logger.info(f"Time taken to build: {e-s}s")
    return engine


def get_refit_info_cache(key):
    refit_info_path = get_cache_path(key, "refit_info") + ".st"
    if os.path.exists(refit_info_path):
        return TorchTensorRTRefitInfo(refit_info_path).load()
    return None


def gen_refit_info(key):
    refit_info_path = get_cache_path(key, "refit_info") + ".st"
    return TorchTensorRTRefitInfo(refit_info_path)


class TorchTensorRTRefitInfo:
    def __init__(self, info_path) -> None:
        self.info_path = info_path
        self.tensor_gen_map = None
        self.constant_params_dict = None

    def save(self):
        safetensors.torch.save_file(
            self.constant_params_dict,
            self.info_path,
            metadata={"tensor_gen_map": json.dumps(self.tensor_gen_map)},
        )

    def load(self):
        self.constant_params_dict = safetensors.torch.load_file(self.info_path)
        with safetensors.safe_open(self.info_path, "torch") as st:
            if st.metadata() is not None:
                self.tensor_gen_map = json.loads(st.metadata()["tensor_gen_map"])
        return self


def sha256sum_state_dict(state_dict: dict[str, torch.Tensor]):
    hasher = hashlib.sha256()

    for k, v in state_dict.items():
        tensor_bytes = v.cpu().detach().numpy().astype(numpy.float16).data.tobytes()
        hasher.update(tensor_bytes)

    return hasher.hexdigest()
