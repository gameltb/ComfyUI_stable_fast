import gc
import hashlib
import os
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, List

import torch as th
from torch.cuda import nvtx

import comfy.ldm.modules.diffusionmodules.openaimodel
import comfy.model_management
import comfy.model_patcher
import comfy.cldm.cldm
import comfy.gligen

from .comfy_trace_utilities import hash_arg
from .tensorrt_utilities import Engine


@dataclass
class TensorRTEngineConfig:
    enable_cuda_graph: bool
    keep_width: int = 768
    keep_height: int = 768
    keep_batch_size: int = 2
    keep_embedding_block: int = 2


class CallableTensorRTEngineWrapper:
    def __init__(self, tensorrt_context, identification) -> None:
        self.tensorrt_context: TensorRTEngineContext = tensorrt_context
        self.identification = identification + self.__class__.__name__

        self.engine: Engine = None
        self.onnx_cache = None
        self.input_shape_info = None
        self.input_profile_info = None

        self.engine_comfy_model_patcher_wrapper = None
        self.device_memory_size = 0

        self.engine_cache_map = {}

    def gen_onnx_args(self, kwargs):
        args = []
        args_name = []
        for arg_name, arg in kwargs.items():
            args.append(arg)
            if arg != None:
                args_name.append(arg_name)

        return args, args_name, None

    def gen_onnx_outputs(self, module):
        return ["output"]

    def gen_tensorrt_args(self, kwargs):
        input_shape_info = {}
        feed_dict = {}
        for arg_name, arg in kwargs.items():
            if arg != None:
                feed_dict[arg_name] = arg
                input_shape_info[arg_name] = tuple(arg.shape)

        return feed_dict, input_shape_info

    def gen_tensorrt_args_profile(self, input_shape_info):
        return {k: [v, v, v] for k, v in input_shape_info.items()}

    def gen_tensorrt_outputs(self, output):
        return output["output"]

    def is_profile_compatible(self, input_profile_info, input_shape_info):
        if input_profile_info == None:
            return False
        if len(input_profile_info) != len(input_shape_info):
            return False
        for arg_name, shape in input_shape_info.items():
            profile = input_profile_info.get(arg_name, None)
            if profile == None:
                return False
            if len(profile[0]) != len(shape):
                return False
            for d, mind, maxd in zip(shape, profile[0], profile[2]):
                if d < mind or d > maxd:
                    return False
        return True

    def __call__(self, module, /, **kwargs: Any) -> Any:
        feed_dict, input_shape_info = self.gen_tensorrt_args(kwargs)

        if self.engine == None or not self.is_profile_compatible(
            self.input_profile_info, input_shape_info
        ):
            self.input_shape_info = input_shape_info
            input_profile_info = self.gen_tensorrt_args_profile(input_shape_info)

            engine_cache_key = (
                hash_arg(self.tensorrt_context.unet_config),
                hash_arg(self.identification),
                hash_arg(input_profile_info),
            )

            if engine_cache_key in self.engine_cache_map:
                (
                    self.engine,
                    self.engine_comfy_model_patcher_wrapper,
                ) = self.engine_cache_map[engine_cache_key]
                self.input_profile_info = input_profile_info
            else:
                engine = get_engine_with_cache(
                    engine_cache_key, TensorRTEngineConfig(enable_cuda_graph=False)
                )

                if self.onnx_cache == None:
                    module.to(device=self.tensorrt_context.cuda_device)
                    args, args_name, dynamic_axes = self.gen_onnx_args(kwargs)
                    self.onnx_cache = BytesIO()
                    try:
                        th.onnx.export(
                            module,
                            tuple(args),
                            self.onnx_cache,
                            export_params=True,
                            verbose=False,
                            do_constant_folding=True,
                            input_names=args_name,
                            output_names=self.gen_onnx_outputs(module),
                            dynamic_axes=dynamic_axes,
                        )
                    except Exception as e:
                        self.onnx_cache = None
                        raise e

                nvtx.range_push("offload origin model")
                module.to(device="cpu")
                gc.collect()
                comfy.model_management.soft_empty_cache()
                nvtx.range_pop()

                if engine == None:
                    comfy.model_management.free_memory(
                        6 * 1024 * 1024 * 1024, self.tensorrt_context.cuda_device
                    )
                    engine = gen_engine(
                        engine_cache_key,
                        self.onnx_cache.getvalue(),
                        [input_profile_info],
                        self.tensorrt_context.dtype,
                    )
                    self.onnx_cache.seek(0)
                    engine.refit_simple(self.onnx_cache, reset_zero=True)
                    engine.save_engine()

                self.engine = engine
                try:
                    nvtx.range_push("load engine")
                    if self.engine.engine == None:
                        self.engine.load()
                    self.device_memory_size = self.engine.engine.device_memory_size
                    nvtx.range_push("refit engine")
                    self.onnx_cache.seek(0)
                    self.engine.refit_simple(self.onnx_cache)
                    nvtx.range_pop()
                    self.engine.activate(True)
                    self.engine_comfy_model_patcher_wrapper = (
                        TensorRTEngineComfyModelPatcherWrapper(
                            engine,
                            load_device=self.tensorrt_context.cuda_device,
                            offload_device="cpu",
                            size=self.device_memory_size,
                        )
                    )
                    comfy.model_management.load_models_gpu(
                        [
                            *self.tensorrt_context.keep_models,
                            self.engine_comfy_model_patcher_wrapper,
                            *get_additional_keep_models(),
                        ],
                        self.device_memory_size,
                    )
                    self.input_profile_info = input_profile_info
                    self.engine_cache_map[engine_cache_key] = (
                        self.engine,
                        self.engine_comfy_model_patcher_wrapper,
                    )
                    nvtx.range_pop()
                except Exception as e:
                    self.engine = None
                    raise e

        if self.engine.engine == None:
            comfy.model_management.load_models_gpu(
                [
                    *self.tensorrt_context.keep_models,
                    self.engine_comfy_model_patcher_wrapper,
                    *get_additional_keep_models(),
                ],
                self.device_memory_size,
            )

        self.engine.allocate_buffers(
            feed_dict,
            device=self.tensorrt_context.cuda_device,
            allocate_input_buffers=False,
        )
        output = self.engine.infer(feed_dict, self.tensorrt_context.cuda_stream)
        output = self.gen_tensorrt_outputs(output)
        self.engine.release_buffers()

        return output


class TensorRTEngineComfyModelPatcherWrapper(comfy.model_patcher.ModelPatcher):
    def patch_model(self, device_to=None):
        if device_to is not None:
            if self.model.engine == None:
                self.model.load()
                self.model.activate(True)
            self.current_device = device_to

        return self.model

    def unpatch_model(self, device_to=None):
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
    model_type: str = ""
    keep_models: List = field(default_factory=lambda: [])
    dtype: object = th.float16


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


def get_engine_path(key):
    engine_cache_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "tensorrt_engine_cache"
    )
    if not os.path.exists(engine_cache_dir):
        os.makedirs(engine_cache_dir, exist_ok=True)
    basename = hashlib.sha256(str(key).encode()).hexdigest()
    return os.path.join(engine_cache_dir, basename + ".trt")


def get_engine_with_cache(key, config):
    engine_path = get_engine_path(key)
    if os.path.exists(engine_path):
        return Engine(engine_path, enable_cuda_graph=config.enable_cuda_graph)
    return None


def gen_engine(key, onnx_buff, input_profile, dtype):
    engine = Engine(get_engine_path(key))
    s = time.time()
    engine.build(
        onnx_buff,
        dtype=dtype,
        enable_refit=True,
        timing_cache=TIMING_CACHE_PATH,
        input_profile=input_profile,
    )
    e = time.time()
    print(f"Time taken to build: {e-s}s")
    return engine
