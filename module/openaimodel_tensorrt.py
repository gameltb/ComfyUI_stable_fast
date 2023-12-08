import gc
import hashlib
import yaml
import os
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict

import torch as th
from torch.cuda import nvtx

import comfy.ldm.modules.diffusionmodules.openaimodel
import comfy.model_management
import comfy.model_patcher
from comfy.ldm.modules.diffusionmodules.openaimodel import forward_timestep_embed

from .comfy_trace_utilities import hash_arg
from .tensorrt_utilities import Engine

TENSORRT_CONTEXT_KEY = "tensorrt_context"

origin_forward_timestep_embed = forward_timestep_embed


@dataclass
class TensorRTEngineConfig:
    enable_cuda_graph: bool


class CallableTensorRTEngineWarper:
    def __init__(self, tensorrt_context, identification) -> None:
        self.tensorrt_context: TensorRTEngineCacheContext = tensorrt_context
        self.identification = identification + self.__class__.__name__

        self.engine: Engine = None
        self.onnx_cache = None
        self.input_shape_info = None
        self.input_profile_info = None

        self.engine_comfy_model_patcher_warper = None
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
                    self.engine_comfy_model_patcher_warper,
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
                            output_names=["output"],
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
                    engine = gen_engine(
                        engine_cache_key,
                        self.onnx_cache.getvalue(),
                        [input_profile_info],
                    )
                    self.onnx_cache.seek(0)
                    engine.refit_simple(self.onnx_cache, reset_zero=True)
                    engine.save_engine()
                    del engine
                    engine = get_engine_with_cache(
                        engine_cache_key, TensorRTEngineConfig(enable_cuda_graph=False)
                    )

                self.engine = engine
                try:
                    nvtx.range_push("load engine")
                    self.engine.load()
                    self.device_memory_size = self.engine.engine.device_memory_size
                    self.engine.unload()
                    self.engine_comfy_model_patcher_warper = (
                        TensorRTEngineComfyModelPatcherWarper(
                            engine,
                            load_device=self.tensorrt_context.cuda_device,
                            offload_device="cpu",
                            size=self.device_memory_size,
                        )
                    )
                    comfy.model_management.load_models_gpu(
                        [
                            self.tensorrt_context.origin_model_patcher,
                            self.engine_comfy_model_patcher_warper,
                        ],
                        self.device_memory_size,
                    )
                    nvtx.range_push("refit engine")
                    self.onnx_cache.seek(0)
                    self.engine.refit_simple(self.onnx_cache)
                    nvtx.range_pop()
                    self.input_profile_info = input_profile_info
                    self.engine_cache_map[engine_cache_key] = (
                        self.engine,
                        self.engine_comfy_model_patcher_warper,
                    )
                    nvtx.range_pop()
                except Exception as e:
                    self.engine = None
                    raise e

        if self.engine.engine == None:
            comfy.model_management.load_models_gpu(
                [
                    self.tensorrt_context.origin_model_patcher,
                    self.engine_comfy_model_patcher_warper,
                ],
                self.device_memory_size,
            )

        self.engine.allocate_buffers(feed_dict)
        return self.engine.infer(feed_dict, self.tensorrt_context.cuda_stream)["output"]


class CallableTensorRTEngineWarperDynamicShapeForwardTimestep(
    CallableTensorRTEngineWarper
):
    args_name = [
        "x",
        "emb",
        "context",
        "output_shape_tensor",
        "time_context",
        "image_only_indicator",
    ]

    def gen_onnx_args(self, kwargs):
        args_name = []
        args = []
        for arg_name in self.args_name:
            args.append(kwargs.get(arg_name, None))
            if args[-1] != None:
                args_name.append(arg_name)
        dynamic_axes = {
            "x": {0: "B", 2: "H", 3: "W"},
            "emb": {0: "B"},
            "context": {0: "B", 1: "E"},
            "output_shape_tensor": {0: "B", 2: "OH", 3: "OW"},
        }
        for k in list(dynamic_axes.keys()):
            if not k in args_name:
                dynamic_axes.pop(k)
        return args, args_name, dynamic_axes

    def gen_tensorrt_args(self, kwargs):
        input_shape_info = {}
        feed_dict = {}
        for arg_name in self.args_name:
            arg = kwargs.get(arg_name, None)
            if arg != None:
                feed_dict[arg_name] = arg
                input_shape_info[arg_name] = tuple(arg.shape)

        return feed_dict, input_shape_info

    def gen_tensorrt_args_profile(self, input_shape_info):
        min_input_profile_info = {
            "x": {0: 1, 2: 1, 3: 1},
            "emb": {0: 1},
            "context": {0: 1, 1: 77},
            "output_shape_tensor": {0: 1, 2: 1, 3: 1},
        }
        input_profile_info = {}
        for arg_name in self.args_name:
            shape_info = input_shape_info.get(arg_name, None)
            min_shape_config = min_input_profile_info.get(arg_name, None)
            if shape_info != None:
                min_shape_info = list(shape_info)
                if min_shape_config != None:
                    for k, v in min_shape_config.items():
                        min_shape_info[k] = v
                input_profile_info[arg_name] = [
                    tuple(min_shape_info),
                    shape_info,
                    shape_info,
                ]

        return input_profile_info


class TensorRTEngineComfyModelPatcherWarper(comfy.model_patcher.ModelPatcher):
    def patch_model(self, device_to=None):
        if device_to is not None:
            self.model.load()
            self.model.activate(True)
            # self.model.to(device_to)
            self.current_device = device_to

        return self.model

    def unpatch_model(self, device_to=None):
        if device_to is not None:
            self.model.offload()
            # self.model.to(device_to)
            self.current_device = device_to


@dataclass
class TensorRTEngineCacheContext:
    block_cache: Dict[str, CallableTensorRTEngineWarper] = field(
        default_factory=lambda: {}
    )
    cuda_device = None
    cuda_stream = None
    unet_config: dict = None
    model_type: str = ""
    origin_model_patcher: comfy.model_patcher.ModelPatcher = None

    def dump_input_profile_info(self):
        input_shape_info_map = {}
        for key in sorted(self.block_cache):
            input_shape_info_map[key] = self.block_cache[key].input_shape_info
        print(yaml.safe_dump(input_shape_info_map))


class ForwardTimestepEmbedModule(th.nn.Module):
    def __init__(self, ts, transformer_options={}, num_video_frames=None):
        super().__init__()
        self.module = ts
        self.transformer_options = transformer_options
        self.num_video_frames = num_video_frames

    def forward(
        self,
        x,
        emb,
        context=None,
        output_shape_tensor=None,
        time_context=None,
        image_only_indicator=None,
    ):
        return origin_forward_timestep_embed(
            self.module,
            x,
            emb,
            context=context,
            transformer_options=self.transformer_options,
            output_shape=output_shape_tensor
            if output_shape_tensor == None
            else output_shape_tensor.shape,
            time_context=time_context,
            num_video_frames=self.num_video_frames,
            image_only_indicator=image_only_indicator,
        )


def hook_forward_timestep_embed(
    ts,
    x,
    emb,
    context=None,
    transformer_options={},
    output_shape=None,
    time_context=None,
    num_video_frames=None,
    image_only_indicator=None,
):
    module = ForwardTimestepEmbedModule(ts, transformer_options, num_video_frames)
    tensorrt_context: TensorRTEngineCacheContext = transformer_options.get(
        TENSORRT_CONTEXT_KEY, None
    )
    if tensorrt_context != None:
        block_key = str(transformer_options["block"])
        block = tensorrt_context.block_cache.get(block_key, None)
        if block == None:
            tensorrt_context.block_cache[
                block_key
            ] = CallableTensorRTEngineWarperDynamicShapeForwardTimestep(
                tensorrt_context, block_key
            )
        return tensorrt_context.block_cache[block_key](
            module,
            x=x,
            emb=emb,
            context=context,
            output_shape_tensor=output_shape
            if output_shape == None
            else th.empty((output_shape), device=x.device, dtype=x.dtype),
            time_context=time_context,
            image_only_indicator=image_only_indicator,
        )
    return module(x, emb, context, time_context, image_only_indicator)


def do_hook_forward_timestep_embed():
    comfy.ldm.modules.diffusionmodules.openaimodel.forward_timestep_embed = (
        hook_forward_timestep_embed
    )


def undo_hook_forward_timestep_embed():
    comfy.ldm.modules.diffusionmodules.openaimodel.forward_timestep_embed = (
        origin_forward_timestep_embed
    )


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


def gen_engine(key, onnx_buff, input_profile):
    engine = Engine(get_engine_path(key))
    s = time.time()
    engine.build(
        onnx_buff,
        fp16=True,
        enable_refit=True,
        timing_cache=TIMING_CACHE_PATH,
        input_profile=input_profile,
    )
    e = time.time()
    print(f"Time taken to build: {e-s}s")
    return engine
