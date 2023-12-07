import gc
import hashlib
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
        self.identification = identification

        self.engine: Engine = None
        self.onnx_cache = None
        self.input_shape_info = None

        self.engine_comfy_model_patcher_warper = None
        self.device_memory_size = 0

        self.engine_cache_map = {}

    def __call__(self, module, /, **kwargs: Any) -> Any:
        args = []
        args_name = []
        input_shape_info = {}
        feed_dict = {}
        for arg_name, arg in kwargs.items():
            args.append(arg)
            args_name.append(arg_name)
            if arg != None:
                feed_dict[arg_name] = arg
                input_shape_info[arg_name] = [
                    tuple(arg.shape),
                    tuple(arg.shape),
                    tuple(arg.shape),
                ]

        if self.engine == None or self.input_shape_info != input_shape_info:
            engine_cache_key = (
                hash_arg(self.tensorrt_context.unet_config),
                hash_arg(self.identification),
                hash_arg(input_shape_info),
            )

            if engine_cache_key in self.engine_cache_map:
                (
                    self.engine,
                    self.engine_comfy_model_patcher_warper,
                ) = self.engine_cache_map[engine_cache_key]
                self.input_shape_info = input_shape_info
            else:
                engine = get_engine_with_cache(
                    engine_cache_key, TensorRTEngineConfig(enable_cuda_graph=False)
                )

                if self.onnx_cache == None or (
                    self.input_shape_info != input_shape_info and engine == None
                ):
                    module.to(device=args[0].device)
                    self.onnx_cache = BytesIO()
                    th.onnx.export(
                        module,
                        tuple(args),
                        self.onnx_cache,
                        export_params=True,
                        verbose=False,
                        do_constant_folding=True,
                        input_names=args_name,
                        output_names=["output"],
                    )

                nvtx.range_push("offload origin model")
                module.to(device="cpu")
                gc.collect()
                comfy.model_management.soft_empty_cache()
                nvtx.range_pop()

                if engine == None:
                    engine = gen_engine(
                        engine_cache_key, self.onnx_cache.getvalue(), input_shape_info
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
                            load_device=args[0].device,
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
                    self.input_shape_info = input_shape_info
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
    cuda_stream = None
    unet_config = None
    origin_model_patcher: comfy.model_patcher.ModelPatcher = None


class ForwardTimestepEmbedModule(th.nn.Module):
    def __init__(
        self, ts, transformer_options={}, output_shape=None, num_video_frames=None
    ):
        super().__init__()
        self.module = ts
        self.transformer_options = transformer_options
        self.output_shape = output_shape
        self.num_video_frames = num_video_frames

    def forward(
        self, x, emb, context=None, time_context=None, image_only_indicator=None
    ):
        return origin_forward_timestep_embed(
            self.module,
            x,
            emb,
            context=context,
            transformer_options=self.transformer_options,
            output_shape=self.output_shape,
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
    module = ForwardTimestepEmbedModule(
        ts, transformer_options, output_shape, num_video_frames
    )
    tensorrt_context: TensorRTEngineCacheContext = transformer_options.get(
        TENSORRT_CONTEXT_KEY, None
    )
    if tensorrt_context != None:
        block_key = str(transformer_options["block"])
        block = tensorrt_context.block_cache.get(block_key, None)
        if block == None:
            tensorrt_context.block_cache[block_key] = CallableTensorRTEngineWarper(
                tensorrt_context, block_key
            )
        return tensorrt_context.block_cache[block_key](
            module,
            x=x,
            emb=emb,
            context=context,
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
        input_profile=[input_profile],
    )
    e = time.time()
    print(f"Time taken to build: {e-s}s")
    return engine
