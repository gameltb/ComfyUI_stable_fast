import gc
import hashlib
import os
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import Dict

import torch as th
from torch.cuda import nvtx

import comfy.ldm.modules.diffusionmodules.openaimodel
import comfy.model_management
from comfy.ldm.modules.diffusionmodules.openaimodel import \
    forward_timestep_embed

from .comfy_trace_utilities import hash_arg
from .tensorrt_utilities import Engine

TENSORRT_CONTEXT_KEY = "tensorrt_context"

origin_forward_timestep_embed = forward_timestep_embed


@dataclass
class TensorRTEngineConfig:
    enable_cuda_graph: bool


@dataclass
class TensorRTEngineCacheContextBlockItem:
    engine_cache: Engine = None
    onnx_cache: BytesIO = None
    pytorch_model_device: th.device = None
    input_shape_info = None


@dataclass
class TensorRTEngineCacheContext:
    block_cache: Dict[str, TensorRTEngineCacheContextBlockItem] = field(
        default_factory=lambda: {}
    )
    cuda_stream = None
    unet_config = None


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
        block = tensorrt_context.block_cache.get(
            block_key, TensorRTEngineCacheContextBlockItem()
        )
        args = (x, emb, context, time_context, image_only_indicator)
        args_name = ["x", "emb", "context", "time_context", "image_only_indicator"]
        input_shape_info = {}
        feed_dict = {}
        for arg_name, arg in zip(args_name, args):
            if arg != None:
                feed_dict[arg_name] = arg
                input_shape_info[arg_name] = [
                    tuple(arg.shape),
                    tuple(arg.shape),
                    tuple(arg.shape),
                ]

        if block.engine_cache == None or block.input_shape_info != input_shape_info:
            del block.engine_cache
            block.engine_cache = None

            engine_cache_key = (
                hash_arg(tensorrt_context.unet_config),
                hash_arg(block_key),
                hash_arg(input_shape_info),
            )

            engine = get_engine_with_cache(engine_cache_key, TensorRTEngineConfig(enable_cuda_graph=False))

            if block.onnx_cache == None or (
                block.input_shape_info != input_shape_info
                and engine == None
            ):
                ts.to(device=x.device)
                block.onnx_cache = BytesIO()
                th.onnx.export(
                    module,
                    tuple(args),
                    block.onnx_cache,
                    export_params=True,
                    verbose=False,
                    do_constant_folding=True,
                    input_names=args_name,
                    output_names=["h"],
                )
            tensorrt_context.block_cache[block_key] = block

            nvtx.range_push("offload origin model")
            ts.to(device="cpu")
            gc.collect()
            comfy.model_management.soft_empty_cache()
            nvtx.range_pop()

            if engine == None:
                engine = gen_engine(engine_cache_key, block.onnx_cache.getvalue(), input_shape_info)
                block.onnx_cache.seek(0)
                engine.refit_simple(block.onnx_cache, reset_zero=True)
                engine.save_engine()
                del engine
                engine = get_engine_with_cache(engine_cache_key, TensorRTEngineConfig(enable_cuda_graph=False))

            block.engine_cache = engine
            try:
                nvtx.range_push("load engine")
                block.engine_cache.load()
                block.engine_cache.activate(True)
                nvtx.range_push("refit engine")
                block.onnx_cache.seek(0)
                block.engine_cache.refit_simple(block.onnx_cache)
                nvtx.range_pop()
                block.input_shape_info = input_shape_info
                nvtx.range_pop()
            except Exception as e:
                block.engine_cache = None
                raise e

        block.engine_cache.allocate_buffers(feed_dict)
        return block.engine_cache.infer(feed_dict, tensorrt_context.cuda_stream)[
            "h"
        ]
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
