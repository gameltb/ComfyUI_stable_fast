import yaml
from dataclasses import dataclass, field
from typing import Any, Dict

import torch as th

import comfy.ldm.modules.diffusionmodules.openaimodel
import comfy.model_management
import comfy.model_patcher
from .tensorrt_wrapper import TensorRTEngineContext, CallableTensorRTEngineWrapper
from .comfy_trace.openaimodel import (
    ForwardTimestepEmbedModule,
    origin_forward_timestep_embed,
)


TENSORRT_CONTEXT_KEY = "tensorrt_context"


@dataclass
class TensorRTEngineBlockContext:
    block_cache: Dict[str, CallableTensorRTEngineWrapper] = field(
        default_factory=lambda: {}
    )
    tensorrt_context: TensorRTEngineContext = field(
        default_factory=lambda: TensorRTEngineContext()
    )

    def dump_input_profile_info(self):
        input_shape_info_map = {}
        for key in sorted(self.block_cache):
            input_shape_info_map[key] = self.block_cache[key].input_shape_info
        print(yaml.safe_dump(input_shape_info_map))


class CallableTensorRTEngineWrapperDynamicShapeForwardTimestep(
    CallableTensorRTEngineWrapper
):
    args_name = [
        "x",
        "emb",
        "context",
        "output_shape_tensor",
        "time_context",
        "image_only_indicator",
    ]

    def gen_onnx_args(self, kwargs,module=None):
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
        for arg_name, shape_info in input_shape_info.items():
            min_shape_config = min_input_profile_info.get(arg_name, None)
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
    tensorrt_block_context: TensorRTEngineBlockContext = transformer_options.get(
        TENSORRT_CONTEXT_KEY, None
    )
    if tensorrt_block_context != None:
        block_key = str(transformer_options["block"])
        block = tensorrt_block_context.block_cache.get(block_key, None)
        if block == None:
            tensorrt_block_context.block_cache[
                block_key
            ] = CallableTensorRTEngineWrapperDynamicShapeForwardTimestep(
                tensorrt_block_context.tensorrt_context, block_key
            )
        return tensorrt_block_context.block_cache[block_key](
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
