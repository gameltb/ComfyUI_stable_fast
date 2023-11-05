import logging
import packaging.version
from dataclasses import dataclass
from typing import Union
import functools
import torch
import sfast
from sfast.jit import passes
from sfast.jit.trace_helper import (lazy_trace, to_module)
from sfast.jit import utils as jit_utils
from sfast.cuda.graphs import make_dynamic_graphed_callable

logger = logging.getLogger()


class CompilationConfig:

    @dataclass
    class Default:
        memory_format: torch.memory_format = torch.channels_last
        enable_jit: bool = True
        enable_jit_freeze: bool = True
        enable_cnn_optimization: bool = True
        prefer_lowp_gemm: bool = True
        enable_xformers: bool = False
        enable_cuda_graph: bool = False
        enable_triton: bool = False


def compile_unet(unet_funtion, config, device):
    enable_cuda_graph = config.enable_cuda_graph and device.type == 'cuda'

    with torch.no_grad():
        if config.enable_xformers:
            if config.enable_jit:
                from sfast.utils.xformers_attention import xformers_memory_efficient_attention
                from xformers import ops

                ops.memory_efficient_attention = xformers_memory_efficient_attention

        if config.enable_jit:
            modify_model = functools.partial(
                _modify_model,
                enable_cnn_optimization=config.enable_cnn_optimization,
                prefer_lowp_gemm=config.prefer_lowp_gemm,
                enable_triton=config.enable_triton,
                memory_format=config.memory_format,
            )

            def ts_compiler(
                m,
                call_helper,
                inputs,
                kwarg_inputs,
                freeze=False,
                enable_cuda_graph=False,
            ):
                with torch.jit.optimized_execution(True):
                    if freeze:
                        # raw freeze causes Tensor reference leak
                        # because the constant Tensors in the GraphFunction of
                        # the compilation unit are never freed.
                        m.eval()
                        m = jit_utils.better_freeze(m)
                    modify_model(m)

                if enable_cuda_graph:
                    m = make_dynamic_graphed_callable(m)
                return m

            unet_forward = lazy_trace(to_module(unet_funtion),
                                      ts_compiler=functools.partial(
                                          ts_compiler,
                                          freeze=config.enable_jit_freeze,
                                          enable_cuda_graph=enable_cuda_graph,
                                      ),
                                      check_trace=False,
                                      strict=False)

            @functools.wraps(unet_funtion)
            def unet_forward_wrapper(sample, t, *args, **kwargs):
                t = t.to(device=sample.device)
                return unet_forward(sample, t, *args, **kwargs)

            return unet_forward_wrapper

        return unet_funtion


def _modify_model(m,
                  enable_cnn_optimization=True,
                  prefer_lowp_gemm=True,
                  enable_triton=False,
                  memory_format=None):
    if enable_triton:
        from sfast.jit.passes import triton_passes

    torch._C._jit_pass_inline(m.graph)
    passes.jit_pass_remove_dropout(m.graph)

    passes.jit_pass_remove_contiguous(m.graph)
    passes.jit_pass_replace_view_with_reshape(m.graph)
    if enable_triton:
        triton_passes.jit_pass_optimize_reshape(m.graph)

        # triton_passes.jit_pass_optimize_cnn(m.graph)

        triton_passes.jit_pass_fuse_group_norm_silu(m.graph)
        triton_passes.jit_pass_optimize_group_norm(m.graph)

    passes.jit_pass_optimize_linear(m.graph)

    if memory_format is not None:
        sfast._C._jit_pass_convert_op_input_tensors(
            m.graph,
            'aten::_convolution',
            indices=[0],
            memory_format=memory_format)

    if enable_cnn_optimization:
        passes.jit_pass_optimize_cnn(m.graph)

    if prefer_lowp_gemm:
        passes.jit_pass_prefer_lowp_gemm(m.graph)
        passes.jit_pass_fuse_lowp_linear_add(m.graph)
