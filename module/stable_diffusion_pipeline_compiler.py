import functools
import logging
import threading

import torch
from sfast.compilers.stable_diffusion_pipeline_compiler import _modify_model
from sfast.cuda.graphs import make_dynamic_graphed_callable
from sfast.jit import utils as jit_utils
from sfast.jit.trace_helper import hash_arg, trace_with_kwargs

logger = logging.getLogger()


def lazy_trace(func, *, ts_compiler=None, **kwargs_):
    class TraceModule:
        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.traced_modules = {}

        def __call__(self, *args, **kwargs):
            key = (hash_arg(args), hash_arg(kwargs))
            traced_module = self.traced_modules.get(key)
            if traced_module is None:
                with self.lock:
                    traced_module = self.traced_modules.get(key)
                    if traced_module is None:
                        logger.info(
                            f'Tracing {getattr(func, "__name__", func.__class__.__name__)}'
                        )
                        traced_m, call_helper = trace_with_kwargs(
                            func, args, kwargs, **kwargs_
                        )
                        if ts_compiler is not None:
                            traced_m = ts_compiler(traced_m, call_helper, args, kwargs)
                        traced_module = call_helper(traced_m)
                        self.traced_modules[key] = traced_module
            return traced_module

        def to(self, device):
            for v in self.traced_modules.values():
                v.to(device)

        def to_empty(self, device):
            for v in self.traced_modules.values():
                v.to_empty(device=device)

    return TraceModule()


def compile_unet(unet_module, config, device):
    enable_cuda_graph = config.enable_cuda_graph and device.type == "cuda"

    with torch.no_grad():
        if config.enable_xformers:
            if config.enable_jit:
                from sfast.utils.xformers_attention import \
                    xformers_memory_efficient_attention
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

            unet_forward = lazy_trace(
                unet_module,
                ts_compiler=functools.partial(
                    ts_compiler,
                    freeze=config.enable_jit_freeze,
                    enable_cuda_graph=enable_cuda_graph,
                ),
                check_trace=False,
                strict=False,
            )

            return unet_forward

        return unet_module
