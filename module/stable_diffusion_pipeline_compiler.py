import functools
import logging
from dataclasses import dataclass

import torch
from sfast.compilers.stable_diffusion_pipeline_compiler import (
    _enable_xformers,
    _modify_model,
)
from sfast.cuda.graphs import make_dynamic_graphed_callable
from sfast.jit import utils as jit_utils
from sfast.jit.trace_helper import to_module, trace_with_kwargs

from .comfy_trace_utilities import BaseModelApplyModel

logger = logging.getLogger()


@dataclass
class TracedModuleCacheItem:
    module: object
    patch_id: int
    device: str


class LazyTraceModule:
    traced_modules = {}

    def __init__(self, config=None, patch_id=None, **kwargs_) -> None:
        self.config = config
        self.patch_id = patch_id
        self.kwargs_ = kwargs_
        self.modify_model = functools.partial(
            _modify_model,
            enable_cnn_optimization=config.enable_cnn_optimization,
            prefer_lowp_gemm=config.prefer_lowp_gemm,
            enable_triton=config.enable_triton,
            enable_triton_reshape=config.enable_triton,
            memory_format=config.memory_format,
        )
        self.cuda_graph_modules = {}

    def ts_compiler(
        self,
        m,
    ):
        with torch.jit.optimized_execution(True):
            if self.config.enable_jit_freeze:
                # raw freeze causes Tensor reference leak
                # because the constant Tensors in the GraphFunction of
                # the compilation unit are never freed.
                m.eval()
                m = jit_utils.better_freeze(m)
            self.modify_model(m)

        if self.config.enable_cuda_graph:
            m = make_dynamic_graphed_callable(m)
        return m

    def __call__(self, model_function, *args, **kwargs):
        module = BaseModelApplyModel(model_function, args, kwargs)
        args, kwargs = module.convert_args()
        key = module.gen_cache_key()

        traced_module = self.cuda_graph_modules.get(key)
        if traced_module is None and not (
            self.config.enable_cuda_graph or self.config.enable_jit_freeze
        ):
            traced_module_cache = self.traced_modules.get(key)
            if not traced_module_cache is None:
                if (
                    traced_module_cache.patch_id != self.patch_id
                    or traced_module_cache.device == "meta"
                ):
                    model_function_module = to_module(model_function)
                    next(
                        next(traced_module_cache.module.children()).children()
                    ).load_state_dict(
                        model_function_module.state_dict(), strict=False, assign=True
                    )
                    traced_module_cache.device = None
                    traced_module_cache.patch_id = self.patch_id
                traced_module = traced_module_cache.module

        if traced_module is None:

            def m_trace(m_model_function, m_args, m_kwargs):
                func = to_module(m_model_function)
                logger.info(
                    f'Tracing {getattr(func, "__name__", func.__class__.__name__)}'
                )
                return trace_with_kwargs(func, m_args, m_kwargs, **self.kwargs_)

            traced_m, call_helper = module.do_with_convert_module(m_trace)

            traced_m = self.ts_compiler(traced_m)
            traced_module = call_helper(traced_m)
            if self.config.enable_cuda_graph or self.config.enable_jit_freeze:
                self.cuda_graph_modules[key] = traced_module
            else:
                self.traced_modules[key] = TracedModuleCacheItem(
                    module=traced_module, patch_id=self.patch_id, device=None
                )

        return traced_module(*args, **kwargs)

    def to_empty(self):
        for v in self.traced_modules.values():
            v.module.to_empty(device="meta")
            v.device = "meta"


def build_lazy_trace_module(config, device, patch_id):
    config.enable_cuda_graph = config.enable_cuda_graph and device.type == "cuda"

    if config.enable_xformers:
        _enable_xformers(None)

    return LazyTraceModule(
        config=config,
        patch_id=patch_id,
        check_trace=True,
        strict=True,
    )
