import functools
import logging
from dataclasses import dataclass

import torch
from sfast.compilers.stable_diffusion_pipeline_compiler import _modify_model
from sfast.cuda.graphs import make_dynamic_graphed_callable
from sfast.jit import utils as jit_utils
from sfast.jit.trace_helper import hash_arg, trace_with_kwargs
from sfast.jit.trace_helper import to_module

logger = logging.getLogger()

from .nodes_freelunch import FreeU, FreeU_V2
from .openaimodel import PatchUNetModel
from .nodes_model_downscale import (
    PatchModelAddDownscale_input_block_patch,
    PatchModelAddDownscale_output_block_patch,
)

PATCH_PATCH_MAP = {
    "FreeU.patch.<locals>.output_block_patch": FreeU,
    "FreeU_V2.patch.<locals>.output_block_patch": FreeU_V2,
    "PatchModelAddDownscale.patch.<locals>.input_block_patch": PatchModelAddDownscale_input_block_patch,
    "PatchModelAddDownscale.patch.<locals>.output_block_patch": PatchModelAddDownscale_output_block_patch,
}


@dataclass
class TracedModuleCacheItem:
    module: object
    patch_id: int
    device: str


def gen_comfy_unet_cache_key(unet_config, args, kwargs, patch_module):
    key_kwargs = {}
    for k, v in kwargs.items():
        if k == "transformer_options":
            nv = {}
            for tk, tv in v.items():
                if not tk in ("patches"):  # ,"cond_or_uncond"
                    nv[tk] = tv
            v = nv
        key_kwargs[k] = v

    patch_module_cache_key = {}
    for patch_type_name, patch_list in patch_module.items():
        patch_module_cache_key[patch_type_name] = []
        for patch in patch_list:
            patch_module_cache_key[patch_type_name].append(patch.gen_cache_key())

    return (
        hash_arg(unet_config),
        hash_arg(args),
        hash_arg(key_kwargs),
        hash_arg(patch_module_cache_key),
    )


def convert_comfy_args(args, kwargs):
    transformer_options = kwargs.get("transformer_options", {})
    patches = transformer_options.get("patches", {})

    patch_module = {}
    patch_module_parameter = {}

    for patch_type_name, patch_list in patches.items():
        patch_module[patch_type_name] = []
        patch_module_parameter[patch_type_name] = []
        for patch in patch_list:
            if patch.__qualname__ in PATCH_PATCH_MAP:
                patch, parameter = PATCH_PATCH_MAP[patch.__qualname__].from_closure(
                    patch, transformer_options
                )
                patch_module[patch_type_name].append(patch)
                patch_module_parameter[patch_type_name].append(parameter)
                # output_block_patch_module.append(torch.jit.script(patch))
            else:
                print(f"\33[93mWarning: Ignore patch {patch.__qualname__}.\33[0m")

    transformer_options["patches"] = patch_module_parameter

    return patch_module


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

    def get_traced_module(self, model_function, *args, **kwargs):
        unet_config = model_function.__self__.model_config.unet_config

        patch_module = convert_comfy_args(args, kwargs)
        key = gen_comfy_unet_cache_key(unet_config, args, kwargs, patch_module)

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
            func = to_module(model_function)
            logger.info(f'Tracing {getattr(func, "__name__", func.__class__.__name__)}')

            if len(patch_module) > 0:
                func.module.diffusion_model = PatchUNetModel.cast_from(
                    func.module.diffusion_model
                )
                try:
                    func.module.diffusion_model.set_patch_module(patch_module)

                    traced_m, call_helper = trace_with_kwargs(
                        func, args, kwargs, **self.kwargs_
                    )
                finally:
                    func.module.diffusion_model = (
                        func.module.diffusion_model.cast_to_base_model()
                    )
            else:
                traced_m, call_helper = trace_with_kwargs(
                    func, args, kwargs, **self.kwargs_
                )

            traced_m = self.ts_compiler(traced_m)
            traced_module = call_helper(traced_m)
            if self.config.enable_cuda_graph or self.config.enable_jit_freeze:
                self.cuda_graph_modules[key] = traced_module
            else:
                self.traced_modules[key] = TracedModuleCacheItem(
                    module=traced_module, patch_id=self.patch_id, device=None
                )

        return traced_module

    def to_empty(self):
        for v in self.traced_modules.values():
            v.module.to_empty(device="meta")
            v.device = "meta"


def build_lazy_trace_module(config, device, patch_id):
    config.enable_cuda_graph = config.enable_cuda_graph and device.type == "cuda"

    if config.enable_xformers:
        from sfast.utils.xformers_attention import (
            xformers_memory_efficient_attention,
        )
        from xformers import ops

        ops.memory_efficient_attention = xformers_memory_efficient_attention

    return LazyTraceModule(
        config=config,
        patch_id=patch_id,
        check_trace=True,
        strict=True,
    )
