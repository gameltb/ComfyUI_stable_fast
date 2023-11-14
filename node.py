import time

import torch

from sfast.compilers.stable_diffusion_pipeline_compiler import (
    CompilationConfig, compile_unet)


def is_cuda_malloc_async():
    return "cudaMallocAsync" in torch.cuda.get_allocator_backend()


def gen_stable_fast_config():
    config = CompilationConfig.Default()
    # xformers and triton are suggested for achieving best performance.
    # It might be slow for triton to generate, compile and fine-tune kernels.
    try:
        import xformers

        config.enable_xformers = True
    except ImportError:
        print("xformers not installed, skip")
    try:
        import triton

        config.enable_triton = True
    except ImportError:
        print("triton not installed, skip")

    if config.enable_triton and is_cuda_malloc_async():
        print("disable stable fast triton because of cudaMallocAsync")
        config.enable_triton = False

    # CUDA Graph is suggested for small batch sizes.
    # After capturing, the model only accepts one fixed image size.
    # If you want the model to be dynamic, don't enable it.
    config.enable_cuda_graph = True
    return config


class ApplyStableFastUnet:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
            }
        }

    RETURN_TYPES = ("MODEL", )
    FUNCTION = "apply_stable_fast"

    CATEGORY = "loaders"

    def apply_stable_fast(self, model):
        model = model.clone()

        config = gen_stable_fast_config()

        assert hasattr(model.model, "diffusion_model")
        unet = model.model.diffusion_model
        # Keep the original forward function for recompilation
        original_unet_forward = unet.forward

        def compile_unet_(unet, config):
            if not getattr(unet, "_has_compiled", False):
                print("Compiling...")
                compile_unet(unet, config)
                unet._has_compiled = True
            return unet

        to = unet.to

        def to_(*args, **kwargs):
            to(*args, **kwargs)
            device = args[0] if len(args) > 0 else kwargs.get("device")
            if isinstance(device, torch.device):
                if config.enable_cuda_graph or config.enable_jit_freeze:
                    if device.type == "cpu":
                        print(
                            "\33[93mWarning: Your graphics card doesn't have enough video memory to keep the model. Disable stable fast cuda graph, Flexibility will be improved but speed will be lost.\33[0m"
                        )
                        # comfyui tell we should move to cpu. but we can't do it with cuda graph and freeze now.
                        config.enable_cuda_graph = False
                        config.enable_jit_freeze = False
                        unet.forward = original_unet_forward
                        model.model.diffusion_model = compile_unet_(unet, config)
                else:
                    if hasattr(unet.forward, "_traced_modules"):
                        print(f"\33[93mTransfer model to {device}.\33[0m")
                        traced_modules = unet._traced_modules
                        for k, v in traced_modules.items():
                            v.to(device)

            return self

        unet.to = to_

        model.model.diffusion_model = compile_unet_(unet, config)
        print("Stable fast mode enabled, even if you remove this loader, it will still take effect.")

        return (model, )
