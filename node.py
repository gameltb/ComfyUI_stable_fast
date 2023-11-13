import time

import torch
from sfast.compilers.stable_diffusion_pipeline_compiler import CompilationConfig
from sfast.jit.trace_helper import to_module

from .module.stable_diffusion_pipeline_compiler import compile_unet


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
    # config.enable_jit_freeze = False
    return config


class StableFastPatch:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.stable_fast_model = None
        self.offload_flag = False
        self.model_device = torch.device("cpu")

    def __call__(self, model_function, params):
        input_x = params.get("input")
        timestep_ = params.get("timestep")
        c = params.get("c")

        # disable with accelerate for now
        if hasattr(model_function.__self__, "hf_device_map"):
            return model_function(input_x, timestep_, **c)

        model_function_module = to_module(model_function)

        if self.stable_fast_model is None:
            self.stable_fast_model = compile_unet(
                model_function_module, self.config, input_x.device
            )

        if self.config.enable_cuda_graph or self.config.enable_jit_freeze:
            return self.stable_fast_model(input_x, timestep_, **c)(
                input_x, timestep_, **c
            )
        else:
            stable_fast_model_function = self.stable_fast_model(input_x, timestep_, **c)
            if self.offload_flag:
                if self.model_device != self.model.offload_device:
                    next(
                        next(stable_fast_model_function.children()).children()
                    ).load_state_dict(
                        model_function_module.state_dict(), strict=False, assign=True
                    )
                    self.model_device = self.model.offload_device
            return stable_fast_model_function(input_x, timestep_, **c)

    def to(self, device):
        if type(device) == torch.device:
            self.model_device = device
            if self.config.enable_cuda_graph or self.config.enable_jit_freeze:
                if device.type == "cpu":
                    # comfyui tell we should move to cpu. but we cannt do it with cuda graph and freeze now.
                    del self.stable_fast_model
                    self.stable_fast_model = None
                    self.config.enable_cuda_graph = False
                    self.config.enable_jit_freeze = False
                    print(
                        "\33[93mWarning: Your graphics card doesn't have enough video memory to keep the model. Disable stable fast cuda graph, Flexibility will be improved but speed will be lost.\33[0m"
                    )
            else:
                if self.stable_fast_model != None and device.type == "cpu":
                    self.offload_flag = True
                    self.stable_fast_model.to_empty("meta")
        return self


class ApplyStableFastUnet:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_stable_fast"

    CATEGORY = "loaders"

    def apply_stable_fast(self, model):
        config = gen_stable_fast_config()

        if config.memory_format is not None:
            model.model.to(memory_format=config.memory_format)

        patch = StableFastPatch(model, config)
        model_stable_fast = model.clone()
        model_stable_fast.set_model_unet_function_wrapper(patch)
        return (model_stable_fast,)
