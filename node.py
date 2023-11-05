from .module.stable_diffusion_pipeline_compiler import (CompilationConfig, compile_unet)

class StableFastPatch:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.stable_fast_model = None

    def __call__(self, model_function, params):
        input_x = params.get("input")
        timestep_ = params.get("timestep")
        c = params.get("c")

        if self.stable_fast_model is None:
            self.stable_fast_model = compile_unet(model_function, self.config, input_x.device)

        return self.stable_fast_model(input_x, timestep_, **c)

    def to(self, device):
        pass


class ApplyStableFastUnet:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",), }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_stable_fast"

    CATEGORY = "loaders"

    def apply_stable_fast(self, model):
        config = CompilationConfig.Default()

        if config.memory_format is not None:
            model.model.to(memory_format=config.memory_format)

        # xformers and triton are suggested for achieving best performance.
        # It might be slow for triton to generate, compile and fine-tune kernels.
        try:
            import xformers
            config.enable_xformers = True
        except ImportError:
            print('xformers not installed, skip')
        try:
            import triton
            config.enable_triton = True
        except ImportError:
            print('triton not installed, skip')
        # CUDA Graph is suggested for small batch sizes.
        # After capturing, the model only accepts one fixed image size.
        # If you want the model to be dynamic, don't enable it.
        config.enable_cuda_graph = False
        config.enable_jit_freeze = False

        patch = StableFastPatch(model, config)
        model_stable_fast = model.clone()
        model_stable_fast.set_model_unet_function_wrapper(patch)
        return (model_stable_fast,)
