import copy
import enum

import comfy.model_management
import comfy.model_patcher
import nodes
import torch

from .module.comfy_trace.model_base import (
    UNetModelModuleFactory,
)
from .module.comfy_trace.sd import VAEDecodeModule
from .module.controlnet_tensorrt import (
    CallableTensorRTEngineWrapperDynamicShapeControlNet,
)
from .module.openaimodel_tensorrt import (
    TENSORRT_CONTEXT_KEY,
    CallableTensorRTEngineWrapperDynamicShapeUNetModelForward,
    TensorRTEngineBlockContext,
    do_hook_forward_timestep_embed,
    undo_hook_forward_timestep_embed,
)
from .module.sd_tensorrt import CallableTensorRTEngineWrapperDynamicShapeVAEDecode
from .module.tensorrt_wrapper import TensorRTEngineConfig, TensorRTEngineContext


class BlockTensorRTPatch(torch.nn.Module):
    def __init__(self, config, model_config):
        super().__init__()
        self.model: torch.nn.Module = None
        self.model_config = model_config
        self.config = config
        self.model_device = torch.device("cpu")
        self.tensorrt_module = None

    def __deepcopy__(self, memo=None):
        return self

    @property
    def dtype(self):
        return self.model.dtype

    def warmup(
        self,
        x,
        timesteps,
        context,
        y,
        control,
        transformer_options,
        **kwargs,
    ):
        warmup_input_x = torch.zeros(
            (
                self.config.keep_batch_size * 2,
                x.shape[1],
                int(self.config.keep_height / 8),
                int(self.config.keep_width / 8),
            ),
            device=x.device,
            dtype=x.dtype,
        )
        warmup_x = warmup_input_x
        warmup_timesteps = torch.ones(
            (self.config.keep_batch_size * 2,),
            device=timesteps.device,
            dtype=timesteps.dtype,
        )
        warmup_context = None
        if context is not None:
            warmup_context = torch.zeros(
                (
                    self.config.keep_batch_size * 2,
                    self.config.keep_embedding_block * 77,
                    context.shape[2],
                ),
                device=context.device,
                dtype=context.dtype,
            )
        warmup_y = None
        if y is not None:
            warmup_y = torch.zeros(
                (
                    self.config.keep_batch_size * 2,
                    y.shape[1],
                ),
                device=y.device,
                dtype=y.dtype,
            )

        self(
            warmup_x,
            warmup_timesteps,
            warmup_context,
            warmup_y,
            None,
            {},
            **kwargs,
        )

    def __call__(
        self,
        x,
        timesteps=None,
        context=None,
        y=None,
        control=None,
        transformer_options={},
        **kwargs,
    ):
        if self.tensorrt_module is None:
            self.tensorrt_module = TensorRTEngineBlockContext()
            self.tensorrt_module.tensorrt_context.keep_models.append(self.model)
            self.warmup(
                x,
                timesteps,
                context,
                y,
                control,
                transformer_options,
                **kwargs,
            )

        self.tensorrt_module.tensorrt_context.model_type = (
            self.model_config.__class__.__name__
        )
        self.tensorrt_module.tensorrt_context.unet_config = (
            self.model_config.unet_config
        )
        self.tensorrt_module.tensorrt_context.cuda_stream = torch.cuda.current_stream()
        self.tensorrt_module.tensorrt_context.cuda_device = x.device
        transformer_options[TENSORRT_CONTEXT_KEY] = self.tensorrt_module

        do_hook_forward_timestep_embed()
        try:
            out = self.model(
                x,
                timesteps,
                context,
                y,
                control,
                transformer_options,
                **kwargs,
            )
        finally:
            undo_hook_forward_timestep_embed()
            transformer_options.pop(TENSORRT_CONTEXT_KEY)

        return out

    def to(self, device):
        if type(device) is torch.device:
            self.model_device = device
        return self


class UnetTensorRTPatch(BlockTensorRTPatch):
    def __init__(self, config, model_config):
        super().__init__(config, model_config)
        self.tensorrt_context = TensorRTEngineContext()

    def __call__(
        self,
        x,
        timesteps=None,
        context=None,
        y=None,
        control=None,
        transformer_options={},
        **kwargs,
    ):
        if self.tensorrt_module is None:
            devices = set((v.device for v in self.model.state_dict().values()))
            if torch.device("cpu") in devices:
                self.tensorrt_context.enable_weight_streaming = True
            self.tensorrt_module = (
                CallableTensorRTEngineWrapperDynamicShapeUNetModelForward(
                    self.tensorrt_context, ""
                )
            )
            if control is None:
                self.warmup(
                    x,
                    timesteps,
                    context,
                    y,
                    control,
                    transformer_options,
                    **kwargs,
                )

        self.tensorrt_context.model_type = self.model_config.__class__.__name__
        self.tensorrt_context.unet_config = self.model_config.unet_config

        self.tensorrt_context.cuda_stream = torch.cuda.current_stream()
        self.tensorrt_context.cuda_device = x.device
        # self.tensorrt_context.dtype = input_x.dtype

        module_factory = UNetModelModuleFactory(
            self.model,
            self.model_config,
            x=x,
            timesteps=timesteps,
            context=context,
            y=y,
            control=control,
            transformer_options=transformer_options,
            **kwargs,
        )

        with module_factory.converted_module_context() as (m_model, m_kwargs):
            out = self.tensorrt_module(m_model, **m_kwargs)

        return out


class ModelUnetFunctionWrapper:
    def __init__(self, patch):
        self.patch = patch

    def __deepcopy__(self, memo=None):
        return self

    def __call__(self, model_function, params):
        input_x = params.get("input")
        timestep_ = params.get("timestep")
        c = params.get("c")

        origin_diffusion_model = model_function.__self__.diffusion_model
        self.patch.model = origin_diffusion_model
        model_function.__self__.diffusion_model = self.patch
        try:
            out = model_function(input_x, timestep_, **c)
        finally:
            model_function.__self__.diffusion_model = origin_diffusion_model

        return out


def hook_memory_required(input_shape):
    return 0


class TensorRTEngineOriginModelPatcherWrapper_BlockPatch(
    comfy.model_patcher.ModelPatcher
):
    @staticmethod
    def cast_from(other):
        tcls = comfy.model_patcher.ModelPatcher
        if isinstance(other, tcls):
            other.__class__ = TensorRTEngineOriginModelPatcherWrapper_BlockPatch
            return other
        raise ValueError(f"instance must be {tcls.__qualname__}")

    def patch_init(self, tensorrt_module_patch):
        self.tensorrt_module_patch = tensorrt_module_patch

    def patch_deinit(self):
        self.tensorrt_module_patch = None
        del self.tensorrt_module_patch

    def cast_to_base_model(self):
        self.patch_deinit()
        self.__class__ = comfy.model_patcher.ModelPatcher
        return self

    def patch_model(self, device_to=None, *arg, **kwargs):
        model = super().patch_model()

        if device_to is not None:
            for name, module in model.named_children():
                if name in ("diffusion_model",):
                    for name, module in module.named_children():
                        if not name in (
                            "input_blocks",
                            "middle_block",
                            "output_blocks",
                        ):
                            module.to(device_to)
                else:
                    module.to(device_to)
            self.current_device = device_to

        return model

    def __del__(self):
        self.model.to(self.current_device)


class TensorRTEngineOriginModelPatcherWrapper_UnetPatch(
    comfy.model_patcher.ModelPatcher
):
    @staticmethod
    def cast_from(other):
        tcls = comfy.model_patcher.ModelPatcher
        if isinstance(other, tcls):
            other.__class__ = TensorRTEngineOriginModelPatcherWrapper_UnetPatch
            return other
        raise ValueError(f"instance must be {tcls.__qualname__}")

    def patch_init(self, tensorrt_module_patch):
        self.tensorrt_module_patch = tensorrt_module_patch

    def patch_deinit(self):
        self.tensorrt_module_patch = None
        del self.tensorrt_module_patch

    def cast_to_base_model(self):
        self.patch_deinit()
        self.__class__ = comfy.model_patcher.ModelPatcher
        return self

    def model_size(self):
        if (
            self.tensorrt_module_patch is None
            or self.tensorrt_module_patch.tensorrt_module is None
        ):
            return super().model_size()
        return 0

    def patch_model_lowvram(
        self,
        device_to=None,
        lowvram_model_memory=0,
        force_patch_weights=False,
        *arg,
        **kwargs,
    ):
        if (
            self.tensorrt_module_patch is None
            or self.tensorrt_module_patch.tensorrt_module is None
        ):
            return super().patch_model_lowvram(
                device_to=device_to,
                lowvram_model_memory=lowvram_model_memory,
                force_patch_weights=force_patch_weights,
                *arg,
                **kwargs,
            )
        return self.patch_model(
            device_to=device_to,
        )

    def patch_model(self, device_to=None, *arg, **kwargs):
        model = super().patch_model()

        if device_to is not None:
            self.current_device = device_to

        return model

    def __del__(self):
        self.model.to(self.current_device)


class PatchType(enum.Enum):
    UNET = UnetTensorRTPatch, TensorRTEngineOriginModelPatcherWrapper_UnetPatch
    UNET_BLOCK = BlockTensorRTPatch, TensorRTEngineOriginModelPatcherWrapper_BlockPatch


class ApplyTensorRTUnet:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "enable_cuda_graph": ("BOOLEAN", {"default": True}),
                "patch_type": ([e.name for e in PatchType], {"default": "UNET"}),
                "hook_memory_require": ("BOOLEAN", {"default": True}),
                "keep_width": (
                    "INT",
                    {"default": 768, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8},
                ),
                "keep_height": (
                    "INT",
                    {"default": 768, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8},
                ),
                "keep_batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "keep_embedding_block": ("INT", {"default": 2, "min": 1, "max": 4096}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_tensorrt"

    CATEGORY = "loaders"

    def apply_tensorrt(
        self,
        model,
        enable_cuda_graph,
        patch_type,
        hook_memory_require,
        keep_width,
        keep_height,
        keep_batch_size,
        keep_embedding_block,
    ):
        config = TensorRTEngineConfig(
            enable_cuda_graph=enable_cuda_graph,
            keep_width=keep_width,
            keep_height=keep_height,
            keep_batch_size=keep_batch_size,
            keep_embedding_block=keep_embedding_block,
        )
        patch_type_clss = PatchType[patch_type].value
        model_tensor_rt = model.clone()
        patch = patch_type_clss[0](config, model.model.model_config)
        model_tensor_rt = patch_type_clss[1].cast_from(model_tensor_rt)
        patch.model = model_tensor_rt
        model_tensor_rt.set_model_unet_function_wrapper(ModelUnetFunctionWrapper(patch))
        model_tensor_rt.patch_init(patch)
        if hook_memory_require:
            model_tensor_rt.add_object_patch("memory_required", hook_memory_required)
        return (model_tensor_rt,)


class VAEDecodeTensorRTPatch:
    def __init__(self, model, config):
        self.model = model
        self.org_decode = model.first_stage_model.decode
        self.config = config
        self.tensorrt_context = TensorRTEngineContext()
        self.tensorrt_module = None

    def warmup(self, samples_in):
        warmup_samples = torch.zeros(
            (
                1,
                samples_in.shape[1],
                int(self.config.keep_height / 8),
                int(self.config.keep_width / 8),
            ),
            device=samples_in.device,
            dtype=samples_in.dtype,
        )

        self(warmup_samples)

    def __call__(self, samples_in):
        if self.tensorrt_module is None:
            self.tensorrt_module = CallableTensorRTEngineWrapperDynamicShapeVAEDecode(
                self.tensorrt_context, ""
            )
            self.warmup(samples_in)

        self.tensorrt_context.cuda_stream = torch.cuda.current_stream()
        self.tensorrt_context.cuda_device = samples_in.device
        self.tensorrt_context.dtype = samples_in.dtype

        batch_number = 1
        pixel_samples = torch.empty(
            (
                samples_in.shape[0],
                3,
                round(samples_in.shape[2] * 8),
                round(samples_in.shape[3] * 8),
            ),
            device=samples_in.device,
        )
        for x in range(0, samples_in.shape[0], batch_number):
            samples = samples_in[x : x + batch_number]
            pixel_samples[x : x + batch_number] = self.tensorrt_module(
                VAEDecodeModule(self.model.first_stage_model, self.org_decode),
                samples=samples,
            )
        return pixel_samples


class ApplyTensorRTVaeDecoder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
                "enable_cuda_graph": ("BOOLEAN", {"default": False}),
                "keep_width": (
                    "INT",
                    {"default": 768, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8},
                ),
                "keep_height": (
                    "INT",
                    {"default": 768, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8},
                ),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "apply_tensorrt"

    CATEGORY = "loaders"

    def apply_tensorrt(
        self,
        vae,
        enable_cuda_graph,
        keep_width,
        keep_height,
    ):
        # hook comfy/sd.py#VAE.patcher
        config = TensorRTEngineConfig(
            enable_cuda_graph=enable_cuda_graph,
            keep_width=keep_width,
            keep_height=keep_height,
        )
        patch = VAEDecodeTensorRTPatch(vae, config)
        vae_tensor_rt = copy.copy(vae)
        vae_tensor_rt.patcher = vae_tensor_rt.patcher.clone()
        vae_tensor_rt.patcher.add_object_patch("decode", patch)
        return (vae_tensor_rt,)


class ControlNetTensorRTPatch:
    def __init__(self, control_model, config):
        self.control_model = control_model
        self.config = config
        self.tensorrt_context = TensorRTEngineContext()
        self.tensorrt_module = None
        self.dtype = torch.float16

    def state_dict(self):
        return self.control_model.state_dict()

    def to(self, device):
        return self.control_model.to(device)

    def warmup(self, x, hint, timesteps, context, y=None):
        warmup_x = torch.zeros(
            (
                self.config.keep_batch_size * 2,
                x.shape[1],
                int(self.config.keep_height / 8),
                int(self.config.keep_width / 8),
            ),
            device=x.device,
            dtype=x.dtype,
        )
        warmup_hint = torch.zeros(
            (
                self.config.keep_batch_size,
                hint.shape[1],
                self.config.keep_height,
                self.config.keep_width,
            ),
            device=hint.device,
            dtype=hint.dtype,
        )
        warmup_timesteps = torch.ones(
            (self.config.keep_batch_size * 2,),
            device=timesteps.device,
            dtype=timesteps.dtype,
        )
        warmup_context = torch.zeros(
            (
                self.config.keep_batch_size * 2,
                self.config.keep_embedding_block * 77,
                context.shape[2],
            ),
            device=context.device,
            dtype=context.dtype,
        )

        self(warmup_x, warmup_hint, warmup_timesteps, warmup_context, y)

    def __call__(self, x, hint, timesteps, context, y=None):
        if self.tensorrt_module == None:
            self.tensorrt_module = CallableTensorRTEngineWrapperDynamicShapeControlNet(
                self.tensorrt_context, ""
            )
            self.warmup(x, hint, timesteps, context, y)

        self.tensorrt_context.cuda_stream = torch.cuda.current_stream()
        self.tensorrt_context.cuda_device = x.device
        self.tensorrt_context.dtype = x.dtype

        return self.tensorrt_module(
            self.control_model,
            x=x,
            hint=hint,
            timesteps=timesteps,
            context=context,
            y=y,
        )


class ApplyTensorRTControlNet:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "control_net": ("CONTROL_NET",),
                "enable_cuda_graph": ("BOOLEAN", {"default": True}),
                "keep_width": (
                    "INT",
                    {"default": 768, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8},
                ),
                "keep_height": (
                    "INT",
                    {"default": 768, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8},
                ),
                "keep_batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            }
        }

    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "apply_tensorrt"

    CATEGORY = "loaders"

    def apply_tensorrt(
        self,
        control_net,
        enable_cuda_graph,
        keep_width,
        keep_height,
        keep_batch_size,
    ):
        # hook comfy/controlnet.py#ControlNet.control_model_wrapped
        config = TensorRTEngineConfig(
            enable_cuda_graph=enable_cuda_graph,
            keep_width=keep_width,
            keep_height=keep_height,
            keep_batch_size=keep_batch_size,
        )
        patch = ControlNetTensorRTPatch(control_net.control_model, config)
        control_net_tensor_rt = copy.copy(control_net)
        control_net_tensor_rt.control_model = patch
        control_net_tensor_rt = control_net_tensor_rt.copy()
        return (control_net_tensor_rt,)
