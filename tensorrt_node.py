import enum
import gc
from io import BytesIO

import torch
from torch.cuda import nvtx

import comfy.model_management
import comfy.model_patcher

from .module.comfy_trace_utilities import BaseModelApplyModel
from .module.openaimodel_tensorrt import (
    TENSORRT_CONTEXT_KEY,
    TensorRTEngineCacheContext,
    TensorRTEngineConfig,
    do_hook_forward_timestep_embed,
    gen_engine,
    get_engine_with_cache,
    undo_hook_forward_timestep_embed,
)


def make_BaseModelApplyModel_FuncModule(model_function):
    class BaseModelApplyModelFuncModule(torch.nn.Module):
        def __init__(self, func, module=None):
            super().__init__()
            self.func = func
            self.module = module

        def forward(
            self,
            x,
            t,
            c_concat=None,
            c_crossattn=None,
            y=None,
            control=None,
            transformer_options={},
        ):
            kwargs = {"y": y}
            return self.func(
                x,
                t,
                c_concat=c_concat,
                c_crossattn=c_crossattn,
                control=control,
                transformer_options=transformer_options,
                **kwargs,
            )

    return BaseModelApplyModelFuncModule(model_function, model_function.__self__).eval()


def gen_onnx_module(
    model_function_module, args, input_names, output_names, onnx_output
):
    # script_module = torch.jit.trace(model_function, example_inputs=args, example_kwarg_inputs=kwargs)

    torch.onnx.export(
        model_function_module,
        tuple(args),
        onnx_output,
        export_params=True,
        verbose=False,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
    )


def gen_control_params(control_params):
    root_name = "control"
    control_params_name_list = []
    control_params_name_list_shape_info = {}
    control_params_map = {}
    for key in control_params:
        for i, v in enumerate(control_params[key]):
            control_params_name = f"{root_name}_{key}_{i}"
            control_params_name_list.append(control_params_name)
            control_params_name_list_shape_info[control_params_name] = [
                tuple(v.shape),
                tuple(v.shape),
                tuple(v.shape),
            ]
            control_params_map[control_params_name] = v
    return (
        control_params_name_list,
        control_params_name_list_shape_info,
        control_params_map,
    )


class TensorRTPatch:
    tensor_rt_engine_cache = {}

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.engine = None
        self.onnx_buff = None
        self.profile_shape_info = None
        self.profile_shape_info_key_set = None

    def __call__(self, model_function, params):
        input_x = params.get("input")
        timestep_ = params.get("timestep")
        c = params.get("c")

        # disable with accelerate for now
        if hasattr(model_function.__self__, "hf_device_map"):
            return model_function(input_x, timestep_, **c)

        nvtx.range_push("args")
        module = BaseModelApplyModel(model_function, (input_x, timestep_), c)
        args, kwargs = module.convert_args()

        onnx_args = [input_x, timestep_]
        onnx_input_names = ["input_x", "timestep"]
        onnx_output_names = ["output"]
        profile_shape_info = {
            "input_x": [
                tuple(input_x.shape),
                tuple(input_x.shape),
                tuple(input_x.shape),
            ],
            "timestep": [
                tuple(timestep_.shape),
                tuple(timestep_.shape),
                tuple(timestep_.shape),
            ],
        }
        feed_dict = {
            "input_x": input_x.float(),
            "timestep": timestep_.float(),
        }

        for kwarg_name in ["c_concat", "c_crossattn", "y"]:
            kwarg = kwargs.get(kwarg_name, None)
            onnx_args.append(kwarg)
            if kwarg != None:
                onnx_input_names.append(kwarg_name)
                profile_shape_info[kwarg_name] = [
                    tuple(kwarg.shape),
                    tuple(kwarg.shape),
                    tuple(kwarg.shape),
                ]
                feed_dict[kwarg_name] = kwarg.float()

        control = kwargs.get("control", None)
        onnx_args.append(control)
        if control != None:
            name_list, shape_info, control_params = gen_control_params(control)
            onnx_input_names.extend(name_list)
            profile_shape_info.update(shape_info)
            feed_dict.update(control_params)

        onnx_args.append({})
        nvtx.range_pop()

        if self.engine == None or self.profile_shape_info != profile_shape_info:
            self.deactivate()

            key = module.gen_cache_key(profile_shape_info)

            engine = get_engine_with_cache(key, self.config)

            if (
                self.onnx_buff == None
                or (engine == None and self.profile_shape_info != profile_shape_info)
                or self.profile_shape_info_key_set != set(profile_shape_info.keys())
            ):
                model_function.__self__.to(device=input_x.device)
                self.onnx_buff = BytesIO()
                gen_onnx_module(
                    make_BaseModelApplyModel_FuncModule(model_function),
                    onnx_args,
                    onnx_input_names,
                    onnx_output_names,
                    self.onnx_buff,
                )
                self.profile_shape_info_key_set = set(profile_shape_info.keys())

            nvtx.range_push("offload origin model")
            model_function.__self__.to(device="cpu")
            gc.collect()
            comfy.model_management.soft_empty_cache()
            nvtx.range_pop()

            if engine == None:
                engine = gen_engine(
                    key, self.onnx_buff.getvalue(), [profile_shape_info]
                )
                self.onnx_buff.seek(0)
                engine.refit_simple(self.onnx_buff, reset_zero=True)
                engine.save_engine()
                del engine
                engine = get_engine_with_cache(key, self.config)

            self.engine = engine
            try:
                nvtx.range_push("load engine")
                self.activate()
                nvtx.range_push("refit engine")
                self.onnx_buff.seek(0)
                self.engine.refit_simple(self.onnx_buff)
                nvtx.range_pop()
                self.profile_shape_info = profile_shape_info
                nvtx.range_pop()
            except Exception as e:
                self.engine = None
                raise e

        nvtx.range_push("forward")

        self.cudaStream = torch.cuda.current_stream()
        self.engine.allocate_buffers(feed_dict)

        out = self.engine.infer(feed_dict, self.cudaStream)["output"]

        nvtx.range_pop()
        return out

    def activate(self):
        nvtx.range_push("load engine byte")
        self.engine.load()
        nvtx.range_pop()
        self.engine_vram_req = self.engine.engine.device_memory_size
        nvtx.range_push("activate engine")
        self.engine.activate(True)
        nvtx.range_pop()

    def deactivate(self):
        del self.engine
        self.engine = None

    def to(self, device):
        if type(device) == torch.device:
            if device.type == "cpu" and self.engine != None:
                self.deactivate()
            # else:
            #     self.activate()
        return self


class BlockTensorRTPatch:
    tensorrt_context_cache = {}

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.model_device = torch.device("cpu")
        self.tensorrt_context_cache[id(self)] = TensorRTEngineCacheContext(
            origin_model_patcher=model
        )

    def __call__(self, model_function, params):
        input_x = params.get("input")
        timestep_ = params.get("timestep")
        c = params.get("c")

        # disable with accelerate for now
        if hasattr(model_function.__self__, "hf_device_map"):
            return model_function(input_x, timestep_, **c)

        self.tensorrt_context_cache[
            id(self)
        ].model_type = model_function.__self__.model_config.__class__.__name__
        self.tensorrt_context_cache[
            id(self)
        ].unet_config = model_function.__self__.model_config.unet_config
        self.tensorrt_context_cache[id(self)].cuda_stream = torch.cuda.current_stream()
        self.tensorrt_context_cache[id(self)].cuda_device = input_x.device
        c["transformer_options"][TENSORRT_CONTEXT_KEY] = self.tensorrt_context_cache[
            id(self)
        ]

        do_hook_forward_timestep_embed()
        try:
            out = model_function(input_x, timestep_, **c)
        finally:
            undo_hook_forward_timestep_embed()
            c["transformer_options"].pop(TENSORRT_CONTEXT_KEY)

        return out

    def to(self, device):
        if type(device) == torch.device:
            self.model_device = device
        return self

    def __del__(self):
        self.tensorrt_context_cache.pop(id(self))


def hook_memory_required(input_shape):
    return 0


class TensorRTEngineOriginModelPatcherWarper_BlockPatch(
    comfy.model_patcher.ModelPatcher
):
    @staticmethod
    def cast_from(other):
        tcls = comfy.model_patcher.ModelPatcher
        if isinstance(other, tcls):
            other.__class__ = TensorRTEngineOriginModelPatcherWarper_BlockPatch
            return other
        raise ValueError(f"instance must be {tcls.__qualname__}")

    def cast_to_base_model(self):
        self.__class__ = comfy.model_patcher.ModelPatcher
        return self

    def patch_model(self, device_to=None):
        model = super().patch_model()

        if device_to is not None:
            for name, module in model.named_children():
                if name in ("diffusion_model"):
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


class PatchType(enum.Enum):
    UNET = TensorRTPatch
    UNET_BLOCK = BlockTensorRTPatch


class ApplyTensorRTUnet:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "enable_cuda_graph": ("BOOLEAN", {"default": True}),
                "patch_type": ([e.name for e in PatchType], {"default": "UNET_BLOCK"}),
                "hook_memory_require": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_tensorrt"

    CATEGORY = "loaders"

    def apply_tensorrt(self, model, enable_cuda_graph, patch_type, hook_memory_require):
        config = TensorRTEngineConfig(enable_cuda_graph=enable_cuda_graph)
        patch = None
        for e in PatchType:
            if e.name == patch_type:
                patch = e.value(model, config)
        assert patch != None
        model_tensor_rt = model.clone()
        if isinstance(patch, BlockTensorRTPatch):
            model_tensor_rt = (
                TensorRTEngineOriginModelPatcherWarper_BlockPatch.cast_from(
                    model_tensor_rt
                )
            )
        model_tensor_rt.set_model_unet_function_wrapper(patch)
        if hook_memory_require:
            model_tensor_rt.add_object_patch("memory_required", hook_memory_required)
        return (model_tensor_rt,)
