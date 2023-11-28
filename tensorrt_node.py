import torch
import os
from torch.cuda import nvtx

from .module.stable_diffusion_pipeline_compiler import (
    gen_comfy_unet_cache_key,
    convert_comfy_args,
    to_module,
)
from .module.tensorrt_utilities import Engine

NO_PARAMS_ONNX_PATH = "/tmp/test_NO_PARAMS.onnx"
PARAMS_ONNX_PATH = "/tmp/test_PARAMS.onnx"
TRT_PATH = "/tmp/test.trt"

def gen_onnx_module(model_function, input_x, timestep_, **kwargs):
    unet_config = model_function.__self__.model_config.unet_config

    patch_module = convert_comfy_args((input_x, timestep_), kwargs)
    key = gen_comfy_unet_cache_key(
        unet_config, (input_x, timestep_), kwargs, patch_module
    )

    model_function_module = to_module(model_function)

    args = [input_x, timestep_]
    input_names = ["input_x", "timestep"]

    for kwarg_name in ["c_concat", "c_crossattn", "control"]:
        kwarg = kwargs.get(kwarg_name, None)
        args.append(kwarg)
        if kwarg != None:
            input_names.append(kwarg_name)

    # script_module = torch.jit.trace(model_function, example_inputs=args, example_kwarg_inputs=kwargs)

    if not os.path.exists(NO_PARAMS_ONNX_PATH):
        torch.onnx.export(
            model_function_module,
            (*args,),
            NO_PARAMS_ONNX_PATH,
            export_params=False,
            verbose=True,
            do_constant_folding=True,
            input_names=input_names,
            output_names=["output"],
        )

    if not os.path.exists(PARAMS_ONNX_PATH):
        torch.onnx.export(
            model_function_module,
            (*args,),
            PARAMS_ONNX_PATH,
            export_params=True,
            verbose=True,
            do_constant_folding=True,
            input_names=input_names,
            output_names=["output"],
        )


class TensorrtPatch:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.stable_fast_model = None

        if os.path.exists(TRT_PATH):
            self.engine = Engine(TRT_PATH)

    def __call__(self, model_function, params):
        input_x = params.get("input")
        timestep_ = params.get("timestep")
        c = params.get("c")

        # disable with accelerate for now
        if hasattr(model_function.__self__, "hf_device_map"):
            return model_function(input_x, timestep_, **c)

        gen_onnx_module(model_function, input_x, timestep_, **c)

        nvtx.range_push("forward")
        feed_dict = {
            "input_x": input_x.float(),
            "timestep": timestep_.float(),
        }
        for kwarg_name in ["c_concat", "c_crossattn", "control"]:
            if kwarg_name in c:
                feed_dict[kwarg_name] = c[kwarg_name].float()

        tmp = torch.empty(
            self.engine_vram_req, dtype=torch.uint8, device=input_x.device
        )
        self.engine.context.device_memory = tmp.data_ptr()
        self.cudaStream = torch.cuda.current_stream().cuda_stream
        self.engine.allocate_buffers(feed_dict)

        out = self.engine.infer(feed_dict, self.cudaStream)["output"]

        nvtx.range_pop()
        return out

    def activate(self):
        self.engine.load()
        print(self.engine)
        self.engine_vram_req = self.engine.engine.device_memory_size
        self.engine.activate(True)

    def deactivate(self):
        self.shape_hash = 0
        del self.engine

    def to(self, device):
        if type(device) == torch.device:
            if device.type == "cpu":
                self.deactivate()
            else:
                self.activate()
        return self


class ApplyTensorRTUnet:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_tensorrt"

    CATEGORY = "loaders"

    def apply_tensorrt(self, model):
        patch = TensorrtPatch(model, None)
        model_stable_fast = model.clone()
        model_stable_fast.set_model_unet_function_wrapper(patch)
        return (model_stable_fast,)
