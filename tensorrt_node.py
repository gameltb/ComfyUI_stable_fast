import torch
import os
from torch.cuda import nvtx
from dataclasses import dataclass
from io import BytesIO
import hashlib
import time

from .module.stable_diffusion_pipeline_compiler import (
    gen_comfy_unet_cache_key,
    convert_comfy_args,
    to_module,
)
from .module.tensorrt_utilities import Engine


@dataclass
class TensorRTEngineCacheItem:
    engine: object
    patch_id: int
    device: str


TIMING_CACHE_PATH = os.path.join(
    os.path.dirname(__file__), "tensorrt_engine_cache", "timing_cache.cache"
)
if not os.path.exists(TIMING_CACHE_PATH):
    with open(TIMING_CACHE_PATH, "wb") as f:
        pass


def get_key_hash(key):
    return hashlib.sha256(str(key).encode()).hexdigest()


def get_engine_path(key):
    engine_cache_dir = os.path.join(os.path.dirname(__file__), "tensorrt_engine_cache")
    if not os.path.exists(engine_cache_dir):
        os.makedirs(engine_cache_dir, exist_ok=True)
    basename = hashlib.sha256(str(key).encode()).hexdigest()
    return os.path.join(engine_cache_dir, basename + ".trt")


def get_engine_with_cache(key):
    engine_path = get_engine_path(key)
    if os.path.exists(engine_path):
        return Engine(engine_path)
    return None


def gen_engine(key, onnx_buff, input_profile):
    engine = Engine(get_engine_path(key))
    s = time.time()
    engine.build(
        onnx_buff,
        fp16=True,
        enable_refit=True,
        timing_cache=TIMING_CACHE_PATH,
        input_profile=[input_profile],
    )
    e = time.time()
    print(f"Time taken to build: {e-s}s")
    return engine


def gen_onnx_module(model_function, args, input_names, output_names, onnx_output):
    model_function_module = to_module(model_function)
    # script_module = torch.jit.trace(model_function, example_inputs=args, example_kwarg_inputs=kwargs)

    torch.onnx.export(
        model_function_module,
        (*args,),
        onnx_output,
        export_params=True,
        verbose=True,
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

    def __call__(self, model_function, params):
        input_x = params.get("input")
        timestep_ = params.get("timestep")
        c = params.get("c")

        # disable with accelerate for now
        if hasattr(model_function.__self__, "hf_device_map"):
            return model_function(input_x, timestep_, **c)

        nvtx.range_push("args")
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

        for kwarg_name in ["c_concat", "c_crossattn"]:
            kwarg = c.get(kwarg_name, None)
            onnx_args.append(kwarg)
            if kwarg != None:
                onnx_input_names.append(kwarg_name)
                profile_shape_info[kwarg_name] = [
                    tuple(kwarg.shape),
                    tuple(kwarg.shape),
                    tuple(kwarg.shape),
                ]
                feed_dict[kwarg_name] = c[kwarg_name].float()

        control = c.get("control", None)
        onnx_args.append(control)
        if control != None:
            name_list, shape_info, control_params = gen_control_params(control)
            onnx_input_names.extend(name_list)
            profile_shape_info.update(shape_info)
            feed_dict.update(control_params)

        onnx_args.append({})
        nvtx.range_pop()

        if self.engine == None or self.profile_shape_info != profile_shape_info:
            unet_config = model_function.__self__.model_config.unet_config

            patch_module = convert_comfy_args((input_x, timestep_), c)
            key = gen_comfy_unet_cache_key(
                unet_config, (input_x, timestep_), c, patch_module, profile_shape_info
            )

            self.engine = get_engine_with_cache(key)

            if self.onnx_buff == None:
                self.onnx_buff = BytesIO()
                gen_onnx_module(
                    model_function,
                    onnx_args,
                    onnx_input_names,
                    onnx_output_names,
                    self.onnx_buff,
                )

            nvtx.range_push("offload origin model")
            model_function.__self__.to(device="cpu")
            nvtx.range_pop()

            if self.engine == None:
                self.engine = gen_engine(
                    key, self.onnx_buff.getvalue(), profile_shape_info
                )
                self.onnx_buff.seek(0)
                self.engine.refit_simple(self.onnx_buff, reset_zero=True)
                self.engine.save_engine()
                self.deactivate()
                self.engine = get_engine_with_cache(key)

            nvtx.range_push("load engine")
            self.activate()
            nvtx.range_push("refit engine")
            self.onnx_buff.seek(0)
            self.engine.refit_simple(self.onnx_buff)
            nvtx.range_pop()
            self.profile_shape_info = profile_shape_info
            nvtx.range_pop()

        nvtx.range_push("forward")
        tmp_buff = torch.empty(
            self.engine_vram_req, dtype=torch.uint8, device=input_x.device
        )
        self.engine.context.device_memory = tmp_buff.data_ptr()
        self.cudaStream = torch.cuda.current_stream().cuda_stream
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
        patch = TensorRTPatch(model, None)
        model_stable_fast = model.clone()
        model_stable_fast.set_model_unet_function_wrapper(patch)
        return (model_stable_fast,)
