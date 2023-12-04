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


def hash_arg(arg):
    # micro optimization: bool obj is an instance of int
    if isinstance(arg, (str, int, float, bytes)):
        return arg
    if isinstance(arg, (tuple, list)):
        return tuple(map(hash_arg, arg))
    if isinstance(arg, dict):
        return tuple(
            sorted(
                ((hash_arg(k), hash_arg(v)) for k, v in arg.items()), key=lambda x: x[0]
            )
        )
    return type(arg)


class BaseModelApplyModel:
    def __init__(self, model_function, args, kwargs) -> None:
        self.model_function = model_function
        self.unet_config = model_function.__self__.model_config.unet_config
        self.args = args
        self.kwargs = kwargs
        self.patch_module = {}
        self.patch_module_parameter = {}

    def convert_args(self):
        transformer_options = self.kwargs.get("transformer_options", {})
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

        self.patch_module = patch_module
        self.patch_module_parameter = patch_module_parameter
        return self.args, self.kwargs

    def gen_cache_key(self, shape_info={}):
        key_kwargs = {}
        for k, v in self.kwargs.items():
            if k == "transformer_options":
                nv = {}
                for tk, tv in v.items():
                    if not tk in ("patches"):  # ,"cond_or_uncond"
                        nv[tk] = tv
                v = nv
            key_kwargs[k] = v

        patch_module_cache_key = {}
        for patch_type_name, patch_list in self.patch_module.items():
            patch_module_cache_key[patch_type_name] = []
            for patch in patch_list:
                patch_module_cache_key[patch_type_name].append(patch.gen_cache_key())

        return (
            hash_arg(self.unet_config),
            hash_arg(self.args),
            hash_arg(key_kwargs),
            hash_arg(patch_module_cache_key),
            hash_arg(shape_info),
        )

    def do_with_convert_module(self, func):
        if len(self.patch_module) > 0:
            self.model_function.__self__.diffusion_model = PatchUNetModel.cast_from(
                self.model_function.__self__.diffusion_model
            )
            try:
                self.model_function.__self__.diffusion_model.set_patch_module(self.patch_module)

                result = func(self.model_function, self.args, self.kwargs)
            finally:
                self.model_function.__self__.diffusion_model = (
                    self.model_function.__self__.diffusion_model.cast_to_base_model()
                )
        else:
            result = func(self.model_function, self.args, self.kwargs)

        return result
