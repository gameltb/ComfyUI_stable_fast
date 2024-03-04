import torch

from .tensorrt_wrapper import CallableTensorRTEngineWrapper


class CallableTensorRTEngineWrapperDynamicShapeBaseModelApplyModel(
    CallableTensorRTEngineWrapper
):
    args_name = [
        "input_x",
        "timestep",
        "c_concat",
        "c_crossattn",
        "y",
        "control",
    ]

    def gen_onnx_args(self, kwargs, module=None):
        dynamic_axes = {
            "input_x": {0: "B", 2: "H", 3: "W"},
            "timestep": {0: "B"},
            "c_crossattn": {0: "B", 1: "E"},
            "y": {0: "B"},
        }
        args_name = []
        args = []
        for arg_name in self.args_name:
            arg = kwargs.get(arg_name, None)
            if arg != None or not isinstance(
                module, (torch.jit.ScriptFunction, torch.jit.ScriptModule)
            ):
                args.append(arg)
                if arg != None:
                    if arg_name == "control":
                        control_params = arg
                        for key in control_params:
                            for i, v in enumerate(control_params[key]):
                                control_params_name = f"{arg_name}_{key}_{i}"
                                args_name.append(control_params_name)
                                dynamic_axes[control_params_name] = {
                                    0: "B",
                                    2: f"{control_params_name}_H",
                                    3: f"{control_params_name}_W",
                                }
                    else:
                        args_name.append(arg_name)
        if not isinstance(module, (torch.jit.ScriptFunction, torch.jit.ScriptModule)):
            args.append({})
        for k in list(dynamic_axes.keys()):
            if not k in args_name:
                dynamic_axes.pop(k)
        return args, args_name, dynamic_axes

    def gen_tensorrt_args(self, kwargs):
        input_shape_info = {}
        feed_dict = {}
        for arg_name in self.args_name:
            arg = kwargs.get(arg_name, None)
            if arg != None:
                if arg_name == "control":
                    control_params = arg
                    for key in control_params:
                        for i, v in enumerate(control_params[key]):
                            control_params_name = f"{arg_name}_{key}_{i}"
                            feed_dict[control_params_name] = v
                            input_shape_info[control_params_name] = tuple(v.shape)
                else:
                    feed_dict[arg_name] = arg
                    input_shape_info[arg_name] = tuple(arg.shape)

        return feed_dict, input_shape_info

    def gen_tensorrt_args_profile(self, input_shape_info):
        min_input_profile_info = {
            "input_x": {0: 1, 2: 2, 3: 2},
            "timestep": {0: 1},
            "c_crossattn": {0: 1, 1: 77},
            "y": {0: 1},
        }
        input_profile_info = {}
        for arg_name, shape_info in input_shape_info.items():
            if arg_name.startswith("control"):
                min_shape_config = {0: 1, 2: 1, 3: 1}
            else:
                min_shape_config = min_input_profile_info.get(arg_name, None)
            min_shape_info = list(shape_info)
            if min_shape_config != None:
                for k, v in min_shape_config.items():
                    min_shape_info[k] = v
            input_profile_info[arg_name] = [
                tuple(min_shape_info),
                shape_info,
                shape_info,
            ]

        return input_profile_info
