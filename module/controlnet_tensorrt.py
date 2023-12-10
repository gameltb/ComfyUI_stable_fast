from .tensorrt_wrapper import CallableTensorRTEngineWrapper


class CallableTensorRTEngineWrapperDynamicShapeControlNet(
    CallableTensorRTEngineWrapper
):
    args_name = ["x", "hint", "timesteps", "context", "y"]

    def gen_onnx_args(self, kwargs):
        args_name = []
        args = []
        for arg_name in self.args_name:
            args.append(kwargs.get(arg_name, None))
            if args[-1] != None:
                args_name.append(arg_name)
        dynamic_axes = {
            "x": {0: "B", 2: "H", 3: "W"},
            "hint": {0: "HB", 2: "8H", 3: "8W"},
            "timesteps": {0: "B"},
            "context": {0: "B", 1: "77E"},
        }
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
                feed_dict[arg_name] = arg
                input_shape_info[arg_name] = tuple(arg.shape)

        return feed_dict, input_shape_info

    def gen_tensorrt_args_profile(self, input_shape_info):
        min_input_profile_info = {
            "x": {0: 1, 2: 8, 3: 8},
            "hint": {0: 1, 2: 64, 3: 64},
            "timesteps": {0: 1},
            "context": {0: 1, 1: 77},
        }
        input_profile_info = {}
        for arg_name in self.args_name:
            shape_info = input_shape_info.get(arg_name, None)
            min_shape_config = min_input_profile_info.get(arg_name, None)
            if shape_info != None:
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

    def gen_onnx_outputs(self, module):
        outputs_name = []
        for i in range(len(module.input_blocks) + 1):
            outputs_name.append(f"output_{i}")
        self.outputs_name = outputs_name
        return outputs_name

    def gen_tensorrt_outputs(self, output_map):
        output = []
        for output_name in self.outputs_name:
            output.append(output_map[output_name])
        return output
