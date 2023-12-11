import torch as th

from .tensorrt_wrapper import CallableTensorRTEngineWrapper


class VAEDecodeModule(th.nn.Module):
    def __init__(self, module, decode):
        super().__init__()
        self.module = module
        self.decode = decode

    def forward(self, samples):
        return self.decode(samples)


class CallableTensorRTEngineWrapperDynamicShapeVAEDecode(CallableTensorRTEngineWrapper):
    args_name = [
        "samples",
    ]

    def gen_onnx_args(self, kwargs):
        args_name = []
        args = []
        for arg_name in self.args_name:
            args.append(kwargs.get(arg_name, None))
            if args[-1] != None:
                args_name.append(arg_name)
        dynamic_axes = {
            "samples": {2: "H", 3: "W"},
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
            "samples": {2: 2, 3: 2},
        }
        input_profile_info = {}
        for arg_name, shape_info in input_shape_info.items():
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
