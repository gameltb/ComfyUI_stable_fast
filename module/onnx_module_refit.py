import logging
from collections import OrderedDict
from dataclasses import asdict, dataclass

import onnx
import torch
from onnx import helper, numpy_helper

_logger = logging.getLogger(__name__)


@dataclass
class ParamsDictGenMapValue:
    op: str
    args: list


def make_module_onnx_tensor_gen_map_by_params_dict(
    module: torch.nn.Module, params_dict: dict[str, torch.Tensor]
):
    params_dict_gen_map = {}

    params_dict_dataptr_map = {v.data_ptr(): k for k, v in params_dict.items()}

    not_found_state_dict_list = []
    for k, v in module.state_dict().items():
        if v.data_ptr() in params_dict_dataptr_map:
            params_dict_key = params_dict_dataptr_map[v.data_ptr()]
            assert params_dict_key not in params_dict_gen_map
            if params_dict[params_dict_key].shape == v.shape:
                params_dict_gen_map[params_dict_key] = asdict(
                    ParamsDictGenMapValue("rename", [k])
                )
                # torch.testing.assert_close()
            elif params_dict[params_dict_key].squeeze().shape == v.shape:
                params_dict_gen_map[params_dict_key] = asdict(
                    ParamsDictGenMapValue(
                        "reshape", [k, list(params_dict[params_dict_key].shape)]
                    )
                )
                # torch.testing.assert_close()
            elif params_dict[params_dict_key].transpose(0, 1).shape == v.shape:
                params_dict_gen_map[params_dict_key] = asdict(
                    ParamsDictGenMapValue("transpose", [k, [0, 1]])
                )
                # torch.testing.assert_close()
            else:
                assert False, (
                    k,
                    v.shape,
                    params_dict_key,
                    params_dict[params_dict_key].shape,
                )
        else:
            not_found_state_dict_list.append(k)

    not_found_key_set = set(params_dict.keys()) - set(params_dict_gen_map.keys())
    for not_found_key in not_found_key_set:
        _logger.warning(not_found_key)
    assert len(not_found_key_set) == 0
    return params_dict_gen_map


def make_module_onnx_tensor_gen_map_by_onnx_model(
    module: torch.nn.Module,
    onnx_model: str,
) -> dict:
    # TODO

    return params_dict_gen_map


def make_params_dict_by_module(
    module: torch.nn.Module, params_dict_gen_map: dict[str, dict]
):
    params_dict = {}

    module_state_dict: dict[str, torch.Tensor] = module.state_dict()

    op_map = {
        "rename": lambda name: module_state_dict[name],
        "reshape": lambda name, shape: module_state_dict[name].reshape(tuple(shape)),
        "transpose": lambda name, dims: module_state_dict[name].transpose(*dims),
    }

    for k, v in params_dict_gen_map.items():
        op = v["op"]
        args = v["args"]

        params_dict[k] = op_map[op](*args)

    return params_dict


def make_constant_params_dict_by_onnx_model(
    onnx_model_path,
):
    constant_params_dict = {}

    onnx_model = onnx.load(onnx_model_path)
    for node in onnx_model.graph.node:
        if node.op_type == "Constant":
            for output in node.output:
                if "Constant" in output:
                    attrs = OrderedDict(
                        (a.name, helper.get_attribute_value(a)) for a in node.attribute
                    )
                    ndarry = numpy_helper.to_array(attrs["value"])
                    try:
                        constant_params_dict[output] = torch.Tensor(ndarry.copy())
                    except Exception:
                        print(output, ndarry)
                        continue

    return constant_params_dict
