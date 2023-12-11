#
# Copyright 2022 The HuggingFace Inc. team.
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import copy
from collections import OrderedDict
from enum import Enum, auto
from logging import error, warning

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import tensorrt as trt
import torch
import zstandard
from polygraphy import util
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import (
    ModifyNetworkOutputs,
    Profile,
    bytes_from_engine,
    engine_from_bytes,
    engine_from_network,
    network_from_onnx_bytes,
    network_from_onnx_path,
    save_engine,
)
from polygraphy.logger import G_LOGGER
from safetensors.numpy import load_file, save_file
from torch.cuda import nvtx
from tqdm import tqdm

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
G_LOGGER.module_severity = G_LOGGER.VERBOSE

# Map of numpy dtype -> torch dtype
numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool

# Map of torch dtype -> numpy dtype
torch_to_numpy_dtype_dict = {
    value: key for (key, value) in numpy_to_torch_dtype_dict.items()
}


class TQDMProgressMonitor(trt.IProgressMonitor):
    def __init__(self):
        trt.IProgressMonitor.__init__(self)
        self._active_phases = {}
        self._step_result = True
        self.max_indent = 5

    def phase_start(self, phase_name, parent_phase, num_steps):
        leave = False
        try:
            if parent_phase is not None:
                nbIndents = (
                    self._active_phases.get(parent_phase, {}).get(
                        "nbIndents", self.max_indent
                    )
                    + 1
                )
                if nbIndents >= self.max_indent:
                    return
            else:
                nbIndents = 0
                leave = True
            self._active_phases[phase_name] = {
                "tq": tqdm(
                    total=num_steps, desc=phase_name, leave=leave, position=nbIndents
                ),
                "nbIndents": nbIndents,
                "parent_phase": parent_phase,
            }
        except KeyboardInterrupt:
            # The phase_start callback cannot directly cancel the build, so request the cancellation from within step_complete.
            _step_result = False

    def phase_finish(self, phase_name):
        try:
            if phase_name in self._active_phases.keys():
                self._active_phases[phase_name]["tq"].update(
                    self._active_phases[phase_name]["tq"].total
                    - self._active_phases[phase_name]["tq"].n
                )

                parent_phase = self._active_phases[phase_name].get("parent_phase", None)
                while parent_phase is not None:
                    self._active_phases[parent_phase]["tq"].refresh()
                    parent_phase = self._active_phases[parent_phase].get(
                        "parent_phase", None
                    )
                if (
                    self._active_phases[phase_name]["parent_phase"]
                    in self._active_phases.keys()
                ):
                    self._active_phases[
                        self._active_phases[phase_name]["parent_phase"]
                    ]["tq"].refresh()
                del self._active_phases[phase_name]
            pass
        except KeyboardInterrupt:
            _step_result = False

    def step_complete(self, phase_name, step):
        try:
            if phase_name in self._active_phases.keys():
                self._active_phases[phase_name]["tq"].update(
                    step - self._active_phases[phase_name]["tq"].n
                )
            return self._step_result
        except KeyboardInterrupt:
            # There is no need to propagate this exception to TensorRT. We can simply cancel the build.
            return False


class Engine:
    def __init__(self, engine_path, enable_cuda_graph=False):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.shared_device_memory = None

        self.enable_cuda_graph = enable_cuda_graph
        self.cuda_graph_instance = None  # cuda graph
        self.inferred = False
        self.cuda_graph_stream = None

        self.refited_engine_byte = None

    def __del__(self):
        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def refit_simple(self, onnx_path, dump_refit_path=None, reset_zero=False):
        def convert_int64(arr):
            # TODO: smarter conversion
            if len(arr.shape) == 0:
                return np.int32(arr)
            return arr

        def add_to_map(refit_dict, name, values):
            if name in refit_dict:
                assert refit_dict[name] is None
                if values.dtype == np.int64:
                    values = convert_int64(values)
                refit_dict[name] = values

        print(f"Refitting TensorRT engine with {onnx_path} weights")

        # Construct refit dictionary
        refit_dict = {}
        refitter = trt.Refitter(self.engine, TRT_LOGGER)
        all_weights = refitter.get_all()
        for layer_name, role in zip(all_weights[0], all_weights[1]):
            # for specialized roles, use a unique name in the map:
            if role == trt.WeightsRole.KERNEL:
                name = layer_name + "_TRTKERNEL"
            elif role == trt.WeightsRole.BIAS:
                name = layer_name + "_TRTBIAS"
            else:
                name = layer_name

            assert name not in refit_dict, "Found duplicate layer: " + name
            refit_dict[name] = None

        for n in gs.import_onnx(onnx.load(onnx_path)).toposort().nodes:
            # Constant nodes in ONNX do not have inputs but have a constant output
            if n.op == "Constant":
                name = n.outputs[0].name
                if isinstance(n.outputs[0], gs.Constant):
                    try:
                        add_to_map(refit_dict, name, n.outputs[0].values)
                    except:
                        error(f"Failed to add Constant {name}")

            # Handle scale and bias weights
            elif n.op == "Conv":
                if n.inputs[1].__class__ == gs.Constant:
                    name = n.name + "_TRTKERNEL"
                    try:
                        add_to_map(refit_dict, name, n.inputs[1].values)
                    except:
                        error(f"Failed to add Conv {name}")

                if n.inputs[2].__class__ == gs.Constant:
                    name = n.name + "_TRTBIAS"
                    try:
                        add_to_map(refit_dict, name, n.inputs[2].values)
                    except:
                        error(f"Failed to add Conv {name}")

            # For all other nodes: find node inputs that are initializers (AKA gs.Constant)
            else:
                for inp in n.inputs:
                    name = inp.name
                    if inp.__class__ == gs.Constant:
                        try:
                            add_to_map(refit_dict, name, inp.values)
                        except:
                            error(f"Failed to add Constant {name}")

        if dump_refit_path is not None:
            print("Finished refit. Dumping result to disk.")
            save_file(
                refit_dict, dump_refit_path
            )  # TODO need to come up with delta system to save only changed weights
            return

        for layer_name, weights_role in zip(all_weights[0], all_weights[1]):
            if weights_role == trt.WeightsRole.KERNEL:
                custom_name = layer_name + "_TRTKERNEL"
            elif weights_role == trt.WeightsRole.BIAS:
                custom_name = layer_name + "_TRTBIAS"
            else:
                custom_name = layer_name

            # Skip refitting Trilu for now; scalar weights of type int64 value 1 - for clip model
            if layer_name.startswith("onnx::Trilu"):
                continue

            if refit_dict[custom_name] is not None:
                refitter.set_weights(
                    layer_name,
                    weights_role,
                    refit_dict[custom_name]
                    if not reset_zero
                    else np.zeros_like(refit_dict[custom_name]),
                )
            else:
                print(f"[W] No refit weights for layer: {layer_name}")

        if not refitter.refit_cuda_engine():
            raise Exception("Failed to refit!")

    def build(
        self,
        onnx_path,
        dtype,
        input_profile=None,
        enable_refit=False,
        enable_preview=False,
        enable_all_tactics=False,
        timing_cache=None,
        update_output_names=None,
    ):
        print(f"Building TensorRT engine for : {self.engine_path}")
        config_kwargs = {}
        if not enable_all_tactics:
            config_kwargs["tactic_sources"] = []

        if type(onnx_path) == bytes:
            network = network_from_onnx_bytes(
                onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM]
            )
        else:
            network = network_from_onnx_path(
                onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM]
            )
        if update_output_names:
            print(f"Updating network outputs to {update_output_names}")
            network = ModifyNetworkOutputs(network, update_output_names)

        input_names = set()
        nd = network[1]
        for i in range(nd.num_inputs):
            input_names.add(nd.get_input(i).name)

        p = [Profile()]
        if input_profile:
            p = [Profile() for i in range(len(input_profile))]
            for _p, i_profile in zip(p, input_profile):
                for name, dims in i_profile.items():
                    if not name in input_names:
                        continue
                    assert len(dims) == 3
                    _p.add(name, min=dims[0], opt=dims[1], max=dims[2])

        builder = network[0]
        config = builder.create_builder_config()
        config.progress_monitor = TQDMProgressMonitor()

        if dtype == torch.float16:
            config.set_flag(trt.BuilderFlag.FP16)
        elif dtype == torch.bfloat16:
            config.set_flag(trt.BuilderFlag.BF16)
        config.set_flag(trt.BuilderFlag.REFIT) if enable_refit else None

        # config.set_preview_feature(
        #     trt.PreviewFeature.DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805, False
        # )
        # config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS) | 1 << int(trt.TacticSource.CUBLAS_LT))

        cache = None
        try:
            with util.LockFile(timing_cache):
                timing_cache_data = util.load_file(
                    timing_cache, description="tactic timing cache"
                )
                cache = config.create_timing_cache(timing_cache_data)
        except FileNotFoundError:
            warning(
                "Timing cache file {} not found, falling back to empty timing cache.".format(
                    timing_cache
                )
            )
        if cache is not None:
            config.set_timing_cache(cache, ignore_mismatch=True)

        profiles = copy.deepcopy(p)
        for profile in profiles:
            # Last profile is used for set_calibration_profile.
            calib_profile = profile.fill_defaults(network[1]).to_trt(
                builder, network[1]
            )
            config.add_optimization_profile(calib_profile)

        try:
            self.engine = engine_from_network(
                network,
                config,
                save_timing_cache=timing_cache,
            )
        except Exception as e:
            raise Exception(f"Failed to build engine: {e}")
        self.update_binding_set()

    def save_engine(self):
        print(f"Saveing TensorRT engine: {self.engine_path}")
        with zstandard.open(self.engine_path, "wb") as zwfp:
            zwfp.write(bytes_from_engine(self.engine))

    def load(self):
        if self.refited_engine_byte != None:
            print(f"Loading TensorRT engine from byte cache.")
            self.engine = engine_from_bytes(self.refited_engine_byte)
            self.refited_engine_byte = None
        else:
            print(f"Loading TensorRT engine: {self.engine_path}")
            with zstandard.open(self.engine_path, "rb") as zrfp:
                self.engine = engine_from_bytes(zrfp.read())
        self.update_binding_set()

    def update_binding_set(self):
        self.binding_set = set()
        for idx in range(self.engine.num_io_tensors):
            self.binding_set.add(self.engine[idx])

    def unload(self):
        del self.context
        self.context = None
        del self.engine
        self.engine = None

    def offload(self):
        if self.refited_engine_byte == None:
            self.refited_engine_byte = bytes_from_engine(self.engine)
        self.unload()
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.shared_device_memory = None

        self.cuda_graph_instance = None
        self.inferred = False
        self.cuda_graph_stream = None

    def activate(self, reuse_device_memory=None):
        if reuse_device_memory:
            self.context = self.engine.create_execution_context_without_device_memory()
        #    self.context.device_memory = reuse_device_memory
        else:
            self.context = self.engine.create_execution_context()

    def allocate_buffers(
        self, shape_dict=None, device="cuda", allocate_input_buffers=True
    ):
        nvtx.range_push("allocate_buffers")
        for idx in range(self.engine.num_io_tensors):
            binding = self.engine[idx]
            if shape_dict and binding in shape_dict:
                shape = shape_dict[binding].shape
            else:
                shape = self.context.get_binding_shape(idx)
            if binding in self.tensors and self.tensors[binding].shape == shape:
                continue
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            if self.engine.binding_is_input(binding):
                self.context.set_binding_shape(idx, shape)
            if self.engine.binding_is_input(binding):
                if not allocate_input_buffers or not binding in shape_dict:
                    continue
            tensor = torch.empty(
                tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype], device=device
            )
            self.tensors[binding] = tensor
        if not self.enable_cuda_graph or self.shared_device_memory == None:
            self.shared_device_memory = torch.empty(
                self.engine.device_memory_size, dtype=torch.uint8, device=device
            )
            self.context.device_memory = self.shared_device_memory.data_ptr()
        nvtx.range_pop()

    def release_buffers(self):
        self.tensors = OrderedDict()

    def infer(self, feed_dict, stream):
        nvtx.range_push("set_tensors")
        for name, buf in feed_dict.items():
            if name in self.tensors:
                self.tensors[name].copy_(buf)
            elif name in self.binding_set:
                dtype = trt.nptype(self.engine.get_binding_dtype(name))
                self.tensors[name] = buf.to(dtype=numpy_to_torch_dtype_dict[dtype])

        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())
        nvtx.range_pop()
        nvtx.range_push("execute")
        if self.enable_cuda_graph and self.cuda_graph_instance is not None:
            self.cuda_graph_instance.replay()
        elif self.enable_cuda_graph and self.inferred:
            # capture cuda graph
            infer_graph = torch.cuda.CUDAGraph()
            self.cuda_graph_stream = torch.cuda.Stream()

            with torch.cuda.graph(infer_graph, stream=self.cuda_graph_stream):
                noerror = self.context.execute_async_v3(
                    self.cuda_graph_stream.cuda_stream
                )

            if not noerror:
                raise ValueError("ERROR: inference failed.")

            self.cuda_graph_instance = infer_graph
        else:
            noerror = self.context.execute_async_v3(stream.cuda_stream)
            if not noerror:
                raise ValueError("ERROR: inference failed.")
            self.inferred = True
        nvtx.range_pop()

        if not self.enable_cuda_graph:
            del self.shared_device_memory
            self.shared_device_memory = None

        return self.tensors

    def set_static_dict_input(self, feed_dict):
        nvtx.range_push("set_tensors")
        for name, tensor in feed_dict.items():
            dtype = trt.nptype(self.engine.get_binding_dtype(name))
            feed_dict[name] = tensor.to(dtype=numpy_to_torch_dtype_dict[dtype])
            self.context.set_tensor_address(name, feed_dict[name].data_ptr())
        nvtx.range_pop()

    def __str__(self):
        out = ""
        for opt_profile in range(self.engine.num_optimization_profiles):
            for binding_idx in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(binding_idx)
                shape = self.engine.get_profile_shape(opt_profile, name)
                out += f"\t{name} = {shape}\n"
        return out
