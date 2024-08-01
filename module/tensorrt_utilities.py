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
from logging import warning

import numpy as np
import tensorrt as trt
import torch
import zstandard
from polygraphy import util
from polygraphy.backend.trt import (
    ModifyNetworkOutputs,
    Profile,
    bytes_from_engine,
    engine_from_bytes,
    engine_from_network,
    network_from_onnx_bytes,
    network_from_onnx_path,
)
from polygraphy.logger import G_LOGGER
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

        self.last_device_memory_size = 0

    def __del__(self):
        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def refit_simple(self, onnx_model):
        print(f"Refitting TensorRT engine with {onnx_model} weights")

        refitter = trt.Refitter(self.engine, TRT_LOGGER)
        parser_refitter = trt.OnnxParserRefitter(refitter, TRT_LOGGER)
        if type(onnx_model) is bytes:
            result = parser_refitter.refit_from_bytes(onnx_model)
        else:
            result = parser_refitter.refit_from_file(onnx_model)

        if not result or not refitter.refit_cuda_engine():
            raise Exception("Failed to refit!")

    def refit_from_dict(
        self,
        refit_weights: dict[str, torch.Tensor],
        constant_refit_weights: dict[str, torch.Tensor],
    ):
        # Initialize refitter
        refitter = trt.Refitter(self.engine, TRT_LOGGER)

        refitted_weights = set()
        print(f"[I] Total refittable weights {len(refitter.get_all_weights())}.")

        # iterate through all tensorrt refittable weights
        for trt_weight_name in refitter.get_all_weights():
            # get weight from state dict
            if trt_weight_name in refit_weights:
                refit_weight = refit_weights[trt_weight_name]
            elif trt_weight_name in constant_refit_weights:
                refit_weight = constant_refit_weights[trt_weight_name]
                # print(refit_weight)
            else:
                continue

            trt_datatype = refitter.get_weights_prototype(trt_weight_name).dtype
            if trt_datatype == trt.DataType.FLOAT:
                refit_weight = refit_weight.float()
            elif trt_datatype == trt.DataType.HALF:
                refit_weight = refit_weight.half()
            else:
                print("unhandled", trt_datatype)
                continue

            # trt.Weight and trt.TensorLocation
            trt_wt_tensor = trt.Weights(
                trt_datatype,
                refit_weight.data_ptr(),
                torch.numel(refit_weight),
            )
            trt_wt_location = (
                trt.TensorLocation.DEVICE
                if refit_weight.is_cuda
                else trt.TensorLocation.HOST
            )

            self.buffers[trt_weight_name] = refit_weight

            # apply refit
            assert refitter.set_named_weights(trt_weight_name, trt_wt_tensor, trt_wt_location)
            refitted_weights.add(trt_weight_name)

        # assert set(refitted_weights) == set(refit_weights.keys())
        if not refitter.refit_cuda_engine():
            print("Error: failed to refit new weights.")
            exit(0)

        print(f"[I] Total refitted weights {len(refitted_weights)}.")

    def build(
        self,
        onnx_model,
        dtype,
        input_profile=None,
        enable_refit=False,
        enable_weight_streaming=False,
        enable_all_tactics=False,
        timing_cache=None,
        update_output_names=None,
    ):
        print(f"Building TensorRT engine for : {self.engine_path}")
        config_kwargs = {}
        if not enable_all_tactics:
            config_kwargs["tactic_sources"] = []

        if type(onnx_model) is bytes:
            network = network_from_onnx_bytes(
                onnx_model,
                flags=[
                    trt.OnnxParserFlag.NATIVE_INSTANCENORM,
                ],
                strongly_typed=enable_weight_streaming,
            )
        else:
            network = network_from_onnx_path(
                onnx_model,
                flags=[
                    trt.OnnxParserFlag.NATIVE_INSTANCENORM,
                ],
                strongly_typed=enable_weight_streaming,
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
                    if name not in input_names:
                        continue
                    assert len(dims) == 3
                    _p.add(name, min=dims[0], opt=dims[1], max=dims[2])

        builder = network[0]
        config = builder.create_builder_config()
        config.progress_monitor = TQDMProgressMonitor()

        if not enable_weight_streaming:
            if dtype == torch.float16:
                config.set_flag(trt.BuilderFlag.FP16)
            elif dtype == torch.bfloat16:
                config.set_flag(trt.BuilderFlag.BF16)

        if enable_refit:
            config.set_flag(trt.BuilderFlag.STRIP_PLAN)
            # Slower than REFIT_IDENTICAL
            # config.set_flag(trt.BuilderFlag.REFIT)
            config.set_flag(trt.BuilderFlag.REFIT_IDENTICAL)

        if enable_weight_streaming:
            config.set_flag(trt.BuilderFlag.WEIGHT_STREAMING)
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
        if self.refited_engine_byte is not None:
            print("Loading TensorRT engine from byte cache.")
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
        if self.refited_engine_byte is None:
            serialization_config = self.engine.create_serialization_config()
            serialization_config.flags &= ~(
                1 << int(trt.SerializationFlag.EXCLUDE_WEIGHTS)
            )
            self.refited_engine_byte = self.engine.serialize_with_config(
                serialization_config
            )
            self.buffers.clear()
        self.unload()
        self.tensors = OrderedDict()
        self.shared_device_memory = None

        self.cuda_graph_instance = None
        self.inferred = False
        self.cuda_graph_stream = None

    def is_weight_streaming_engine(self):
        return self.engine.streamable_weights_size > 0

    def activate(
        self, reuse_device_memory=None, memory_limit_size=1000 * 1000 * 1000 * 3
    ):
        if self.context is None:
            if self.is_weight_streaming_engine():

                def update_budget_size():
                    budget_size = memory_limit_size - self.engine.device_memory_size_v2
                    if budget_size < 0:
                        budget_size = 0
                    self.engine.weight_streaming_budget_v2 = (
                        budget_size
                        if budget_size < self.engine.streamable_weights_size
                        else self.engine.streamable_weights_size
                    )

                # if weight_streaming enable , device_memory_size_v2 will change.
                update_budget_size()
                update_budget_size()

            if reuse_device_memory:
                self.context = (
                    self.engine.create_execution_context_without_device_memory()
                )
            #    self.context.device_memory = reuse_device_memory
            else:
                self.context = self.engine.create_execution_context()
            assert self.context is not None

    def get_device_memory_size(self):
        if self.engine is not None:
            if self.is_weight_streaming_engine():
                self.last_device_memory_size = (
                    self.engine.device_memory_size_v2
                    + self.engine.weight_streaming_budget_v2
                )
            else:
                self.last_device_memory_size = self.engine.device_memory_size_v2
        return self.last_device_memory_size

    def allocate_buffers(
        self, shape_dict=None, device="cuda", allocate_input_buffers=True
    ):
        nvtx.range_push("allocate_buffers")
        for idx in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(idx)

            if shape_dict and tensor_name in shape_dict:
                shape = shape_dict[tensor_name].shape
            else:
                shape = self.context.get_tensor_shape(tensor_name)
            shape = list(shape)
            if (
                tensor_name in self.tensors
                and list(self.tensors[tensor_name].shape) == shape
            ):
                continue
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(tensor_name, shape)
                if not allocate_input_buffers or tensor_name not in shape_dict:
                    continue
            tensor = torch.empty(
                tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype], device=device
            )
            self.tensors[tensor_name] = tensor
        if self.shared_device_memory is None:
            self.shared_device_memory = torch.empty(
                self.engine.device_memory_size_v2, dtype=torch.uint8, device=device
            )
            self.context.set_device_memory(
                self.shared_device_memory.data_ptr(), self.engine.device_memory_size_v2
            )
        nvtx.range_pop()

    def release_buffers(self):
        self.tensors = OrderedDict()

    def infer(
        self,
        feed_dict,
        stream: torch.cuda.Stream,
        stream_sync=False,
        free_shared_device_memory=True,
    ):
        nvtx.range_push("set_tensors")
        for name, buf in feed_dict.items():
            if name in self.tensors:
                self.tensors[name].copy_(buf)
            elif name in self.binding_set:
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
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

        if stream_sync:
            stream.synchronize()

        if not self.enable_cuda_graph and free_shared_device_memory:
            del self.shared_device_memory
            self.shared_device_memory = None

        return self.tensors

    def set_static_dict_input(self, feed_dict):
        nvtx.range_push("set_tensors")
        for name, tensor in feed_dict.items():
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            feed_dict[name] = tensor.to(dtype=numpy_to_torch_dtype_dict[dtype])
            self.context.set_tensor_address(name, feed_dict[name].data_ptr())
        nvtx.range_pop()

    def __str__(self):
        out = ""
        for opt_profile in range(self.engine.num_optimization_profiles):
            for binding_idx in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(binding_idx)
                shape = self.engine.get_tensor_profile_shape(opt_profile, name)
                out += f"\t{name} = {shape}\n"
        return out
