NO_PARAMS_ONNX_PATH = "/tmp/test_NO_PARAMS.onnx"
PARAMS_ONNX_PATH = "/tmp/test_PARAMS.onnx"
TRT_PATH = "/tmp/test.trt"
from module.tensorrt_utilities import Engine
import time
engine = Engine(TRT_PATH)
s = time.time()
ret = engine.build(PARAMS_ONNX_PATH, fp16=True, enable_refit=False)
e = time.time()
print(f"Time taken to build: {e-s}s")