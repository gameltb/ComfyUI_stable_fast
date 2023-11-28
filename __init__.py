from .node import ApplyStableFastUnet
from .tensorrt_node import ApplyTensorRTUnet

NODE_CLASS_MAPPINGS = {
    "ApplyStableFastUnet": ApplyStableFastUnet,
    "ApplyTensorRTUnet": ApplyTensorRTUnet,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyStableFastUnet": "Apply StableFast Unet",
    "ApplyTensorRTUnet": "Apply TensorRT Unet",
}
