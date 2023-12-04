from .node import ApplyStableFastUnet

NODE_CLASS_MAPPINGS = {
    "ApplyStableFastUnet": ApplyStableFastUnet,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyStableFastUnet": "Apply StableFast Unet",
}

try:
    from .tensorrt_node import ApplyTensorRTUnet

    TRT_NODE_CLASS_MAPPINGS = {
        "ApplyTensorRTUnet": ApplyTensorRTUnet,
    }
    TRT_NODE_DISPLAY_NAME_MAPPINGS = {
        "ApplyTensorRTUnet": "Apply TensorRT Unet",
    }
    NODE_CLASS_MAPPINGS.update(TRT_NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(TRT_NODE_DISPLAY_NAME_MAPPINGS)
except Exception as e:
    print("ComfyUI_stable_fast: tensorrt_node import failed.")
