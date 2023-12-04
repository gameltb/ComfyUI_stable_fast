# ComfyUI_stable_fast

Experimental usage of [stable-fast](https://github.com/chengzeyi/stable-fast) and TensorRT.

# Installation

```bash
git clone https://github.com/gameltb/ComfyUI_stable_fast custom_nodes/ComfyUI_stable_fast
```

## stable-fast

[stable-fast installation](https://github.com/chengzeyi/stable-fast?tab=readme-ov-file#installation)

## TensorRT(testing)

> [!NOTE]
>
> Currently only tested on linux, Not tested on Windows.

The following needs to be installed when you use TensorRT.  
Even if you don't install these, the stable-fast node is still available.

```bash
pip install onnx zstandard
pip install --pre --upgrade --extra-index-url https://pypi.nvidia.com tensorrt
pip install onnx-graphsurgeon polygraphy --extra-index-url https://pypi.ngc.nvidia.com
```

## Usage

Please refer to the [screenshot](#screenshot)

## stable-fast

It can work with Lora, ControlNet and lcm. SD1.5 and SSD-1B are supported. SDXL should work.  
Run ComfyUI with `--disable-cuda-malloc` may be possible to optimize the speed further.

> [!NOTE]
>
> - FreeU and PatchModelAddDownscale are now supported experimentally, Just use the comfy node normally.
> - If you are using WSL, please do not install Triton for the time being due to bugs.
> - stable fast not work well with accelerate, So this node has no effect when the vram is low. For example: 6G vram card run SDXL.
> - stable fast will optimize the speed when generating images using the same model for the second time. if you switch models or Lora frequently, please consider disable enable_cuda_graph.
> - stable fast should be directly connected to ksampler, and it is better not to have other nodes between them.

## TensorRT(testing)

Run ComfyUI with `--disable-xformers` and use `Apply TensorRT Unet` like `Apply StableFast Unet`.  
The Engine will be cached in `tensorrt_engine_cache`, Each is about 2MB.

> [!NOTE]
>
> - If you encounter an error after updating, you can try deleting the `tensorrt_engine_cache`.

# Table

## Features

|                  | Stable Fast           | TensorRT |
| ---------------- | --------------------- | -------- |
| SD1.5            | &check;               | &check;  |
| SDXL             | untested(Should work) | WIP      |
| SSD-1B           | &check;               | WIP      |
| Lora             | &check;               | &check;  |
| ControlNet Unet  | &check;               | &check;  |
| VAE              | WIP                   | WIP      |
| ControlNet Model | WIP                   | WIP      |

## Nodes Tested

|                        | Stable Fast | TensorRT |
| ---------------------- | ----------- | -------- |
| Load LoRA              | &check;     | &check;  |
| FreeU(FreeU_V2)        | &check;     | &cross;  |
| PatchModelAddDownscale | &check;     | WIP      |

## Speed Test

TBD

# Screenshot

![sd1.5](asset/scr.png)
![ssd-1b](asset/scr1.png)
