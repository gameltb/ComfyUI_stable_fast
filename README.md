# ComfyUI_stable_fast

Experimental usage of [stable-fast](https://github.com/chengzeyi/stable-fast) and TensorRT.

[Speed Test](##speed-test)

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

Run ComfyUI with `--disable-xformers --force-fp16 --fp16-vae` and use `Apply TensorRT Unet` like `Apply StableFast Unet`.  
The Engine will be cached in `tensorrt_engine_cache`.

> [!NOTE]
>
> - If you encounter an error after updating, you can try deleting the `tensorrt_engine_cache`.

# Table

## Features

|                  | Stable Fast           | TensorRT(UNET) | TensorRT(UNET_BLOCK) |
| ---------------- | --------------------- | -------------- | -------------------- |
| SD1.5            | &check;               | &check;        | &check;              |
| SDXL             | untested(Should work) | WIP            | untested             |
| SSD-1B           | &check;               | WIP            | &check;              |
| Lora             | &check;               | &check;        | &check;              |
| ControlNet Unet  | &check;               | &check;        | &check;              |
| VAE decode       | WIP                   | &check;        | &check;              |
| ControlNet Model | WIP                   | WIP            | WIP                  |

## Nodes Tested

|                        | Stable Fast | TensorRT(UNET) | TensorRT(UNET_BLOCK) |
| ---------------------- | ----------- | -------------- | -------------------- |
| Load LoRA              | &check;     | &check;        | &check;              |
| FreeU(FreeU_V2)        | &check;     | &cross;        | &check;              |
| PatchModelAddDownscale | &check;     | WIP            | &check;              |

## Speed Test

### GeForce RTX 3060 Mobile

GeForce RTX 3060 Mobile (80W) 6GB, Linux , torch 2.1.1, stable fast 0.0.14, tensorrt 9.2.0.post12.dev5, xformers 0.0.23.  
[workflow](./tests/workflow.json): SD1.5, 512x512 bantch_size 1, euler_ancestral karras, 20 steps, use fp16.

Test Stable Fast and xformers run ComfyUI with `--disable-cuda-malloc`.  
Test TensorRT and pytorch run ComfyUI with `--disable-xformers`.

For the TensorRT first launch, it will take up to 10 minutes to build the engine; with timing cache, it will reduce to about 2–3 minutes; with engine cache, it will reduce to about 20–30 seconds for now.

#### Avg it/s

|                               | Stable Fast(enable_cuda_graph) | TensorRT(UNET) | TensorRT(UNET_BLOCK) | pytorch cross attention | xformers |
| ----------------------------- | ------------------------------ | -------------- | -------------------- | ----------------------- | -------- |
|                               | 10.10 it/s                     | 10.95it/s      | 10.66it/s            | 7.02it/s                | 7.90it/s |
| enable FreeU                  | 9.42 it/s                      | &cross;        | 10.04it/s            | 6.75it/s                | 7.54it/s |
| enable PatchModelAddDownscale | 10.81 it/s                     | &cross;        | 11.30it/s            | 7.46it/s                | 8.41it/s |

#### Avg time spent

| workflow                      | Stable Fast(enable_cuda_graph) | TensorRT(UNET) | TensorRT(UNET_BLOCK) | pytorch cross attention | xformers |
| ----------------------------- | ------------------------------ | -------------- | -------------------- | ----------------------- | -------- |
|                               | 2.21s (first 17s)              | 2.05s          | 2.10s                | 3.06s                   | 2.76s    |
| enable FreeU                  | 2.35s (first 18.5s)            | &cross;        | 2.24s                | 3.18s                   | 2.88     |
| enable PatchModelAddDownscale | 2.08s (first 31.37s)           | &cross;        | 2.03s                | 2.89s                   | 2.61s    |

# Screenshot

![sd1.5](asset/scr.png)
![ssd-1b](asset/scr1.png)
