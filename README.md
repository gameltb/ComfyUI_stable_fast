# ComfyUI_stable_fast

Experimental usage of [stable-fast](https://github.com/chengzeyi/stable-fast) and TensorRT.

> [!NOTE]
>
> Official TensorRT node https://github.com/comfyanonymous/ComfyUI_TensorRT  
> This repo is still experimental, just want to try TensorRT that doesn't need to be compiled repeatedly.

[Speed Test](#speed-test)

# Update

- 2024-07-31 : Unfortunately, using the same engine on different models will result in a slight variation in the results or complete unusability. Added an option to allow building dedicated engines for different models. However, some models still have different outputs than PyTorch.
- 2024-07-29 : significantly improved performance of starting and switching TensorRT models when there is an engine cache on PyTorch 2.4.0. add WEIGHT_STREAMING support, you can run SDXL on 6GB device with TensorRT. However, the engine unloading caused by VAE decoding can greatly slow down the overall generation speed.

# Installation

```bash
git clone https://github.com/gameltb/ComfyUI_stable_fast custom_nodes/ComfyUI_stable_fast
```

## stable-fast

You'll need to follow the guide below to enable stable fast node.

[stable-fast installation](https://github.com/chengzeyi/stable-fast?tab=readme-ov-file#installation)

> [!NOTE]
>
> Requires stable-fast >= 1.0.0 .

## TensorRT(testing)

> [!NOTE]
>
> Currently only tested on linux, Not tested on Windows.

The following needs to be installed when you use TensorRT.

```bash
pip install onnx zstandard onnxscript --upgrade
pip install --pre --upgrade --extra-index-url https://pypi.nvidia.com tensorrt==10.2.0
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
> - stable fast not work well with accelerate, So this node has no effect when the vram is low. For example: 6G vram card run SDXL.
> - stable fast will optimize the speed when generating images using the same model for the second time. if you switch models or Lora frequently, please consider disable enable_cuda_graph.
> - **It is better to connect the `Apply StableFast Unet` node directly to the `KSampler` node, and there should be no nodes between them that will change the weight, such as the `Load LoRA` node, but for some nodes, placing it between them can prevent useless recompilation caused by modifying the node parameters, such as the `FreeU` node, you can try to use other nodes, but I can't guarantee that it will work properly.**

## TensorRT

Run ComfyUI with `--disable-xformers --force-fp16 --fp16-vae` and use `Apply TensorRT Unet` like `Apply StableFast Unet`.  
The Engine will be cached in `tensorrt_engine_cache`.

> [!NOTE]
>
> - If you encounter an error after updating, you can try deleting the `tensorrt_engine_cache`.

### Apply TensorRT Unet Node

- enable_cuda_graph
  - With or without CUDA Graph, this should make it slightly faster, but at the moment there is a problem with the implementation and this has no effect. Also, even if it works, it won't work with WEIGHT_STREAMING.
- patch_type
  - UNET compiles the whole unet as a model, and it's faster. However, some nodes are unusable because TensorRT does not support some operations in PyTorch, such as FreeU nodes. Also, if you don't have enough video memory to put down the entire model, you'll need to select this option to use TensorRT, otherwise it's likely to be slower than running directly.
  - UNET_BLOCK splits unet into several small models to allow pytorch to perform operations between them that TensorRT does not support. It takes quite a bit of time to compile and load, but the speed of completion is not much compared to XXX. It may not be acceptable to use this option most of the time.
- keep_width
- keep_height
- keep_batch_size
- keep_embedding_block
  - The parameters starting with `keep_` above are used when building the engine, and they specify the maximum value of the parameters that the engine accepts. At the same time, the node will look up the cached engine based on these values, so if you want to build the engine as few times as possible, keep a fixed set of values based on different types of models such as sd15 or sdxl. If one of the parameters you use is greater than them, it will trigger the build. embedding_block is related to the length of your prompt, and the longer the length, the greater the value.
- use_dedicated_engine
  - building dedicated engines for different models.

When you use ControlNet, different control image sizes will cause the engine to compile for now.

# Table

## Features

|                  | Stable Fast           | TensorRT(UNET) | TensorRT(UNET_BLOCK) |
| ---------------- | --------------------- | -------------- | -------------------- |
| SD1.5            | &check;               | &check;        | &check;              |
| SDXL             | untested(Should work) | &check;        | untested             |
| SSD-1B           | &check;               | &check;        | &check;              |
| Lora             | &check;               | &check;        | &check;              |
| ControlNet Unet  | &check;               | &check;        | &check;              |
| VAE decode       | WIP                   | &check;        | -                    |
| ControlNet Model | WIP                   | WIP            | -                    |

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

###### TensorRT Note

For the TensorRT first launch, it will take up to 10 minutes to build the engine; with timing cache, it will reduce to about 2–3 minutes; with engine cache, it will reduce to about 20–30 seconds for now.

#### Avg it/s

|                                  | Stable Fast (enable_cuda_graph) | TensorRT (UNET) | TensorRT (UNET_BLOCK) | pytorch cross attention | xformers |
| -------------------------------- | ------------------------------- | --------------- | --------------------- | ----------------------- | -------- |
|                                  | 10.10 it/s                      | 10.95it/s       | 10.66it/s             | 7.02it/s                | 7.90it/s |
| enable FreeU                     | 9.42 it/s                       | &cross;         | 10.04it/s             | 6.75it/s                | 7.54it/s |
| enable Patch Model Add Downscale | 10.81 it/s                      | &cross;         | 11.30it/s             | 7.46it/s                | 8.41it/s |

#### Avg time spent

| workflow                         | Stable Fast (enable_cuda_graph) | TensorRT (UNET) | TensorRT (UNET_BLOCK) | pytorch cross attention | xformers |
| -------------------------------- | ------------------------------- | --------------- | --------------------- | ----------------------- | -------- |
|                                  | 2.21s (first 17s)               | 2.05s           | 2.10s                 | 3.06s                   | 2.76s    |
| enable FreeU                     | 2.35s (first 18.5s)             | &cross;         | 2.24s                 | 3.18s                   | 2.88     |
| enable Patch Model Add Downscale | 2.08s (first 31.37s)            | &cross;         | 2.03s                 | 2.89s                   | 2.61s    |

# Screenshot

![sd1.5](asset/scr.png)
![ssd-1b](asset/scr1.png)
