# InternImage Wrapper

This repository provides an implementation for calling the "InternImage" model in a way similar to the timm library.
You can use InternImage as feature extractor.

## Install

Before installing this repository, you need to install torch, timm, and CUDA
to build CUDA ext_module.

```bash
pip install torch timm
python setup.py build install --user
```

## Provided Model

- internimage_t_1k_224
- internimage_s_1k_224
- internimage_b_1k_224
- internimage_l_22kto1k_384
- internimage_xl_22kto1k_384
- internimage_h_22kto1k_384
- internimage_h_22kto1k_640
- internimage_g_22kto1k_512

## Usage

```python
import torch
import internimage

model = internimage.create_model("internimage_b_1k_224")
model = internimage.create_model("internimage_b_1k_224", features_only=True)
model = internimage.create_model("internimage_b_1k_224", features_only=True, out_indices=[2, 3])
```

## References

- [Official Implementation](https://github.com/OpenGVLab/InternImage)
- [paper](https://arxiv.org/abs/2211.05778)
- [timm](https://github.com/huggingface/pytorch-image-models)