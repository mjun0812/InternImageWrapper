from typing import Optional

import torch

from .model import InternImage

"""
internimage_t_1k_224
internimage_s_1k_224
internimage_b_1k_224
internimage_l_22kto1k_384
internimage_xl_22kto1k_384
internimage_h_22kto1k_384
internimage_h_22kto1k_640
internimage_g_22kto1k_512
"""

DEFAULT_CONFIG = dict(
    core_op="DCNv3",
    num_classes=1000,
    channels=64,
    depths=[4, 4, 18, 4],
    groups=[4, 8, 16, 32],
    layer_scale=None,
    offset_scale=1.0,
    post_norm=False,
    mlp_ratio=4.0,
    res_post_norm=False,  # for InternImage-H/G
    dw_kernel_size=None,  # for InternImage-H/G
    use_clip_projector=False,  # for InternImage-H/G
    level2_post_norm=False,  # for InternImage-H/G
    level2_post_norm_block_ids=None,  # for InternImage-H/G
    center_feature_scale=False,  # for InternImage-H/G
    remove_center=False,
)


def internimage_t_1k_224(**kwargs) -> dict:
    config = DEFAULT_CONFIG.copy()
    config.update(
        dict(
            channels=64,
            depths=[4, 4, 18, 4],
            groups=[4, 8, 16, 32],
            offset_scale=1.0,
            mlp_ratio=4.0,
        )
    )
    config.update(kwargs)
    return config


def internimage_s_1k_224(**kwargs) -> dict:
    config = DEFAULT_CONFIG.copy()
    config.update(
        dict(
            channels=80,
            depths=[4, 4, 21, 4],
            groups=[5, 10, 20, 40],
            layer_scale=1e-5,
            offset_scale=1.0,
            mlp_ratio=4.0,
            post_norm=True,
        )
    )
    config.update(kwargs)
    return config


def internimage_b_1k_224(**kwargs) -> dict:
    config = DEFAULT_CONFIG.copy()
    config.update(
        dict(
            channels=112,
            depths=[4, 4, 21, 4],
            groups=[7, 14, 28, 56],
            layer_scale=1e-5,
            offset_scale=1.0,
            post_norm=True,
            mlp_ratio=4.0,
        )
    )
    config.update(kwargs)
    return config


def internimage_l_22kto1k_384(**kwargs) -> dict:
    config = DEFAULT_CONFIG.copy()
    config.update(
        dict(
            channels=160,
            depths=[5, 5, 22, 5],
            groups=[10, 20, 40, 80],
            layer_scale=1e-5,
            offset_scale=2.0,
            post_norm=True,
            mlp_ratio=4.0,
        )
    )
    config.update(kwargs)
    return config


def internimage_xl_22kto1k_384(**kwargs) -> dict:
    config = DEFAULT_CONFIG.copy()
    config.update(
        dict(
            channels=192,
            depths=[5, 5, 24, 5],
            groups=[12, 24, 48, 96],
            layer_scale=1e-5,
            offset_scale=2.0,
            mlp_ratio=4.0,
            post_norm=True,
        )
    )
    config.update(kwargs)
    return config


def internimage_h_22kto1k_384(**kwargs) -> dict:
    config = DEFAULT_CONFIG.copy()
    config.update(
        dict(
            channels=320,
            depths=[6, 6, 32, 6],
            groups=[10, 20, 40, 80],
            layer_scale=None,
            offset_scale=1.0,
            post_norm=False,
            mlp_ratio=4.0,
            res_post_norm=True,  # for InternImage-H/G
            dw_kernel_size=5,  # for InternImage-H/G
            use_clip_projector=True,  # for InternImage-H/G
            level2_post_norm=True,  # for InternImage-H/G
            level2_post_norm_block_ids=[5, 11, 17, 23, 29],  # for InternImage-H/G
            center_feature_scale=True,  # for InternImage-H/G
        )
    )
    config.update(kwargs)
    return config


def internimage_h_22kto1k_640(**kwargs) -> dict:
    config = DEFAULT_CONFIG.copy()
    config.update(
        dict(
            channels=320,
            depths=[6, 6, 32, 6],
            groups=[10, 20, 40, 80],
            layer_scale=None,
            offset_scale=1.0,
            post_norm=False,
            mlp_ratio=4.0,
            res_post_norm=True,  # for InternImage-H/G
            dw_kernel_size=5,  # for InternImage-H/G
            use_clip_projector=True,  # for InternImage-H/G
            level2_post_norm=True,  # for InternImage-H/G
            level2_post_norm_block_ids=[5, 11, 17, 23, 29],  # for InternImage-H/G
            center_feature_scale=True,  # for InternImage-H/G
        )
    )
    config.update(kwargs)
    return config


def internimage_g_22kto1k_512(**kwargs) -> dict:
    config = DEFAULT_CONFIG.copy()
    config.update(
        dict(
            channels=512,
            depths=[2, 2, 48, 4],
            groups=[16, 32, 64, 128],
            layer_scale=None,
            offset_scale=1.0,
            post_norm=True,
            mlp_ratio=4.0,
            dw_kernel_size=5,  # for InternImage-H/G
            use_clip_projector=True,  # for InternImage-H/G
            level2_post_norm=True,  # for InternImage-H/G
            level2_post_norm_block_ids=[
                5,
                11,
                17,
                23,
                29,
                35,
                41,
                47,
            ],  # for InternImage-H/G
            center_feature_scale=True,  # for InternImage-H/G
        )
    )
    config.update(kwargs)
    return config


MODELS = {
    "internimage_t_1k_224": {
        "config": internimage_t_1k_224,
        "pretrained": "https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_t_1k_224.pth",
    },
    "internimage_s_1k_224": {
        "config": internimage_s_1k_224,
        "pretrained": "https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_s_1k_224.pth",
    },
    "internimage_b_1k_224": {
        "config": internimage_b_1k_224,
        "pretrained": "https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_b_1k_224.pth",
    },
    "internimage_l_22kto1k_384": {
        "config": internimage_l_22kto1k_384,
        "pretrained": "https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_l_22kto1k_384.pth",
    },
    "internimage_xl_22kto1k_384": {
        "config": internimage_xl_22kto1k_384,
        "pretrained": "https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_xl_22kto1k_384.pth",
    },
    "internimage_h_22kto1k_384": {
        "config": internimage_h_22kto1k_384,
        "pretrained": "https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_h_jointto22k_384.pth",
    },
    "internimage_h_22kto1k_640": {
        "config": internimage_h_22kto1k_640,
        "pretrained": "https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_h_22kto1k_640.pth",
    },
    "internimage_g_22kto1k_512": {
        "config": internimage_g_22kto1k_512,
        "pretrained": "https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_g_22kto1k_512.pth",
    },
}


def create_model(
    model_name: str,
    pretrained: bool = True,
    features_only: bool = False,
    out_indices: Optional[list[int]] = [0, 1, 2, 3],
    **kwargs,
):
    assert model_name in list(MODELS.keys()), f"model {model_name} is not supported"

    model_info = MODELS[model_name]
    config = model_info["config"](**kwargs)
    model = InternImage(**config, features_only=features_only, out_indices=out_indices)

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url=model_info["pretrained"], map_location="cpu", check_hash=True
        )
        msg = model.load_state_dict(checkpoint["model"], strict=False)
        print(msg)
    return model
