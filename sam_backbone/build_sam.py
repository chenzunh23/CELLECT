# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .sam import Sam
from .transformer import TwoWayTransformer


RGB_PIXEL_MEAN = [123.675, 116.28, 103.53]
RGB_PIXEL_STD = [58.395, 57.12, 57.375]
ASTRO_PIXEL_MEAN = [0.0, 0.0, 0.0]
ASTRO_PIXEL_STD = [1.0, 1.0, 1.0]


def build_sam_vit_h(
    checkpoint=None,
    scaling_mode=None,
    astro_rgb_mode="none",
    astro_rgb_low_sigma=None,
    astro_rgb_none_std=None,
    astro_preprocess_in_model=False,
    astro_preprocess_clip_sigma=3.0,
    astro_preprocess_sigma_iters=-1,
    astro_preprocess_z_clip=None,
):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
        scaling_mode=scaling_mode,
        astro_rgb_mode=astro_rgb_mode,
        astro_rgb_low_sigma=astro_rgb_low_sigma,
        astro_rgb_none_std=astro_rgb_none_std,
        astro_preprocess_in_model=astro_preprocess_in_model,
        astro_preprocess_clip_sigma=astro_preprocess_clip_sigma,
        astro_preprocess_sigma_iters=astro_preprocess_sigma_iters,
        astro_preprocess_z_clip=astro_preprocess_z_clip,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(
    checkpoint=None,
    scaling_mode=None,
    astro_rgb_mode="none",
    astro_rgb_low_sigma=None,
    astro_rgb_none_std=None,
    astro_preprocess_in_model=False,
    astro_preprocess_clip_sigma=3.0,
    astro_preprocess_sigma_iters=-1,
    astro_preprocess_z_clip=None,
):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
        scaling_mode=scaling_mode,
        astro_rgb_mode=astro_rgb_mode,
        astro_rgb_low_sigma=astro_rgb_low_sigma,
        astro_rgb_none_std=astro_rgb_none_std,
        astro_preprocess_in_model=astro_preprocess_in_model,
        astro_preprocess_clip_sigma=astro_preprocess_clip_sigma,
        astro_preprocess_sigma_iters=astro_preprocess_sigma_iters,
        astro_preprocess_z_clip=astro_preprocess_z_clip,
    )


def build_sam_vit_b(
    checkpoint=None,
    scaling_mode=None,
    astro_rgb_mode="none",
    astro_rgb_low_sigma=None,
    astro_rgb_none_std=None,
    astro_preprocess_in_model=False,
    astro_preprocess_clip_sigma=3.0,
    astro_preprocess_sigma_iters=-1,
    astro_preprocess_z_clip=None,
):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        scaling_mode=scaling_mode,
        astro_rgb_mode=astro_rgb_mode,
        astro_rgb_low_sigma=astro_rgb_low_sigma,
        astro_rgb_none_std=astro_rgb_none_std,
        astro_preprocess_in_model=astro_preprocess_in_model,
        astro_preprocess_clip_sigma=astro_preprocess_clip_sigma,
        astro_preprocess_sigma_iters=astro_preprocess_sigma_iters,
        astro_preprocess_z_clip=astro_preprocess_z_clip,
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def convert_astro(mean, std, mode="astro_rgb", low_sigma=None):
    if mode == "none":
        return mean, std
    elif mode == "astro_rgb":
        new_mean = [(m + 3.0) * 255 / 6.0 for m in mean]
        new_std = [s * 255 / 6.0 for s in std]
        return new_mean, new_std
    elif mode == "astro_rgb1":
        new_mean = [(m + 1.0) * 255 / 4.0 for m in mean]
        new_std = [s * 255 / 4.0 for s in std]
        return new_mean, new_std
    elif mode == "astro_rgb2":
        low = 5.0 if low_sigma is None else float(low_sigma)
        scale = low + 3.0
        new_mean = [(m + low) * 255 / scale for m in mean]
        new_std = [s * 255 / scale for s in std]
        return new_mean, new_std
    raise ValueError(f"Unknown astro_rgb_mode: {mode}")


def get_pixel_stats(
    scaling_mode=None,
    astro_rgb_mode="none",
    astro_rgb_low_sigma=None,
    astro_rgb_none_std=None,
):
    if scaling_mode == "astro_rgb":
        if astro_rgb_mode == "none":
            std_value = 1.0 if astro_rgb_none_std is None else float(astro_rgb_none_std)
            return ASTRO_PIXEL_MEAN, [std_value] * 3
        return convert_astro(
            ASTRO_PIXEL_MEAN,
            ASTRO_PIXEL_STD,
            mode=astro_rgb_mode,
            low_sigma=astro_rgb_low_sigma,
        )

    if astro_rgb_mode != "none":
        return convert_astro(
            ASTRO_PIXEL_MEAN,
            ASTRO_PIXEL_STD,
            mode=astro_rgb_mode,
            low_sigma=astro_rgb_low_sigma,
        )

    return RGB_PIXEL_MEAN, RGB_PIXEL_STD


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    scaling_mode=None,
    astro_rgb_mode="none",
    astro_rgb_low_sigma=None,
    astro_rgb_none_std=None,
    astro_preprocess_in_model=False,
    astro_preprocess_clip_sigma=3.0,
    astro_preprocess_sigma_iters=-1,
    astro_preprocess_z_clip=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    mean, std = get_pixel_stats(
        scaling_mode=scaling_mode,
        astro_rgb_mode=astro_rgb_mode,
        astro_rgb_low_sigma=astro_rgb_low_sigma,
        astro_rgb_none_std=astro_rgb_none_std,
    )
    print(f"Using SAM pixel normalization with mean={mean} and std={std}")
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=mean,
        pixel_std=std,
        astro_preprocess_in_model=astro_preprocess_in_model,
        astro_preprocess_clip_sigma=astro_preprocess_clip_sigma,
        astro_preprocess_sigma_iters=astro_preprocess_sigma_iters,
        astro_preprocess_z_clip=astro_preprocess_z_clip,
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        # Deal with torch.compile
        try:
            sam.load_state_dict(state_dict)
        except Exception:
            # Add _orig_mod to state dict keys to load into compiled modules
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("image_encoder."):
                    new_k = k.replace("image_encoder.", "image_encoder._orig_mod.")
                elif k.startswith("prompt_encoder."):
                    new_k = k.replace("prompt_encoder.", "prompt_encoder._orig_mod.")
                elif k.startswith("mask_decoder."):
                    new_k = k.replace("mask_decoder.", "mask_decoder._orig_mod.")
                else:
                    new_k = k
                new_state_dict[new_k] = v
            torch.set_float32_matmul_precision("high")
            sam.load_state_dict(new_state_dict)
    return sam
