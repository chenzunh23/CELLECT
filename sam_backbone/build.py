from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .image_encoder import ImageEncoderViT
from .preprocess import astro_preprocess, pad_to_patch_multiple, pad_to_square, per_band_stats
from .style_conditioning import ImageStyleRouter


SAM_ENCODER_CONFIGS: Dict[str, Dict[str, Any]] = {
    "vit_b": {
        "encoder_embed_dim": 768,
        "encoder_depth": 12,
        "encoder_num_heads": 12,
        "encoder_global_attn_indexes": (2, 5, 8, 11),
    },
    "vit_l": {
        "encoder_embed_dim": 1024,
        "encoder_depth": 24,
        "encoder_num_heads": 16,
        "encoder_global_attn_indexes": (5, 11, 17, 23),
    },
    "vit_h": {
        "encoder_embed_dim": 1280,
        "encoder_depth": 32,
        "encoder_num_heads": 16,
        "encoder_global_attn_indexes": (7, 15, 23, 31),
    },
}


def build_sam_image_encoder(
    model_type: str = "vit_b",
    *,
    checkpoint: Optional[str | Path] = None,
    image_size: int = 512,
    patch_size: int = 16,
    out_chans: int = 256,
    strict: bool = False,
    style_prompt_dim: int = 0,
    style_prompt_layers: Sequence[int] = (),
    style_adapter_dim: int = 32,
) -> ImageEncoderViT:
    """Build a 512-native SAM image encoder and optionally load SAM weights."""

    key = str(model_type).lower()
    if key == "default":
        key = "vit_h"
    if key not in SAM_ENCODER_CONFIGS:
        raise ValueError(f"unknown SAM encoder type {model_type!r}; expected one of {sorted(SAM_ENCODER_CONFIGS)}")
    cfg = SAM_ENCODER_CONFIGS[key]
    encoder = ImageEncoderViT(
        depth=cfg["encoder_depth"],
        embed_dim=cfg["encoder_embed_dim"],
        img_size=int(image_size),
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=cfg["encoder_num_heads"],
        patch_size=int(patch_size),
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=tuple(cfg["encoder_global_attn_indexes"]),
        window_size=14,
        in_chans=3,
        out_chans=int(out_chans),
        style_prompt_dim=int(style_prompt_dim),
        style_prompt_layers=tuple(style_prompt_layers),
        style_adapter_dim=int(style_adapter_dim),
    )
    if checkpoint is not None:
        load_sam_encoder_checkpoint(encoder, checkpoint, strict=strict)
    return encoder


def load_sam_encoder_checkpoint(
    encoder: ImageEncoderViT,
    checkpoint: str | Path | Mapping[str, Tensor],
    *,
    strict: bool = False,
) -> nn.modules.module._IncompatibleKeys:
    """Load official SAM checkpoints into the resized image encoder.

    Official checkpoints contain prompt encoder and mask decoder weights and
    were trained at 1024/16, so absolute and global relative position tensors
    are resized to the current model shapes before loading.
    """

    state = _load_state_dict(checkpoint)
    encoder_state = _extract_encoder_state(state)
    model_state = encoder.state_dict()
    adapted: Dict[str, Tensor] = {}
    skipped: list[str] = []
    for key, value in encoder_state.items():
        if key not in model_state:
            skipped.append(key)
            continue
        target = model_state[key]
        if tuple(value.shape) == tuple(target.shape):
            adapted[key] = value
            continue
        resized = _resize_encoder_tensor(key, value, target)
        if resized is None:
            skipped.append(key)
            continue
        adapted[key] = resized

    incompatible = encoder.load_state_dict(adapted, strict=False)
    if strict and (incompatible.missing_keys or incompatible.unexpected_keys or skipped):
        raise RuntimeError(
            "SAM encoder checkpoint did not load strictly: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}, skipped={skipped}"
        )
    return incompatible


class SamPerBandImageEncoder(nn.Module):
    """Run one shared SAM image encoder independently for every band.

    Forward input is either ``[B, bands, H, W]`` or ``[B, bands, 3, H, W]``.
    Grayscale inputs are repeated to RGB; 3-channel inputs are passed to SAM as
    native RGB-like channels.  The encoder output is reshaped back to
    ``[B, bands, 256, H/16, W/16]``.
    """

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        *,
        num_bands: int = 5,
        astro_preprocess_in_model: bool = True,
        astro_preprocess_clip_sigma: float = 3.0,
        astro_preprocess_sigma_iters: int = -1,
        astro_preprocess_z_clip: Optional[Tuple[float, float]] = None,
        style_prompt_enabled: bool = False,
        style_prompt_dim: int = 32,
        style_router_temperature: float = 1.0,
        dynamic_image_size: bool = False,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.num_bands = int(num_bands)
        self.astro_preprocess_in_model = bool(astro_preprocess_in_model)
        self.astro_preprocess_clip_sigma = float(astro_preprocess_clip_sigma)
        self.astro_preprocess_sigma_iters = int(astro_preprocess_sigma_iters)
        self.astro_preprocess_z_clip = astro_preprocess_z_clip
        self.style_prompt_enabled = bool(style_prompt_enabled)
        self.style_router_temperature = float(style_router_temperature)
        self.dynamic_image_size = bool(dynamic_image_size)
        if self.style_router_temperature <= 0.0:
            raise ValueError("style_router_temperature must be positive")
        if self.style_prompt_enabled:
            self.style_router = ImageStyleRouter(num_bands=self.num_bands)
            self.style_prompt_raw = nn.Parameter(torch.empty(int(style_prompt_dim)))
            self.style_prompt_processed = nn.Parameter(torch.empty(int(style_prompt_dim)))
            nn.init.normal_(self.style_prompt_raw, std=0.02)
            nn.init.normal_(self.style_prompt_processed, std=0.02)
        else:
            self.style_router = None
            self.register_parameter("style_prompt_raw", None)
            self.register_parameter("style_prompt_processed", None)

    @property
    def img_size(self) -> int:
        return int(self.image_encoder.img_size)

    @property
    def patch_size(self) -> int:
        return int(self.image_encoder.patch_size)

    def forward(
        self,
        x: Tensor,
        *,
        zscale_cache: Optional[Tensor] = None,
        input_is_preprocessed: bool = False,
        return_flat: bool = False,
        return_stats: bool = False,
        return_input: bool = False,
    ) -> Tensor | dict[str, Tensor | dict[str, Tensor] | int]:
        if x.ndim == 4:
            has_rgb_axis = False
            batch, bands, height, width = x.shape
        elif x.ndim == 5 and int(x.shape[2]) == 3:
            has_rgb_axis = True
            batch, bands, _rgb, height, width = x.shape
        else:
            raise ValueError(
                f"SamPerBandImageEncoder expects [B, band, H, W] or [B, band, 3, H, W], got shape {tuple(x.shape)}"
            )
        if bands != self.num_bands and self.style_router is not None:
            raise ValueError(
                f"input has {bands} bands but encoder style router was configured for {self.num_bands}"
            )

        if zscale_cache is not None:
            if has_rgb_axis:
                raise ValueError("zscale_cache is only supported for grayscale [B, band, H, W] SAM inputs")
            x = astro_preprocess(x, zscale_cache=zscale_cache, cache_is_preprocessed=True).to(device=x.device)
            processed_rgb = None
        elif self.astro_preprocess_in_model and not input_is_preprocessed:
            if has_rgb_axis:
                raise ValueError("in-model astro preprocessing is only supported for grayscale [B, band, H, W] SAM inputs")
            x = astro_preprocess(
                x,
                clip_sigma=self.astro_preprocess_clip_sigma,
                sigma_iters=self.astro_preprocess_sigma_iters,
                z_clip=self.astro_preprocess_z_clip,
            )
            processed_rgb = None
        else:
            x = torch.nan_to_num(x.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
            processed_rgb = x if has_rgb_axis else None

        processed = x.mean(dim=2) if has_rgb_axis else x
        style_logit: Optional[Tensor] = None
        style_alpha: Optional[Tensor] = None
        flat_style_prompt: Optional[Tensor] = None
        if self.style_router is not None:
            style_logit = self.style_router(processed)
            style_alpha = torch.sigmoid(style_logit / self.style_router_temperature)
            sample_prompt = (
                (1.0 - style_alpha[:, None]) * self.style_prompt_raw[None]
                + style_alpha[:, None] * self.style_prompt_processed[None]
            )
            flat_style_prompt = sample_prompt[:, None, :].expand(-1, bands, -1).reshape(batch * bands, -1)
        stats = per_band_stats(processed) if return_stats else None
        if processed_rgb is not None:
            flat_rgb_input = processed_rgb.reshape(batch * bands, 3, height, width)
            if self.dynamic_image_size:
                flat_rgb = pad_to_patch_multiple(
                    flat_rgb_input,
                    self.patch_size,
                    max_size=self.img_size,
                ).contiguous()
            else:
                flat_rgb = pad_to_square(flat_rgb_input, self.img_size).contiguous()
        else:
            padded = (
                pad_to_patch_multiple(processed, self.patch_size, max_size=self.img_size)
                if self.dynamic_image_size
                else pad_to_square(processed, self.img_size)
            )
            padded_height, padded_width = (int(v) for v in padded.shape[-2:])
            flat = padded.reshape(batch * bands, 1, padded_height, padded_width)
            flat_rgb = flat.expand(-1, 3, -1, -1).contiguous()
        padded_height, padded_width = (int(v) for v in flat_rgb.shape[-2:])
        flat_features = self.image_encoder(flat_rgb, style_prompt=flat_style_prompt)
        features = flat_features.reshape(batch, bands, *flat_features.shape[1:])

        if not return_flat and not return_stats and not return_input:
            return features

        out: dict[str, Tensor | dict[str, Tensor] | int] = {
            "features": features,
            "flat_features": flat_features,
            "flat_batch_size": int(batch * bands),
            "batch_size": int(batch),
            "num_bands": int(bands),
            "input_height": int(height),
            "input_width": int(width),
            "padded_height": padded_height,
            "padded_width": padded_width,
        }
        if stats is not None:
            out["per_band_stats"] = stats
        if return_input:
            out["preprocessed_images"] = processed
        if style_logit is not None and style_alpha is not None:
            out["style_logit"] = style_logit
            out["style_alpha"] = style_alpha
        if return_flat:
            out["flat_rgb"] = flat_rgb
        return out


def build_per_band_sam_encoder(
    model_type: str = "vit_b",
    *,
    checkpoint: Optional[str | Path] = None,
    num_bands: int = 5,
    image_size: int = 512,
    patch_size: int = 16,
    strict: bool = False,
    astro_preprocess_in_model: bool = True,
    astro_preprocess_clip_sigma: float = 3.0,
    astro_preprocess_sigma_iters: int = -1,
    astro_preprocess_z_clip: Optional[Tuple[float, float]] = None,
    style_prompt_enabled: bool = False,
    style_prompt_dim: int = 32,
    style_prompt_layers: Sequence[int] = (),
    style_adapter_dim: int = 32,
    style_router_temperature: float = 1.0,
    dynamic_image_size: bool = False,
) -> SamPerBandImageEncoder:
    encoder = build_sam_image_encoder(
        model_type,
        checkpoint=checkpoint,
        image_size=image_size,
        patch_size=patch_size,
        strict=strict,
        style_prompt_dim=int(style_prompt_dim) if style_prompt_enabled else 0,
        style_prompt_layers=tuple(style_prompt_layers) if style_prompt_enabled else (),
        style_adapter_dim=int(style_adapter_dim),
    )
    return SamPerBandImageEncoder(
        encoder,
        num_bands=num_bands,
        astro_preprocess_in_model=astro_preprocess_in_model,
        astro_preprocess_clip_sigma=astro_preprocess_clip_sigma,
        astro_preprocess_sigma_iters=astro_preprocess_sigma_iters,
        astro_preprocess_z_clip=astro_preprocess_z_clip,
        style_prompt_enabled=style_prompt_enabled,
        style_prompt_dim=style_prompt_dim,
        style_router_temperature=style_router_temperature,
        dynamic_image_size=dynamic_image_size,
    )


def _load_state_dict(checkpoint: str | Path | Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    if isinstance(checkpoint, Mapping):
        state = checkpoint
    else:
        path = Path(checkpoint).expanduser()
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
    if isinstance(state, Mapping) and "model" in state and isinstance(state["model"], Mapping):
        state = state["model"]
    elif isinstance(state, Mapping) and "state_dict" in state and isinstance(state["state_dict"], Mapping):
        state = state["state_dict"]
    return state


def _extract_encoder_state(state: Mapping[str, Tensor]) -> Dict[str, Tensor]:
    out: Dict[str, Tensor] = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        new_key = str(key)
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        if new_key.startswith("image_encoder._orig_mod."):
            new_key = new_key[len("image_encoder._orig_mod.") :]
        elif new_key.startswith("image_encoder."):
            new_key = new_key[len("image_encoder.") :]
        elif new_key.startswith("encoder."):
            new_key = new_key[len("encoder.") :]
        elif new_key.startswith(("prompt_encoder.", "mask_decoder.")):
            continue
        out[new_key] = value.detach().cpu()
    return out


def _resize_encoder_tensor(key: str, value: Tensor, target: Tensor) -> Optional[Tensor]:
    if key == "pos_embed" and value.ndim == 4 and target.ndim == 4:
        return _resize_pos_embed(value, target)
    if key.endswith(("rel_pos_h", "rel_pos_w")) and value.ndim == 2 and target.ndim == 2:
        return _resize_rel_pos(value, target)
    if key == "patch_embed.proj.weight" and value.ndim == 4 and target.ndim == 4:
        return _resize_patch_embed(value, target)
    return None


def _resize_pos_embed(value: Tensor, target: Tensor) -> Tensor:
    if value.shape[1] % target.shape[1] == 0 and value.shape[2] % target.shape[2] == 0:
        step_h = value.shape[1] // target.shape[1]
        step_w = value.shape[2] // target.shape[2]
        sliced = value[:, ::step_h, ::step_w]
        if tuple(sliced.shape) == tuple(target.shape):
            return sliced.to(dtype=target.dtype)
    resized = F.interpolate(
        value.permute(0, 3, 1, 2).float(),
        size=tuple(target.shape[1:3]),
        mode="bicubic",
        align_corners=False,
    ).permute(0, 2, 3, 1)
    return resized.to(dtype=target.dtype)


def _resize_rel_pos(value: Tensor, target: Tensor) -> Tensor:
    resized = F.interpolate(
        value.reshape(1, value.shape[0], value.shape[1]).permute(0, 2, 1).float(),
        size=target.shape[0],
        mode="linear",
        align_corners=False,
    ).permute(0, 2, 1).reshape(target.shape)
    return resized.to(dtype=target.dtype)


def _resize_patch_embed(value: Tensor, target: Tensor) -> Optional[Tensor]:
    if value.shape[0] != target.shape[0]:
        return None
    work = value.float()
    if value.shape[-2:] != target.shape[-2:]:
        old_h, old_w = int(value.shape[-2]), int(value.shape[-1])
        new_h, new_w = int(target.shape[-2]), int(target.shape[-1])
        if old_h % new_h == 0 and old_w % new_w == 0:
            # Cellpose-SAM style: initialize a smaller patch kernel by sampling
            # the official 16x16 SAM kernel on the corresponding stride grid.
            work = work[:, :, :: old_h // new_h, :: old_w // new_w]
        else:
            work = F.interpolate(work, size=tuple(target.shape[-2:]), mode="bicubic", align_corners=False)
    if work.shape[1] == target.shape[1]:
        return work.to(dtype=target.dtype)
    if target.shape[1] == 3 and work.shape[1] >= 3:
        return work[:, :3].to(dtype=target.dtype)
    if work.shape[1] == 3 and target.shape[1] != 3:
        base = work.mean(dim=1, keepdim=True)
        return (base.repeat(1, target.shape[1], 1, 1) / float(target.shape[1])).to(dtype=target.dtype)
    return None


def available_model_types() -> Sequence[str]:
    return tuple(SAM_ENCODER_CONFIGS.keys())
