from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F


ArrayLike = Union[np.ndarray, Tensor]


def _as_float_tensor(x: ArrayLike, *, device: Optional[torch.device] = None) -> Tensor:
    if not torch.is_tensor(x):
        x = torch.from_numpy(np.asarray(x))
    if device is not None:
        x = x.to(device=device)
    return x.to(dtype=torch.float32)


def astro_preprocess(
    x: ArrayLike,
    *,
    zscale_cache: Optional[ArrayLike] = None,
    cache_is_preprocessed: bool = True,
    clip_sigma: float = 3.0,
    sigma_iters: int = -1,
    z_clip: Optional[Tuple[float, float]] = None,
) -> Tensor:
    """SAM-style per-image, per-band normalization.

    Input and output are CHW or BCHW.  If ``zscale_cache`` is supplied and
    ``cache_is_preprocessed`` is true, the cache is returned after finite-value
    cleanup.  This lets the model consume CELLECT zscale cache tensors without
    recomputing sigma-clipped statistics in the training loop.
    """

    source = zscale_cache if zscale_cache is not None else x
    out = _as_float_tensor(source)
    if cache_is_preprocessed and zscale_cache is not None:
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    squeeze_batch = out.ndim == 3
    if squeeze_batch:
        work = out.unsqueeze(0)
    elif out.ndim == 4:
        work = out
    else:
        raise ValueError(f"astro_preprocess expects CHW or BCHW input, got shape {tuple(out.shape)}")

    normalized = torch.empty_like(work)
    for batch_idx in range(work.shape[0]):
        for band_idx in range(work.shape[1]):
            vals = work[batch_idx, band_idx]
            finite = torch.isfinite(vals)
            if not bool(finite.any()):
                normalized[batch_idx, band_idx] = torch.zeros_like(vals)
                continue

            finite_vals = vals[finite]
            raw_median = torch.median(finite_vals)
            raw_sigma = torch.std(finite_vals, unbiased=False)
            if not bool(torch.isfinite(raw_sigma)) or float(raw_sigma) <= 0.0:
                raw_sigma = torch.ones((), dtype=vals.dtype, device=vals.device)
            clip_hi = raw_median + float(clip_sigma) * raw_sigma

            clipped_vals = torch.minimum(finite_vals, clip_hi)
            mean, std = astro_sigma_clipped_mean_std(
                clipped_vals,
                vals,
                clip_sigma=clip_sigma,
                sigma_iters=sigma_iters,
            )
            if not bool(torch.isfinite(std)) or float(std) <= 0.0:
                std = torch.ones((), dtype=vals.dtype, device=vals.device)

            safe = torch.where(finite, vals, mean)
            clipped = torch.minimum(safe, clip_hi)
            z = (clipped - mean) / std
            if z_clip is not None:
                z = torch.clamp(z, float(z_clip[0]), float(z_clip[1]))
            normalized[batch_idx, band_idx] = z

    return normalized[0] if squeeze_batch else normalized


def astro_sigma_clipped_mean_std(
    clipped_vals: Tensor,
    like: Tensor,
    *,
    clip_sigma: float,
    sigma_iters: int,
) -> Tuple[Tensor, Tensor]:
    """Match the astropy-backed sigma-clipped mean/std helper used for SAM preprocessing."""

    try:
        from astropy.stats import sigma_clipped_stats
    except Exception as exc:
        raise RuntimeError("astro_preprocess requires astropy unless a zscale cache tensor is supplied.") from exc

    arr = clipped_vals.detach().cpu().numpy().astype(np.float64, copy=False)
    maxiters = None if int(sigma_iters) < 0 else int(sigma_iters)
    mean, _median, std = sigma_clipped_stats(
        arr,
        sigma=float(clip_sigma),
        maxiters=maxiters,
    )
    if not np.isfinite(mean):
        mean = float(np.nanmean(arr))
    if not np.isfinite(std) or std <= 0:
        std = float(np.nanstd(arr))
    mean_t = torch.tensor(float(mean), dtype=like.dtype, device=like.device)
    std_t = torch.tensor(float(std), dtype=like.dtype, device=like.device)
    return mean_t, std_t


def pad_to_square(x: Tensor, size: int) -> Tensor:
    """Pad BCHW/CHW tensors on bottom/right to ``size`` square."""

    if x.ndim not in (3, 4):
        raise ValueError(f"pad_to_square expects CHW or BCHW, got shape {tuple(x.shape)}")
    h, w = x.shape[-2:]
    if h > size or w > size:
        raise ValueError(f"input spatial size {(h, w)} exceeds configured SAM image size {(size, size)}")
    return F.pad(x, (0, int(size) - w, 0, int(size) - h))


def per_band_stats(x: Tensor) -> dict[str, Tensor]:
    """Return finite-aware per item/per band summary tensors for BCHW inputs."""

    if x.ndim != 4:
        raise ValueError(f"per_band_stats expects BCHW input, got shape {tuple(x.shape)}")
    finite = torch.isfinite(x)
    safe = torch.where(finite, x, torch.zeros_like(x))
    count = finite.flatten(2).sum(dim=-1).clamp_min(1)
    total = safe.flatten(2).sum(dim=-1)
    mean = total / count
    centered = torch.where(finite, x - mean[..., None, None], torch.zeros_like(x))
    var = centered.flatten(2).pow(2).sum(dim=-1) / count
    return {
        "mean": mean,
        "std": torch.sqrt(var.clamp_min(0.0)),
        "finite_fraction": finite.flatten(2).float().mean(dim=-1),
    }
