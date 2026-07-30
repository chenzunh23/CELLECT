"""Shared single-band scaling utilities for SAM/CELLECT preprocessing.

The functions here mirror the diagnostics in ``visualize_sam_input_scalings.py``
so preprocessing and visualization use the same definitions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import make_lupton_rgb


def finite_values(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("image has no finite pixels")
    return values.astype(np.float64, copy=False)


def standardize_by_self(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    values = finite_values(arr)
    mean = float(np.mean(values))
    std = float(np.std(values))
    if not np.isfinite(std) or std <= 0.0:
        std = 1.0
    safe = np.nan_to_num(arr, nan=mean, posinf=mean, neginf=mean).astype(np.float32, copy=False)
    z = ((safe - mean) / std).astype(np.float32)
    z_clip = np.clip(z, -3.0, 3.0).astype(np.float32)
    finite = z[np.isfinite(z)]
    return z, z_clip, {
        "pixel_mean": mean,
        "pixel_std": std,
        "zmax_pixel_fraction": float(np.count_nonzero(z_clip >= 3.0) / z_clip.size),
        "zmin_pixel_fraction": float(np.count_nonzero(z_clip <= -3.0) / z_clip.size),
        "full_z_p99": float(np.percentile(finite, 99.0)),
        "full_z_p99_9": float(np.percentile(finite, 99.9)),
        "full_z_max": float(np.max(finite)),
    }


def current_sam_zscore(
    image: np.ndarray,
    *,
    clip_sigma: float = 3.0,
    z_clip: tuple[float, float] = (-3.0, 3.0),
) -> tuple[np.ndarray, dict[str, float]]:
    values = finite_values(image)
    raw_min = float(np.min(values))
    raw_median = float(np.median(values))
    raw_sigma = float(np.std(values))
    if not np.isfinite(raw_sigma) or raw_sigma <= 0.0:
        raw_sigma = 1.0
    clip_hi = raw_median + float(clip_sigma) * raw_sigma
    clipped_values = np.minimum(values, clip_hi)
    mean, median, std = sigma_clipped_stats(clipped_values, sigma=float(clip_sigma), maxiters=None)
    mean = float(mean) if np.isfinite(mean) else float(np.mean(clipped_values))
    median = float(median) if np.isfinite(median) else float(np.median(clipped_values))
    std = float(std) if np.isfinite(std) and std > 0 else float(np.std(clipped_values))
    if not np.isfinite(std) or std <= 0.0:
        std = 1.0
    safe = np.where(np.isfinite(image), image, mean).astype(np.float32, copy=False)
    capped = np.minimum(safe, clip_hi)
    z = np.clip((capped - mean) / std, z_clip[0], z_clip[1]).astype(np.float32)
    return z, {
        "raw_median": raw_median,
        "raw_min": raw_min,
        "raw_sigma": raw_sigma,
        "clip_hi": clip_hi,
        "mean": mean,
        "median": median,
        "std": std,
        "clip_hi_pixel_fraction": float(np.count_nonzero(values >= clip_hi) / values.size),
        "zmax_pixel_fraction": float(np.count_nonzero(z >= z_clip[1]) / z.size),
    }


def no_first_clip_zscore(
    image: np.ndarray,
    *,
    clip_sigma: float = 3.0,
    z_clip: tuple[float, float] = (-3.0, 3.0),
) -> tuple[np.ndarray, dict[str, float]]:
    values = finite_values(image)
    mean, _median, std = sigma_clipped_stats(values, sigma=float(clip_sigma), maxiters=None)
    mean = float(mean) if np.isfinite(mean) else float(np.mean(values))
    std = float(std) if np.isfinite(std) and std > 0 else float(np.std(values))
    if not np.isfinite(std) or std <= 0.0:
        std = 1.0
    safe = np.where(np.isfinite(image), image, mean).astype(np.float32, copy=False)
    upper = z_clip[1]
    z_raw = (safe - mean) / std
    if np.isfinite(upper):
        z = np.clip(z_raw, z_clip[0], upper).astype(np.float32)
    else:
        z = np.maximum(z_raw, z_clip[0]).astype(np.float32)
    return z, {
        "mean": mean,
        "std": std,
        "zmax_pixel_fraction": float(np.count_nonzero(z >= upper) / z.size) if np.isfinite(upper) else 0.0,
    }


def log_single(image: np.ndarray, *, minimum: float, high_pct: float, a: float) -> tuple[np.ndarray, dict[str, float]]:
    values = finite_values(image)
    hi = float(np.percentile(values, float(high_pct)))
    minimum = float(minimum)
    if not np.isfinite(hi) or hi <= minimum:
        hi = float(np.max(values))
    if not np.isfinite(hi) or hi <= minimum:
        hi = minimum + 1.0
    safe = np.nan_to_num(image, nan=minimum, posinf=hi, neginf=minimum).astype(np.float32, copy=False)
    x = np.clip((safe - minimum) / (hi - minimum), 0.0, 1.0)
    a = float(a) if np.isfinite(a) and a > 0.0 else 300.0
    y = np.log1p(a * x) / np.log(a)
    return y.astype(np.float32), {
        "minimum": minimum,
        "hi": hi,
        "a": a,
        "high_clip_fraction": float(np.count_nonzero(values >= hi) / values.size),
    }


def lupton_single(image: np.ndarray, *, minimum: float, stretch: float, q: float) -> tuple[np.ndarray, dict[str, float]]:
    rgb = make_lupton_rgb(
        image,
        image,
        image,
        minimum=float(minimum),
        stretch=float(stretch),
        Q=float(q),
        output_dtype=float,
    )
    return np.asarray(rgb[..., 0], dtype=np.float32), {
        "minimum": float(minimum),
        "stretch": float(stretch),
        "q": float(q),
    }


def lupton_soft_float(image: np.ndarray, *, minimum: float, stretch: float, q: float) -> tuple[np.ndarray, dict[str, float]]:
    safe = np.nan_to_num(image, nan=minimum, posinf=minimum, neginf=minimum).astype(np.float32, copy=False)
    positive = np.maximum(safe - float(minimum), 0.0)
    stretch = float(stretch) if np.isfinite(stretch) and stretch > 0.0 else 1.0
    q = float(q) if np.isfinite(q) and q > 0.0 else 1.0
    y = np.arcsinh(q * positive / stretch) / q
    values = finite_values(y)
    vmax = float(np.percentile(values, 99.9))
    if np.isfinite(vmax) and vmax > 0.0:
        y = np.clip(y / vmax, 0.0, 1.0)
    return y.astype(np.float32), {
        "minimum": float(minimum),
        "stretch": stretch,
        "q": q,
        "p99_9_before_display_norm": vmax,
    }


def anscombe_single(image: np.ndarray, *, scale: float) -> tuple[np.ndarray, dict[str, float]]:
    values = finite_values(image)
    minimum = float(np.min(values))
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Anscombe scale must be finite and positive, got {scale!r}")
    safe = np.nan_to_num(image, nan=minimum, posinf=minimum, neginf=minimum).astype(np.float64, copy=False)
    shifted_scaled = np.maximum((safe - minimum) * scale, 0.0)
    transformed = 2.0 * np.sqrt(shifted_scaled + 3.0 / 8.0)
    return transformed.astype(np.float32), {
        "minimum": minimum,
        "scale": scale,
        "mean_minus_minimum": float(np.mean(values) - minimum),
        "scaled_mean_minus_minimum": float((np.mean(values) - minimum) * scale),
    }


def resolve_minimum(mode: str, fixed: float, zscore_stats: dict[str, float]) -> float:
    if mode == "fixed":
        return float(fixed)
    if mode == "zscore-mean":
        return float(zscore_stats["mean"])
    if mode == "zscore-median":
        return float(zscore_stats["median"])
    if mode == "raw-median":
        return float(zscore_stats["raw_median"])
    if mode == "image-min":
        return float(zscore_stats["raw_min"])
    raise ValueError(f"unsupported minimum mode: {mode}")


def read_image_plane(path: Path, *, hdu: int = 1) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=True, ignore_missing_end=True) as hdul:
        if hdu < len(hdul) and hdul[hdu].data is not None and np.asarray(hdul[hdu].data).ndim == 2:
            return np.asarray(hdul[hdu].data, dtype=np.float32), hdul[hdu].header.copy()
        for candidate in hdul:
            if candidate.data is not None and np.asarray(candidate.data).ndim == 2:
                return np.asarray(candidate.data, dtype=np.float32), candidate.header.copy()
    raise ValueError(f"no 2D image HDU found: {path}")


def build_bright_mask(
    image: np.ndarray,
    *,
    mode: str = "log-lupton",
    threshold: float = 3.0,
    dilation: int = 2,
    log_a: float = 300.0,
    log_high_percentile: float = 99.5,
    lupton_stretch: float = 0.5,
    lupton_q: float = 20.0,
    anscombe_scale: float = 1000.0,
) -> np.ndarray:
    """Build a saturated/bright-region mask for PU background suppression."""

    normalized_mode = str(mode).strip().lower().replace("_", "-")
    if normalized_mode in {"none", "raw", "zscore-no-upper", "zscore-unbounded"}:
        return np.zeros(np.asarray(image).shape, dtype=bool)
    current_z, current_stats = current_sam_zscore(image)
    if normalized_mode in {
        "zscore",
        "zscore-noclip",
        "zscore-no-clip",
        "raw-zscore",
    }:
        if normalized_mode in {"zscore-noclip", "zscore-no-clip", "raw-zscore"}:
            z, _stats = no_first_clip_zscore(image, z_clip=(-3.0, float("inf")))
        else:
            z = current_z
        bright = z >= float(threshold)
    elif normalized_mode in {"log-lupton", "zscore-lupton-log", "lupton-log"}:
        log_map, _log_stats = log_single(
            image,
            minimum=float(current_stats["raw_min"]),
            high_pct=float(log_high_percentile),
            a=float(log_a),
        )
        lupton_map, _lupton_stats = lupton_single(
            image,
            minimum=float(current_stats["mean"]),
            stretch=float(lupton_stretch),
            q=float(lupton_q),
        )
        _log_z, log_zclip, _log_zstats = standardize_by_self(log_map)
        _lupton_z, lupton_zclip, _lupton_zstats = standardize_by_self(lupton_map)
        bright = (log_zclip >= float(threshold)) & (lupton_zclip >= float(threshold))
    elif normalized_mode == "anscombe":
        anscombe_map, _stats = anscombe_single(image, scale=float(anscombe_scale))
        _z, zclip, _zstats = standardize_by_self(anscombe_map)
        bright = zclip >= float(threshold)
    else:
        raise ValueError(f"Unknown bright mask mode: {mode!r}")

    if int(dilation) > 0 and np.any(bright):
        try:
            from scipy import ndimage

            bright = ndimage.binary_dilation(bright, iterations=int(dilation))
        except Exception:
            pass
    return np.asarray(bright, dtype=bool)


def scale_training_image(
    image: np.ndarray,
    *,
    mode: str = "astro-zscore",
    z_clip: tuple[float, float] | None = None,
    log_a: float = 300.0,
    log_high_percentile: float = 99.5,
    lupton_stretch: float = 0.5,
    lupton_q: float = 20.0,
    anscombe_scale: float = 1000.0,
) -> np.ndarray:
    """Return one preprocessed image plane for direct-Zarr SAM training.

    Modes ending in ``-rgb`` return ``[3, H, W]`` arrays.  The three-channel
    mode ``zscore-log-lupton-rgb`` uses channel order ``zscore, log, lupton``.
    Single-scaling RGB modes repeat the same transformed plane three times so
    the SAM encoder receives a native RGB-shaped tensor.
    """

    normalized_mode = str(mode).strip().lower().replace("_", "-")
    if normalized_mode in {"astro-zscore", "legacy", "zscale"}:
        raise ValueError("astro-zscore is implemented by astro_zscale_preprocess in the caller")

    if normalized_mode in {"zscore-no-upper", "zscore-no-upper-rgb", "zscore-unbounded", "zscore-unbounded-rgb"}:
        z, _stats = no_first_clip_zscore(image, z_clip=(-3.0, float("inf")))
        return np.stack([z, z, z], axis=0).astype(np.float32, copy=False) if normalized_mode.endswith("-rgb") else z

    current_z, current_stats = current_sam_zscore(image, z_clip=z_clip if z_clip is not None else (-3.0, 3.0))
    if normalized_mode in {"zscore", "zscore-rgb"}:
        return (
            np.stack([current_z, current_z, current_z], axis=0).astype(np.float32, copy=False)
            if normalized_mode.endswith("-rgb")
            else current_z
        )

    if normalized_mode in {"zscore-log-lupton-rgb", "zscore-lupton-log-rgb", "log-lupton-rgb"}:
        log_map, _log_stats = log_single(
            image,
            minimum=float(current_stats["raw_min"]),
            high_pct=float(log_high_percentile),
            a=float(log_a),
        )
        lupton_map, _lupton_stats = lupton_single(
            image,
            minimum=float(current_stats["mean"]),
            stretch=float(lupton_stretch),
            q=float(lupton_q),
        )
        _log_z, log_zclip, _log_zstats = standardize_by_self(log_map)
        _lupton_z, lupton_zclip, _lupton_zstats = standardize_by_self(lupton_map)
        return np.stack([current_z, log_zclip, lupton_zclip], axis=0).astype(np.float32, copy=False)

    if normalized_mode in {"anscombe", "anscombe-rgb"}:
        anscombe_map, _stats = anscombe_single(image, scale=float(anscombe_scale))
        _z, zclip, _zstats = standardize_by_self(anscombe_map)
        return np.stack([zclip, zclip, zclip], axis=0).astype(np.float32, copy=False) if normalized_mode.endswith("-rgb") else zclip

    raise ValueError(f"Unknown training image scaling mode: {mode!r}")


def build_bright_mask_from_fits(path: Path, *, hdu: int = 1, **kwargs: object) -> np.ndarray:
    image, _header = read_image_plane(Path(path), hdu=int(hdu))
    return build_bright_mask(image, **kwargs)
