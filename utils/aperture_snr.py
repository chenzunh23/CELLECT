"""Fixed-aperture SNR utilities based on source-free aperture placement.

This follows the aperture-RMS convention used in
``codex-product-backup/analysis/2026-05-07_exposure_depth_sweep``:
measure the noise from many source-free circular aperture sums, then divide
source aperture fluxes by that aperture-level RMS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class ApertureNoiseModel:
    aperture_radius: float
    aperture_area_pixels: int
    sky_aperture_count: int
    background: float
    sigma: float
    robust_sigma: float
    clip_rounds: int
    clip_sigma: float
    trusted: bool


@dataclass(frozen=True)
class ApertureSnrResult:
    snr: float
    flux: float
    background: float
    sigma: float
    aperture_pixels: int
    sky_aperture_count: int
    trusted: bool
    internal: bool
    background_method: str = "global"
    local_sky_aperture_count: int = 0


@dataclass(frozen=True)
class ApertureNoiseFields:
    model: ApertureNoiseModel
    sky_centers: np.ndarray
    aperture_sum: np.ndarray


def circular_kernel(radius: float) -> np.ndarray:
    r = float(radius)
    half = int(math.ceil(r))
    yy, xx = np.mgrid[-half : half + 1, -half : half + 1]
    kernel = (xx * xx + yy * yy) <= r * r
    if not bool(np.any(kernel)):
        raise ValueError(f"empty circular aperture kernel for radius={radius}")
    return kernel.astype(np.float32)


def robust_sigma(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    med = float(np.median(arr))
    return float(1.4826 * np.median(np.abs(arr - med)))


def sigma_clip_stats(values: np.ndarray, *, rounds: int = 2, sigma: float = 3.0) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    keep = finite.copy()
    clip_lo = float("nan")
    clip_hi = float("nan")
    for _index in range(int(rounds)):
        current = arr[keep]
        if current.size == 0:
            break
        center = float(np.median(current))
        scale = robust_sigma(current)
        if not np.isfinite(scale) or scale <= 0.0:
            break
        clip_lo = center - float(sigma) * scale
        clip_hi = center + float(sigma) * scale
        keep = finite & (arr >= clip_lo) & (arr <= clip_hi)
    clipped = arr[keep]
    if clipped.size == 0:
        return {
            "count": 0,
            "median": None,
            "mean": None,
            "std": None,
            "robust_rms": None,
            "clip_lo": None,
            "clip_hi": None,
        }
    return {
        "count": int(clipped.size),
        "median": float(np.median(clipped)),
        "mean": float(np.mean(clipped)),
        "std": float(np.std(clipped)),
        "robust_rms": robust_sigma(clipped),
        "clip_lo": float(clip_lo),
        "clip_hi": float(clip_hi),
    }


def _row_ellipse_params(row: dict[str, float]) -> tuple[float, float, float, float, float]:
    x = float(row.get("x", row.get("X_IMAGE", 0.0)))
    y = float(row.get("y", row.get("Y_IMAGE", 0.0)))
    major = max(abs(float(row.get("major", row.get("a", 1.0)))), 1.0)
    minor = max(abs(float(row.get("minor", row.get("b", 1.0)))), 1.0)
    theta = float(row.get("theta", row.get("theta_rad", row.get("theta_deg", 0.0))))
    if abs(theta) > 2.0 * math.pi:
        theta = math.radians(theta)
    return x, y, major, minor, theta


def ellipse_union_mask(
    shape: tuple[int, int],
    rows: Sequence[dict[str, float]],
    *,
    scale: float = 1.0,
) -> np.ndarray:
    """Rasterize a union mask from image-coordinate ellipse rows."""
    height, width = int(shape[0]), int(shape[1])
    mask = np.zeros((height, width), dtype=bool)
    for row in rows:
        x, y, major, minor, theta = _row_ellipse_params(row)
        major = max(major * float(scale), 1.0)
        minor = max(minor * float(scale), 1.0)
        pad = int(math.ceil(max(major, minor))) + 2
        x0 = max(0, int(math.floor(x)) - pad)
        x1 = min(width, int(math.floor(x)) + pad + 1)
        y0 = max(0, int(math.floor(y)) - pad)
        y1 = min(height, int(math.floor(y)) + pad + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        ct, st = math.cos(theta), math.sin(theta)
        dx = xx - x
        dy = yy - y
        xp = dx * ct + dy * st
        yp = -dx * st + dy * ct
        mask[y0:y1, x0:x1] |= (xp / major) ** 2 + (yp / minor) ** 2 <= 1.0
    return mask


def build_aperture_noise_model(
    image: np.ndarray,
    *,
    source_mask: np.ndarray | None = None,
    source_rows: Sequence[dict[str, float]] | None = None,
    source_mask_scale: float = 1.0,
    aperture_radius: float = 5.0,
    clip_rounds: int = 2,
    clip_sigma: float = 3.0,
    min_sky_apertures: int = 16,
) -> tuple[ApertureNoiseModel, np.ndarray]:
    """Estimate aperture-level background and sigma from source-free apertures."""
    fields = build_aperture_noise_fields(
        image,
        source_mask=source_mask,
        source_rows=source_rows,
        source_mask_scale=source_mask_scale,
        aperture_radius=aperture_radius,
        clip_rounds=clip_rounds,
        clip_sigma=clip_sigma,
        min_sky_apertures=min_sky_apertures,
    )
    return fields.model, fields.sky_centers


def build_aperture_noise_fields(
    image: np.ndarray,
    *,
    source_mask: np.ndarray | None = None,
    source_rows: Sequence[dict[str, float]] | None = None,
    source_mask_scale: float = 1.0,
    aperture_radius: float = 5.0,
    clip_rounds: int = 2,
    clip_sigma: float = 3.0,
    min_sky_apertures: int = 16,
) -> ApertureNoiseFields:
    """Estimate aperture noise and retain aperture sums for local backgrounds."""
    arr = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(arr)
    if source_mask is None:
        source_mask = np.zeros(arr.shape, dtype=bool)
    else:
        source_mask = np.asarray(source_mask, dtype=bool).copy()
    if source_rows is not None:
        source_mask |= ellipse_union_mask(arr.shape, source_rows, scale=float(source_mask_scale))

    kernel = circular_kernel(float(aperture_radius))
    aperture_area = int(np.sum(kernel))
    finite_count = ndimage.convolve(finite.astype(np.float32), kernel, mode="constant", cval=0.0)
    source_count = ndimage.convolve(source_mask.astype(np.float32), kernel, mode="constant", cval=1.0)
    safe_image = np.where(finite, arr, 0.0).astype(np.float32, copy=False)
    aperture_sum = ndimage.convolve(safe_image, kernel, mode="constant", cval=0.0)
    sky_centers = (finite_count == float(aperture_area)) & (source_count == 0.0)
    values = np.asarray(aperture_sum[sky_centers], dtype=np.float64)
    stats = sigma_clip_stats(values, rounds=int(clip_rounds), sigma=float(clip_sigma))

    raw_background = float(np.median(values)) if values.size else float("nan")
    raw_robust = robust_sigma(values)
    background = stats["median"] if stats["median"] is not None else raw_background
    sigma = stats["std"] if stats["std"] is not None else raw_robust
    robust = stats["robust_rms"] if stats["robust_rms"] is not None else raw_robust
    trusted = (
        int(values.size) >= int(min_sky_apertures)
        and background is not None
        and sigma is not None
        and np.isfinite(float(sigma))
        and float(sigma) > 0.0
    )
    model = ApertureNoiseModel(
        aperture_radius=float(aperture_radius),
        aperture_area_pixels=aperture_area,
        sky_aperture_count=int(values.size),
        background=float(background) if background is not None else float("nan"),
        sigma=float(sigma) if sigma is not None else float("nan"),
        robust_sigma=float(robust) if robust is not None else float("nan"),
        clip_rounds=int(clip_rounds),
        clip_sigma=float(clip_sigma),
        trusted=bool(trusted),
    )
    return ApertureNoiseFields(model=model, sky_centers=sky_centers, aperture_sum=aperture_sum.astype(np.float32, copy=False))


def measure_aperture_snr(
    image: np.ndarray,
    row: dict[str, float],
    noise: ApertureNoiseModel,
    *,
    sky_centers: np.ndarray | None = None,
    aperture_sum: np.ndarray | None = None,
    local_background_radius: float = 0.0,
    min_local_sky_apertures: int = 8,
) -> ApertureSnrResult:
    """Measure fixed-aperture SNR at one row center using a precomputed noise model."""
    arr = np.asarray(image, dtype=np.float32)
    height, width = arr.shape
    x, y, major, minor, _theta = _row_ellipse_params(row)
    radius = float(noise.aperture_radius)
    pad = int(math.ceil(radius)) + 2
    x0 = max(0, int(math.floor(x)) - pad)
    x1 = min(width, int(math.floor(x)) + pad + 1)
    y0 = max(0, int(math.floor(y)) - pad)
    y1 = min(height, int(math.floor(y)) + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return ApertureSnrResult(
            float("nan"),
            float("nan"),
            noise.background,
            noise.sigma,
            0,
            noise.sky_aperture_count,
            trusted=False,
            internal=False,
            background_method="none",
            local_sky_aperture_count=0,
        )
    yy, xx = np.mgrid[y0:y1, x0:x1]
    aperture = (xx - x) ** 2 + (yy - y) ** 2 <= radius * radius
    local = arr[y0:y1, x0:x1]
    expected_aperture_pixels = int(np.sum(aperture))
    finite_ap = aperture & np.isfinite(local)
    aperture_pixels = int(np.sum(finite_ap))
    ap_sum = float(np.sum(local[finite_ap], dtype=np.float64)) if aperture_pixels else float("nan")
    area_fraction = aperture_pixels / max(float(noise.aperture_area_pixels), 1.0)
    background_sum = float(noise.background)
    background_method = "global"
    local_count = 0
    if (
        sky_centers is not None
        and aperture_sum is not None
        and float(local_background_radius) > 0.0
        and np.asarray(sky_centers).shape == arr.shape
        and np.asarray(aperture_sum).shape == arr.shape
    ):
        yy_all, xx_all = np.mgrid[:height, :width]
        nearby = ((xx_all - x) ** 2 + (yy_all - y) ** 2 <= float(local_background_radius) ** 2) & np.asarray(sky_centers, dtype=bool)
        local_values = np.asarray(aperture_sum[nearby], dtype=np.float64)
        local_values = local_values[np.isfinite(local_values)]
        local_count = int(local_values.size)
        if local_count >= int(min_local_sky_apertures):
            stats = sigma_clip_stats(local_values, rounds=int(noise.clip_rounds), sigma=float(noise.clip_sigma))
            if stats["median"] is not None:
                background_sum = float(stats["median"])
                background_method = "local"

    background = background_sum * area_fraction
    sigma = float(noise.sigma) * math.sqrt(max(area_fraction, 0.0))
    flux = ap_sum - background if np.isfinite(ap_sum) and np.isfinite(background) else float("nan")
    snr = flux / sigma if np.isfinite(flux) and np.isfinite(sigma) and sigma > 0.0 else float("nan")
    edge_radius = max(major, minor, radius)
    internal = bool(
        x - edge_radius >= 0.0
        and y - edge_radius >= 0.0
        and x + edge_radius < width
        and y + edge_radius < height
    )
    trusted = bool(noise.trusted and expected_aperture_pixels > 0 and aperture_pixels == expected_aperture_pixels and internal)
    return ApertureSnrResult(
        float(snr),
        float(flux),
        float(background),
        float(sigma),
        aperture_pixels,
        int(noise.sky_aperture_count),
        trusted,
        internal,
        background_method,
        local_count,
    )
