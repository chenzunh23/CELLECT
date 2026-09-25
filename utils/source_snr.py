"""Shared source SNR measurements for CELLECT visualizers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy import ndimage

from utils.aperture_snr import (
    ApertureNoiseFields,
    ApertureSnrResult,
    build_aperture_noise_fields,
    ellipse_union_mask,
    robust_sigma,
)


@dataclass(frozen=True)
class SourceSnrConfig:
    aperture_radius: float = 5.0
    snr_clip_rounds: int = 2
    snr_clip_sigma: float = 3.0
    min_sky_apertures: int = 16
    source_only_scale: float = 1.2
    background_box: int = 65
    high_threshold_sigma: float = 3.0
    high_dilation_radius: int = 5
    annulus_inner_radius: float = 10.0
    annulus_outer_radius: float = 15.0
    min_annulus_pixels: int = 100


@dataclass(frozen=True)
class SourceSnrSummary:
    method: str
    n_sources: int
    n_good: int
    n_bad: int
    background_stage: str
    noise_stage: str


def _disk_structure(radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy <= radius * radius)


def _ellipse_params(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    x = float(row.get("x", row.get("X_IMAGE", 0.0)))
    y = float(row.get("y", row.get("Y_IMAGE", 0.0)))
    major = max(abs(float(row.get("major", row.get("a", 1.0)))), 1.0)
    minor = max(abs(float(row.get("minor", row.get("b", 1.0)))), 1.0)
    theta = float(row.get("theta", row.get("theta_rad", row.get("theta_deg", 0.0))))
    if abs(theta) > 2.0 * math.pi:
        theta = math.radians(theta)
    return x, y, major, minor, theta


def _source_mask(rows: Sequence[dict[str, Any]], shape: tuple[int, int], *, scale: float) -> np.ndarray:
    mask_rows = []
    for row in rows:
        if "x" not in row or "y" not in row:
            continue
        x, y, major, minor, theta = _ellipse_params(row)
        mask_rows.append({"x": x, "y": y, "major": major, "minor": minor, "theta": theta})
    return ellipse_union_mask(shape, mask_rows, scale=float(scale))


def estimate_background(image: np.ndarray, *, box: int = 65) -> np.ndarray:
    """Estimate a per-pixel background with the median-filter path used by diagnostics."""
    arr = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(arr)
    fill = float(np.nanmedian(arr[finite])) if bool(finite.any()) else 0.0
    safe = np.where(finite, arr, fill).astype(np.float32, copy=False)
    size = max(3, int(box))
    if size % 2 == 0:
        size += 1
    return ndimage.median_filter(safe, size=size, mode="reflect").astype(np.float32, copy=False)


def background_residual(
    image: np.ndarray,
    rows: Sequence[dict[str, Any]],
    *,
    config: SourceSnrConfig = SourceSnrConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(image, dtype=np.float32)
    background = estimate_background(arr, box=int(config.background_box))
    residual = (arr - background).astype(np.float32, copy=False)
    finite = residual[np.isfinite(residual)]
    center = float(np.median(finite)) if finite.size else 0.0
    sigma = robust_sigma(finite)
    if np.isfinite(sigma) and sigma > 0.0:
        high_mask = residual > center + float(config.high_threshold_sigma) * sigma
    else:
        high_mask = np.zeros_like(residual, dtype=bool)
    high_mask = ndimage.binary_dilation(high_mask, structure=_disk_structure(int(config.high_dilation_radius)))
    source_mask = _source_mask(rows, arr.shape, scale=float(config.source_only_scale))
    return residual, background, high_mask | source_mask


def build_ap2_noise_fields(
    image: np.ndarray,
    rows: Sequence[dict[str, Any]],
    *,
    config: SourceSnrConfig = SourceSnrConfig(),
) -> tuple[ApertureNoiseFields, np.ndarray]:
    residual, _background, total_mask = background_residual(image, rows, config=config)
    fields = build_aperture_noise_fields(
        residual,
        source_mask=total_mask,
        aperture_radius=float(config.aperture_radius),
        clip_rounds=int(config.snr_clip_rounds),
        clip_sigma=float(config.snr_clip_sigma),
        min_sky_apertures=int(config.min_sky_apertures),
    )
    return fields, residual


def _annulus_flux(
    image: np.ndarray,
    row: dict[str, Any],
    source_rows: Sequence[dict[str, Any]],
    *,
    config: SourceSnrConfig,
) -> tuple[float, float, int, int]:
    height, width = image.shape
    x, y, _major, _minor, _theta = _ellipse_params(row)
    r_ap = float(config.aperture_radius)
    r_in = float(config.annulus_inner_radius)
    r_out = float(config.annulus_outer_radius)
    pad = int(math.ceil(r_out)) + 2
    x0 = max(0, int(math.floor(x)) - pad)
    x1 = min(width, int(math.floor(x)) + pad + 1)
    y0 = max(0, int(math.floor(y)) - pad)
    y1 = min(height, int(math.floor(y)) + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return float("nan"), float("nan"), 0, 0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr2 = (xx - x) ** 2 + (yy - y) ** 2
    aperture = rr2 <= r_ap * r_ap
    annulus = (rr2 >= r_in * r_in) & (rr2 <= r_out * r_out)
    other_rows = [other for other in source_rows if other is not row]
    other_mask = _source_mask(other_rows, (height, width), scale=1.0)[y0:y1, x0:x1]
    local = np.asarray(image[y0:y1, x0:x1], dtype=np.float32)
    finite_ap = aperture & np.isfinite(local)
    finite_ann = annulus & np.isfinite(local) & (~other_mask)
    aperture_pixels = int(np.sum(finite_ap))
    annulus_pixels = int(np.sum(finite_ann))
    if aperture_pixels <= 0:
        return float("nan"), float("nan"), annulus_pixels, aperture_pixels
    background_per_pixel = float(np.median(local[finite_ann])) if annulus_pixels >= int(config.min_annulus_pixels) else float("nan")
    aperture_sum = float(np.sum(local[finite_ap], dtype=np.float64))
    background = background_per_pixel * aperture_pixels if np.isfinite(background_per_pixel) else float("nan")
    flux = aperture_sum - background if np.isfinite(background) else float("nan")
    return flux, background, annulus_pixels, aperture_pixels


def measure_ap2_snr(
    image: np.ndarray,
    rows: Sequence[dict[str, Any]],
    *,
    config: SourceSnrConfig = SourceSnrConfig(),
) -> tuple[list[dict[str, Any]], SourceSnrSummary]:
    arr = np.asarray(image, dtype=np.float32)
    fields, _residual = build_ap2_noise_fields(arr, rows, config=config)
    out: list[dict[str, Any]] = []
    n_good = 0
    n_bad = 0
    for row in rows:
        flux, aperture_background, annulus_pixels, aperture_pixels = _annulus_flux(arr, row, rows, config=config)
        area_fraction = aperture_pixels / max(float(fields.model.aperture_area_pixels), 1.0)
        sigma = float(fields.model.sigma) * math.sqrt(max(area_fraction, 0.0))
        snr = flux / sigma if np.isfinite(flux) and np.isfinite(sigma) and sigma > 0.0 else float("nan")
        x, y, major, minor, _theta = _ellipse_params(row)
        edge_radius = max(major, minor, float(config.aperture_radius))
        internal = bool(x - edge_radius >= 0.0 and y - edge_radius >= 0.0 and x + edge_radius < arr.shape[1] and y + edge_radius < arr.shape[0])
        trusted = bool(fields.model.trusted and aperture_pixels > 0 and annulus_pixels >= int(config.min_annulus_pixels) and internal and np.isfinite(snr))
        new_row = dict(row)
        new_row.update(
            {
                "snr_ap2": float(snr),
                "snr": float(snr),
                "snr_method": "ap2",
                "snr_trusted": int(trusted),
                "snr_bad": int(not trusted),
                "snr_flux": float(flux),
                "snr_background": float(aperture_background),
                "snr_sigma": float(sigma),
                "snr_aperture_pixels": int(aperture_pixels),
                "snr_sky_aperture_count": int(fields.model.sky_aperture_count),
                "snr_annulus_pixels": int(annulus_pixels),
            }
        )
        n_good += int(trusted)
        n_bad += int(not trusted)
        out.append(new_row)
    return out, SourceSnrSummary("ap2", len(out), n_good, n_bad, "median background", "aperture RMS")


def _ellipse_local_mask(shape: tuple[int, int], row: dict[str, Any]) -> tuple[slice, slice, np.ndarray, bool]:
    height, width = shape
    x, y, major, minor, theta = _ellipse_params(row)
    pad = int(math.ceil(max(major, minor))) + 2
    x0 = max(0, int(math.floor(x)) - pad)
    x1 = min(width, int(math.floor(x)) + pad + 1)
    y0 = max(0, int(math.floor(y)) - pad)
    y1 = min(height, int(math.floor(y)) + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return slice(0, 0), slice(0, 0), np.zeros((0, 0), dtype=bool), False
    yy, xx = np.mgrid[y0:y1, x0:x1]
    ct, st = math.cos(theta), math.sin(theta)
    dx = xx - x
    dy = yy - y
    xp = dx * ct + dy * st
    yp = -dx * st + dy * ct
    mask = (xp / major) ** 2 + (yp / minor) ** 2 <= 1.0
    internal = bool(x - pad >= 0 and y - pad >= 0 and x + pad < width and y + pad < height)
    return slice(y0, y1), slice(x0, x1), mask, internal


def measure_kron_snr(
    image: np.ndarray,
    rows: Sequence[dict[str, Any]],
    *,
    config: SourceSnrConfig = SourceSnrConfig(),
) -> tuple[list[dict[str, Any]], SourceSnrSummary]:
    arr = np.asarray(image, dtype=np.float32)
    residual, _background, total_mask = background_residual(arr, rows, config=config)
    sky = np.isfinite(residual) & (~total_mask)
    values = residual[sky]
    pixel_sigma = robust_sigma(values)
    trusted_noise = bool(values.size >= max(100, int(config.min_sky_apertures)) and np.isfinite(pixel_sigma) and pixel_sigma > 0.0)
    out: list[dict[str, Any]] = []
    n_good = 0
    n_bad = 0
    for row in rows:
        ys, xs, mask, internal = _ellipse_local_mask(arr.shape, row)
        local = residual[ys, xs]
        finite = mask & np.isfinite(local)
        pixels = int(np.sum(finite))
        flux = float(np.sum(local[finite], dtype=np.float64)) if pixels else float("nan")
        sigma = float(pixel_sigma) * math.sqrt(float(pixels)) if trusted_noise and pixels > 0 else float("nan")
        snr = flux / sigma if np.isfinite(flux) and np.isfinite(sigma) and sigma > 0.0 else float("nan")
        trusted = bool(trusted_noise and internal and pixels > 0 and np.isfinite(snr))
        new_row = dict(row)
        new_row.update(
            {
                "snr_kron": float(snr),
                "snr": float(snr),
                "snr_method": "kron",
                "snr_trusted": int(trusted),
                "snr_bad": int(not trusted),
                "snr_flux": float(flux),
                "snr_background": 0.0,
                "snr_sigma": float(sigma),
                "snr_aperture_pixels": int(pixels),
                "snr_sky_pixel_count": int(values.size),
            }
        )
        n_good += int(trusted)
        n_bad += int(not trusted)
        out.append(new_row)
    return out, SourceSnrSummary("kron", len(out), n_good, n_bad, "median background", "per-pixel RMS")

def disk_kernel(radius: float) -> np.ndarray:
    half = int(math.ceil(float(radius)))
    yy, xx = np.mgrid[-half : half + 1, -half : half + 1]
    return (xx * xx + yy * yy <= float(radius) ** 2).astype(np.float32)


def robust_sigma_value(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    med = float(np.median(arr))
    sigma = float(1.4826 * np.median(np.abs(arr - med)))
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = float(np.std(arr))
    return sigma


def clipped_median_std(values: np.ndarray, *, rounds: int = 2, sigma: float = 3.0) -> tuple[float, float, int]:
    arr = np.asarray(values, dtype=np.float64)
    keep = np.isfinite(arr)
    for _ in range(int(rounds)):
        cur = arr[keep]
        if cur.size == 0:
            break
        med = float(np.median(cur))
        scale = robust_sigma_value(cur)
        if not np.isfinite(scale) or scale <= 0.0:
            break
        keep &= np.abs(arr - med) <= float(sigma) * scale
    cur = arr[keep]
    if cur.size == 0:
        return float("nan"), float("nan"), 0
    std = float(np.std(cur))
    if not np.isfinite(std) or std <= 0.0:
        std = robust_sigma_value(cur)
    return float(np.median(cur)), float(std), int(cur.size)


def block_background(image: np.ndarray, sky_mask: np.ndarray | None, *, box: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(arr)
    if sky_mask is None:
        use = finite
    else:
        use = finite & np.asarray(sky_mask, dtype=bool)
    global_med, global_sigma, global_count = clipped_median_std(arr[use], rounds=2, sigma=3.0)
    if not np.isfinite(global_med):
        global_med = float(np.nanmedian(arr[finite])) if np.any(finite) else 0.0
    if not np.isfinite(global_sigma) or global_sigma <= 0.0:
        global_sigma = robust_sigma_value(arr[finite]) if np.any(finite) else 1.0
    if not np.isfinite(global_sigma) or global_sigma <= 0.0:
        global_sigma = 1.0

    bg = np.full(arr.shape, global_med, dtype=np.float32)
    sigma_map = np.full(arr.shape, global_sigma, dtype=np.float32)
    count_map = np.zeros(arr.shape, dtype=np.int32)
    ny, nx = arr.shape
    box = int(max(box, 8))
    min_pix = max(64, box * box // 16)
    for y0 in range(0, ny, box):
        y1 = min(ny, y0 + box)
        for x0 in range(0, nx, box):
            x1 = min(nx, x0 + box)
            block_use = use[y0:y1, x0:x1]
            med, sig, count = clipped_median_std(arr[y0:y1, x0:x1][block_use], rounds=2, sigma=3.0)
            if count < min_pix or not np.isfinite(med) or not np.isfinite(sig) or sig <= 0.0:
                med, sig = global_med, global_sigma
            bg[y0:y1, x0:x1] = np.float32(med)
            sigma_map[y0:y1, x0:x1] = np.float32(sig)
            count_map[y0:y1, x0:x1] = int(count)
    return bg, sigma_map, count_map


def aperture_noise_by_block(residual: np.ndarray, sky_mask: np.ndarray, *, radius: float, box: int, min_sky_apertures: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    kernel = disk_kernel(radius)
    aperture_area = int(np.sum(kernel))
    finite = np.isfinite(residual)
    safe = np.where(finite, residual, 0.0).astype(np.float32, copy=False)
    finite_count = ndimage.convolve(finite.astype(np.float32), kernel, mode="constant", cval=0.0)
    sky_count = ndimage.convolve(np.asarray(sky_mask, dtype=bool).astype(np.float32), kernel, mode="constant", cval=0.0)
    aperture_sum = ndimage.convolve(safe, kernel, mode="constant", cval=0.0)
    sky_centers = (finite_count == float(aperture_area)) & (sky_count == float(aperture_area))
    global_med, global_sigma, global_count = clipped_median_std(aperture_sum[sky_centers], rounds=2, sigma=3.0)
    if not np.isfinite(global_sigma) or global_sigma <= 0.0:
        pix_sigma = robust_sigma_value(residual[np.asarray(sky_mask, dtype=bool) & finite])
        global_sigma = pix_sigma * math.sqrt(aperture_area) if np.isfinite(pix_sigma) and pix_sigma > 0 else 1.0

    sigma_map = np.full(residual.shape, global_sigma, dtype=np.float32)
    count_map = np.zeros(residual.shape, dtype=np.int32)
    ny, nx = residual.shape
    box = int(max(box, 8))
    for y0 in range(0, ny, box):
        y1 = min(ny, y0 + box)
        for x0 in range(0, nx, box):
            x1 = min(nx, x0 + box)
            vals = aperture_sum[y0:y1, x0:x1][sky_centers[y0:y1, x0:x1]]
            _med, sig, count = clipped_median_std(vals, rounds=2, sigma=3.0)
            if count < min_sky_apertures or not np.isfinite(sig) or sig <= 0.0:
                sig = global_sigma
            sigma_map[y0:y1, x0:x1] = np.float32(sig)
            count_map[y0:y1, x0:x1] = int(count)
    return sigma_map, count_map, aperture_sum.astype(np.float32, copy=False), aperture_area


def measure_local_aperture_snr(image, x, y, *, major=None, minor=None,
                               sky_mask=None, excluded_mask=None, radius=16.0,
                               box=128, min_sky_apertures=8):
    """A2744 block-background/blank-aperture estimator, with optional LSST sky.

    Input is linear flux, not scaled RGB. NaN pixels remain unobserved. Without
    an external sky mask the reference high-threshold heuristic is explicit in
    ``background_method``; it is not an LSST detection product.
    """
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 2 or not arr.size or radius <= 0 or box < 8:
        raise ValueError("requires a nonempty 2-D image, positive radius and box >= 8")
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("source coordinates must be equal-length vectors")
    finite = np.isfinite(arr)
    excluded = np.zeros(arr.shape, bool) if excluded_mask is None else np.asarray(excluded_mask, bool)
    if excluded.shape != arr.shape or (sky_mask is not None and np.shape(sky_mask) != arr.shape):
        raise ValueError("sky/excluded mask shape mismatch")
    if sky_mask is None:
        bg0, _, _ = block_background(arr, finite, box=box)
        resid0 = arr - bg0
        sigma = robust_sigma_value(resid0[finite & ~excluded])
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = robust_sigma_value(resid0[finite])
        high = finite & (resid0 > 3 * sigma)
        high = ndimage.binary_dilation(high, iterations=5)
        sky = finite & ~excluded & ~high
    else:
        sky = np.asarray(sky_mask, bool) & finite & ~excluded
    background, _, bg_count = block_background(arr, sky, box=box)
    residual = arr - background
    sigma_map, counts, _, expected_area = aperture_noise_by_block(
        residual, sky, radius=radius, box=box, min_sky_apertures=min_sky_apertures)
    major = np.full(x.shape, radius) if major is None else np.asarray(major, float)
    minor = np.full(x.shape, radius) if minor is None else np.asarray(minor, float)
    if major.shape != x.shape or minor.shape != x.shape:
        raise ValueError("source axes length mismatch")
    snr, flux = np.full(x.shape, np.nan, np.float32), np.full(x.shape, np.nan, np.float32)
    trusted, pixels = np.zeros(x.shape, bool), np.zeros(x.shape, np.int32)
    h, w = arr.shape
    for i in np.flatnonzero(np.isfinite(x) & np.isfinite(y)):
        pad = int(math.ceil(radius)) + 2
        x0, x1 = max(0, math.floor(x[i]) - pad), min(w, math.floor(x[i]) + pad + 1)
        y0, y1 = max(0, math.floor(y[i]) - pad), min(h, math.floor(y[i]) + pad + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        local = residual[y0:y1, x0:x1]
        aperture = (xx - x[i])**2 + (yy - y[i])**2 <= radius**2
        ap = aperture & np.isfinite(local)
        pixels[i] = ap.sum()
        if not pixels[i]:
            continue
        xi, yi = int(np.clip(round(x[i]), 0, w - 1)), int(np.clip(round(y[i]), 0, h - 1))
        sigma = sigma_map[yi, xi] * math.sqrt(pixels[i] / expected_area)
        flux[i] = np.sum(local[ap], dtype=np.float64)
        # A fallback sigma=1 in an empty sky is not a valid measurement.
        if sky.any() and np.isfinite(sigma) and sigma > 0:
            snr[i] = flux[i] / sigma
        edge = max(major[i], minor[i], radius)
        internal = x[i] - edge >= 0 and y[i] - edge >= 0 and x[i] + edge < w and y[i] + edge < h
        # A fractional center changes the rasterized area relative to the integer kernel.
        complete = pixels[i] == np.count_nonzero(aperture)
        trusted[i] = complete and counts[yi, xi] >= min_sky_apertures and internal and np.isfinite(snr[i])
    return dict(snr=snr, flux=flux, trusted=trusted, aperture_pixels=pixels,
                background=background, background_pixel_count=bg_count, sky_mask=sky,
                aperture_sigma=sigma_map, sky_aperture_count=counts,
                background_method="lsst_sky" if sky_mask is not None else "high_threshold_heuristic")


def measure_source_snrs(
    image: np.ndarray,
    rows: Sequence[dict[str, Any]],
    *,
    method: str = "ap2",
    config: SourceSnrConfig = SourceSnrConfig(),
) -> tuple[list[dict[str, Any]], SourceSnrSummary]:
    normalized = str(method or "ap2").strip().lower()
    if normalized in {"ap2", "aperture", "aperture2"}:
        return measure_ap2_snr(image, rows, config=config)
    if normalized in {"kron", "kron_snr", "kron-snr"}:
        return measure_kron_snr(image, rows, config=config)
    raise ValueError(f"unknown SNR method: {method}")


def filter_snr_rows(rows: Sequence[dict[str, Any]], *, threshold: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cut = float(threshold)
    for row in rows:
        snr = float(row.get("snr", float("nan")))
        bad = bool(int(row.get("snr_bad", 0))) or not np.isfinite(snr)
        if bad or snr >= cut:
            item = dict(row)
            item["class_id"] = 7 if bad else 1
            out.append(item)
    return out
