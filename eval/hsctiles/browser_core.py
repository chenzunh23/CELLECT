from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/{os.environ.get('USER', str(os.getuid()))}_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
from astropy.visualization import ZScaleInterval
from PIL import Image, ImageDraw, ImageFont
import torch

from data_filtering.sam_input_scaling import (
    anscombe_single as display_anscombe_single,
    current_sam_zscore as display_sam_zscore,
    lupton_single as display_lupton_single,
)

DEFAULT_CELLECT_ROOT = Path(__file__).resolve().parents[2]
CELLECT_ROOT = Path(os.environ.get("CELLECT_ROOT", str(DEFAULT_CELLECT_ROOT))).expanduser().resolve()
if str(CELLECT_ROOT) not in sys.path:
    sys.path.insert(0, str(CELLECT_ROOT))

from eval.datasets import (  # noqa: E402
    DEFAULT_HSC_RAW_BANDS,
    DEFAULT_HSC_COADD_FITS_ROOT,
    DEFAULT_HSC_DENOISED_FITS_ROOT,
    DEFAULT_HSC_NOISY_FITS_ROOT,
    DEFAULT_HSC_WEIGHT_ROOT,
    DEFAULT_HSC_RAW_ROOT,
    DEFAULT_JWST_NIRCAM_ROOT,
    DEFAULT_MESSIER_ROOT,
    DEFAULT_ZTF_BANDS,
    DEFAULT_ZTF_CUT_ORIGIN_DIR,
    DEFAULT_ZTF_FIELD,
    DEFAULT_ZTF_ROOT,
    DEFAULT_ZTF_TILE_SIZE,
    FrameRef,
    HscImageAccess,
    HscRawAccess,
    JwstNircamAccess,
    MessierAccess,
    ZtfAccess,
)
from eval.datasets.base import patch_sort_key  # noqa: E402
from eval.eval_utils import (  # noqa: E402
    detection_rows,
    draw_ellipses,
    infer_cellect,
    load_cellect_model,
    make_training_rgb,
    select_band_outputs,
    zscale_gray,
)
from utils.source_snr import SourceSnrConfig, filter_snr_rows, measure_source_snrs  # noqa: E402


DEFAULT_ROOT = DEFAULT_HSC_RAW_ROOT
DEFAULT_BANDS = DEFAULT_HSC_RAW_BANDS
DEFAULT_CHECKPOINT = Path("/data/czh23/ckpts/sam_anscombe_0803/epoch_0030.pt")


PAGES_DIR = Path(__file__).with_name("pages")
HTML_PATH = PAGES_DIR / "index.html"
ASSETS_DIR = CELLECT_ROOT / "eval" / "assets"


def _slug_text(text: str, *, fallback: str = "run") -> str:
    clean = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(text).strip())
    clean = "_".join(part for part in clean.split("_") if part)
    return clean or fallback


def _session_name(run_name: str, stamp: str) -> str:
    run_name = str(run_name or "").strip()
    if not run_name:
        return stamp
    return f"{_slug_text(run_name)}_{stamp}"


def _parse_tile_xy(tile_id: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"x(\d+)_y(\d+)", str(tile_id))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _tile_id_candidates(text: str, y: int | None = None) -> list[str]:
    raw = str(text or "").strip()
    values: list[tuple[int, int]] = []
    match = re.fullmatch(r"x(\d+)_y(\d+)", raw)
    if match:
        values.append((int(match.group(1)), int(match.group(2))))
    elif "," in raw:
        parts = [part.strip() for part in raw.split(",", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            values.append((int(parts[0]), int(parts[1])))
    elif y is not None and raw:
        values.append((int(raw), int(y)))
    out: list[str] = []
    for x_value, y_value in values:
        for candidate in (f"x{x_value}_y{y_value}", f"x{x_value:03d}_y{y_value:03d}"):
            if candidate not in out:
                out.append(candidate)
    if raw and raw not in out:
        out.insert(0, raw)
    return out


def _tile_heat_color(count: int) -> tuple[int, int, int]:
    count = int(count)
    if count <= 0:
        return (178, 181, 176)
    level = min(15, max(1, count))
    t = (level - 1) / 14.0 if level > 1 else 0.0
    if t <= 0.5:
        u = t / 0.5
        c0 = np.array([35, 165, 95], dtype=np.float32)
        c1 = np.array([245, 225, 40], dtype=np.float32)
    else:
        u = (t - 0.5) / 0.5
        c0 = np.array([245, 225, 40], dtype=np.float32)
        c1 = np.array([215, 55, 45], dtype=np.float32)
    color = np.rint((1.0 - u) * c0 + u * c1).astype(int)
    return int(color[0]), int(color[1]), int(color[2])


def load_html() -> bytes:
    return HTML_PATH.read_bytes()


def display_gray(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    try:
        lo, hi = ZScaleInterval().get_limits(finite)
    except Exception:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = np.clip((np.nan_to_num(arr, nan=lo) - lo) / (hi - lo), 0.0, 1.0)
    return np.flipud(np.rint(255.0 * scaled).astype(np.uint8))


def _finite_stats(image: np.ndarray) -> tuple[np.ndarray, float, float]:
    arr = np.asarray(image, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return finite, 0.0, 1.0
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return finite, lo, hi


def _percentile_or_default(values: np.ndarray, pct: float, default: float) -> float:
    if values.size == 0:
        return float(default)
    pct = min(100.0, max(0.0, float(pct)))
    value = float(np.percentile(values, pct))
    return value if np.isfinite(value) else float(default)


def _display_limits(
    image: np.ndarray,
    *,
    custom: bool,
    low_pct: float,
    high_pct: float,
    low_value: float | None,
    high_value: float | None,
    default_low: float | None = None,
    default_high: float | None = None,
) -> tuple[float, float]:
    finite, image_min, image_max = _finite_stats(image)
    if custom:
        lo = float(low_value) if low_value is not None and np.isfinite(low_value) else _percentile_or_default(finite, low_pct, image_min)
        hi = float(high_value) if high_value is not None and np.isfinite(high_value) else _percentile_or_default(finite, high_pct, image_max)
    else:
        lo = image_min if default_low is None else float(default_low)
        hi = image_max if default_high is None else float(default_high)
    if not np.isfinite(lo):
        lo = image_min
    if not np.isfinite(hi):
        hi = image_max
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _normalize_between(image: np.ndarray, lo: float, hi: float) -> np.ndarray:
    safe = np.nan_to_num(np.asarray(image, dtype=np.float32), nan=float(lo), posinf=float(hi), neginf=float(lo))
    return np.clip((safe - float(lo)) / max(float(hi) - float(lo), 1e-6), 0.0, 1.0).astype(np.float32)


def _display_scaled_plane(
    image: np.ndarray,
    *,
    display_scaling: str,
    scale_custom: bool = False,
    scale_low_pct: float = 0.0,
    scale_high_pct: float = 100.0,
    scale_low_value: float | None = None,
    scale_high_value: float | None = None,
    log_a: float = 1000.0,
    lupton_stretch: float = 0.5,
    lupton_q: float = 20.0,
    anscombe_scale: float = 1000.0,
) -> np.ndarray:
    mode = str(display_scaling or "zscale").strip().lower().replace("_", "-")
    arr = np.asarray(image, dtype=np.float32)
    finite, image_min, image_max = _finite_stats(arr)
    if mode == "zscale":
        try:
            lo, hi = ZScaleInterval().get_limits(finite)
        except Exception:
            lo, hi = image_min, image_max
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = image_min, image_max
        return _normalize_between(arr, lo, hi)
    if mode == "zmax":
        lo, hi = _display_limits(
            arr,
            custom=scale_custom,
            low_pct=scale_low_pct,
            high_pct=scale_high_pct,
            low_value=scale_low_value,
            high_value=scale_high_value,
        )
        return _normalize_between(arr, lo, hi)
    if mode == "asinh":
        _z, stats = display_sam_zscore(arr)
        default_low = float(stats.get("zscore_median", stats.get("median", image_min)))
        lo, hi = _display_limits(
            arr,
            custom=scale_custom,
            low_pct=scale_low_pct,
            high_pct=scale_high_pct,
            low_value=scale_low_value,
            high_value=scale_high_value,
            default_low=default_low,
            default_high=image_max,
        )
        x = _normalize_between(arr, lo, hi)
        return np.clip(np.arcsinh(10.0 * x) / 3.0, 0.0, 1.0).astype(np.float32)
    if mode == "log":
        lo, hi = _display_limits(
            arr,
            custom=scale_custom,
            low_pct=scale_low_pct,
            high_pct=scale_high_pct,
            low_value=scale_low_value,
            high_value=scale_high_value,
            default_low=image_min,
            default_high=image_max,
        )
        x = _normalize_between(arr, lo, hi)
        a = float(log_a) if np.isfinite(float(log_a)) and float(log_a) > 0.0 else 1000.0
        return np.clip(np.log1p(a * x) / np.log1p(a), 0.0, 1.0).astype(np.float32)
    if mode == "lupton":
        _z, stats = display_sam_zscore(arr)
        default_low = float(stats.get("zscore_median", stats.get("median", image_min)))
        lo, hi = _display_limits(
            arr,
            custom=scale_custom,
            low_pct=scale_low_pct,
            high_pct=scale_high_pct,
            low_value=scale_low_value,
            high_value=scale_high_value,
            default_low=default_low,
            default_high=image_max,
        )
        clipped = np.clip(np.nan_to_num(arr, nan=lo, posinf=hi, neginf=lo), lo, hi)
        plane, _stats = display_lupton_single(clipped, minimum=lo, stretch=lupton_stretch, q=lupton_q)
        return np.clip(np.nan_to_num(plane, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
    if mode == "anscombe":
        base = np.clip(np.nan_to_num(arr, nan=image_min, posinf=image_max, neginf=image_min), image_min, image_max)
        plane, _stats = display_anscombe_single(base, scale=anscombe_scale, clip=False)
        lo, hi = _display_limits(
            plane,
            custom=scale_custom,
            low_pct=scale_low_pct,
            high_pct=scale_high_pct,
            low_value=None,
            high_value=None,
        )
        return _normalize_between(plane, lo, hi)
    if mode == "square":
        lo, hi = _display_limits(
            arr,
            custom=scale_custom,
            low_pct=scale_low_pct,
            high_pct=scale_high_pct,
            low_value=scale_low_value,
            high_value=scale_high_value,
        )
        x = np.maximum(np.nan_to_num(arr, nan=lo, posinf=hi, neginf=lo) - lo, 0.0)
        y = x * x
        return np.clip(y / max((hi - lo) * (hi - lo), 1e-6), 0.0, 1.0).astype(np.float32)
    raise ValueError(f"unknown display scaling: {display_scaling}")


def _smooth_float_image(image: np.ndarray, sigma: float) -> np.ndarray:
    sigma = float(sigma)
    arr = np.asarray(image, dtype=np.float32)
    if sigma <= 0.0:
        return arr
    finite = np.isfinite(arr)
    fill = float(np.nanmedian(arr[finite])) if np.any(finite) else 0.0
    arr = np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill).astype(np.float32, copy=False)
    radius = max(1, int(np.ceil(2.0 * sigma)))
    axis = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (axis / sigma) ** 2)
    kernel /= np.sum(kernel)

    def convolve_axis(values: np.ndarray, axis_index: int) -> np.ndarray:
        pad = [(0, 0)] * values.ndim
        pad[axis_index] = (radius, radius)
        padded = np.pad(values, pad, mode="reflect")
        return np.apply_along_axis(lambda line: np.convolve(line, kernel, mode="valid"), axis_index, padded)

    out = convolve_axis(arr, 1)
    out = convolve_axis(out, 0)
    return out.astype(np.float32, copy=False)


def _boxcar_float_image(image: np.ndarray, radius: int) -> np.ndarray:
    radius = int(radius)
    arr = np.asarray(image, dtype=np.float32)
    if radius <= 0:
        return arr
    finite = np.isfinite(arr)
    fill = float(np.nanmedian(arr[finite])) if np.any(finite) else 0.0
    arr = np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill).astype(np.float32, copy=False)
    size = 2 * radius + 1
    padded = np.pad(arr, ((radius, radius), (radius, radius)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    total = integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size] + integral[:-size, :-size]
    return (total / float(size * size)).astype(np.float32, copy=False)


def _tophat_float_image(image: np.ndarray, radius: int) -> np.ndarray:
    radius = int(radius)
    arr = np.asarray(image, dtype=np.float32)
    if radius <= 0:
        return arr
    finite = np.isfinite(arr)
    fill = float(np.nanmedian(arr[finite])) if np.any(finite) else 0.0
    arr = np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill).astype(np.float32, copy=False)
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = ((xx * xx + yy * yy) <= radius * radius).astype(np.float32)
    denom = float(mask.sum())
    padded = np.pad(arr, ((radius, radius), (radius, radius)), mode="reflect")
    out = np.zeros_like(arr, dtype=np.float32)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            weight = float(mask[dy, dx])
            if weight:
                out += weight * padded[dy : dy + arr.shape[0], dx : dx + arr.shape[1]]
    return (out / denom).astype(np.float32, copy=False)


def _display_filter_image(
    image: np.ndarray,
    *,
    smooth_mode: str = "none",
    smooth_sigma: float = 1.0,
    smooth_radius: int = 1,
) -> np.ndarray:
    mode = str(smooth_mode or "none").replace("_", "-").lower()
    if mode in {"", "none", "off"}:
        return np.asarray(image, dtype=np.float32)
    if mode == "gaussian":
        return _smooth_float_image(image, float(smooth_sigma))
    if mode == "boxcar":
        return _boxcar_float_image(image, int(smooth_radius))
    if mode == "tophat":
        return _tophat_float_image(image, int(smooth_radius))
    raise ValueError(f"unknown smoothing mode: {smooth_mode}")


def _png_bytes(rgb_or_gray: np.ndarray) -> bytes:
    arr = np.asarray(rgb_or_gray)
    handle = io.BytesIO()
    if arr.ndim == 2:
        image = Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
    else:
        image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
    image.save(handle, format="PNG", optimize=True)
    return handle.getvalue()


def _overlay_png_bytes(image: np.ndarray, rows: list[dict[str, float]], *, draw_centers: bool, invert_background: bool = False) -> bytes:
    rgb = draw_ellipses(
        image,
        rows,
        color="cyan",
        draw_centers=draw_centers,
        point_color="blue",
        invert_background=invert_background,
    )
    arr = np.clip(np.rint(np.flipud(rgb) * 255.0), 0, 255).astype(np.uint8)
    return _png_bytes(arr)


def _overlay_uint8(image: np.ndarray, rows: list[dict[str, float]], *, draw_centers: bool, invert_background: bool = False) -> np.ndarray:
    rgb = draw_ellipses(
        image,
        rows,
        color="cyan",
        draw_centers=draw_centers,
        point_color="blue",
        invert_background=invert_background,
    )
    return np.clip(np.rint(np.flipud(rgb) * 255.0), 0, 255).astype(np.uint8)


def _input_overlay_channel(scaled: np.ndarray, scaling: str) -> tuple[int, np.ndarray] | None:
    label = str(scaling).replace("_", "-").lower()
    arr = np.asarray(scaled, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] <= 0:
        return None
    if "log-lupton" in label or "lupton" in label:
        idx = min(2, arr.shape[0] - 1)
        return idx, arr[idx]
    if "anscombe" in label or "zscore" in label:
        return 0, arr[0]
    return None


def _input_shape_overlay_png_bytes(
    scaled: np.ndarray,
    rows: list[dict[str, float]],
    *,
    scaling: str,
    clip_threshold: float,
    draw_centers: bool,
    invert_background: bool = False,
    use_row_colors: bool = False,
) -> bytes:
    selection = _input_overlay_channel(scaled, scaling)
    if selection is None:
        gray = display_gray(np.asarray(scaled[0] if np.asarray(scaled).ndim == 3 else scaled))
        if invert_background:
            gray = 255 - gray
        return _png_bytes(gray)
    channel_idx, channel = selection
    rgb = draw_ellipses(
        channel,
        rows,
        color=None if use_row_colors else "cyan",
        draw_centers=draw_centers,
        point_color=None if use_row_colors else "blue",
        invert_background=invert_background,
        input_scaled_background=True,
        input_scaling=scaling,
        input_channel_index=channel_idx,
        input_clip_threshold=float(clip_threshold),
    )
    arr = np.clip(np.rint(np.flipud(rgb) * 255.0), 0, 255).astype(np.uint8)
    return _png_bytes(arr)


def _input_display_png_bytes(
    scaled: np.ndarray,
    *,
    scaling: str,
    clip_threshold: float,
    invert_background: bool = False,
) -> bytes:
    return _input_shape_overlay_png_bytes(
        scaled,
        [],
        scaling=scaling,
        clip_threshold=float(clip_threshold),
        draw_centers=False,
        invert_background=invert_background,
    )


def _input_shape_overlay_uint8(
    scaled: np.ndarray,
    rows: list[dict[str, float]],
    *,
    scaling: str,
    clip_threshold: float,
    draw_centers: bool,
    smooth_mode: str = "none",
    smooth_sigma: float = 1.0,
    smooth_radius: int = 1,
    invert_background: bool = False,
    use_row_colors: bool = False,
) -> np.ndarray:
    selection = _input_overlay_channel(scaled, scaling)
    if selection is None:
        gray = display_gray(np.asarray(scaled[0] if np.asarray(scaled).ndim == 3 else scaled))
        if invert_background:
            gray = 255 - gray
        return np.repeat(gray[..., None], 3, axis=2)
    channel_idx, channel = selection
    channel = _display_filter_image(
        channel,
        smooth_mode=smooth_mode,
        smooth_sigma=smooth_sigma,
        smooth_radius=smooth_radius,
    )
    rgb = draw_ellipses(
        channel,
        rows,
        color=None if use_row_colors else "cyan",
        draw_centers=draw_centers,
        point_color=None if use_row_colors else "blue",
        invert_background=invert_background,
        input_scaled_background=True,
        input_scaling=scaling,
        input_channel_index=channel_idx,
        input_clip_threshold=float(clip_threshold),
    )
    return np.clip(np.rint(np.flipud(rgb) * 255.0), 0, 255).astype(np.uint8)


def _input_display_uint8(
    scaled: np.ndarray,
    *,
    scaling: str,
    clip_threshold: float,
    smooth_mode: str = "none",
    smooth_sigma: float = 1.0,
    smooth_radius: int = 1,
    invert_background: bool = False,
) -> np.ndarray:
    return _input_shape_overlay_uint8(
        scaled,
        [],
        scaling=scaling,
        clip_threshold=float(clip_threshold),
        draw_centers=False,
        smooth_mode=smooth_mode,
        smooth_sigma=smooth_sigma,
        smooth_radius=smooth_radius,
        invert_background=invert_background,
    )


def _display_input_uint8(
    image: np.ndarray,
    *,
    display_scaling: str,
    scale_custom: bool = False,
    scale_low_pct: float = 0.0,
    scale_high_pct: float = 100.0,
    scale_low_value: float | None = None,
    scale_high_value: float | None = None,
    smooth_mode: str = "none",
    smooth_sigma: float = 1.0,
    smooth_radius: int = 1,
    invert_background: bool = False,
) -> np.ndarray:
    plane = _display_scaled_plane(
        image,
        display_scaling=display_scaling,
        scale_custom=scale_custom,
        scale_low_pct=scale_low_pct,
        scale_high_pct=scale_high_pct,
        scale_low_value=scale_low_value,
        scale_high_value=scale_high_value,
    )
    plane = _display_filter_image(
        plane,
        smooth_mode=smooth_mode,
        smooth_sigma=smooth_sigma,
        smooth_radius=smooth_radius,
    )
    gray = np.clip(np.rint(np.nan_to_num(plane, nan=0.0, posinf=1.0, neginf=0.0) * 255.0), 0, 255).astype(np.uint8)
    if invert_background:
        gray = 255 - gray
    return np.repeat(np.flipud(gray)[..., None], 3, axis=2)


def _display_input_shape_overlay_uint8(
    image: np.ndarray,
    rows: list[dict[str, float]],
    *,
    display_scaling: str,
    scale_custom: bool = False,
    scale_low_pct: float = 0.0,
    scale_high_pct: float = 100.0,
    scale_low_value: float | None = None,
    scale_high_value: float | None = None,
    draw_centers: bool,
    smooth_mode: str = "none",
    smooth_sigma: float = 1.0,
    smooth_radius: int = 1,
    invert_background: bool = False,
    use_row_colors: bool = False,
) -> np.ndarray:
    plane = _display_scaled_plane(
        image,
        display_scaling=display_scaling,
        scale_custom=scale_custom,
        scale_low_pct=scale_low_pct,
        scale_high_pct=scale_high_pct,
        scale_low_value=scale_low_value,
        scale_high_value=scale_high_value,
    )
    plane = _display_filter_image(
        plane,
        smooth_mode=smooth_mode,
        smooth_sigma=smooth_sigma,
        smooth_radius=smooth_radius,
    )
    rgb = draw_ellipses(
        plane,
        rows,
        color=None if use_row_colors else "cyan",
        draw_centers=draw_centers,
        point_color=None if use_row_colors else "blue",
        invert_background=invert_background,
        input_scaled_background=True,
        input_scaling="zscore-no-upper",
        input_channel_index=0,
        input_clip_threshold=1.0,
    )
    return np.clip(np.rint(np.flipud(rgb) * 255.0), 0, 255).astype(np.uint8)


def _draw_centers_on_uint8(image_rgb: np.ndarray, rows: list[dict[str, float]], *, radius: int = 5) -> np.ndarray:
    arr = np.asarray(image_rgb).copy()
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    height, width = arr.shape[:2]
    pil = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(pil)
    color = (0, 90, 255)
    for row in rows:
        x = int(round(float(row.get("x", 0.0))))
        y = height - 1 - int(round(float(row.get("y", 0.0))))
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        draw.line((x - radius, y, x + radius, y), fill=color, width=1)
        draw.line((x, y - radius, x, y + radius), fill=color, width=1)
    return np.asarray(pil, dtype=np.uint8)


def _save_titled_png(path: Path, image_rgb: np.ndarray, title: str, *, min_image_size: int = 512) -> None:
    arr = np.asarray(image_rgb)
    if arr.ndim == 2:
        pil = Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
    else:
        pil = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
    scale = max(1, int(np.ceil(float(min_image_size) / float(max(pil.size)))))
    if scale > 1:
        pil = pil.resize((pil.width * scale, pil.height * scale), Image.Resampling.NEAREST)
    title_suffix = f" ({scale}x)" if scale > 1 else ""
    font = ImageFont.load_default()
    title_text = f"{title}{title_suffix}"
    title_h = 30
    canvas = Image.new("RGB", (pil.width, pil.height + title_h), "white")
    canvas.paste(pil, (0, title_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title_text, fill=(20, 20, 20), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _save_snr_overlay_png(
    path: Path,
    image: np.ndarray,
    rows: list[dict[str, Any]],
    *,
    title: str,
    threshold: float = 5.0,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as patheffects
    from matplotlib.patches import Ellipse

    arr = np.asarray(image, dtype=np.float32)
    height, width = arr.shape
    fig, ax = plt.subplots(figsize=(width / 100.0, height / 100.0), dpi=100)
    ax.imshow(zscale_gray(arr), origin="lower", cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    for row in sorted(rows, key=lambda item: abs(float(item.get("major", 1.0)) * float(item.get("minor", 1.0))), reverse=True):
        snr = float(row.get("snr", float("nan")))
        bad = bool(int(row.get("snr_bad", 0))) or not np.isfinite(snr)
        color = "0.65" if bad else ("lime" if snr > float(threshold) else "red")
        x = float(row.get("x", 0.0))
        y = float(row.get("y", 0.0))
        major = max(abs(float(row.get("major", 1.0))), 1.0)
        minor = max(abs(float(row.get("minor", 1.0))), 1.0)
        theta = float(row.get("theta", 0.0))
        if abs(theta) > 2.0 * np.pi:
            theta = np.deg2rad(theta)
        ax.add_patch(
            Ellipse(
                (x, y),
                width=2.0 * major,
                height=2.0 * minor,
                angle=float(np.rad2deg(theta)),
                fill=False,
                edgecolor=color,
                linewidth=1.1,
                alpha=0.95,
            )
        )
        label = "bad" if bad else f"{snr:.1f}"
        ax.text(
            x + 4.0,
            y + 4.0,
            label,
            color=color,
            fontsize=7,
            fontweight="bold",
            path_effects=[patheffects.withStroke(linewidth=1.6, foreground="black")],
        )
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 0.94)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def dataset_label(dataset_id: str) -> str:
    return {
        "hsc_raw": "HSC raw tiles",
        "sitian": "Sitian",
        "hsc_image": "HSC coadd/noisy/denoised",
        "ztf": "ZTF",
        "jwst": "JWST NIRCam",
    }.get(str(dataset_id), str(dataset_id))


def make_access(dataset_id: str, args: argparse.Namespace, tract: str):
    def path_arg(name: str, default: Path) -> Path:
        value = getattr(args, name, None)
        return Path(default if value is None else value)

    dataset_id = str(dataset_id)
    if dataset_id == "hsc_raw":
        return HscRawAccess(Path(args.root), tract)
    if dataset_id == "hsc_image":
        return HscImageAccess(
            Path(args.hsc_image_root),
            tract,
            weight_root=path_arg("hsc_weight_root", DEFAULT_HSC_WEIGHT_ROOT),
            coadd_weight_root=path_arg("hsc_coadd_weight_root", DEFAULT_HSC_WEIGHT_ROOT),
            variant_weight_root=path_arg("hsc_variant_weight_root", DEFAULT_HSC_WEIGHT_ROOT),
            coadd_fits_root=path_arg("hsc_coadd_fits_root", DEFAULT_HSC_COADD_FITS_ROOT),
            noisy_fits_root=path_arg("hsc_noisy_fits_root", DEFAULT_HSC_NOISY_FITS_ROOT),
            denoised_fits_root=path_arg("hsc_denoised_fits_root", DEFAULT_HSC_DENOISED_FITS_ROOT),
        )
    if dataset_id == "sitian":
        return MessierAccess(Path(args.messier_root), tract, selection_mode=str(args.messier_tile_mode))
    if dataset_id == "ztf":
        return ZtfAccess(
            Path(args.ztf_root),
            tract,
            ccd=str(args.ztf_ccd),
            tile_size=int(args.ztf_tile_size),
            cut_origin_dir=Path(args.ztf_cut_origin_dir) if args.ztf_cut_origin_dir else None,
        )
    if dataset_id == "jwst":
        return JwstNircamAccess(
            Path(getattr(args, "jwst_root", DEFAULT_JWST_NIRCAM_ROOT)),
            "default",
            tile_size=int(getattr(args, "jwst_tile_size", 512)),
        )
    raise KeyError(f"unknown dataset: {dataset_id}")


class BrowserState:
    def __init__(
        self,
        args: argparse.Namespace,
        *,
        tract: str,
        patches: list[str],
        bands: list[str],
        n_tiles: int | None,
        all_tiles: bool,
        frames_per_tile: int | None = None,
        tiles_per_page: int | None = None,
        run_name: str = "",
        stamp: str | None = None,
    ) -> None:
        self.run_name = str(run_name or "")
        self.stamp = str(stamp or time.strftime("%Y%m%d_%H%M%S"))
        self.session_name = _session_name(self.run_name, self.stamp)
        self.dataset_id = str(getattr(args, "dataset", "hsc_raw"))
        self.args_tile_selection_mode = str(args.messier_tile_mode) if self.dataset_id == "sitian" else "default"
        self.access = make_access(self.dataset_id, args, tract)
        self.root = Path(getattr(self.access, "root", args.root)).expanduser().resolve()
        self.tract = str(tract)
        self.patches = sorted([str(value) for value in patches], key=patch_sort_key)
        self.patch = self.patches[0]
        self.bands = [str(value) for value in bands]
        if self.dataset_id == "sitian":
            default_n_tiles = int(args.messier_n_tiles)
            default_frames_per_tile = 1
        elif self.dataset_id == "ztf":
            default_n_tiles = int(args.ztf_n_tiles)
            default_frames_per_tile = int(args.ztf_frames_per_tile)
        elif self.dataset_id == "jwst":
            default_n_tiles = int(args.jwst_n_tiles)
            default_frames_per_tile = 1
        else:
            default_n_tiles = int(args.n_tiles)
            default_frames_per_tile = int(args.frames_per_tile)
        self.n_tiles_requested = None if all_tiles else int(n_tiles if n_tiles is not None else default_n_tiles)
        self.all_tiles = bool(all_tiles)
        self.frames_per_tile = max(1, int(frames_per_tile if frames_per_tile is not None else default_frames_per_tile))
        if self.dataset_id == "jwst":
            self.frames_per_tile = 1
        self.tiles_per_page = max(1, int(tiles_per_page if tiles_per_page is not None else args.tiles_per_page))
        self.detect_batch_size = max(1, int(args.detect_batch_size))
        self.seed = int(args.seed)
        self.visit = int(args.visit) if args.visit is not None else None
        self.frame_rank = int(args.frame_rank)
        self.strict_visit = bool(args.strict_visit)
        self.checkpoint = Path(args.checkpoint).expanduser().resolve()
        self.config = Path(args.config).expanduser().resolve() if args.config else None
        self.device_name = str(args.device)
        self.amp = str(args.amp)
        self.scaling_mode = str(args.scaling_mode)
        self.clip_threshold = float(args.clip_threshold)
        self.log_a = float(args.log_a)
        self.log_high_percentile = float(args.log_high_percentile)
        self.lupton_stretch = float(args.lupton_stretch)
        self.lupton_q = float(args.lupton_q)
        self.anscombe_clip = bool(args.anscombe_clip)
        self.anscombe_scale = float(args.anscombe_scale)
        self.confidence_threshold = float(args.confidence_threshold)
        self.confidence_score = str(args.confidence_score)
        self.nms_radius = int(getattr(args, "active_nms_radius", args.nms_radius))
        self.center_refinement = str(args.center_refinement)
        self.center_refinement_radius = int(args.center_refinement_radius)
        self.shape_overlay_centers = bool(args.shape_overlay_centers)
        self.session_dir = Path(args.session_dir).expanduser().resolve()
        self.export_dir = Path(args.export_dir).expanduser().resolve()
        self.selected: set[str] = set()
        self.warnings: list[str] = []
        self.bands_by_patch: dict[str, list[str]] = {}
        self.selected_tiles_by_patch: dict[str, list[str]] = {}
        self.frame_slots_by_patch: dict[str, dict[str, int]] = {}
        self.refs_by_patch: dict[str, list[FrameRef]] = {}
        self.ref_by_token: dict[str, FrameRef] = {}
        self.input_png_by_token: dict[str, bytes] = {}
        self.detect_png_by_token: dict[str, bytes] = {}
        self.input_shape_png_by_token: dict[str, bytes] = {}
        self.detect_rows_by_token: dict[str, list[dict[str, float]]] = {}
        self.snr_rows_by_token_method: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.snr_config = SourceSnrConfig()
        self.tile_map_png_by_patch: dict[str, bytes] = {}
        self._model_by_bands: dict[tuple[str, ...], tuple[torch.nn.Module, dict[str, Any]]] = {}
        self._model_lock = threading.Lock()
        for patch in self.patches:
            self.selected_tiles_by_patch[patch] = self._choose_tiles(patch, self.seed + 1009 * len(self.selected_tiles_by_patch))
        self._build_refs()
        self._build_tile_maps()
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._write_manifest()

    def _choose_tiles(self, patch: str, seed: int) -> list[str]:
        bands = self._usable_bands_for_patch(patch)
        self.bands_by_patch[patch] = bands
        if not bands:
            self.warnings.append(f"{patch}: no requested bands are available; page will be empty")
            return []
        try:
            tiles = self.access.choose_tiles(
                patch,
                bands,
                n_tiles=self.n_tiles_requested,
                all_tiles=self.all_tiles,
                seed=seed,
                mode=self.args_tile_selection_mode,
            )
        except Exception as exc:
            self.warnings.append(f"{patch}: tile selection failed: {exc}")
            print(f"WARNING: {patch}: tile selection failed: {exc}", flush=True)
            return []
        if not tiles:
            self.warnings.append(f"{patch}: no common tiles for available bands {bands}; page will be empty")
        return tiles

    def _usable_bands_for_patch(self, patch: str) -> list[str]:
        if self.dataset_id == "jwst" and hasattr(self.access, "image_file"):
            usable = []
            for band in self.bands:
                try:
                    self.access.image_file(band, patch)
                except Exception as exc:
                    self.warnings.append(f"{patch}: skip unavailable {band}: {exc}")
                    print(f"WARNING: {patch}: skip unavailable {band}: {exc}", flush=True)
                    continue
                usable.append(band)
            return usable
        if self.dataset_id in {"hsc_raw", "ztf"} and hasattr(self.access, "valid_tiles_for_band"):
            usable = []
            for band in self.bands:
                try:
                    valid = self.access.valid_tiles_for_band(band, patch)
                except Exception as exc:
                    self.warnings.append(f"{patch}: skip unavailable {band}: {exc}")
                    print(f"WARNING: {patch}: skip unavailable {band}: {exc}", flush=True)
                    continue
                if valid:
                    usable.append(band)
                else:
                    self.warnings.append(f"{patch}: skip empty {band}")
                    print(f"WARNING: {patch}: skip empty {band}", flush=True)
            return usable
        return list(self.bands)

    def _bands_for_patch(self, patch: str | None = None) -> list[str]:
        return self.bands_by_patch.get(str(patch or self.patch), list(self.bands))

    def _tile_slot_count(self, patch: str, tile_id: str) -> int:
        return self.access.tile_slot_count(
            patch,
            tile_id,
            self._bands_for_patch(patch),
            frames_per_tile=self.frames_per_tile,
            visit=self.visit,
        )

    def _build_refs(self) -> None:
        token = 0
        for patch in self.patches:
            refs: list[FrameRef] = []
            self.frame_slots_by_patch[patch] = {}
            bands = self._bands_for_patch(patch)
            for tile_id in self.selected_tiles_by_patch[patch]:
                slot_count = self._tile_slot_count(patch, tile_id)
                self.frame_slots_by_patch[patch][tile_id] = slot_count
                for frame_slot in range(slot_count):
                    for band in bands:
                        try:
                            ref = self.access.make_ref(
                                token=f"P{token:05d}",
                                patch=patch,
                                band=band,
                                tile_id=tile_id,
                                frame_slot=frame_slot,
                                frame_rank=self.frame_rank,
                                frames_per_tile=self.frames_per_tile,
                                visit=self.visit,
                                strict_visit=self.strict_visit,
                            )
                        except Exception as exc:
                            message = f"{patch} {tile_id} slot {frame_slot + 1}: skip {band}: {exc}"
                            self.warnings.append(message)
                            print(f"WARNING: {message}", flush=True)
                            continue
                        refs.append(ref)
                        self.ref_by_token[ref.token] = ref
                        token += 1
            self.refs_by_patch[patch] = refs

    def _all_map_tile_ids(self, patch: str) -> list[str]:
        if self.dataset_id == "jwst":
            return list(self.selected_tiles_by_patch.get(patch, []))
        if self.dataset_id in {"hsc_raw", "ztf", "jwst"} and hasattr(self.access, "valid_tiles_for_band"):
            try:
                sets = [self.access.valid_tiles_for_band(band, patch) for band in self._bands_for_patch(patch)]
                if sets:
                    return sorted(
                        set.intersection(*sets),
                        key=lambda text: tuple(int(v) for v in re.findall(r"\d+", text)),
                    )
            except Exception as exc:
                print(f"WARNING: failed to collect full tile map for {patch}; using selected tiles only: {exc}", flush=True)
        return list(self.selected_tiles_by_patch.get(patch, []))

    def _tile_map_counts(self, patch: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for tile_id in self._all_map_tile_ids(patch):
            counts[tile_id] = 0
        for tile_id, slot_count in self.frame_slots_by_patch.get(patch, {}).items():
            counts[tile_id] = int(slot_count)
        return counts

    def _make_tile_map_png(self, patch: str) -> bytes:
        counts = self._tile_map_counts(patch)
        parsed = {tile_id: xy for tile_id, xy in ((tile_id, _parse_tile_xy(tile_id)) for tile_id in counts) if xy is not None}
        if not parsed:
            return (ASSETS_DIR / "blank.png").read_bytes()

        xs = sorted({xy[0] for xy in parsed.values()})
        ys = sorted({xy[1] for xy in parsed.values()})
        x_index = {value: idx for idx, value in enumerate(xs)}
        y_index = {value: idx for idx, value in enumerate(ys)}
        nx = len(xs)
        ny = len(ys)
        cell = max(34, min(58, int(900 / max(nx, ny, 1))))
        left = 58
        right = 78
        top = 56
        bottom = 54
        width = left + nx * cell + right
        height = top + ny * cell + bottom
        image = Image.new("RGB", (width, height), (246, 246, 243))
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        title = f"{dataset_label(self.dataset_id)} {self.tract}/{patch}: selected groups per tile"
        draw.text((left, 16), title, fill=(25, 25, 25), font=font)

        for tile_id, (tx, ty) in parsed.items():
            ix = x_index[tx]
            iy = y_index[ty]
            # Show tile y increasing upward, matching astronomical tile-map convention.
            py = ny - 1 - iy
            x0 = left + ix * cell
            y0 = top + py * cell
            count = int(counts.get(tile_id, 0))
            color = _tile_heat_color(count)
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=color, outline=(92, 92, 88), width=1)
            text = str(count)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            draw.text((x0 + (cell - tw) / 2, y0 + (cell - th) / 2), text, fill=(10, 10, 10), font=font)

        for tx in xs:
            ix = x_index[tx]
            x = left + ix * cell + cell / 2
            label = str(tx)
            bbox = draw.textbbox((0, 0), label, font=font)
            draw.text((x - (bbox[2] - bbox[0]) / 2, top + ny * cell + 8), label, fill=(35, 35, 35), font=font)
        for ty in ys:
            iy = y_index[ty]
            py = ny - 1 - iy
            y = top + py * cell + cell / 2
            label = str(ty)
            bbox = draw.textbbox((0, 0), label, font=font)
            draw.text((left - 10 - (bbox[2] - bbox[0]), y - (bbox[3] - bbox[1]) / 2), label, fill=(35, 35, 35), font=font)
        draw.text((left + nx * cell / 2 - 26, height - 24), "tile x", fill=(35, 35, 35), font=font)
        draw.text((10, top + ny * cell / 2 - 8), "tile y", fill=(35, 35, 35), font=font)

        cb_x0 = left + nx * cell + 26
        cb_y0 = top
        cb_w = 22
        cb_h = ny * cell
        for j in range(cb_h):
            t = 1.0 - j / max(1, cb_h - 1)
            level = int(round(t * 15))
            color = _tile_heat_color(level)
            draw.line((cb_x0, cb_y0 + j, cb_x0 + cb_w, cb_y0 + j), fill=color)
        draw.rectangle((cb_x0, cb_y0, cb_x0 + cb_w, cb_y0 + cb_h), outline=(60, 60, 60), width=1)
        draw.text((cb_x0 + cb_w + 6, cb_y0 - 2), "15+", fill=(35, 35, 35), font=font)
        draw.text((cb_x0 + cb_w + 6, cb_y0 + cb_h - 10), "0", fill=(35, 35, 35), font=font)

        path = self.session_dir / f"tile_map_{_slug_text(str(patch), fallback='patch')}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        return path.read_bytes()

    def _build_tile_maps(self) -> None:
        for patch in self.patches:
            self.tile_map_png_by_patch[patch] = self._make_tile_map_png(patch)

    @property
    def refs(self) -> list[FrameRef]:
        return self.refs_by_patch.get(self.patch, [])

    @property
    def selected_tiles(self) -> list[str]:
        return self.selected_tiles_by_patch.get(self.patch, [])

    @property
    def n_pages(self) -> int:
        return max(1, int(np.ceil(len(self.selected_tiles) / self.tiles_per_page)))

    def _write_manifest(self) -> None:
        payload = {
            "root": str(self.root),
            "dataset": self.dataset_id,
            "dataset_label": dataset_label(self.dataset_id),
            "run_name": self.run_name,
            "session_name": self.session_name,
            "tract": self.tract,
            "patches": self.patches,
            "bands": self.bands,
            "bands_by_patch": self.bands_by_patch,
            "tile_size": int(getattr(self.access, "tile_size", 256)),
            "n_tiles_by_patch": {patch: len(values) for patch, values in self.selected_tiles_by_patch.items()},
            "frames_per_tile": self.frames_per_tile,
            "tiles_per_page": self.tiles_per_page,
            "detect_batch_size": self.detect_batch_size,
            "n_candidates_by_patch": {patch: len(values) for patch, values in self.refs_by_patch.items()},
            "selected_tile_ids_by_patch": self.selected_tiles_by_patch,
            "checkpoint": str(self.checkpoint),
            "scaling_mode": self.scaling_mode,
            "visit": self.visit,
            "note": (
                f"{dataset_label(self.dataset_id)} frames; HSC image detection uses FITS crops with scaling={self.scaling_mode}."
                if self.dataset_id == "hsc_image"
                else f"{dataset_label(self.dataset_id)} frames; detection uses browser scaling={self.scaling_mode}."
            ),
        }
        (self.session_dir / "browser_manifest.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def state_payload(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "dataset": self.dataset_id,
            "dataset_label": dataset_label(self.dataset_id),
            "run_name": self.run_name,
            "session_name": self.session_name,
            "tract": self.tract,
            "started": True,
            "patches": self.patches,
            "patch": self.patch,
            "bands": self._bands_for_patch(self.patch),
            "requested_bands": self.bands,
            "tile_size": int(getattr(self.access, "tile_size", 256)),
            "n_tiles": len(self.selected_tiles),
            "frames_per_tile": self.frames_per_tile,
            "tiles_per_page": self.tiles_per_page,
            "detect_batch_size": self.detect_batch_size,
            "nms_radius": self.nms_radius,
            "n_candidates": len(self.refs),
            "n_selected": len(self.selected),
            "n_pages": self.n_pages,
            "checkpoint_name": self.checkpoint.name,
            "scaling_mode": self.scaling_mode,
            "visit": self.visit,
            "session_dir": str(self.session_dir),
            "export_dir": str(self.export_dir),
            "warnings": self.warnings[-20:],
        }

    def set_patch(self, patch: str) -> None:
        if patch not in self.refs_by_patch:
            raise KeyError(f"unknown patch: {patch}")
        self.patch = patch

    def _tile_summary(self, tile_id: str) -> dict[str, Any]:
        refs = [ref for ref in self.refs if ref.tile_id == tile_id]
        if not refs:
            raise KeyError(f"tile is not loaded in current patch: {tile_id}")
        ref = refs[0]
        index = self.selected_tiles.index(tile_id)
        return {
            "tile_id": tile_id,
            "tile_index": int(ref.tile_index),
            "x0": int(ref.x0),
            "y0": int(ref.y0),
            "x1": int(ref.x1),
            "y1": int(ref.y1),
            "page": int(index // self.tiles_per_page),
            "page_display": int(index // self.tiles_per_page + 1),
            "n_pages": int(self.n_pages),
        }

    def find_tile(self, mode: str, x: Any, y: Any = None) -> dict[str, Any]:
        mode = str(mode)
        if mode == "tile_xy":
            candidates = _tile_id_candidates(str(x), int(y) if y not in (None, "") else None)
            tile_id = next((value for value in candidates if value in self.selected_tiles), candidates[0])
            if tile_id not in self.selected_tiles:
                raise KeyError(
                    f"{' or '.join(candidates)} is not loaded in current patch; use max/all tiles if it was not sampled"
                )
            return self._tile_summary(tile_id)
        if mode == "pixel_xy":
            x = int(x)
            y = int(y)
            matches = []
            seen: set[str] = set()
            for ref in self.refs:
                if ref.tile_id in seen:
                    continue
                seen.add(ref.tile_id)
                if int(ref.x0) <= x < int(ref.x1) and int(ref.y0) <= y < int(ref.y1):
                    matches.append(ref.tile_id)
            if not matches:
                raise KeyError(
                    f"no loaded tile contains pixel ({x}, {y}) in current patch; use max/all tiles if it was not sampled"
                )
            matches = sorted(matches, key=lambda tile_id: self.selected_tiles.index(tile_id))
            return self._tile_summary(matches[0])
        raise ValueError(f"unknown search mode: {mode}")

    def page_payload(self, page_index: int) -> dict[str, Any]:
        page_index = max(0, min(page_index, self.n_pages - 1))
        start = page_index * self.tiles_per_page
        tile_ids = set(self.selected_tiles[start : start + self.tiles_per_page])
        refs = []
        for ref in self.refs:
            if ref.tile_id in tile_ids:
                row = ref.to_dict()
                row["selected"] = ref.token in self.selected
                row["image_url"] = f"/image/{ref.token}.png"
                row["input_image_url"] = f"/image/{ref.token}.png?input=1&v={len(self.input_png_by_token)}"
                row["detected"] = ref.token in self.detect_png_by_token
                row["detect_image_url"] = f"/image/{ref.token}.png?detect=1&v={len(self.detect_png_by_token)}"
                row["input_shape_detected"] = ref.token in self.input_shape_png_by_token
                row["input_shape_image_url"] = f"/image/{ref.token}.png?input_shape=1&detect=1&v={len(self.input_shape_png_by_token)}"
                row["n_detections"] = len(self.detect_rows_by_token.get(ref.token, ()))
                refs.append(row)
        return {"page": page_index, "n_pages": self.n_pages, "bands": self._bands_for_patch(self.patch), "tile_ids": sorted(tile_ids), "candidates": refs}

    def set_selected(self, tokens: list[str], selected: bool) -> None:
        for token in tokens:
            if token not in self.ref_by_token:
                raise KeyError(f"unknown token: {token}")
            if selected:
                self.selected.add(token)
            else:
                self.selected.discard(token)

    def selected_refs(self) -> list[FrameRef]:
        return [ref for ref in self.refs if ref.token in self.selected]

    @staticmethod
    def _normalize_snr_method(method: str) -> str:
        normalized = str(method or "ap2").strip().lower().replace("_", "-")
        return "kron" if normalized in {"kron", "kron-snr"} else "ap2"

    def _snr_rows_for_token(self, token: str, method: str) -> list[dict[str, Any]]:
        method = self._normalize_snr_method(method)
        key = (str(token), method)
        cached = self.snr_rows_by_token_method.get(key)
        if cached is not None:
            return cached
        if token not in self.detect_rows_by_token:
            return []
        ref = self.ref_by_token[token]
        image = self._detection_image_for_ref(ref)
        rows, _summary = measure_source_snrs(
            image,
            self.detect_rows_by_token[token],
            method=method,
            config=self.snr_config,
        )
        self.snr_rows_by_token_method[key] = rows
        return rows

    def _snr_filtered_rows(self, token: str, method: str, threshold: float) -> list[dict[str, Any]]:
        return filter_snr_rows(self._snr_rows_for_token(token, method), threshold=float(threshold))

    def prepare_snr_page(self, page_index: int, method: str, threshold: float = 5.0) -> dict[str, Any]:
        method = self._normalize_snr_method(method)
        threshold = float(threshold)
        refs_by_key = self._refs_by_tile_slot()
        tokens: list[str] = []
        for tile_id, frame_slot in self._page_tile_slots(page_index):
            per_band = refs_by_key.get((tile_id, frame_slot), {})
            for band in self._bands_for_patch(self.patch):
                ref = per_band.get(band)
                if ref is not None and ref.token in self.detect_rows_by_token:
                    tokens.append(ref.token)
        n_sources = 0
        n_bad = 0
        by_token: dict[str, dict[str, Any]] = {}
        for token in tokens:
            rows = self._snr_rows_for_token(token, method)
            good_rows = [
                row for row in rows
                if int(row.get("snr_bad", 0)) == 0 and np.isfinite(float(row.get("snr", float("nan")))) and float(row.get("snr", float("nan"))) >= threshold
            ]
            bad_rows = [row for row in rows if int(row.get("snr_bad", 0)) != 0 or not np.isfinite(float(row.get("snr", float("nan"))))]
            n_sources += len(good_rows)
            n_bad += len(bad_rows)
            by_token[token] = {
                "method": method,
                "threshold": threshold,
                "sources": int(len(good_rows)),
                "bad": int(len(bad_rows)),
                "total": int(len(rows)),
            }
        return {
            "stage": "done",
            "method": method,
            "threshold": threshold,
            "n_images": len(tokens),
            "n_sources": int(n_sources),
            "n_bad": int(n_bad),
            "by_token": by_token,
        }

    def _detection_image_for_ref(self, ref: FrameRef) -> np.ndarray:
        reader = getattr(self.access, "read_detection_frame", None)
        if callable(reader):
            return np.asarray(reader(ref), dtype=np.float32)
        return np.asarray(self.access.read_frame(ref), dtype=np.float32)

    def _scaled_input_for_ref(self, ref: FrameRef) -> np.ndarray:
        image = self._detection_image_for_ref(ref)
        return make_training_rgb(
            image,
            mode=self.scaling_mode,
            clip_threshold=self.clip_threshold,
            log_a=self.log_a,
            log_high_percentile=self.log_high_percentile,
            lupton_stretch=self.lupton_stretch,
            lupton_q=self.lupton_q,
            anscombe_clip=self.anscombe_clip,
            anscombe_scale=self.anscombe_scale,
        )

    def _scaled_input_for_token(self, token: str) -> np.ndarray:
        return self._scaled_input_for_ref(self.ref_by_token[token])

    def scale_stats(self, token: str) -> dict[str, Any]:
        if token not in self.ref_by_token:
            raise KeyError(f"unknown token: {token}")
        image = self._detection_image_for_ref(self.ref_by_token[token])
        finite, image_min, image_max = _finite_stats(image)
        percentiles = [0.0, 50.0, 90.0, 95.0, 99.5, 99.9, 99.95, 99.99, 100.0]
        values = {str(p): _percentile_or_default(finite, p, image_min if p <= 50 else image_max) for p in percentiles}
        try:
            zlo, zhi = ZScaleInterval().get_limits(finite)
        except Exception:
            zlo, zhi = image_min, image_max
        if not np.isfinite(zlo) or not np.isfinite(zhi) or zhi <= zlo:
            zlo, zhi = image_min, image_max
        return {
            "token": token,
            "min": image_min,
            "max": image_max,
            "zscale_min": float(zlo),
            "zscale_max": float(zhi),
            "percentiles": values,
            "finite_fraction": float(finite.size / np.asarray(image).size) if np.asarray(image).size else 0.0,
        }

    def image_png(
        self,
        token: str,
        *,
        detect: bool = False,
        input_image: bool = False,
        input_shape: bool = False,
        show_shape: bool = True,
        show_center: bool = False,
        smooth_mode: str = "none",
        smooth_sigma: float = 1.0,
        smooth_radius: int = 1,
        invert_background: bool = False,
        display_scaling: str = "zscale",
        scale_custom: bool = False,
        scale_low_pct: float = 0.0,
        scale_high_pct: float = 100.0,
        scale_low_value: float | None = None,
        scale_high_value: float | None = None,
        snr_filter: bool = False,
        snr_method: str = "ap2",
        snr_threshold: float = 5.0,
    ) -> bytes:
        use_input = bool(input_image or input_shape)
        if detect and token in self.detect_rows_by_token:
            rows = self._snr_filtered_rows(token, snr_method, snr_threshold) if bool(snr_filter) else self.detect_rows_by_token[token]
            if use_input:
                scaled = self._scaled_input_for_token(token)
                if show_shape:
                    arr = _input_shape_overlay_uint8(
                        scaled,
                        rows,
                        scaling=self.scaling_mode,
                        clip_threshold=self.clip_threshold,
                        draw_centers=show_center,
                        smooth_mode=smooth_mode,
                        smooth_sigma=smooth_sigma,
                        smooth_radius=smooth_radius,
                        invert_background=invert_background,
                        use_row_colors=bool(snr_filter),
                    )
                else:
                    arr = _input_display_uint8(
                        scaled,
                        scaling=self.scaling_mode,
                        clip_threshold=self.clip_threshold,
                        smooth_mode=smooth_mode,
                        smooth_sigma=smooth_sigma,
                        smooth_radius=smooth_radius,
                        invert_background=invert_background,
                    )
                    if show_center:
                        arr = _draw_centers_on_uint8(arr, rows)
                return _png_bytes(arr)
            image = _display_filter_image(
                self._detection_image_for_ref(self.ref_by_token[token]),
                smooth_mode=smooth_mode,
                smooth_sigma=smooth_sigma,
                smooth_radius=smooth_radius,
            )
            if show_shape:
                arr = _display_input_shape_overlay_uint8(
                    image,
                    rows,
                    display_scaling=display_scaling,
                    scale_custom=scale_custom,
                    scale_low_pct=scale_low_pct,
                    scale_high_pct=scale_high_pct,
                    scale_low_value=scale_low_value,
                    scale_high_value=scale_high_value,
                    draw_centers=show_center,
                    invert_background=invert_background,
                    use_row_colors=bool(snr_filter),
                )
            else:
                arr = _display_input_uint8(
                    image,
                    display_scaling=display_scaling,
                    scale_custom=scale_custom,
                    scale_low_pct=scale_low_pct,
                    scale_high_pct=scale_high_pct,
                    scale_low_value=scale_low_value,
                    scale_high_value=scale_high_value,
                    invert_background=invert_background,
                )
                if show_center:
                    arr = _draw_centers_on_uint8(arr, rows)
            return _png_bytes(arr)
        if input_image or input_shape:
            return _png_bytes(
                _input_display_uint8(
                    self._scaled_input_for_token(token),
                    scaling=self.scaling_mode,
                    clip_threshold=self.clip_threshold,
                    smooth_mode=smooth_mode,
                    smooth_sigma=smooth_sigma,
                    smooth_radius=smooth_radius,
                    invert_background=invert_background,
                )
            )
        image = _display_filter_image(
            self.access.read_frame(self.ref_by_token[token]),
            smooth_mode=smooth_mode,
            smooth_sigma=smooth_sigma,
            smooth_radius=smooth_radius,
        )
        return _png_bytes(
            _display_input_uint8(
                image,
                display_scaling=display_scaling,
                scale_custom=scale_custom,
                scale_low_pct=scale_low_pct,
                scale_high_pct=scale_high_pct,
                scale_low_value=scale_low_value,
                scale_high_value=scale_high_value,
                invert_background=invert_background,
            )
        )

    def raw_image_png(self, token: str) -> bytes:
        image = self.access.read_frame(self.ref_by_token[token])
        return _png_bytes(display_gray(image))

    def _load_model(self, bands: list[str] | None = None) -> tuple[torch.nn.Module, dict[str, Any]]:
        key = tuple(str(band) for band in (bands or self._bands_for_patch(self.patch)))
        with self._model_lock:
            if key not in self._model_by_bands:
                device = torch.device(self.device_name)
                model, cfg = load_cellect_model(
                    self.checkpoint,
                    self.config,
                    device,
                    list(key),
                    dynamic_image_size=True,
                )
                self._model_by_bands[key] = (model, dict(cfg or {}))
            model, cfg = self._model_by_bands[key]
            return model, dict(cfg or {})

    def _page_tile_slots(self, page_index: int) -> list[tuple[str, int]]:
        page_index = max(0, min(page_index, self.n_pages - 1))
        start = page_index * self.tiles_per_page
        tile_ids = self.selected_tiles[start : start + self.tiles_per_page]
        slot_counts = self.frame_slots_by_patch.get(self.patch, {})
        return [(tile_id, frame_slot) for tile_id in tile_ids for frame_slot in range(int(slot_counts.get(tile_id, 0)))]

    def _refs_by_tile_slot(self) -> dict[tuple[str, int], dict[str, FrameRef]]:
        refs_by_key: dict[tuple[str, int], dict[str, FrameRef]] = {}
        for ref in self.refs:
            refs_by_key.setdefault((ref.tile_id, ref.frame_slot), {})[ref.band] = ref
        return refs_by_key

    def _detect_tile_slot(
        self,
        tile_id: str,
        frame_slot: int,
        refs_by_key: dict[tuple[str, int], dict[str, FrameRef]] | None = None,
    ) -> dict[str, int]:
        refs_by_key = refs_by_key or self._refs_by_tile_slot()
        per_band = refs_by_key.get((tile_id, frame_slot), {})
        bands = [band for band in self._bands_for_patch(self.patch) if band in per_band]
        if not bands:
            return {"n_images": 0, "n_detections": 0}
        if all(per_band[band].token in self.detect_rows_by_token for band in bands):
            return {
                "n_images": len(bands),
                "n_detections": sum(len(self.detect_rows_by_token.get(per_band[band].token, ())) for band in bands),
            }
        try:
            model, _cfg = self._load_model(bands)
        except Exception as exc:
            message = f"{tile_id} slot {frame_slot + 1}: detection skipped for bands {bands}: {exc}"
            self.warnings.append(message)
            print(f"WARNING: {message}", flush=True)
            return {"n_images": 0, "n_detections": 0}
        device = torch.device(self.device_name)
        raw_images = [self._detection_image_for_ref(per_band[band]) for band in bands]
        scaled = [
            make_training_rgb(
                image,
                mode=self.scaling_mode,
                clip_threshold=self.clip_threshold,
                log_a=self.log_a,
                log_high_percentile=self.log_high_percentile,
                lupton_stretch=self.lupton_stretch,
                lupton_q=self.lupton_q,
                anscombe_clip=self.anscombe_clip,
                anscombe_scale=self.anscombe_scale,
            )
            for image in raw_images
        ]
        tensor = torch.from_numpy(np.stack(scaled, axis=0).astype(np.float32, copy=False))[None]
        outputs = infer_cellect(model=model, image_tensor=tensor, device=device, amp=self.amp)
        total_images = 0
        total_detections = 0
        for band_idx, band in enumerate(bands):
            ref = per_band[band]
            rows = detection_rows(
                select_band_outputs(outputs, band_idx),
                threshold=self.confidence_threshold,
                nms_radius=self.nms_radius,
                confidence_score=self.confidence_score,
                center_refinement=self.center_refinement,
                center_refinement_radius=self.center_refinement_radius,
                width=per_band[band].width,
                height=per_band[band].height,
            )
            self.detect_rows_by_token[ref.token] = rows
            for method in ("ap2", "kron"):
                self.snr_rows_by_token_method.pop((ref.token, method), None)
            self.detect_png_by_token[ref.token] = _overlay_png_bytes(
                raw_images[band_idx],
                rows,
                draw_centers=self.shape_overlay_centers,
            )
            self.input_shape_png_by_token[ref.token] = _input_shape_overlay_png_bytes(
                scaled[band_idx],
                rows,
                scaling=self.scaling_mode,
                clip_threshold=self.clip_threshold,
                draw_centers=self.shape_overlay_centers,
            )
            total_images += 1
            total_detections += len(rows)
        return {"n_images": total_images, "n_detections": total_detections}

    @staticmethod
    def _select_sample_outputs(outputs: dict[str, torch.Tensor], sample_idx: int) -> dict[str, torch.Tensor]:
        selected = {}
        for key, value in outputs.items():
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] > sample_idx:
                selected[key] = value[sample_idx : sample_idx + 1]
            else:
                selected[key] = value
        return selected

    def _detect_tile_slots(
        self,
        slots: list[tuple[str, int]],
        refs_by_key: dict[tuple[str, int], dict[str, FrameRef]] | None = None,
    ) -> dict[str, int]:
        refs_by_key = refs_by_key or self._refs_by_tile_slot()
        pending: list[tuple[str, int, dict[str, FrameRef]]] = []
        for tile_id, frame_slot in slots:
            per_band = refs_by_key.get((tile_id, frame_slot), {})
            bands = [band for band in self._bands_for_patch(self.patch) if band in per_band]
            if not bands:
                continue
            if all(per_band[band].token in self.detect_rows_by_token for band in bands):
                continue
            pending.append((tile_id, frame_slot, {band: per_band[band] for band in bands}))
        if pending:
            device = torch.device(self.device_name)
            grouped: dict[tuple[str, ...], list[tuple[str, int, dict[str, FrameRef]]]] = {}
            for item in pending:
                grouped.setdefault(tuple(item[2]), []).append(item)
            for bands_key, group_items in grouped.items():
                bands = list(bands_key)
                try:
                    model, _cfg = self._load_model(bands)
                except Exception as exc:
                    message = f"detection skipped for bands {bands}: {exc}"
                    self.warnings.append(message)
                    print(f"WARNING: {message}", flush=True)
                    continue
                for start in range(0, len(group_items), self.detect_batch_size):
                    chunk = group_items[start : start + self.detect_batch_size]
                    raw_by_sample: list[list[np.ndarray]] = []
                    scaled_by_sample = []
                    for _tile_id, _frame_slot, per_band in chunk:
                        raw_images = [self._detection_image_for_ref(per_band[band]) for band in bands]
                        raw_by_sample.append(raw_images)
                        scaled_by_sample.append(
                            [
                                make_training_rgb(
                                    image,
                                    mode=self.scaling_mode,
                                    clip_threshold=self.clip_threshold,
                                    log_a=self.log_a,
                                    log_high_percentile=self.log_high_percentile,
                                    lupton_stretch=self.lupton_stretch,
                                    lupton_q=self.lupton_q,
                                    anscombe_clip=self.anscombe_clip,
                                    anscombe_scale=self.anscombe_scale,
                                )
                                for image in raw_images
                            ]
                        )
                    tensor = torch.from_numpy(np.stack(scaled_by_sample, axis=0).astype(np.float32, copy=False))
                    outputs = infer_cellect(model=model, image_tensor=tensor, device=device, amp=self.amp)
                    for sample_idx, (_tile_id, _frame_slot, per_band) in enumerate(chunk):
                        sample_outputs = self._select_sample_outputs(outputs, sample_idx)
                        for band_idx, band in enumerate(bands):
                            ref = per_band[band]
                            rows = detection_rows(
                                select_band_outputs(sample_outputs, band_idx),
                                threshold=self.confidence_threshold,
                                nms_radius=self.nms_radius,
                                confidence_score=self.confidence_score,
                                center_refinement=self.center_refinement,
                                center_refinement_radius=self.center_refinement_radius,
                                width=ref.width,
                                height=ref.height,
                            )
                            self.detect_rows_by_token[ref.token] = rows
                            for method in ("ap2", "kron"):
                                self.snr_rows_by_token_method.pop((ref.token, method), None)
                            self.detect_png_by_token[ref.token] = _overlay_png_bytes(
                                raw_by_sample[sample_idx][band_idx],
                                rows,
                                draw_centers=self.shape_overlay_centers,
                            )
                            self.input_shape_png_by_token[ref.token] = _input_shape_overlay_png_bytes(
                                scaled_by_sample[sample_idx][band_idx],
                                rows,
                                scaling=self.scaling_mode,
                                clip_threshold=self.clip_threshold,
                                draw_centers=self.shape_overlay_centers,
                            )
        total_images = 0
        total_detections = 0
        for tile_id, frame_slot in slots:
            per_band = refs_by_key.get((tile_id, frame_slot), {})
            bands = [band for band in self._bands_for_patch(self.patch) if band in per_band]
            if not bands:
                continue
            total_images += len(bands)
            total_detections += sum(len(self.detect_rows_by_token.get(per_band[band].token, ())) for band in bands)
        return {"n_images": total_images, "n_detections": total_detections}

    def detect_page(self, page_index: int) -> dict[str, Any]:
        refs_by_key = self._refs_by_tile_slot()
        return self._detect_tile_slots(self._page_tile_slots(page_index), refs_by_key)

    def write_selection_csv(self, out_dir: Path | None = None) -> Path:
        refs = self.selected_refs()
        if not refs:
            raise RuntimeError("no selected candidates")
        target_dir = (out_dir or self.session_dir).expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / "selection.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            rows = [ref.to_dict() for ref in refs]
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def export_selected(self, write_png: bool = True) -> dict[str, Any]:
        refs = self.selected_refs()
        if not refs:
            raise RuntimeError("no selected candidates")
        self.export_dir.mkdir(parents=True, exist_ok=True)
        selection_csv = self.write_selection_csv(self.export_dir)
        refs_by_key = self._refs_by_tile_slot()
        selected_slots = sorted({(ref.tile_id, ref.frame_slot) for ref in refs})
        self._detect_tile_slots(selected_slots, refs_by_key)
        manifest = []
        for ref in refs:
            image = self.access.read_frame(ref)
            detection_image = self._detection_image_for_ref(ref)
            out_dir = self.export_dir / ref.tract / ref.patch / ref.band / ref.tile_id
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = ref.candidate_id
            npz_path = out_dir / f"{stem}.npz"
            np.savez_compressed(npz_path, image=image)
            png_path = out_dir / f"{stem}.png"
            detect_png_path = out_dir / f"{stem}_detect_overlay.png"
            input_shape_png_path = out_dir / f"{stem}_input_shape_overlay.png"
            ap2_snr_png_path = out_dir / f"{stem}_ap2_snr_overlay.png"
            kron_snr_png_path = out_dir / f"{stem}_kron_snr_overlay.png"
            detect_csv_path = out_dir / f"{stem}_detections.csv"
            if write_png:
                Image.fromarray(display_gray(image), mode="L").convert("RGB").save(png_path)
            detect_rows = self.detect_rows_by_token.get(ref.token, [])
            fieldnames = sorted({key for item in detect_rows for key in item}) if detect_rows else [
                "x",
                "y",
                "score",
                "major",
                "minor",
                "theta",
            ]
            with detect_csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(detect_rows)
            if write_png:
                overlay = _overlay_uint8(detection_image, detect_rows, draw_centers=self.shape_overlay_centers)
                _save_titled_png(
                    detect_png_path,
                    overlay,
                    f"{ref.candidate_id} detections={len(detect_rows)}",
                    min_image_size=512,
                )
                scaled = self._scaled_input_for_ref(ref)
                input_overlay = _input_shape_overlay_uint8(
                    scaled,
                    detect_rows,
                    scaling=self.scaling_mode,
                    clip_threshold=self.clip_threshold,
                    draw_centers=self.shape_overlay_centers,
                )
                _save_titled_png(
                    input_shape_png_path,
                    input_overlay,
                    f"{ref.candidate_id} input shape detections={len(detect_rows)}",
                    min_image_size=512,
                )
                for method, snr_png_path, label in (
                    ("ap2", ap2_snr_png_path, "AP2"),
                    ("kron", kron_snr_png_path, "Kron"),
                ):
                    snr_rows = self._snr_rows_for_token(ref.token, method)
                    _save_snr_overlay_png(
                        snr_png_path,
                        detection_image,
                        snr_rows,
                        title=f"{ref.candidate_id} {label} SNR, red<5 green>5 gray=bad",
                        threshold=5.0,
                    )
            row = {
                **ref.to_dict(),
                "npz_path": str(npz_path),
                "png_path": str(png_path) if write_png else "",
                "detect_png_path": str(detect_png_path) if write_png else "",
                "input_shape_png_path": str(input_shape_png_path) if write_png else "",
                "ap2_snr_png_path": str(ap2_snr_png_path) if write_png else "",
                "kron_snr_png_path": str(kron_snr_png_path) if write_png else "",
                "detect_csv_path": str(detect_csv_path),
                "export_subdir": str(out_dir),
                "n_detections": len(detect_rows),
            }
            (out_dir / f"{stem}.json").write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            manifest.append(row)
        with (self.export_dir / "export_manifest.jsonl").open("w", encoding="utf-8") as handle:
            for row in manifest:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return {
            "n_exported": len(manifest),
            "out_dir": str(self.export_dir),
            "selection_csv": str(selection_csv),
            "n_detections": sum(int(row["n_detections"]) for row in manifest),
        }
