#!/usr/bin/env python3
"""Clean implementation for FITS AMG runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

CELLECT_ROOT = Path(__file__).resolve().parents[2]
if str(CELLECT_ROOT) not in sys.path:
    sys.path.insert(0, str(CELLECT_ROOT))

try:
    from astropy.io import fits
    from astropy.stats import sigma_clipped_stats
    from astropy.visualization import ZScaleInterval
except Exception as exc:  # pragma: no cover
    raise RuntimeError("astropy is required. Install with: pip install astropy") from exc

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Pillow is required. Install with: pip install pillow") from exc

from sam_backbone import SamAutomaticMaskGenerator, sam_model_registry


FITS_EXTS = (".fits.gz", ".fits", ".fit", ".fts")
RGB_WEIGHTS = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)


@dataclass(frozen=True)
class ChannelStats:
    mean: float
    center: float
    sigma: float
    raw_mean: float
    raw_median: float
    raw_sigma: float
    clipped_count: int


@dataclass(frozen=True)
class AstroInput:
    sam_input: np.ndarray
    save_rgb: np.ndarray
    overlay_rgb: np.ndarray
    crop_y0: int = 0
    crop_x0: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM automatic mask generation on FITS images."
    )
    parser.add_argument("--input", type=str, required=True, nargs="+")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model-type", type=str, default="vit_h", choices=["default", "vit_h", "vit_l", "vit_b"])
    parser.add_argument("--checkpoint", type=str, default="/home/chenzunhao/sam_vit_h_4b8939.pth")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hdu", type=int, default=0)
    parser.add_argument("--low-percentile", type=float, default=0.1)
    parser.add_argument("--high-percentile", type=float, default=99.5)
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument("--overlay-style", choices=["boundary", "fill"], default="fill")
    parser.add_argument("--boundary-color", type=int, nargs=3, default=[255, 255, 255])
    parser.add_argument(
        "--scaling-mode",
        type=str,
        default="robust",
        choices=["robust", "lupton", "lupton_rgb", "linear_rgb", "astro_rgb"],
    )
    parser.add_argument(
        "--astro-rgb-mode",
        type=str,
        default="none",
        choices=["none", "astro_rgb", "astro_rgb1", "astro_rgb2", "both"],
        help="Use both to emit astro_rgb and astro_rgb_none outputs for --scaling-mode astro_rgb.",
    )
    parser.add_argument(
        "--astro-stats-mode",
        type=str,
        default="sigmaclip",
        choices=["bgd", "sigmaclip"],
        help="bgd uses median/MAD; sigmaclip uses sigma-clipped median/std.",
    )
    parser.add_argument(
        "--astro-stats-input",
        type=str,
        nargs=3,
        default=None,
        help=(
            "Optional full-frame FITS triplet used only to estimate astro_rgb stats. "
            "Use this when --input contains small ROI crops but normalization should come from the original image."
        ),
    )
    parser.add_argument("--astro-rgb-low-sigma", type=float, default=None)
    parser.add_argument("--astro-crop-size", type=int, default=1024)
    parser.add_argument(
        "--astro-preprocess-in-model",
        action="store_true",
        help="Pass raw FITS float crops to SAM and run astro normalization inside Sam.preprocess().",
    )
    parser.add_argument("--astro-preprocess-clip-sigma", type=float, default=3.0)
    parser.add_argument(
        "--astro-preprocess-sigma-iters",
        type=int,
        default=-1,
        help="Astropy sigma_clip maxiters inside Sam.preprocess(); -1 means iterate to convergence.",
    )
    parser.add_argument("--astro-preprocess-z-clip", type=float, nargs=2, default=None)
    parser.add_argument(
        "--astro-hardcode-mean",
        type=float,
        nargs=3,
        default=None,
        metavar=("R", "G", "B"),
        help="Debug override: use these per-channel means immediately before network normalization.",
    )
    parser.add_argument(
        "--astro-hardcode-std",
        type=float,
        nargs=3,
        default=None,
        metavar=("R", "G", "B"),
        help="Debug override: use these per-channel stds immediately before network normalization.",
    )
    parser.add_argument(
        "--astro-hardcode-clip-hi",
        type=float,
        nargs=3,
        default=None,
        metavar=("R", "G", "B"),
        help="Debug override: cap each channel at these FITS values before network normalization.",
    )
    parser.add_argument(
        "--astro-hardcode-z-clip",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
        help="Debug override: clip final z-score before feeding/mapping to SAM.",
    )
    parser.add_argument("--no-save-fits", action="store_true")
    parser.add_argument("--save-json", action="store_true", help="Save full mask JSON. Disabled by default because binary masks are large.")

    parser.add_argument("--points-per-side", type=int, default=64)
    parser.add_argument("--points-per-batch", type=int, default=128)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.88)
    parser.add_argument("--stability-score-thresh", type=float, default=0.95)
    parser.add_argument("--box-nms-thresh", type=float, default=0.7)
    parser.add_argument("--crop-n-layers", type=int, default=0)
    parser.add_argument("--crop-nms-thresh", type=float, default=0.7)
    parser.add_argument("--crop-overlap-ratio", type=float, default=512 / 1500)
    parser.add_argument("--crop-n-points-downscale-factor", type=int, default=1)
    parser.add_argument("--min-mask-region-area", type=int, default=0)
    parser.add_argument("--max-mask-area-ratio", type=float, default=0.15)
    return parser.parse_args()


def collect_fits_paths(path_str: str) -> List[Path]:
    path = Path(path_str)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input path not found: {path}")
    files = [p for p in sorted(path.iterdir()) if p.is_file() and p.name.lower().endswith(FITS_EXTS)]
    if not files:
        raise FileNotFoundError(f"No FITS files found in directory: {path}")
    return files


def strip_fits_suffix(path: Path) -> str:
    stem = path.name
    for suffix in FITS_EXTS:
        if stem.lower().endswith(suffix):
            return stem[: -len(suffix)]
    return path.stem


def read_fits_2d(path: Path, hdu: int) -> Tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=True) as hdul:
        if hdu >= len(hdul):
            raise IndexError(f"HDU index {hdu} out of range for file: {path}")
        data = hdul[hdu].data
        if data is None:
            raise ValueError(f"No image data in HDU {hdu}: {path}")
        if data.ndim != 2:
            raise ValueError(f"Only 2D FITS supported. Got shape {data.shape} in {path}")
        return np.asarray(data, dtype=np.float32), hdul[hdu].header.copy()


def finite_values(image: np.ndarray) -> np.ndarray:
    vals = image[np.isfinite(image)]
    if vals.size == 0:
        raise ValueError("Image has no finite values.")
    return vals.astype(np.float64, copy=False)


def robust_to_uint8(image: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    vals = finite_values(image)
    lo = float(np.percentile(vals, low_pct))
    hi = float(np.percentile(vals, high_pct))
    if not np.isfinite(hi - lo) or hi <= lo:
        hi = lo + 1e-6
    y = (np.clip(image, lo, hi) - lo) / (hi - lo)
    y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
    return np.round(np.clip(y, 0.0, 1.0) * 255.0).astype(np.uint8)


def zscale_to_uint8(image: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    """Map a FITS-like image to uint8 using astropy's zscale interval."""

    image = np.asarray(image, dtype=np.float32)
    vals = finite_values(image)
    try:
        lo, hi = ZScaleInterval().get_limits(vals)
    except Exception:
        lo = float(np.percentile(vals, low_pct))
        hi = float(np.percentile(vals, high_pct))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(image, dtype=np.uint8)
    y = (np.clip(image, lo, hi) - lo) / (hi - lo)
    y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
    return np.round(np.clip(y, 0.0, 1.0) * 255.0).astype(np.uint8)


def lupton_to_uint8(image: np.ndarray, low_pct: float, high_pct: float, q: float = 10.0) -> np.ndarray:
    vals = finite_values(image)
    minimum = float(np.percentile(vals, low_pct))
    upper = float(np.percentile(vals, high_pct))
    stretch = upper - minimum
    if not np.isfinite(stretch) or stretch <= 0:
        stretch = float(np.nanmax(vals) - minimum)
    if not np.isfinite(stretch) or stretch <= 0:
        stretch = 1e-6
    x = np.clip((image.astype(np.float64) - minimum) / stretch, 0.0, None)
    y = np.arcsinh(q * x) / np.arcsinh(q) if q > 0 else np.clip(x, 0.0, 1.0)
    y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
    return np.round(np.clip(y, 0.0, 1.0) * 255.0).astype(np.uint8)


def estimate_channel_stats(image: np.ndarray, mode: str) -> ChannelStats:
    vals = finite_values(image)
    raw_mean = float(np.mean(vals))
    raw_median = float(np.median(vals))
    raw_sigma = float(np.std(vals))
    if not np.isfinite(raw_sigma) or raw_sigma <= 0:
        raw_sigma = 1.0
    bright_limit = raw_median + 3.0 * raw_sigma
    bright_clipped = np.minimum(vals, bright_limit)
    clipped_count = int(np.count_nonzero(vals > bright_limit))

    if mode == "bgd":
        center = float(np.median(bright_clipped))
        mad = float(np.median(np.abs(bright_clipped - center)))
        sigma = 1.4826 * mad
        mean = float(np.mean(bright_clipped))
    elif mode == "sigmaclip":
        mean, median, std = sigma_clipped_stats(bright_clipped, sigma=3.0, maxiters=None)
        center = float(median)
        sigma = float(std)
        mean = float(mean)
    else:
        raise ValueError(f"Unknown astro stats mode: {mode}")
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.std(bright_clipped))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    return ChannelStats(
        mean=mean,
        center=center,
        sigma=sigma,
        raw_mean=raw_mean,
        raw_median=raw_median,
        raw_sigma=raw_sigma,
        clipped_count=clipped_count,
    )


def estimate_rgb_stats(
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stats = [estimate_channel_stats(x, mode) for x in (r, g, b)]
    for name, stat in zip(("R", "G", "B"), stats):
        print(
            f"[STATS {mode} {name}] raw_mean={stat.raw_mean:.6g} raw_median={stat.raw_median:.6g} "
            f"raw_sigma={stat.raw_sigma:.6g} "
            f"clip_hi={stat.raw_median + 3.0 * stat.raw_sigma:.6g} clipped_pixels={stat.clipped_count} "
            f"mean={stat.mean:.6g} center={stat.center:.6g} sigma={stat.sigma:.6g} "
        )
    center = np.array([s.mean for s in stats], dtype=np.float32).reshape(1, 1, 3)
    sigma = np.array([s.sigma for s in stats], dtype=np.float32).reshape(1, 1, 3)
    raw_median = np.array([s.raw_median for s in stats], dtype=np.float32).reshape(1, 1, 3)
    raw_sigma = np.array([s.raw_sigma for s in stats], dtype=np.float32).reshape(1, 1, 3)
    return center, sigma, raw_median, raw_sigma


def normalize_astro_crop(
    rgb_crop: np.ndarray,
    mean: np.ndarray,
    sigma: np.ndarray,
    raw_median: np.ndarray,
    raw_sigma: np.ndarray,
) -> np.ndarray:
    safe = np.where(np.isfinite(rgb_crop), rgb_crop, mean)
    clipped = np.minimum(safe, raw_median + 3.0 * raw_sigma)
    return (clipped - mean) / sigma


def subtract_background(image: np.ndarray, mode: str) -> np.ndarray:
    return image - estimate_channel_stats(image, mode).center


def make_lupton_rgb(r: np.ndarray, g: np.ndarray, b: np.ndarray, low_pct: float, high_pct: float, q: float = 10.0) -> np.ndarray:
    r0 = subtract_background(r, "bgd")
    g0 = subtract_background(g, "bgd")
    b0 = subtract_background(b, "bgd")
    intensity = (r0 + g0 + b0) / 3.0
    vals = finite_values(intensity)
    minimum = float(np.percentile(vals, low_pct))
    upper = float(np.percentile(vals, high_pct))
    stretch = upper - minimum
    if not np.isfinite(stretch) or stretch <= 0:
        stretch = 1e-6
    scaled_i = np.clip((intensity - minimum) / stretch, 0.0, None)
    f = np.arcsinh(q * scaled_i) / np.arcsinh(q) if q > 0 else np.clip(scaled_i, 0.0, 1.0)
    i_safe = np.where(intensity <= 0, 1e-6, intensity)
    rgb = np.stack([r0 * f / i_safe, g0 * f / i_safe, b0 * f / i_safe], axis=-1)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def make_linear_rgb(r: np.ndarray, g: np.ndarray, b: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    channels = [robust_to_uint8(x, low_pct, high_pct).astype(np.float32) / 255.0 for x in (r, g, b)]
    return np.stack(channels, axis=-1)


def make_gray_quicklook_from_rgb(r: np.ndarray, g: np.ndarray, b: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    gray_float = (
        np.asarray(r, dtype=np.float32) * float(RGB_WEIGHTS[0])
        + np.asarray(g, dtype=np.float32) * float(RGB_WEIGHTS[1])
        + np.asarray(b, dtype=np.float32) * float(RGB_WEIGHTS[2])
    )
    gray = zscale_to_uint8(gray_float, low_pct, high_pct)
    return np.repeat(gray[..., None], 3, axis=2)


def crop_rgb(r: np.ndarray, g: np.ndarray, b: np.ndarray, crop_size: int) -> Tuple[np.ndarray, int, int]:
    h, w = r.shape
    ch = min(h, crop_size)
    cw = min(w, crop_size)
    y0 = 0
    x0 = w - cw
    return np.stack([r[y0 : y0 + ch, x0 : x0 + cw], g[y0 : y0 + ch, x0 : x0 + cw], b[y0 : y0 + ch, x0 : x0 + cw]], axis=-1).astype(np.float32), y0, x0


def astro_sigmas(mode: str, low_sigma_override: float | None) -> Tuple[float, float]:
    if mode == "astro_rgb":
        return 3.0, 3.0
    if mode == "astro_rgb1":
        return 1.0, 3.0
    if mode == "astro_rgb2":
        return 5.0 if low_sigma_override is None else float(low_sigma_override), 3.0
    raise ValueError(f"Unknown astro_rgb_mode for clipped mapping: {mode}")


def build_astro_input(
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    mode: str,
    stats_mode: str,
    low_sigma_override: float | None,
    crop_size: int,
    low_pct: float,
    high_pct: float,
    stats_arrays: Tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    hardcode_mean: Sequence[float] | None = None,
    hardcode_std: Sequence[float] | None = None,
    hardcode_clip_hi: Sequence[float] | None = None,
    hardcode_z_clip: Sequence[float] | None = None,
    preprocess_in_model: bool = False,
) -> AstroInput:
    rgb_crop, y0, x0 = crop_rgb(r, g, b, crop_size)
    if preprocess_in_model:
        save_rgb = np.stack(
            [robust_to_uint8(rgb_crop[..., c], low_pct, high_pct) for c in range(3)],
            axis=-1,
        )
        overlay_rgb = make_gray_quicklook_from_rgb(r, g, b, low_pct, high_pct)
        print("[STATS model-preprocess] passing raw FITS float crop to Sam.preprocess()")
        return AstroInput(
            sam_input=rgb_crop.astype(np.float32),
            save_rgb=save_rgb,
            overlay_rgb=overlay_rgb,
            crop_y0=y0,
            crop_x0=x0,
        )
    if hardcode_mean is not None or hardcode_std is not None or hardcode_clip_hi is not None:
        if hardcode_mean is None or hardcode_std is None or hardcode_clip_hi is None:
            raise ValueError("--astro-hardcode-mean/std/clip-hi must be provided together.")
        center = np.asarray(hardcode_mean, dtype=np.float32).reshape(1, 1, 3)
        sigma = np.asarray(hardcode_std, dtype=np.float32).reshape(1, 1, 3)
        clip_hi = np.asarray(hardcode_clip_hi, dtype=np.float32).reshape(1, 1, 3)
        sigma = np.where((np.isfinite(sigma)) & (sigma > 0), sigma, 1.0)
        safe = np.where(np.isfinite(rgb_crop), rgb_crop, center)
        z = (np.minimum(safe, clip_hi) - center) / sigma
        print(
            "[STATS hardcode] "
            f"mean={center.reshape(3).tolist()} std={sigma.reshape(3).tolist()} "
            f"clip_hi={clip_hi.reshape(3).tolist()}"
        )
    else:
        stats_r, stats_g, stats_b = stats_arrays if stats_arrays is not None else (r, g, b)
        center, sigma, raw_median, raw_sigma = estimate_rgb_stats(stats_r, stats_g, stats_b, stats_mode)
        z = normalize_astro_crop(rgb_crop, center, sigma, raw_median, raw_sigma)
    if hardcode_z_clip is not None:
        z_low, z_high = float(hardcode_z_clip[0]), float(hardcode_z_clip[1])
        if not (np.isfinite(z_low) and np.isfinite(z_high) and z_low < z_high):
            raise ValueError("--astro-hardcode-z-clip requires finite LOW < HIGH.")
        z = np.clip(z, z_low, z_high)
        print(f"[STATS hardcode] final z clipped to [{z_low:g}, {z_high:g}]")
    if mode == "none":
        sam_input = z.astype(np.float32)
        save_rgb = z_to_uint8(z)
    else:
        low_sigma, high_sigma = astro_sigmas(mode, low_sigma_override)
        mapped = (np.clip(z, -low_sigma, high_sigma) + low_sigma) / (low_sigma + high_sigma)
        save_rgb = np.round(np.clip(mapped, 0.0, 1.0) * 255.0).astype(np.uint8)
        sam_input = save_rgb

    overlay_rgb = make_gray_quicklook_from_rgb(r, g, b, low_pct, high_pct)
    return AstroInput(sam_input=sam_input, save_rgb=save_rgb, overlay_rgb=overlay_rgb, crop_y0=y0, crop_x0=x0)


def z_to_uint8(z: np.ndarray, low: float = -3.0, high: float = 3.0) -> np.ndarray:
    y = (np.clip(z, low, high) - low) / (high - low)
    return np.round(y * 255.0).astype(np.uint8)


def label_boundaries(label_map: np.ndarray) -> np.ndarray:
    up = np.zeros_like(label_map)
    down = np.zeros_like(label_map)
    left = np.zeros_like(label_map)
    right = np.zeros_like(label_map)
    up[1:, :] = label_map[:-1, :]
    down[:-1, :] = label_map[1:, :]
    left[:, 1:] = label_map[:, :-1]
    right[:, :-1] = label_map[:, 1:]
    return ((label_map != up) | (label_map != down) | (label_map != left) | (label_map != right)) & (label_map > 0)


def colorize_labels(label_map: np.ndarray, seed: int = 1234) -> np.ndarray:
    lut = np.zeros((int(label_map.max()) + 1, 3), dtype=np.uint8)
    if lut.shape[0] > 1:
        lut[1:] = np.random.default_rng(seed).integers(0, 256, size=(lut.shape[0] - 1, 3), dtype=np.uint8)
    return lut[label_map]


def blend_overlay(
    base_u8: np.ndarray,
    label_map: np.ndarray,
    alpha: float,
    boundary_rgb: Tuple[int, int, int],
    style: str,
) -> np.ndarray:
    if base_u8.ndim == 2:
        base_rgb = np.repeat(base_u8[..., None], 3, axis=2).astype(np.float32)
    elif base_u8.ndim == 3 and base_u8.shape[2] == 3:
        base_rgb = base_u8.astype(np.float32)
    else:
        raise ValueError(f"Overlay base must be HxW or HxWx3, got shape {base_u8.shape}")

    if style == "fill":
        mask = label_map > 0
        if np.any(mask):
            colors = colorize_labels(label_map).astype(np.float32)
            base_rgb[mask] = (1.0 - alpha) * base_rgb[mask] + alpha * colors[mask]
    elif style != "boundary":
        raise ValueError(f"Unknown overlay style: {style}")

    boundary = label_boundaries(label_map)
    if np.any(boundary):
        base_rgb[boundary] = np.array(boundary_rgb, dtype=np.float32)
    return np.clip(base_rgb, 0, 255).astype(np.uint8)


def bbox_xywh(seg: np.ndarray) -> List[int]:
    ys, xs = np.where(seg)
    if ys.size == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


def refresh_mask_area(mask: Dict, h: int, w: int) -> int:
    seg = mask.get("segmentation")
    if isinstance(seg, np.ndarray):
        if seg.shape != (h, w):
            return int(mask.get("area", 0))
        area = int(np.count_nonzero(seg))
        mask["area"] = area
        mask["bbox"] = bbox_xywh(seg.astype(bool))
        return area
    return int(mask.get("area", 0))


def filter_small_masks(masks: List[Dict], h: int, w: int, min_mask_region_area: int) -> Tuple[List[Dict], int]:
    if min_mask_region_area <= 0:
        return masks, 0
    kept = []
    removed = 0
    for mask in masks:
        area = refresh_mask_area(mask, h, w)
        if area < min_mask_region_area:
            removed += 1
        else:
            kept.append(mask)
    return kept, removed


def filter_large_masks(masks: List[Dict], h: int, w: int, max_mask_area_ratio: float) -> Tuple[List[Dict], int]:
    if max_mask_area_ratio >= 1.0:
        return masks, 0
    if max_mask_area_ratio <= 0.0:
        raise ValueError("--max-mask-area-ratio must be in (0, 1].")
    area_limit = h * w * float(max_mask_area_ratio)
    kept = []
    removed = 0
    for mask in masks:
        area = refresh_mask_area(mask, h, w)
        if area > area_limit:
            removed += 1
        else:
            kept.append(mask)
    return kept, removed


def make_label_map(masks: List[Dict], h: int, w: int) -> np.ndarray:
    label_map = np.zeros((h, w), dtype=np.int32)
    obj_id = 1
    for mask in sorted(masks, key=lambda m: float(m.get("area", 0)), reverse=True):
        seg = mask.get("segmentation")
        if isinstance(seg, np.ndarray) and seg.shape == (h, w):
            assign = seg.astype(bool) & (label_map == 0)
            if np.any(assign):
                label_map[assign] = obj_id
                obj_id += 1
    return label_map


def make_filtered_label_map(
    masks: List[Dict], h: int, w: int, min_mask_region_area: int
) -> Tuple[np.ndarray, List[Dict], int]:
    if min_mask_region_area <= 0:
        return make_label_map(masks, h, w), masks, 0

    label_map = np.zeros((h, w), dtype=np.int32)
    kept = []
    removed = 0
    obj_id = 1
    for mask in sorted(masks, key=lambda m: float(m.get("area", 0)), reverse=True):
        seg = mask.get("segmentation")
        if not isinstance(seg, np.ndarray) or seg.shape != (h, w):
            continue
        assign = seg.astype(bool) & (label_map == 0)
        area = int(np.count_nonzero(assign))
        if area < min_mask_region_area:
            removed += 1
            continue

        filtered_mask = dict(mask)
        filtered_mask["segmentation"] = assign
        filtered_mask["area"] = area
        filtered_mask["bbox"] = bbox_xywh(assign)
        kept.append(filtered_mask)
        label_map[assign] = obj_id
        obj_id += 1
    return label_map, kept, removed


def expand_crop_masks(masks: List[Dict], full_shape: Tuple[int, int], crop_shape: Tuple[int, int], y0: int, x0: int) -> None:
    full_h, full_w = full_shape
    crop_h, crop_w = crop_shape
    for mask in masks:
        seg = mask.get("segmentation")
        if not isinstance(seg, np.ndarray) or seg.shape == full_shape:
            continue
        full_seg = np.zeros((full_h, full_w), dtype=seg.dtype)
        full_seg[y0 : y0 + crop_h, x0 : x0 + crop_w] = seg
        mask["segmentation"] = full_seg


def masks_to_csv(masks: Sequence[Dict], csv_path: Path) -> None:
    header = [
        "id", "area", "bbox_x0", "bbox_y0", "bbox_w", "bbox_h", "point_input_x", "point_input_y",
        "predicted_iou", "stability_score", "crop_box_x0", "crop_box_y0", "crop_box_w", "crop_box_h",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, mask in enumerate(masks):
            point = mask.get("point_coords", [[0, 0]])[0]
            writer.writerow([
                i + 1,
                mask.get("area", 0),
                *mask.get("bbox", [0, 0, 0, 0]),
                *point,
                mask.get("predicted_iou", 0),
                mask.get("stability_score", 0),
                *mask.get("crop_box", [0, 0, 0, 0]),
            ])


def serializable_masks(masks: Sequence[Dict]) -> List[Dict]:
    out = []
    for mask in masks:
        item = {}
        for key, value in mask.items():
            item[key] = value.tolist() if isinstance(value, np.ndarray) else value
        out.append(item)
    return out


def save_outputs(
    stem: str,
    out_dir: Path,
    original: np.ndarray,
    header: fits.Header,
    sam_input_preview: np.ndarray,
    label_map: np.ndarray,
    overlay_u8: np.ndarray,
    masks: List[Dict],
    tag: str,
    save_fits: bool,
    save_json: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_fits:
        fits.writeto(out_dir / f"{stem}_{tag}_sam_labelmap.fits", label_map.astype(np.int32), header=header, overwrite=True)
        hdu0 = fits.PrimaryHDU(data=original.astype(np.float32), header=header)
        hdu0.header["HIERARCH SAM.NMASK"] = int(len(masks))
        hdu1 = fits.ImageHDU(data=label_map.astype(np.int32), name="LABELMAP")
        hdu2 = fits.ImageHDU(data=np.transpose(overlay_u8, (2, 0, 1)), name="OVERLAYRGB")
        hdu3 = fits.ImageHDU(data=sam_input_preview.astype(np.uint8), name="SAMINPUT8")
        fits.HDUList([hdu0, hdu1, hdu2, hdu3]).writeto(out_dir / f"{stem}_{tag}_sam_bundle.fits", overwrite=True)

    input_png = out_dir / f"{stem}_{tag}_input.png"
    Image.fromarray(np.flipud(sam_input_preview), mode="RGB").save(input_png)
    Image.fromarray(np.flipud(overlay_u8), mode="RGB").save(out_dir / f"{stem}_{tag}_overlay.png")
    Image.fromarray(np.flipud(overlay_u8), mode="RGB").save(out_dir / f"{stem}_{tag}_overlay.tif", compression="tiff_lzw")
    masks_to_csv(masks, out_dir / f"{stem}_{tag}_metadata.csv")
    if save_json:
        with (out_dir / f"{stem}_{tag}_masks.json").open("w") as f:
            json.dump(serializable_masks(masks), f, indent=2)


def build_single_input(image: np.ndarray, scaling_mode: str, low_pct: float, high_pct: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scaling_mode == "lupton":
        gray = lupton_to_uint8(image, low_pct, high_pct)
    else:
        gray = robust_to_uint8(image, low_pct, high_pct)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    return rgb, rgb, gray


def build_triplet_input(
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    scaling_mode: str,
    low_pct: float,
    high_pct: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scaling_mode == "lupton_rgb":
        rgb_float = make_lupton_rgb(r, g, b, low_pct, high_pct)
    else:
        rgb_float = make_linear_rgb(r, g, b, low_pct, high_pct)
    rgb = np.round(np.clip(rgb_float, 0.0, 1.0) * 255.0).astype(np.uint8)
    gray = np.round(np.dot(rgb_float, RGB_WEIGHTS) * 255.0).astype(np.uint8)
    return rgb, rgb, gray


def run_generator(generator: SamAutomaticMaskGenerator, sam_input: np.ndarray) -> List[Dict]:
    return generator.generate(sam_input)


def build_generator(args: argparse.Namespace, astro_rgb_mode: str) -> SamAutomaticMaskGenerator:
    sam = sam_model_registry[args.model_type](
        checkpoint=args.checkpoint,
        scaling_mode=args.scaling_mode,
        astro_rgb_mode=astro_rgb_mode,
        astro_rgb_low_sigma=args.astro_rgb_low_sigma,
        astro_preprocess_in_model=args.astro_preprocess_in_model,
        astro_preprocess_clip_sigma=args.astro_preprocess_clip_sigma,
        astro_preprocess_sigma_iters=args.astro_preprocess_sigma_iters,
        astro_preprocess_z_clip=args.astro_preprocess_z_clip,
    )
    sam = sam.to(device=args.device)
    kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
        "output_mode": "binary_mask",
    }
    return SamAutomaticMaskGenerator(sam, **kwargs)


def run_one_single(args: argparse.Namespace, generator: SamAutomaticMaskGenerator, path: Path, out_dir: Path) -> None:
    image, header = read_fits_2d(path, args.hdu)
    sam_input, preview_rgb, gray = build_single_input(image, args.scaling_mode, args.low_percentile, args.high_percentile)
    masks = run_generator(generator, sam_input)
    masks, removed_small = filter_small_masks(masks, image.shape[0], image.shape[1], args.min_mask_region_area)
    masks, removed_large = filter_large_masks(masks, image.shape[0], image.shape[1], args.max_mask_area_ratio)
    label_map, masks, removed_label_small = make_filtered_label_map(
        masks, image.shape[0], image.shape[1], args.min_mask_region_area
    )
    removed_small += removed_label_small
    overlay = blend_overlay(preview_rgb, label_map, float(args.overlay_alpha), tuple(args.boundary_color), args.overlay_style)
    tag = args.scaling_mode
    save_outputs(
        strip_fits_suffix(path),
        out_dir,
        image,
        header,
        preview_rgb,
        label_map,
        overlay,
        masks,
        tag,
        not args.no_save_fits,
        args.save_json,
    )
    print(
        f"[OK] {path.name} {tag}: {len(masks)} masks, "
        f"removed_small={removed_small}, removed_large={removed_large}"
    )


def run_one_triplet(
    args: argparse.Namespace,
    generator: SamAutomaticMaskGenerator,
    paths: Sequence[Path],
    out_dir: Path,
    astro_mode: str | None = None,
) -> None:
    arrays = []
    headers = []
    for path in paths:
        image, header = read_fits_2d(path, args.hdu)
        arrays.append(image)
        headers.append(header)
    r, g, b = arrays
    if r.shape != g.shape or r.shape != b.shape:
        raise ValueError("Input band images must have identical shapes")

    stem = "_".join(strip_fits_suffix(p) for p in paths)
    original_stack = np.stack([r, g, b], axis=0)
    if args.scaling_mode == "astro_rgb":
        assert astro_mode is not None
        stats_arrays = None
        stats_inputs = getattr(args, "astro_stats_input", None)
        if stats_inputs is not None:
            stats_paths = [Path(p) for p in stats_inputs]
            if len(stats_paths) != 3:
                raise ValueError("--astro-stats-input must contain exactly three FITS paths")
            stats_arrays = tuple(read_fits_2d(p, args.hdu)[0] for p in stats_paths)
            if len({arr.shape for arr in stats_arrays}) != 1:
                raise ValueError(f"Astro stats inputs must have identical shapes, got {[a.shape for a in stats_arrays]}")
        astro_input = build_astro_input(
            r,
            g,
            b,
            astro_mode,
            args.astro_stats_mode,
            args.astro_rgb_low_sigma,
            args.astro_crop_size,
            args.low_percentile,
            args.high_percentile,
            stats_arrays=stats_arrays,
            hardcode_mean=args.astro_hardcode_mean,
            hardcode_std=args.astro_hardcode_std,
            hardcode_clip_hi=args.astro_hardcode_clip_hi,
            hardcode_z_clip=args.astro_hardcode_z_clip,
            preprocess_in_model=args.astro_preprocess_in_model,
        )
        masks = run_generator(generator, astro_input.sam_input)
        expand_crop_masks(masks, r.shape, astro_input.sam_input.shape[:2], astro_input.crop_y0, astro_input.crop_x0)
        base_tag = "astro_rgb_none" if astro_mode == "none" else astro_mode
        tag = f"{base_tag}_{args.astro_stats_mode}"
        preview_rgb = astro_input.save_rgb
        overlay_base = astro_input.overlay_rgb
    else:
        sam_input, preview_rgb, gray = build_triplet_input(r, g, b, args.scaling_mode, args.low_percentile, args.high_percentile)
        masks = run_generator(generator, sam_input)
        tag = args.scaling_mode
        overlay_base = make_gray_quicklook_from_rgb(r, g, b, args.low_percentile, args.high_percentile)

    masks, removed_small = filter_small_masks(masks, r.shape[0], r.shape[1], args.min_mask_region_area)
    masks, removed_large = filter_large_masks(masks, r.shape[0], r.shape[1], args.max_mask_area_ratio)
    label_map, masks, removed_label_small = make_filtered_label_map(
        masks, r.shape[0], r.shape[1], args.min_mask_region_area
    )
    removed_small += removed_label_small
    overlay = blend_overlay(overlay_base, label_map, float(args.overlay_alpha), tuple(args.boundary_color), args.overlay_style)
    save_outputs(
        stem,
        out_dir,
        original_stack,
        headers[0],
        preview_rgb,
        label_map,
        overlay,
        masks,
        tag,
        not args.no_save_fits,
        args.save_json,
    )
    print(
        f"[OK] {stem} {tag}: {len(masks)} masks, "
        f"removed_small={removed_small}, removed_large={removed_large}"
    )


def grouped(paths: Sequence[Path], size: int) -> Iterable[Sequence[Path]]:
    if len(paths) % size != 0:
        raise ValueError(f"Expected input count to be a multiple of {size}, got {len(paths)}")
    for i in range(0, len(paths), size):
        yield paths[i : i + size]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output)
    input_paths: List[Path] = []
    for item in args.input:
        input_paths.extend(collect_fits_paths(item))

    if args.scaling_mode in {"lupton_rgb", "linear_rgb", "astro_rgb"}:
        triplets = list(grouped(input_paths, 3))
        if args.scaling_mode == "astro_rgb":
            modes = ["astro_rgb", "none"] if args.astro_rgb_mode == "both" else [args.astro_rgb_mode]
            for mode in modes:
                print(f"Loading SAM model for astro_rgb_mode={mode}")
                generator = build_generator(args, mode)
                for triplet in triplets:
                    run_one_triplet(args, generator, triplet, out_dir, astro_mode=mode)
        else:
            generator = build_generator(args, "none")
            for triplet in triplets:
                run_one_triplet(args, generator, triplet, out_dir)
    else:
        generator = build_generator(args, "none")
        for path in input_paths:
            run_one_single(args, generator, path, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
