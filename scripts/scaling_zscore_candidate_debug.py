#!/usr/bin/env python3
"""Compare old raw-first scaling with zscore-first scaling candidates.

The left column reproduces the old bright-region scaling path:
    raw FITS -> log/lupton/anscombe -> self standardize -> clip

The right column applies a candidate zscore-first path:
    raw FITS -> robust zscore -> log/lupton/anscombe -> self standardize -> clip
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cellect")

REPO_ROOT = Path("/home/czh23/CELLECT")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ZScaleInterval
from scipy import ndimage

from data_filtering.sam_input_scaling import anscombe_single, log_single, lupton_single, standardize_by_self


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits-root", type=Path, default=Path("/data/shared/Subaru/9813"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tract", default="9813")
    parser.add_argument("--patch", default="4,5")
    parser.add_argument("--bands", nargs="+", default=["HSC-G", "HSC-I", "NB1010"])
    parser.add_argument("--visualize-bands", nargs="+", default=["HSC-I", "NB1010"])
    parser.add_argument("--tiles", nargs="+", default=["4,6", "7,10", "9,1", "7,1"])
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-step", type=int, default=368)
    parser.add_argument("--log-a", type=float, default=1000.0)
    parser.add_argument("--log-high-percentile", type=float, default=99.5)
    parser.add_argument("--old-lupton-q", type=float, default=20.0)
    parser.add_argument("--old-lupton-stretch", type=float, default=0.5)
    parser.add_argument("--old-anscombe-scale", type=float, default=1000.0)
    parser.add_argument("--new-lupton-q", type=float, required=True)
    parser.add_argument("--new-lupton-stretch", type=float, required=True)
    parser.add_argument("--new-anscombe-scale", type=float, default=1000.0)
    parser.add_argument("--bright-threshold", type=float, default=5.0)
    parser.add_argument("--clip-threshold", type=float, default=5.0)
    parser.add_argument("--dilation", type=int, default=2)
    parser.add_argument("--zscore-sigma", type=float, default=3.0)
    parser.add_argument("--zscore-maxiters", type=int, default=5)
    return parser.parse_args()


def parse_tiles(values: list[str]) -> list[tuple[int, int]]:
    out = []
    for value in values:
        row, col = value.split(",", 1)
        out.append((int(row), int(col)))
    return out


def read_image(path: Path) -> np.ndarray:
    with fits.open(path, memmap=True) as hdul:
        return np.asarray(hdul[1].data, dtype=np.float32)


def robust_zscore_no_raw_clip(
    image: np.ndarray,
    *,
    sigma: float,
    maxiters: int,
) -> tuple[np.ndarray, dict[str, float]]:
    vals = np.asarray(image, dtype=np.float64)
    finite = np.isfinite(vals)
    finite_vals = vals[finite]
    mean, median, zsigma = sigma_clipped_stats(finite_vals, sigma=float(sigma), maxiters=int(maxiters))
    median = float(median) if np.isfinite(median) else float(np.median(finite_vals))
    zsigma = float(zsigma) if np.isfinite(zsigma) and zsigma > 0 else float(np.std(finite_vals))
    if not np.isfinite(zsigma) or zsigma <= 0:
        zsigma = 1.0
    safe = np.where(finite, vals, median)
    z = ((safe - median) / zsigma).astype(np.float32)
    raw_std = float(np.std(finite_vals))
    return z, {
        "raw_mean": float(np.mean(finite_vals)),
        "raw_median": float(np.median(finite_vals)),
        "raw_std": raw_std,
        "zscore_mean": float(mean) if np.isfinite(mean) else float(np.mean(finite_vals)),
        "zscore_median": median,
        "zscore_sigma": zsigma,
        "raw_std_over_zscore_sigma": float(raw_std / zsigma),
    }


def transform_then_normalize(
    input_image: np.ndarray,
    *,
    log_a: float,
    log_high_percentile: float,
    lupton_q: float,
    lupton_stretch: float,
    lupton_minimum: float,
    anscombe_scale: float,
    clip_threshold: float,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    finite = input_image[np.isfinite(input_image)]
    input_min = float(np.min(finite))
    maps = {
        "log": log_single(input_image, minimum=input_min, high_pct=log_high_percentile, a=log_a)[0],
        "lupton": lupton_single(
            input_image,
            minimum=lupton_minimum,
            stretch=lupton_stretch,
            q=lupton_q,
        )[0],
        "anscombe": anscombe_single(input_image, scale=anscombe_scale, clip=False)[0],
    }
    planes = {}
    stats = {}
    for mode, transformed in maps.items():
        _z, zclip, stat = standardize_by_self(transformed, clip_threshold=clip_threshold)
        planes[mode] = zclip
        stats[mode] = {f"normalize_{key}": value for key, value in stat.items()}
    return planes, stats


def bright_mask(planes: dict[str, np.ndarray], mode: str, *, threshold: float, dilation: int) -> np.ndarray:
    if mode == "log_lupton":
        mask = (planes["log"] >= threshold) & (planes["lupton"] >= threshold)
    elif mode == "anscombe":
        mask = planes["anscombe"] >= threshold
    else:
        raise ValueError(mode)
    if dilation > 0 and np.any(mask):
        mask = ndimage.binary_dilation(mask, iterations=dilation)
    return np.asarray(mask, dtype=bool)


def component_stats(mask: np.ndarray) -> dict[str, float | int]:
    labels, nlabels = ndimage.label(mask)
    areas = np.bincount(labels.ravel())[1:] if nlabels else np.asarray([], dtype=np.int64)
    return {
        "pixels": int(mask.sum()),
        "fraction": float(mask.mean()),
        "components": int(nlabels),
        "component_area_median": float(np.median(areas)) if areas.size else 0.0,
        "component_area_p90": float(np.quantile(areas, 0.9)) if areas.size else 0.0,
        "component_area_max": int(areas.max()) if areas.size else 0,
        "components_area_ge_1000": int(np.count_nonzero(areas >= 1000)) if areas.size else 0,
    }


def compare_masks(old_mask: np.ndarray, new_mask: np.ndarray) -> dict[str, object]:
    old_pixels = int(old_mask.sum())
    new_pixels = int(new_mask.sum())
    intersection = int(np.count_nonzero(old_mask & new_mask))
    union = int(np.count_nonzero(old_mask | new_mask))
    return {
        "old_raw_first": component_stats(old_mask),
        "new_zscore_first": component_stats(new_mask),
        "intersection_pixels": intersection,
        "union_pixels": union,
        "jaccard": float(intersection / union) if union else 1.0,
        "old_only_pixels": int(np.count_nonzero(old_mask & ~new_mask)),
        "new_only_pixels": int(np.count_nonzero(new_mask & ~old_mask)),
        "old_only_fraction_of_old": float(np.count_nonzero(old_mask & ~new_mask) / old_pixels) if old_pixels else 0.0,
        "new_only_fraction_of_new": float(np.count_nonzero(new_mask & ~old_mask) / new_pixels) if new_pixels else 0.0,
        "new_zscore_first_over_old_raw_first_pixel_ratio": float(new_pixels / old_pixels) if old_pixels else math.inf,
    }


def zscale_display(image: np.ndarray) -> np.ndarray:
    values = image[np.isfinite(image)]
    if values.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    try:
        vmin, vmax = ZScaleInterval().get_limits(values)
    except Exception:
        vmin, vmax = np.nanpercentile(values, [1, 99])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = np.nanpercentile(values, [1, 99])
    return np.clip((np.nan_to_num(image, nan=vmin) - vmin) / max(vmax - vmin, 1e-6), 0, 1).astype(np.float32)


def overlay_rgb(base: np.ndarray, mask: np.ndarray, color: tuple[float, float, float], *, alpha: float = 0.42) -> np.ndarray:
    rgb = np.dstack([base, base, base]).astype(np.float32)
    rgb[mask] = (1 - alpha) * rgb[mask] + alpha * np.asarray(color, dtype=np.float32)
    return np.clip(rgb, 0, 1)


def display_plane(zclip: np.ndarray, *, clip_threshold: float) -> np.ndarray:
    return np.clip((zclip + clip_threshold) / (2.0 * clip_threshold), 0, 1)


def save_bright_compare(
    out_dir: Path,
    band: str,
    patch: str,
    mode: str,
    image: np.ndarray,
    old_mask: np.ndarray,
    new_mask: np.ndarray,
    *,
    suffix: str,
) -> str:
    display = zscale_display(image)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
    panels = [
        ("old raw-first", old_mask, (1.0, 0.74, 0.0)),
        ("new zscore-first", new_mask, (0.0, 0.85, 0.45)),
        ("old only", old_mask & ~new_mask, (1.0, 0.18, 0.18)),
        ("new only", new_mask & ~old_mask, (0.0, 0.45, 1.0)),
    ]
    for ax, (title, mask, color) in zip(axes.ravel(), panels):
        ax.imshow(overlay_rgb(display, mask, color), origin="lower", interpolation="nearest")
        ax.set_title(f"{band} {mode} {title}", fontsize=10)
        ax.set_axis_off()
    path = out_dir / f"{band}_{patch.replace(',', '_')}_{mode}_{suffix}_vs_old_bright_region.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def save_tile_scaling(
    out_dir: Path,
    band: str,
    patch: str,
    old_planes: dict[str, np.ndarray],
    new_planes: dict[str, np.ndarray],
    row: int,
    col: int,
    *,
    tile_size: int,
    tile_step: int,
    clip_threshold: float,
    suffix: str,
    old_title: str,
    new_title: str,
) -> str:
    x0 = col * tile_step
    y0 = row * tile_step
    x1 = min(x0 + tile_size, old_planes["log"].shape[1])
    y1 = min(y0 + tile_size, old_planes["log"].shape[0])
    modes = ["log", "lupton", "anscombe"]
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 10.8), constrained_layout=True)
    for i, mode in enumerate(modes):
        for j, (title, planes) in enumerate([(old_title, old_planes), (new_title, new_planes)]):
            ax = axes[i, j]
            ax.imshow(
                display_plane(planes[mode][y0:y1, x0:x1], clip_threshold=clip_threshold),
                cmap="gray",
                origin="lower",
                interpolation="nearest",
                vmin=0,
                vmax=1,
            )
            if i == 0:
                ax.set_title(title, fontsize=10)
            if j == 0:
                ax.set_ylabel(mode, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"{band} {patch} grid_r{row:02d}_c{col:02d}", fontsize=12)
    path = out_dir / f"{band}_{patch.replace(',', '_')}_grid_r{row:02d}_c{col:02d}_{suffix}_vs_old_scaling.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tiles = parse_tiles(args.tiles)
    suffix = (
        f"zscore_q{args.new_lupton_q:g}_s{args.new_lupton_stretch:g}"
        f"_a{args.new_anscombe_scale:g}"
    ).replace(".", "p")
    old_title = (
        f"old raw: Q={args.old_lupton_q:g},s={args.old_lupton_stretch:g},"
        f"A={args.old_anscombe_scale:g}"
    )
    new_title = (
        f"zscore: Q={args.new_lupton_q:g},s={args.new_lupton_stretch:g},"
        f"A={args.new_anscombe_scale:g}"
    )

    summary = {
        "description": "Old raw-first scaling versus candidate zscore-first scaling.",
        "left_column": "raw FITS -> old log/lupton/anscombe -> self standardize -> clip",
        "right_column": "raw FITS -> robust zscore(no raw clip, no z clip) -> candidate log/lupton/anscombe -> self standardize -> clip",
        "output_dir": str(args.out_dir),
        "patch": args.patch,
        "suffix": suffix,
        "old_params": {
            "log_a": args.log_a,
            "log_high_percentile": args.log_high_percentile,
            "lupton_q": args.old_lupton_q,
            "lupton_stretch": args.old_lupton_stretch,
            "anscombe_scale": args.old_anscombe_scale,
        },
        "new_params": {
            "log_a": args.log_a,
            "log_high_percentile": args.log_high_percentile,
            "lupton_q": args.new_lupton_q,
            "lupton_stretch": args.new_lupton_stretch,
            "anscombe_scale": args.new_anscombe_scale,
            "lupton_minimum_after_zscore": 0.0,
        },
        "bright_threshold": args.bright_threshold,
        "clip_threshold": args.clip_threshold,
        "dilation": args.dilation,
        "bands": {},
        "tile_figures": [],
        "bright_region_figures": [],
    }
    rows = []
    scale_rows = []

    for band in args.bands:
        print(f"processing {band}", flush=True)
        fits_path = args.fits_root / band / args.patch / f"calexp-{band}-{args.tract}-{args.patch}.fits"
        image = read_image(fits_path)
        zimage, zstats = robust_zscore_no_raw_clip(
            image,
            sigma=args.zscore_sigma,
            maxiters=args.zscore_maxiters,
        )
        old_planes, old_stats = transform_then_normalize(
            image,
            log_a=args.log_a,
            log_high_percentile=args.log_high_percentile,
            lupton_q=args.old_lupton_q,
            lupton_stretch=args.old_lupton_stretch,
            lupton_minimum=float(zstats["zscore_median"]),
            anscombe_scale=args.old_anscombe_scale,
            clip_threshold=args.clip_threshold,
        )
        new_planes, new_stats = transform_then_normalize(
            zimage,
            log_a=args.log_a,
            log_high_percentile=args.log_high_percentile,
            lupton_q=args.new_lupton_q,
            lupton_stretch=args.new_lupton_stretch,
            lupton_minimum=0.0,
            anscombe_scale=args.new_anscombe_scale,
            clip_threshold=args.clip_threshold,
        )
        band_entry = {"fits_path": str(fits_path), "zscore_stats": zstats, "modes": {}}
        for transform_mode in ["log", "lupton", "anscombe"]:
            old_norm = old_stats[transform_mode]
            new_norm = new_stats[transform_mode]
            scale_rows.append(
                {
                    "band": band,
                    "transform": transform_mode,
                    "old_raw_first_normalize_mean": old_norm["normalize_pixel_mean"],
                    "old_raw_first_normalize_std": old_norm["normalize_pixel_std"],
                    "new_zscore_first_normalize_mean": new_norm["normalize_pixel_mean"],
                    "new_zscore_first_normalize_std": new_norm["normalize_pixel_std"],
                    "new_over_old_normalize_std": (
                        new_norm["normalize_pixel_std"] / old_norm["normalize_pixel_std"]
                        if old_norm["normalize_pixel_std"]
                        else math.nan
                    ),
                    "old_raw_first_zclip_hi_fraction": old_norm["normalize_zmax_pixel_fraction"],
                    "new_zscore_first_zclip_hi_fraction": new_norm["normalize_zmax_pixel_fraction"],
                }
            )
        for bright_mode in ["log_lupton", "anscombe"]:
            old_mask = bright_mask(
                old_planes,
                bright_mode,
                threshold=args.bright_threshold,
                dilation=args.dilation,
            )
            new_mask = bright_mask(
                new_planes,
                bright_mode,
                threshold=args.bright_threshold,
                dilation=args.dilation,
            )
            comparison = compare_masks(old_mask, new_mask)
            figure = save_bright_compare(
                args.out_dir,
                band,
                args.patch,
                bright_mode,
                image,
                old_mask,
                new_mask,
                suffix=suffix,
            )
            summary["bright_region_figures"].append(figure)
            band_entry["modes"][bright_mode] = comparison
            row = {"band": band, "mode": bright_mode}
            row.update(
                {
                    "old_raw_first_pixels": comparison["old_raw_first"]["pixels"],
                    "old_raw_first_fraction": comparison["old_raw_first"]["fraction"],
                    "old_raw_first_components": comparison["old_raw_first"]["components"],
                    "old_raw_first_area_median": comparison["old_raw_first"]["component_area_median"],
                    "old_raw_first_area_p90": comparison["old_raw_first"]["component_area_p90"],
                    "old_raw_first_area_max": comparison["old_raw_first"]["component_area_max"],
                    "old_raw_first_components_area_ge_1000": comparison["old_raw_first"]["components_area_ge_1000"],
                    "new_zscore_first_pixels": comparison["new_zscore_first"]["pixels"],
                    "new_zscore_first_fraction": comparison["new_zscore_first"]["fraction"],
                    "new_zscore_first_components": comparison["new_zscore_first"]["components"],
                    "new_zscore_first_area_median": comparison["new_zscore_first"]["component_area_median"],
                    "new_zscore_first_area_p90": comparison["new_zscore_first"]["component_area_p90"],
                    "new_zscore_first_area_max": comparison["new_zscore_first"]["component_area_max"],
                    "new_zscore_first_components_area_ge_1000": comparison["new_zscore_first"]["components_area_ge_1000"],
                    "new_zscore_first_over_old_raw_first_pixel_ratio": comparison[
                        "new_zscore_first_over_old_raw_first_pixel_ratio"
                    ],
                    "jaccard": comparison["jaccard"],
                    "old_only_fraction_of_old": comparison["old_only_fraction_of_old"],
                    "new_only_fraction_of_new": comparison["new_only_fraction_of_new"],
                }
            )
            rows.append(row)
        if band in set(args.visualize_bands):
            for row_idx, col_idx in tiles:
                figure = save_tile_scaling(
                    args.out_dir,
                    band,
                    args.patch,
                    old_planes,
                    new_planes,
                    row_idx,
                    col_idx,
                    tile_size=args.tile_size,
                    tile_step=args.tile_step,
                    clip_threshold=args.clip_threshold,
                    suffix=suffix,
                    old_title=old_title,
                    new_title=new_title,
                )
                summary["tile_figures"].append(figure)
        summary["bands"][band] = band_entry

    summary_path = args.out_dir / f"{suffix}_vs_old_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    bright_csv = args.out_dir / f"{suffix}_vs_old_bright_region_summary.csv"
    with bright_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    scale_csv = args.out_dir / f"{suffix}_vs_old_normalize_scale_summary.csv"
    with scale_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scale_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scale_rows)
    print(f"wrote {args.out_dir}")
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
