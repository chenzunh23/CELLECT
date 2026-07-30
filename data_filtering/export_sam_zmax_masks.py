#!/usr/bin/env python3
"""Export masks of pixels equal to the maximum value after SAM astro zscore preprocessing."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellect_mplconfig")

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ZScaleInterval
from matplotlib.colors import ListedColormap


DEFAULT_BANDS = ("HSC-I",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="*", type=Path, default=[])
    parser.add_argument("--coadd-root", type=Path, default=Path("/data/shared/Subaru"))
    parser.add_argument("--tract", default="9813")
    parser.add_argument("--patches", nargs="+", default=[])
    parser.add_argument("--bands", nargs="+", default=list(DEFAULT_BANDS))
    parser.add_argument("--out-dir", type=Path, default=Path("output/data_filter_0727/sam_zmax_masks"))
    parser.add_argument("--hdu", default="IMAGE")
    parser.add_argument("--crop", nargs=3, type=int, metavar=("X0", "Y0", "SIZE"), default=None)
    parser.add_argument("--clip-sigma", type=float, default=3.0)
    parser.add_argument(
        "--initial-clip-sigmas",
        nargs="*",
        type=float,
        default=[],
        help="Run several raw bright-clip sigmas while keeping the later sigma-clipped stats sigma fixed.",
    )
    parser.add_argument(
        "--stats-sigma",
        type=float,
        default=None,
        help="Sigma used by astropy sigma_clipped_stats after the raw bright clip. Defaults to --clip-sigma.",
    )
    parser.add_argument("--sigma-iters", type=int, default=-1)
    parser.add_argument("--z-clip", nargs=2, type=float, default=[-3.0, 3.0])
    parser.add_argument("--eps", type=float, default=1.0e-6)
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--write-zscore", action="store_true")
    return parser.parse_args()


def calexp_path(root: Path, tract: str, band: str, patch: str) -> Path:
    return root / str(tract) / band / patch / f"calexp-{band}-{tract}-{patch}.fits"


def input_paths(args: argparse.Namespace) -> list[Path]:
    paths = [p.expanduser() for p in args.inputs]
    for patch in args.patches:
        for band in args.bands:
            paths.append(calexp_path(args.coadd_root, args.tract, band, patch))
    if not paths:
        raise ValueError("provide --inputs or --patches/--bands")
    return paths


def find_image_hdu(hdul: fits.HDUList, hdu_spec: str):
    if str(hdu_spec).isdigit():
        hdu = hdul[int(hdu_spec)]
        if hdu.data is not None and getattr(hdu.data, "ndim", None) == 2:
            return hdu
    else:
        try:
            hdu = hdul[hdu_spec]
            if hdu.data is not None and getattr(hdu.data, "ndim", None) == 2:
                return hdu
        except Exception:
            pass
    for hdu in hdul:
        if hdu.data is not None and getattr(hdu.data, "ndim", None) == 2:
            return hdu
    raise KeyError("no 2D image HDU found")


def read_image(path: Path, hdu_spec: str) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=True, ignore_missing_end=True) as hdul:
        hdu = find_image_hdu(hdul, hdu_spec)
        image = np.asarray(hdu.data, dtype=np.float32)
        header = hdu.header.copy()
    return image, header


def crop_image(image: np.ndarray, header: fits.Header, crop: list[int] | None) -> tuple[np.ndarray, fits.Header, tuple[int, int]]:
    if crop is None:
        return image, header, (0, 0)
    x0, y0, size = map(int, crop)
    x1 = min(image.shape[1], x0 + size)
    y1 = min(image.shape[0], y0 + size)
    if x0 < 0 or y0 < 0 or x0 >= image.shape[1] or y0 >= image.shape[0] or x1 <= x0 or y1 <= y0:
        raise ValueError(f"invalid crop {crop} for image shape {image.shape}")
    out_header = header.copy()
    out_header["LTV1"] = float(header.get("LTV1", 0.0)) - float(x0)
    out_header["LTV2"] = float(header.get("LTV2", 0.0)) - float(y0)
    if "CRPIX1" in out_header:
        out_header["CRPIX1"] = float(out_header["CRPIX1"]) - float(x0)
    if "CRPIX2" in out_header:
        out_header["CRPIX2"] = float(out_header["CRPIX2"]) - float(y0)
    return image[y0:y1, x0:x1], out_header, (x0, y0)


def sam_astro_zscore(
    image: np.ndarray,
    *,
    clip_sigma: float,
    initial_clip_sigma: float | None = None,
    stats_sigma: float | None = None,
    sigma_iters: int,
    z_clip: tuple[float, float] | None,
) -> tuple[np.ndarray, dict[str, float]]:
    vals = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(vals)
    if not np.any(finite):
        return np.zeros_like(vals, dtype=np.float32), {
            "raw_median": float("nan"),
            "raw_sigma": float("nan"),
            "clip_hi": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "z_max": 0.0,
        }
    finite_vals = vals[finite].astype(np.float64, copy=False)
    raw_median = float(np.median(finite_vals))
    raw_sigma = float(np.std(finite_vals))
    if not math.isfinite(raw_sigma) or raw_sigma <= 0.0:
        raw_sigma = 1.0
    initial_sigma = float(clip_sigma if initial_clip_sigma is None else initial_clip_sigma)
    stats_sigma_value = float(clip_sigma if stats_sigma is None else stats_sigma)
    clip_hi = raw_median + initial_sigma * raw_sigma
    clipped_vals = np.minimum(finite_vals, clip_hi)
    maxiters = None if int(sigma_iters) < 0 else int(sigma_iters)
    mean, _median, std = sigma_clipped_stats(clipped_vals, sigma=stats_sigma_value, maxiters=maxiters)
    if not math.isfinite(float(mean)):
        mean = float(np.nanmean(clipped_vals))
    if not math.isfinite(float(std)) or float(std) <= 0.0:
        std = float(np.nanstd(clipped_vals))
    if not math.isfinite(float(std)) or float(std) <= 0.0:
        std = 1.0
    safe = np.where(finite, vals.astype(np.float64), float(mean))
    clipped = np.minimum(safe, clip_hi)
    z = ((clipped - float(mean)) / float(std)).astype(np.float32)
    if z_clip is not None:
        z = np.clip(z, float(z_clip[0]), float(z_clip[1])).astype(np.float32)
    return z, {
        "raw_median": raw_median,
        "raw_sigma": raw_sigma,
        "initial_clip_sigma": initial_sigma,
        "stats_sigma": stats_sigma_value,
        "clip_hi": float(clip_hi),
        "mean": float(mean),
        "std": float(std),
        "z_min": float(np.nanmin(z)),
        "z_max": float(np.nanmax(z)),
    }


def zmax_mask(z: np.ndarray, eps: float) -> np.ndarray:
    finite = np.isfinite(z)
    if not np.any(finite):
        return np.zeros(z.shape, dtype=bool)
    max_value = float(np.nanmax(z[finite]))
    return finite & (z >= max_value - float(eps))


def raw_clip_plateau_mask(image: np.ndarray, clip_hi: float) -> np.ndarray:
    finite = np.isfinite(image)
    return finite & (image >= float(clip_hi))


def zscale_limits(image: np.ndarray) -> tuple[float, float]:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    if finite.size > 2_000_000:
        finite = finite[:: int(math.ceil(finite.size / 2_000_000))]
    try:
        lo, hi = ZScaleInterval().get_limits(finite)
    except Exception:
        lo, hi = np.nanpercentile(finite, [1, 99])
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanpercentile(finite, 1)), float(np.nanpercentile(finite, 99))
    return float(lo), float(hi)


def downsample_image(image: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return image
    h, w = image.shape
    hh = h // factor
    ww = w // factor
    trimmed = image[: hh * factor, : ww * factor]
    return np.nanmean(trimmed.reshape(hh, factor, ww, factor), axis=(1, 3))


def downsample_bool(mask: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return mask
    h, w = mask.shape
    hh = h // factor
    ww = w // factor
    trimmed = mask[: hh * factor, : ww * factor]
    return trimmed.reshape(hh, factor, ww, factor).any(axis=(1, 3))


def save_overlay(path: Path, image: np.ndarray, z: np.ndarray, mask: np.ndarray, downsample: int) -> None:
    image_ds = downsample_image(image, downsample)
    mask_ds = downsample_bool(mask, downsample)
    lo, hi = zscale_limits(image_ds)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), constrained_layout=True)
    axes[0].imshow(image_ds, origin="lower", cmap="gray", vmin=lo, vmax=hi)
    axes[0].imshow(np.ma.masked_where(~mask_ds, mask_ds), origin="lower", cmap=ListedColormap(["red"]), alpha=0.55)
    if np.any(mask_ds):
        axes[0].contour(mask_ds.astype(np.uint8), levels=[0.5], colors=["red"], linewidths=0.7)
    axes[0].set_title(f"zscale + zmax mask; pixels={int(np.count_nonzero(mask))}")
    z_ds = downsample_image(z, downsample)
    axes[1].imshow(z_ds, origin="lower", cmap="magma", vmin=float(np.nanpercentile(z_ds, 1)), vmax=float(np.nanmax(z_ds)))
    axes[1].imshow(np.ma.masked_where(~mask_ds, mask_ds), origin="lower", cmap=ListedColormap(["cyan"]), alpha=0.45)
    axes[1].set_title(f"SAM zscore; max={float(np.nanmax(z)):.4g}")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_rawclip_overlay(path: Path, image: np.ndarray, mask: np.ndarray, title: str, downsample: int) -> None:
    image_ds = downsample_image(image, downsample)
    mask_ds = downsample_bool(mask, downsample)
    lo, hi = zscale_limits(image_ds)
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.imshow(image_ds, origin="lower", cmap="gray", vmin=lo, vmax=hi)
    ax.imshow(np.ma.masked_where(~mask_ds, mask_ds), origin="lower", cmap=ListedColormap(["red"]), alpha=0.55)
    if np.any(mask_ds):
        ax.contour(mask_ds.astype(np.uint8), levels=[0.5], colors=["red"], linewidths=0.7)
    ax.set_title(f"{title}; pixels={int(np.count_nonzero(mask))}")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_comparison_overlay(
    path: Path,
    image: np.ndarray,
    results: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]],
    downsample: int,
) -> None:
    image_ds = downsample_image(image, downsample)
    lo, hi = zscale_limits(image_ds)
    n = len(results)
    fig, axes = plt.subplots(3, n, figsize=(5.6 * n, 15.5), constrained_layout=True)
    if n == 1:
        axes = np.asarray(axes).reshape(3, 1)
    for col, (sigma, z, z_mask, raw_mask, stats) in enumerate(results):
        raw_ds = downsample_bool(raw_mask, downsample)
        axes[0, col].imshow(image_ds, origin="lower", cmap="gray", vmin=lo, vmax=hi)
        axes[0, col].imshow(np.ma.masked_where(~raw_ds, raw_ds), origin="lower", cmap=ListedColormap(["red"]), alpha=0.55)
        if np.any(raw_ds):
            axes[0, col].contour(raw_ds.astype(np.uint8), levels=[0.5], colors=["red"], linewidths=0.7)
        axes[0, col].set_title(
            f"raw clip plateau {sigma:g}sigma\n"
            f"pixels={int(np.count_nonzero(raw_mask))} ({np.count_nonzero(raw_mask) / raw_mask.size:.3f})"
        )

        mask = z_mask
        mask_ds = downsample_bool(mask, downsample)
        axes[1, col].imshow(image_ds, origin="lower", cmap="gray", vmin=lo, vmax=hi)
        axes[1, col].imshow(np.ma.masked_where(~mask_ds, mask_ds), origin="lower", cmap=ListedColormap(["cyan"]), alpha=0.50)
        if np.any(mask_ds):
            axes[1, col].contour(mask_ds.astype(np.uint8), levels=[0.5], colors=["cyan"], linewidths=0.7)
        axes[1, col].set_title(
            f"final zmax after zscore\n"
            f"pixels={int(np.count_nonzero(mask))} ({np.count_nonzero(mask) / mask.size:.3f})"
        )
        z_ds = downsample_image(z, downsample)
        axes[2, col].imshow(
            z_ds,
            origin="lower",
            cmap="magma",
            vmin=float(np.nanpercentile(z_ds, 1)),
            vmax=float(np.nanmax(z_ds)),
        )
        axes[2, col].imshow(np.ma.masked_where(~mask_ds, mask_ds), origin="lower", cmap=ListedColormap(["cyan"]), alpha=0.45)
        axes[2, col].set_title(f"clip_hi={stats['clip_hi']:.4g}, mean={stats['mean']:.4g}, std={stats['std']:.4g}")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(path, dpi=170)
    plt.close(fig)


def stem_for(path: Path, origin: tuple[int, int]) -> str:
    stem = path.name.replace(".fits", "").replace(",", "_")
    if origin != (0, 0):
        stem += f"_x{origin[0]}_y{origin[1]}"
    return stem


def process_one(path: Path, args: argparse.Namespace) -> dict[str, object]:
    image, header = read_image(path, args.hdu)
    image, header, origin = crop_image(image, header, args.crop)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem_for(path, origin)
    sigmas = list(args.initial_clip_sigmas) if args.initial_clip_sigmas else [float(args.clip_sigma)]
    rows = []
    comparison = []
    for sigma in sigmas:
        z, stats = sam_astro_zscore(
            image,
            clip_sigma=args.clip_sigma,
            initial_clip_sigma=float(sigma),
            stats_sigma=args.stats_sigma,
            sigma_iters=args.sigma_iters,
            z_clip=tuple(args.z_clip) if args.z_clip is not None else None,
        )
        mask = zmax_mask(z, args.eps)
        raw_mask = raw_clip_plateau_mask(image, stats["clip_hi"])
        suffix = f"_initclip{float(sigma):g}sigma" if args.initial_clip_sigmas else ""
        suffix = suffix.replace(".", "p")
        mask_path = args.out_dir / f"{stem}{suffix}_sam_zmax_mask.fits"
        mask_header = header.copy()
        for key, value in stats.items():
            if math.isfinite(float(value)):
                mask_header[f"Z{key[:6].upper()}"] = float(value)
        fits.writeto(mask_path, mask.astype(np.uint8), mask_header, overwrite=True)
        raw_mask_path = args.out_dir / f"{stem}{suffix}_sam_rawclip_plateau_mask.fits"
        fits.writeto(raw_mask_path, raw_mask.astype(np.uint8), mask_header, overwrite=True)
        if args.write_zscore:
            fits.writeto(args.out_dir / f"{stem}{suffix}_sam_zscore.fits", z.astype(np.float32), header, overwrite=True)
        save_overlay(args.out_dir / f"{stem}{suffix}_sam_zmax_mask_overlay.png", image, z, mask, int(args.downsample))
        save_rawclip_overlay(
            args.out_dir / f"{stem}{suffix}_sam_rawclip_plateau_overlay.png",
            image,
            raw_mask,
            title=f"raw clip plateau {float(sigma):g}sigma",
            downsample=int(args.downsample),
        )
        row = {
            "input": str(path),
            "stem": stem,
            "x0": origin[0],
            "y0": origin[1],
            "height": image.shape[0],
            "width": image.shape[1],
            "mask_pixels": int(np.count_nonzero(mask)),
            "mask_fraction": float(np.count_nonzero(mask) / mask.size),
            "rawclip_plateau_pixels": int(np.count_nonzero(raw_mask)),
            "rawclip_plateau_fraction": float(np.count_nonzero(raw_mask) / raw_mask.size),
        }
        row.update(stats)
        rows.append(row)
        comparison.append((float(sigma), z, mask, raw_mask, stats))
    if len(comparison) > 1:
        save_comparison_overlay(args.out_dir / f"{stem}_sam_zmax_mask_initial_clip_compare.png", image, comparison, int(args.downsample))
    return rows


def main() -> int:
    args = parse_args()
    rows = []
    for path in input_paths(args):
        new_rows = process_one(path, args)
        rows.extend(new_rows)
        for row in new_rows:
            print(row, flush=True)
    summary_path = args.out_dir / "sam_zmax_mask_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["input"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
