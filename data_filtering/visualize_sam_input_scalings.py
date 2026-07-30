#!/usr/bin/env python3
"""Visualize candidate single-band contrast scalings for SAM/CELLECT inputs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from astropy.stats import sigma_clipped_stats
    from astropy.visualization import make_lupton_rgb
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This script requires astropy and matplotlib.") from exc


@dataclass(frozen=True)
class InputSpec:
    name: str
    band: str
    path: Path
    dataset: str = "coadd"
    group: str = ""
    patch: str = ""
    crop_x0: int | None = None
    crop_y0: int | None = None
    crop_size: int | None = None
    log_a: float | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("output/data_filter_0727/sam_input_scaling_heatmaps"))
    p.add_argument("--preprocessed-root", type=Path, default=Path("/data/czh23/preprocessed"))
    p.add_argument("--calexp-root", type=Path, default=Path("/data/shared/Subaru"))
    p.add_argument("--root", type=Path, default=None, help="Legacy coadd cutout root. Defaults to PREPROCESSED_ROOT/9813/4,5/cutouts.")
    p.add_argument("--sam-root", type=Path, default=Path("zangetsu_demo/data/sam_x18204_y20924"))
    p.add_argument("--tract", default="9813")
    p.add_argument("--patch", default="4,5")
    p.add_argument("--datasets", nargs="+", default=["coadd"], choices=["coadd", "denoised", "noisy"])
    p.add_argument("--groups", nargs="+", default=["group_00"])
    p.add_argument("--regular-source", choices=["preprocessed", "calexp"], default="preprocessed")
    p.add_argument("--regular-tiles", nargs="+", default=["r4c6", "r7c10", "r5c4"])
    p.add_argument("--regular-tile-size", type=int, default=512)
    p.add_argument(
        "--custom-regions",
        nargs="*",
        default=[],
        metavar="BAND:PATCH:NAME:X:Y:SIZE[:LOG_A]",
        help="Crop full calexp regions by parent/physical center. Example: NB0387:7,6:c29210_26110:29210:26110:1024:3000",
    )
    p.add_argument(
        "--log-a-by-band",
        nargs="*",
        default=[],
        metavar="BAND=VALUE",
        help="Override --log-a for selected bands.",
    )
    p.add_argument(
        "--include-regular-cutouts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the regular r4c6/r7c10 cutouts.",
    )
    p.add_argument(
        "--include-sam-cutout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the sam_x18204_y20924 cutout for the selected datasets.",
    )
    p.add_argument("--bands", nargs="+", default=["HSC-I", "HSC-Y"])
    p.add_argument("--hdu", type=int, default=1)
    p.add_argument("--asinh-low-percentile", type=float, default=0.1)
    p.add_argument("--asinh-high-percentile", type=float, default=99.5)
    p.add_argument("--lupton-minimum", type=float, default=0.0)
    p.add_argument(
        "--lupton-minimum-mode",
        choices=["fixed", "zscore-mean", "zscore-median", "raw-median", "image-min"],
        default="fixed",
        help="Source for Lupton minimum. fixed uses --lupton-minimum; zscore-* reuses the current SAM zscore background estimate.",
    )
    p.add_argument("--lupton-stretch", type=float, default=0.5)
    p.add_argument("--lupton-q", type=float, default=20.0)
    p.add_argument("--include-log-scale", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--log-a", type=float, default=300.0)
    p.add_argument("--log-high-percentile", type=float, default=99.5)
    p.add_argument(
        "--include-anscombe-scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include an Anscombe variance-stabilizing transform followed by self-standardization.",
    )
    p.add_argument(
        "--anscombe-scale",
        type=float,
        default=1000.0,
        help="Multiply image-minus-finite-minimum by this factor before applying 2*sqrt(x + 3/8).",
    )
    p.add_argument(
        "--log-minimum-mode",
        choices=["same-as-lupton", "fixed", "zscore-mean", "zscore-median", "raw-median", "image-min"],
        default="image-min",
        help="Source for log scaling minimum. image-min uses the finite image minimum, which is usually a stable background-side zero point for HSC cutouts.",
    )
    p.add_argument("--log-minimum", type=float, default=0.0)
    return p.parse_args()


def read_image(path: Path, hdu: int) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=True) as hdul:
        if hdu < len(hdul) and hdul[hdu].data is not None:
            return np.asarray(hdul[hdu].data, dtype=np.float32), hdul[hdu].header.copy()
        for candidate in hdul:
            if candidate.data is not None and np.asarray(candidate.data).ndim == 2:
                return np.asarray(candidate.data, dtype=np.float32), candidate.header.copy()
    raise ValueError(f"no 2D image HDU found: {path}")


def origin_from_ltv(header: fits.Header) -> tuple[int, int]:
    if "LTV1" not in header or "LTV2" not in header:
        return (0, 0)
    return -int(round(float(header["LTV1"]))), -int(round(float(header["LTV2"])))


def crop_by_parent_box(
    image: np.ndarray,
    header: fits.Header,
    *,
    parent_x0: int,
    parent_y0: int,
    size: int,
) -> tuple[np.ndarray, fits.Header]:
    origin_x, origin_y = origin_from_ltv(header)
    local_x0 = int(round(parent_x0 - origin_x))
    local_y0 = int(round(parent_y0 - origin_y))
    local_x1 = local_x0 + int(size)
    local_y1 = local_y0 + int(size)
    if local_x0 < 0 or local_y0 < 0 or local_x1 > image.shape[1] or local_y1 > image.shape[0]:
        raise ValueError(
            f"crop [{parent_x0}:{parent_x0 + size}, {parent_y0}:{parent_y0 + size}] "
            f"is outside {image.shape[::-1]} with parent origin {(origin_x, origin_y)}"
        )
    cropped = np.asarray(image[local_y0:local_y1, local_x0:local_x1], dtype=np.float32)
    out_header = header.copy()
    if "LTV1" in out_header:
        out_header["LTV1"] = float(out_header["LTV1"]) - local_x0
    if "LTV2" in out_header:
        out_header["LTV2"] = float(out_header["LTV2"]) - local_y0
    for key, delta in (("CRPIX1", local_x0), ("CRPIX2", local_y0)):
        if key in out_header:
            out_header[key] = float(out_header[key]) - float(delta)
    out_header["CELCTX0"] = int(parent_x0)
    out_header["CELCTY0"] = int(parent_y0)
    out_header["CELCTSZ"] = int(size)
    return cropped, out_header


def read_spec_image(spec: InputSpec, hdu: int) -> tuple[np.ndarray, fits.Header]:
    image, header = read_image(spec.path, hdu)
    if spec.crop_x0 is None or spec.crop_y0 is None or spec.crop_size is None:
        return image, header
    return crop_by_parent_box(
        image,
        header,
        parent_x0=spec.crop_x0,
        parent_y0=spec.crop_y0,
        size=spec.crop_size,
    )


def finite_values(image: np.ndarray) -> np.ndarray:
    vals = np.asarray(image, dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("image has no finite pixels")
    return vals.astype(np.float64, copy=False)


def current_sam_zscore(image: np.ndarray, *, clip_sigma: float = 3.0, z_clip: tuple[float, float] = (-3.0, 3.0)) -> tuple[np.ndarray, dict[str, float]]:
    vals = finite_values(image)
    raw_min = float(np.min(vals))
    raw_median = float(np.median(vals))
    raw_sigma = float(np.std(vals))
    if not np.isfinite(raw_sigma) or raw_sigma <= 0:
        raw_sigma = 1.0
    clip_hi = raw_median + float(clip_sigma) * raw_sigma
    clipped_vals = np.minimum(vals, clip_hi)
    mean, median, std = sigma_clipped_stats(clipped_vals, sigma=float(clip_sigma), maxiters=None)
    mean = float(mean) if np.isfinite(mean) else float(np.mean(clipped_vals))
    median = float(median) if np.isfinite(median) else float(np.median(clipped_vals))
    std = float(std) if np.isfinite(std) and std > 0 else float(np.std(clipped_vals))
    if not np.isfinite(std) or std <= 0:
        std = 1.0
    safe = np.where(np.isfinite(image), image, mean).astype(np.float32, copy=False)
    capped = np.minimum(safe, clip_hi)
    z = (capped - mean) / std
    z = np.clip(z, z_clip[0], z_clip[1]).astype(np.float32)
    return z, {
        "raw_median": raw_median,
        "raw_min": raw_min,
        "raw_sigma": raw_sigma,
        "clip_hi": clip_hi,
        "mean": mean,
        "median": median,
        "std": std,
        "clip_hi_pixel_fraction": float(np.count_nonzero(vals >= clip_hi) / vals.size),
        "zmax_pixel_fraction": float(np.count_nonzero(z >= z_clip[1]) / z.size),
    }


def no_first_clip_zscore(image: np.ndarray, *, clip_sigma: float = 3.0, z_clip: tuple[float, float] = (-3.0, 3.0)) -> tuple[np.ndarray, dict[str, float]]:
    vals = finite_values(image)
    mean, _median, std = sigma_clipped_stats(vals, sigma=float(clip_sigma), maxiters=None)
    mean = float(mean) if np.isfinite(mean) else float(np.mean(vals))
    std = float(std) if np.isfinite(std) and std > 0 else float(np.std(vals))
    if not np.isfinite(std) or std <= 0:
        std = 1.0
    safe = np.where(np.isfinite(image), image, mean).astype(np.float32, copy=False)
    z = (safe - mean) / std
    z = np.clip(z, z_clip[0], z_clip[1]).astype(np.float32)
    return z, {
        "mean": mean,
        "std": std,
        "zmax_pixel_fraction": float(np.count_nonzero(z >= z_clip[1]) / z.size),
    }


def ds9_asinh(image: np.ndarray, *, low_pct: float, high_pct: float) -> tuple[np.ndarray, dict[str, float]]:
    vals = finite_values(image)
    lo = float(np.percentile(vals, low_pct))
    hi = float(np.percentile(vals, high_pct))
    if not np.isfinite(hi) or not np.isfinite(lo) or hi <= lo:
        lo = float(np.min(vals))
        hi = float(np.max(vals))
    if not np.isfinite(hi) or not np.isfinite(lo) or hi <= lo:
        hi = lo + 1.0
    x = (np.nan_to_num(image, nan=lo, posinf=hi, neginf=lo).astype(np.float32) - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0)
    y = np.arcsinh(10.0 * x) / 3.0
    return y.astype(np.float32), {
        "lo": lo,
        "hi": hi,
        "high_clip_fraction": float(np.count_nonzero(vals >= hi) / vals.size),
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
    y = np.asarray(rgb[..., 0], dtype=np.float32)
    return y, {
        "minimum": float(minimum),
        "stretch": float(stretch),
        "q": float(q),
        "output_dtype": "float",
    }


def log_single(
    image: np.ndarray,
    *,
    minimum: float,
    high_pct: float,
    a: float,
) -> tuple[np.ndarray, dict[str, float]]:
    vals = finite_values(image)
    hi = float(np.percentile(vals, float(high_pct)))
    minimum = float(minimum)
    if not np.isfinite(hi) or hi <= minimum:
        hi = float(np.max(vals))
    if not np.isfinite(hi) or hi <= minimum:
        hi = minimum + 1.0
    safe = np.nan_to_num(image, nan=minimum, posinf=hi, neginf=minimum).astype(np.float32, copy=False)
    x = np.clip((safe - minimum) / (hi - minimum), 0.0, 1.0)
    a = float(a) if np.isfinite(a) and a > 0 else 300.0
    y = np.log1p(a * x) / np.log(a)
    return y.astype(np.float32), {
        "minimum": minimum,
        "hi": hi,
        "a": a,
        "high_clip_fraction": float(np.count_nonzero(vals >= hi) / vals.size),
    }


def anscombe_single(image: np.ndarray, *, scale: float) -> tuple[np.ndarray, dict[str, float]]:
    """Apply the classical Anscombe transform after shifting/scaling the image.

    HSC inputs are background-subtracted and can contain negative pixels, while
    the Anscombe transform expects non-negative Poisson-like values.  Shifting
    by the finite image minimum preserves all pixel ordering and makes the
    transform well-defined.  Scaling before the transform keeps small
    mean-minus-minimum differences from being dominated by the 3/8 correction.
    """
    vals = finite_values(image)
    minimum = float(np.min(vals))
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Anscombe scale must be finite and positive, got {scale!r}")
    safe = np.nan_to_num(image, nan=minimum, posinf=minimum, neginf=minimum).astype(
        np.float64, copy=False
    )
    shifted_scaled = np.maximum((safe - minimum) * scale, 0.0)
    transformed = 2.0 * np.sqrt(shifted_scaled + 3.0 / 8.0)
    return transformed.astype(np.float32), {
        "minimum": minimum,
        "scale": scale,
        "mean_minus_minimum": float(np.mean(vals) - minimum),
        "scaled_mean_minus_minimum": float((np.mean(vals) - minimum) * scale),
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


def lupton_soft_float(image: np.ndarray, *, minimum: float, stretch: float, q: float) -> tuple[np.ndarray, dict[str, float]]:
    safe = np.nan_to_num(image, nan=minimum, posinf=minimum, neginf=minimum).astype(np.float32, copy=False)
    positive = np.maximum(safe - float(minimum), 0.0)
    stretch = float(stretch) if np.isfinite(stretch) and stretch > 0 else 1.0
    q = float(q) if np.isfinite(q) and q > 0 else 1.0
    # DeepDISC uses astropy Lupton with minimum=0 by default for HSC, which is
    # already approximately background-subtracted.  This single-band diagnostic
    # applies the same asinh softening directly to the positive intensity.
    y = np.arcsinh(q * positive / stretch) / q
    vals = finite_values(y)
    vmax = float(np.percentile(vals, 99.9))
    if np.isfinite(vmax) and vmax > 0:
        y = np.clip(y / vmax, 0.0, 1.0)
    return y.astype(np.float32), {
        "minimum": float(minimum),
        "stretch": stretch,
        "q": q,
        "p99_9_before_display_norm": vmax,
    }


def standardize_by_self(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    vals = finite_values(arr)
    mean = float(np.mean(vals))
    std = float(np.std(vals))
    if not np.isfinite(std) or std <= 0:
        std = 1.0
    safe = np.nan_to_num(arr, nan=mean, posinf=mean, neginf=mean).astype(np.float32, copy=False)
    z = ((safe - mean) / std).astype(np.float32)
    z_clip = np.clip(z, -3.0, 3.0).astype(np.float32)
    return z, z_clip, {
        "pixel_mean": mean,
        "pixel_std": std,
        "zmax_pixel_fraction": float(np.count_nonzero(z_clip >= 3.0) / z_clip.size),
        "zmin_pixel_fraction": float(np.count_nonzero(z_clip <= -3.0) / z_clip.size),
        "full_z_p99": float(np.percentile(z[np.isfinite(z)], 99.0)),
        "full_z_p99_9": float(np.percentile(z[np.isfinite(z)], 99.9)),
        "full_z_max": float(np.max(z[np.isfinite(z)])),
    }


def write_float_fits(path: Path, arr: np.ndarray, header: fits.Header) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(path, np.asarray(arr, dtype=np.float32), header=header, overwrite=True)


def display_name(spec: InputSpec) -> str:
    group = f" {spec.group}" if spec.group else ""
    patch = f" {spec.patch}" if spec.patch else ""
    return f"{spec.dataset}{group}{patch} {spec.name} {spec.band}"


def plot_clipped_comparison(title_name: str, maps: list[tuple[str, np.ndarray, dict[str, float]]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(maps), figsize=(4.2 * len(maps), 4.4), constrained_layout=True)
    for ax, (title, arr, stats) in zip(np.ravel(axes), maps):
        im = ax.imshow(arr, origin="lower", cmap="magma", interpolation="nearest", vmin=-3.0, vmax=3.0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=8)
        if "zmax_pixel_fraction" in stats:
            ax.text(
                0.02,
                0.98,
                f"max frac={stats['zmax_pixel_fraction'] * 100:.2f}%",
                transform=ax.transAxes,
                va="top",
                ha="left",
                color="white",
                fontsize=8,
                bbox={"facecolor": "black", "alpha": 0.45, "pad": 2, "edgecolor": "none"},
            )
    fig.suptitle(f"{title_name}: candidate inputs clipped to [-3, 3]", fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_asinh_lupton_details(
    title_name: str,
    maps: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(3, len(maps), figsize=(4.4 * len(maps), 11.6), constrained_layout=True)
    for col, (title, raw, full, clipped, stats) in enumerate(maps):
        for row, (kind, arr) in enumerate((("raw mapping", raw), ("self-standardized full z", full), ("z clipped [-3,3]", clipped))):
            ax = axes[row, col] if len(maps) > 1 else axes[row]
            im_kwargs = {"origin": "lower", "cmap": "magma", "interpolation": "nearest"}
            if row == 2:
                im_kwargs.update({"vmin": -3.0, "vmax": 3.0})
            im = ax.imshow(arr, **im_kwargs)
            ax.set_title(f"{title}\n{kind}", fontsize=10)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=8)
            if row == 0:
                text = (
                    f"raw mean={stats['raw_mean']:.4g}\n"
                    f"raw std={stats['raw_std']:.4g}"
                )
            elif row == 1:
                text = (
                    f"mean={stats['pixel_mean']:.4g}\n"
                    f"std={stats['pixel_std']:.4g}\n"
                    f"p99.9={stats['full_z_p99_9']:.3g}"
                )
            else:
                text = f"+3 frac={stats['zmax_pixel_fraction'] * 100:.2f}%"
            ax.text(
                0.02,
                0.98,
                text,
                transform=ax.transAxes,
                va="top",
                ha="left",
                color="white",
                fontsize=8,
                bbox={"facecolor": "black", "alpha": 0.45, "pad": 2, "edgecolor": "none"},
            )
    fig.suptitle(f"{title_name}: nonlinear scaling details", fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def regular_cutout_path(
    *,
    preprocessed_root: Path,
    tract: str,
    patch: str,
    dataset: str,
    group: str,
    tile: str,
    band: str,
    legacy_coadd_root: Path,
) -> Path:
    if dataset == "coadd":
        return legacy_coadd_root / tile / band / f"calexp-{band}-{tract}-{patch}.fits"
    tile_name = f"{group}_{tile}"
    return (
        preprocessed_root
        / dataset
        / str(tract)
        / patch
        / "cutouts"
        / tile_name
        / band
        / f"{dataset}-{band}-{tract}-{patch}-{group}.fits"
    )


def sam_cutout_path(
    *,
    sam_root: Path,
    tract: str,
    patch: str,
    dataset: str,
    group: str,
    tile: str,
    band: str,
) -> Path:
    if dataset == "coadd":
        return sam_root / dataset / str(tract) / patch / "cutouts" / tile / band / f"calexp-{band}-{tract}-{patch}.fits"
    tile_name = f"{group}_{tile}"
    return (
        sam_root
        / dataset
        / str(tract)
        / patch
        / "cutouts"
        / tile_name
        / band
        / f"{dataset}-{band}-{tract}-{patch}-{group}.fits"
    )


def parse_key_value_floats(entries: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"expected KEY=VALUE entry, got {entry!r}")
        key, value = entry.split("=", 1)
        out[key] = float(value)
    return out


def calexp_path(*, calexp_root: Path, tract: str, band: str, patch: str) -> Path:
    return calexp_root / str(tract) / band / patch / f"calexp-{band}-{tract}-{patch}.fits"


def parse_tile_xy(tile_name: str) -> tuple[int, int]:
    parts = tile_name.split("_")
    x_part = next((part for part in parts if part.startswith("x")), None)
    y_part = next((part for part in parts if part.startswith("y")), None)
    if x_part is None or y_part is None:
        raise ValueError(f"tile name does not contain x/y parent origin: {tile_name}")
    return int(x_part[1:]), int(y_part[1:])


def parse_custom_region(entry: str, default_log_a: float, log_a_by_band: dict[str, float]) -> dict[str, object]:
    fields = entry.split(":")
    if len(fields) not in (6, 7):
        raise ValueError(
            "custom region must be BAND:PATCH:NAME:X:Y:SIZE[:LOG_A], "
            f"got {entry!r}"
        )
    band, patch, name, x_text, y_text, size_text = fields[:6]
    log_a = float(fields[6]) if len(fields) == 7 else log_a_by_band.get(band, default_log_a)
    size = int(round(float(size_text)))
    center_x = float(x_text)
    center_y = float(y_text)
    return {
        "band": band,
        "patch": patch,
        "name": name,
        "x0": int(round(center_x - size / 2.0)),
        "y0": int(round(center_y - size / 2.0)),
        "size": size,
        "log_a": log_a,
    }


def build_specs(args: argparse.Namespace) -> list[InputSpec]:
    tiles = {
        "r4c6": "grid_r04_c06_x18108_y21372",
        "r7c10": "grid_r07_c10_x19588_y22476",
        "r5c4": "grid_r05_c04_x17372_y21740",
    }
    log_a_by_band = parse_key_value_floats(args.log_a_by_band)
    legacy_coadd_root = (
        args.root
        if args.root is not None
        else args.preprocessed_root / str(args.tract) / args.patch / "cutouts"
    )
    specs: list[InputSpec] = []
    for dataset in args.datasets:
        groups = [""] if dataset == "coadd" else list(args.groups)
        for group in groups:
            if bool(args.include_regular_cutouts):
                for name in args.regular_tiles:
                    if name not in tiles:
                        raise ValueError(f"unknown regular tile {name!r}; known tiles: {sorted(tiles)}")
                    tile = tiles[name]
                    for band in args.bands:
                        crop_x0 = crop_y0 = crop_size = None
                        if args.regular_source == "calexp":
                            path = calexp_path(
                                calexp_root=args.calexp_root,
                                tract=str(args.tract),
                                band=band,
                                patch=args.patch,
                            )
                            crop_x0, crop_y0 = parse_tile_xy(tile)
                            crop_size = int(args.regular_tile_size)
                        else:
                            path = regular_cutout_path(
                                preprocessed_root=args.preprocessed_root,
                                tract=str(args.tract),
                                patch=args.patch,
                                dataset=dataset,
                                group=group,
                                tile=tile,
                                band=band,
                                legacy_coadd_root=legacy_coadd_root,
                            )
                        if path.exists():
                            specs.append(
                                InputSpec(
                                    name=name,
                                    band=band,
                                    path=path,
                                    dataset=dataset,
                                    group=group,
                                    patch=args.patch,
                                    crop_x0=crop_x0,
                                    crop_y0=crop_y0,
                                    crop_size=crop_size,
                                    log_a=log_a_by_band.get(band, args.log_a),
                                )
                            )
                        else:
                            print(f"skip missing {dataset} {group} {name} {band}: {path}")
            if bool(args.include_sam_cutout):
                for band in args.bands:
                    path = sam_cutout_path(
                        sam_root=args.sam_root,
                        tract=str(args.tract),
                        patch=args.patch,
                        dataset=dataset,
                        group=group,
                        tile="sam_x18204_y20924",
                        band=band,
                    )
                    if path.exists():
                        specs.append(
                            InputSpec(
                                name="sam_x18204_y20924",
                                band=band,
                                path=path,
                                dataset=dataset,
                                group=group,
                                patch=args.patch,
                                log_a=log_a_by_band.get(band, args.log_a),
                            )
                        )
                    else:
                        print(f"skip missing {dataset} {group} sam_x18204_y20924 {band}: {path}")
    for entry in args.custom_regions:
        region = parse_custom_region(entry, args.log_a, log_a_by_band)
        band = str(region["band"])
        patch = str(region["patch"])
        path = calexp_path(calexp_root=args.calexp_root, tract=str(args.tract), band=band, patch=patch)
        if path.exists():
            specs.append(
                InputSpec(
                    name=str(region["name"]),
                    band=band,
                    path=path,
                    dataset="coadd",
                    patch=patch,
                    crop_x0=int(region["x0"]),
                    crop_y0=int(region["y0"]),
                    crop_size=int(region["size"]),
                    log_a=float(region["log_a"]),
                )
            )
        else:
            print(f"skip missing custom region {entry}: {path}")
    return specs


def main() -> int:
    args = parse_args()
    rows: list[dict[str, object]] = []
    specs = build_specs(args)
    if not specs:
        raise FileNotFoundError("no input FITS matched the requested datasets/groups/bands/tiles")
    for spec in specs:
        image, header = read_spec_image(spec, args.hdu)
        current_zclip, current_stats = current_sam_zscore(image)
        nofirst_zclip, nofirst_stats = no_first_clip_zscore(image)
        asinh_map, asinh_stats = ds9_asinh(
            image,
            low_pct=args.asinh_low_percentile,
            high_pct=args.asinh_high_percentile,
        )
        lupton_minimum = resolve_minimum(args.lupton_minimum_mode, args.lupton_minimum, current_stats)
        lupton_map, lupton_stats = lupton_single(
            image,
            minimum=lupton_minimum,
            stretch=args.lupton_stretch,
            q=args.lupton_q,
        )
        asinh_z, asinh_zclip, asinh_zstats = standardize_by_self(asinh_map)
        lupton_z, lupton_zclip, lupton_zstats = standardize_by_self(lupton_map)
        asinh_zstats.update({f"source_{k}": v for k, v in asinh_stats.items()})
        lupton_zstats.update({f"source_{k}": v for k, v in lupton_stats.items()})
        asinh_zstats.update(
            {
                "raw_mean": float(np.nanmean(asinh_map)),
                "raw_std": float(np.nanstd(asinh_map)),
            }
        )
        lupton_zstats.update(
            {
                "raw_mean": float(np.nanmean(lupton_map)),
                "raw_std": float(np.nanstd(lupton_map)),
            }
        )
        anscombe_map = anscombe_z = anscombe_zclip = None
        anscombe_zstats: dict[str, float] = {}
        if args.include_anscombe_scale:
            anscombe_map, anscombe_stats = anscombe_single(image, scale=args.anscombe_scale)
            anscombe_z, anscombe_zclip, anscombe_zstats = standardize_by_self(anscombe_map)
            anscombe_zstats.update({f"source_{k}": v for k, v in anscombe_stats.items()})
            anscombe_zstats.update(
                {
                    "raw_mean": float(np.nanmean(anscombe_map)),
                    "raw_std": float(np.nanstd(anscombe_map)),
                }
            )
        log_map = log_z = log_zclip = None
        log_zstats: dict[str, float] = {}
        if args.include_log_scale:
            log_a = float(spec.log_a) if spec.log_a is not None else float(args.log_a)
            if args.log_minimum_mode == "same-as-lupton":
                log_minimum = lupton_minimum
            else:
                log_minimum = resolve_minimum(args.log_minimum_mode, args.log_minimum, current_stats)
            log_map, log_stats = log_single(
                image,
                minimum=log_minimum,
                high_pct=args.log_high_percentile,
                a=log_a,
            )
            log_z, log_zclip, log_zstats = standardize_by_self(log_map)
            log_zstats.update({f"source_{k}": v for k, v in log_stats.items()})
            log_zstats.update(
                {
                    "raw_mean": float(np.nanmean(log_map)),
                    "raw_std": float(np.nanstd(log_map)),
                }
            )
        scalings = [
            ("sam_zscore_rawclip3", (current_zclip, current_stats)),
            (
                f"asinh_selfstd_p{args.asinh_low_percentile:g}_p{args.asinh_high_percentile:g}",
                (asinh_zclip, asinh_zstats),
            ),
            (
                f"lupton_selfstd_s{args.lupton_stretch:g}_Q{args.lupton_q:g}",
                (lupton_zclip, lupton_zstats),
            ),
            ("zscore_no_first_raw_clip", (nofirst_zclip, nofirst_stats)),
        ]
        if args.include_anscombe_scale:
            scalings.insert(
                3,
                (
                    f"anscombe_selfstd_scale{args.anscombe_scale:g}",
                    (anscombe_zclip, anscombe_zstats),
                ),
            )
        if args.include_log_scale:
            scalings.insert(
                4 if args.include_anscombe_scale else 3,
                (
                    f"log_selfstd_a{log_a:g}_p{args.log_high_percentile:g}",
                    (log_zclip, log_zstats),
                ),
            )
        safe_group = spec.group if spec.group else "nogroup"
        safe_patch = spec.patch if spec.patch else args.patch
        safe_name = f"{spec.dataset}_{safe_group}_{safe_patch}_{spec.name}_{spec.band}".replace(",", "_").replace("-", "_")
        title_name = display_name(spec)
        plot_clipped_comparison(
            title_name,
            [(title, arr, stats) for title, (arr, stats) in scalings],
            args.out_dir / f"{safe_name}_scaling_heatmaps.png",
        )
        standardized = [
            (
                f"asinh p{args.asinh_low_percentile:g}-p{args.asinh_high_percentile:g}",
                asinh_map,
                asinh_z,
                asinh_zclip,
                asinh_zstats,
            ),
            (
                f"lupton min={lupton_minimum:.4g} stretch={args.lupton_stretch:g} Q={args.lupton_q:g}",
                lupton_map,
                lupton_z,
                lupton_zclip,
                lupton_zstats,
            ),
        ]
        if args.include_anscombe_scale:
            standardized.append(
                (
                    f"anscombe min={float(anscombe_zstats['source_minimum']):.4g} "
                    f"scale={args.anscombe_scale:g}",
                    anscombe_map,
                    anscombe_z,
                    anscombe_zclip,
                    anscombe_zstats,
                )
            )
        if args.include_log_scale:
            standardized.append(
                (
                    f"log min={float(log_zstats['source_minimum']):.4g} a={log_a:g} p{args.log_high_percentile:g}",
                    log_map,
                    log_z,
                    log_zclip,
                    log_zstats,
                )
            )
        plot_asinh_lupton_details(
            title_name,
            standardized,
            args.out_dir / f"{safe_name}_self_standardized_full_vs_clipped_heatmaps.png",
        )
        for title, (arr, stats) in scalings:
            out_fits = args.out_dir / "fits" / f"{safe_name}_{title.replace('[', '').replace(']', '').replace(',', '_')}.fits"
            write_float_fits(out_fits, arr, header)
            row = {
                "region": spec.name,
                "band": spec.band,
                "dataset": spec.dataset,
                "group": spec.group,
                "patch": safe_patch,
                "crop_x0": spec.crop_x0,
                "crop_y0": spec.crop_y0,
                "crop_size": spec.crop_size,
                "input_path": str(spec.path),
                "scaling": title,
                "fits": str(out_fits),
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
                "mean": float(np.nanmean(arr)),
                "median": float(np.nanmedian(arr)),
                "p99": float(np.nanpercentile(arr[np.isfinite(arr)], 99.0)),
            }
            row.update(stats)
            rows.append(row)
        for title, raw, full, clipped, stats in standardized:
            safe_title = (
                title.replace(" ", "_")
                .replace("=", "")
                .replace("[", "")
                .replace("]", "")
                .replace(",", "_")
                .replace("-", "_")
            )
            full_fits = args.out_dir / "fits" / f"{safe_name}_{safe_title}_self_standardized_full_z.fits"
            clip_fits = args.out_dir / "fits" / f"{safe_name}_{safe_title}_self_standardized_zclip_m3_p3.fits"
            raw_fits = args.out_dir / "fits" / f"{safe_name}_{safe_title}_raw_mapping.fits"
            write_float_fits(raw_fits, raw, header)
            write_float_fits(full_fits, full, header)
            write_float_fits(clip_fits, clipped, header)
            for variant, arr, out_fits in (
                ("raw_mapping", raw, raw_fits),
                ("full_z", full, full_fits),
                ("zclip_m3_p3", clipped, clip_fits),
            ):
                finite = arr[np.isfinite(arr)]
                row = {
                    "region": spec.name,
                    "band": spec.band,
                    "dataset": spec.dataset,
                    "group": spec.group,
                    "patch": safe_patch,
                    "crop_x0": spec.crop_x0,
                    "crop_y0": spec.crop_y0,
                    "crop_size": spec.crop_size,
                    "input_path": str(spec.path),
                    "scaling": f"{title}_{variant}",
                    "fits": str(out_fits),
                    "min": float(np.nanmin(arr)),
                    "max": float(np.nanmax(arr)),
                    "mean": float(np.nanmean(arr)),
                    "median": float(np.nanmedian(arr)),
                    "p99": float(np.nanpercentile(finite, 99.0)),
                }
                row.update(stats)
                rows.append(row)
        print(f"wrote {args.out_dir / f'{safe_name}_scaling_heatmaps.png'}")
        print(f"wrote {args.out_dir / f'{safe_name}_self_standardized_full_vs_clipped_heatmaps.png'}")

    summary_path = args.out_dir / "scaling_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
