#!/usr/bin/env python3
"""Export downsampled Astropy-zscale PNGs for HSC calexp patch quality checks."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cellect_mplconfig")


def normalize_band_dir(band: str) -> str:
    band = band.strip().upper()
    if band.startswith("HSC-") or band.startswith("NB"):
        return band
    return f"HSC-{band}"


def parse_patches(values: Iterable[str]) -> list[str]:
    patches: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in str(value).split(";"):
            patch = item.strip()
            if not patch:
                continue
            expanded = [f"{x},{y}" for x in range(9) for y in range(9)] if patch.lower() == "all" else [patch]
            for candidate in expanded:
                if candidate not in seen:
                    patches.append(candidate)
                    seen.add(candidate)
    return patches


def discover_bands(data_root: Path, tract: str) -> list[str]:
    root = data_root / str(tract)
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir() and (path.name.startswith("HSC-") or path.name.startswith("NB")))


def find_calexp(path: Path) -> Path | None:
    files = sorted(path.glob("calexp-*.fits")) + sorted(path.glob("calexp-*.fits.gz"))
    return files[0] if files else None


def read_calexp_image(path: Path, hdu_index: int | None) -> tuple[np.ndarray, int]:
    from astropy.io import fits

    with fits.open(path, memmap=True, ignore_missing_end=True) as hdul:
        if hdu_index is not None:
            data = hdul[int(hdu_index)].data
            if data is None or np.asarray(data).ndim != 2:
                raise ValueError(f"HDU {hdu_index} does not contain a 2D image")
            return np.asarray(data, dtype=np.float32), int(hdu_index)
        for index, hdu in enumerate(hdul):
            data = hdu.data
            if data is not None and np.asarray(data).ndim == 2:
                return np.asarray(data, dtype=np.float32), index
    raise ValueError("no 2D image HDU found")


def finite_sample(image: np.ndarray, max_samples: int) -> np.ndarray:
    finite = np.isfinite(image)
    if not bool(finite.any()):
        return np.asarray([], dtype=np.float32)
    values = image[finite]
    if values.size <= max_samples:
        return values.astype(np.float32, copy=False)
    stride = max(1, int(math.ceil(values.size / max_samples)))
    return values[::stride].astype(np.float32, copy=False)


def zscale_limits(image: np.ndarray, *, max_samples: int, contrast: float) -> tuple[float, float]:
    from astropy.visualization import ZScaleInterval

    sample = finite_sample(image, max_samples=max_samples)
    if sample.size == 0:
        return 0.0, 1.0
    try:
        lo, hi = ZScaleInterval(contrast=contrast).get_limits(sample)
    except Exception:
        lo, hi = np.nanpercentile(sample, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        median = float(np.nanmedian(sample))
        sigma = float(np.nanstd(sample))
        if not np.isfinite(sigma) or sigma <= 0.0:
            sigma = 1.0
        lo, hi = median - sigma, median + sigma
    return float(lo), float(hi)


def downsample_mean(image: np.ndarray, factor: int) -> np.ndarray:
    factor = max(1, int(factor))
    if factor == 1:
        return image
    height, width = image.shape
    out_h = max(1, height // factor)
    out_w = max(1, width // factor)
    trimmed = image[: out_h * factor, : out_w * factor]
    with np.errstate(invalid="ignore"):
        return np.nanmean(trimmed.reshape(out_h, factor, out_w, factor), axis=(1, 3)).astype(np.float32, copy=False)


def choose_downsample_factor(shape: tuple[int, int], max_size: int, explicit: int | None) -> int:
    if explicit is not None and explicit > 0:
        return int(explicit)
    return max(1, int(math.ceil(max(shape) / max(1, int(max_size)))))


def image_to_uint8(image: np.ndarray, lo: float, hi: float, downsample: int) -> np.ndarray:
    small = downsample_mean(image, downsample)
    scaled = (small - lo) / max(hi - lo, 1e-6)
    scaled = np.clip(scaled, 0.0, 1.0)
    scaled[~np.isfinite(scaled)] = 0.0
    return np.round(scaled * 255.0).astype(np.uint8)


def write_png(gray: np.ndarray, path: Path, *, title: str | None = None) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig_w = max(4.0, gray.shape[1] / 180.0)
    fig_h = max(4.0, gray.shape[0] / 180.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=180)
    ax.imshow(gray, cmap="gray", origin="lower", interpolation="nearest", vmin=0, vmax=255)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=8)
    fig.tight_layout(pad=0.02)
    fig.savefig(path)
    plt.close(fig)


def write_contact_sheet(thumbs: dict[str, np.ndarray], out_path: Path, *, band_dir: str, thumb_size: int) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(9, 9, figsize=(13.5, 13.5), dpi=180)
    for y in range(9):
        for x in range(9):
            patch = f"{x},{y}"
            ax = axes[8 - y, x]
            ax.set_axis_off()
            gray = thumbs.get(patch)
            if gray is None:
                ax.text(0.5, 0.5, patch + "\nmissing", ha="center", va="center", fontsize=5, color="red")
                continue
            ax.imshow(gray, cmap="gray", origin="lower", interpolation="nearest", vmin=0, vmax=255)
            ax.text(
                0.02,
                0.98,
                patch,
                ha="left",
                va="top",
                transform=ax.transAxes,
                fontsize=5,
                color="yellow",
                bbox={"facecolor": "black", "alpha": 0.45, "pad": 0.5, "edgecolor": "none"},
            )
    fig.suptitle(f"{band_dir} calexp zscale overview | thumbnails <= {thumb_size}px", fontsize=12)
    fig.tight_layout(pad=0.05)
    fig.savefig(out_path)
    plt.close(fig)


def summarize_image(image: np.ndarray, lo: float, hi: float, *, hdu: int, path: Path, band_dir: str, patch: str, factor: int) -> dict[str, object]:
    finite = np.isfinite(image)
    finite_count = int(finite.sum())
    total = int(image.size)
    row: dict[str, object] = {
        "band": band_dir,
        "patch": patch,
        "path": str(path),
        "hdu": hdu,
        "height": int(image.shape[0]),
        "width": int(image.shape[1]),
        "downsample": int(factor),
        "file_size": int(path.stat().st_size),
        "finite_fraction": finite_count / max(total, 1),
        "zscale_lo": lo,
        "zscale_hi": hi,
    }
    if finite_count:
        values = finite_sample(image, max_samples=2_000_000)
        row.update(
            {
                "min": float(np.nanmin(values)),
                "p01": float(np.nanpercentile(values, 1.0)),
                "p50": float(np.nanpercentile(values, 50.0)),
                "p99": float(np.nanpercentile(values, 99.0)),
                "max": float(np.nanmax(values)),
                "zero_fraction_finite": float(np.mean(values == 0.0)),
            }
        )
    else:
        row.update({"min": "", "p01": "", "p50": "", "p99": "", "max": "", "zero_fraction_finite": ""})
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "band",
        "patch",
        "path",
        "hdu",
        "height",
        "width",
        "downsample",
        "file_size",
        "finite_fraction",
        "zero_fraction_finite",
        "zscale_lo",
        "zscale_hi",
        "min",
        "p01",
        "p50",
        "p99",
        "max",
        "status",
        "error",
        "png",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render HSC calexp FITS patches as downsampled Astropy-zscale PNGs.")
    parser.add_argument("--data-root", type=Path, default=Path("/data/shared/Subaru"))
    parser.add_argument("--tract", default="9813")
    parser.add_argument("--bands", nargs="*", default=None, help="Bands or band directories. Default: discover from data root.")
    parser.add_argument("--patches", nargs="+", default=["all"])
    parser.add_argument("--out-dir", type=Path, default=Path("output/calexp_quality_overview"))
    parser.add_argument("--hdu", type=int, default=None, help="FITS HDU to read. Default: first 2D image HDU.")
    parser.add_argument("--downsample", type=int, default=None, help="Explicit integer downsample factor.")
    parser.add_argument("--max-size", type=int, default=1024, help="Auto downsample so max image side is <= this size.")
    parser.add_argument("--thumb-size", type=int, default=256, help="Max side length for contact-sheet thumbnails.")
    parser.add_argument("--zscale-samples", type=int, default=200_000)
    parser.add_argument("--zscale-contrast", type=float, default=0.25)
    parser.add_argument("--no-contact-sheet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bands = [normalize_band_dir(band) for band in args.bands] if args.bands else discover_bands(args.data_root, args.tract)
    patches = parse_patches(args.patches)
    if not bands:
        print(f"no bands found under {args.data_root / str(args.tract)}", file=sys.stderr)
        return 2

    all_rows: list[dict[str, object]] = []
    for band_dir in bands:
        band_rows: list[dict[str, object]] = []
        thumbs: dict[str, np.ndarray] = {}
        for patch in patches:
            patch_dir = args.data_root / str(args.tract) / band_dir / patch
            calexp = find_calexp(patch_dir)
            png_path = args.out_dir / str(args.tract) / band_dir / "patch_pngs" / f"{band_dir}_{args.tract}_{patch.replace(',', '_')}_calexp_zscale.png"
            row: dict[str, object] = {"band": band_dir, "patch": patch, "path": "", "status": "missing", "error": "", "png": str(png_path)}
            if calexp is None:
                row["error"] = f"no calexp FITS in {patch_dir}"
                band_rows.append(row)
                all_rows.append(row)
                print(f"{band_dir} {patch}: missing", file=sys.stderr)
                continue
            try:
                image, used_hdu = read_calexp_image(calexp, args.hdu)
                lo, hi = zscale_limits(image, max_samples=int(args.zscale_samples), contrast=float(args.zscale_contrast))
                factor = choose_downsample_factor(image.shape, int(args.max_size), args.downsample)
                gray = image_to_uint8(image, lo, hi, factor)
                title = f"{band_dir} {args.tract} {patch} | hdu={used_hdu} | z=({lo:.3g},{hi:.3g}) | ds={factor}x"
                write_png(gray, png_path, title=title)
                thumb_factor = choose_downsample_factor(image.shape, int(args.thumb_size), None)
                thumbs[patch] = image_to_uint8(image, lo, hi, thumb_factor)
                row = summarize_image(image, lo, hi, hdu=used_hdu, path=calexp, band_dir=band_dir, patch=patch, factor=factor)
                row.update({"status": "ok", "error": "", "png": str(png_path)})
                print(f"{band_dir} {patch}: ok -> {png_path}", file=sys.stderr)
            except Exception as exc:  # noqa: BLE001 - continue through bad FITS files.
                row.update({"path": str(calexp), "status": "error", "error": f"{type(exc).__name__}: {exc}"})
                print(f"{band_dir} {patch}: ERROR {exc}", file=sys.stderr)
            band_rows.append(row)
            all_rows.append(row)

        band_out = args.out_dir / str(args.tract) / band_dir
        write_csv(band_out / f"{band_dir}_{args.tract}_calexp_quality_summary.csv", band_rows)
        if not args.no_contact_sheet:
            write_contact_sheet(
                thumbs,
                band_out / f"{band_dir}_{args.tract}_calexp_zscale_contact_sheet.png",
                band_dir=band_dir,
                thumb_size=int(args.thumb_size),
            )
    write_csv(args.out_dir / str(args.tract) / f"{args.tract}_calexp_quality_summary.csv", all_rows)
    print(f"wrote {args.out_dir / str(args.tract)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
