#!/usr/bin/env python3
"""Overlay preprocessed per-band background masks on a variant Zangetsu cutout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval


DEFAULT_DATA_ROOT = Path("/nvme0/zc/scarlet/preprocessed")
DEFAULT_OUT_DIR = Path("/home/czh23/CELLECT/zangetsu_demo/output/confidence_map_overlays_0628/training_background")
DEFAULT_BANDS = ("HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y")


def _zscale_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)
    if not bool(finite.any()):
        scaled = np.zeros_like(image, dtype=np.float32)
    else:
        interval = ZScaleInterval()
        try:
            vmin, vmax = interval.get_limits(image[finite])
        except Exception:
            vmin, vmax = float(np.nanpercentile(image[finite], 1.0)), float(np.nanpercentile(image[finite], 99.0))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = float(np.nanmin(image[finite])), float(np.nanmax(image[finite]))
        scaled = np.clip((image - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    return np.repeat(scaled[..., None], 3, axis=2)


def _find_image(path_root: Path, band: str, variant: str, tract: int, patch: str, group: str) -> Path:
    band_dir = path_root / band
    matches = sorted(band_dir.glob(f"{variant}-{band}-{tract}-{patch}-{group}.fits"))
    if not matches:
        matches = sorted(band_dir.glob("*.fits"))
    if not matches:
        raise FileNotFoundError(f"no FITS image found in {band_dir}")
    return matches[0]


def _read_background(target_path: Path) -> np.ndarray:
    with np.load(target_path) as data:
        if "background_mask" not in data:
            raise KeyError(f"{target_path} has no background_mask")
        return np.asarray(data["background_mask"], dtype=bool)


def _overlay(image: np.ndarray, background: np.ndarray, alpha: float) -> np.ndarray:
    rgb = _zscale_rgb(image)
    color = np.asarray([0.05, 0.35, 1.0], dtype=np.float32)
    mask = np.asarray(background, dtype=bool)
    rgb[mask] = (1.0 - float(alpha)) * rgb[mask] + float(alpha) * color[None, :]
    return np.clip(rgb, 0.0, 1.0)


def _save(path: Path, rgb: np.ndarray, title: str, alpha: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    ax.imshow(rgb, origin="lower", interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(
        handles=[mpatches.Patch(color=(0.05, 0.35, 1.0, alpha), label="training background")],
        loc="lower right",
        fontsize=7,
        framealpha=0.72,
    )
    fig.tight_layout(pad=0.2)
    fig.savefig(path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dataset", default="denoised")
    parser.add_argument("--variant", default=None, help="FITS basename. Defaults to --dataset.")
    parser.add_argument("--tract", type=int, default=9813)
    parser.add_argument("--patch", default="4,5")
    parser.add_argument("--group", default="group_01")
    parser.add_argument("--tile", default="sam_x18204_y20924")
    parser.add_argument("--bands", nargs="+", default=list(DEFAULT_BANDS))
    parser.add_argument("--alpha", type=float, default=0.62)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    variant = str(args.variant or args.dataset)
    tile = f"{args.group}_{args.tile}" if args.group and not str(args.tile).startswith(f"{args.group}_") else str(args.tile)
    patch_root = data_root / str(args.dataset) / str(args.tract) / str(args.patch)
    image_root = patch_root / "cutouts" / tile

    summary: list[dict[str, object]] = []
    for band in args.bands:
        image_path = _find_image(image_root, band, variant, args.tract, args.patch, args.group)
        target_path = patch_root / "band_targets" / band / f"{tile}.npz"
        background = _read_background(target_path)
        image = np.asarray(fits.getdata(image_path, ext=1), dtype=np.float32)
        rgb = _overlay(image, background, args.alpha)
        prefix = f"{args.dataset}_{args.patch.replace(',', '_')}_{tile}_{band.replace('-', '_')}"
        out_path = out_dir / f"{prefix}_actual_background_overlay.png"
        _save(out_path, rgb, f"{args.dataset} {args.patch}/{tile} {band}: actual training background", args.alpha)
        row = {
            "dataset": str(args.dataset),
            "band": band,
            "tile": tile,
            "image": str(image_path),
            "target": str(target_path),
            "output": str(out_path),
            "background_pixels": int(np.count_nonzero(background)),
            "total_pixels": int(background.size),
            "background_fraction": float(np.count_nonzero(background) / max(background.size, 1)),
        }
        summary.append(row)
        print(f"wrote {out_path} background={row['background_pixels']}/{row['total_pixels']}")

    summary_path = out_dir / f"{args.dataset}_{args.patch.replace(',', '_')}_{tile}_background_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
