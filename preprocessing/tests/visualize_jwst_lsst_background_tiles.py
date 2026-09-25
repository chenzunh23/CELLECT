#!/usr/bin/env python3
"""Visualize LSST detection background masks on selected JWST cutouts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from matplotlib.colors import ListedColormap


DEFAULT_RAW_ROOT = Path("/data/shared/jwst_foundation/raw/COSMOS_1727_1837_5893")
DEFAULT_BACKGROUND_ROOT = Path("/data/czh23/JWST/lsst_background_masks")
DEFAULT_OUT_DIR = Path("/home/czh23/analysis/2026-09/2026-09-06/jwst_lsst_background_static")
SIZE = 4096
DS = 2


def raw_fits_path(root: Path, pointing: str, band: str) -> Path:
    point = f"{int(pointing):04d}" if str(pointing).isdigit() else str(pointing)
    return root / f"Pointing_{point}" / f"COSMOS_pointing_{point}_{band.upper()}_detector_p001.fits"


def background_npz_path(root: Path, tract: str, pointing: str, band: str) -> Path:
    point = f"{int(pointing):04d}" if str(pointing).isdigit() else str(pointing)
    return root / "jwst" / tract / point / "group_00" / band.lower() / "background_mask.npz"


def summary_path(root: Path, tract: str, pointing: str, band: str) -> Path:
    return background_npz_path(root, tract, pointing, band).with_name("summary.json")


def first_image_hdu(hdul: fits.HDUList):
    for hdu in hdul:
        if hdu.data is not None and getattr(hdu.data, "ndim", 0) == 2:
            return hdu
    raise ValueError("no 2-D image HDU found")


def stretch(cut: np.ndarray) -> np.ndarray:
    finite = cut[np.isfinite(cut)]
    if finite.size == 0:
        return np.zeros(cut.shape, dtype=np.float32)
    sample = finite[:: max(1, finite.size // 500_000)]
    try:
        lo, hi = ZScaleInterval(contrast=0.25, krej=2.5, max_iterations=5).get_limits(sample)
    except Exception:
        lo, hi = np.nanpercentile(sample, [0.5, 99.7])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.nanpercentile(sample, [0.5, 99.7])
    image = np.nan_to_num(cut, nan=lo, posinf=hi, neginf=lo)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def downsample_bool(mask: np.ndarray) -> np.ndarray:
    h = mask.shape[0] // DS
    w = mask.shape[1] // DS
    return mask[: h * DS, : w * DS].reshape(h, DS, w, DS).max(axis=(1, 3))


def regions_for_shape(nx: int, ny: int) -> dict[str, tuple[int, int]]:
    return {
        "left_top_4096": (0, ny - SIZE),
        "center_4096": (nx // 2 - SIZE // 2, ny // 2 - SIZE // 2),
        "top_row_col3_4096": (2 * SIZE, ny - SIZE),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--background-root", type=Path, default=DEFAULT_BACKGROUND_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tract", default="default")
    parser.add_argument("--pointing", default="0019")
    parser.add_argument("--band", default="f444w")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_path = raw_fits_path(args.raw_root.expanduser().resolve(), args.pointing, args.band)
    mask_path = background_npz_path(args.background_root.expanduser().resolve(), args.tract, args.pointing, args.band)
    stat_path = summary_path(args.background_root.expanduser().resolve(), args.tract, args.pointing, args.band)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(mask_path) as data:
        background = np.asarray(data["background_mask"], dtype=bool)
    with fits.open(raw_path, memmap=True) as hdul:
        hdu = first_image_hdu(hdul)
        image = hdu.data
        ny, nx = image.shape
        regions = regions_for_shape(nx, ny)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
        rows: list[dict[str, object]] = []
        for ax, (name, (x0, y0)) in zip(axes, regions.items(), strict=True):
            cut = np.asarray(image[y0 : y0 + SIZE, x0 : x0 + SIZE], dtype=np.float32)
            bg_cut = background[y0 : y0 + SIZE, x0 : x0 + SIZE]
            footprint_cut = ~bg_cut
            finite_cut = np.isfinite(cut)
            display = stretch(cut[::DS, ::DS])

            ax.imshow(display, origin="lower", cmap="gray", interpolation="nearest")
            ax.imshow(
                np.ma.masked_where(~downsample_bool(footprint_cut), downsample_bool(footprint_cut)),
                origin="lower",
                cmap=ListedColormap(["#ff3030"]),
                alpha=0.24,
                interpolation="nearest",
            )
            ax.imshow(
                np.ma.masked_where(~downsample_bool(bg_cut), downsample_bool(bg_cut)),
                origin="lower",
                cmap=ListedColormap(["#00d060"]),
                alpha=0.08,
                interpolation="nearest",
            )
            ax.set_title(
                f"{name}\nbackground {bg_cut.mean():.3f}, footprint {footprint_cut.mean():.3f}",
                fontsize=11,
            )
            ax.set_xlim(0, SIZE // DS)
            ax.set_ylim(0, SIZE // DS)
            ax.set_axis_off()
            rows.append(
                {
                    "region": name,
                    "x0": x0,
                    "y0": y0,
                    "background_pixels": int(np.count_nonzero(bg_cut)),
                    "footprint_pixels": int(np.count_nonzero(footprint_cut)),
                    "finite_pixels": int(np.count_nonzero(finite_cut)),
                    "background_fraction": float(bg_cut.mean()),
                    "footprint_fraction": float(footprint_cut.mean()),
                    "nonfinite_fraction": float((~finite_cut).mean()),
                }
            )

    handles = [
        plt.Line2D([0], [0], color="#ff3030", lw=6, alpha=0.55, label="LSST detection footprint"),
        plt.Line2D([0], [0], color="#00d060", lw=6, alpha=0.35, label="LSST background"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10, framealpha=0.9)
    title = f"Pointing {args.pointing} {args.band.upper()}: LSST static detection background"
    if stat_path.exists():
        info = json.loads(stat_path.read_text())
        title += (
            f"  mode={info.get('detection_mode')} "
            f"thr={info.get('threshold_value')} grow={info.get('n_sigma_to_grow')}"
        )
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))

    out_png = out_dir / f"pointing{args.pointing}_{args.band.lower()}_three_blocks_lsst_background.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    out_csv = out_dir / f"pointing{args.pointing}_{args.band.lower()}_three_blocks_lsst_background_summary.csv"
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(out_png)
    print(out_csv)


if __name__ == "__main__":
    main()
