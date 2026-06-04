#!/usr/bin/env python3
"""Visualize Zangetsu PU partitions with strict-ignore as the fourth panel."""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
import sys

import matplotlib
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import ZScaleInterval

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from astro_data_preprocessing import (
    TileSpec,
    _band_det_path,
    _crop_full_mask_for_tile,
    _find_image_hdu_index,
    _origin_from_ltv,
    _read_det_background_mask,
    crop_catalog_for_tile,
    make_pu_dense_targets,
)


COLORS = {
    "clean": (0.0, 0.8, 0.2),
    "strict_ignore": (1.0, 0.72, 0.0),
    "ignore": (1.0, 0.05, 0.05),
    "background": (0.0, 0.45, 1.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("zangetsu_demo/preprocessed"))
    parser.add_argument("--official-coadd-root", type=Path, default=Path("/data1/czh23/Subaru/9813"))
    parser.add_argument("--datasets", nargs="+", default=["coadd", "noisy", "denoised"])
    parser.add_argument("--tract", default="9813")
    parser.add_argument("--patch", default="6,1")
    parser.add_argument("--tile", default="zangetsu_lower_right_x27366_y6453")
    parser.add_argument("--band", default="HSC-I")
    parser.add_argument("--x0", type=int, default=27366)
    parser.add_argument("--y0", type=int, default=6453)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--margin", type=float, default=64.0)
    parser.add_argument("--out-dir", type=Path, default=Path("zangetsu_demo/pu_partition_overlays"))
    return parser.parse_args()


def zscale_limits(image: np.ndarray) -> tuple[float, float]:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    try:
        return tuple(float(v) for v in ZScaleInterval().get_limits(finite))
    except Exception:
        return tuple(float(v) for v in np.nanpercentile(finite, [1, 99]))


def read_image(path: Path) -> np.ndarray:
    with fits.open(path, memmap=True, ignore_missing_end=True) as hdul:
        for hdu in hdul:
            if hdu.data is not None and getattr(hdu.data, "ndim", 0) == 2:
                return np.asarray(hdu.data, dtype=np.float32)
    raise ValueError(f"no 2D image found in {path}")


def read_image_shape_and_origin(path: Path) -> tuple[tuple[int, int], tuple[int, int]]:
    with fits.open(path, memmap=True, ignore_missing_end=True) as hdul:
        hdu = hdul[_find_image_hdu_index(hdul)]
        return tuple(int(v) for v in hdu.data.shape[-2:]), _origin_from_ltv(hdu.header)


def rgba(mask: np.ndarray, color: tuple[float, float, float], alpha: float) -> np.ndarray:
    out = np.zeros((*mask.shape, 4), dtype=np.float32)
    out[mask] = (color[0], color[1], color[2], alpha)
    return out


def ellipse_area(row) -> float:
    try:
        major = float(row["ellipse_major_sigma"])
        minor = float(row["ellipse_minor_sigma"])
    except Exception:
        return -1.0
    if not np.isfinite(major) or not np.isfinite(minor):
        return -1.0
    return major * minor


def sorted_by_area_desc(table: Table) -> list:
    rows = list(table)
    rows.sort(key=ellipse_area, reverse=True)
    return rows


def paint_one_instance(
    overlay: np.ndarray,
    row,
    spec: TileSpec,
    color: tuple[float, float, float],
    *,
    alpha: float,
    ellipse_sigma: float,
) -> None:
    try:
        x = float(row["base_SdssCentroid_x"] - spec.x0)
        y = float(row["base_SdssCentroid_y"] - spec.y0)
        major = float(row["ellipse_major_sigma"]) * ellipse_sigma
        minor = float(row["ellipse_minor_sigma"]) * ellipse_sigma
        theta = float(row["ellipse_theta"])
    except Exception:
        return
    if not all(np.isfinite(v) for v in (x, y, major, minor, theta)):
        return
    major = max(major, 1.5)
    minor = max(minor, 1.5)
    radius = int(math.ceil(max(major, minor))) + 2
    cx = int(round(x))
    cy = int(round(y))
    y0 = max(0, cy - radius)
    y1 = min(overlay.shape[0], cy + radius + 1)
    x0 = max(0, cx - radius)
    x1 = min(overlay.shape[1], cx + radius + 1)
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx = xx - x
    dy = yy - y
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    xr = cos_t * dx + sin_t * dy
    yr = -sin_t * dx + cos_t * dy
    inside = (xr / major) ** 2 + (yr / minor) ** 2 <= 1.0
    overlay[y0:y1, x0:x1][inside] = (color[0], color[1], color[2], alpha)


def instance_overlay(table: Table, spec: TileSpec, shape: tuple[int, int], *, ellipse_sigma: float) -> np.ndarray:
    overlay = np.zeros((*shape, 4), dtype=np.float32)
    palette = plt.get_cmap("tab20")
    for idx, row in enumerate(sorted_by_area_desc(table)):
        paint_one_instance(overlay, row, spec, palette(idx % 20)[:3], alpha=0.46, ellipse_sigma=ellipse_sigma)
    return overlay


def local_rows(table: Table, *, x0: int, y0: int, size: int, margin: float) -> Table:
    if len(table) == 0:
        return table
    x = np.asarray(table["base_SdssCentroid_x"], dtype=float) - x0
    y = np.asarray(table["base_SdssCentroid_y"], dtype=float) - y0
    keep = np.isfinite(x) & np.isfinite(y) & (x >= -margin) & (x < size + margin) & (y >= -margin) & (y < size + margin)
    return table[keep]


def read_table(path: Path) -> Table:
    if not path.exists():
        return Table()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Table.read(path)


def official_background_mask(args: argparse.Namespace) -> np.ndarray | None:
    det_path = _band_det_path(args.official_coadd_root, args.band, int(args.tract), args.patch)
    if det_path is None:
        return None
    image_path = args.official_coadd_root / args.band / args.patch / f"calexp-{args.band}-{args.tract}-{args.patch}.fits"
    if not image_path.exists():
        return None
    shape_yx, origin_xy = read_image_shape_and_origin(image_path)
    full_background = _read_det_background_mask(det_path, shape_yx, origin_xy=origin_xy)
    spec = TileSpec(name=args.tile, x0=args.x0, y0=args.y0, size=args.size)
    return _crop_full_mask_for_tile(full_background, spec, origin_xy)


def plot_dataset(args: argparse.Namespace, dataset: str) -> Path:
    patch_root = args.root / dataset / args.tract / args.patch
    image_dir = patch_root / "cutouts" / args.tile / args.band
    image_path = next(image_dir.glob("*.fits"))
    image = read_image(image_path)
    catalog_path = patch_root / "band_reference_catalogs" / args.band / f"meas-{args.band}-{args.tract}-{args.patch}.fits"
    center_path = patch_root / "band_reference_center_only" / args.band / f"meas-{args.band}-{args.tract}-{args.patch}.fits"
    ignore_path = patch_root / "band_reference_ignore" / args.band / f"meas-{args.band}-{args.tract}-{args.patch}.fits"
    strict_path = patch_root / "band_reference_strict_ignore" / args.band / f"meas-{args.band}-{args.tract}-{args.patch}.fits"
    clean_table = read_table(catalog_path)
    center_table = read_table(center_path)
    ignore_table = read_table(ignore_path)
    strict_table = read_table(strict_path)
    clean_tile = local_rows(clean_table, x0=args.x0, y0=args.y0, size=args.size, margin=args.margin)
    spec = TileSpec(name=args.tile, x0=args.x0, y0=args.y0, size=args.size)
    center_tile = crop_catalog_for_tile(
        center_table,
        spec,
        x_col="base_SdssCentroid_x",
        y_col="base_SdssCentroid_y",
        margin=args.margin,
    ) if len(center_table) else center_table
    ignore_tile = crop_catalog_for_tile(
        ignore_table,
        spec,
        x_col="base_SdssCentroid_x",
        y_col="base_SdssCentroid_y",
        margin=args.margin,
    ) if len(ignore_table) else ignore_table
    strict_tile = crop_catalog_for_tile(
        strict_table,
        spec,
        x_col="base_SdssCentroid_x",
        y_col="base_SdssCentroid_y",
        margin=args.margin,
    ) if len(strict_table) else strict_table
    lsst_background = official_background_mask(args)
    targets = make_pu_dense_targets(
        clean_tile,
        center_tile,
        ignore_tile,
        spec,
        x_col="base_SdssCentroid_x",
        y_col="base_SdssCentroid_y",
        ellipse_sigma=1.0,
        confidence_levels=5,
        core_radius=2,
        center_only_weight=0.25,
        lsst_background_mask=lsst_background,
        strict_ignore_sources=strict_tile,
    )
    masks = {
        "clean": targets["clean_mask"] > 0,
        "strict_ignore": targets["strict_ignore_mask"] > 0,
        "ignore": targets["ignore_mask"] > 0,
        "background": targets["background_mask"] > 0,
    }

    vmin, vmax = zscale_limits(image)
    partition = np.zeros((*image.shape, 4), dtype=np.float32)
    for name in ("background", "clean", "strict_ignore", "ignore"):
        alpha = 0.23 if name == "background" else 0.43
        layer = rgba(masks[name], COLORS[name], alpha)
        replace = layer[..., 3] > 0
        partition[replace] = layer[replace]
    clean_overlay = instance_overlay(clean_tile, spec, masks["clean"].shape, ellipse_sigma=1.0)
    strict_overlay = rgba(masks["strict_ignore"], COLORS["strict_ignore"], 0.55)
    background_overlay = rgba(masks["background"], COLORS["background"], 0.45)

    fig, axes = plt.subplots(1, 5, figsize=(21, 4.8), constrained_layout=True)
    for ax in axes:
        ax.imshow(image, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_xlim(-0.5, args.size - 0.5)
        ax.set_ylim(-0.5, args.size - 0.5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].set_title("zscale")
    axes[1].imshow(partition, origin="lower")
    axes[1].set_title("PU partitions")
    axes[1].legend(
        handles=[Patch(facecolor=COLORS[name], alpha=0.55, label=name) for name in ("clean", "strict_ignore", "ignore", "background")],
        loc="lower left",
        fontsize=7,
        framealpha=0.75,
    )
    axes[2].imshow(clean_overlay, origin="lower")
    axes[2].set_title(f"clean instances n={len(clean_tile)}")
    axes[3].imshow(strict_overlay, origin="lower")
    axes[3].set_title(f"strict_ignore px={int(np.count_nonzero(masks['strict_ignore']))}")
    axes[4].imshow(background_overlay, origin="lower")
    bg_source = "official det" if lsst_background is not None else "none"
    axes[4].set_title(f"background px={int(np.count_nonzero(masks['background']))}")

    for row in clean_tile:
        x = float(row["base_SdssCentroid_x"] - args.x0)
        y = float(row["base_SdssCentroid_y"] - args.y0)
        if -args.margin <= x < args.size + args.margin and -args.margin <= y < args.size + args.margin:
            axes[2].plot(x, y, "+", color="white", markersize=3.5, markeredgewidth=0.8)

    fig.suptitle(
        f"{dataset} {args.band} {args.tract}/{args.patch} {args.tile}: "
        f"center_only panel replaced by strict_ignore; background={bg_source}"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{dataset}_{args.band}_{args.patch.replace(',', '_')}_{args.tile}_strict_ignore_panel.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> int:
    args = parse_args()
    for dataset in args.datasets:
        out = plot_dataset(args, dataset)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
