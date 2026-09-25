#!/usr/bin/env python3
"""Visualize selected COSMOS-Web blended segmentation masks."""

from __future__ import annotations

import csv
import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import ZScaleInterval
from astropy.wcs import FITSFixedWarning, WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse, Patch
from scipy import ndimage


RAW = Path(
    "/data/shared/jwst_foundation/raw/COSMOS_1727_1837_5893/"
    "Pointing_0019/COSMOS_pointing_0019_F444W_detector_p001.fits"
)
CATALOG = Path("/data/shared/jwst_foundation/catalog/COSMOS_1727_1837_5893/COSMOSWeb_mastercatalog_v1.1.fits")
SEG = Path(
    "/data/shared/jwst_foundation/catalog/COSMOS_1727_1837_5893/"
    "segmentation_maps/detection_chi2pos_SWLW_B3_segmap_v1.3.fits"
)
OUT_DIR = Path("/home/czh23/analysis/2026-09/2026-09-06/jwst_segmentation_blends")

GROUPS = {
    "ids_462527_462528": [462527, 462528],
    "ids_465127_465128_465129": [465127, 465128, 465129],
}
COLORS = ["#00ffff", "#ff3b30", "#ffd400", "#00d060", "#ff00ff"]


def first_image_hdu(hdul: fits.HDUList):
    for hdu in hdul:
        if hdu.data is not None and getattr(hdu.data, "ndim", 0) == 2:
            return hdu
    raise ValueError("no 2-D image HDU found")


def stretch(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.float32)
    sample = finite[:: max(1, finite.size // 400_000)]
    try:
        lo, hi = ZScaleInterval(contrast=0.25, krej=2.5, max_iterations=5).get_limits(sample)
    except Exception:
        lo, hi = np.nanpercentile(sample, [0.5, 99.7])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.nanpercentile(sample, [0.5, 99.7])
    return np.clip((np.nan_to_num(image, nan=lo, posinf=hi, neginf=lo) - lo) / (hi - lo), 0.0, 1.0)


def local_affine_raw_to_map(raw_wcs: WCS, map_wcs: WCS, x0: int, y0: int, size: int) -> np.ndarray:
    xs = np.linspace(x0, x0 + size - 1, 5)
    ys = np.linspace(y0, y0 + size - 1, 5)
    xx, yy = np.meshgrid(xs, ys)
    ra, dec = raw_wcs.all_pix2world(xx.ravel(), yy.ravel(), 0)
    mx, my = map_wcs.all_world2pix(ra, dec, 0)
    mat = np.column_stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
    ok = np.isfinite(mx) & np.isfinite(my)
    cx, *_ = np.linalg.lstsq(mat[ok], mx[ok], rcond=None)
    cy, *_ = np.linalg.lstsq(mat[ok], my[ok], rcond=None)
    return np.vstack([cx, cy])


def sample_seg_cutout(raw_wcs: WCS, seg_hdu, seg_wcs: WCS, x0: int, y0: int, size: int) -> np.ndarray:
    ny, nx = seg_hdu.shape
    affine = local_affine_raw_to_map(raw_wcs, seg_wcs, x0, y0, size)
    yy, xx = np.mgrid[y0 : y0 + size, x0 : x0 + size]
    mx = affine[0, 0] * xx + affine[0, 1] * yy + affine[0, 2]
    my = affine[1, 0] * xx + affine[1, 1] * yy + affine[1, 2]
    xi = np.rint(mx).astype(np.int64)
    yi = np.rint(my).astype(np.int64)
    valid = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
    out = np.zeros((size, size), dtype=np.int32)
    if not bool(np.any(valid)):
        return out
    xmin, xmax = int(xi[valid].min()), int(xi[valid].max())
    ymin, ymax = int(yi[valid].min()), int(yi[valid].max())
    section = np.asarray(seg_hdu.section[ymin : ymax + 1, xmin : xmax + 1])
    values = np.zeros(valid.shape, dtype=section.dtype)
    values[valid] = section[yi[valid] - ymin, xi[valid] - xmin]
    out[valid] = values[valid].astype(np.int32)
    return out


def load_sources(ids: list[int], raw_wcs: WCS) -> list[dict[str, object]]:
    tab = Table.read(CATALOG, hdu=1)
    all_ids = np.asarray(tab["id"], dtype=np.int64)
    rows = [int(np.flatnonzero(all_ids == sid)[0]) for sid in ids]
    ra = np.asarray(tab["ra"][rows], dtype=float)
    dec = np.asarray(tab["dec"][rows], dtype=float)
    x, y = raw_wcs.all_world2pix(ra, dec, 0)
    out = []
    for sid, row, xx, yy in zip(ids, rows, x, y, strict=False):
        out.append(
            {
                "id": int(sid),
                "tile": str(tab["tile"][row]),
                "segment_id": int(tab["segment-id"][row]),
                "x": float(xx),
                "y": float(yy),
                "x_image": float(tab["x_image"][row]),
                "y_image": float(tab["y_image"][row]),
                "mag_auto_f444w": float(tab["mag_auto_f444w"][row]),
                "seg_area": float(tab["seg_area"][row]),
                "kron2_area": float(tab["kron2_area"][row]),
                "kron2_a": float(tab["kron2_a"][row]),
                "kron2_b": float(tab["kron2_b"][row]),
                "theta_world": float(tab["theta_world"][row]),
            }
        )
    return out


def draw_ellipse(ax, src: dict[str, object], x0: int, y0: int, pixscale: float, color: str, lw: float = 1.0) -> None:
    a = float(src["kron2_a"])
    b = float(src["kron2_b"])
    theta = float(src["theta_world"])
    if not np.isfinite(a + b + theta) or a <= 0 or b <= 0:
        return
    ax.add_patch(
        Ellipse(
            (float(src["x"]) - x0, float(src["y"]) - y0),
            2.0 * a / pixscale,
            2.0 * b / pixscale,
            angle=theta,
            fill=False,
            ec=color,
            lw=lw,
            alpha=0.95,
        )
    )


def mask_edges(mask: np.ndarray) -> np.ndarray:
    if not bool(np.any(mask)):
        return mask
    return mask ^ ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)


def plot_group(name: str, ids: list[int], raw_hdu, raw_wcs: WCS, seg_hdu, seg_wcs: WCS, pixscale: float) -> list[dict[str, object]]:
    sources = load_sources(ids, raw_wcs)
    xs = np.asarray([float(s["x"]) for s in sources])
    ys = np.asarray([float(s["y"]) for s in sources])
    max_axis_pix = max(float(s["kron2_a"]) for s in sources) / pixscale
    size = int(np.ceil(max(768.0, (xs.max() - xs.min()) + (ys.max() - ys.min()) + 2.8 * max_axis_pix)))
    size = min(max(size, 768), 2200)
    x0 = int(round(float(xs.mean()) - size / 2))
    y0 = int(round(float(ys.mean()) - size / 2))
    ny, nx = raw_hdu.shape
    x0 = max(0, min(nx - size, x0))
    y0 = max(0, min(ny - size, y0))
    image = np.asarray(raw_hdu.section[y0 : y0 + size, x0 : x0 + size], dtype=np.float32)
    seg = sample_seg_cutout(raw_wcs, seg_hdu, seg_wcs, x0, y0, size)
    display = stretch(image)

    masks = [seg == int(src["segment_id"]) for src in sources]
    union = np.zeros(seg.shape, dtype=bool)
    for mask in masks:
        union |= mask

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3), dpi=150)
    titles = ["raw + Kron", "segmentation masks", "seg masks + Kron"]
    for ax, title in zip(axes, titles, strict=False):
        ax.imshow(display, origin="lower", cmap="gray", interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.set_axis_off()

    for color, src in zip(COLORS, sources, strict=False):
        draw_ellipse(axes[0], src, x0, y0, pixscale, color, lw=1.1)
    for color, mask, src in zip(COLORS, masks, sources, strict=False):
        rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
        rgb = tuple(int(color[i : i + 2], 16) / 255.0 for i in (1, 3, 5))
        rgba[mask, :3] = rgb
        rgba[mask, 3] = 0.32
        axes[1].imshow(rgba, origin="lower", interpolation="nearest")
        edge = mask_edges(mask)
        axes[1].contour(edge.astype(np.uint8), levels=[0.5], colors=[color], linewidths=0.8, origin="lower")
        axes[1].plot(float(src["x"]) - x0, float(src["y"]) - y0, marker="+", color=color, ms=8, mew=1.2)
        axes[2].imshow(rgba, origin="lower", interpolation="nearest")
        axes[2].contour(edge.astype(np.uint8), levels=[0.5], colors=[color], linewidths=0.8, origin="lower")
        draw_ellipse(axes[2], src, x0, y0, pixscale, color, lw=1.0)

    handles = [
        Patch(facecolor=color, edgecolor=color, alpha=0.35, label=f"id {src['id']} seg {src['segment_id']}")
        for color, src in zip(COLORS, sources, strict=False)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=max(1, len(handles)), fontsize=9, framealpha=0.9)
    fig.suptitle(f"Pointing 19 F444W B3 blended seg masks: {', '.join(map(str, ids))}", fontsize=13)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    out_png = OUT_DIR / f"pointing0019_f444w_{name}_seg_blend.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    rows: list[dict[str, object]] = []
    for i, (src, mask) in enumerate(zip(sources, masks, strict=False)):
        rows.append(
            {
                "group": name,
                "kind": "source",
                "id_a": int(src["id"]),
                "id_b": "",
                "tile": src["tile"],
                "segment_id": int(src["segment_id"]),
                "raw_x": float(src["x"]),
                "raw_y": float(src["y"]),
                "cutout_x0": x0,
                "cutout_y0": y0,
                "map_pixels_in_cutout": int(mask.sum()),
                "catalog_seg_area": float(src["seg_area"]),
                "catalog_kron2_area": float(src["kron2_area"]),
                "mag_auto_f444w": float(src["mag_auto_f444w"]),
                "overlap_pixels": "",
                "touches_after_1pix_dilation": "",
                "nearest_seg_pixel_distance": "",
            }
        )
        for j in range(i + 1, len(sources)):
            other = masks[j]
            overlap = int(np.logical_and(mask, other).sum())
            dil_touch = bool(np.any(ndimage.binary_dilation(mask, structure=np.ones((3, 3), dtype=bool)) & other))
            if bool(mask.any()) and bool(other.any()):
                dist = float(ndimage.distance_transform_edt(~other)[mask].min())
            else:
                dist = math.nan
            rows.append(
                {
                    "group": name,
                    "kind": "pair",
                    "id_a": int(src["id"]),
                    "id_b": int(sources[j]["id"]),
                    "tile": "B3",
                    "segment_id": "",
                    "raw_x": "",
                    "raw_y": "",
                    "cutout_x0": x0,
                    "cutout_y0": y0,
                    "map_pixels_in_cutout": "",
                    "catalog_seg_area": "",
                    "catalog_kron2_area": "",
                    "mag_auto_f444w": "",
                    "overlap_pixels": overlap,
                    "touches_after_1pix_dilation": dil_touch,
                    "nearest_seg_pixel_distance": dist,
                }
            )
    print(out_png)
    return rows


def main() -> None:
    warnings.filterwarnings("ignore", category=FITSFixedWarning)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    with fits.open(RAW, memmap=True) as raw_hdul, fits.open(SEG, memmap=False) as seg_hdul:
        raw_hdu = first_image_hdu(raw_hdul)
        seg_hdu = first_image_hdu(seg_hdul)
        raw_wcs = WCS(raw_hdu.header).celestial
        seg_wcs = WCS(seg_hdu.header).celestial
        pixscale = float(np.mean(proj_plane_pixel_scales(raw_wcs)) * 3600.0)
        for name, ids in GROUPS.items():
            all_rows.extend(plot_group(name, ids, raw_hdu, raw_wcs, seg_hdu, seg_wcs, pixscale))

    out_csv = OUT_DIR / "pointing0019_f444w_selected_seg_blend_stats.csv"
    with out_csv.open("w", newline="") as f:
        fieldnames = [
            "group",
            "kind",
            "id_a",
            "id_b",
            "tile",
            "segment_id",
            "raw_x",
            "raw_y",
            "cutout_x0",
            "cutout_y0",
            "map_pixels_in_cutout",
            "catalog_seg_area",
            "catalog_kron2_area",
            "mag_auto_f444w",
            "overlap_pixels",
            "touches_after_1pix_dilation",
            "nearest_seg_pixel_distance",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(out_csv)


if __name__ == "__main__":
    main()
