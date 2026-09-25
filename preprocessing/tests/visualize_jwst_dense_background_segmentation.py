#!/usr/bin/env python3
"""Visualize JWST dense-target background and isolated segmentation components."""

from __future__ import annotations

import csv
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import ZScaleInterval
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.wcs import FITSFixedWarning
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse
from scipy import ndimage


RAW_BASE = Path("/data/shared/jwst_foundation/raw/COSMOS_1727_1837_5893")
CATALOG = Path("/data/shared/jwst_foundation/catalog/COSMOS_1727_1837_5893/COSMOSWeb_mastercatalog_v1.1.fits")
SEG_DIR = Path("/data/shared/jwst_foundation/catalog/COSMOS_1727_1837_5893/segmentation_maps")
STAR_DIR = Path("/data/shared/jwst_foundation/catalog/COSMOS_1727_1837_5893/star_masks")
OUT_DIR = Path("/home/czh23/analysis/2026-09/2026-09-06/jwst_dense_background_segmentation")

POINTING = "0019"
BAND = "f444w"
SIZE = 4096
DS = 2
DILATE_PIX = 8
CHUNK_ROWS = 64
MIN_SEG_AREA = 8
SEG_TILE_MULT = 1_000_000


def raw_fits_path(pointing: str, band: str) -> Path:
    return RAW_BASE / f"Pointing_{pointing}" / f"COSMOS_pointing_{pointing}_{band.upper()}_detector_p001.fits"


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
    return np.clip((np.nan_to_num(cut, nan=lo, posinf=hi, neginf=lo) - lo) / (hi - lo), 0.0, 1.0).astype(
        np.float32
    )


def disk(radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def candidate_maps(paths: list[Path], raw_wcs: WCS, x0: int, y0: int) -> list[Path]:
    xs = np.linspace(x0, x0 + SIZE - 1, 5)
    ys = np.linspace(y0, y0 + SIZE - 1, 5)
    xx, yy = np.meshgrid(xs, ys)
    ra, dec = raw_wcs.all_pix2world(xx.ravel(), yy.ravel(), 0)
    out: list[Path] = []
    for path in paths:
        with fits.open(path, memmap=True, do_not_scale_image_data=True) as hdul:
            hdu = first_image_hdu(hdul)
            wcs = WCS(hdu.header).celestial
            ny, nx = hdu.shape
        mx, my = wcs.all_world2pix(ra, dec, 0)
        inside = np.isfinite(mx) & np.isfinite(my) & (mx >= 0) & (mx < nx) & (my >= 0) & (my < ny)
        if bool(np.any(inside)):
            out.append(path)
    return out


def tile_name(path: Path) -> str:
    match = re.search(r"_(A\d+|B\d+)(?:_|\\.)", path.name)
    if not match:
        match = re.search(r"jwst_(A\d+|B\d+)\\.fits", path.name)
    if not match:
        raise ValueError(f"cannot parse COSMOS-Web tile name from {path.name}")
    return match.group(1)


def tile_code(name: str) -> int:
    arm = 0 if name[0].upper() == "A" else 1
    return arm * 100 + int(name[1:])


def sample_map_full(paths: list[Path], raw_wcs: WCS, x0: int, y0: int, *, kind: str) -> np.ndarray:
    if kind == "seg":
        canvas = np.zeros((SIZE, SIZE), dtype=np.int32)
    else:
        canvas = np.zeros((SIZE, SIZE), dtype=bool)
    if not paths:
        return canvas
    x_grid = x0 + np.arange(SIZE, dtype=float)
    for path in paths:
        with fits.open(path, memmap=False, do_not_scale_image_data=False) as hdul:
            hdu = first_image_hdu(hdul)
            wcs = WCS(hdu.header).celestial
            ny, nx = hdu.shape
            for y_start in range(0, SIZE, CHUNK_ROWS):
                y_end = min(SIZE, y_start + CHUNK_ROWS)
                y_grid = y0 + np.arange(y_start, y_end, dtype=float)
                xx, yy = np.meshgrid(x_grid, y_grid)
                ra, dec = raw_wcs.all_pix2world(xx, yy, 0)
                mx, my = wcs.all_world2pix(ra, dec, 0)
                xi = np.rint(mx).astype(np.int64)
                yi = np.rint(my).astype(np.int64)
                valid = np.isfinite(mx) & np.isfinite(my) & (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
                if not bool(np.any(valid)):
                    continue
                xmin, xmax = int(xi[valid].min()), int(xi[valid].max())
                ymin, ymax = int(yi[valid].min()), int(yi[valid].max())
                section = np.asarray(hdu.section[ymin : ymax + 1, xmin : xmax + 1])
                values = np.zeros(valid.shape, dtype=section.dtype)
                values[valid] = section[yi[valid] - ymin, xi[valid] - xmin]
                if kind == "seg":
                    put = values > 0
                    canvas[y_start:y_end][put] = values[put].astype(np.int32)
                else:
                    canvas[y_start:y_end] |= values > 0
    return canvas


def local_affine_raw_to_map(raw_wcs: WCS, map_wcs: WCS, x0: int, y0: int) -> np.ndarray:
    """Fit a local affine transform from raw pixels to map pixels for one cutout."""

    xs = np.linspace(x0, x0 + SIZE - 1, 5)
    ys = np.linspace(y0, y0 + SIZE - 1, 5)
    xx, yy = np.meshgrid(xs, ys)
    ra, dec = raw_wcs.all_pix2world(xx.ravel(), yy.ravel(), 0)
    mx, my = map_wcs.all_world2pix(ra, dec, 0)
    mat = np.column_stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
    ok = np.isfinite(mx) & np.isfinite(my)
    if int(ok.sum()) < 3:
        raise ValueError("not enough finite WCS samples for affine fit")
    coeff_x, *_ = np.linalg.lstsq(mat[ok], mx[ok], rcond=None)
    coeff_y, *_ = np.linalg.lstsq(mat[ok], my[ok], rcond=None)
    return np.vstack([coeff_x, coeff_y])


def sample_map_full_affine(paths: list[Path], raw_wcs: WCS, x0: int, y0: int, *, kind: str) -> np.ndarray:
    if kind == "seg":
        canvas = np.zeros((SIZE, SIZE), dtype=np.int32)
    else:
        canvas = np.zeros((SIZE, SIZE), dtype=bool)
    if not paths:
        return canvas
    x_grid = x0 + np.arange(SIZE, dtype=float)
    for path in paths:
        seg_offset = tile_code(tile_name(path)) * SEG_TILE_MULT if kind == "seg" else 0
        with fits.open(path, memmap=False, do_not_scale_image_data=False) as hdul:
            hdu = first_image_hdu(hdul)
            wcs = WCS(hdu.header).celestial
            ny, nx = hdu.shape
            affine = local_affine_raw_to_map(raw_wcs, wcs, x0, y0)
            for y_start in range(0, SIZE, CHUNK_ROWS):
                y_end = min(SIZE, y_start + CHUNK_ROWS)
                y_grid = y0 + np.arange(y_start, y_end, dtype=float)
                xx, yy = np.meshgrid(x_grid, y_grid)
                mx = affine[0, 0] * xx + affine[0, 1] * yy + affine[0, 2]
                my = affine[1, 0] * xx + affine[1, 1] * yy + affine[1, 2]
                xi = np.rint(mx).astype(np.int64)
                yi = np.rint(my).astype(np.int64)
                valid = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
                if not bool(np.any(valid)):
                    continue
                xmin, xmax = int(xi[valid].min()), int(xi[valid].max())
                ymin, ymax = int(yi[valid].min()), int(yi[valid].max())
                section = np.asarray(hdu.section[ymin : ymax + 1, xmin : xmax + 1])
                values = np.zeros(valid.shape, dtype=section.dtype)
                values[valid] = section[yi[valid] - ymin, xi[valid] - xmin]
                if kind == "seg":
                    put = values > 0
                    canvas[y_start:y_end][put] = (seg_offset + values[put].astype(np.int32)).astype(np.int32)
                else:
                    canvas[y_start:y_end] |= values > 0
    return canvas


def downsample_bool(mask: np.ndarray) -> np.ndarray:
    h = mask.shape[0] // DS
    w = mask.shape[1] // DS
    return mask[: h * DS, : w * DS].reshape(h, DS, w, DS).max(axis=(1, 3))


def segmentation_edges(mask: np.ndarray) -> np.ndarray:
    edge = np.zeros(mask.shape, dtype=bool)
    edge[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    edge[1:, :] |= mask[1:, :] != mask[:-1, :]
    return edge & mask


def load_catalog(raw_wcs: WCS) -> dict[str, np.ndarray]:
    tab = Table.read(CATALOG, hdu=1)
    ra = np.asarray(tab["ra"], dtype=float)
    dec = np.asarray(tab["dec"], dtype=float)
    x, y = raw_wcs.all_world2pix(ra, dec, 0)
    return {
        "id": np.asarray(tab["id"], dtype=np.int64),
        "segment_id": np.asarray(tab["segment-id"], dtype=np.int64),
        "tile": np.asarray(tab["tile"]).astype(str),
        "ra": ra,
        "dec": dec,
        "x": np.asarray(x, dtype=float),
        "y": np.asarray(y, dtype=float),
        "kron2_a": np.asarray(tab["kron2_a"], dtype=float),
        "kron2_b": np.asarray(tab["kron2_b"], dtype=float),
        "theta_world": np.asarray(tab["theta_world"], dtype=float),
        "mag_auto": np.asarray(tab[f"mag_auto_{BAND}"], dtype=float),
    }


def draw_ellipse(ax, x: float, y: float, a_arcsec: float, b_arcsec: float, theta: float, pixscale: float, color: str) -> None:
    if not np.isfinite(x + y + a_arcsec + b_arcsec + theta) or a_arcsec <= 0 or b_arcsec <= 0:
        return
    ax.add_patch(
        Ellipse(
            (x / DS, y / DS),
            2.0 * a_arcsec / pixscale / DS,
            2.0 * b_arcsec / pixscale / DS,
            angle=theta,
            fill=False,
            ec=color,
            lw=0.6,
            alpha=0.9,
            zorder=5,
        )
    )


def region_sources(catalog: dict[str, np.ndarray], x0: int, y0: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = catalog["x"] - float(x0)
    y = catalog["y"] - float(y0)
    keep = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < SIZE) & (y >= 0) & (y < SIZE)
    idx = np.flatnonzero(keep)
    return idx, x[idx], y[idx]


def isolated_seg_components(
    seg: np.ndarray,
    star_for_bg: np.ndarray,
    catalog: dict[str, np.ndarray],
    x0: int,
    y0: int,
    *,
    structure: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    isolated_mask = np.zeros(seg.shape, dtype=bool)
    rows: list[dict[str, float | int]] = []
    seg_ids = np.unique(seg)
    seg_ids = seg_ids[seg_ids > 0]
    id_to_row: dict[int, int] = {}
    for row, (tile, seg_id) in enumerate(zip(catalog["tile"], catalog["segment_id"], strict=False)):
        code = tile_code(str(tile)) * SEG_TILE_MULT + int(seg_id)
        id_to_row[code] = int(row)
    for seg_id in seg_ids:
        row = id_to_row.get(int(seg_id), -1)
        if row < 0:
            continue
        yy, xx = np.nonzero(seg == int(seg_id))
        area = int(yy.size)
        if area < MIN_SEG_AREA:
            continue
        ymin, ymax = int(yy.min()), int(yy.max()) + 1
        xmin, xmax = int(xx.min()), int(xx.max()) + 1
        if ymin <= 0 or xmin <= 0 or ymax >= SIZE or xmax >= SIZE:
            continue
        y0b = max(0, ymin - DILATE_PIX)
        y1b = min(SIZE, ymax + DILATE_PIX)
        x0b = max(0, xmin - DILATE_PIX)
        x1b = min(SIZE, xmax + DILATE_PIX)
        sub = seg[y0b:y1b, x0b:x1b]
        comp = sub == int(seg_id)
        expanded = ndimage.binary_dilation(comp, structure=structure)
        if bool(np.any(expanded & (sub > 0) & (sub != int(seg_id)))):
            continue
        if bool(np.any(expanded & star_for_bg[y0b:y1b, x0b:x1b])):
            continue
        sx = float(catalog["x"][row] - float(x0))
        sy = float(catalog["y"][row] - float(y0))
        if not (0 <= sx < SIZE and 0 <= sy < SIZE):
            continue
        isolated_mask[y0b:y1b, x0b:x1b] |= expanded & comp
        rows.append(
            {
                "source_id": int(catalog["id"][row]),
                "tile": str(catalog["tile"][row]),
                "segment_id": int(catalog["segment_id"][row]),
                "area_pix": area,
                "x": sx,
                "y": sy,
                "mag_auto": float(catalog["mag_auto"][row]),
                "kron2_a": float(catalog["kron2_a"][row]),
                "kron2_b": float(catalog["kron2_b"][row]),
                "theta_world": float(catalog["theta_world"][row]),
            }
        )
    return isolated_mask, rows


def main() -> None:
    warnings.filterwarnings("ignore", category=FITSFixedWarning)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = raw_fits_path(POINTING, BAND)
    seg_paths_all = sorted(SEG_DIR.glob("*_segmap_*.fits"))
    star_paths_all = sorted(STAR_DIR.glob("*.fits"))
    structure = disk(DILATE_PIX)

    with fits.open(raw_path, memmap=True) as hdul:
        hdu = first_image_hdu(hdul)
        data = hdu.data
        raw_wcs = WCS(hdu.header).celestial
        pixscale = float(np.mean(proj_plane_pixel_scales(raw_wcs)) * 3600.0)
        ny, nx = data.shape
        regions = {
            "left_top_4096": (0, ny - SIZE),
            "center_4096": (nx // 2 - SIZE // 2, ny // 2 - SIZE // 2),
            "top_row_col3_4096": (2 * SIZE, ny - SIZE),
        }
        catalog = load_catalog(raw_wcs)

        bg_fig, bg_axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
        iso_fig, iso_axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
        summary_rows: list[dict[str, object]] = []
        isolated_rows: list[dict[str, object]] = []

        for bg_ax, iso_ax, (name, (x0, y0)) in zip(bg_axes, iso_axes, regions.items()):
            cut = np.asarray(data[y0 : y0 + SIZE, x0 : x0 + SIZE], dtype=np.float32)
            display = stretch(cut[::DS, ::DS])
            seg_candidates = candidate_maps(seg_paths_all, raw_wcs, x0, y0)
            star_candidates = candidate_maps(star_paths_all, raw_wcs, x0, y0)
            print(f"[region] {name}: seg_maps={len(seg_candidates)} star_masks={len(star_candidates)}", flush=True)
            seg = sample_map_full_affine(seg_candidates, raw_wcs, x0, y0, kind="seg")
            star_raw = sample_map_full_affine(star_candidates, raw_wcs, x0, y0, kind="star")
            star_for_bg = ndimage.binary_fill_holes(star_raw)
            seg_dil = ndimage.binary_dilation(seg > 0, structure=structure)
            star_dil = ndimage.binary_dilation(star_for_bg, structure=structure)
            background = ~(seg_dil | star_dil) & np.isfinite(cut)

            idx, sx, sy = region_sources(catalog, x0, y0)
            xi = np.rint(sx).astype(np.int64)
            yi = np.rint(sy).astype(np.int64)
            center_in_bg = np.zeros(len(idx), dtype=bool)
            center_valid = (xi >= 0) & (xi < SIZE) & (yi >= 0) & (yi < SIZE)
            center_in_bg[center_valid] = background[yi[center_valid], xi[center_valid]]

            bg_ax.imshow(display, origin="lower", cmap="gray", interpolation="nearest")
            bg_ax.imshow(
                np.ma.masked_where(~downsample_bool(background), downsample_bool(background)),
                origin="lower",
                cmap=ListedColormap(["#00d060"]),
                alpha=0.24,
                interpolation="nearest",
            )
            bg_ax.imshow(
                np.ma.masked_where(~downsample_bool(seg_dil | star_dil), downsample_bool(seg_dil | star_dil)),
                origin="lower",
                cmap=ListedColormap(["#ff3030"]),
                alpha=0.14,
                interpolation="nearest",
            )
            bg_ax.scatter(sx[center_in_bg] / DS, sy[center_in_bg] / DS, s=8, c="#ff00ff", marker="+", linewidths=0.45)
            bg_ax.set_title(f"{name}: bg centers {int(center_in_bg.sum())}", fontsize=11)
            bg_ax.set_xlim(0, SIZE // DS)
            bg_ax.set_ylim(0, SIZE // DS)
            bg_ax.set_axis_off()

            iso_mask, iso_rows = isolated_seg_components(seg, star_for_bg, catalog, x0, y0, structure=structure)
            iso_ax.imshow(display, origin="lower", cmap="gray", interpolation="nearest")
            iso_ax.imshow(
                np.ma.masked_where(~downsample_bool(iso_mask), downsample_bool(iso_mask)),
                origin="lower",
                cmap=ListedColormap(["#00ffff"]),
                alpha=0.28,
                interpolation="nearest",
            )
            iso_edge = downsample_bool(segmentation_edges(iso_mask))
            iso_ax.imshow(
                np.ma.masked_where(~iso_edge, iso_edge),
                origin="lower",
                cmap=ListedColormap(["#00ffff"]),
                alpha=0.85,
                interpolation="nearest",
            )
            for row in iso_rows:
                draw_ellipse(
                    iso_ax,
                    float(row["x"]),
                    float(row["y"]),
                    float(row["kron2_a"]),
                    float(row["kron2_b"]),
                    float(row["theta_world"]),
                    pixscale,
                    "#ffd400",
                )
                row_out = {"region": name, **row}
                isolated_rows.append(row_out)
            iso_ax.set_title(f"{name}: isolated seg {len(iso_rows)}", fontsize=11)
            iso_ax.set_xlim(0, SIZE // DS)
            iso_ax.set_ylim(0, SIZE // DS)
            iso_ax.set_axis_off()

            summary_rows.append(
                {
                    "region": name,
                    "x0": x0,
                    "y0": y0,
                    "background_pixels": int(background.sum()),
                    "background_fraction": float(background.mean()),
                    "seg_pixels": int((seg > 0).sum()),
                    "star_raw_pixels": int(star_raw.sum()),
                    "star_hole_filled_pixels": int(star_for_bg.sum()),
                    "excluded_dilated_pixels": int((seg_dil | star_dil).sum()),
                    "catalog_sources": int(len(idx)),
                    "source_centers_in_background": int(center_in_bg.sum()),
                    "isolated_seg_components": int(len(iso_rows)),
                    "seg_maps": ",".join(p.name for p in seg_candidates),
                    "star_masks": ",".join(p.name for p in star_candidates),
                }
            )

        handles = [
            plt.Line2D([0], [0], color="#00d060", lw=5, alpha=0.35, label="initial background"),
            plt.Line2D([0], [0], color="#ff3030", lw=5, alpha=0.22, label="seg/star excluded after 8 pix dilation"),
            plt.Line2D([0], [0], marker="+", color="#ff00ff", lw=0, label="catalog center in background"),
        ]
        bg_fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10, framealpha=0.9)
        bg_fig.suptitle(f"Pointing {POINTING} {BAND.upper()}: initial background from segmap and JWST star mask", fontsize=13)
        bg_fig.tight_layout(rect=(0, 0.06, 1, 0.95))
        bg_png = OUT_DIR / f"pointing{POINTING}_{BAND}_three_blocks_initial_background_centers.png"
        bg_fig.savefig(bg_png, bbox_inches="tight")
        plt.close(bg_fig)

        handles = [
            plt.Line2D([0], [0], color="#00ffff", lw=5, alpha=0.4, label="isolated segmentation component"),
            plt.Line2D([0], [0], color="#ffd400", lw=1.4, label="catalog Kron shape"),
        ]
        iso_fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10, framealpha=0.9)
        iso_fig.suptitle(
            f"Pointing {POINTING} {BAND.upper()}: isolated seg components with matched Kron shapes", fontsize=13
        )
        iso_fig.tight_layout(rect=(0, 0.06, 1, 0.95))
        iso_png = OUT_DIR / f"pointing{POINTING}_{BAND}_three_blocks_isolated_seg_kron.png"
        iso_fig.savefig(iso_png, bbox_inches="tight")
        plt.close(iso_fig)

    with (OUT_DIR / f"pointing{POINTING}_{BAND}_background_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (OUT_DIR / f"pointing{POINTING}_{BAND}_isolated_seg_components.csv").open("w", newline="") as f:
        fieldnames = [
            "region",
            "source_id",
            "tile",
            "segment_id",
            "area_pix",
            "x",
            "y",
            "mag_auto",
            "kron2_a",
            "kron2_b",
            "theta_world",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(isolated_rows)
    readme = OUT_DIR / "README.txt"
    readme.write_text(
        "\n".join(
            [
                "JWST dense-target background / segmentation diagnostic.",
                f"Raw image: {raw_path}",
                f"Catalog: {CATALOG}",
                "Initial background = finite pixels outside (segmentation map > 0 dilated by 8 pix) and outside",
                "(JWST star mask after binary_fill_holes, dilated by 8 pix).",
                "The hole-filled star mask is used only for the background exclusion so sources carved out of",
                "star-mask artifacts are not accidentally counted as background.",
                "Isolated segmentation components require: catalog id match, no contact with another seg id within",
                "8 pixels, no overlap with hole-filled JWST star mask within 8 pixels, not touching the cutout edge.",
            ]
        )
        + "\n"
    )
    print(bg_png)
    print(iso_png)


if __name__ == "__main__":
    main()
