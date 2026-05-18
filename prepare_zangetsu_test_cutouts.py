"""Prepare non-overlapping Zangetsu 512x512 HSC test cutouts.

The Zangetsu 4.30 arcmin denoised images are 1536x1536, so they are split into
exactly 3x3 tiles. Reference catalogs are cropped from the HSC-I meas table and
filtered with the same 3-sigma ellipse-area quality cut used by the HSC
training preprocessing. Training/eval code can still choose parent/child rows
with ``--source-filter``.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
from astropy.io import fits
from astropy.table import Table


ZANGETSU_ROOT = Path("/home/chenzunhao/Zangetsu_4.30arcmin")
CATALOG_ROOT = Path("/home/chenzunhao/2026-05-01_scarlet_deblend_demo/catalog_zangetsu")
CATALOG_PATH = CATALOG_ROOT / "HSC-I" / "meas-HSC-I-9813-6,1.fits"
OUTPUT_ROOT = Path("/home/chenzunhao/CELLECT/output/hsc_test")
BANDS = ("HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y", "NB0816", "NB0921", "NB1010")
TILE_SIZE = 512
GRID_SIZE = 3
X_COL = "base_SdssCentroid_x"
Y_COL = "base_SdssCentroid_y"
SHAPE_SOURCE = "sdss"
MAX_AREA_3SIGMA = 400.0


def _parent_origin_from_ltv(header: fits.Header) -> tuple[int, int]:
    if "LTV1" not in header or "LTV2" not in header:
        raise KeyError("Zangetsu FITS header must contain LTV1/LTV2")
    return int(round(-float(header["LTV1"]))), int(round(-float(header["LTV2"])))


def _crop_header(header: fits.Header, *, local_x0: int, local_y0: int, global_x0: int, global_y0: int) -> fits.Header:
    out = header.copy()
    if "CRPIX1" in out:
        out["CRPIX1"] = float(out["CRPIX1"]) - float(local_x0)
    if "CRPIX2" in out:
        out["CRPIX2"] = float(out["CRPIX2"]) - float(local_y0)
    out["LTV1"] = -float(global_x0)
    out["LTV2"] = -float(global_y0)
    return out


def crop_denoised_fits(source: Path, dest: Path, *, local_x0: int, local_y0: int, global_x0: int, global_y0: int) -> None:
    with fits.open(source, memmap=True) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError(f"Expected 2D FITS image for {source}, got {data.shape}")
        crop = data[local_y0 : local_y0 + TILE_SIZE, local_x0 : local_x0 + TILE_SIZE].copy()
        if crop.shape != (TILE_SIZE, TILE_SIZE):
            raise ValueError(f"Crop {source} at ({local_x0},{local_y0}) produced {crop.shape}")
        if not np.isfinite(crop).all():
            finite = np.isfinite(crop)
            fill = float(np.nanmedian(crop[finite])) if finite.any() else 0.0
            crop = np.where(finite, crop, fill).astype(np.float32, copy=False)
        header = _crop_header(hdul[0].header, local_x0=local_x0, local_y0=local_y0, global_x0=global_x0, global_y0=global_y0)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data=crop, header=header).writeto(dest, overwrite=True)


def ellipse_parameters(table: Table, *, shape_source: str = SHAPE_SOURCE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if shape_source == "hsm" and {
        "ext_shapeHSM_HsmSourceMoments_xx",
        "ext_shapeHSM_HsmSourceMoments_yy",
        "ext_shapeHSM_HsmSourceMoments_xy",
    }.issubset(table.colnames):
        xx = np.asarray(table["ext_shapeHSM_HsmSourceMoments_xx"], dtype=np.float32)
        yy = np.asarray(table["ext_shapeHSM_HsmSourceMoments_yy"], dtype=np.float32)
        xy = np.asarray(table["ext_shapeHSM_HsmSourceMoments_xy"], dtype=np.float32)
    elif {"base_SdssShape_xx", "base_SdssShape_yy", "base_SdssShape_xy"}.issubset(table.colnames):
        xx = np.asarray(table["base_SdssShape_xx"], dtype=np.float32)
        yy = np.asarray(table["base_SdssShape_yy"], dtype=np.float32)
        xy = np.asarray(table["base_SdssShape_xy"], dtype=np.float32)
    else:
        n = len(table)
        xx = np.full(n, 4.0, dtype=np.float32)
        yy = np.full(n, 4.0, dtype=np.float32)
        xy = np.zeros(n, dtype=np.float32)

    xx = np.maximum(xx, 0.25)
    yy = np.maximum(yy, 0.25)
    trace = xx + yy
    delta = np.sqrt(np.maximum((xx - yy) ** 2 + 4.0 * xy**2, 0.0))
    major = np.sqrt(np.maximum(0.5 * (trace + delta), 0.25))
    minor = np.sqrt(np.maximum(0.5 * (trace - delta), 0.25))
    angle = 0.5 * np.arctan2(2.0 * xy, xx - yy)
    return major.astype(np.float32), minor.astype(np.float32), angle.astype(np.float32)


def add_ellipse_columns(table: Table, *, shape_source: str = SHAPE_SOURCE) -> Table:
    out = table.copy(copy_data=True)
    major, minor, angle = ellipse_parameters(out, shape_source=shape_source)
    out["ellipse_major_sigma"] = major
    out["ellipse_minor_sigma"] = minor
    out["ellipse_theta"] = angle
    out["ellipse_area_3sigma"] = (9.0 * math.pi * major * minor).astype(np.float32)
    return out


def crop_catalog(table: Table, *, tile_name: str, global_x0: int, global_y0: int) -> Table:
    x = np.asarray(table[X_COL], dtype=np.float64)
    y = np.asarray(table[Y_COL], dtype=np.float64)
    area = np.asarray(table["ellipse_area_3sigma"], dtype=np.float32)
    inside = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(area)
        & (area <= MAX_AREA_3SIGMA)
        & (x >= global_x0)
        & (x < global_x0 + TILE_SIZE)
        & (y >= global_y0)
        & (y < global_y0 + TILE_SIZE)
    )
    out = table[inside].copy(copy_data=True)
    out["centroid_local_x"] = np.asarray(out[X_COL], dtype=np.float32) - float(global_x0)
    out["centroid_local_y"] = np.asarray(out[Y_COL], dtype=np.float32) - float(global_y0)
    out.meta["TILE"] = tile_name
    out.meta["X0"] = int(global_x0)
    out.meta["Y0"] = int(global_y0)
    out.meta["SHAPESRC"] = SHAPE_SOURCE
    out.meta["MAXA3SIG"] = float(MAX_AREA_3SIGMA)
    return out


def count_spatial_rows(table: Table, *, global_x0: int, global_y0: int) -> int:
    x = np.asarray(table[X_COL], dtype=np.float64)
    y = np.asarray(table[Y_COL], dtype=np.float64)
    inside = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= global_x0)
        & (x < global_x0 + TILE_SIZE)
        & (y >= global_y0)
        & (y < global_y0 + TILE_SIZE)
    )
    return int(inside.sum())


def write_catalog(table: Table, fits_path: Path, csv_path: Path) -> None:
    fits_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.write(fits_path, format="fits", overwrite=True)
    one_dim = Table()
    for name in table.colnames:
        arr = np.asarray(table[name])
        if arr.ndim <= 1:
            one_dim[name] = table[name]
        else:
            one_dim[name] = [json.dumps(np.asarray(row).tolist(), separators=(",", ":")) for row in arr]
    one_dim.write(csv_path, format="ascii.csv", overwrite=True)


def main() -> None:
    first_image = ZANGETSU_ROOT / "HSC-I" / "denoised.fits"
    with fits.open(first_image, memmap=True) as hdul:
        shape = tuple(hdul[0].data.shape)
        parent_x0, parent_y0 = _parent_origin_from_ltv(hdul[0].header)
    if shape != (TILE_SIZE * GRID_SIZE, TILE_SIZE * GRID_SIZE):
        raise ValueError(f"Expected Zangetsu image shape {(TILE_SIZE * GRID_SIZE, TILE_SIZE * GRID_SIZE)}, got {shape}")

    catalog = add_ellipse_columns(Table.read(CATALOG_PATH, hdu=1), shape_source=SHAPE_SOURCE)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, object]] = []
    cutout_paths: Dict[str, Dict[str, str]] = {}

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            local_x0 = col * TILE_SIZE
            local_y0 = row * TILE_SIZE
            global_x0 = parent_x0 + local_x0
            global_y0 = parent_y0 + local_y0
            tile_name = f"zangetsu_r{row:02d}_c{col:02d}_x{global_x0}_y{global_y0}"

            band_paths: Dict[str, str] = {}
            for band in BANDS:
                source = ZANGETSU_ROOT / band / "denoised.fits"
                if not source.exists():
                    continue
                dest = OUTPUT_ROOT / "cutouts" / tile_name / band / "denoised.fits"
                crop_denoised_fits(
                    source,
                    dest,
                    local_x0=local_x0,
                    local_y0=local_y0,
                    global_x0=global_x0,
                    global_y0=global_y0,
                )
                band_paths[band] = str(dest)

            spatial_rows = count_spatial_rows(catalog, global_x0=global_x0, global_y0=global_y0)
            tile_catalog = crop_catalog(catalog, tile_name=tile_name, global_x0=global_x0, global_y0=global_y0)
            ref_fits = OUTPUT_ROOT / "reference_catalogs" / f"{tile_name}_meas.fits"
            ref_csv = OUTPUT_ROOT / "reference_catalogs_csv" / f"{tile_name}_meas.csv"
            write_catalog(tile_catalog, ref_fits, ref_csv)

            nchild0 = int((np.asarray(tile_catalog["deblend_nChild"], dtype=np.int64) == 0).sum()) if "deblend_nChild" in tile_catalog.colnames else None
            parent0 = int((np.asarray(tile_catalog["parent"], dtype=np.int64) == 0).sum()) if "parent" in tile_catalog.colnames else None
            manifest_rows.append(
                {
                    "name": tile_name,
                    "row": row,
                    "col": col,
                    "x0": global_x0,
                    "y0": global_y0,
                    "x1": global_x0 + TILE_SIZE,
                    "y1": global_y0 + TILE_SIZE,
                    "size": TILE_SIZE,
                    "spatial_catalog_rows": spatial_rows,
                    "catalog_rows": len(tile_catalog),
                    "area_filtered_rows": spatial_rows - len(tile_catalog),
                    "nchild0_rows": nchild0,
                    "parent0_rows": parent0,
                }
            )
            cutout_paths[tile_name] = band_paths

    with (OUTPUT_ROOT / "tiles.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    metadata = {
        "zangetsu_root": str(ZANGETSU_ROOT),
        "catalog_path": str(CATALOG_PATH),
        "output_root": str(OUTPUT_ROOT),
        "bands": list(BANDS),
        "tile_size": TILE_SIZE,
        "grid_size": GRID_SIZE,
        "parent_origin": [parent_x0, parent_y0],
        "num_tiles": len(manifest_rows),
        "shape_source": SHAPE_SOURCE,
        "max_area_3sigma": MAX_AREA_3SIGMA,
        "note": "Reference catalogs are spatial crops filtered by finite 3-sigma ellipse area <= max_area_3sigma; use --source-filter nchild0 during eval/training for leaf-source GT.",
    }
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (OUTPUT_ROOT / "cutout_paths.json").write_text(json.dumps(cutout_paths, indent=2), encoding="utf-8")

    total_rows = sum(int(row["catalog_rows"]) for row in manifest_rows)
    total_spatial = sum(int(row["spatial_catalog_rows"]) for row in manifest_rows)
    total_area_filtered = sum(int(row["area_filtered_rows"]) for row in manifest_rows)
    total_nchild0 = sum(int(row["nchild0_rows"] or 0) for row in manifest_rows)
    print(f"wrote {len(manifest_rows)} Zangetsu test tiles to {OUTPUT_ROOT}")
    print(
        f"catalog rows across tiles={total_rows}, spatial rows before area filter={total_spatial}, "
        f"area-filtered rows={total_area_filtered}, deblend_nChild==0 rows={total_nchild0}"
    )


if __name__ == "__main__":
    main()
