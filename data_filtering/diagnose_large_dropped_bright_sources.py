#!/usr/bin/env python3
"""Diagnose dropped bright clusters in large log+lupton bright components."""

from __future__ import annotations

import argparse
import csv
import math
import os
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cellect")

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from astropy.units import UnitsWarning
from astropy.wcs import WCS

import build_external_bright_labels as external


warnings.filterwarnings("ignore", category=UnitsWarning)


FLAG_NAMES = (
    "base_SdssCentroid_flag",
    "base_SdssCentroid_flag_edge",
    "base_SdssCentroid_flag_notAtMaximum",
    "base_SdssShape_flag",
    "base_SdssShape_flag_shift",
    "base_SdssShape_flag_maxIter",
    "base_PixelFlags_flag",
    "base_PixelFlags_flag_edge",
    "base_PixelFlags_flag_saturated",
    "base_PixelFlags_flag_saturatedCenter",
    "base_PixelFlags_flag_bad",
    "base_PixelFlags_flag_bright_object",
    "base_PixelFlags_flag_bright_objectCenter",
    "ext_photometryKron_KronFlux_flag",
    "ext_photometryKron_KronFlux_flag_edge",
    "ext_photometryKron_KronFlux_flag_bad_radius",
    "ext_photometryKron_KronFlux_flag_bad_shape",
    "ext_photometryKron_KronFlux_flag_bad_shape_no_psf",
    "ext_photometryKron_KronFlux_flag_used_minimum_radius",
    "ext_photometryKron_KronFlux_flag_used_psf_radius",
    "ext_photometryKron_KronFlux_flag_small_radius",
    "deblend_tooManyPeaks",
    "deblend_parentTooBig",
    "deblend_masked",
    "deblend_skipped",
    "deblend_hasStrayFlux",
    "modelfit_CModel_flag",
    "modelfit_CModel_flag_region_maxBadPixelFraction",
    "modelfit_CModel_flag_badCentroid",
    "base_ClassificationExtendedness_flag",
)

MASK_PLANES = ("BRIGHT_OBJECT", "SAT", "BAD", "NO_DATA", "EDGE", "UNMASKEDNAN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/data/shared/Subaru"))
    parser.add_argument("--refit-root", type=Path, default=Path("/data/czh23/refit"))
    parser.add_argument("--external-root", type=Path, default=Path("output/data_filter_0728/external_bright_labels"))
    parser.add_argument("--gaia-fits", type=Path, default=Path("output/gaia_dr3_cosmos.fits"))
    parser.add_argument("--tract", default="9813")
    parser.add_argument("--patch", default="4,5")
    parser.add_argument("--bands", nargs="+", default=["HSC-I", "HSC-Y"])
    parser.add_argument("--component-area-min", type=int, default=1000)
    parser.add_argument("--small-source-area", type=float, default=1000.0)
    parser.add_argument("--shape-area-max", type=float, default=10000.0)
    parser.add_argument("--axis-ratio-max", type=float, default=5.0)
    parser.add_argument("--neighbor-radius", type=float, default=50.0)
    parser.add_argument("--out-dir", type=Path, default=Path("output/data_filter_0728/external_bright_labels/9813/4,5"))
    return parser.parse_args()


def flag_map(table: Table) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, value in table.meta.items():
        if not key.startswith("TFLAG"):
            continue
        try:
            index = int(key[5:]) - 1
        except ValueError:
            continue
        out[str(value)] = index
    return out


def bool_flag(table: Table, fmap: dict[str, int], row_index: int, name: str) -> bool | None:
    if name not in fmap:
        return None
    flags = np.asarray(table["flags"][row_index], dtype=bool)
    index = fmap[name]
    if index < 0 or index >= flags.shape[0]:
        return None
    return bool(flags[index])


def true_flag_names(table: Table, fmap: dict[str, int], row_index: int, names: tuple[str, ...]) -> list[str]:
    return [name for name in names if bool_flag(table, fmap, row_index, name) is True]


def mask_bits(header: fits.Header) -> dict[str, int]:
    return {name: int(header[f"MP_{name}"]) for name in MASK_PLANES if f"MP_{name}" in header}


def center_mask_names(mask: np.ndarray, bits: dict[str, int], x: float, y: float) -> list[str]:
    xi = int(round(x))
    yi = int(round(y))
    if not (0 <= xi < mask.shape[1] and 0 <= yi < mask.shape[0]):
        return ["OUT_OF_BOUNDS"]
    value = int(mask[yi, xi])
    return [name for name, bit in bits.items() if value & (1 << int(bit))]


def safe_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def read_csv_dict(path: Path, key: str | None = None) -> dict[str, dict[str, str]] | list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if key is None:
        return rows
    return {str(row[key]): row for row in rows}


def source_categories(
    *,
    row: dict[str, str],
    refit: dict[str, str] | None,
    table: Table,
    fmap: dict[str, int],
    row_index: int,
    center_masks: list[str],
    nearest_gaia_arcsec: float,
    nearby_dropped_count: int,
    nearby_supervised_count: int,
) -> list[str]:
    categories: list[str] = []
    area = safe_float(row["area"])
    axis_ratio = safe_float(row["axis_ratio"])
    if area < 1000.0:
        categories.append("small_bright_region_ignore_ok")
    if nearest_gaia_arcsec > 3.0:
        categories.append("no_gaia_within_3arcsec")
    if nearest_gaia_arcsec > 10.0:
        categories.append("no_gaia_within_10arcsec")
    if center_masks:
        categories.append("center_on_mask:" + "|".join(center_masks))
    if area >= 10000.0:
        categories.append("shape_area_ge_10000")
    if axis_ratio > 5.0:
        categories.append("axis_ratio_gt_5")
    if bool_flag(table, fmap, row_index, "base_SdssShape_flag"):
        categories.append("base_SdssShape_flag")
    if bool_flag(table, fmap, row_index, "base_SdssCentroid_flag"):
        categories.append("base_SdssCentroid_flag")
    if bool_flag(table, fmap, row_index, "detect_isPrimary") is False:
        categories.append("not_detect_isPrimary")
    parent = int(safe_float(table["parent"][row_index], 0.0)) if "parent" in table.colnames else 0
    n_child = int(safe_float(table["deblend_nChild"][row_index], 0.0)) if "deblend_nChild" in table.colnames else 0
    if parent != 0:
        categories.append("child_source_parent_nonzero")
    if n_child > 0:
        categories.append("parent_source_has_children")
    if nearby_dropped_count > 1:
        categories.append("nearby_dropped_sources_within_50px")
    if nearby_supervised_count > 0:
        categories.append("nearby_supervised_sources_within_50px")
    if refit is not None:
        if refit.get("measurement_surface", "").startswith("direct_footprint"):
            categories.append("direct_footprint_fallback")
        if refit.get("heavyfp_nan0_good", "").lower() == "false":
            categories.append("heavyfp_nan0_not_good")
        if refit.get("proxy_nan0_good", "").lower() == "false":
            categories.append("proxy_nan0_not_good")
        if refit.get("proxy_nan0_fallback_large_aperture", "").lower() == "true":
            categories.append("proxy_fallback_large_aperture")
        ap_count = safe_float(refit.get("aperture_pixel_count", "nan"))
        if math.isfinite(ap_count) and area > 0 and ap_count / area < 0.3 and area > 500:
            categories.append("aperture_pixel_fraction_lt_0p3")
    true_flags = true_flag_names(table, fmap, row_index, FLAG_NAMES)
    if true_flags:
        categories.append("has_bad_quality_flags")
    if not categories:
        categories.append("no_obvious_catalog_flag_or_mask")
    return categories


def process_band(args: argparse.Namespace, band: str, gaia_coords: SkyCoord, gaia_table: Table) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    image, mask, image_header, mask_header = external.read_exposure(
        external.image_path(args.data_root, args.tract, band, args.patch)
    )
    bright_mask, component_labels = external.build_bright_components(
        image,
        log_a=300.0,
        log_high_percentile=99.5,
        lupton_stretch=0.5,
        lupton_q=20.0,
        threshold=2.99,
        dilation=2,
    )
    component_area = np.bincount(component_labels.ravel())
    bits = mask_bits(mask_header)
    wcs = WCS(image_header)
    table = Table.read(external.meas_path(args.data_root, args.tract, band, args.patch), hdu=1)
    fmap = flag_map(table)
    refit_rows = read_csv_dict(external.refit_path(args.refit_root, args.tract, band, args.patch), key="row_index")
    external_rows = read_csv_dict(
        args.external_root / args.tract / args.patch / band / f"{args.tract}_{args.patch.replace(',', '_')}_{band}_bright_reclassification.csv"
    )
    all_rows = list(external_rows)
    dropped = [row for row in all_rows if row["final_label"] == "dropped_bright_cluster"]
    supervised = [row for row in all_rows if row["final_label"] in {"clean", "center_only", "strict_center_only"}]
    details: list[dict[str, object]] = []
    for row in dropped:
        comp = int(row["component_id"])
        comp_area = int(component_area[comp]) if comp > 0 and comp < len(component_area) else 0
        if comp_area <= int(args.component_area_min):
            continue
        x = safe_float(row["x"])
        y = safe_float(row["y"])
        row_index = int(safe_float(row["row_index"], -1.0))
        refit = refit_rows.get(str(row_index))
        sky = wcs.pixel_to_world(x, y)
        sep = SkyCoord(ra=sky.ra.deg * u.deg, dec=sky.dec.deg * u.deg).separation(gaia_coords).arcsec
        nearest_idx = int(np.argmin(sep))
        nearest_gaia_arcsec = float(sep[nearest_idx])
        center_masks = center_mask_names(mask, bits, x, y)
        nearby_dropped_count = sum(
            math.hypot(x - safe_float(other["x"]), y - safe_float(other["y"])) <= float(args.neighbor_radius)
            for other in dropped
        )
        nearby_supervised_count = sum(
            math.hypot(x - safe_float(other["x"]), y - safe_float(other["y"])) <= float(args.neighbor_radius)
            for other in supervised
        )
        flags = true_flag_names(table, fmap, row_index, FLAG_NAMES)
        parent = int(safe_float(table["parent"][row_index], 0.0)) if "parent" in table.colnames else 0
        n_child = int(safe_float(table["deblend_nChild"][row_index], 0.0)) if "deblend_nChild" in table.colnames else 0
        categories = source_categories(
            row=row,
            refit=refit,
            table=table,
            fmap=fmap,
            row_index=row_index,
            center_masks=center_masks,
            nearest_gaia_arcsec=nearest_gaia_arcsec,
            nearby_dropped_count=nearby_dropped_count,
            nearby_supervised_count=nearby_supervised_count,
        )
        details.append(
            {
                "band": band,
                "source_id": row["source_id"],
                "row_index": row_index,
                "x": x,
                "y": y,
                "mag": safe_float(row["mag"]),
                "area": safe_float(row["area"]),
                "axis_ratio": safe_float(row["axis_ratio"]),
                "component_id": comp,
                "component_area": comp_area,
                "cluster_id": row["cluster_id"],
                "cluster_size": row["cluster_size"],
                "existing_label": row["existing_label"],
                "final_label": row["final_label"],
                "reason": row["reason"],
                "parent": parent,
                "deblend_nChild": n_child,
                "class": row["class"],
                "center_mask": "|".join(center_masks),
                "true_flags": "|".join(flags),
                "nearest_gaia_source_id": int(gaia_table["source_id"][nearest_idx]),
                "nearest_gaia_g_mag": float(gaia_table["phot_g_mean_mag"][nearest_idx]),
                "nearest_gaia_arcsec": nearest_gaia_arcsec,
                "nearby_dropped_count_50px": nearby_dropped_count,
                "nearby_supervised_count_50px": nearby_supervised_count,
                "measurement_surface": refit.get("measurement_surface", "") if refit else "",
                "footprint_area": safe_float(refit.get("footprint_area", "nan")) if refit else float("nan"),
                "aperture_pixel_count": safe_float(refit.get("aperture_pixel_count", "nan")) if refit else float("nan"),
                "aperture_pixel_fraction": (
                    safe_float(refit.get("aperture_pixel_count", "nan")) / safe_float(row["area"])
                    if refit and safe_float(row["area"]) > 0
                    else float("nan")
                ),
                "heavy_value_count": safe_float(refit.get("heavy_value_count", "nan")) if refit else float("nan"),
                "finite_value_count": safe_float(refit.get("finite_value_count", "nan")) if refit else float("nan"),
                "proxy_nan0_good": refit.get("proxy_nan0_good", "") if refit else "",
                "heavyfp_nan0_good": refit.get("heavyfp_nan0_good", "") if refit else "",
                "proxy_nan0_fallback_large_aperture": refit.get("proxy_nan0_fallback_large_aperture", "") if refit else "",
                "diagnostic_categories": ";".join(categories),
            }
        )
    summary_counter: Counter[str] = Counter()
    for row in details:
        summary_counter.update(str(row["diagnostic_categories"]).split(";"))
    summary = [
        {"band": band, "category": key, "count": count}
        for key, count in summary_counter.most_common()
    ]
    return details, summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    gaia = Table.read(args.gaia_fits)
    gaia_coords = SkyCoord(
        ra=np.asarray(gaia["ra"], dtype=float) * u.deg,
        dec=np.asarray(gaia["dec"], dtype=float) * u.deg,
    )
    all_details: list[dict[str, object]] = []
    all_summary: list[dict[str, object]] = []
    for band in args.bands:
        details, summary = process_band(args, band, gaia_coords, gaia)
        all_details.extend(details)
        all_summary.extend(summary)
        print(f"{band}: large-component dropped sources={len(details)}")
        for row in summary[:12]:
            print(f"  {row['category']}: {row['count']}")
    prefix = f"{args.tract}_{args.patch.replace(',', '_')}"
    write_csv(args.out_dir / f"{prefix}_large_component_dropped_diagnostics.csv", all_details)
    write_csv(args.out_dir / f"{prefix}_large_component_dropped_summary.csv", all_summary)
    print(f"wrote {args.out_dir / f'{prefix}_large_component_dropped_diagnostics.csv'}")
    print(f"wrote {args.out_dir / f'{prefix}_large_component_dropped_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
