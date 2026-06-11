#!/usr/bin/env python
"""Export DS9 region files from preprocessed PU clean source catalogs.

The preprocessing pipeline writes the final training clean catalogs to
<root>/<tract>/<patch>/band_reference_catalogs/<band>/meas-<band>-<tract>-<patch>.fits.
This script exports those final catalogs, so the REG files match the data that
training actually sees.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from astropy.table import Table
from astropy.io import fits


CATALOG_DIRS = {
    "clean": "band_reference_catalogs",
    "center_only": "band_reference_center_only",
    "ordinary_ignore": "band_reference_ignore",
    "strict_center_only": "band_reference_strict_center_only",
    "strict_ignore": "band_reference_strict_ignore",
    "pu_all": "band_reference_pu_all",
}

CLASS_COLORS = {
    "clean": "green",
    "center_only": "yellow",
    "ordinary_ignore": "red",
    "strict_center_only": "magenta",
    "strict_ignore": "magenta",
    "pu_all": "cyan",
}


def _split_words(value: Optional[str]) -> list[str]:
    if value is None:
        return []
    return [item for item in re.split(r"[\s;]+", value.strip()) if item]


def _read_table(path: Path) -> Table:
    if not path.exists():
        raise FileNotFoundError(path)
    return Table.read(path)


def _find_column(table: Table, candidates: Iterable[str], *, role: str) -> str:
    for name in candidates:
        if name in table.colnames:
            return name
    raise KeyError(f"Could not find {role} column; tried {', '.join(candidates)}")


def _ensure_ellipse_columns(table: Table, *, shape_source: str) -> Table:
    needed = {"ellipse_major_sigma", "ellipse_minor_sigma", "ellipse_theta"}
    if needed.issubset(table.colnames):
        return table
    from astro_data_preprocessing import add_ellipse_columns

    return add_ellipse_columns(table, shape_source=shape_source)


def _catalog_path(root: Path, tract: str, patch: str, band: str, class_name: str) -> Path:
    rel_dir = CATALOG_DIRS[class_name]
    path = root / tract / patch / rel_dir / band / f"meas-{band}-{tract}-{patch}.fits"
    if class_name == "strict_center_only" and not path.exists():
        return root / tract / patch / CATALOG_DIRS["strict_ignore"] / band / f"meas-{band}-{tract}-{patch}.fits"
    return path

def _calexp_path(root: Path, tract: str, patch: str, band: str) -> Path:
    return root / tract / band / patch / f"calexp-{band}-{tract}-{patch}.fits"


def _meas_catalog_path(root: Path, tract: str, patch: str, band: str) -> Path:
    return root / tract / band / patch / f"meas-{band}-{tract}-{patch}.fits"


def _patch_origin_from_calexp(path: Path) -> tuple[float, float]:
    """Return full-pixel origin of a patch calexp.

    HSC calexp images store the patch-local image with LTV offsets.  Source
    catalogs use full-pixel tract coordinates, while DS9 image regions for a
    patch calexp need patch-local coordinates.  CRVAL1A/CRVAL2A are alternate
    WCS values and are not reliable patch origins.
    """

    if not path.exists():
        print(f"[WARNING] Calexp file not found; assuming patch origin (0,0): {path}", flush=True)
        return 0.0, 0.0
    with fits.open(path, memmap=True, lazy_load_hdus=True, ignore_missing_end=True) as hdulist:
        for hdu in hdulist:
            header = hdu.header
            if int(header.get("NAXIS", 0)) < 2:
                continue
            if "LTV1" in header and "LTV2" in header:
                x0 = -float(header["LTV1"])
                y0 = -float(header["LTV2"])
                print(f"Found patch origin from LTV for {path}: x0={x0:g}, y0={y0:g}", flush=True)
                return x0, y0
    print(f"[WARNING] No LTV1/LTV2 found; assuming patch origin (0,0): {path}", flush=True)
    return 0.0, 0.0


def _parse_tile_origin(tile_name: Optional[str]) -> tuple[Optional[float], Optional[float]]:
    if not tile_name:
        return None, None
    match = re.search(r"_x(-?\d+(?:\.\d+)?)_y(-?\d+(?:\.\d+)?)", tile_name)
    if not match:
        raise ValueError(f"Could not parse x/y origin from tile name: {tile_name}")
    print(f"Parsed tile origin from tile name {tile_name}: x0={match.group(1)}, y0={match.group(2)}", flush=True)
    return float(match.group(1)), float(match.group(2))


def _crop_mask(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x0: Optional[float],
    y0: Optional[float],
    width: Optional[float],
    height: Optional[float],
    margin: float,
    crop: bool = False,
) -> np.ndarray:
    valid = np.isfinite(x) & np.isfinite(y)
    if x0 is None or y0 is None:
        return valid
    if width is None or height is None:
        raise ValueError("--crop-width/--crop-height or --crop-size is required when using crop coordinates")
    return (
        valid
        & (x >= float(x0) - margin)
        & (x < float(x0) + float(width) + margin)
        & (y >= float(y0) - margin)
        & (y < float(y0) + float(height) + margin)
    )


def _region_line(
    *,
    x: float,
    y: float,
    major: float,
    minor: float,
    theta: float,
    color: str,
    width: int,
    point_size: int,
    max_ellipse_area_as_point: float,
) -> str:
    area = math.pi * major * minor
    if (
        not np.isfinite(major)
        or not np.isfinite(minor)
        or not np.isfinite(theta)
        or major <= 0
        or minor <= 0
        or area > max_ellipse_area_as_point
    ):
        return f"point({x + 1.0:.3f},{y + 1.0:.3f}) # point=circle {point_size} color={color} width={width}\n"
    return (
        f"ellipse({x + 1.0:.3f},{y + 1.0:.3f},{major:.3f},{minor:.3f},"
        f"{math.degrees(theta):.3f}) # color={color} width={width}\n"
    )


def _write_reg(
    path: Path,
    rows: list[dict[str, object]],
    *,
    point_size: int,
    max_ellipse_area_as_point: float,
    coordinate_system: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Region file format: DS9 version 4.1\n")
        handle.write(
            'global color=green dashlist=8 3 width=2 font="helvetica 12 normal roman" '
            "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n"
        )
        handle.write(f"{coordinate_system}\n")
        for row in rows:
            handle.write(
                _region_line(
                    x=float(row["x"]),
                    y=float(row["y"]),
                    major=float(row["major"]),
                    minor=float(row["minor"]),
                    theta=float(row["theta"]),
                    color=str(row["color"]),
                    width=int(row["width"]),
                    point_size=point_size,
                    max_ellipse_area_as_point=max_ellipse_area_as_point,
                )
            )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "tract",
        "patch",
        "band",
        "class",
        "source_id",
        "x",
        "y",
        "patch_x",
        "patch_y",
        "major",
        "minor",
        "theta_rad",
        "theta_deg",
        "ellipse_area",
        "mag",
        "pu_reason",
        "ap2_flux",
        "kron_flux",
        "ap2_mag",
        "kron_mag",
        "ap2_minus_kron_mag",
        "ap2_kron_filter",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _rows_from_table(
    table: Table,
    *,
    tract: str,
    patch: str,
    band: str,
    class_name: str,
    patch_origin_x: float,
    patch_origin_y: float,
    crop_x0: Optional[float],
    crop_y0: Optional[float],
    width: Optional[float],
    height: Optional[float],
    margin: float,
    local_coordinates: bool,
    region_coordinates: str,
    shape_source: str,
    reg_width: int,
) -> list[dict[str, object]]:
    if class_name == "clean" and "pu_refit_kron_radius_matched" in table.colnames:
        matched = np.asarray(table["pu_refit_kron_radius_matched"], dtype=bool)
        missing = int(np.count_nonzero(~matched))
        if missing:
            print(
                f"WARNING: {tract}/{patch} {band} clean catalog contains {missing}/{len(table)} rows "
                "without matched kron-refit radius. With the current preprocessing defaults these rows "
                "would be ordinary_ignore, so this catalog is likely stale and should be regenerated.",
                flush=True,
            )
    table = _ensure_ellipse_columns(table, shape_source=shape_source)
    x_col = _find_column(table, ("base_SdssCentroid_x", "base_SdssShape_x", "slot_Centroid_x", "x"), role="x")
    y_col = _find_column(table, ("base_SdssCentroid_y", "base_SdssShape_y", "slot_Centroid_y", "y"), role="y")
    x = np.asarray(table[x_col], dtype=np.float64)
    y = np.asarray(table[y_col], dtype=np.float64)
    mask = _crop_mask(x, y, x0=crop_x0, y0=crop_y0, width=width, height=height, margin=margin)
    major = np.asarray(table["ellipse_major_sigma"], dtype=np.float64)
    minor = np.asarray(table["ellipse_minor_sigma"], dtype=np.float64)
    theta = np.asarray(table["ellipse_theta"], dtype=np.float64)
    source_id = np.asarray(table["id"]) if "id" in table.colnames else np.arange(len(table))
    mag = np.asarray(table["pu_mag"], dtype=np.float64) if "pu_mag" in table.colnames else None
    reason = np.asarray(table["pu_reason"], dtype=str) if "pu_reason" in table.colnames else None

    rows: list[dict[str, object]] = []
    for idx in np.flatnonzero(mask):
        full_x = float(x[idx])
        full_y = float(y[idx])
        patch_x = full_x - float(patch_origin_x)
        patch_y = full_y - float(patch_origin_y)
        if region_coordinates == "physical":
            rx = full_x
            ry = full_y
        elif local_coordinates and crop_x0 is not None and crop_y0 is not None:
            rx = full_x - float(crop_x0)
            ry = full_y - float(crop_y0)
        else:
            rx = patch_x
            ry = patch_y
        row_major = float(major[idx])
        row_minor = float(minor[idx])
        row_theta = float(theta[idx])
        rows.append(
            {
                "tract": tract,
                "patch": patch,
                "band": band,
                "class": class_name,
                "source_id": int(source_id[idx]) if np.isfinite(source_id[idx]) else "",
                "x": rx,
                "y": ry,
                "patch_x": patch_x,
                "patch_y": patch_y,
                "major": row_major,
                "minor": row_minor,
                "theta": row_theta,
                "theta_rad": row_theta,
                "theta_deg": math.degrees(row_theta) if np.isfinite(row_theta) else "",
                "ellipse_area": math.pi * row_major * row_minor if np.isfinite(row_major * row_minor) else "",
                "mag": float(mag[idx]) if mag is not None and np.isfinite(mag[idx]) else "",
                "pu_reason": str(reason[idx]) if reason is not None else "",
                "color": CLASS_COLORS.get(class_name, "green"),
                "width": reg_width,
            }
        )
    rows.sort(key=lambda item: float(item["ellipse_area"]) if item["ellipse_area"] != "" else 0.0, reverse=True)
    return rows


def _format_threshold(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _safe_mag(flux: float, zeropoint: float) -> float:
    if not np.isfinite(flux) or flux <= 0.0:
        return float("nan")
    return float(zeropoint) - 2.5 * math.log10(float(flux))


def _load_ap2_kron_photometry(
    *,
    path: Path,
    ids: Iterable[int],
    ap2_flux_column: str,
    kron_flux_column: str,
    zeropoint: float,
) -> dict[int, dict[str, float]]:
    wanted = set(int(value) for value in ids if value != "")
    if not wanted:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"meas catalog for ap2/kron filtering not found: {path}")
    with fits.open(path, memmap=True, lazy_load_hdus=True, ignore_missing_end=True) as hdul:
        table = hdul[1].data
        names = set(table.columns.names)
        for column in ("id", ap2_flux_column, kron_flux_column):
            if column not in names:
                raise KeyError(f"{path} missing required column for ap2/kron filtering: {column}")
        ids_array = np.asarray(table["id"], dtype=np.int64)
        mask = np.isin(ids_array, np.asarray(list(wanted), dtype=np.int64))
        out: dict[int, dict[str, float]] = {}
        for source_id, ap2_flux, kron_flux in zip(
            ids_array[mask],
            table[ap2_flux_column][mask],
            table[kron_flux_column][mask],
        ):
            ap2_mag = _safe_mag(float(ap2_flux), zeropoint)
            kron_mag = _safe_mag(float(kron_flux), zeropoint)
            diff = ap2_mag - kron_mag if np.isfinite(ap2_mag) and np.isfinite(kron_mag) else float("nan")
            out[int(source_id)] = {
                "ap2_flux": float(ap2_flux),
                "kron_flux": float(kron_flux),
                "ap2_mag": ap2_mag,
                "kron_mag": kron_mag,
                "ap2_minus_kron_mag": diff,
            }
        return out


def _apply_ap2_kron_filter(
    rows: list[dict[str, object]],
    *,
    photometry: dict[int, dict[str, float]],
    abs_max: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    kept: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    invalid: list[dict[str, object]] = []
    for row in rows:
        item = dict(row)
        try:
            source_id = int(item["source_id"])
        except Exception:
            source_id = -1
        phot = photometry.get(source_id)
        if phot is None:
            item["ap2_kron_filter"] = "invalid_missing_source_id"
            invalid.append(item)
            continue
        for key, value in phot.items():
            item[key] = value if np.isfinite(value) else ""
        diff = phot["ap2_minus_kron_mag"]
        if not np.isfinite(diff):
            item["ap2_kron_filter"] = "invalid_flux"
            invalid.append(item)
        elif abs(float(diff)) <= float(abs_max):
            item["ap2_kron_filter"] = "kept"
            kept.append(item)
        else:
            item["ap2_kron_filter"] = "rejected_abs_gt_threshold"
            rejected.append(item)
    return kept, rejected, invalid


def export_one(args: argparse.Namespace, *, patch: str, band: str) -> dict[str, object]:
    calexp_path = _calexp_path(Path(args.data), str(args.tract), patch, band)
    patch_origin_x, patch_origin_y = _patch_origin_from_calexp(calexp_path)

    crop_local_x0, crop_local_y0 = _parse_tile_origin(args.tile_name)
    if args.crop_x0 is not None:
        crop_local_x0 = float(args.crop_x0)
    if args.crop_y0 is not None:
        crop_local_y0 = float(args.crop_y0)
    if crop_local_x0 is not None and crop_local_y0 is not None:
        if args.crop_coordinates == "full":
            crop_x0 = float(crop_local_x0)
            crop_y0 = float(crop_local_y0)
            crop_label_x0 = crop_x0
            crop_label_y0 = crop_y0
        else:
            crop_x0 = float(crop_local_x0) + float(patch_origin_x)
            crop_y0 = float(crop_local_y0) + float(patch_origin_y)
            crop_label_x0 = float(crop_local_x0)
            crop_label_y0 = float(crop_local_y0)
    else:
        crop_x0 = None
        crop_y0 = None
        crop_label_x0 = None
        crop_label_y0 = None
    width = float(args.crop_size) if args.crop_size is not None else args.crop_width
    height = float(args.crop_size) if args.crop_size is not None else args.crop_height

    patch_label = patch.replace(",", "_")
    crop_suffix = ""
    if crop_label_x0 is not None and crop_label_y0 is not None:
        crop_suffix = f"_x{int(crop_label_x0)}_y{int(crop_label_y0)}"
        if width is not None and height is not None:
            crop_suffix += f"_w{int(width)}_h{int(height)}"

    output_dir = Path(args.output_dir) / str(args.tract) / patch / band
    all_rows: list[dict[str, object]] = []
    class_counts: dict[str, int] = {}
    for class_name in args.classes:
        path = _catalog_path(Path(args.root), str(args.tract), patch, band, class_name)
        if not path.exists():
            if args.allow_missing_classes:
                class_counts[class_name] = 0
                continue
            raise FileNotFoundError(f"{class_name} catalog not found: {path}")
        table = _read_table(path)
        if len(table) == 0:
            print(f"WARNING: {patch} {band} {class_name} catalog is empty: {path}", flush=True)
        rows = _rows_from_table(
            table,
            tract=str(args.tract),
            patch=patch,
            band=band,
            class_name=class_name,
            patch_origin_x=patch_origin_x,
            patch_origin_y=patch_origin_y,
            crop_x0=crop_x0,
            crop_y0=crop_y0,
            width=width,
            height=height,
            margin=float(args.crop_margin),
            local_coordinates=bool(args.local_coordinates),
            region_coordinates=str(args.region_coordinates),
            shape_source=str(args.shape_source),
            reg_width=int(args.reg_width),
        )
        class_counts[class_name] = len(rows)
        if len(table) and not rows:
            print(
                f"WARNING: {patch} {band} {class_name} catalog has {len(table)} rows but export selected 0. "
                "Check crop coordinates and --crop-coordinates.",
                flush=True,
            )
        all_rows.extend(rows)
        if args.write_class_files:
            stem = f"{str(args.tract)}_{patch_label}_{band}_{class_name}{crop_suffix}"
            _write_reg(
                output_dir / f"{stem}.reg",
                rows,
                point_size=int(args.point_size),
                max_ellipse_area_as_point=float(args.max_ellipse_area_as_point),
                coordinate_system=str(args.region_coordinates),
            )
            _write_csv(output_dir / f"{stem}.csv", rows)

    combined_stem = f"{str(args.tract)}_{patch_label}_{band}_{'_'.join(args.classes)}{crop_suffix}"
    reg_path = output_dir / f"{combined_stem}.reg"
    csv_path = output_dir / f"{combined_stem}.csv"
    _write_reg(
        reg_path,
        all_rows,
        point_size=int(args.point_size),
        max_ellipse_area_as_point=float(args.max_ellipse_area_as_point),
        coordinate_system=str(args.region_coordinates),
    )
    _write_csv(csv_path, all_rows)
    ap2_summary: dict[str, object] = {}
    if args.ap2_kron_abs_max is not None:
        threshold = float(args.ap2_kron_abs_max)
        photometry = _load_ap2_kron_photometry(
            path=_meas_catalog_path(Path(args.data), str(args.tract), patch, band),
            ids=[int(row["source_id"]) for row in all_rows if row.get("source_id") != ""],
            ap2_flux_column=str(args.ap2_flux_column),
            kron_flux_column=str(args.kron_flux_column),
            zeropoint=float(args.photometry_zeropoint),
        )
        kept, rejected, invalid = _apply_ap2_kron_filter(all_rows, photometry=photometry, abs_max=threshold)
        suffix = f"_abs_ap2_minus_kron_le{_format_threshold(threshold)}"
        kept_reg_path = output_dir / f"{combined_stem}{suffix}.reg"
        kept_csv_path = output_dir / f"{combined_stem}{suffix}.csv"
        rejected_reg_path = output_dir / f"{combined_stem}_abs_ap2_minus_kron_gt{_format_threshold(threshold)}.reg"
        rejected_csv_path = output_dir / f"{combined_stem}_abs_ap2_minus_kron_gt{_format_threshold(threshold)}.csv"
        invalid_reg_path = output_dir / f"{combined_stem}_ap2_kron_invalid.reg"
        invalid_csv_path = output_dir / f"{combined_stem}_ap2_kron_invalid.csv"
        for out_reg, out_csv, out_rows in (
            (kept_reg_path, kept_csv_path, kept),
            (rejected_reg_path, rejected_csv_path, rejected),
            (invalid_reg_path, invalid_csv_path, invalid),
        ):
            _write_reg(
                out_reg,
                out_rows,
                point_size=int(args.point_size),
                max_ellipse_area_as_point=float(args.max_ellipse_area_as_point),
                coordinate_system=str(args.region_coordinates),
            )
            _write_csv(out_csv, out_rows)
        ap2_summary = {
            "ap2_kron_abs_max": threshold,
            "ap2_kron_kept_rows": len(kept),
            "ap2_kron_rejected_rows": len(rejected),
            "ap2_kron_invalid_rows": len(invalid),
            "ap2_kron_kept_reg": str(kept_reg_path),
            "ap2_kron_kept_csv": str(kept_csv_path),
            "ap2_flux_column": str(args.ap2_flux_column),
            "kron_flux_column": str(args.kron_flux_column),
        }
        print(
            f"ap2/kron filter {patch} {band}: kept={len(kept)} rejected={len(rejected)} "
            f"invalid={len(invalid)} threshold={threshold:g}",
            flush=True,
        )
    return {
        "patch": patch,
        "band": band,
        "rows": len(all_rows),
        "class_counts": class_counts,
        "reg": str(reg_path),
        "csv": str(csv_path),
        **ap2_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Root containing original catalog data")
    parser.add_argument("--root", type=Path, required=True, help="Preprocessed root containing <tract>/<patch>.")
    parser.add_argument("--tract", default="9813")
    parser.add_argument("--patches", nargs="+", required=True)
    parser.add_argument("--bands", nargs="+", required=True)
    parser.add_argument("--classes", nargs="+", default=["clean"], choices=sorted(CATALOG_DIRS))
    parser.add_argument("--output-dir", type=Path, default=Path("output/preprocessed_clean_regions"))
    parser.add_argument("--shape-source", default="kron", choices=("kron", "sdss", "hsm", "circular_kron"))
    parser.add_argument("--tile-name", default=None, help="Optional tile name containing _x<X>_y<Y>.")
    parser.add_argument("--crop-x0", type=float, default=None)
    parser.add_argument("--crop-y0", type=float, default=None)
    parser.add_argument("--crop-size", type=float, default=None)
    parser.add_argument("--crop-width", type=float, default=None)
    parser.add_argument("--crop-height", type=float, default=None)
    parser.add_argument(
        "--crop-coordinates",
        choices=("local", "full"),
        default="local",
        help="Interpret --crop-x0/--crop-y0 and --tile-name origins as patch-local or full-pixel coordinates.",
    )
    parser.add_argument("--crop-margin", type=float, default=0.0)
    parser.add_argument("--local-coordinates", action="store_true", help="Write x/y relative to crop origin.")
    parser.add_argument("--reg-width", type=int, default=2)
    parser.add_argument(
        "--region-coordinates",
        choices=("physical", "image"),
        default="physical",
        help=(
            "Coordinate system written in the DS9 REG file. physical writes catalog full-pixel coordinates, "
            "which is usually what DS9 expects for HSC patch calexp files. image writes patch-local pixels."
        ),
    )
    parser.add_argument(
        "--ap2-kron-abs-max",
        type=float,
        default=None,
        help=(
            "Optional extra photometric cleanup. When set, write additional REG/CSV files that keep only "
            "rows with abs(ap2_mag-kron_mag) <= this value, plus rejected and invalid files."
        ),
    )
    parser.add_argument(
        "--ap2-flux-column",
        default="base_CircularApertureFlux_6_0_instFlux",
        help="Fixed-aperture flux column used for ap2/kron cleanup. HSC 2 arcsec aperture is radius 6 pix.",
    )
    parser.add_argument("--kron-flux-column", default="ext_photometryKron_KronFlux_instFlux")
    parser.add_argument("--photometry-zeropoint", type=float, default=27.0)
    parser.add_argument("--point-size", type=int, default=8)
    parser.add_argument("--max-ellipse-area-as-point", type=float, default=40000.0)
    parser.add_argument("--write-class-files", action="store_true", help="Also write one REG/CSV per class.")
    parser.add_argument("--allow-missing-classes", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    patches = [patch for value in args.patches for patch in _split_words(value)]
    bands = [band for value in args.bands for band in _split_words(value)]
    summaries = []
    for patch in patches:
        for band in bands:
            summary = export_one(args, patch=patch, band=band)
            summaries.append(summary)
            print(
                f"wrote {summary['rows']} rows for {summary['patch']} {summary['band']}: "
                f"{summary['reg']}",
                flush=True,
            )

    summary_path = Path(args.output_dir) / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = (
            "patch",
            "band",
            "rows",
            "class_counts",
            "reg",
            "csv",
            "ap2_kron_abs_max",
            "ap2_kron_kept_rows",
            "ap2_kron_rejected_rows",
            "ap2_kron_invalid_rows",
            "ap2_kron_kept_reg",
            "ap2_kron_kept_csv",
            "ap2_flux_column",
            "kron_flux_column",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    print(f"summary written to {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
