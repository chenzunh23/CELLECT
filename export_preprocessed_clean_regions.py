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
    "strict_ignore": "band_reference_strict_ignore",
    "pu_all": "band_reference_pu_all",
}

CLASS_COLORS = {
    "clean": "green",
    "center_only": "yellow",
    "ordinary_ignore": "red",
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
    return root / tract / patch / rel_dir / band / f"meas-{band}-{tract}-{patch}.fits"

def _calexp_path(root: Path, tract: str, patch: str, band: str) -> Path:
    return root / tract / band / patch / f"calexp-{band}-{tract}-{patch}.fits"

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
    if x0 is None or y0 is None or crop is False:
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


def _write_reg(path: Path, rows: list[dict[str, object]], *, point_size: int, max_ellipse_area_as_point: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Region file format: DS9 version 4.1\n")
        handle.write(
            'global color=green dashlist=8 3 width=2 font="helvetica 12 normal roman" '
            "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n"
        )
        handle.write("image\n")
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
    x0: Optional[float],
    y0: Optional[float],
    width: Optional[float],
    height: Optional[float],
    margin: float,
    local_coordinates: bool,
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
    mask = _crop_mask(x, y, x0=x0, y0=y0, width=width, height=height, margin=margin, crop=local_coordinates)
    major = np.asarray(table["ellipse_major_sigma"], dtype=np.float64)
    minor = np.asarray(table["ellipse_minor_sigma"], dtype=np.float64)
    theta = np.asarray(table["ellipse_theta"], dtype=np.float64)
    source_id = np.asarray(table["id"]) if "id" in table.colnames else np.arange(len(table))
    mag = np.asarray(table["pu_mag"], dtype=np.float64) if "pu_mag" in table.colnames else None
    reason = np.asarray(table["pu_reason"], dtype=str) if "pu_reason" in table.colnames else None

    rows: list[dict[str, object]] = []
    for idx in np.flatnonzero(mask):
        px = float(x[idx])
        py = float(y[idx])
        # Subtract tile origin
        px = px - float(x0) if x0 is not None else px
        py = py - float(y0) if y0 is not None else py
        rx = px - float(x0) if local_coordinates and x0 is not None else px
        ry = py - float(y0) if local_coordinates and y0 is not None else py
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
                "patch_x": px,
                "patch_y": py,
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


def export_one(args: argparse.Namespace, *, patch: str, band: str) -> dict[str, object]:
    calexp_path = _calexp_path(Path(args.data), str(args.tract), patch, band)

    # Read table header using fits
    if not calexp_path.exists():
        print(f'[WARNING] Calexp file not found for {args.tract}/{patch} {band}: {calexp_path}', flush=True)
        calexp_header = {}
    else:
        with fits.open(calexp_path) as hdulist:
            calexp_header = hdulist[1].header
    x_patch, y_patch = 0, 0
    if "CRVAL1A" in calexp_header and "CRVAL2A" in calexp_header:
        x_patch, y_patch = calexp_header["CRVAL1A"], calexp_header["CRVAL2A"]
        print(
            f"Found WCS reference in calexp header for {args.tract}/{patch} {band}: "
            f"CRVAL1A={calexp_header['CRVAL1A']}, CRVAL2A={calexp_header['CRVAL2A']}",
            flush=True,
        )
    else:
        print(f'[WARNING] No WCS reference found in calexp header for {args.tract}/{patch} {band}, setting origin to (0,0)', flush=True)
    x0, y0 = _parse_tile_origin(args.tile_name)
    if args.crop_x0 is not None:
        x0 = float(args.crop_x0)
    if args.crop_y0 is not None:
        y0 = float(args.crop_y0)
    x0 = x0 + x_patch if x0 is not None else x_patch
    y0 = y0 + y_patch if y0 is not None else y_patch
    width = float(args.crop_size) if args.crop_size is not None else args.crop_width
    height = float(args.crop_size) if args.crop_size is not None else args.crop_height

    patch_label = patch.replace(",", "_")
    crop_suffix = ""
    if x0 is not None and y0 is not None:
        crop_suffix = f"_x{int(x0)}_y{int(y0)}"
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
        rows = _rows_from_table(
            _read_table(path),
            tract=str(args.tract),
            patch=patch,
            band=band,
            class_name=class_name,
            x0=x0,
            y0=y0,
            width=width,
            height=height,
            margin=float(args.crop_margin),
            local_coordinates=bool(args.local_coordinates),
            shape_source=str(args.shape_source),
            reg_width=int(args.reg_width),
        )
        class_counts[class_name] = len(rows)
        all_rows.extend(rows)
        if args.write_class_files:
            stem = f"{str(args.tract)}_{patch_label}_{band}_{class_name}{crop_suffix}"
            _write_reg(
                output_dir / f"{stem}.reg",
                rows,
                point_size=int(args.point_size),
                max_ellipse_area_as_point=float(args.max_ellipse_area_as_point),
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
    )
    _write_csv(csv_path, all_rows)
    return {
        "patch": patch,
        "band": band,
        "rows": len(all_rows),
        "class_counts": class_counts,
        "reg": str(reg_path),
        "csv": str(csv_path),
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
    parser.add_argument("--crop-margin", type=float, default=0.0)
    parser.add_argument("--local-coordinates", action="store_true", help="Write x/y relative to crop origin.")
    parser.add_argument("--reg-width", type=int, default=2)
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
        writer = csv.DictWriter(handle, fieldnames=("patch", "band", "rows", "class_counts", "reg", "csv"))
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)
    print(f"summary written to {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
