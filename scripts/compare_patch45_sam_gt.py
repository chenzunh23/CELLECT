#!/usr/bin/env python
"""Compare old Zangetsu sam tile GT against current patch 4,5 preprocessing."""

from __future__ import annotations

import argparse
import csv
import math
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
from astropy.io.fits.verify import VerifyWarning
from astropy.table import Table
from astropy.units import UnitsWarning

warnings.filterwarnings("ignore", category=UnitsWarning)
warnings.filterwarnings("ignore", category=VerifyWarning)

CLASS_DIRS = {
    "clean": "band_reference_catalogs",
    "center_only": "band_reference_center_only",
    "strict_center_only": "band_reference_strict_center_only",
    "ignore": "band_reference_ignore",
    "rejected": "band_reference_rejected",
    "dropped": "band_reference_dropped",
    "pu_all": "band_reference_pu_all",
}
GT_CLASSES = ("clean", "center_only", "strict_center_only")


def first_column(table: Table, names: Iterable[str], default: float = np.nan) -> np.ndarray:
    out = np.full(len(table), float(default), dtype=np.float64)
    filled = np.zeros(len(table), dtype=bool)
    for name in names:
        if name not in table.colnames:
            continue
        values = np.asarray(table[name], dtype=np.float64)
        take = ~filled & np.isfinite(values)
        out[take] = values[take]
        filled[take] = True
    return out


def table_xy(table: Table) -> tuple[np.ndarray, np.ndarray]:
    x = first_column(table, ("base_SdssCentroid_x", "base_SdssShape_x", "slot_Centroid_x", "x"))
    y = first_column(table, ("base_SdssCentroid_y", "base_SdssShape_y", "slot_Centroid_y", "y"))
    return x, y


def ellipse_from_table(table: Table) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xx = first_column(table, ("base_SdssShape_xx", "ext_shapeHSM_HsmSourceMoments_xx"))
    yy = first_column(table, ("base_SdssShape_yy", "ext_shapeHSM_HsmSourceMoments_yy"))
    xy = first_column(table, ("base_SdssShape_xy", "ext_shapeHSM_HsmSourceMoments_xy"), default=0.0)
    kron = first_column(
        table,
        ("pu_refit_kron_radius", "ext_photometryKron_KronFlux_radius", "ext_photometryKron_KronFlux_radius_for_radius"),
    )
    xx = np.maximum(xx, 0.25)
    yy = np.maximum(yy, 0.25)
    trace = xx + yy
    delta = np.sqrt(np.maximum((xx - yy) ** 2 + 4.0 * xy**2, 0.0))
    sdss_major = np.sqrt(np.maximum(0.5 * (trace + delta), 0.25))
    sdss_minor = np.sqrt(np.maximum(0.5 * (trace - delta), 0.25))
    det = np.sqrt(np.maximum(sdss_major * sdss_minor, 0.0))
    valid = np.isfinite(det) & (det > 0) & np.isfinite(kron) & (kron > 0)
    major = np.full(len(table), np.nan, dtype=np.float64)
    minor = np.full(len(table), np.nan, dtype=np.float64)
    theta = np.full(len(table), np.nan, dtype=np.float64)
    scale = np.zeros(len(table), dtype=np.float64)
    scale[valid] = kron[valid] / det[valid]
    major[valid] = sdss_major[valid] * scale[valid]
    minor[valid] = sdss_minor[valid] * scale[valid]
    theta[valid] = 0.5 * np.arctan2(2.0 * xy[valid], xx[valid] - yy[valid])
    return major, minor, theta


def catalog_path(root: Path, tract: str, patch: str, band: str, class_name: str) -> Path:
    return root / tract / patch / CLASS_DIRS[class_name] / band / f"meas-{band}-{tract}-{patch}.fits"


def read_class(root: Path, tract: str, patch: str, band: str, class_name: str, *, tile_origin: tuple[float, float]) -> list[dict[str, object]]:
    path = catalog_path(root, tract, patch, band, class_name)
    if not path.exists():
        return []
    table = Table.read(path)
    x, y = table_xy(table)
    major, minor, theta = ellipse_from_table(table)
    ids = np.asarray(table["id"], dtype=np.int64) if "id" in table.colnames else np.arange(len(table), dtype=np.int64)
    rows: list[dict[str, object]] = []
    for i in range(len(table)):
        rows.append(
            {
                "class": class_name,
                "id": int(ids[i]),
                "x": float(x[i]),
                "y": float(y[i]),
                "local_x": float(x[i] - tile_origin[0]),
                "local_y": float(y[i] - tile_origin[1]),
                "major": float(major[i]),
                "minor": float(minor[i]),
                "theta": float(theta[i]),
                "pu_reason": str(table["pu_reason"][i]) if "pu_reason" in table.colnames else "",
                "pu_mag": float(table["pu_mag"][i]) if "pu_mag" in table.colnames and np.isfinite(table["pu_mag"][i]) else np.nan,
                "ap2_minus_kron": float(table["pu_ap2_minus_kron_mag"][i])
                if "pu_ap2_minus_kron_mag" in table.colnames and np.isfinite(table["pu_ap2_minus_kron_mag"][i])
                else np.nan,
                "pu_remeasure_class": str(table["pu_remeasure_class"][i]) if "pu_remeasure_class" in table.colnames else "",
                "pu_remeasure_surface": str(table["pu_remeasure_surface"][i]) if "pu_remeasure_surface" in table.colnames else "",
                "pu_remeasure_reason": str(table["pu_remeasure_reason"][i]) if "pu_remeasure_reason" in table.colnames else "",
                "pu_refit_matched": bool(table["pu_refit_kron_radius_matched"][i])
                if "pu_refit_kron_radius_matched" in table.colnames
                else "",
                "footprint_area": float(table["base_FootprintArea_value"][i])
                if "base_FootprintArea_value" in table.colnames and np.isfinite(table["base_FootprintArea_value"][i])
                else np.nan,
            }
        )
    return rows


def inside(rows: list[dict[str, object]], size: float = 512.0) -> list[dict[str, object]]:
    return [r for r in rows if 0.0 <= float(r["local_x"]) < size and 0.0 <= float(r["local_y"]) < size]


def nearest_indices(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(a) == 0 or len(b) == 0:
        return np.full(len(a), -1, dtype=int), np.full(len(a), np.inf, dtype=np.float64)
    dist2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
    idx = np.argmin(dist2, axis=1)
    dist = np.sqrt(dist2[np.arange(len(a)), idx])
    return idx, dist


def mutual_matches(old_rows: list[dict[str, object]], new_rows: list[dict[str, object]], threshold: float) -> dict[int, tuple[int, float]]:
    old_xy = np.asarray([[float(r["local_x"]), float(r["local_y"])] for r in old_rows], dtype=np.float64)
    new_xy = np.asarray([[float(r["local_x"]), float(r["local_y"])] for r in new_rows], dtype=np.float64)
    old_to_new, old_dist = nearest_indices(old_xy, new_xy)
    new_to_old, _new_dist = nearest_indices(new_xy, old_xy)
    matches: dict[int, tuple[int, float]] = {}
    for old_idx, new_idx in enumerate(old_to_new):
        if new_idx < 0 or old_dist[old_idx] >= threshold:
            continue
        if new_to_old[new_idx] == old_idx:
            matches[old_idx] = (int(new_idx), float(old_dist[old_idx]))
    return matches


def nearest_row(row: dict[str, object], candidates: list[dict[str, object]]) -> tuple[dict[str, object] | None, float]:
    if not candidates:
        return None, math.inf
    xy = np.asarray([[float(r["local_x"]), float(r["local_y"])] for r in candidates], dtype=np.float64)
    p = np.asarray([float(row["local_x"]), float(row["local_y"])], dtype=np.float64)
    dist = np.sqrt(((xy - p[None, :]) ** 2).sum(axis=1))
    idx = int(np.argmin(dist))
    return candidates[idx], float(dist[idx])


def reg_line(row: dict[str, object], color: str) -> str:
    x = float(row["local_x"])
    y = float(row["local_y"])
    major = float(row["major"])
    minor = float(row["minor"])
    theta = float(row["theta"])
    area = math.pi * major * minor if np.isfinite(major * minor) else math.inf
    if not np.isfinite(area) or area <= 0 or area > 40000:
        return f"point({x + 1:.3f},{y + 1:.3f}) # point=circle 5 color={color} width=2\n"
    return f"ellipse({x + 1:.3f},{y + 1:.3f},{major:.3f},{minor:.3f},{math.degrees(theta):.3f}) # color={color} width=2\n"


def write_reg(path: Path, rows: list[dict[str, object]], color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Region file format: DS9 version 4.1\n")
        handle.write('global color=red dashlist=8 3 width=2 font="helvetica 12 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
        handle.write("image\n")
        for row in rows:
            handle.write(reg_line(row, color))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, default=Path("zangetsu_demo/data/sam_x18204_y20924/coadd"))
    parser.add_argument("--new-root", type=Path, default=Path("/nvme0/zc/scarlet/debug_patch45_preprocess/preprocessed"))
    parser.add_argument("--tract", default="9813")
    parser.add_argument("--patch", default="4,5")
    parser.add_argument("--band", default="HSC-I")
    parser.add_argument("--tile-origin", nargs=2, type=float, default=(18204.0, 20924.0))
    parser.add_argument("--match-radius", type=float, default=3.0)
    parser.add_argument("--out-dir", type=Path, default=Path("output/patch45_gt_crossmatch_260708"))
    args = parser.parse_args()

    tile_origin = (float(args.tile_origin[0]), float(args.tile_origin[1]))
    old_by_class = {
        cls: inside(read_class(args.old_root, args.tract, args.patch, args.band, cls, tile_origin=tile_origin))
        for cls in GT_CLASSES
    }
    new_by_class = {
        cls: inside(read_class(args.new_root, args.tract, args.patch, args.band, cls, tile_origin=tile_origin))
        for cls in CLASS_DIRS
    }
    old_gt = [row for cls in GT_CLASSES for row in old_by_class[cls]]
    new_gt = [row for cls in GT_CLASSES for row in new_by_class[cls]]
    new_all = [row for cls in CLASS_DIRS for row in new_by_class[cls]]
    matches = mutual_matches(old_gt, new_gt, float(args.match_radius))
    missing_old_indices = [i for i in range(len(old_gt)) if i not in matches]
    matched_new_indices = {new_idx for new_idx, _dist in matches.values()}
    new_unmatched_indices = [i for i in range(len(new_gt)) if i not in matched_new_indices]
    missing_rows = []
    new_by_id = {}
    for row in new_all:
        new_by_id.setdefault(int(row["id"]), []).append(row)
    for i in missing_old_indices:
        old = old_gt[i]
        same_id = new_by_id.get(int(old["id"]), [])
        same_id_inside = same_id[0] if same_id else None
        nearest_all, nearest_all_dist = nearest_row(old, new_all)
        nearest_gt, nearest_gt_dist = nearest_row(old, new_gt)
        evidence = same_id_inside or nearest_all
        missing_rows.append(
            {
                "old_id": old["id"],
                "old_class": old["class"],
                "old_local_x": old["local_x"],
                "old_local_y": old["local_y"],
                "old_pu_mag": old["pu_mag"],
                "old_ap2_minus_kron": old["ap2_minus_kron"],
                "old_footprint_area": old["footprint_area"],
                "new_same_id_class": same_id_inside["class"] if same_id_inside else "",
                "new_same_id_local_x": same_id_inside["local_x"] if same_id_inside else "",
                "new_same_id_local_y": same_id_inside["local_y"] if same_id_inside else "",
                "new_same_id_pu_mag": same_id_inside["pu_mag"] if same_id_inside else "",
                "new_same_id_ap2_minus_kron": same_id_inside["ap2_minus_kron"] if same_id_inside else "",
                "new_same_id_pu_reason": same_id_inside["pu_reason"] if same_id_inside else "",
                "new_same_id_remeasure_class": same_id_inside["pu_remeasure_class"] if same_id_inside else "",
                "new_same_id_remeasure_surface": same_id_inside["pu_remeasure_surface"] if same_id_inside else "",
                "new_same_id_remeasure_reason": same_id_inside["pu_remeasure_reason"] if same_id_inside else "",
                "new_same_id_refit_matched": same_id_inside["pu_refit_matched"] if same_id_inside else "",
                "nearest_new_all_class": nearest_all["class"] if nearest_all else "",
                "nearest_new_all_id": nearest_all["id"] if nearest_all else "",
                "nearest_new_all_dist": nearest_all_dist,
                "nearest_new_gt_class": nearest_gt["class"] if nearest_gt else "",
                "nearest_new_gt_id": nearest_gt["id"] if nearest_gt else "",
                "nearest_new_gt_dist": nearest_gt_dist,
                "suggested_current_status": evidence["class"] if evidence else "absent_from_current_class_catalogs",
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "old_gt_missing_from_current_gt.csv"
    fieldnames = list(missing_rows[0].keys()) if missing_rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        if fieldnames:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(missing_rows)

    write_reg(args.out_dir / "old_gt_missing_from_current_gt.reg", [old_gt[i] for i in missing_old_indices], "red")
    write_reg(args.out_dir / "old_gt_matched_current_gt.reg", [old_gt[i] for i in sorted(matches)], "green")
    write_reg(args.out_dir / "current_gt_not_in_old_gt.reg", [new_gt[i] for i in new_unmatched_indices], "cyan")
    current_extra_rows = []
    for i in new_unmatched_indices:
        row = new_gt[i]
        nearest_old, nearest_old_dist = nearest_row(row, old_gt)
        current_extra_rows.append(
            {
                "new_id": row["id"],
                "new_class": row["class"],
                "new_local_x": row["local_x"],
                "new_local_y": row["local_y"],
                "new_pu_mag": row["pu_mag"],
                "new_ap2_minus_kron": row["ap2_minus_kron"],
                "new_footprint_area": row["footprint_area"],
                "new_pu_reason": row["pu_reason"],
                "new_remeasure_class": row["pu_remeasure_class"],
                "new_remeasure_surface": row["pu_remeasure_surface"],
                "new_remeasure_reason": row["pu_remeasure_reason"],
                "nearest_old_id": nearest_old["id"] if nearest_old else "",
                "nearest_old_class": nearest_old["class"] if nearest_old else "",
                "nearest_old_dist": nearest_old_dist,
            }
        )
    with (args.out_dir / "current_gt_not_in_old_gt.csv").open("w", newline="", encoding="utf-8") as handle:
        if current_extra_rows:
            writer = csv.DictWriter(handle, fieldnames=list(current_extra_rows[0].keys()))
            writer.writeheader()
            writer.writerows(current_extra_rows)
    counts = {
        "old_gt": len(old_gt),
        "new_gt": len(new_gt),
        "matched": len(matches),
        "old_missing_from_new_gt": len(missing_old_indices),
        "new_gt_not_in_old_gt": len(new_unmatched_indices),
        "old_by_class": {cls: len(rows) for cls, rows in old_by_class.items()},
        "new_by_class": {cls: len(rows) for cls, rows in new_by_class.items()},
        "missing_suggested_status": {},
    }
    for row in missing_rows:
        status = str(row["suggested_current_status"])
        counts["missing_suggested_status"][status] = counts["missing_suggested_status"].get(status, 0) + 1
    (args.out_dir / "summary.json").write_text(__import__("json").dumps(counts, indent=2), encoding="utf-8")
    print(__import__("json").dumps(counts, indent=2))
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
