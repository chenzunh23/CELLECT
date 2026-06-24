#!/usr/bin/env python3
"""Make DS9 region diagnostics for patch 9813/4,5 detection eval outputs."""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from astropy.table import Table


PATCH = "4,5"
TRACT = "9813"
TILE_SIZE = 512
ARCSEC_PER_PIXEL = 0.168
MATCH_RADIUS_PIX = 0.5 / ARCSEC_PER_PIXEL
MAG_LIMITS = {
    "HSC-G": 27.4,
    "HSC-R": 27.1,
    "HSC-I": 26.9,
    "HSC-Z": 26.3,
    "HSC-Y": 25.3,
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path
    eval_csv: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="output/detection_only_b5/reg_diagnostics_patch45")
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--num-tiles", type=int, default=2)
    parser.add_argument("--mag-zp", type=float, default=27.0)
    parser.add_argument("--bands", nargs="+", default=["HSC-G", "HSC-R", "HSC-I"])
    parser.add_argument(
        "--denoised-root",
        default="/home/czh23/denoised/9813",
        help="Root that contains patch directories for denoised data.",
    )
    parser.add_argument(
        "--coadd-root",
        default="/nvme0/zc/scarlet/preprocessed/9813",
        help="Root that contains patch directories for official coadd data.",
    )
    parser.add_argument(
        "--denoised-eval-csv",
        default="output/detection_only_b5/eval_sources_denoised_4,5.csv",
    )
    parser.add_argument(
        "--coadd-eval-csv",
        default="output/detection_only_b5/eval_sources_4,5_6,1_clean.csv",
    )
    parser.add_argument(
        "--tiles",
        nargs="*",
        default=None,
        help="Optional explicit tile names, e.g. grid_r00_c00_x15900_y19900.",
    )
    return parser.parse_args()


def tile_xy0(tile_name: str) -> tuple[int, int]:
    match = re.search(r"_x(-?\d+)_y(-?\d+)$", tile_name)
    if not match:
        raise ValueError(f"Cannot parse tile origin from {tile_name!r}")
    return int(match.group(1)), int(match.group(2))


def read_eval_rows(path: Path, patch: str = PATCH) -> dict[str, list[dict[str, str]]]:
    rows_by_tile: dict[str, list[dict[str, str]]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("patch") != patch:
                continue
            tile = row.get("tile_name") or row.get("record", "").split("/")[-1]
            if not tile:
                continue
            rows_by_tile.setdefault(tile, []).append(row)
    return rows_by_tile


@lru_cache(maxsize=128)
def read_table(path: str) -> Table:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Table.read(path)


def table_path(root: Path, kind: str, band: str) -> Path:
    return root / PATCH / kind / band / f"meas-{band}-{TRACT}-{PATCH}.fits"


def tile_table_path(root: Path, kind: str, tile_name: str) -> Path:
    return root / PATCH / kind / f"{tile_name}_meas.fits"


def available_bands(root: Path) -> list[str]:
    base = root / PATCH / "band_reference_catalogs"
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir())


def first_existing_xy_columns(table: Table) -> tuple[str, str]:
    candidates = (
        ("centroid_local_x", "centroid_local_y"),
        ("base_SdssShape_x", "base_SdssShape_y"),
        ("base_SdssCentroid_x", "base_SdssCentroid_y"),
        ("base_NaiveCentroid_x", "base_NaiveCentroid_y"),
        ("deblend_psfCenter_x", "deblend_psfCenter_y"),
        ("slot_Centroid_x", "slot_Centroid_y"),
    )
    names = set(table.colnames)
    for x_name, y_name in candidates:
        if x_name in names and y_name in names:
            return x_name, y_name
    raise KeyError("No supported centroid columns found")


def finite_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def local_rows(table: Table, tile_name: str) -> tuple[Table, np.ndarray, np.ndarray]:
    x0, y0 = tile_xy0(tile_name)
    x_name, y_name = first_existing_xy_columns(table)
    x = finite_array(table[x_name])
    y = finite_array(table[y_name])
    if x_name == "centroid_local_x":
        lx, ly = x, y
    else:
        lx, ly = x - x0, y - y0
    keep = np.isfinite(lx) & np.isfinite(ly) & (lx >= 0) & (lx < TILE_SIZE) & (ly >= 0) & (ly < TILE_SIZE)
    return table[keep], lx[keep], ly[keep]


def mag_from_row(row, zp: float) -> float:
    if "pu_mag" in row.colnames:
        try:
            mag = float(row["pu_mag"])
            if np.isfinite(mag):
                return mag
        except Exception:
            pass
    for col in ("ext_photometryKron_KronFlux_instFlux", "base_SdssShape_instFlux", "base_PsfFlux_instFlux"):
        if col not in row.colnames:
            continue
        try:
            flux = float(row[col])
        except Exception:
            continue
        if np.isfinite(flux) and flux > 0:
            return float(zp - 2.5 * math.log10(flux))
    return float("nan")


def load_masks(root: Path, band: str, tile_name: str) -> tuple[np.ndarray, np.ndarray]:
    candidates = (
        root / PATCH / "band_targets" / band / f"{tile_name}.npz",
        root / PATCH / "targets" / f"{tile_name}.npz",
    )
    for path in candidates:
        if not path.exists():
            continue
        with np.load(path) as data:
            clean = data["clean_mask"].astype(bool) if "clean_mask" in data else np.zeros((TILE_SIZE, TILE_SIZE), bool)
            background = (
                data["background_mask"].astype(bool)
                if "background_mask" in data
                else np.zeros((TILE_SIZE, TILE_SIZE), bool)
            )
        return clean, background
    return np.zeros((TILE_SIZE, TILE_SIZE), bool), np.zeros((TILE_SIZE, TILE_SIZE), bool)


def greedy_match(pred_xy: np.ndarray, gt_xy: np.ndarray, radius: float) -> tuple[dict[int, int], set[int]]:
    if len(pred_xy) == 0 or len(gt_xy) == 0:
        return {}, set()
    pairs: list[tuple[float, int, int]] = []
    r2 = radius * radius
    for pi, pred in enumerate(pred_xy):
        delta = gt_xy - pred
        d2 = np.sum(delta * delta, axis=1)
        candidates = np.where(d2 <= r2)[0]
        for gi in candidates:
            pairs.append((float(d2[gi]), pi, int(gi)))
    pairs.sort(key=lambda item: item[0])
    pred_to_gt: dict[int, int] = {}
    used_gt: set[int] = set()
    for _, pi, gi in pairs:
        if pi in pred_to_gt or gi in used_gt:
            continue
        pred_to_gt[pi] = gi
        used_gt.add(gi)
    return pred_to_gt, used_gt


def nearest_mag_in_clean_range(
    xy: np.ndarray,
    source_rows: Table,
    source_x: np.ndarray,
    source_y: np.ndarray,
    band: str,
    zp: float,
) -> bool:
    if len(source_rows) == 0:
        return False
    d2 = (source_x - xy[0]) ** 2 + (source_y - xy[1]) ** 2
    idx = int(np.argmin(d2))
    mag = mag_from_row(source_rows[idx], zp)
    limit = MAG_LIMITS.get(band)
    return bool(limit is not None and np.isfinite(mag) and (limit - 5.0) <= mag < limit)


def ellipse_region(row, x: float, y: float, color: str, width: int = 1) -> str:
    try:
        major = float(row["ellipse_major_sigma"])
        minor = float(row["ellipse_minor_sigma"])
        theta = float(row["ellipse_theta"])
    except Exception:
        major = minor = 4.0
        theta = 0.0
    if not np.isfinite(major) or major <= 0:
        major = 4.0
    if not np.isfinite(minor) or minor <= 0:
        minor = max(1.5, min(major, 4.0))
    if not np.isfinite(theta):
        theta = 0.0
    return f"ellipse({x + 1:.3f},{y + 1:.3f},{major:.3f},{minor:.3f},{math.degrees(theta):.3f}) # color={color} width={width}"


def point_region(x: float, y: float, color: str, radius: float = 4.0, width: int = 1) -> str:
    return f"circle({x + 1:.3f},{y + 1:.3f},{radius:.2f}) # color={color} width={width}"


def pred_rows_for_band(rows: list[dict[str, str]], band: str) -> np.ndarray:
    points = []
    for row in rows:
        if row.get("band") != band:
            continue
        if row.get("source_type") not in ("band", ""):
            continue
        try:
            x = float(row["x_local"])
            y = float(row["y_local"])
        except Exception:
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        if row.get("strict_ignore") == "1":
            continue
        points.append((x, y))
    return np.asarray(points, dtype=float).reshape(-1, 2)


def make_tile_band_regs(
    spec: DatasetSpec,
    tile_name: str,
    band: str,
    rows: list[dict[str, str]],
    out_dir: Path,
) -> dict[str, int | str]:
    gt_lines = [
        "# Region file format: DS9 version 4.1",
        f"# {spec.name} {PATCH}/{tile_name} {band}: clean GT ellipses and ordinary-ignore GT centers",
        "image",
    ]
    det_lines = [
        "# Region file format: DS9 version 4.1",
        f"# {spec.name} {PATCH}/{tile_name} {band}: FN centers and clean/background-region FP centers",
        "# FP magenta: unmatched prediction center inside clean_mask | background_mask.",
        "image",
    ]
    tp_lines = [
        "# Region file format: DS9 version 4.1",
        f"# {spec.name} {PATCH}/{tile_name} {band}: clean TP GT ellipses and matched prediction centers",
        "# blue ellipse: matched clean GT shape; cyan circle: matched prediction center.",
        "image",
    ]
    summary = {
        "clean_gt": 0,
        "ordinary_ignore": 0,
        "tp": 0,
        "fn": 0,
        "fp_clean_background": 0,
    }

    tile_clean_path = tile_table_path(spec.root, "reference_catalogs", tile_name)
    tile_ignore_path = tile_table_path(spec.root, "ignore_catalogs", tile_name)
    tile_clean_rows = Table()
    tile_clean_x = np.asarray([], dtype=float)
    tile_clean_y = np.asarray([], dtype=float)
    tile_ignore_rows = Table()
    tile_ignore_x = np.asarray([], dtype=float)
    tile_ignore_y = np.asarray([], dtype=float)
    if tile_clean_path.exists():
        tile_clean_rows, tile_clean_x, tile_clean_y = local_rows(read_table(str(tile_clean_path)), tile_name)
    if tile_ignore_path.exists():
        tile_ignore_rows, tile_ignore_x, tile_ignore_y = local_rows(read_table(str(tile_ignore_path)), tile_name)

    gt_lines.append(f"# merged tile catalogs: clean={len(tile_clean_rows)} ordinary_ignore={len(tile_ignore_rows)}")
    for row, x, y in zip(tile_clean_rows, tile_clean_x, tile_clean_y):
        gt_lines.append(ellipse_region(row, float(x), float(y), "green", width=1))
    for x, y in zip(tile_ignore_x, tile_ignore_y):
        gt_lines.append(point_region(float(x), float(y), "yellow", radius=3.0, width=1))
    summary["clean_gt"] += int(len(tile_clean_rows))
    summary["ordinary_ignore"] += int(len(tile_ignore_rows))

    clean_rows, clean_x, clean_y = tile_clean_rows, tile_clean_x, tile_clean_y
    ignore_rows, ignore_x, ignore_y = tile_ignore_rows, tile_ignore_x, tile_ignore_y
    pred_xy = pred_rows_for_band(rows, band)
    clean_xy = np.column_stack([clean_x, clean_y]) if len(clean_rows) else np.zeros((0, 2), dtype=float)
    ignore_xy = np.column_stack([ignore_x, ignore_y]) if len(ignore_rows) else np.zeros((0, 2), dtype=float)
    pred_to_clean, clean_used = greedy_match(pred_xy, clean_xy, MATCH_RADIUS_PIX)
    clean_to_pred = {gi: pi for pi, gi in pred_to_clean.items()}
    remaining_pred_idx = [idx for idx in range(len(pred_xy)) if idx not in pred_to_clean]
    if remaining_pred_idx:
        rem_xy = pred_xy[remaining_pred_idx]
        rem_to_ignore, _ = greedy_match(rem_xy, ignore_xy, MATCH_RADIUS_PIX)
        ordinary_matched_pred = {remaining_pred_idx[idx] for idx in rem_to_ignore}
    else:
        ordinary_matched_pred = set()

    clean_mask, background_mask = load_masks(spec.root, band, tile_name)
    clean_bg_mask = clean_mask | background_mask
    det_lines.append(f"# {band}: pred={len(pred_xy)} tp={len(clean_used)} fn={len(clean_rows) - len(clean_used)}")
    tp_lines.append(f"# {band}: tp={len(clean_used)}")
    for gi in sorted(clean_used):
        row = clean_rows[gi]
        gx, gy = float(clean_x[gi]), float(clean_y[gi])
        px, py = pred_xy[clean_to_pred[gi]]
        tp_lines.append(ellipse_region(row, gx, gy, "blue", width=1))
        tp_lines.append(point_region(float(px), float(py), "cyan", radius=2.5, width=1))
        summary["tp"] += 1
    for gi in range(len(clean_rows)):
        if gi in clean_used:
            continue
        det_lines.append(point_region(float(clean_x[gi]), float(clean_y[gi]), "red", radius=4.0, width=2))
        summary["fn"] += 1

    for pi, xy in enumerate(pred_xy):
        if pi in pred_to_clean or pi in ordinary_matched_pred:
            continue
        xi, yi = int(round(float(xy[0]))), int(round(float(xy[1])))
        if not (0 <= xi < TILE_SIZE and 0 <= yi < TILE_SIZE and clean_bg_mask[yi, xi]):
            continue
        det_lines.append(point_region(float(xy[0]), float(xy[1]), "magenta", radius=3.5, width=2))
        summary["fp_clean_background"] += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    safe_tile = tile_name.replace(",", "_")
    safe_band = band.replace("-", "_")
    gt_path = out_dir / f"{spec.name}_{PATCH.replace(',', '_')}_{safe_tile}_{safe_band}_clean_gt_ordinary_ignore.reg"
    det_path = out_dir / f"{spec.name}_{PATCH.replace(',', '_')}_{safe_tile}_{safe_band}_fn_fp_clean_background.reg"
    tp_path = out_dir / f"{spec.name}_{PATCH.replace(',', '_')}_{safe_tile}_{safe_band}_clean_tp.reg"
    gt_path.write_text("\n".join(gt_lines) + "\n")
    det_path.write_text("\n".join(det_lines) + "\n")
    tp_path.write_text("\n".join(tp_lines) + "\n")
    summary["gt_reg"] = str(gt_path)
    summary["det_reg"] = str(det_path)
    summary["tp_reg"] = str(tp_path)
    return summary


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    denoised = DatasetSpec("denoised", Path(args.denoised_root), Path(args.denoised_eval_csv))
    coadd = DatasetSpec("coadd", Path(args.coadd_root), Path(args.coadd_eval_csv))
    den_rows = read_eval_rows(denoised.eval_csv)
    coadd_rows = read_eval_rows(coadd.eval_csv)
    if args.tiles:
        tiles = args.tiles
    else:
        common = sorted(set(den_rows) & set(coadd_rows))
        rng = random.Random(args.seed)
        tiles = rng.sample(common, min(args.num_tiles, len(common)))
    if not tiles:
        raise RuntimeError("No common patch 4,5 tiles found in the eval CSV files")
    print("selected tiles:", ", ".join(tiles))
    for tile in tiles:
        for spec, rows_by_tile in ((denoised, den_rows), (coadd, coadd_rows)):
            if tile not in rows_by_tile:
                print(f"skip {spec.name} {tile}: no eval rows")
                continue
            row_bands = {row.get("band") for row in rows_by_tile[tile] if row.get("band")}
            for band in args.bands:
                if band not in row_bands:
                    print(f"skip {spec.name} {tile} {band}: no eval rows")
                    continue
                summary = make_tile_band_regs(spec, tile, band, rows_by_tile[tile], out_dir)
                print(
                    f"{spec.name} {tile} {band}: clean_gt={summary['clean_gt']} "
                    f"ordinary_ignore={summary['ordinary_ignore']} tp={summary['tp']} "
                    f"fn={summary['fn']} fp_clean_background={summary['fp_clean_background']}"
                )
                print("  ", summary["gt_reg"])
                print("  ", summary["det_reg"])
                print("  ", summary["tp_reg"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
