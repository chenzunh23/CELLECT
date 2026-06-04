#!/usr/bin/env python3
"""Create per-band DS9 REG diagnostics for the Zangetsu demo eval outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from astropy.table import Table


BANDS = ["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"]
TRACT = "9813"
PATCH = "6,1"
TILE_SIZE = 512
MATCH_RADIUS_PIX = 0.5 / 0.168


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path
    eval_csv: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="zangetsu_demo/reg_diagnostics_by_band")
    parser.add_argument("--coadd-root", default="zangetsu_demo/preprocessed/coadd/9813")
    parser.add_argument("--denoised-root", default="zangetsu_demo/preprocessed/denoised/9813")
    parser.add_argument("--coadd-eval-csv", default="zangetsu_demo/eval/coadd/eval_sources.csv")
    parser.add_argument("--denoised-eval-csv", default="zangetsu_demo/eval/denoised/eval_sources.csv")
    parser.add_argument(
        "--extra-dataset",
        action="append",
        default=[],
        metavar="NAME:ROOT:EVAL_CSV",
        help="Additional dataset to include in REG/metrics output.",
    )
    parser.add_argument("--bands", nargs="+", default=BANDS)
    return parser.parse_args()


def tile_xy0(tile_name: str) -> tuple[int, int]:
    match = re.search(r"_x(-?\d+)_y(-?\d+)", tile_name)
    if not match:
        raise ValueError(f"Cannot parse tile origin from {tile_name!r}")
    return int(match.group(1)), int(match.group(2))


def read_eval_rows(path: Path) -> dict[str, list[dict[str, str]]]:
    rows_by_tile: dict[str, list[dict[str, str]]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("patch") != PATCH:
                continue
            tile = row.get("tile_name") or row.get("record", "").split("/")[-1]
            if tile:
                rows_by_tile.setdefault(tile, []).append(row)
    return rows_by_tile


@lru_cache(maxsize=128)
def read_table(path: str) -> Table:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Table.read(path)


def table_path(root: Path, kind: str, band: str) -> Path:
    return root / PATCH / kind / band / f"meas-{band}-{TRACT}-{PATCH}.fits"


def xy_from_table(table: Table, tile_name: str) -> tuple[np.ndarray, np.ndarray]:
    x0, y0 = tile_xy0(tile_name)
    names = set(table.colnames)
    for x_name, y_name in (
        ("base_SdssShape_x", "base_SdssShape_y"),
        ("base_SdssCentroid_x", "base_SdssCentroid_y"),
        ("base_NaiveCentroid_x", "base_NaiveCentroid_y"),
        ("deblend_psfCenter_x", "deblend_psfCenter_y"),
        ("centroid_local_x", "centroid_local_y"),
    ):
        if x_name in names and y_name in names:
            x = np.asarray(table[x_name], dtype=float)
            y = np.asarray(table[y_name], dtype=float)
            if x_name == "centroid_local_x":
                return x, y
            return x - x0, y - y0
    raise KeyError("No supported centroid columns found")


def local_rows(table: Table, tile_name: str) -> tuple[Table, np.ndarray, np.ndarray]:
    lx, ly = xy_from_table(table, tile_name)
    keep = np.isfinite(lx) & np.isfinite(ly) & (lx >= 0) & (lx < TILE_SIZE) & (ly >= 0) & (ly < TILE_SIZE)
    return table[keep], lx[keep], ly[keep]


def load_masks(root: Path, band: str, tile_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = root / PATCH / "band_targets" / band / f"{tile_name}.npz"
    with np.load(path) as data:
        clean = data["clean_mask"].astype(bool) if "clean_mask" in data else np.zeros((TILE_SIZE, TILE_SIZE), bool)
        background = data["background_mask"].astype(bool) if "background_mask" in data else np.zeros((TILE_SIZE, TILE_SIZE), bool)
        ignore = data["ignore_mask"].astype(bool) if "ignore_mask" in data else np.zeros((TILE_SIZE, TILE_SIZE), bool)
        strict = data["strict_ignore_mask"].astype(bool) if "strict_ignore_mask" in data else np.zeros((TILE_SIZE, TILE_SIZE), bool)
        ordinary_ignore = ignore & ~strict
    return clean, background, ordinary_ignore


def greedy_match(pred_xy: np.ndarray, gt_xy: np.ndarray, radius: float) -> tuple[dict[int, int], set[int]]:
    if len(pred_xy) == 0 or len(gt_xy) == 0:
        return {}, set()
    pairs: list[tuple[float, int, int]] = []
    r2 = radius * radius
    for pi, pred in enumerate(pred_xy):
        d2 = np.sum((gt_xy - pred) ** 2, axis=1)
        for gi in np.where(d2 <= r2)[0]:
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


def ellipse_region(row, x: float, y: float, color: str, width: int = 1) -> str:
    major = float(row["ellipse_major_sigma"]) if "ellipse_major_sigma" in row.colnames else 4.0
    minor = float(row["ellipse_minor_sigma"]) if "ellipse_minor_sigma" in row.colnames else 4.0
    theta = float(row["ellipse_theta"]) if "ellipse_theta" in row.colnames else 0.0
    if not np.isfinite(major) or major <= 0:
        major = 4.0
    if not np.isfinite(minor) or minor <= 0:
        minor = 4.0
    if not np.isfinite(theta):
        theta = 0.0
    return f"ellipse({x + 1:.3f},{y + 1:.3f},{major:.3f},{minor:.3f},{math.degrees(theta):.3f}) # color={color} width={width}"


def point_region(x: float, y: float, color: str, radius: float = 4.0, width: int = 1) -> str:
    return f"circle({x + 1:.3f},{y + 1:.3f},{radius:.2f}) # color={color} width={width}"


def pred_rows_for_band(rows: list[dict[str, str]], band: str) -> np.ndarray:
    pts = []
    for row in rows:
        if row.get("band") != band or row.get("strict_ignore") == "1":
            continue
        try:
            x, y = float(row["x_local"]), float(row["y_local"])
        except Exception:
            continue
        if np.isfinite(x) and np.isfinite(y):
            pts.append((x, y))
    return np.asarray(pts, dtype=np.float32).reshape(-1, 2)


def write_tile_band(spec: DatasetSpec, tile_name: str, band: str, rows: list[dict[str, str]], out_dir: Path) -> dict[str, int]:
    clean_path = table_path(spec.root, "band_reference_catalogs", band)
    ignore_path = table_path(spec.root, "band_reference_ignore", band)
    clean_rows, clean_x, clean_y = local_rows(read_table(str(clean_path)), tile_name)
    ignore_rows, ignore_x, ignore_y = local_rows(read_table(str(ignore_path)), tile_name) if ignore_path.exists() else (Table(), np.array([]), np.array([]))
    pred_xy = pred_rows_for_band(rows, band)
    clean_xy = np.column_stack([clean_x, clean_y]) if len(clean_rows) else np.zeros((0, 2), dtype=np.float32)
    ignore_xy = np.column_stack([ignore_x, ignore_y]) if len(ignore_rows) else np.zeros((0, 2), dtype=np.float32)
    pred_to_clean, clean_used = greedy_match(pred_xy, clean_xy, MATCH_RADIUS_PIX)
    clean_to_pred = {gi: pi for pi, gi in pred_to_clean.items()}
    remaining = [idx for idx in range(len(pred_xy)) if idx not in pred_to_clean]
    if remaining:
        rem_to_ignore, ignore_used = greedy_match(pred_xy[remaining], ignore_xy, MATCH_RADIUS_PIX)
        ordinary_matched_pred = {remaining[idx] for idx in rem_to_ignore}
    else:
        ignore_used = set()
        ordinary_matched_pred = set()

    clean_mask, background_mask, ordinary_ignore_mask = load_masks(spec.root, band, tile_name)
    clean_bg = clean_mask | background_mask

    header = ["# Region file format: DS9 version 4.1", "image"]
    gt_lines = header + [f"# {spec.name} {PATCH}/{tile_name} {band}: clean GT and ordinary ignore"]
    det_lines = header + [f"# {spec.name} {PATCH}/{tile_name} {band}: FN and clean/background FP"]
    tp_lines = header + [f"# {spec.name} {PATCH}/{tile_name} {band}: clean TP"]

    for row, x, y in zip(clean_rows, clean_x, clean_y):
        gt_lines.append(ellipse_region(row, float(x), float(y), "green", width=2))
    for x, y in zip(ignore_x, ignore_y):
        gt_lines.append(point_region(float(x), float(y), "yellow", radius=3.0, width=2))

    for gi in sorted(clean_used):
        row = clean_rows[gi]
        px, py = pred_xy[clean_to_pred[gi]]
        tp_lines.append(ellipse_region(row, float(clean_x[gi]), float(clean_y[gi]), "blue", width=2))
        tp_lines.append(point_region(float(px), float(py), "cyan", radius=2.5, width=2))
    for gi in range(len(clean_rows)):
        if gi not in clean_used:
            det_lines.append(point_region(float(clean_x[gi]), float(clean_y[gi]), "red", radius=4.0, width=2))

    clean_fp_count = 0
    ordinary_fp_count = 0
    for pi, xy in enumerate(pred_xy):
        if pi in pred_to_clean or pi in ordinary_matched_pred:
            continue
        xi, yi = int(round(float(xy[0]))), int(round(float(xy[1])))
        if 0 <= xi < TILE_SIZE and 0 <= yi < TILE_SIZE:
            if clean_bg[yi, xi]:
                det_lines.append(point_region(float(xy[0]), float(xy[1]), "magenta", radius=3.5, width=2))
                clean_fp_count += 1
            elif ordinary_ignore_mask[yi, xi]:
                ordinary_fp_count += 1

    dataset_out_dir = out_dir / spec.name
    dataset_out_dir.mkdir(parents=True, exist_ok=True)
    safe_tile = tile_name.replace(",", "_")
    safe_band = band.replace("-", "_")
    prefix = f"{spec.name}_{PATCH.replace(',', '_')}_{safe_tile}_{safe_band}"
    (dataset_out_dir / f"{prefix}_clean_gt_ordinary_ignore.reg").write_text("\n".join(gt_lines) + "\n")
    (dataset_out_dir / f"{prefix}_fn_fp_clean_background.reg").write_text("\n".join(det_lines) + "\n")
    (dataset_out_dir / f"{prefix}_clean_tp.reg").write_text("\n".join(tp_lines) + "\n")
    clean_gt = len(clean_rows)
    clean_tp = len(clean_used)
    clean_fn = clean_gt - clean_tp
    ordinary_gt = len(ignore_rows)
    ordinary_tp = len(ignore_used)
    ordinary_fn = ordinary_gt - ordinary_tp
    return {
        "clean_GT": clean_gt,
        "clean_TP": clean_tp,
        "clean_FP": clean_fp_count,
        "clean_FN": clean_fn,
        "ordinary_GT": ordinary_gt,
        "ordinary_TP": ordinary_tp,
        "ordinary_FP": ordinary_fp_count,
        "ordinary_FN": ordinary_fn,
        "total_GT": clean_gt + ordinary_gt,
        "total_TP": clean_tp + ordinary_tp,
        "total_FP": clean_fp_count + ordinary_fp_count,
        "total_FN": clean_fn + ordinary_fn,
    }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    specs = [
        DatasetSpec("coadd", Path(args.coadd_root), Path(args.coadd_eval_csv)),
        DatasetSpec("denoised", Path(args.denoised_root), Path(args.denoised_eval_csv)),
    ]
    for item in args.extra_dataset:
        parts = str(item).split(":", 2)
        if len(parts) != 3:
            raise ValueError(f"--extra-dataset must be NAME:ROOT:EVAL_CSV, got {item!r}")
        specs.append(DatasetSpec(parts[0], Path(parts[1]), Path(parts[2])))
    metric_rows: list[dict[str, object]] = []
    for spec in specs:
        rows_by_tile = read_eval_rows(spec.eval_csv)
        for tile_name in sorted(rows_by_tile):
            rows = rows_by_tile[tile_name]
            for band in args.bands:
                summary = write_tile_band(spec, tile_name, band, rows, out_dir)
                print(
                    f"{spec.name} {tile_name} {band}: "
                    f"clean TP/FP/GT={summary['clean_TP']}/{summary['clean_FP']}/{summary['clean_GT']} "
                    f"ordinary TP/FP/GT={summary['ordinary_TP']}/{summary['ordinary_FP']}/{summary['ordinary_GT']} "
                    f"total TP/FP/GT={summary['total_TP']}/{summary['total_FP']}/{summary['total_GT']}"
                )
                metric_rows.append(
                    {
                        "level": "band",
                        "dataset": spec.name,
                        "tile": tile_name,
                        "band": band,
                        "GT": int(summary["clean_GT"]),
                        "TP": int(summary["clean_TP"]),
                        "FP": int(summary["clean_FP"]),
                        "FN": int(summary["clean_FN"]),
                        **{name: int(summary[name]) for name in (
                            "clean_GT",
                            "clean_TP",
                            "clean_FP",
                            "clean_FN",
                            "ordinary_GT",
                            "ordinary_TP",
                            "ordinary_FP",
                            "ordinary_FN",
                            "total_GT",
                            "total_TP",
                            "total_FP",
                            "total_FN",
                        )},
                    }
                )
    aggregate_rows: list[dict[str, object]] = []
    for keys, level in (
        (("dataset", "tile"), "tile"),
        (("dataset", "band"), "band_total"),
        (("dataset",), "dataset_total"),
    ):
        groups: dict[tuple[object, ...], dict[str, object]] = {}
        for row in metric_rows:
            key = tuple(row[name] for name in keys)
            group = groups.setdefault(
                key,
                {
                    "level": level,
                    "dataset": row["dataset"],
                    "tile": row["tile"] if "tile" in keys else "ALL",
                    "band": row["band"] if "band" in keys else "ALL",
                    "GT": 0,
                    "TP": 0,
                    "FP": 0,
                    "FN": 0,
                    "clean_GT": 0,
                    "clean_TP": 0,
                    "clean_FP": 0,
                    "clean_FN": 0,
                    "ordinary_GT": 0,
                    "ordinary_TP": 0,
                    "ordinary_FP": 0,
                    "ordinary_FN": 0,
                    "total_GT": 0,
                    "total_TP": 0,
                    "total_FP": 0,
                    "total_FN": 0,
                },
            )
            for name in (
                "GT",
                "TP",
                "FP",
                "FN",
                "clean_GT",
                "clean_TP",
                "clean_FP",
                "clean_FN",
                "ordinary_GT",
                "ordinary_TP",
                "ordinary_FP",
                "ordinary_FN",
                "total_GT",
                "total_TP",
                "total_FP",
                "total_FN",
            ):
                group[name] = int(group[name]) + int(row[name])
        aggregate_rows.extend(groups.values())

    def _add_rates(row: dict[str, object]) -> dict[str, object]:
        row = dict(row)
        for prefix in ("clean", "ordinary", "total"):
            tp = int(row[f"{prefix}_TP"])
            fp = int(row[f"{prefix}_FP"])
            fn = int(row[f"{prefix}_FN"])
            row[f"{prefix}_precision"] = tp / (tp + fp) if tp + fp > 0 else float("nan")
            row[f"{prefix}_recall"] = tp / (tp + fn) if tp + fn > 0 else float("nan")
        row["precision"] = row["clean_precision"]
        row["recall"] = row["clean_recall"]
        return row

    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = [_add_rates(row) for row in metric_rows + aggregate_rows]
    metrics_csv = out_dir / "zangetsu_reg_metrics.csv"
    fieldnames = [
        "level",
        "dataset",
        "tile",
        "band",
        "GT",
        "TP",
        "FP",
        "FN",
        "precision",
        "recall",
        "clean_GT",
        "clean_TP",
        "clean_FP",
        "clean_FN",
        "clean_precision",
        "clean_recall",
        "ordinary_GT",
        "ordinary_TP",
        "ordinary_FP",
        "ordinary_FN",
        "ordinary_precision",
        "ordinary_recall",
        "total_GT",
        "total_TP",
        "total_FP",
        "total_FN",
        "total_precision",
        "total_recall",
    ]
    with metrics_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    metrics_json = out_dir / "zangetsu_reg_metrics.json"
    metrics_json.write_text(json.dumps(all_rows, indent=2, allow_nan=True) + "\n")
    total_rows = [
        {
            "level": row["level"],
            "dataset": row["dataset"],
            "tile": row["tile"],
            "band": row["band"],
            "total_GT": row["total_GT"],
            "total_TP": row["total_TP"],
            "total_FP": row["total_FP"],
            "total_FN": row["total_FN"],
            "total_precision": row["total_precision"],
            "total_recall": row["total_recall"],
        }
        for row in all_rows
    ]
    total_csv = out_dir / "zangetsu_reg_total_metrics.csv"
    total_fieldnames = [
        "level",
        "dataset",
        "tile",
        "band",
        "total_GT",
        "total_TP",
        "total_FP",
        "total_FN",
        "total_precision",
        "total_recall",
    ]
    with total_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=total_fieldnames)
        writer.writeheader()
        writer.writerows(total_rows)
    total_json = out_dir / "zangetsu_reg_total_metrics.json"
    total_json.write_text(json.dumps(total_rows, indent=2, allow_nan=True) + "\n")
    print(f"metrics_csv: {metrics_csv}")
    print(f"metrics_json: {metrics_json}")
    print(f"total_metrics_csv: {total_csv}")
    print(f"total_metrics_json: {total_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
