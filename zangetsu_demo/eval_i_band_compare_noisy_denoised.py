#!/usr/bin/env python3
"""Run I-band Zangetsu eval and write comparison REG diagnostics.

Outputs three REG files per dataset/tile:

1. all detected centers
2. all detected predicted ellipses
3. clean/background-region FN GT ellipses

For noisy/denoised/coadd comparisons, detections are first matched to clean GT.
Two detections are considered the same source if they match the same clean GT.
Only detections unmatched to clean GT on both sides are then cross-matched with
a 3 arcsec radius. Common detections are green/cyan; denoised-only detections
are magenta; noisy-only detections are orange; coadd-only detections are blue.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from astropy.table import Table
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astro_train_data import AstroCutoutDataset, collate_cutouts, discover_cutout_records

from eval_i_band_shape_regs import (
    BAND,
MATCH_RADIUS_PIX,
    PATCH,
    TRACT,
    _ellipse_from_row,
    _ellipse_pred,
    _greedy_match,
    _load_masks,
    _local_rows,
    _make_model,
    _point,
    _read_config,
    _run_predictions,
    _table,
)


FALLBACK_COMPARE_RADIUS_ARCSEC = 3.0
FALLBACK_COMPARE_RADIUS_PIX = FALLBACK_COMPARE_RADIUS_ARCSEC / 0.168


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path


REG_HEADER = [
    "# Region file format: DS9 version 4.1",
    'global color=green dashlist=8 3 width=2 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1',
    "image",
]


def _read_checkpoint_epoch(path: Path) -> object:
    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception:
        return ""
    if isinstance(ckpt, dict):
        return ckpt.get("epoch", "")
    return ""


def _prediction_xy(rows: list[dict]) -> np.ndarray:
    if not rows:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray([[row["x"], row["y"]] for row in rows], dtype=np.float32).reshape(-1, 2)


def _spatial_unique_flags(rows: list[dict], other_rows: list[dict], *, radius: float) -> list[bool]:
    xy = _prediction_xy(rows)
    other_xy = _prediction_xy(other_rows)
    matched, _used_other = _greedy_match(xy, other_xy, radius)
    return [idx not in matched for idx in range(len(rows))]


def _clean_xy_for_tile(dataset_root: Path, tile_name: str) -> np.ndarray:
    clean_table = _load_clean_table(dataset_root)
    _clean_rows, clean_x, clean_y = _local_rows(clean_table, tile_name)
    return (
        np.column_stack([clean_x, clean_y]).astype(np.float32)
        if len(clean_x)
        else np.zeros((0, 2), dtype=np.float32)
    )


def _unique_flags_by_gt_then_radius(
    rows: list[dict],
    other_rows: list[dict],
    *,
    clean_xy: np.ndarray,
    other_clean_xy: np.ndarray,
    fallback_radius: float,
) -> list[bool]:
    """Mark rows not recovered by the counterpart.

    The comparison first uses clean GT identity. A detection matched to clean GT
    is recovered only if the counterpart has a detection matched to the same GT.
    Predictions unmatched to clean GT are compared only against counterpart
    predictions that are also unmatched to clean GT, using a wider 3 arcsec
    spatial radius.
    """

    xy = _prediction_xy(rows)
    other_xy = _prediction_xy(other_rows)
    pred_to_gt, _used_gt = _greedy_match(xy, clean_xy, MATCH_RADIUS_PIX)
    other_to_gt, _other_used_gt = _greedy_match(other_xy, other_clean_xy, MATCH_RADIUS_PIX)
    other_gt_set = set(other_to_gt.values())

    unique = [True for _ in range(len(rows))]
    unmatched_rows: list[int] = []
    for idx in range(len(rows)):
        gt_idx = pred_to_gt.get(idx)
        if gt_idx is not None:
            unique[idx] = gt_idx not in other_gt_set
        else:
            unmatched_rows.append(idx)

    other_unmatched = [idx for idx in range(len(other_rows)) if idx not in other_to_gt]
    if unmatched_rows and other_unmatched:
        sub_xy = xy[np.asarray(unmatched_rows, dtype=np.int64)]
        sub_other_xy = other_xy[np.asarray(other_unmatched, dtype=np.int64)]
        matched_sub, _used_other = _greedy_match(sub_xy, sub_other_xy, fallback_radius)
        for rel_idx in matched_sub:
            unique[unmatched_rows[rel_idx]] = False
    return unique


def _color_for_detection(dataset_name: str, unique: bool) -> str:
    if dataset_name == "denoised":
        return "magenta" if unique else "green"
    if dataset_name == "noisy":
        return "orange" if unique else "green"
    if dataset_name == "coadd":
        return "blue" if unique else "cyan"
    return "cyan"


def _safe_tile_name(tile_name: str) -> str:
    return tile_name.replace(",", "_")


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _load_clean_table(dataset_root: Path) -> Table:
    return _table(dataset_root / PATCH / "band_reference_catalogs" / BAND / f"meas-{BAND}-{TRACT}-{PATCH}.fits")


def _write_dataset_tile_regs(
    *,
    out_dir: Path,
    dataset_name: str,
    dataset_root: Path,
    tile_name: str,
    pred_rows: list[dict],
    unique_flags: list[bool],
) -> dict[str, object]:
    clean_table = _load_clean_table(dataset_root)
    clean_rows, clean_x, clean_y = _local_rows(clean_table, tile_name)
    clean_xy = (
        np.column_stack([clean_x, clean_y]).astype(np.float32)
        if len(clean_x)
        else np.zeros((0, 2), dtype=np.float32)
    )
    pred_xy = _prediction_xy(pred_rows)
    _pred_to_clean, clean_used = _greedy_match(pred_xy, clean_xy, MATCH_RADIUS_PIX)
    clean_mask, background_mask, _ordinary_ignore = _load_masks(dataset_root, tile_name)
    clean_bg = clean_mask | background_mask

    prefix = f"{dataset_name}_{PATCH.replace(',', '_')}_{_safe_tile_name(tile_name)}_{BAND.replace('-', '_')}"
    ds_out = out_dir / dataset_name
    center_path = ds_out / f"{prefix}_all_detected_centers_compare.reg"
    shape_path = ds_out / f"{prefix}_all_detected_shapes_compare.reg"
    fn_path = ds_out / f"{prefix}_clean_background_fn_shapes.reg"

    legend = {
        "coadd": "cyan=matched with denoised, blue=coadd-only",
        "noisy": "green=matched with denoised, orange=noisy-only",
        "denoised": "green=matched with noisy, magenta=denoised-only",
    }[dataset_name]
    center_lines = REG_HEADER + [f"# {dataset_name} {tile_name} {BAND}: all detected centers; {legend}"]
    shape_lines = REG_HEADER + [f"# {dataset_name} {tile_name} {BAND}: all detected predicted ellipses; {legend}"]
    fn_lines = REG_HEADER + [
        f"# {dataset_name} {tile_name} {BAND}: clean/background-region FN GT ellipses; red=unmatched clean GT"
    ]

    if len(unique_flags) != len(pred_rows):
        unique_flags = [False] * len(pred_rows)
    for row, is_unique in zip(pred_rows, unique_flags):
        color = _color_for_detection(dataset_name, bool(is_unique))
        center_lines.append(_point(float(row["x"]), float(row["y"]), color, radius=3.0, width=2))
        shape_lines.append(
            _ellipse_pred(
                float(row["x"]),
                float(row["y"]),
                float(row["major"]),
                float(row["minor"]),
                float(row["theta"]),
                color,
                width=2,
            )
        )

    fn_count = 0
    for gt_idx, (row, x, y) in enumerate(zip(clean_rows, clean_x, clean_y)):
        if gt_idx in clean_used:
            continue
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if 0 <= xi < clean_bg.shape[1] and 0 <= yi < clean_bg.shape[0] and clean_bg[yi, xi]:
            fn_lines.append(_ellipse_from_row(row, float(x), float(y), "red", width=2))
            fn_count += 1

    _write_lines(center_path, center_lines)
    _write_lines(shape_path, shape_lines)
    _write_lines(fn_path, fn_lines)
    return {
        "dataset": dataset_name,
        "tile": tile_name,
        "band": BAND,
        "pred": len(pred_rows),
        "unique_vs_counterpart": int(sum(bool(flag) for flag in unique_flags)),
        "clean_gt": len(clean_rows),
        "clean_tp": len(clean_used),
        "clean_background_fn": fn_count,
        "center_reg": str(center_path),
        "shape_reg": str(shape_path),
        "fn_reg": str(fn_path),
    }


def _predict_dataset(
    *,
    dataset: DatasetSpec,
    cfg: dict,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, list[dict]], list[str]]:
    records = discover_cutout_records(dataset.root, bands=[BAND])
    records = [rec for rec in records if rec.patch == PATCH]
    if not records:
        raise RuntimeError(f"No {BAND} records found under {dataset.root} for patch {PATCH}")
    ds = AstroCutoutDataset(
        records,
        fits_hdu=int(cfg.get("fits_hdu", 1)),
        confidence_levels=5,
        ellipse_sigma=float(cfg.get("ellipse_sigma", 2.0)),
        core_radius=int(cfg.get("core_radius", 2)),
        shape_source=str(cfg.get("shape_source", "kron")),
        source_filter=str(cfg.get("source_filter", "nchild0")),
        load_eval_ignore_sources=True,
        augment=False,
    )
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False, num_workers=0, collate_fn=collate_cutouts)
    predictions = _run_predictions(model, loader, device=device, cfg=cfg)
    tile_names = sorted(str(rec.tile_name) for rec in records)
    for tile_name in tile_names:
        predictions.setdefault(tile_name, [])
    return predictions, tile_names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("output/per_band_data_filtered_0609/best.pt"))
    parser.add_argument("--config", type=Path, default=Path("output/per_band_data_filtered_0609/run_config.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("zangetsu_demo/eval_per_band_data_filtered_0609_i"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    datasets = [
        DatasetSpec("coadd", Path("zangetsu_demo/preprocessed/coadd/9813")),
        DatasetSpec("noisy", Path("zangetsu_demo/preprocessed/noisy/9813")),
        DatasetSpec("denoised", Path("zangetsu_demo/preprocessed/denoised/9813")),
    ]
    cfg = _read_config(args.config)
    device = torch.device(args.device)
    model = _make_model(cfg, args.checkpoint, device)
    metrics: list[dict[str, object]] = []
    predictions_by_dataset: dict[str, dict[str, list[dict]]] = {}
    tile_names_by_dataset: dict[str, list[str]] = {}

    for dataset in datasets:
        preds, tile_names = _predict_dataset(
            dataset=dataset,
            cfg=cfg,
            model=model,
            device=device,
            batch_size=args.batch_size,
        )
        predictions_by_dataset[dataset.name] = preds
        tile_names_by_dataset[dataset.name] = tile_names

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        for tile_name in tile_names_by_dataset[dataset.name]:
            pred_rows = predictions_by_dataset[dataset.name].get(tile_name, [])
            clean_xy = _clean_xy_for_tile(dataset.root, tile_name)
            if dataset.name == "denoised":
                other_rows = predictions_by_dataset["noisy"].get(tile_name, [])
                other_clean_xy = _clean_xy_for_tile(Path("zangetsu_demo/preprocessed/noisy/9813"), tile_name)
                flags = _unique_flags_by_gt_then_radius(
                    pred_rows,
                    other_rows,
                    clean_xy=clean_xy,
                    other_clean_xy=other_clean_xy,
                    fallback_radius=FALLBACK_COMPARE_RADIUS_PIX,
                )
            elif dataset.name == "noisy":
                other_rows = predictions_by_dataset["denoised"].get(tile_name, [])
                other_clean_xy = _clean_xy_for_tile(Path("zangetsu_demo/preprocessed/denoised/9813"), tile_name)
                flags = _unique_flags_by_gt_then_radius(
                    pred_rows,
                    other_rows,
                    clean_xy=clean_xy,
                    other_clean_xy=other_clean_xy,
                    fallback_radius=FALLBACK_COMPARE_RADIUS_PIX,
                )
            else:
                other_rows = predictions_by_dataset["denoised"].get(tile_name, [])
                other_clean_xy = _clean_xy_for_tile(Path("zangetsu_demo/preprocessed/denoised/9813"), tile_name)
                flags = _unique_flags_by_gt_then_radius(
                    pred_rows,
                    other_rows,
                    clean_xy=clean_xy,
                    other_clean_xy=other_clean_xy,
                    fallback_radius=FALLBACK_COMPARE_RADIUS_PIX,
                )
            row = _write_dataset_tile_regs(
                out_dir=args.out_dir,
                dataset_name=dataset.name,
                dataset_root=dataset.root,
                tile_name=tile_name,
                pred_rows=pred_rows,
                unique_flags=flags,
            )
            metrics.append(row)
            print(
                f"{dataset.name} {tile_name} {BAND}: "
                f"pred={row['pred']} unique={row['unique_vs_counterpart']} "
                f"clean TP/FN/GT={row['clean_tp']}/{row['clean_background_fn']}/{row['clean_gt']}"
            )

    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": _read_checkpoint_epoch(args.checkpoint),
        "band": BAND,
        "match_radius_pix": MATCH_RADIUS_PIX,
        "metrics": metrics,
    }
    fieldnames = list(metrics[0].keys()) if metrics else []
    with (args.out_dir / "i_band_compare_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)
    (args.out_dir / "i_band_compare_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote metrics: {args.out_dir / 'i_band_compare_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
