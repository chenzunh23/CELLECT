#!/usr/bin/env python3
"""Evaluate I-band checkpoints on Zangetsu demo cutouts and write ellipse REG files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from astropy.table import Table
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astro_cellect2d import MultiBandAstroCELLECT2D
from astro_train_data import AstroCutoutDataset, collate_cutouts, discover_cutout_records
from astro_train_ops import detect_centers, detect_centers_with_en, unwrap_model


TRACT = "9813"
PATCH = "6,1"
BAND = "HSC-I"
MATCH_RADIUS_PIX = 0.5 / 0.168


@dataclass(frozen=True)
class ModelSpec:
    name: str
    out_dir: Path
    checkpoint: Path
    config: Path


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path


def _read_config(path: Path) -> dict:
    return json.loads(path.read_text())["args"]


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _table(path: Path) -> Table:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Table.read(path)


def _tile_xy0(tile_name: str) -> tuple[int, int]:
    import re

    match = re.search(r"_x(-?\d+)_y(-?\d+)", tile_name)
    if not match:
        return 0, 0
    return int(match.group(1)), int(match.group(2))


def _xy_from_table(table: Table, tile_name: str) -> tuple[np.ndarray, np.ndarray]:
    x0, y0 = _tile_xy0(tile_name)
    names = set(table.colnames)
    for x_name, y_name in (
        ("base_SdssShape_x", "base_SdssShape_y"),
        ("base_SdssCentroid_x", "base_SdssCentroid_y"),
        ("base_NaiveCentroid_x", "base_NaiveCentroid_y"),
        ("deblend_psfCenter_x", "deblend_psfCenter_y"),
        ("centroid_local_x", "centroid_local_y"),
    ):
        if x_name in names and y_name in names:
            x = np.asarray(table[x_name], dtype=np.float32)
            y = np.asarray(table[y_name], dtype=np.float32)
            if x_name == "centroid_local_x":
                return x, y
            return x - float(x0), y - float(y0)
    raise KeyError(f"No supported centroid columns in {table.colnames}")


def _local_rows(table: Table, tile_name: str, size: int = 512) -> tuple[Table, np.ndarray, np.ndarray]:
    x, y = _xy_from_table(table, tile_name)
    keep = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < size) & (y >= 0) & (y < size)
    return table[keep], x[keep], y[keep]


def _ellipse_from_row(row, x: float, y: float, color: str, width: int = 2) -> str:
    names = set(row.colnames) if hasattr(row, "colnames") else set()
    major = _safe_float(row["ellipse_major_sigma"], 4.0) if "ellipse_major_sigma" in names else 4.0
    minor = _safe_float(row["ellipse_minor_sigma"], 4.0) if "ellipse_minor_sigma" in names else 4.0
    theta = _safe_float(row["ellipse_theta"], 0.0) if "ellipse_theta" in names else 0.0
    if not math.isfinite(major) or major <= 0:
        major = 4.0
    if not math.isfinite(minor) or minor <= 0:
        minor = 4.0
    if not math.isfinite(theta):
        theta = 0.0
    return f"ellipse({x + 1:.3f},{y + 1:.3f},{major:.3f},{minor:.3f},{math.degrees(theta):.3f}) # color={color} width={width}"


def _ellipse_pred(x: float, y: float, major: float, minor: float, theta: float, color: str, width: int = 2) -> str:
    major = max(abs(float(major)), 1.0)
    minor = max(abs(float(minor)), 1.0)
    theta = float(theta) if math.isfinite(float(theta)) else 0.0
    return f"ellipse({x + 1:.3f},{y + 1:.3f},{major:.3f},{minor:.3f},{math.degrees(theta):.3f}) # color={color} width={width}"


def _point(x: float, y: float, color: str, radius: float = 3.0, width: int = 2) -> str:
    return f"circle({x + 1:.3f},{y + 1:.3f},{radius:.3f}) # color={color} width={width}"


def _greedy_match(pred_xy: np.ndarray, gt_xy: np.ndarray, radius: float) -> tuple[dict[int, int], set[int]]:
    if pred_xy.size == 0 or gt_xy.size == 0:
        return {}, set()
    pairs: list[tuple[float, int, int]] = []
    r2 = float(radius) ** 2
    for pi, pred in enumerate(pred_xy):
        d2 = np.sum((gt_xy - pred[None, :]) ** 2, axis=1)
        for gi in np.flatnonzero(d2 <= r2):
            pairs.append((float(d2[gi]), int(pi), int(gi)))
    pairs.sort(key=lambda item: item[0])
    pred_to_gt: dict[int, int] = {}
    used_gt: set[int] = set()
    for _dist, pi, gi in pairs:
        if pi in pred_to_gt or gi in used_gt:
            continue
        pred_to_gt[pi] = gi
        used_gt.add(gi)
    return pred_to_gt, used_gt


def _load_masks(root: Path, tile_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = root / PATCH / "band_targets" / BAND / f"{tile_name}.npz"
    with np.load(path) as data:
        clean = np.asarray(data["clean_mask"], dtype=bool) if "clean_mask" in data else np.zeros((512, 512), bool)
        background = np.asarray(data["background_mask"], dtype=bool) if "background_mask" in data else np.zeros((512, 512), bool)
        ignore = np.asarray(data["ignore_mask"], dtype=bool) if "ignore_mask" in data else np.zeros((512, 512), bool)
        strict = np.asarray(data["strict_ignore_mask"], dtype=bool) if "strict_ignore_mask" in data else np.zeros((512, 512), bool)
    return clean, background, ignore & ~strict


def _flatten_outputs(outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.reshape(value.shape[0] * value.shape[1], *value.shape[2:]) for key, value in outputs.items()}


@torch.no_grad()
def _run_predictions(model: torch.nn.Module, loader: DataLoader, *, device: torch.device, cfg: dict) -> dict[str, list[dict]]:
    model.eval()
    base_model = unwrap_model(model)
    predictions: dict[str, list[dict]] = {}
    threshold = float(cfg.get("confidence_threshold", 2.0))
    nms_radius = int(cfg.get("nms_radius", 1))
    confidence_score = str(cfg.get("confidence_score", "cellect"))
    center_refinement = str(cfg.get("center_refinement", "softargmax"))
    center_refinement_radius = int(cfg.get("center_refinement_radius", 1))
    use_en = bool(cfg.get("use_en_postprocess", True))
    en_threshold = float(cfg.get("en_postprocess_threshold", 0.6))
    candidate_count = int(cfg.get("matcher_candidate_count", 5))
    for batch in loader:
        image = batch["image"].to(device=device, dtype=torch.float32)
        outputs = model(image)
        if use_en and hasattr(base_model, "EN"):
            pred_list = detect_centers_with_en(
                base_model,
                outputs,
                threshold=threshold,
                nms_radius=nms_radius,
                confidence_score=confidence_score,
                match_radius=MATCH_RADIUS_PIX,
                candidate_count=candidate_count,
                en_threshold=en_threshold,
                center_refinement=center_refinement,
                center_refinement_radius=center_refinement_radius,
            )
        elif outputs["seg_logits"].ndim == 5:
            pred_list = detect_centers(
                _flatten_outputs(outputs),
                threshold=threshold,
                nms_radius=nms_radius,
                confidence_score=confidence_score,
                center_refinement=center_refinement,
                center_refinement_radius=center_refinement_radius,
            )
        else:
            pred_list = detect_centers(
                outputs,
                threshold=threshold,
                nms_radius=nms_radius,
                confidence_score=confidence_score,
                center_refinement=center_refinement,
                center_refinement_radius=center_refinement_radius,
            )
        shape = outputs["shape"]
        if shape.ndim == 5:
            shape = shape[:, 0]
        for item_idx, pred_xy in enumerate(pred_list):
            tile_name = str(batch["tile_name"][item_idx])
            rows = predictions.setdefault(tile_name, [])
            for xy in np.asarray(pred_xy, dtype=np.float32).reshape(-1, 2):
                xi = int(round(float(xy[0])))
                yi = int(round(float(xy[1])))
                if xi < 0 or yi < 0 or yi >= shape.shape[-2] or xi >= shape.shape[-1]:
                    continue
                shp = shape[item_idx, :, yi, xi].detach().cpu().numpy().astype(float)
                rows.append({"x": float(xy[0]), "y": float(xy[1]), "major": shp[0], "minor": shp[1], "theta": shp[2]})
    return predictions


def _make_model(cfg: dict, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = MultiBandAstroCELLECT2D(
        num_bands=1,
        seg_classes=int(cfg.get("seg_classes", 2)),
        confidence_levels=5,
        embedding_dim=int(cfg.get("embedding_dim", 64)),
        base_channels=int(cfg.get("base_channels", 32)),
        shape_channels=3,
        candidate_count=int(cfg.get("matcher_candidate_count", 5)),
        shape_feature_dim=6,
    ).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    if isinstance(ckpt, dict):
        if hasattr(model, "EX") and ckpt.get("EX") is not None:
            model.EX.load_state_dict(ckpt["EX"])
        if hasattr(model, "EN") and ckpt.get("EN") is not None:
            model.EN.load_state_dict(ckpt["EN"])
    return model


def _write_tile_regs(
    *,
    out_dir: Path,
    model_name: str,
    dataset: DatasetSpec,
    tile_name: str,
    pred_rows: list[dict],
) -> dict[str, object]:
    clean_table = _table(dataset.root / PATCH / "band_reference_catalogs" / BAND / f"meas-{BAND}-{TRACT}-{PATCH}.fits")
    ignore_path = dataset.root / PATCH / "band_reference_ignore" / BAND / f"meas-{BAND}-{TRACT}-{PATCH}.fits"
    ignore_table = _table(ignore_path) if ignore_path.exists() else Table()
    clean_rows, clean_x, clean_y = _local_rows(clean_table, tile_name)
    if len(ignore_table):
        ignore_rows, ignore_x, ignore_y = _local_rows(ignore_table, tile_name)
    else:
        ignore_rows, ignore_x, ignore_y = Table(), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    pred_xy = np.asarray([[row["x"], row["y"]] for row in pred_rows], dtype=np.float32).reshape(-1, 2)
    clean_xy = np.column_stack([clean_x, clean_y]).astype(np.float32) if len(clean_x) else np.zeros((0, 2), np.float32)
    ignore_xy = np.column_stack([ignore_x, ignore_y]).astype(np.float32) if len(ignore_x) else np.zeros((0, 2), np.float32)
    pred_to_clean, clean_used = _greedy_match(pred_xy, clean_xy, MATCH_RADIUS_PIX)
    remaining = [idx for idx in range(len(pred_xy)) if idx not in pred_to_clean]
    if remaining:
        rem_to_ignore, ignore_used = _greedy_match(pred_xy[remaining], ignore_xy, MATCH_RADIUS_PIX)
        pred_to_ignore = {remaining[ri]: gi for ri, gi in rem_to_ignore.items()}
    else:
        ignore_used = set()
        pred_to_ignore = {}

    clean_mask, background_mask, ordinary_ignore_mask = _load_masks(dataset.root, tile_name)
    clean_bg = clean_mask | background_mask
    header = [
        "# Region file format: DS9 version 4.1",
        'global color=green dashlist=8 3 width=2 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1',
        "image",
    ]
    gt_lines = header + [f"# {model_name} {dataset.name} {tile_name} {BAND}: clean GT ellipses and ordinary ignore centers"]
    tp_lines = header + [f"# {model_name} {dataset.name} {tile_name} {BAND}: clean/ordinary TP ellipses"]
    det_lines = header + [f"# {model_name} {dataset.name} {tile_name} {BAND}: FN centers and unmatched FP predicted ellipses"]
    pred_lines = header + [f"# {model_name} {dataset.name} {tile_name} {BAND}: all predicted ellipses"]

    for row, x, y in zip(clean_rows, clean_x, clean_y):
        gt_lines.append(_ellipse_from_row(row, float(x), float(y), "green", width=2))
    for x, y in zip(ignore_x, ignore_y):
        gt_lines.append(_point(float(x), float(y), "yellow", radius=3.0, width=2))

    for pi, row in enumerate(pred_rows):
        if pi in pred_to_clean:
            color = "cyan"
        elif pi in pred_to_ignore:
            color = "yellow"
        else:
            xi = int(round(row["x"]))
            yi = int(round(row["y"]))
            if 0 <= xi < 512 and 0 <= yi < 512 and clean_bg[yi, xi]:
                color = "magenta"
            elif 0 <= xi < 512 and 0 <= yi < 512 and ordinary_ignore_mask[yi, xi]:
                color = "orange"
            else:
                color = "gray"
        pred_lines.append(_ellipse_pred(row["x"], row["y"], row["major"], row["minor"], row["theta"], color, width=2))

    for gi in sorted(clean_used):
        row = clean_rows[gi]
        tp_lines.append(_ellipse_from_row(row, float(clean_x[gi]), float(clean_y[gi]), "cyan", width=2))
    for gi in sorted(ignore_used):
        tp_lines.append(_point(float(ignore_x[gi]), float(ignore_y[gi]), "yellow", radius=3.0, width=2))
    for gi in range(len(clean_rows)):
        if gi not in clean_used:
            det_lines.append(_point(float(clean_x[gi]), float(clean_y[gi]), "red", radius=3.5, width=2))

    clean_fp = 0
    ordinary_fp = 0
    for pi, row in enumerate(pred_rows):
        if pi in pred_to_clean or pi in pred_to_ignore:
            continue
        xi = int(round(row["x"]))
        yi = int(round(row["y"]))
        if 0 <= xi < 512 and 0 <= yi < 512 and clean_bg[yi, xi]:
            det_lines.append(_ellipse_pred(row["x"], row["y"], row["major"], row["minor"], row["theta"], "magenta", width=2))
            clean_fp += 1
        elif 0 <= xi < 512 and 0 <= yi < 512 and ordinary_ignore_mask[yi, xi]:
            det_lines.append(_ellipse_pred(row["x"], row["y"], row["major"], row["minor"], row["theta"], "orange", width=2))
            ordinary_fp += 1

    tile_out = out_dir / model_name / dataset.name
    tile_out.mkdir(parents=True, exist_ok=True)
    safe_tile = tile_name.replace(",", "_")
    prefix = f"{dataset.name}_{PATCH.replace(',', '_')}_{safe_tile}_{BAND.replace('-', '_')}"
    paths = {
        "gt_reg": tile_out / f"{prefix}_clean_gt_ordinary_ignore.reg",
        "tp_reg": tile_out / f"{prefix}_clean_ordinary_tp.reg",
        "det_reg": tile_out / f"{prefix}_fn_fp_clean_ordinary.reg",
        "pred_reg": tile_out / f"{prefix}_predicted_ellipses.reg",
    }
    paths["gt_reg"].write_text("\n".join(gt_lines) + "\n")
    paths["tp_reg"].write_text("\n".join(tp_lines) + "\n")
    paths["det_reg"].write_text("\n".join(det_lines) + "\n")
    paths["pred_reg"].write_text("\n".join(pred_lines) + "\n")

    return {
        "model": model_name,
        "dataset": dataset.name,
        "tile": tile_name,
        "band": BAND,
        "clean_GT": len(clean_rows),
        "clean_TP": len(clean_used),
        "clean_FP": clean_fp,
        "clean_FN": len(clean_rows) - len(clean_used),
        "ordinary_GT": len(ignore_rows),
        "ordinary_TP": len(ignore_used),
        "ordinary_FP": ordinary_fp,
        "ordinary_FN": len(ignore_rows) - len(ignore_used),
        "total_pred": len(pred_rows),
        **{key: str(value) for key, value in paths.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("zangetsu_demo/eval_i_band_shape_regs"))
    parser.add_argument("--checkpoint-name", default="best.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    models = [
        ModelSpec("per_band_b5_new", Path("output/per_band_b5_new"), Path("output/per_band_b5_new") / args.checkpoint_name, Path("output/per_band_b5_new/run_config.json")),
        ModelSpec("per_band_b5_optimized", Path("output/per_band_b5_optimized"), Path("output/per_band_b5_optimized") / args.checkpoint_name, Path("output/per_band_b5_optimized/run_config.json")),
    ]
    datasets = [
        DatasetSpec("coadd", Path("zangetsu_demo/preprocessed/coadd/9813")),
        DatasetSpec("noisy", Path("zangetsu_demo/preprocessed/noisy/9813")),
        DatasetSpec("denoised", Path("zangetsu_demo/preprocessed/denoised/9813")),
    ]
    device = torch.device(args.device)
    all_metrics: list[dict[str, object]] = []
    for model_spec in models:
        cfg = _read_config(model_spec.config)
        model = _make_model(cfg, model_spec.checkpoint, device)
        for dataset in datasets:
            records = discover_cutout_records(dataset.root, bands=[BAND])
            records = [rec for rec in records if rec.patch == PATCH]
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
            loader = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=collate_cutouts)
            predictions = _run_predictions(model, loader, device=device, cfg=cfg)
            for tile_name in sorted(predictions):
                metrics = _write_tile_regs(
                    out_dir=args.out_dir,
                    model_name=model_spec.name,
                    dataset=dataset,
                    tile_name=tile_name,
                    pred_rows=predictions[tile_name],
                )
                all_metrics.append(metrics)
                print(
                    f"{model_spec.name} {dataset.name} {tile_name}: "
                    f"clean TP/FP/GT={metrics['clean_TP']}/{metrics['clean_FP']}/{metrics['clean_GT']} "
                    f"ordinary TP/FP/GT={metrics['ordinary_TP']}/{metrics['ordinary_FP']}/{metrics['ordinary_GT']} "
                    f"pred={metrics['total_pred']}"
                )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_metrics[0].keys()) if all_metrics else []
    with (args.out_dir / "i_band_shape_reg_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metrics)
    (args.out_dir / "i_band_shape_reg_metrics.json").write_text(json.dumps(all_metrics, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
