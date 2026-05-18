"""Export AstroCELLECT false-positive regions for selected HSC cutouts.

This script compares the single-band no-EN baseline checkpoint with the EN
checkpoint on two diagnostic cutouts:

* the SAM 512x512 cutout from ``output/hsc_astro_preprocessed``
* the last Zangetsu test tile from ``output/hsc_test``

False positives are predictions unmatched to the reference centers within the
same 0.5 arcsec / 0.168 arcsec-per-pixel radius used elsewhere in this repo.
Each output DS9 region file contains both methods, distinguished only by color.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from astro_cellect2d import AstroUNet2D, FusedEncoderMultiBandAstroCELLECT2D, MultiBandAstroCELLECT2D
from astro_train_eval import (
    AstroCutoutDataset,
    CutoutRecord,
    collate_cutouts,
    detect_centers,
    detect_centers_with_en,
    discover_cutout_records,
)


@dataclass(frozen=True)
class CutoutSpec:
    name: str
    root: Path
    fits_hdu: int


@dataclass(frozen=True)
class ExperimentSpec:
    label: str
    checkpoint: Path
    color: str
    use_en_postprocess: bool


DEFAULT_CUTOUTS = (
    CutoutSpec(
        name="sam_x18204_y20924",
        root=Path("/home/chenzunhao/CELLECT/output/hsc_astro_preprocessed"),
        fits_hdu=1,
    ),
    CutoutSpec(
        name="zangetsu_r02_c02_x27366_y7477",
        root=Path("/home/chenzunhao/CELLECT/output/hsc_test"),
        fits_hdu=0,
    ),
)

DEFAULT_EXPERIMENTS = (
    ExperimentSpec(
        label="astro_hsc_i_en",
        checkpoint=Path("/home/chenzunhao/CELLECT/output/astro_hsc_i_en/best.pt"),
        color="magenta",
        use_en_postprocess=True,
    ),
    ExperimentSpec(
        label="astro_hsc_i_cellect_conf",
        checkpoint=Path("/home/chenzunhao/CELLECT/output/astro_hsc_i_cellect_conf/best.pt"),
        color="cyan",
        use_en_postprocess=False,
    ),
)


def _build_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    bands = ckpt_args.get("bands", ["HSC-I"])
    num_bands = len(bands)
    embedding_dim = int(ckpt_args.get("embedding_dim", 64))
    base_channels = int(ckpt_args.get("base_channels", 32))
    model_variant = checkpoint.get("model_variant") if isinstance(checkpoint, dict) else None

    if model_variant in ("per_band", "fused_encoder"):
        model_cls = MultiBandAstroCELLECT2D if model_variant == "per_band" else FusedEncoderMultiBandAstroCELLECT2D
        model = model_cls(
            num_bands=num_bands,
            seg_classes=3,
            confidence_levels=5,
            embedding_dim=embedding_dim,
            base_channels=base_channels,
            shape_channels=3,
            candidate_count=int(ckpt_args.get("matcher_candidate_count", 5)),
            shape_feature_dim=6,
        ).to(device)
    else:
        model = AstroUNet2D(
            in_channels=num_bands,
            seg_classes=3,
            confidence_levels=5,
            embedding_dim=embedding_dim,
            base_channels=base_channels,
            shape_channels=3,
        ).to(device)

    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state)
    if hasattr(model, "EX") and hasattr(model, "EN") and isinstance(checkpoint, dict):
        if checkpoint.get("EX") is not None:
            model.EX.load_state_dict(checkpoint["EX"])
        if checkpoint.get("EN") is not None:
            model.EN.load_state_dict(checkpoint["EN"])
    model.eval()
    return model


def _select_record(spec: CutoutSpec, bands: Sequence[str]) -> CutoutRecord:
    records = discover_cutout_records(spec.root, bands=bands)
    matches = [record for record in records if record.name == spec.name]
    if len(matches) != 1:
        raise RuntimeError(f"expected one record named {spec.name!r} under {spec.root}, found {len(matches)}")
    return matches[0]


def _match_predictions(
    pred_xy: np.ndarray,
    gt_xy: np.ndarray,
    radius: float,
) -> Tuple[np.ndarray, np.ndarray, int, int, int]:
    pred_xy = np.asarray(pred_xy, dtype=np.float32)
    gt_xy = np.asarray(gt_xy, dtype=np.float32)
    pred_used = np.zeros(pred_xy.shape[0], dtype=bool)
    gt_used = np.zeros(gt_xy.shape[0], dtype=bool)
    if pred_xy.size == 0 or gt_xy.size == 0:
        return pred_used, gt_used, 0, int(pred_xy.shape[0]), int(gt_xy.shape[0])

    dist = np.sqrt(((pred_xy[:, None, :] - gt_xy[None, :, :]) ** 2).sum(axis=2))
    pairs: List[Tuple[float, int, int]] = []
    for pred_idx in range(dist.shape[0]):
        gt_idx = int(np.argmin(dist[pred_idx]))
        if dist[pred_idx, gt_idx] <= radius:
            pairs.append((float(dist[pred_idx, gt_idx]), pred_idx, gt_idx))
    pairs.sort()
    tp = 0
    for _distance, pred_idx, gt_idx in pairs:
        if pred_used[pred_idx] or gt_used[gt_idx]:
            continue
        pred_used[pred_idx] = True
        gt_used[gt_idx] = True
        tp += 1
    fp = int(pred_xy.shape[0] - tp)
    fn = int(gt_xy.shape[0] - tp)
    return pred_used, gt_used, tp, fp, fn


def _shape_map(outputs: Dict[str, torch.Tensor]) -> np.ndarray:
    shape = outputs["shape"]
    if shape.ndim == 5:
        return shape[0, 0].detach().cpu().numpy().astype(np.float32)
    return shape[0].detach().cpu().numpy().astype(np.float32)


def _ellipse_from_prediction(
    shape: np.ndarray,
    x: float,
    y: float,
    *,
    sigma: float,
    min_axis: float,
) -> Tuple[float, float, float]:
    h, w = shape.shape[-2:]
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    if xi < 0 or xi >= w or yi < 0 or yi >= h:
        return min_axis, min_axis, 0.0
    major = float(shape[0, yi, xi])
    minor = float(shape[1, yi, xi]) if shape.shape[0] > 1 else major
    theta = float(shape[2, yi, xi]) if shape.shape[0] > 2 else 0.0
    if not np.isfinite(major):
        major = min_axis / max(float(sigma), 1e-6)
    if not np.isfinite(minor):
        minor = major
    if not np.isfinite(theta):
        theta = 0.0
    return max(abs(major) * sigma, min_axis), max(abs(minor) * sigma, min_axis), math.degrees(theta)


@torch.no_grad()
def _run_one(
    experiment: ExperimentSpec,
    record: CutoutRecord,
    *,
    fits_hdu: int,
    device: torch.device,
    match_radius: float,
    confidence_threshold: float,
    nms_radius: int,
    confidence_score: str,
    source_filter: str,
    shape_source: str,
    ellipse_sigma: float,
    region_sigma: float,
    min_axis: float,
    en_threshold: float,
    en_candidate_count: int,
) -> Tuple[List[dict], dict]:
    model = _build_model(experiment.checkpoint, device)
    dataset = AstroCutoutDataset(
        [record],
        fits_hdu=fits_hdu,
        confidence_levels=5,
        ellipse_sigma=ellipse_sigma,
        core_radius=2,
        shape_source=shape_source,
        source_filter=source_filter,
        targets_dir=None,
        augment=False,
    )
    batch = collate_cutouts([dataset[0]])
    image = batch["image"].to(device=device, dtype=torch.float32)
    outputs = model(image)

    if experiment.use_en_postprocess and hasattr(model, "EN"):
        pred_list = detect_centers_with_en(
            model,
            outputs,
            threshold=confidence_threshold,
            nms_radius=nms_radius,
            confidence_score=confidence_score,
            match_radius=match_radius,
            candidate_count=en_candidate_count,
            en_threshold=en_threshold,
        )
        pred_xy = pred_list[0]
    elif outputs["seg_logits"].ndim == 5:
        flat = {key: value.reshape(value.shape[0] * value.shape[1], *value.shape[2:]) for key, value in outputs.items()}
        pred_xy = detect_centers(
            flat,
            threshold=confidence_threshold,
            nms_radius=nms_radius,
            confidence_score=confidence_score,
        )[0]
    else:
        pred_xy = detect_centers(
            outputs,
            threshold=confidence_threshold,
            nms_radius=nms_radius,
            confidence_score=confidence_score,
        )[0]

    gt_xy = batch["centers"][0].numpy().astype(np.float32)
    pred_used, _gt_used, tp, fp, fn = _match_predictions(pred_xy, gt_xy, match_radius)
    shape = _shape_map(outputs)

    rows: List[dict] = []
    for fp_index in np.flatnonzero(~pred_used):
        x, y = pred_xy[int(fp_index)]
        a, b, angle = _ellipse_from_prediction(shape, float(x), float(y), sigma=region_sigma, min_axis=min_axis)
        rows.append(
            {
                "cutout": record.name,
                "experiment": experiment.label,
                "color": experiment.color,
                "fp_index": int(fp_index),
                "image_x": float(x) + 1.0,
                "image_y": float(y) + 1.0,
                "ellipse_a_pix": a,
                "ellipse_b_pix": b,
                "ellipse_angle_deg": angle,
            }
        )

    metrics = {
        "cutout": record.name,
        "experiment": experiment.label,
        "checkpoint": str(experiment.checkpoint),
        "use_en_postprocess": bool(experiment.use_en_postprocess and hasattr(model, "EN")),
        "reference_count": int(gt_xy.shape[0]),
        "prediction_count": int(pred_xy.shape[0]),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
    }
    return rows, metrics


def _write_reg(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Region file format: DS9 version 4.1\n")
        handle.write(
            'global color=green dashlist=8 3 width=2 font="helvetica 14 bold roman" '
            "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n"
        )
        handle.write("image\n")
        for row in rows:
            handle.write(
                f"ellipse({row['image_x']:.3f},{row['image_y']:.3f},"
                f"{row['ellipse_a_pix']:.3f},{row['ellipse_b_pix']:.3f},"
                f"{row['ellipse_angle_deg']:.2f}) # color={row['color']}\n"
            )


def _write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    out_dir = Path("/home/chenzunhao/CELLECT/output/astro_fp_regions_en_vs_noen")
    bands = ("HSC-I",)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    match_radius = 0.5 / 0.168

    all_fp_rows: List[dict] = []
    metrics_rows: List[dict] = []
    reg_paths: Dict[str, str] = {}

    for cutout in DEFAULT_CUTOUTS:
        record = _select_record(cutout, bands=bands)
        cutout_rows: List[dict] = []
        for experiment in DEFAULT_EXPERIMENTS:
            rows, metrics = _run_one(
                experiment,
                record,
                fits_hdu=cutout.fits_hdu,
                device=device,
                match_radius=match_radius,
                confidence_threshold=0.0,
                nms_radius=1,
                confidence_score="cellect",
                source_filter="nchild0",
                shape_source="sdss",
                ellipse_sigma=2.0,
                region_sigma=3.0,
                min_axis=1.5,
                en_threshold=0.6,
                en_candidate_count=5,
            )
            cutout_rows.extend(rows)
            all_fp_rows.extend(rows)
            metrics_rows.append(metrics)
        reg_path = out_dir / f"{cutout.name}_fp_en_magenta_noen_cyan.reg"
        _write_reg(reg_path, cutout_rows)
        reg_paths[cutout.name] = str(reg_path)

    _write_csv(
        out_dir / "fp_regions.csv",
        all_fp_rows,
        (
            "cutout",
            "experiment",
            "color",
            "fp_index",
            "image_x",
            "image_y",
            "ellipse_a_pix",
            "ellipse_b_pix",
            "ellipse_angle_deg",
        ),
    )
    _write_csv(
        out_dir / "detection_counts.csv",
        metrics_rows,
        (
            "cutout",
            "experiment",
            "checkpoint",
            "use_en_postprocess",
            "reference_count",
            "prediction_count",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
        ),
    )

    summary = {
        "output_dir": str(out_dir),
        "regions": reg_paths,
        "counts_csv": str(out_dir / "detection_counts.csv"),
        "fp_csv": str(out_dir / "fp_regions.csv"),
        "match_radius_pix": match_radius,
        "match_radius_arcsec": 0.5,
        "pixel_scale_arcsec": 0.168,
        "source_filter": "nchild0",
        "region_sigma": 3.0,
        "metrics": metrics_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
