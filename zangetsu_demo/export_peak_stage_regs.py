#!/usr/bin/env python3
"""Export DS9 REG overlays for CELLECT peak-selection stages on the SAM demo cutout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Sequence

import torch
from torch.utils.data import DataLoader

CELLECT_ROOT = Path("/home/czh23/CELLECT")
if str(CELLECT_ROOT) not in sys.path:
    sys.path.insert(0, str(CELLECT_ROOT))

from astro_train_data import AstroCutoutDataset, collate_cutouts, discover_cutout_records  # noqa: E402
from astro_train_ops import (  # noqa: E402
    ORDINAL_EXPECTATION_MERGE_RADIUS,
    _compute_detection_peak_maps,
    _confidence_detection_score,
    _merge_close_centers_by_score,
)
from sam_backbone import build_sam_cellect2d  # noqa: E402


TRACT = "9813"
PATCH = "4,5"
DEFAULT_TILE_COADD = "sam_x18204_y20924"
DEFAULT_TILE_GROUP = "group_01_sam_x18204_y20924"
DEFAULT_BANDS = ("HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y")
DEFAULT_BAND = "HSC-I"
DEFAULT_DATA_ROOT = CELLECT_ROOT / "zangetsu_demo/data/sam_x18204_y20924"
DEFAULT_OUT_DIR = CELLECT_ROOT / "zangetsu_demo/output/peak_stage_regs"
REG_HEADER = [
    "# Region file format: DS9 version 4.1",
    'global color=green dashlist=8 3 width=2 font="helvetica 10 normal roman" '
    "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1",
    "image",
]
STAGE_SPECS = (
    ("spatial_localmax", "yellow"),
    ("seed_candidates", "cyan"),
    ("seed_after_foreground_gate", "green"),
    ("seed_after_threshold", "orange"),
    ("final_peaks", "red"),
    ("center_pooled_candidates", "magenta"),
    ("center_after_threshold", "blue"),
    ("ordinal_expectation_thr2p1", "white"),
    ("ordinal_expectation_thr2p0", "pink"),
    ("ordinal_expectation_thr1p5", "blue"),
    ("ordinal_prob_thr0p45", "lime"),
)


def _read_config(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    args = dict(payload.get("args", {}))
    args["_top"] = payload
    return args


def _strip_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key[7:] if str(key).startswith("module.") else key: value for key, value in state.items()}


def _checkpoint_variant(checkpoint: Path) -> str:
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    if isinstance(ckpt, dict):
        if ckpt.get("model_variant"):
            return str(ckpt["model_variant"])
        if isinstance(ckpt.get("args"), dict) and ckpt["args"].get("model_variant"):
            return str(ckpt["args"]["model_variant"])
    if isinstance(state, dict):
        first_key = next(iter(state), "")
        if str(first_key).startswith(("encoder.", "module.encoder.")):
            return "sam_per_band"
    return ""


def _resolve_config(checkpoint: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    candidates = (
        checkpoint.parent / "run_config.json",
        checkpoint.parent.parent / "run_config.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find run_config.json next to {checkpoint}. Pass --config explicitly."
    )


def _make_model(cfg: dict, checkpoint: Path, device: torch.device, bands: Sequence[str]) -> torch.nn.Module:
    top = cfg.get("_top", {})
    variant = _checkpoint_variant(checkpoint) or str(top.get("model_variant") or cfg.get("model_variant", "sam_per_band"))
    if variant != "sam_per_band":
        raise ValueError(f"{checkpoint} is model_variant={variant!r}; this exporter expects sam_per_band checkpoints")
    base_channels = int(top.get("base_channels") or cfg.get("base_channels", 32))
    model = build_sam_cellect2d(
        str(top.get("sam_model_type") or cfg.get("sam_model_type", "vit_b")),
        checkpoint=None,
        num_bands=len(bands),
        image_size=512,
        patch_size=16,
        seg_classes=int(top.get("seg_classes") or cfg.get("seg_classes", 2)),
        confidence_levels=5,
        embedding_dim=int(cfg.get("embedding_dim", 64)),
        shape_channels=3,
        decoder_channels=(base_channels * 8, base_channels * 4, base_channels * 2, base_channels),
        use_cen=bool(top.get("sam_cen_enabled", not bool(cfg.get("disable_sam_cen", False)))),
        cen_input_image=True,
        cen_width=max(2, base_channels // 4),
        candidate_count=int(cfg.get("matcher_candidate_count", 5)),
        shape_feature_dim=6,
        enable_matchers=False,
        astro_preprocess_in_model=False,
    ).to(device)

    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    incompatible = model.load_state_dict(_strip_module_prefix(state), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            f"[load] {checkpoint.name}: missing={len(incompatible.missing_keys)} "
            f"unexpected={len(incompatible.unexpected_keys)}",
            flush=True,
        )
    model.eval()
    return model


def _default_tile(dataset: str) -> str:
    return DEFAULT_TILE_COADD if str(dataset) == "coadd" else DEFAULT_TILE_GROUP


def _dataset(
    root: Path,
    dataset_name: str,
    bands: Sequence[str],
    cfg: dict,
    tile_name: str,
) -> DataLoader:
    records = discover_cutout_records(root, bands=bands)
    records = [
        rec
        for rec in records
        if str(rec.dataset_source) == str(dataset_name) and rec.patch == PATCH and rec.tile_name == tile_name
    ]
    if len(records) != 1:
        raise RuntimeError(
            f"Expected one record for {dataset_name}/{PATCH}/{tile_name}, got {len(records)} under {root}"
        )
    ds = AstroCutoutDataset(
        records,
        fits_hdu=int(cfg.get("fits_hdu", 1)),
        confidence_levels=5,
        ellipse_sigma=float(cfg.get("ellipse_sigma", 1.0)),
        core_radius=int(cfg.get("core_radius", 2)),
        shape_source=str(cfg.get("shape_source", "kron")),
        source_filter=str(cfg.get("source_filter", "nchild0")),
        load_eval_ignore_sources=True,
        augment=False,
    )
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_cutouts)


def _band_outputs(outputs: dict[str, torch.Tensor], band_idx: int) -> dict[str, torch.Tensor]:
    selected: dict[str, torch.Tensor] = {}
    for key, value in outputs.items():
        if not torch.is_tensor(value):
            continue
        if value.ndim >= 5:
            selected[key] = value[:, band_idx]
        else:
            selected[key] = value
    return selected


def _circle_line(x: float, y: float, radius: float, *, color: str, width: int = 2, text: str = "") -> str:
    suffix = f" # color={color} width={width}"
    if text:
        suffix += f" text={{{text}}}"
    return f"circle({x + 1.0:.3f},{y + 1.0:.3f},{radius:.3f}){suffix}"


def _write_reg(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _peak_stage_maps(debug: Dict[str, object]) -> Dict[str, torch.Tensor]:
    spatial_localmax = debug["spatial_localmax"]
    seed_candidates = debug["seed_candidates"]
    foreground_gate = debug["foreground_gate"]
    threshold_pass = debug["threshold_pass"]
    final_peaks = debug["final_peaks"]
    center_pooled_candidates = debug.get("center_pooled_candidates")
    assert isinstance(spatial_localmax, torch.Tensor)
    assert isinstance(seed_candidates, torch.Tensor)
    assert isinstance(foreground_gate, torch.Tensor)
    assert isinstance(threshold_pass, torch.Tensor)
    assert isinstance(final_peaks, torch.Tensor)
    assert isinstance(center_pooled_candidates, torch.Tensor)
    return {
        "spatial_localmax": spatial_localmax,
        "seed_candidates": seed_candidates,
        "seed_after_foreground_gate": seed_candidates & foreground_gate,
        "seed_after_threshold": seed_candidates & threshold_pass,
        "final_peaks": final_peaks,
        "center_pooled_candidates": center_pooled_candidates,
        "center_after_foreground_gate": center_pooled_candidates & foreground_gate,
        "center_after_threshold": center_pooled_candidates & threshold_pass,
    }


def _dedupe_peak_mask(mask: torch.Tensor, score_map: torch.Tensor, *, min_distance: float) -> torch.Tensor:
    if mask.ndim != 2 or score_map.ndim != 2:
        raise ValueError("mask and score_map must be [H,W]")
    y, x = torch.where(mask)
    if x.numel() == 0 or float(min_distance) <= 0.0:
        return mask
    coords = torch.stack([x, y], dim=1).to(dtype=torch.float32)
    scores = score_map[y, x]
    keep_coords, _keep_scores = _merge_close_centers_by_score(
        coords,
        scores,
        min_distance=float(min_distance),
    )
    out = torch.zeros_like(mask, dtype=torch.bool)
    if keep_coords.numel() == 0:
        return out
    keep_xy = torch.round(keep_coords).to(dtype=torch.long)
    keep_x = keep_xy[:, 0].clamp(0, mask.shape[1] - 1)
    keep_y = keep_xy[:, 1].clamp(0, mask.shape[0] - 1)
    out[keep_y, keep_x] = True
    return out


def _ablation_stage_maps(band_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    confidence = band_outputs["confidence"]
    ordinal_prob = _confidence_detection_score(band_outputs, "ordinal_prob")
    ordinal_prob_pooled = torch.nn.functional.max_pool2d(
        ordinal_prob.unsqueeze(1),
        kernel_size=3,
        stride=1,
        padding=1,
    ).squeeze(1)

    prob = torch.softmax(confidence.float(), dim=1).to(dtype=confidence.dtype)
    level_values = torch.arange(confidence.shape[1], device=prob.device, dtype=prob.dtype).view(1, -1, 1, 1)
    ordinal_expectation = (prob * level_values).sum(dim=1)
    ordinal_expectation_pooled = torch.nn.functional.max_pool2d(
        ordinal_expectation.unsqueeze(1),
        kernel_size=3,
        stride=1,
        padding=1,
    ).squeeze(1)
    ordinal_expectation_thr1p5 = (ordinal_expectation == ordinal_expectation_pooled) & (ordinal_expectation > 1.5)
    ordinal_expectation_thr2p1 = (ordinal_expectation == ordinal_expectation_pooled) & (ordinal_expectation > 2.1)
    ordinal_expectation_thr2p0 = (ordinal_expectation == ordinal_expectation_pooled) & (ordinal_expectation > 2.0)
    return {
        "ordinal_expectation_thr1p5": _dedupe_peak_mask(
            ordinal_expectation_thr1p5[0],
            ordinal_expectation[0],
            min_distance=float(ORDINAL_EXPECTATION_MERGE_RADIUS),
        ).unsqueeze(0),
        "ordinal_expectation_thr2p1": _dedupe_peak_mask(
            ordinal_expectation_thr2p1[0],
            ordinal_expectation[0],
            min_distance=float(ORDINAL_EXPECTATION_MERGE_RADIUS),
        ).unsqueeze(0),
        "ordinal_expectation_thr2p0": _dedupe_peak_mask(
            ordinal_expectation_thr2p0[0],
            ordinal_expectation[0],
            min_distance=float(ORDINAL_EXPECTATION_MERGE_RADIUS),
        ).unsqueeze(0),
        "ordinal_prob_thr0p45": (ordinal_prob == ordinal_prob_pooled) & (ordinal_prob > 0.45),
    }


def export_peak_stage_regs(args: argparse.Namespace) -> dict[str, object]:
    checkpoint = args.checkpoint.resolve()
    config_path = _resolve_config(checkpoint, args.config.resolve() if args.config is not None else None)
    cfg = _read_config(config_path)
    bands = tuple(args.bands)
    if args.band not in bands:
        raise ValueError(f"--band {args.band!r} must be one of {bands}")
    band_idx = bands.index(args.band)
    tile_name = str(args.tile_name) if args.tile_name else _default_tile(str(args.dataset))

    device = torch.device(args.device)
    model = _make_model(cfg, checkpoint, device, bands)
    loader = _dataset(args.data_root, str(args.dataset), bands, cfg, tile_name)
    batch = next(iter(loader))
    image = batch["image"].to(device=device, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(image)
    band_outputs = _band_outputs(outputs, band_idx)

    threshold = float(args.threshold if args.threshold is not None else cfg.get("confidence_threshold", 2.0))
    nms_radius = int(args.nms_radius if args.nms_radius is not None else cfg.get("nms_radius", 1))
    confidence_score = str(args.confidence_score if args.confidence_score is not None else cfg.get("confidence_score", "cellect"))
    center_score, _peaks, debug = _compute_detection_peak_maps(
        band_outputs,
        threshold=threshold,
        nms_radius=nms_radius,
        confidence_score=confidence_score,
    )
    stage_masks = _peak_stage_maps(debug)
    stage_masks.update(_ablation_stage_maps(band_outputs))
    if confidence_score == "ordinal_expectation":
        final_mask = stage_masks["final_peaks"][0]
        stage_masks["final_peaks"] = _dedupe_peak_mask(
            final_mask,
            center_score[0],
            min_distance=float(ORDINAL_EXPECTATION_MERGE_RADIUS),
        ).unsqueeze(0)

    checkpoint_label = args.checkpoint_label or checkpoint.stem
    prefix = (
        f"{checkpoint_label}_{args.dataset}_{PATCH.replace(',', '_')}_{tile_name}_{args.band.replace('-', '_')}"
    )
    out_dir = args.out_dir / str(args.dataset)
    combined_path = out_dir / f"{prefix}_peak_stages.reg"

    combined_lines = REG_HEADER + [
        f"# {checkpoint_label} {args.dataset} {PATCH}/{tile_name} {args.band}: CELLECT peak-selection stages"
    ]
    stage_paths: dict[str, str] = {}
    stage_counts: dict[str, int] = {}
    for stage_name, color in STAGE_SPECS:
        mask = stage_masks[stage_name][0]
        y, x = torch.where(mask)
        stage_counts[stage_name] = int(x.numel())
        stage_lines = REG_HEADER + [
            (
                f"# {checkpoint_label} {args.dataset} {PATCH}/{tile_name} {args.band}: "
                f"{stage_name} count={int(x.numel())}"
            )
        ]
        combined_lines.append(f"# {stage_name} color={color} count={int(x.numel())}")
        for xi, yi in zip(x.tolist(), y.tolist()):
            line = _circle_line(float(xi), float(yi), float(args.center_radius), color=color, width=2)
            stage_lines.append(line)
            combined_lines.append(line)
        stage_path = out_dir / f"{prefix}_{stage_name}.reg"
        _write_reg(stage_path, stage_lines)
        stage_paths[stage_name] = str(stage_path)

    _write_reg(combined_path, combined_lines)
    center_score_map = center_score[0].detach()
    return {
        "combined_reg": str(combined_path),
        "stage_regs": stage_paths,
        "stage_counts": stage_counts,
        "dataset": str(args.dataset),
        "tile_name": tile_name,
        "band": str(args.band),
        "threshold": threshold,
        "nms_radius": nms_radius,
        "confidence_score": confidence_score,
        "center_score_min": float(center_score_map.min().item()),
        "center_score_max": float(center_score_map.max().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint-label", default=None)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dataset", choices=("coadd", "denoised", "noisy"), default="coadd")
    parser.add_argument("--tile-name", default=None, help="Defaults to sam_x18204_y20924 for coadd, or group_01_sam_x18204_y20924 for denoised/noisy.")
    parser.add_argument("--bands", nargs="+", default=list(DEFAULT_BANDS))
    parser.add_argument("--band", default=DEFAULT_BAND)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--nms-radius", type=int, default=None)
    parser.add_argument("--confidence-score", choices=("cellect", "raw", "ordinal_prob", "ordinal_expectation"), default=None)
    parser.add_argument("--center-radius", type=float, default=2.0, help="DS9 circle radius in image pixels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_peak_stage_regs(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
