"""
Train and evaluate the 2D CELLECT-style astronomy model on LSST/HSC FITS cutouts.

The dense losses intentionally keep CELLECT's original constants.  Helper code
for dataset construction, losses, detection, and epoch execution lives in
astro_train_data.py and astro_train_ops.py so this file stays focused on CLI,
model construction, and distributed training orchestration.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from astro_cellect2d import AstroUNet2D, FusedEncoderMultiBandAstroCELLECT2D, MultiBandAstroCELLECT2D
from astro_match_eval import parse_ex_band_pairs as parse_matcher_ex_band_pairs
from astro_train_data import (
    AstroCutoutDataset,
    CutoutRecord,
    _expand_path,
    _record_name_aliases,
    collate_cutouts,
    discover_cutout_records,
    load_meas_catalog,
    make_targets,
    split_records,
)
from astro_train_ops import (
    HardTripletLoss,
    LossWeights,
    build_cellect_style_segmentation,
    center_localization_loss,
    detect_centers,
    detect_centers_with_en,
    detect_centers_with_ex_link,
    dense_losses,
    dense_losses_any,
    embedding_triplet_loss,
    en_deduplicate_centers,
    evaluate_detection,
    generate_pu_pseudo_labels,
    match_points,
    matcher_classification_loss,
    parse_ex_band_pairs,
    run_epoch,
    unwrap_model,
    validate_epoch,
)


class _DDPModelWrapper(nn.Module):
    """Expose matcher parameters to DDP even though EX/EN are used in the loss."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self._ddp_wrapped_model = model

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self._ddp_wrapped_model(image)
        dummy = image.new_zeros(())
        for name in ("EX", "EN"):
            matcher = getattr(self._ddp_wrapped_model, name, None)
            if matcher is None:
                continue
            for param in matcher.parameters():
                dummy = dummy + param.sum() * 0.0
        if dummy.requires_grad:
            outputs = dict(outputs)
            outputs["seg_logits"] = outputs["seg_logits"] + dummy
        return outputs


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _distributed_requested(args: argparse.Namespace) -> bool:
    return bool(args.ddp) or _env_int("WORLD_SIZE", 1) > 1


def _setup_distributed(args: argparse.Namespace) -> Tuple[bool, int, int, int, torch.device, str]:
    distributed = _distributed_requested(args)
    local_rank = _env_int("LOCAL_RANK", args.local_rank)
    backend = args.dist_backend
    if backend == "auto":
        backend = "nccl" if torch.cuda.is_available() and torch.device(args.device).type == "cuda" else "gloo"

    if distributed:
        dist.init_process_group(
            backend=backend,
            init_method=args.dist_url,
            timeout=timedelta(minutes=max(1.0, float(args.ddp_timeout_minutes))),
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    requested = torch.device(args.device)
    if distributed and requested.type == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = requested
    return distributed, rank, world_size, local_rank, device, backend


def _is_main(rank: int) -> bool:
    return int(rank) == 0


def _sync_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _wrap_ddp(model: nn.Module, args: argparse.Namespace, device: torch.device, local_rank: int) -> nn.Module:
    if device.type == "cuda":
        device_ids = [local_rank]
        output_device = local_rank
    else:
        device_ids = None
        output_device = None
    return DistributedDataParallel(
        _DDPModelWrapper(model),
        device_ids=device_ids,
        output_device=output_device,
        find_unused_parameters=False,
    )


def _parse_patch_specs(specs: Iterable[str], patch_file: str | None = None) -> set[str]:
    patches: set[str] = set()
    for spec in specs:
        text = str(spec).strip()
        if text:
            patches.add(text.strip("/"))
    if patch_file:
        for line in Path(patch_file).expanduser().read_text(encoding="utf-8").splitlines():
            text = line.split("#", 1)[0].strip()
            if text:
                patches.add(text.strip("/"))
    return patches


def _record_patch_aliases(rec: CutoutRecord, root: Path) -> set[str]:
    aliases: set[str] = set()
    if rec.patch:
        aliases.add(rec.patch)
        if rec.tract:
            aliases.add(f"{rec.tract}/{rec.patch}")
        elif root.name:
            aliases.add(f"{root.name}/{rec.patch}")
    if rec.relative_root:
        aliases.add(str(rec.relative_root).strip("/"))
    parts = Path(rec.relative_root).parts
    if len(parts) >= 2:
        aliases.add(f"{parts[-2]}/{parts[-1]}")
        aliases.add(str(parts[-1]))
    return aliases


def _filter_records_by_patches(records: Sequence[CutoutRecord], patches: set[str], root: Path) -> list[CutoutRecord]:
    if not patches:
        return list(records)
    wanted = {patch.strip("/") for patch in patches if patch.strip("/")}
    return [rec for rec in records if _record_patch_aliases(rec, root) & wanted]


def _record_patch_label(rec: CutoutRecord, root: Path) -> str:
    if rec.tract and rec.patch:
        return f"{rec.tract}/{rec.patch}"
    if rec.patch and root.name:
        return f"{root.name}/{rec.patch}"
    return rec.patch or rec.relative_root or "-"


def _loader_kwargs(args: argparse.Namespace, *, shuffle: bool, sampler: DistributedSampler | None = None) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "sampler": sampler,
        "num_workers": args.num_workers,
        "collate_fn": collate_cutouts,
        "pin_memory": bool(args.pin_memory),
    }
    if int(args.num_workers) > 0:
        kwargs["persistent_workers"] = bool(args.persistent_workers)
        kwargs["prefetch_factor"] = max(1, int(args.prefetch_factor))
    return kwargs


def _format_float(value: float, digits: int) -> str:
    if not np.isfinite(value):
        return ""
    return f"{float(value):.{int(digits)}f}"


def _band_name(band_idx: int, band_names: Sequence[str]) -> str:
    if 0 <= int(band_idx) < len(band_names):
        return str(band_names[int(band_idx)])
    return f"band{int(band_idx)}"


def _wcs_for_path(path: str, fits_hdu: int, cache: dict[tuple[str, int], object]) -> object | None:
    key = (str(path), int(fits_hdu))
    if key in cache:
        value = cache[key]
        return value if value is not False else None
    try:
        from astropy.io import fits
        from astropy.wcs import WCS
    except Exception:
        cache[key] = False
        return None

    try:
        with fits.open(path, memmap=True) as hdul:
            header = hdul[int(fits_hdu)].header
            wcs = WCS(header).celestial
            if not wcs.has_celestial:
                cache[key] = False
                return None
            cache[key] = wcs
            return wcs
    except Exception:
        cache[key] = False
        return None


def _radec_from_wcs(wcs: object | None, x: float, y: float) -> tuple[float, float]:
    if wcs is None:
        return float("nan"), float("nan")
    try:
        ra, dec = wcs.all_pix2world([float(x)], [float(y)], 0)  # type: ignore[attr-defined]
        return float(ra[0]), float(dec[0])
    except Exception:
        return float("nan"), float("nan")


def _flat_per_band_outputs(outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.reshape(value.shape[0] * value.shape[1], *value.shape[2:]) for key, value in outputs.items()}


@torch.no_grad()
def _write_eval_sources_csv(
    model: nn.Module,
    loader: DataLoader,
    path: Path,
    *,
    device: torch.device,
    fits_hdu: int,
    threshold: float,
    nms_radius: int,
    confidence_score: str,
    center_refinement: str,
    center_refinement_radius: int,
    match_radius: float,
    use_en_postprocess: bool,
    en_candidate_count: int,
    en_threshold: float,
    use_ex_link_postprocess: bool,
    ex_link_threshold: float,
    ex_band_pairs: Sequence[Tuple[int, int]] | None,
    band_names: Sequence[str],
    show_progress: bool,
) -> int:
    model.eval()
    base_model = unwrap_model(model)
    rows: list[dict[str, object]] = []
    wcs_cache: dict[tuple[str, int], object] = {}
    source_id = 0

    def append_row(
        batch: dict[str, object],
        item_idx: int,
        source_type: str,
        x: float,
        y: float,
        *,
        band_idx: int,
        source_index: int,
        member_count: int = 1,
        member_bands: str = "",
        member_centers: object = "",
    ) -> None:
        nonlocal source_id
        def _mask_at(mask_key: str, band_mask_key: str) -> bool:
            mask_obj: object | None = None
            if band_idx >= 0 and band_mask_key in batch:
                band_masks = batch[band_mask_key]  # type: ignore[index]
                try:
                    mask_obj = band_masks[item_idx, band_idx]  # type: ignore[index]
                except Exception:
                    mask_obj = None
            if mask_obj is None and mask_key in batch:
                masks = batch[mask_key]  # type: ignore[index]
                try:
                    mask_obj = masks[item_idx]  # type: ignore[index]
                except Exception:
                    mask_obj = None
            if mask_obj is None:
                return False
            try:
                mask_np = mask_obj.detach().cpu().numpy().astype(bool)  # type: ignore[attr-defined]
            except Exception:
                mask_np = np.asarray(mask_obj, dtype=bool)
            if mask_np.ndim != 2:
                return False
            xi = int(round(float(x)))
            yi = int(round(float(y)))
            if xi < 0 or yi < 0 or yi >= mask_np.shape[0] or xi >= mask_np.shape[1]:
                return False
            return bool(mask_np[yi, xi])

        image_paths = batch["image_paths"][item_idx]  # type: ignore[index]
        image_path = ""
        if isinstance(image_paths, (list, tuple)) and image_paths:
            safe_band_idx = min(max(int(band_idx), 0), len(image_paths) - 1)
            image_path = str(image_paths[safe_band_idx])
        wcs = _wcs_for_path(image_path, fits_hdu, wcs_cache) if image_path else None
        ra_deg, dec_deg = _radec_from_wcs(wcs, x, y)
        x0 = int(batch["x0"][item_idx])  # type: ignore[index]
        y0 = int(batch["y0"][item_idx])  # type: ignore[index]
        strict_ignored = _mask_at("strict_ignore_mask", "band_strict_ignore_mask")
        ignored = _mask_at("ignore_mask", "band_ignore_mask") or strict_ignored
        ordinary_ignored = ignored and not strict_ignored
        source_id += 1
        rows.append(
            {
                "source_id": source_id,
                "record": batch["name"][item_idx],  # type: ignore[index]
                "tract": batch["tract"][item_idx],  # type: ignore[index]
                "patch": batch["patch"][item_idx],  # type: ignore[index]
                "tile_name": batch["tile_name"][item_idx],  # type: ignore[index]
                "source_type": source_type,
                "source_index": int(source_index),
                "band": _band_name(band_idx, band_names) if band_idx >= 0 else "",
                "band_index": int(band_idx) if band_idx >= 0 else "",
                "x_local": _format_float(x, 6),
                "y_local": _format_float(y, 6),
                "x_parent": _format_float(float(x0) + float(x), 6),
                "y_parent": _format_float(float(y0) + float(y), 6),
                "ra_deg": _format_float(ra_deg, 10),
                "dec_deg": _format_float(dec_deg, 10),
                "ignored_by_mask": int(ignored),
                "ordinary_ignore": int(ordinary_ignored),
                "strict_ignore": int(strict_ignored),
                "eval_excluded_by_mask": int(ignored),
                "member_count": int(member_count),
                "member_bands": member_bands,
                "member_centers": json.dumps(member_centers) if member_centers else "",
                "image_path": image_path,
            }
        )

    for batch in tqdm(loader, desc="eval-csv", leave=False, disable=not show_progress):
        image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)  # type: ignore[union-attr]
        outputs = model(image)
        if outputs["seg_logits"].ndim == 5 and use_ex_link_postprocess and hasattr(base_model, "EX"):
            pred_list, components_all = detect_centers_with_ex_link(
                base_model,
                outputs,
                threshold=threshold,
                nms_radius=nms_radius,
                confidence_score=confidence_score,
                match_radius=match_radius,
                candidate_count=en_candidate_count,
                ex_threshold=ex_link_threshold,
                center_refinement=center_refinement,
                center_refinement_radius=center_refinement_radius,
                use_en_postprocess=use_en_postprocess,
                en_threshold=en_threshold,
                band_pairs=ex_band_pairs,
            )
            for item_idx, pred_xy in enumerate(pred_list):
                components = components_all[item_idx] if item_idx < len(components_all) else []
                for source_index, xy in enumerate(np.asarray(pred_xy, dtype=np.float32)):
                    component = components[source_index] if source_index < len(components) else {}
                    members = component.get("members", []) if isinstance(component, dict) else []
                    band_idx = int(members[0][0]) if members else 0
                    member_centers = component.get("member_centers", "") if isinstance(component, dict) else ""
                    member_band_names = []
                    if isinstance(members, list):
                        member_band_names = [_band_name(int(item[0]), band_names) for item in members]
                    append_row(
                        batch,
                        item_idx,
                        "linked",
                        float(xy[0]),
                        float(xy[1]),
                        band_idx=band_idx,
                        source_index=source_index,
                        member_count=len(members) if members else 1,
                        member_bands=",".join(member_band_names),
                        member_centers=member_centers,
                    )
            continue

        band_count = int(outputs["seg_logits"].shape[1]) if outputs["seg_logits"].ndim == 5 else 0
        if use_en_postprocess and hasattr(base_model, "EN"):
            pred_list = detect_centers_with_en(
                base_model,
                outputs,
                threshold=threshold,
                nms_radius=nms_radius,
                confidence_score=confidence_score,
                match_radius=match_radius,
                candidate_count=en_candidate_count,
                en_threshold=en_threshold,
                center_refinement=center_refinement,
                center_refinement_radius=center_refinement_radius,
            )
        elif outputs["seg_logits"].ndim == 5:
            pred_list = detect_centers(
                _flat_per_band_outputs(outputs),
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

        for list_idx, pred_xy in enumerate(pred_list):
            item_idx = list_idx // band_count if band_count else list_idx
            band_idx = list_idx % band_count if band_count else 0
            source_type = "band" if band_count else "fused"
            for source_index, xy in enumerate(np.asarray(pred_xy, dtype=np.float32)):
                append_row(
                    batch,
                    item_idx,
                    source_type,
                    float(xy[0]),
                    float(xy[1]),
                    band_idx=band_idx,
                    source_index=source_index,
                    member_bands=_band_name(band_idx, band_names) if band_count else "",
                )

    fieldnames = [
        "source_id",
        "record",
        "tract",
        "patch",
        "tile_name",
        "source_type",
        "source_index",
        "band",
        "band_index",
        "x_local",
        "y_local",
        "x_parent",
        "y_parent",
        "ra_deg",
        "dec_deg",
        "ignored_by_mask",
        "ordinary_ignore",
        "strict_ignore",
        "eval_excluded_by_mask",
        "member_count",
        "member_bands",
        "member_centers",
        "image_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _checkpoint_payload(
    model: nn.Module,
    *,
    model_variant: str,
    epoch: int,
    args: argparse.Namespace,
    weights: LossWeights,
    center_radius_px: float,
    val_metrics: dict[str, float],
    det_metrics: dict[str, object],
) -> dict[str, object]:
    base_model = unwrap_model(model)
    return {
        "model": base_model.state_dict(),
        "EX": base_model.EX.state_dict() if hasattr(base_model, "EX") else None,
        "EN": base_model.EN.state_dict() if hasattr(base_model, "EN") else None,
        "model_variant": model_variant,
        "epoch": epoch,
        "args": vars(args),
        "loss_weights": asdict(weights),
        "center_radius_px": center_radius_px,
        "val": val_metrics,
        "detection": det_metrics,
    }


def _state_dict_cpu_copy(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in unwrap_model(model).state_dict().items()}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/evaluate AstroCELLECT2D on LSST/HSC FITS cutouts.")
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument(
        "--root",
        default="~/segment-anything/lsst_pipeline/output/cutout_magnitude_experiment_grid",
        help="Root containing cutouts/ and reference_catalogs/, or <tract>/<patch>/cutouts and reference_catalogs.",
    )
    parser.add_argument("--reference-dir", default=None)
    parser.add_argument("--cutout-dir", default=None)
    parser.add_argument(
        "--band-reference-root",
        default=None,
        help="Optional root containing per-band meas catalogs, e.g. catalog/HSC-I/meas-HSC-I-*.fits. "
        "When set, each band uses its own catalog ids/centers for EX/EN training.",
    )
    parser.add_argument(
        "--targets-dir",
        default=None,
        help="Optional directory containing precomputed <tile>.npz dense targets from astro_data_preprocessing.py.",
    )
    parser.add_argument("--bands", nargs="+", default=("HSC-G", "HSC-R", "HSC-I"))
    parser.add_argument("--fits-hdu", type=int, default=1)
    parser.add_argument("--out-dir", default="./output/astro_cellect2d")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true", help="Enable pinned-memory DataLoader transfers to CUDA.")
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        help="Keep DataLoader worker processes alive across epochs. Only used when --num-workers > 0.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="DataLoader prefetch factor when --num-workers > 0.",
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument(
        "--seg-classes",
        type=int,
        default=2,
        help="Number of segmentation logits in the model. New PU training uses 2; set 3 only for old checkpoints.",
    )
    parser.add_argument(
        "--model-variant",
        choices=("auto", "fused", "per_band", "fused_encoder"),
        default="auto",
        help="fused treats bands as channels and outputs one map. per_band runs one shared single-band backbone per band. "
        "fused_encoder runs one multi-band encoder and lightweight per-band heads for EX/EN. "
        "auto uses fused_encoder for multi-band data or when EN is enabled.",
    )
    parser.add_argument(
        "--single-band-detector",
        action="store_true",
        help=(
            "Run all requested bands through one shared 1-channel detector as independent samples, while keeping "
            "per-band outputs and metrics. This forces --model-variant per_band and disables EX/linking."
        ),
    )
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--train-patches",
        nargs="*",
        default=(),
        help="Restrict --mode train training records to specific patches, e.g. 0,0 0,1 or 9813/0,0.",
    )
    parser.add_argument(
        "--train-patches-file",
        default=None,
        help="Optional text file with one train patch per line. Lines may use 0,0 or 9813/0,0; # comments are ignored.",
    )
    parser.add_argument(
        "--val-patches",
        nargs="*",
        default=(),
        help="Restrict --mode train validation records to specific patches, e.g. 6,1 or 9813/6,1.",
    )
    parser.add_argument(
        "--val-patches-file",
        default=None,
        help="Optional text file with one validation patch per line. Lines may use 6,1 or 9813/6,1; # comments are ignored.",
    )
    parser.add_argument(
        "--eval-patches",
        nargs="*",
        default=(),
        help="Restrict --mode eval to specific patches. Accepts patch names like 8,8 or tract/patch like 9813/8,8.",
    )
    parser.add_argument(
        "--eval-patches-file",
        default=None,
        help="Optional text file with one eval patch per line. Lines may use 8,8 or 9813/8,8; # comments are ignored.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument(
        "--fixed-val-names",
        nargs="*",
        default=("sam_x18204_y20924",),
        help="Tile names that must be placed in the validation set. Default keeps the SAM comparison cutout in validation.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ellipse-sigma", type=float, default=2.0)
    parser.add_argument("--core-radius", type=int, default=2)
    parser.add_argument(
        "--shape-source",
        choices=("kron", "sdss", "circular_kron"),
        default="kron",
        help="Pseudo-mask ellipse source: Kron radius plus moment axis ratio, pure Sdss moments, or circular Kron.",
    )
    parser.add_argument(
        "--source-filter",
        choices=("nchild0", "all", "parent", "leaf_child"),
        default="nchild0",
        help="Catalog rows used as center/embedding/eval GT. Default is deblend_nChild==0 leaf sources.",
    )
    parser.add_argument("--confidence-threshold", type=float, default=2.0)
    parser.add_argument(
        "--nms-radius",
        type=int,
        default=1,
        help="Local-max suppression radius. CELLECT uses kernel_size=3, equivalent to radius=1.",
    )
    parser.add_argument(
        "--confidence-score",
        choices=("cellect", "raw", "ordinal_prob"),
        default="cellect",
        help="Score used for center detection. cellect applies DK1 smoothing plus kernel_size=3 local-max logic from CELLECT.",
    )
    parser.add_argument(
        "--center-refinement",
        choices=("integer", "softargmax"),
        default="integer",
        help="Post-process detected peaks. integer preserves old pixel-grid centers; softargmax returns sub-pixel x/y centers.",
    )
    parser.add_argument(
        "--center-refinement-radius",
        type=int,
        default=1,
        help="Radius of the score-map window used by --center-refinement softargmax.",
    )
    parser.add_argument(
        "--eval-sources-csv",
        default=None,
        help="Path for per-source eval detections with pixel and RA/Dec columns. Default: <out-dir>/eval_sources.csv.",
    )
    parser.add_argument(
        "--no-eval-sources-csv",
        action="store_true",
        help="Disable writing the per-source CSV in --mode eval.",
    )
    parser.add_argument("--center-tolerance-arcsec", type=float, default=0.5)
    parser.add_argument("--pixel-scale-arcsec", type=float, default=0.168)
    parser.add_argument(
        "--match-radius",
        type=float,
        default=None,
        help="Center matching radius in pixels. Defaults to center_tolerance_arcsec / pixel_scale_arcsec.",
    )
    # Shut down the time consuming center loss by default since confidence map supervision is usually sufficient for good center detection, and the center loss can be a significant bottleneck when training with many small sources.
    parser.add_argument("--center-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--segmentation-class-weights",
        type=float,
        nargs="+",
        default=(1.0, 32.0),
        help="Binary PU segmentation class weights: background foreground. "
        "If three values are supplied for an old command, the third core-class weight is ignored.",
    )
    parser.add_argument(
        "--seg-loss-weight",
        type=float,
        default=None,
        help="Outer weight for segmentation loss. Defaults to 1.0 normally and 0.0 with --detection-only.",
    )
    parser.add_argument(
        "--seg-loss-stride",
        type=int,
        default=1,
        help="Compute segmentation loss on an avg/max-pooled grid with this stride. "
        "Use 2 or 4 when segmentation is only a light regularizer for detector training.",
    )
    parser.add_argument("--confidence-pos-weight", type=float, default=32.0)
    parser.add_argument("--shape-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--shape-angle-weight",
        type=float,
        default=4.0,
        help="Weight for angular shape loss 1-cos(delta theta) relative to the two axis-length MSE channels.",
    )
    parser.add_argument("--triplet-outer-weight", type=float, default=10.0)
    parser.add_argument("--enable-triplet", action="store_true")
    parser.add_argument(
        "--triplet-max-sources-per-group",
        type=int,
        default=256,
        help="Maximum unique source IDs per image/group used by hard triplet mining. "
        "Set <=0 to use all sources. Dense losses and EX/EN losses are unchanged."
        "IMPORTANT: Sampled based on tile/group even when --triplet-negative-scope is batch.",
    )
    parser.add_argument(
        "--triplet-negative-scope",
        choices=("tile", "batch"),
        default="batch",
        help="Hard-triplet negative mining scope. batch uses all selected sources in the local batch as negatives; "
        "tile keeps the previous behavior and only mines negatives inside each cutout.",
    )
    parser.add_argument(
        "--disable-ex-loss",
        action="store_true",
        help="Disable EX cross-band classification loss. By default EX is trained for per_band multi-band runs.",
    )
    parser.add_argument(
        "--detection-only",
        action="store_true",
        help="Train dense detection only: disable segmentation loss, shape loss, EX loss/linking, while leaving confidence and optional EN active.",
    )
    parser.add_argument(
        "--ex-core-band",
        default="HSC-I",
        help="Core band used only when --ex-band-pairs core is set. Short names like I are accepted.",
    )
    parser.add_argument(
        "--ex-band-pairs",
        nargs="*",
        default=None,
        help="Directed EX training pairs such as HSC-I:HSC-G HSC-I:HSC-R. "
        "Default uses adjacent bidirectional wavelength pairs in --bands order; use 'all' for all directed pairs "
        "or 'core' for the legacy --ex-core-band -> all-other-bands pattern.",
    )
    parser.add_argument(
        "--enable-en-loss",
        action="store_true",
        help="Train EN same-band duplicate/none-of-above classification loss. Useful for single-band duplicate suppression too.",
    )
    parser.add_argument(
        "--use-en-postprocess",
        action="store_true",
        help="Apply CELLECT-style EN same-band deduplication after detect_centers during evaluation/inference.",
    )
    parser.add_argument("--en-postprocess-threshold", type=float, default=0.6)
    parser.add_argument(
        "--use-ex-link-postprocess",
        action="store_true",
        help="Apply CELLECT-style EX cross-band linking after per-band center detection. "
        "When enabled, detection metrics are object-level instead of band-level.",
    )
    parser.add_argument("--ex-link-threshold", type=float, default=0.5)
    parser.add_argument(
        "--detect-every",
        type=int,
        default=5,
        help="During training, run full validation detection every N epochs using the validation forward pass. "
        "Set <=0 to disable train-time detection.",
    )
    parser.add_argument(
        "--train-detect-ex-link",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include EX linking and link_metrics in train-time periodic detection when the model supports EX.",
    )
    parser.add_argument(
        "--ignore-mask-during-detection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ignore predicted centers that fall inside target ignore_mask when computing detection metrics.",
    )
    parser.add_argument(
        "--enable-pu-self-training",
        action="store_true",
        help="Every --pu-self-train-every epochs, run best.pt on the train set and add high-confidence unmatched detections as specially marked pseudo labels.",
    )
    parser.add_argument("--pu-self-train-every", type=int, default=5)
    parser.add_argument("--pu-pseudo-score-percentile-start", type=float, default=99.0)
    parser.add_argument("--pu-pseudo-score-percentile-end", type=float, default=60.0)
    parser.add_argument("--pu-pseudo-clean-iou-threshold", type=float, default=0.33)
    parser.add_argument("--pu-pseudo-axis-ratio-min", type=float, default=0.1)
    parser.add_argument("--pu-pseudo-conf-weight", type=float, default=0.35)
    parser.add_argument("--pu-pseudo-seg-weight", type=float, default=0.25)
    parser.add_argument("--pu-pseudo-shape-weight", type=float, default=0.15)
    parser.add_argument("--pu-pseudo-max-per-record-band", type=int, default=512)
    parser.add_argument("--matcher-candidate-count", type=int, default=5)
    parser.add_argument(
        "--matcher-max-anchors-per-band",
        type=int,
        default=128,
        help="Maximum source anchors per batch item and band for EX/EN classification loss. "
        "Set <=0 to use all anchors.",
    )
    parser.add_argument("--matcher-outer-weight", type=float, default=10.0)
    parser.add_argument(
        "--image-cache-dir",
        default=None,
        help="Optional directory for zscale-preprocessed CHW tensors. Multi-patch caches keep <tract>/<patch>/cutouts.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Enable DistributedDataParallel. Also auto-enables when launched with torchrun WORLD_SIZE>1.",
    )
    parser.add_argument(
        "--dist-backend",
        choices=("auto", "nccl", "gloo"),
        default="auto",
        help="Distributed backend. auto uses nccl for CUDA and gloo for CPU.",
    )
    parser.add_argument(
        "--ddp-timeout-minutes",
        type=float,
        default=60.0,
        help="Process-group timeout for long rank0-only work such as full validation detection or PU pseudo-label generation.",
    )
    parser.add_argument(
        "--patch-val",
        action="store_true",
        help="Split training and validation by random patch instead of random cutout. This is a coarser split that may better reflect generalization to new sky areas, but results in higher variance.",
    )
    parser.add_argument("--dist-url", default="env://", help="Distributed init method, normally env:// for torchrun.")
    parser.add_argument("--local-rank", type=int, default=0, help="Local rank fallback when LOCAL_RANK is not set.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.seg_loss_weight is None:
        args.seg_loss_weight = 0.0 if args.detection_only else 1.0
    if args.single_band_detector:
        if args.use_ex_link_postprocess or args.train_detect_ex_link or not args.disable_ex_loss:
            print("single-band detector mode: forcing model_variant=per_band and disabling EX loss/linking")
        args.model_variant = "per_band"
        args.disable_ex_loss = True
        args.use_ex_link_postprocess = False
        args.train_detect_ex_link = False
    if args.detection_only:
        args.shape_loss_weight = 0.0
        args.disable_ex_loss = True
        args.use_ex_link_postprocess = False
        args.train_detect_ex_link = False
    if len(args.segmentation_class_weights) < 2:
        raise ValueError("--segmentation-class-weights requires at least background and foreground weights")
    if int(args.seg_classes) < 2:
        raise ValueError("--seg-classes must be >= 2")
    distributed, rank, world_size, local_rank, device, backend = _setup_distributed(args)
    is_main = _is_main(rank)

    try:
        seed = int(args.seed) + int(rank)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        root = _expand_path(args.root)
        reference_dir = _expand_path(args.reference_dir) if args.reference_dir else None
        cutout_dir = _expand_path(args.cutout_dir) if args.cutout_dir else None
        band_reference_root = _expand_path(args.band_reference_root) if args.band_reference_root else None
        targets_dir = _expand_path(args.targets_dir) if args.targets_dir else (root / "targets")
        if not targets_dir.exists():
            targets_dir = None
        image_cache_dir = _expand_path(args.image_cache_dir) if args.image_cache_dir else None
        out_dir = _expand_path(args.out_dir)
        if is_main:
            out_dir.mkdir(parents=True, exist_ok=True)
        _sync_distributed()

        if args.mode == "eval" and distributed and not is_main:
            _sync_distributed()
            return

        eval_patch_specs = _parse_patch_specs(args.eval_patches, args.eval_patches_file)
        train_patch_specs = _parse_patch_specs(args.train_patches, args.train_patches_file)
        val_patch_specs = _parse_patch_specs(args.val_patches, args.val_patches_file)
        explicit_train_val_patches = args.mode == "train" and (bool(train_patch_specs) or bool(val_patch_specs))
        records = discover_cutout_records(
            root,
            reference_dir=reference_dir,
            cutout_dir=cutout_dir,
            band_reference_root=band_reference_root,
            bands=args.bands,
            max_records=None if ((args.mode == "eval" and eval_patch_specs) or explicit_train_val_patches) else args.max_records,
        )
        if args.mode == "eval" and eval_patch_specs:
            before_count = len(records)
            records = _filter_records_by_patches(records, eval_patch_specs, root)
            if args.max_records is not None:
                records = records[: int(args.max_records)]
            if not records:
                raise RuntimeError(f"No eval records matched requested patches: {sorted(eval_patch_specs)}")
            if is_main:
                selected_patches = sorted({_record_patch_label(rec, root) for rec in records})
                print(
                    f"Eval patch filter selected {len(records)} / {before_count} records "
                    f"from patches: {selected_patches}"
                )
        if explicit_train_val_patches:
            all_records = list(records)
            if train_patch_specs:
                train_records = _filter_records_by_patches(all_records, train_patch_specs, root)
            elif val_patch_specs:
                val_records_tmp = _filter_records_by_patches(all_records, val_patch_specs, root)
                val_names = {rec.name for rec in val_records_tmp}
                train_records = [rec for rec in all_records if rec.name not in val_names]
            else:
                train_records = list(all_records)

            if val_patch_specs:
                val_records = _filter_records_by_patches(all_records, val_patch_specs, root)
            else:
                train_records, val_records = split_records(
                    train_records,
                    args.val_fraction,
                    args.seed,
                    fixed_val_names=args.fixed_val_names,
                    patch_val=args.patch_val,
                )

            if args.max_records is not None:
                train_records = train_records[: int(args.max_records)]
            if not train_records:
                raise RuntimeError(f"No train records matched requested patches: {sorted(train_patch_specs)}")
            if not val_records:
                raise RuntimeError(f"No validation records matched requested patches: {sorted(val_patch_specs)}")
            train_names = {rec.name for rec in train_records}
            val_names = {rec.name for rec in val_records}
            overlap = sorted(train_names & val_names)
            if overlap:
                raise RuntimeError(
                    "Train and validation patch filters overlap at the cutout level; "
                    f"first overlaps: {overlap[:5]}"
                )
            if is_main:
                train_selected = sorted({_record_patch_label(rec, root) for rec in train_records})
                val_selected = sorted({_record_patch_label(rec, root) for rec in val_records})
                print(
                    f"Train patch filter selected {len(train_records)} records from patches: {train_selected}"
                )
                print(
                    f"Validation patch filter selected {len(val_records)} records from patches: {val_selected}"
                )
        else:
            train_records, val_records = split_records(
                records,
                args.val_fraction,
                args.seed,
                fixed_val_names=args.fixed_val_names,
                patch_val=args.patch_val,
            )
        available_record_names = set()
        for rec in records:
            available_record_names.update(_record_name_aliases(rec))
        missing_fixed_val = sorted(set(args.fixed_val_names) - available_record_names)
        if missing_fixed_val and is_main:
            print(f"WARNING: fixed validation tile(s) not found and cannot be forced into val: {missing_fixed_val}")
        if args.mode == "eval":
            val_records = records

        center_radius_px = (
            float(args.match_radius)
            if args.match_radius is not None
            else float(args.center_tolerance_arcsec) / max(float(args.pixel_scale_arcsec), 1e-12)
        )
        weights = LossWeights(
            segmentation_binary=tuple(float(v) for v in args.segmentation_class_weights[:2]),
            segmentation_outer_weight=float(args.seg_loss_weight),
            segmentation_loss_stride=max(1, int(args.seg_loss_stride)),
            confidence_pos_weight=float(args.confidence_pos_weight),
            shape_outer_weight=float(args.shape_loss_weight),
            center_position=float(args.center_loss_weight),
            shape_angle_weight=float(args.shape_angle_weight),
            triplet_outer_weight=float(args.triplet_outer_weight),
            matcher_outer_weight=float(args.matcher_outer_weight),
        )
        common_ds = dict(
            fits_hdu=args.fits_hdu,
            confidence_levels=5,
            ellipse_sigma=args.ellipse_sigma,
            core_radius=args.core_radius,
            shape_source=args.shape_source,
            source_filter=args.source_filter,
            targets_dir=targets_dir,
            image_cache_dir=image_cache_dir,
        )
        pseudo_label_path = out_dir / "pseudo_labels" / "latest.json" if args.enable_pu_self_training else None
        train_ds = AstroCutoutDataset(
            train_records,
            augment=True,
            pseudo_label_path=pseudo_label_path,
            pseudo_confidence_weight=args.pu_pseudo_conf_weight,
            pseudo_seg_weight=args.pu_pseudo_seg_weight,
            pseudo_shape_weight=args.pu_pseudo_shape_weight,
            **common_ds,
        )
        val_ds = AstroCutoutDataset(
            val_records,
            augment=False,
            load_eval_ignore_sources=args.mode == "eval" or int(args.detect_every) > 0,
            **common_ds,
        )
        pseudo_detect_loader = None
        if args.enable_pu_self_training and is_main and args.mode == "train":
            pseudo_detect_ds = AstroCutoutDataset(train_records, augment=False, **common_ds)
            pseudo_kwargs = {
                "batch_size": args.batch_size,
                "shuffle": False,
                "num_workers": args.num_workers,
                "collate_fn": collate_cutouts,
                "pin_memory": bool(args.pin_memory),
            }
            if int(args.num_workers) > 0:
                pseudo_kwargs["persistent_workers"] = bool(args.persistent_workers)
                pseudo_kwargs["prefetch_factor"] = max(1, int(args.prefetch_factor))
            pseudo_detect_loader = DataLoader(pseudo_detect_ds, **pseudo_kwargs)
        train_sampler = (
            DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
            if distributed and args.mode == "train"
            else None
        )
        val_loader_ds = (
            Subset(val_ds, list(range(rank, len(val_ds), world_size)))
            if distributed and args.mode == "train"
            else val_ds
        )
        train_loader = DataLoader(train_ds, **_loader_kwargs(args, shuffle=train_sampler is None, sampler=train_sampler))
        val_loader = DataLoader(val_loader_ds, **_loader_kwargs(args, shuffle=False))

        if args.model_variant == "auto":
            model_variant = "fused_encoder" if len(args.bands) > 1 or args.enable_en_loss else "fused"
        else:
            model_variant = args.model_variant
        matcher_variant = model_variant in ("per_band", "fused_encoder")
        ex_enabled = matcher_variant and len(args.bands) > 1 and not args.disable_ex_loss
        en_enabled = matcher_variant and bool(args.enable_en_loss)
        en_postprocess_enabled = matcher_variant and (bool(args.use_en_postprocess) or en_enabled)
        ex_link_postprocess_enabled = matcher_variant and len(args.bands) > 1 and bool(args.use_ex_link_postprocess)
        train_detect_ex_link_enabled = ex_enabled and bool(args.train_detect_ex_link)
        ex_band_pairs = (
            parse_matcher_ex_band_pairs(args.bands, core_band=args.ex_core_band, pair_specs=args.ex_band_pairs)
            if ex_enabled
            else tuple()
        )

        if model_variant == "per_band":
            model = MultiBandAstroCELLECT2D(
                num_bands=len(args.bands),
                seg_classes=args.seg_classes,
                confidence_levels=5,
                embedding_dim=args.embedding_dim,
                base_channels=args.base_channels,
                shape_channels=3,
                candidate_count=args.matcher_candidate_count,
                shape_feature_dim=6,
            ).to(device)
        elif model_variant == "fused_encoder":
            model = FusedEncoderMultiBandAstroCELLECT2D(
                num_bands=len(args.bands),
                seg_classes=args.seg_classes,
                confidence_levels=5,
                embedding_dim=args.embedding_dim,
                base_channels=args.base_channels,
                shape_channels=3,
                candidate_count=args.matcher_candidate_count,
                shape_feature_dim=6,
            ).to(device)
        else:
            model = AstroUNet2D(
                in_channels=len(args.bands),
                seg_classes=args.seg_classes,
                confidence_levels=5,
                embedding_dim=args.embedding_dim,
                base_channels=args.base_channels,
                shape_channels=3,
            ).to(device)

        if args.checkpoint:
            ckpt = torch.load(_expand_path(args.checkpoint), map_location=device)
            state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            model.load_state_dict(state)
            if hasattr(model, "EX") and hasattr(model, "EN"):
                if isinstance(ckpt, dict) and ckpt.get("EX") is not None:
                    model.EX.load_state_dict(ckpt["EX"])
                if isinstance(ckpt, dict) and ckpt.get("EN") is not None:
                    model.EN.load_state_dict(ckpt["EN"])

        if is_main and distributed:
            print(
                f"DDP enabled: backend={backend}, world_size={world_size}, "
                f"rank={rank}, local_rank={local_rank}, device={device}"
            )

        if args.mode == "eval":
            dense, det = validate_epoch(
                model,
                val_loader,
                device=device,
                weights=weights,
                triplet_loss_fn=HardTripletLoss(weights.triplet_margin),
                triplet_enabled=args.enable_triplet,
                ex_enabled=ex_enabled,
                en_enabled=en_enabled,
                matcher_candidate_count=args.matcher_candidate_count,
                matcher_max_anchors_per_band=args.matcher_max_anchors_per_band,
                triplet_max_sources_per_group=args.triplet_max_sources_per_group,
                triplet_negative_scope=args.triplet_negative_scope,
                ex_band_pairs=ex_band_pairs,
                center_radius_px=center_radius_px,
                compute_detection=True,
                threshold=args.confidence_threshold,
                nms_radius=args.nms_radius,
                confidence_score=args.confidence_score,
                center_refinement=args.center_refinement,
                center_refinement_radius=args.center_refinement_radius,
                use_en_postprocess=en_postprocess_enabled,
                en_threshold=args.en_postprocess_threshold,
                use_ex_link_postprocess=ex_link_postprocess_enabled,
                ex_link_threshold=args.ex_link_threshold,
                band_names=args.bands,
                ignore_mask_during_detection=bool(args.ignore_mask_during_detection),
                distributed=False,
                show_progress=is_main,
            )
            if is_main:
                if not args.no_eval_sources_csv:
                    eval_csv = _expand_path(args.eval_sources_csv) if args.eval_sources_csv else out_dir / "eval_sources.csv"
                    row_count = _write_eval_sources_csv(
                        model,
                        val_loader,
                        eval_csv,
                        device=device,
                        fits_hdu=args.fits_hdu,
                        threshold=args.confidence_threshold,
                        nms_radius=args.nms_radius,
                        confidence_score=args.confidence_score,
                        center_refinement=args.center_refinement,
                        center_refinement_radius=args.center_refinement_radius,
                        match_radius=center_radius_px,
                        use_en_postprocess=en_postprocess_enabled,
                        en_candidate_count=args.matcher_candidate_count,
                        en_threshold=args.en_postprocess_threshold,
                        use_ex_link_postprocess=ex_link_postprocess_enabled,
                        ex_link_threshold=args.ex_link_threshold,
                        ex_band_pairs=ex_band_pairs,
                        band_names=args.bands,
                        show_progress=True,
                    )
                    det["sources_csv"] = str(eval_csv)
                    det["sources_csv_rows"] = row_count
                print(json.dumps({"dense": dense, "detection": det}, indent=2))
            _sync_distributed()
            return

        if distributed:
            model = _wrap_ddp(model, args, device, local_rank)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
        triplet_loss_fn = HardTripletLoss(weights.triplet_margin)

        metadata = {
            "args": vars(args),
            "loss_weights": asdict(weights),
            "num_records": len(records),
            "num_train": len(train_records),
            "num_val": len(val_records),
            "num_val_local": len(val_loader_ds),
            "fixed_val_names": list(args.fixed_val_names),
            "val_record_names": [rec.name for rec in val_records],
            "train_patch_specs": sorted(train_patch_specs),
            "val_patch_specs": sorted(val_patch_specs),
            "center_radius_px": center_radius_px,
            "center_refinement": str(args.center_refinement),
            "center_refinement_radius": int(args.center_refinement_radius),
            "targets_dir": str(targets_dir) if targets_dir is not None else None,
            "image_cache_dir": str(image_cache_dir) if image_cache_dir is not None else None,
            "band_reference_root": str(band_reference_root) if band_reference_root is not None else None,
            "model_variant": model_variant,
            "single_band_detector": bool(args.single_band_detector),
            "seg_classes": int(args.seg_classes),
            "distributed": distributed,
            "distributed_validation": bool(distributed and args.mode == "train"),
            "world_size": world_size,
            "dist_backend": backend,
            "ex_enabled": ex_enabled,
            "ex_band_pairs": [[str(args.bands[src]), str(args.bands[dst])] for src, dst in ex_band_pairs],
            "matcher_max_anchors_per_band": int(args.matcher_max_anchors_per_band),
            "triplet_negative_scope": str(args.triplet_negative_scope),
            "en_enabled": en_enabled,
            "en_postprocess_enabled": en_postprocess_enabled,
            "ex_link_postprocess_enabled": ex_link_postprocess_enabled,
            "train_detect_ex_link_enabled": train_detect_ex_link_enabled,
            "detect_every": int(args.detect_every),
            "ex_link_threshold": float(args.ex_link_threshold),
            "pu_self_training_enabled": bool(args.enable_pu_self_training),
            "pu_self_train_every": int(args.pu_self_train_every),
            "pu_pseudo_label_path": str(pseudo_label_path) if pseudo_label_path is not None else None,
        }
        if is_main:
            (out_dir / "run_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        _sync_distributed()

        best_val = float("inf")
        best_state_cpu: dict[str, torch.Tensor] | None = None
        for epoch in range(args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_metrics = run_epoch(
                model,
                train_loader,
                optimizer=optimizer,
                device=device,
                weights=weights,
                triplet_loss_fn=triplet_loss_fn,
                triplet_enabled=args.enable_triplet,
                ex_enabled=ex_enabled,
                en_enabled=en_enabled,
                matcher_candidate_count=args.matcher_candidate_count,
                matcher_max_anchors_per_band=args.matcher_max_anchors_per_band,
                triplet_max_sources_per_group=args.triplet_max_sources_per_group,
                triplet_negative_scope=args.triplet_negative_scope,
                ex_band_pairs=ex_band_pairs,
                center_radius_px=center_radius_px,
                distributed=distributed,
                show_progress=is_main,
            )

            eval_model = unwrap_model(model)
            run_detect = int(args.detect_every) > 0 and ((epoch + 1) % int(args.detect_every) == 0)
            val_metrics, det_metrics = validate_epoch(
                eval_model,
                val_loader,
                device=device,
                weights=weights,
                triplet_loss_fn=triplet_loss_fn,
                triplet_enabled=args.enable_triplet,
                ex_enabled=ex_enabled,
                en_enabled=en_enabled,
                matcher_candidate_count=args.matcher_candidate_count,
                matcher_max_anchors_per_band=args.matcher_max_anchors_per_band,
                triplet_max_sources_per_group=args.triplet_max_sources_per_group,
                triplet_negative_scope=args.triplet_negative_scope,
                ex_band_pairs=ex_band_pairs,
                center_radius_px=center_radius_px,
                compute_detection=run_detect,
                threshold=args.confidence_threshold,
                nms_radius=args.nms_radius,
                confidence_score=args.confidence_score,
                center_refinement=args.center_refinement,
                center_refinement_radius=args.center_refinement_radius,
                use_en_postprocess=en_postprocess_enabled,
                en_threshold=args.en_postprocess_threshold,
                use_ex_link_postprocess=train_detect_ex_link_enabled,
                ex_link_threshold=args.ex_link_threshold,
                band_names=args.bands,
                ignore_mask_during_detection=bool(args.ignore_mask_during_detection),
                distributed=distributed,
                show_progress=is_main,
            )
            if not run_detect:
                det_metrics = {"skipped": True, "detect_every": int(args.detect_every)}
            val_total = float(val_metrics["total"])

            val_total_tensor = torch.tensor([val_total], device=device, dtype=torch.float64)
            if distributed:
                dist.broadcast(val_total_tensor, src=0)
            scheduler.step(float(val_total_tensor.item()))

            if is_main:
                log_line = {
                    "epoch": epoch,
                    "train": train_metrics,
                    "val": val_metrics,
                    "detection": det_metrics,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                ckpt = _checkpoint_payload(
                    model,
                    model_variant=model_variant,
                    epoch=epoch,
                    args=args,
                    weights=weights,
                    center_radius_px=center_radius_px,
                    val_metrics=val_metrics,
                    det_metrics=det_metrics,
                )
                torch.save(ckpt, out_dir / "last.pt")
                best_updated = False
                if val_metrics["total"] < best_val:
                    best_val = val_metrics["total"]
                    best_state_cpu = _state_dict_cpu_copy(eval_model)
                    torch.save(ckpt, out_dir / "best.pt")
                    best_updated = True
                log_line["best_updated"] = best_updated
                if args.enable_pu_self_training and int(args.pu_self_train_every) > 0 and ((epoch + 1) % int(args.pu_self_train_every) == 0):
                    if best_state_cpu is None:
                        best_state_cpu = _state_dict_cpu_copy(eval_model)
                        torch.save(ckpt, out_dir / "best.pt")
                    current_state = _state_dict_cpu_copy(eval_model)
                    eval_model.load_state_dict(best_state_cpu)
                    pseudo_path = out_dir / "pseudo_labels" / f"epoch_{epoch + 1:04d}.json"
                    pseudo_summary = generate_pu_pseudo_labels(
                        eval_model,
                        pseudo_detect_loader,
                        device=device,
                        epoch=epoch,
                        total_epochs=args.epochs,
                        output_path=pseudo_path,
                        band_names=args.bands,
                        score_percentile_start=args.pu_pseudo_score_percentile_start,
                        score_percentile_end=args.pu_pseudo_score_percentile_end,
                        min_center_distance_px=center_radius_px,
                        clean_iou_threshold=args.pu_pseudo_clean_iou_threshold,
                        axis_ratio_min=args.pu_pseudo_axis_ratio_min,
                        nms_radius=args.nms_radius,
                        confidence_score=args.confidence_score,
                        ellipse_sigma=args.ellipse_sigma,
                        max_pseudo_per_record_band=args.pu_pseudo_max_per_record_band,
                        show_progress=True,
                    )
                    eval_model.load_state_dict(current_state)
                    latest_path = out_dir / "pseudo_labels" / "latest.json"
                    latest_path.write_text(pseudo_path.read_text(encoding="utf-8"), encoding="utf-8")
                    (out_dir / "pseudo_labels" / "latest_summary.json").write_text(
                        json.dumps(pseudo_summary, indent=2),
                        encoding="utf-8",
                    )
                    log_line["pu_self_training"] = pseudo_summary
                print(json.dumps(log_line, indent=2))
            _sync_distributed()
            if (
                args.enable_pu_self_training
                and pseudo_label_path is not None
                and args.mode == "train"
                and int(args.pu_self_train_every) > 0
                and ((epoch + 1) % int(args.pu_self_train_every) == 0)
            ):
                train_ds.reload_pseudo_labels(pseudo_label_path)
                train_loader = DataLoader(train_ds, **_loader_kwargs(args, shuffle=train_sampler is None, sampler=train_sampler))
    finally:
        _cleanup_distributed()


if __name__ == "__main__":
    main()
