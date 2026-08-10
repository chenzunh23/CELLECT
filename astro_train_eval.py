"""
Train and evaluate the 2D CELLECT-style astronomy model on LSST/HSC FITS cutouts.

The dense losses intentionally keep CELLECT's original constants.  Dataset and
training/evaluation logic lives in astro_train_data.py and astro_train_ops.py;
shared orchestration helpers live in utils/ so this file stays focused on CLI,
model construction, and distributed training orchestration.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import wandb
from dataclasses import asdict, replace
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from astro_cellect2d import AstroUNet2D, FusedEncoderMultiBandAstroCELLECT2D, MultiBandAstroCELLECT2D
from astro_match_eval import parse_ex_band_pairs as parse_matcher_ex_band_pairs
from sam_backbone import build_sam_cellect2d
from astro_train_data import (
    AstroCutoutDataset,
    _expand_path,
    _record_name_aliases,
    collate_cutouts,
    discover_cutout_records,
    load_meas_catalog,
    make_targets,
    split_records,
)
from astro_train_zarr_data import (
    ZarrChunkBatchIterableDataset,
    ZarrChunkLocalBatchSampler,
    ZarrCutoutDataset,
    discover_zarr_image_records,
    discover_zarr_records,
    zarr_passthrough_batch,
)
from astro_train_ops import (
    HardTripletLoss,
    LossWeights,
    active_mask_loss_keys,
    detect_centers,
    detect_centers_with_en,
    detect_centers_with_ex_link,
    evaluate_detection,
    generate_pu_pseudo_labels,
    parse_ex_band_pairs,
    model_forward_with_batch_context,
    run_epoch,
    unwrap_model,
    validate_epoch,
)
from utils.train_eval_utils import (
    WarmupStepIterationLR,
    as_numpy_centers as _as_numpy_centers,
    as_numpy_ids as _as_numpy_ids,
    as_numpy_mask as _as_numpy_mask,
    band_name as _band_name,
    checkpoint_payload as _checkpoint_payload,
    cleanup_distributed as _cleanup_distributed,
    filter_records_by_patches as _filter_records_by_patches,
    flat_per_band_outputs as _flat_per_band_outputs,
    format_float as _format_float,
    greedy_point_mapping as _greedy_point_mapping,
    is_main as _is_main,
    loader_kwargs as _loader_kwargs,
    parse_patch_specs as _parse_patch_specs,
    point_in_mask_np as _point_in_mask_np,
    radec_from_wcs as _radec_from_wcs,
    record_patch_label as _record_patch_label,
    sam_optimizer_param_groups,
    setup_distributed as _setup_distributed,
    state_dict_cpu_copy as _state_dict_cpu_copy,
    sync_distributed as _sync_distributed,
    wcs_for_path as _wcs_for_path,
    wrap_ddp as _wrap_ddp,
)


def _dataset_source_filter(values: Sequence[str]) -> Optional[set[str]]:
    sources = {str(value).strip().lower() for value in values if str(value).strip()}
    if not sources or "all" in sources or "*" in sources:
        return None
    return sources


def _filter_records_by_dataset_sources(records, values: Sequence[str]):
    sources = _dataset_source_filter(values)
    if sources is None:
        return list(records)
    return [
        rec
        for rec in records
        if str(getattr(rec, "dataset_source", "coadd") or "coadd").lower() in sources
    ]


def _runtime_unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap DDP helper layers while preserving torch.compile runtime wrappers."""

    current = model
    while True:
        if hasattr(current, "module"):
            current = current.module  # type: ignore[assignment]
            continue
        if hasattr(current, "_ddp_wrapped_model"):
            current = current._ddp_wrapped_model  # type: ignore[attr-defined,assignment]
            continue
        return current


def _compile_sam_runtime_model(model: nn.Module, args: argparse.Namespace, *, is_main: bool) -> nn.Module:
    if not bool(getattr(args, "sam_compile", False)):
        return model
    if str(getattr(args, "model_variant", "")) != "sam_per_band":
        if is_main:
            print("WARNING: --sam-compile is only used with --model-variant sam_per_band; ignoring.")
        return model
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        raise RuntimeError("--sam-compile requires a PyTorch build with torch.compile")
    kwargs: dict[str, object] = {
        "backend": str(getattr(args, "sam_compile_backend", "inductor")),
        "fullgraph": bool(getattr(args, "sam_compile_fullgraph", False)),
    }
    if bool(getattr(args, "sam_dynamic_image_size", False)):
        kwargs["dynamic"] = True
    mode = str(getattr(args, "sam_compile_mode", "default"))
    if mode and mode != "default":
        kwargs["mode"] = mode
    if is_main:
        print(
            "Compiling SAM runtime model with torch.compile "
            f"(backend={kwargs['backend']}, mode={mode}, fullgraph={kwargs['fullgraph']}). "
            "Checkpoints will still store the uncompiled state_dict."
        )
    return compile_fn(model, **kwargs)


def _compile_sam_mask_decoder_modules(model: nn.Module, args: argparse.Namespace, *, is_main: bool) -> nn.Module:
    if not bool(getattr(args, "sam_compile_mask_decoder", False)):
        return model
    if str(getattr(args, "model_variant", "")) != "sam_per_band":
        if is_main:
            print("WARNING: --sam-compile-mask-decoder is only used with --model-variant sam_per_band; ignoring.")
        return model
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        raise RuntimeError("--sam-compile-mask-decoder requires a PyTorch build with torch.compile")
    kwargs: dict[str, object] = {
        "backend": str(getattr(args, "sam_compile_backend", "inductor")),
        "fullgraph": bool(getattr(args, "sam_compile_fullgraph", False)),
    }
    if bool(getattr(args, "sam_dynamic_image_size", False)):
        kwargs["dynamic"] = True
    mode = str(getattr(args, "sam_compile_mode", "default"))
    if mode and mode != "default":
        kwargs["mode"] = mode
    base_model = _runtime_unwrap_model(model)
    compiled: list[str] = []
    for name in ("prompt_encoder", "mask_decoder"):
        module = getattr(base_model, name, None)
        if isinstance(module, nn.Module):
            setattr(base_model, name, compile_fn(module, **kwargs))
            compiled.append(name)
    if is_main:
        print(
            "Compiling SAM mask-loss decoder modules with torch.compile "
            f"(modules={compiled}, backend={kwargs['backend']}, mode={mode}, fullgraph={kwargs['fullgraph']}).",
            flush=True,
        )
    return model


def _init_wandb_or_disable(
    args: argparse.Namespace,
    *,
    out_dir: Path,
    metadata: dict[str, object],
    is_main: bool,
):
    if not is_main or str(args.wandb_mode) == "disabled":
        return None
    try:
        run = wandb.init(
            project=str(args.wandb_project),
            entity=str(args.wandb_entity) if args.wandb_entity else None,
            name=str(args.wandb_run_name) if args.wandb_run_name else out_dir.name,
            dir=str(out_dir),
            config=metadata,
            mode=str(args.wandb_mode),
        )
    except Exception as exc:
        print(
            "WARNING: wandb initialization failed; continuing with wandb disabled. "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        return None
    wandb.define_metric("iteration")
    wandb.define_metric("epoch")
    wandb.define_metric("train/iteration/*", step_metric="iteration")
    wandb.define_metric("lr/iteration/*", step_metric="iteration")
    wandb.define_metric("lr/epoch/*", step_metric="epoch")
    wandb.define_metric("val/epoch/*", step_metric="epoch")
    wandb.define_metric("train/epoch/*", step_metric="epoch")
    return run


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
    use_ordinal_expectation: bool,
    debug_ordinal_expectation: bool,
    center_refinement: str,
    center_refinement_radius: int,
    match_radius: float,
    pixel_scale_arcsec: float,
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
        match_info: dict[str, object] | None = None,
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
        match_info = match_info or {}
        match_x = match_info.get("match_x_local", "")
        match_y = match_info.get("match_y_local", "")
        match_ra_deg = float("nan")
        match_dec_deg = float("nan")
        if match_x != "" and match_y != "":
            match_ra_deg, match_dec_deg = _radec_from_wcs(wcs, float(match_x), float(match_y))
        strict_center_only = _mask_at("strict_center_only_mask", "band_strict_center_only_mask")
        strict_ignored = _mask_at("strict_ignore_mask", "band_strict_ignore_mask")
        ignored = _mask_at("ignore_mask", "band_ignore_mask")
        ordinary_ignored = ignored
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
                "strict_center_only": int(strict_center_only),
                "strict_ignore": int(strict_ignored),
                "eval_excluded_by_mask": int(ignored),
                "match_status": match_info.get("match_status", ""),
                "matched_catalog": match_info.get("matched_catalog", ""),
                "matched_source_id": match_info.get("matched_source_id", ""),
                "matched_source_index": match_info.get("matched_source_index", ""),
                "match_distance_pix": match_info.get("match_distance_pix", ""),
                "match_distance_arcsec": match_info.get("match_distance_arcsec", ""),
                "matched_x_local": _format_float(float(match_x), 6) if match_x != "" else "",
                "matched_y_local": _format_float(float(match_y), 6) if match_y != "" else "",
                "matched_x_parent": _format_float(float(x0) + float(match_x), 6) if match_x != "" else "",
                "matched_y_parent": _format_float(float(y0) + float(match_y), 6) if match_y != "" else "",
                "matched_ra_deg": _format_float(match_ra_deg, 10),
                "matched_dec_deg": _format_float(match_dec_deg, 10),
                "member_count": int(member_count),
                "member_bands": member_bands,
                "member_centers": json.dumps(member_centers) if member_centers else "",
                "image_path": image_path,
            }
        )

    def _batch_centers(batch: dict[str, object], item_idx: int, band_idx: int, key: str, band_key: str) -> np.ndarray:
        if band_idx >= 0 and band_key in batch:
            try:
                return _as_numpy_centers(batch[band_key][item_idx][band_idx])  # type: ignore[index]
            except Exception:
                pass
        if key in batch:
            try:
                return _as_numpy_centers(batch[key][item_idx])  # type: ignore[index]
            except Exception:
                pass
        return np.zeros((0, 2), dtype=np.float32)

    def _batch_ids(batch: dict[str, object], item_idx: int, band_idx: int) -> np.ndarray:
        if band_idx >= 0 and "band_ids" in batch:
            try:
                return _as_numpy_ids(batch["band_ids"][item_idx][band_idx])  # type: ignore[index]
            except Exception:
                pass
        if "ids" in batch:
            try:
                return _as_numpy_ids(batch["ids"][item_idx])  # type: ignore[index]
            except Exception:
                pass
        return np.zeros((0,), dtype=np.int64)

    def _batch_mask(batch: dict[str, object], item_idx: int, band_idx: int, key: str, band_key: str) -> np.ndarray | None:
        if band_idx >= 0 and band_key in batch:
            try:
                return _as_numpy_mask(batch[band_key][item_idx, band_idx])  # type: ignore[index]
            except Exception:
                pass
        if key in batch:
            try:
                return _as_numpy_mask(batch[key][item_idx])  # type: ignore[index]
            except Exception:
                pass
        return None

    def _match_infos_for_predictions(
        batch: dict[str, object],
        item_idx: int,
        band_idx: int,
        pred_xy: np.ndarray,
    ) -> list[dict[str, object]]:
        pred_xy = _as_numpy_centers(pred_xy)
        clean_xy = _batch_centers(batch, item_idx, band_idx, "centers", "band_centers")
        ordinary_xy = _batch_centers(batch, item_idx, band_idx, "ignore_centers", "band_ignore_centers")
        strict_center_xy = _batch_centers(
            batch,
            item_idx,
            band_idx,
            "strict_center_only_centers",
            "band_strict_center_only_centers",
        )
        strict_ignore_xy = _batch_centers(batch, item_idx, band_idx, "strict_ignore_centers", "band_strict_ignore_centers")
        ordinary_xy = np.unique(
            np.concatenate([ordinary_xy, strict_center_xy, strict_ignore_xy], axis=0),
            axis=0,
        ).astype(np.float32)
        clean_ids = _batch_ids(batch, item_idx, band_idx)
        clean_mask = _batch_mask(batch, item_idx, band_idx, "clean_mask", "band_clean_mask")
        background_mask = _batch_mask(batch, item_idx, band_idx, "background_mask", "band_background_mask")
        clean_map, clean_dist = _greedy_point_mapping(pred_xy, clean_xy, match_radius)
        unmatched = [idx for idx in range(len(pred_xy)) if idx not in clean_map]
        if unmatched:
            unmatched_xy = pred_xy[np.asarray(unmatched, dtype=np.int64)]
            ordinary_rel_map, ordinary_rel_dist = _greedy_point_mapping(unmatched_xy, ordinary_xy, match_radius)
            ordinary_map = {unmatched[rel_idx]: gt_idx for rel_idx, gt_idx in ordinary_rel_map.items()}
            ordinary_dist = {unmatched[rel_idx]: dist for rel_idx, dist in ordinary_rel_dist.items()}
        else:
            ordinary_map = {}
            ordinary_dist = {}

        infos: list[dict[str, object]] = []
        for pred_idx, xy in enumerate(pred_xy):
            info: dict[str, object] = {
                "match_status": "unmatched",
                "matched_catalog": "",
                "matched_source_id": "",
                "matched_source_index": "",
                "match_distance_pix": "",
                "match_distance_arcsec": "",
                "match_x_local": "",
                "match_y_local": "",
            }
            if pred_idx in clean_map:
                gt_idx = clean_map[pred_idx]
                info["match_status"] = "clean_tp"
                info["matched_catalog"] = "clean"
                info["matched_source_index"] = int(gt_idx)
                if gt_idx < len(clean_ids):
                    info["matched_source_id"] = str(clean_ids[gt_idx])
                info["match_distance_pix"] = _format_float(clean_dist[pred_idx], 6)
                info["match_distance_arcsec"] = _format_float(clean_dist[pred_idx] * float(pixel_scale_arcsec), 6)
                info["match_x_local"] = float(clean_xy[gt_idx, 0])
                info["match_y_local"] = float(clean_xy[gt_idx, 1])
            elif pred_idx in ordinary_map:
                gt_idx = ordinary_map[pred_idx]
                info["match_status"] = "ordinary_ignore_tp"
                info["matched_catalog"] = "ordinary_ignore"
                info["matched_source_index"] = int(gt_idx)
                info["match_distance_pix"] = _format_float(ordinary_dist[pred_idx], 6)
                info["match_distance_arcsec"] = _format_float(ordinary_dist[pred_idx] * float(pixel_scale_arcsec), 6)
                info["match_x_local"] = float(ordinary_xy[gt_idx, 0])
                info["match_y_local"] = float(ordinary_xy[gt_idx, 1])
            elif _point_in_mask_np(clean_mask, xy[0], xy[1]) or _point_in_mask_np(background_mask, xy[0], xy[1]):
                info["match_status"] = "fp_clean_background"
            else:
                info["match_status"] = "fp_outside_eval_region"
            infos.append(info)
        return infos

    for batch in tqdm(loader, desc="eval-csv", leave=False, disable=not show_progress):
        image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)  # type: ignore[union-attr]
        outputs = model_forward_with_batch_context(model, image, batch)
        per_band_outputs = outputs["confidence"].ndim == 5
        if per_band_outputs and use_ex_link_postprocess and hasattr(base_model, "EX"):
            pred_list, components_all = detect_centers_with_ex_link(
                base_model,
                outputs,
                threshold=threshold,
                nms_radius=nms_radius,
                confidence_score=confidence_score,
                use_ordinal_expectation=use_ordinal_expectation,
                debug_ordinal_expectation=debug_ordinal_expectation,
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
                pred_arr = np.asarray(pred_xy, dtype=np.float32).reshape(-1, 2)
                components = components_all[item_idx] if item_idx < len(components_all) else []
                linked_band_indices: list[int] = []
                for source_index in range(len(pred_arr)):
                    component = components[source_index] if source_index < len(components) else {}
                    members = component.get("members", []) if isinstance(component, dict) else []
                    linked_band_indices.append(int(members[0][0]) if members else 0)
                linked_match_infos = [
                    _match_infos_for_predictions(batch, item_idx, band_idx, pred_arr[idx : idx + 1])[0]
                    for idx, band_idx in enumerate(linked_band_indices)
                ]
                for source_index, xy in enumerate(pred_arr):
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
                        match_info=linked_match_infos[source_index],
                    )
            continue

        band_count = int(outputs["confidence"].shape[1]) if per_band_outputs else 0
        if use_en_postprocess and hasattr(base_model, "EN"):
            pred_list = detect_centers_with_en(
                base_model,
                outputs,
                threshold=threshold,
                nms_radius=nms_radius,
                confidence_score=confidence_score,
                use_ordinal_expectation=use_ordinal_expectation,
                debug_ordinal_expectation=debug_ordinal_expectation,
                match_radius=match_radius,
                candidate_count=en_candidate_count,
                en_threshold=en_threshold,
                center_refinement=center_refinement,
                center_refinement_radius=center_refinement_radius,
            )
        elif per_band_outputs:
            pred_list = detect_centers(
                _flat_per_band_outputs(outputs),
                threshold=threshold,
                nms_radius=nms_radius,
                confidence_score=confidence_score,
                use_ordinal_expectation=use_ordinal_expectation,
                debug_ordinal_expectation=debug_ordinal_expectation,
                center_refinement=center_refinement,
                center_refinement_radius=center_refinement_radius,
            )
        else:
            pred_list = detect_centers(
                outputs,
                threshold=threshold,
                nms_radius=nms_radius,
                confidence_score=confidence_score,
                use_ordinal_expectation=use_ordinal_expectation,
                debug_ordinal_expectation=debug_ordinal_expectation,
                center_refinement=center_refinement,
                center_refinement_radius=center_refinement_radius,
            )

        for list_idx, pred_xy in enumerate(pred_list):
            item_idx = list_idx // band_count if band_count else list_idx
            band_idx = list_idx % band_count if band_count else 0
            source_type = "band" if band_count else "fused"
            pred_arr = np.asarray(pred_xy, dtype=np.float32).reshape(-1, 2)
            match_infos = _match_infos_for_predictions(batch, item_idx, band_idx, pred_arr)
            for source_index, xy in enumerate(pred_arr):
                append_row(
                    batch,
                    item_idx,
                    source_type,
                    float(xy[0]),
                    float(xy[1]),
                    band_idx=band_idx,
                    source_index=source_index,
                    member_bands=_band_name(band_idx, band_names) if band_count else "",
                    match_info=match_infos[source_index],
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
        "strict_center_only",
        "strict_ignore",
        "eval_excluded_by_mask",
        "match_status",
        "matched_catalog",
        "matched_source_id",
        "matched_source_index",
        "match_distance_pix",
        "match_distance_arcsec",
        "matched_x_local",
        "matched_y_local",
        "matched_x_parent",
        "matched_y_parent",
        "matched_ra_deg",
        "matched_dec_deg",
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


def _write_linking_metrics_json(path: Path, det_metrics: dict[str, object], *, epoch: int | None = None) -> bool:
    link_metrics = det_metrics.get("link_metrics")
    if not isinstance(link_metrics, dict):
        return False
    payload: dict[str, object] = {
        "epoch": epoch,
        "link_metrics": link_metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return True



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/evaluate AstroCELLECT2D on LSST/HSC FITS cutouts.")
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument(
        "--data-format",
        choices=("legacy", "zarr"),
        default="legacy",
        help="Use legacy FITS/NPZ preprocessing or direct patch Zarr stores.",
    )
    parser.add_argument(
        "--zarr-chunk-local-batches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In Zarr mode, batch samples from the same Zarr chunk and shuffle at chunk level.",
    )
    parser.add_argument(
        "--zarr-worker-owned-chunks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In Zarr mode, shard chunks by DDP rank and DataLoader worker; each worker drains one chunk into batches.",
    )
    parser.add_argument(
        "--zarr-shuffle-within-chunk",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In Zarr chunk-local batching, shuffle sample order inside each chunk.",
    )
    parser.add_argument(
        "--zarr-drop-last",
        action="store_true",
        help="Drop incomplete chunk-local batches in Zarr mode.",
    )
    parser.add_argument(
        "--zarr-random-image-batches",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "In Zarr train mode, use single-band image-level stores for SAM detector training. "
            "Batch size then means total images, not multi-band groups."
        ),
    )
    parser.add_argument(
        "--root",
        default="~/segment-anything/lsst_pipeline/output/cutout_magnitude_experiment_grid",
        help="Legacy root containing cutouts/reference_catalogs, or direct Zarr root when --data-format zarr.",
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
    parser.add_argument("--wandb-project", default="Astro_CELLECT2D_SAM")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
        help="Weights & Biases logging mode. Default is disabled to avoid blocking non-interactive DDP runs.",
    )
    parser.add_argument(
        "--wandb-log-interval",
        type=int,
        default=50,
        help="Log training losses/LR/prompt curriculum to wandb every N optimizer iterations. Set <=0 to disable iteration logs.",
    )
    parser.add_argument("--epochs", type=int, default=100)
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
    parser.add_argument(
        "--debug-batch-start",
        type=int,
        default=-1,
        help="Print detailed per-stage timing starting at this zero-based train batch index. Disabled when <0.",
    )
    parser.add_argument(
        "--debug-batch-end",
        type=int,
        default=-1,
        help="Last zero-based train batch index for --debug-batch-start timing. Defaults to the start batch.",
    )
    parser.add_argument(
        "--debug-batch-all-ranks",
        action="store_true",
        help="Print --debug-batch-start timing from every DDP rank instead of rank 0 only.",
    )
    parser.add_argument(
        "--mask-loss-interval",
        type=int,
        default=1,
        help=(
            "Compute SAM mask loss every N training iterations. Skipped iterations avoid forward_sam_masks. "
            "The mask loss is multiplied by N on active iterations unless --no-mask-loss-interval-scale is set."
        ),
    )
    parser.add_argument(
        "--mask-loss-interval-scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scale active intermittent mask-loss steps by --mask-loss-interval to preserve expected mask gradient size.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument(
        "--sam-encoder-lr",
        type=float,
        default=2e-5,
        help="Peak learning rate for the SAM encoder when --model-variant sam_per_band.",
    )
    parser.add_argument(
        "--sam-warmup-ratio",
        type=float,
        default=0.01,
        help="Fraction of total optimizer iterations used for linear warmup in SAM fine-tuning.",
    )
    parser.add_argument(
        "--sam-lr-drop-fractions",
        type=float,
        nargs="+",
        default=(0.70, 0.90),
        help="Fractions of total optimizer iterations where SAM fine-tuning LR is multiplied by --sam-lr-drop-gamma.",
    )
    parser.add_argument(
        "--sam-lr-drop-gamma",
        type=float,
        default=0.1,
        help="Multiplicative LR drop factor for SAM fine-tuning iteration schedule.",
    )
    parser.add_argument(
        "--sam-lr-phase2-epoch",
        type=int,
        default=-1,
        help=(
            "For sam_per_band, override named group base LRs from this epoch onward. "
            "Use with --sam-encoder-lr-after and/or --sam-head-lr-after. "
            "LR=0 freezes that group by clearing gradients before optimizer.step()."
        ),
    )
    parser.add_argument(
        "--sam-encoder-lr-after",
        type=float,
        default=None,
        help="Encoder base LR after --sam-lr-phase2-epoch. Omit to keep --sam-encoder-lr.",
    )
    parser.add_argument(
        "--sam-head-lr-after",
        type=float,
        default=None,
        help=(
            "Proposal head base LR after --sam-lr-phase2-epoch. "
            "This controls the CELLECT confidence/shape decoder, not the SAM mask decoder."
        ),
    )
    parser.add_argument(
        "--sam-decoder-lr-after",
        type=float,
        default=None,
        help="SAM prompt/mask decoder base LR after --sam-lr-phase2-epoch. Omit to keep --lr.",
    )
    parser.add_argument(
        "--sam-fused-adamw",
        action="store_true",
        help=(
            "Use torch AdamW(fused=True) for SAM fine-tuning. Disabled by default because "
            "mask-loss warmup uses a dynamic parameter graph."
        ),
    )
    parser.add_argument(
        "--sam-compile",
        action="store_true",
        help=(
            "Compile --model-variant sam_per_band with torch.compile after checkpoint loading. "
            "Saved checkpoints still use the original uncompiled module state_dict."
        ),
    )
    parser.add_argument("--sam-compile-backend", default="inductor", help="torch.compile backend for --sam-compile.")
    parser.add_argument(
        "--sam-compile-mode",
        choices=("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"),
        default="default",
        help="torch.compile mode for --sam-compile.",
    )
    parser.add_argument(
        "--sam-compile-fullgraph",
        action="store_true",
        help="Pass fullgraph=True to torch.compile for --sam-compile. Default allows graph breaks.",
    )
    parser.add_argument(
        "--sam-compile-mask-decoder",
        action="store_true",
        help=(
            "Also compile SAM prompt_encoder and mask_decoder modules used by forward_sam_masks in mask loss. "
            "Uses the same backend/mode/fullgraph settings as --sam-compile."
        ),
    )
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
        choices=("auto", "fused", "per_band", "fused_encoder", "sam_per_band"),
        default="auto",
        help="fused treats bands as channels and outputs one map. per_band runs one shared single-band backbone per band. "
        "fused_encoder runs one multi-band encoder and lightweight per-band heads for EX/EN. "
        "sam_per_band uses the SAM image encoder plus CELLECT-style dense decoder. "
        "auto uses fused_encoder for multi-band data or when EN is enabled.",
    )
    parser.add_argument("--sam-model-type", choices=("vit_b", "vit_l", "vit_h"), default="vit_b")
    parser.add_argument(
        "--sam-checkpoint",
        default=None,
        help="Official SAM checkpoint for --model-variant sam_per_band. This is separate from --checkpoint.",
    )
    parser.add_argument(
        "--sam-dynamic-image-size",
        action="store_true",
        help=(
            "For sam_per_band inputs up to 512x512, pad only to the nearest patch-size multiple and "
            "dynamically size image/prompt positional encodings and SAM mask outputs. Disabled by default, "
            "which preserves fixed 512x512 SAM padding."
        ),
    )
    parser.add_argument(
        "--disable-sam-cen",
        action="store_true",
        help="Disable the CELLECT-style CEN confidence module in --model-variant sam_per_band.",
    )
    parser.add_argument(
        "--sam-decoder-film",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable zero-initialized decoder FiLM for denoised samples. Coadd/noisy remain on the identity path. "
            "When omitted, this is inferred from --checkpoint and otherwise disabled."
        ),
    )
    parser.add_argument(
        "--sam-encoder-style-prompt",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable image-routed raw/processed prompts in SAM ViT-B blocks 2, 5, and 8. "
            "The router uses image content at inference; dataset labels are only an auxiliary training target."
        ),
    )
    parser.add_argument("--style-prompt-dim", type=int, default=32)
    parser.add_argument("--style-adapter-dim", type=int, default=32)
    parser.add_argument("--style-router-temperature", type=float, default=1.0)
    parser.add_argument(
        "--style-router-loss-weight",
        type=float,
        default=0.1,
        help="Balanced denoised-vs-non-denoised auxiliary router loss; effective only with encoder style prompts.",
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
        "--dataset-sources",
        nargs="*",
        default=("coadd", "denoised"),
        help=(
            "Dataset image variants to load when --root contains multiple variant trees. "
            "Use coadd denoised noisy, or all. Default: coadd denoised."
        ),
    )
    parser.add_argument(
        "--train-dataset-sources",
        nargs="*",
        default=(),
        help="Optional train-only override for --dataset-sources.",
    )
    parser.add_argument(
        "--val-dataset-sources",
        nargs="*",
        default=(),
        help="Optional validation-only override for --dataset-sources.",
    )
    parser.add_argument(
        "--eval-dataset-sources",
        nargs="*",
        default=(),
        help="Optional eval-only override for --dataset-sources.",
    )
    parser.add_argument(
        "--train-patches",
        nargs="*",
        default=(),
        help=(
            "Restrict --mode train training records to patches or source-qualified patches, "
            "e.g. 0,0 9813/0,0 denoised:0,0 noisy:9813/6,1. "
            "Append @group_02 to keep one explicit variant group, or @random to choose one "
            "group per selector using --seed, e.g. denoised:3,4@group_02 or denoised:3,4@random."
        ),
    )
    parser.add_argument(
        "--train-patches-file",
        default=None,
        help=(
            "Optional text file with one train selector per line. Lines may use 0,0, 9813/0,0, "
            "denoised:0,0, denoised:0,0@group_02, or denoised:0,0@random; # comments are ignored."
        ),
    )
    parser.add_argument(
        "--val-patches",
        nargs="*",
        default=(),
        help=(
            "Restrict --mode train validation records to patches or source-qualified patches, "
            "e.g. 6,1, denoised:9813/6,1, or noisy:6,1@group_03."
        ),
    )
    parser.add_argument(
        "--val-patches-file",
        default=None,
        help=(
            "Optional text file with one validation selector per line. Lines may use 6,1, 9813/6,1, "
            "noisy:6,1, noisy:6,1@group_03, or noisy:6,1@random; # comments are ignored."
        ),
    )
    parser.add_argument(
        "--eval-patches",
        nargs="*",
        default=(),
        help=(
            "Restrict --mode eval to specific patches. Accepts 8,8, 9813/8,8, "
            "denoised:8,8, denoised:8,8@group_01, or denoised:8,8@random."
        ),
    )
    parser.add_argument(
        "--eval-patches-file",
        default=None,
        help=(
            "Optional text file with one eval selector per line. Lines may use 8,8, 9813/8,8, "
            "denoised:8,8, denoised:8,8@group_01, or denoised:8,8@random; # comments are ignored."
        ),
    )
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument(
        "--fixed-val-names",
        nargs="*",
        default=("sam_x18204_y20924",),
        help="Tile names that must be placed in the validation set. Default keeps the SAM comparison cutout in validation.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ellipse-sigma", type=float, default=1.0) # Important: We use the relatively large kron aperture in our latest setting, so sigma should be set to 1.0 instead of 2.0
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
    parser.add_argument(
        "--noncoadd-snr-filter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Debug fallback: for noisy/denoised datasets, classify clean GT online by raw-image ap2 annulus SNR. "
            "Default is off because data_preprocessing.sh now writes these targets offline."
        ),
    )
    parser.add_argument("--noncoadd-snr-ignore-thresh", type=float, default=2.0, help="Non-coadd GT below this SNR is moved to ignore.")
    parser.add_argument("--noncoadd-snr-center-only-thresh", type=float, default=3.0, help="Non-coadd GT in [ignore, this) SNR is center-only/low-shape-weight; >= this is normal clean GT.")
    parser.add_argument("--noncoadd-snr-ap-radius", type=float, default=6.0, help="Aperture radius in pixels for non-coadd GT visibility SNR.")
    parser.add_argument("--noncoadd-snr-annulus-r-in", type=float, default=10.0, help="Inner annulus radius in pixels for non-coadd GT visibility SNR.")
    parser.add_argument("--noncoadd-snr-annulus-r-out", type=float, default=15.0, help="Outer annulus radius in pixels for non-coadd GT visibility SNR.")
    parser.add_argument(
        "--noncoadd-snr-annulus-exclude-radius",
        type=float,
        default=6.0,
        help="Exclude pixels within this radius of any clean source center from online non-coadd SNR annuli. Use <=0 to disable.",
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
        choices=("cellect", "raw", "ordinal_prob", "ordinal_expectation"),
        default="cellect",
        help="Score used for center detection. cellect applies DK1 smoothing plus kernel_size=3 local-max logic from CELLECT.",
    )
    parser.add_argument(
        "--use-ordinal-expectation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Override center detection to use ordinal expectation with a fixed threshold of 2.0 and local-max suppression on that score map.",
    )
    parser.add_argument(
        "--debug-ordinal-expectation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add ordinal-expectation candidate maps to astro_train_ops detection debug outputs.",
    )
    parser.add_argument(
        "--debug-detection-metrics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Collect candidate-stage debug histograms and retention metrics during detection eval. Disabled by default to avoid extra peak-map work.",
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
        "--linking-metrics-json",
        default=None,
        help="Path for detection-stage adjacent-band linking metrics in --mode eval. Default: <out-dir>/linking_metrics.json.",
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
    parser.add_argument(
        "--confidence-loss-weight",
        type=float,
        default=1.0,
        help="Outer weight for dense confidence loss. Proposal freezing sets this to 0 after the configured epoch.",
    )
    parser.add_argument("--confidence-pos-weight", type=float, default=32.0)
    parser.add_argument(
        "--confidence-loss-mode",
        choices=("ordinal_legacy", "ce_hard"),
        default="ordinal_legacy",
        help="Dense confidence loss. ordinal_legacy keeps the CELLECT ordinal loss; ce_hard uses weighted 5-class CE.",
    )
    parser.add_argument(
        "--confidence-ce-weights",
        type=float,
        nargs=5,
        default=(1.0, 4.0, 8.0, 16.0, 32.0),
        metavar=("W0", "W1", "W2", "W3", "W4"),
        help="Class weights for --confidence-loss-mode ce_hard, ordered by confidence level 0..4.",
    )
    parser.add_argument(
        "--small-shape-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for an epoch-0 dense loss that suppresses high ordinal confidence "
            "for tiny predicted shapes in ignore/non-clean regions. Default disables it."
        ),
    )
    parser.add_argument("--small-shape-area-min", type=float, default=20.0)
    parser.add_argument("--small-shape-area-tau", type=float, default=5.0)
    parser.add_argument("--small-shape-ordinal-threshold", type=float, default=2.0)
    parser.add_argument(
        "--small-shape-scope",
        choices=("ignore", "non_clean", "unweighted"),
        default="ignore",
        help="Region where the small-shape ordinal suppression loss is applied.",
    )
    parser.add_argument("--shape-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--shape-loss-mode",
        choices=("source_center", "dense_pixel"),
        default="source_center",
        help=(
            "source_center averages a configurable center neighborhood per source, then averages sources; "
            "dense_pixel reproduces the legacy ellipse-pixel-weighted loss."
        ),
    )
    parser.add_argument(
        "--shape-center-size",
        type=int,
        default=3,
        help="Positive odd side length of the per-source shape supervision core, normally 3 or 5.",
    )
    parser.add_argument(
        "--shape-geometry-loss",
        choices=("legacy_area_ratio", "log_spd"),
        default="legacy_area_ratio",
        help=(
            "Per-location ellipse geometry loss. log_spd uses a matrix-free FP32 Log-Euclidean distance "
            "between ellipse covariance matrices; legacy_area_ratio preserves the previous loss."
        ),
    )
    parser.add_argument(
        "--shape-angle-weight",
        type=float,
        default=4.0,
        help=(
            "Weight for the legacy angular loss 1-cos(2*delta theta). "
            "Ignored by --shape-geometry-loss log_spd."
        ),
    )
    parser.add_argument(
        "--freeze-proposal-after-epochs",
        type=int,
        default=-1,
        help=(
            "For sam_per_band, freeze confidence/shape/CEN proposal heads after this many epochs "
            "and set confidence/shape/center losses to 0. Example: 20 trains epochs 0-19 normally."
        ),
    )
    parser.add_argument(
        "--mask-loss-weight",
        type=float,
        default=0.0,
        help="Outer weight for SAM prompt mask-decoder loss. Set >0 to train point+box prompted masks.",
    )
    parser.add_argument(
        "--mask-loss-warmup-epochs",
        type=int,
        default=0,
        help=(
            "Disable SAM mask loss for the first N epochs, then enable --mask-loss-weight. "
            "N=5 disables epochs 0-4 and enables mask loss from epoch 5."
        ),
    )
    parser.add_argument("--mask-dice-weight", type=float, default=0.0)
    parser.add_argument("--mask-bce-weight", type=float, default=0.0)
    parser.add_argument("--mask-centroid-weight", type=float, default=0.2)
    parser.add_argument("--mask-outside-weight", type=float, default=0.5)
    parser.add_argument("--mask-min-area-weight", type=float, default=0.1)
    parser.add_argument("--mask-max-area-weight", type=float, default=0.1)
    parser.add_argument("--mask-pred-iou-weight", type=float, default=0.1)
    parser.add_argument("--mask-stability-weight", type=float, default=0.1)
    parser.add_argument("--mask-unmatched-prompt-weight", type=float, default=0.2)
    parser.add_argument("--center-only-shape-factor", type=float, default=0.2)
    parser.add_argument("--mask-min-area-px", type=float, default=15.0)
    parser.add_argument(
        "--mask-area-ratio-lower",
        type=float,
        default=0.20,
        help="Lower bound for mask_area / prompted_shape_area used by mask area ratio loss.",
    )
    parser.add_argument(
        "--mask-area-ratio-upper",
        type=float,
        default=1.00,
        help="Upper bound for mask_area / prompted_shape_area used by mask area ratio loss.",
    )
    parser.add_argument("--mask-max-area-ratio", type=float, default=0.5)
    parser.add_argument("--mask-pred-iou-thresh", type=float, default=0.8)
    parser.add_argument("--mask-stability-score-thresh", type=float, default=0.95)
    parser.add_argument("--mask-stability-score-offset", type=float, default=1.0)
    parser.add_argument("--mask-stability-temperature", type=float, default=10.0)
    parser.add_argument("--mask-prompt-gt-epochs", type=int, default=5)
    parser.add_argument("--mask-prompt-pred-epoch", type=int, default=30)
    parser.add_argument("--mask-prompt-center-only", action="store_true", help="Use center-only prompts for SAM mask loss instead of center + bbox prompts for greater flexibility.")
    parser.add_argument(
        "--mask-max-gt-per-sample",
        type=int,
        default=0,
        help="Maximum GT prompts per flattened batch/band item for SAM mask loss. <=0 keeps all GT prompts.",
    )
    parser.add_argument(
        "--mask-max-pred-per-sample",
        type=int,
        default=0,
        help="Maximum predicted prompts per flattened batch/band item for mask loss. <=0 keeps all detections.",
    )
    parser.add_argument("--mask-prompt-chunk-size", type=int, default=128)
    parser.add_argument(
        "--mask-selection",
        choices=("pred_iou", "loss"),
        default="pred_iou",
        help=(
            "How to choose among SAM multimask outputs during mask loss. "
            "pred_iou keeps SAM's highest predicted-IoU mask; loss picks the mask with the lowest current mask loss."
        ),
    )
    parser.add_argument(
        "--disable-mask-multimask",
        action="store_true",
        help="Use SAM's single-mask output for mask loss instead of best-of-three multimask supervision.",
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
        help="Keep legacy compatibility flag. Strict masks are treated as low-weight center-only sources, not excluded detections.",
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
        "--amp",
        choices=("off", "bf16"),
        default="bf16",
        help="Automatic mixed precision mode. bf16 is recommended on A800/Hopper/Ampere GPUs.",
    )
    parser.add_argument(
        "--disable-tf32",
        action="store_true",
        help="Disable TF32 matmul/cuDNN acceleration on CUDA devices.",
    )
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
        "--ddp-static-graph",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "DDP static_graph mode. auto keeps static graph except when SAM mask-loss warmup "
            "changes which parameters are used."
        ),
    )
    parser.add_argument(
        "--ddp-find-unused-parameters",
        action="store_true",
        help="Force DDP find_unused_parameters=True. auto also enables this when static graph is disabled.",
    )
    parser.add_argument(
        "--ddp-progress-step-mode",
        choices=("global", "rank"),
        default="global",
        help=(
            "DDP tqdm/W&B x-axis convention. global advances by world_size per synchronized iteration; "
            "rank restores the original per-rank step count. This does not affect optimization or scheduling."
        ),
    )
    parser.add_argument(
        "--patch-val",
        action="store_true",
        help="Split training and validation by random patch instead of random cutout. This is a coarser split that may better reflect generalization to new sky areas, but results in higher variance.",
    )
    parser.add_argument("--dist-url", default="env://", help="Distributed init method, normally env:// for torchrun.")
    parser.add_argument("--local-rank", type=int, default=0, help="Local rank fallback when LOCAL_RANK is not set.")
    parser.add_argument("--ckpt-interval", type=int, default=-1, help="Epoch interval for saving checkpoints. Set <=0 to disable intermediate checkpoints.")
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
    if args.zarr_random_image_batches:
        if args.data_format != "zarr":
            raise ValueError("--zarr-random-image-batches requires --data-format zarr")
        if args.mode != "train":
            print("WARNING: --zarr-random-image-batches is train-only; eval uses regular multiband Zarr discovery.")
            args.zarr_random_image_batches = False
        elif args.model_variant == "auto":
            args.model_variant = "sam_per_band"
        elif args.model_variant != "sam_per_band":
            raise ValueError("--zarr-random-image-batches is currently implemented only for --model-variant sam_per_band")
        args.disable_ex_loss = True
        args.use_ex_link_postprocess = False
        args.train_detect_ex_link = False
        args.enable_triplet = False
    if args.model_variant == "sam_per_band":
        args.seg_loss_weight = 0.0
        args.disable_ex_loss = True
        args.enable_en_loss = False
        args.use_en_postprocess = False
        args.use_ex_link_postprocess = False
        args.train_detect_ex_link = False
        args.enable_triplet = False
        args.enable_pu_self_training = False
    if len(args.segmentation_class_weights) < 2:
        raise ValueError("--segmentation-class-weights requires at least background and foreground weights")
    if int(args.seg_classes) < 2:
        raise ValueError("--seg-classes must be >= 2")
    if int(args.shape_center_size) <= 0 or int(args.shape_center_size) % 2 != 1:
        raise ValueError("--shape-center-size must be a positive odd integer, normally 3 or 5")
    distributed, rank, world_size, local_rank, device, backend = _setup_distributed(args)
    is_main = _is_main(rank)
    run = None
    if device.type == "cuda" and not bool(args.disable_tf32):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    amp_dtype = torch.bfloat16 if device.type == "cuda" and args.amp == "bf16" else None

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
        discover_max_records = None if ((args.mode == "eval" and eval_patch_specs) or explicit_train_val_patches) else args.max_records
        if args.data_format == "zarr":
            if reference_dir is not None or cutout_dir is not None or band_reference_root is not None:
                raise ValueError("--reference-dir/--cutout-dir/--band-reference-root are legacy-only options")
            if args.mode == "train" and bool(args.zarr_random_image_batches):
                records = discover_zarr_image_records(root, bands=args.bands, max_records=discover_max_records)
                if is_main:
                    print(
                        f"Using {len(records)} image-level Zarr records for SAM training; "
                        f"batch-size={args.batch_size} counts images."
                    )
            else:
                records = discover_zarr_records(root, bands=args.bands, max_records=discover_max_records)
        else:
            records = discover_cutout_records(
                root,
                reference_dir=reference_dir,
                cutout_dir=cutout_dir,
                band_reference_root=band_reference_root,
                bands=args.bands,
                max_records=discover_max_records,
            )
        dataset_sources = args.eval_dataset_sources if args.mode == "eval" and args.eval_dataset_sources else args.dataset_sources
        if args.mode == "train" and (args.train_dataset_sources or args.val_dataset_sources):
            dataset_sources = ("all",)
        before_source_count = len(records)
        records = _filter_records_by_dataset_sources(records, dataset_sources)
        if not records:
            raise RuntimeError(
                f"No records matched dataset sources {list(dataset_sources)} out of {before_source_count} discovered records"
            )
        if args.mode == "eval" and eval_patch_specs:
            before_count = len(records)
            records = _filter_records_by_patches(records, eval_patch_specs, root, seed=args.seed)
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
            if args.train_dataset_sources:
                all_train_source_records = _filter_records_by_dataset_sources(all_records, args.train_dataset_sources)
            else:
                all_train_source_records = list(all_records)
            if args.val_dataset_sources:
                all_val_source_records = _filter_records_by_dataset_sources(all_records, args.val_dataset_sources)
            else:
                all_val_source_records = list(all_records)
            if train_patch_specs:
                train_records = _filter_records_by_patches(
                    all_train_source_records,
                    train_patch_specs,
                    root,
                    seed=args.seed,
                )
            elif val_patch_specs:
                val_records_tmp = _filter_records_by_patches(
                    all_val_source_records,
                    val_patch_specs,
                    root,
                    seed=args.seed,
                )
                val_names = {rec.name for rec in val_records_tmp}
                train_records = [rec for rec in all_train_source_records if rec.name not in val_names]
            else:
                train_records = list(all_train_source_records)

            if val_patch_specs:
                val_records = _filter_records_by_patches(
                    all_val_source_records,
                    val_patch_specs,
                    root,
                    seed=args.seed,
                )
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
            if args.mode == "train" and (args.train_dataset_sources or args.val_dataset_sources):
                train_source_records = (
                    _filter_records_by_dataset_sources(records, args.train_dataset_sources)
                    if args.train_dataset_sources
                    else list(records)
                )
            else:
                train_source_records = list(records)
            train_records, val_records = split_records(
                train_source_records,
                args.val_fraction,
                args.seed,
                fixed_val_names=args.fixed_val_names,
                patch_val=args.patch_val,
            )
            if args.mode == "train" and args.val_dataset_sources:
                val_source_records = _filter_records_by_dataset_sources(records, args.val_dataset_sources)
                val_names = {rec.name for rec in val_records}
                val_records = [rec for rec in val_source_records if rec.name in val_names]
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
            confidence_outer_weight=float(args.confidence_loss_weight),
            confidence_pos_weight=float(args.confidence_pos_weight),
            confidence_loss_mode=str(args.confidence_loss_mode),
            confidence_ce_weights=tuple(float(v) for v in args.confidence_ce_weights),
            style_router_loss_weight=(
                float(args.style_router_loss_weight) if bool(args.sam_encoder_style_prompt) else 0.0
            ),
            small_shape_loss_weight=float(args.small_shape_loss_weight),
            small_shape_area_min=float(args.small_shape_area_min),
            small_shape_area_tau=float(args.small_shape_area_tau),
            small_shape_ordinal_threshold=float(args.small_shape_ordinal_threshold),
            small_shape_scope=str(args.small_shape_scope),
            shape_outer_weight=float(args.shape_loss_weight),
            shape_loss_mode=str(args.shape_loss_mode),
            shape_center_size=int(args.shape_center_size),
            shape_geometry_loss=str(args.shape_geometry_loss),
            center_position=float(args.center_loss_weight),
            shape_angle_weight=float(args.shape_angle_weight),
            triplet_outer_weight=float(args.triplet_outer_weight),
            matcher_outer_weight=float(args.matcher_outer_weight),
            mask_outer_weight=float(args.mask_loss_weight),
            mask_loss_warmup_epochs=int(args.mask_loss_warmup_epochs),
            mask_dice=float(args.mask_dice_weight),
            mask_bce=float(args.mask_bce_weight),
            mask_centroid=float(args.mask_centroid_weight),
            mask_outside=float(args.mask_outside_weight),
            mask_min_area=float(args.mask_min_area_weight),
            mask_max_area=float(args.mask_max_area_weight),
            mask_pred_iou=float(args.mask_pred_iou_weight),
            mask_stability=float(args.mask_stability_weight),
            mask_unmatched_prompt=float(args.mask_unmatched_prompt_weight),
            center_only_shape_factor=float(args.center_only_shape_factor),
            mask_min_area_px=float(args.mask_min_area_px),
            mask_area_ratio_lower=float(args.mask_area_ratio_lower),
            mask_area_ratio_upper=float(args.mask_area_ratio_upper),
            mask_max_area_ratio=float(args.mask_max_area_ratio),
            mask_pred_iou_thresh=float(args.mask_pred_iou_thresh),
            mask_stability_score_thresh=float(args.mask_stability_score_thresh),
            mask_stability_score_offset=float(args.mask_stability_score_offset),
            mask_stability_temperature=float(args.mask_stability_temperature),
            mask_prompt_gt_epochs=int(args.mask_prompt_gt_epochs),
            mask_prompt_pred_epoch=int(args.mask_prompt_pred_epoch),
            mask_max_gt_per_sample=int(args.mask_max_gt_per_sample),
            mask_max_pred_per_sample=int(args.mask_max_pred_per_sample),
            mask_prompt_chunk_size=int(args.mask_prompt_chunk_size),
            mask_multimask=not bool(args.disable_mask_multimask),
            mask_selection=str(args.mask_selection),
            mask_prompt_center_only=bool(args.mask_prompt_center_only),
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
            noncoadd_visibility_snr_filter=bool(args.noncoadd_snr_filter),
            noncoadd_visibility_ignore_snr=float(args.noncoadd_snr_ignore_thresh),
            noncoadd_visibility_center_only_snr=float(args.noncoadd_snr_center_only_thresh),
            noncoadd_visibility_ap_radius=float(args.noncoadd_snr_ap_radius),
            noncoadd_visibility_annulus_r_in=float(args.noncoadd_snr_annulus_r_in),
            noncoadd_visibility_annulus_r_out=float(args.noncoadd_snr_annulus_r_out),
            noncoadd_visibility_annulus_exclude_radius=float(args.noncoadd_snr_annulus_exclude_radius),
        )
        pseudo_label_path = out_dir / "pseudo_labels" / "latest.json" if args.enable_pu_self_training else None
        dataset_cls = ZarrCutoutDataset if args.data_format == "zarr" else AstroCutoutDataset
        if args.data_format == "zarr" and args.enable_pu_self_training:
            raise ValueError("--enable-pu-self-training is not yet supported with --data-format zarr")
        train_ds = dataset_cls(
            train_records,
            augment=True,
            pseudo_label_path=pseudo_label_path,
            pseudo_confidence_weight=args.pu_pseudo_conf_weight,
            pseudo_seg_weight=args.pu_pseudo_seg_weight,
            pseudo_shape_weight=args.pu_pseudo_shape_weight,
            **common_ds,
        )
        val_ds = dataset_cls(
            val_records,
            augment=False,
            load_eval_ignore_sources=args.mode == "eval" or int(args.detect_every) > 0,
            **common_ds,
        )
        pseudo_detect_loader = None
        if args.enable_pu_self_training and is_main and args.mode == "train":
            pseudo_detect_ds = dataset_cls(train_records, augment=False, **common_ds)
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
        train_epoch_setter = None
        use_zarr_worker_chunks = args.data_format == "zarr" and bool(args.zarr_worker_owned_chunks)
        use_zarr_chunk_batches = args.data_format == "zarr" and bool(args.zarr_chunk_local_batches)
        if use_zarr_worker_chunks:
            train_iter_ds = (
                ZarrChunkBatchIterableDataset(
                    train_records,
                    batch_size=args.batch_size,
                    shuffle=args.mode == "train",
                    seed=args.seed,
                    num_replicas=world_size if distributed and args.mode == "train" else 1,
                    rank=rank if distributed and args.mode == "train" else 0,
                    drop_last=bool(args.zarr_drop_last),
                    shuffle_within_chunk=bool(args.zarr_shuffle_within_chunk),
                    equalize_replicas=bool(distributed and args.mode == "train"),
                    augment=True,
                    pseudo_label_path=pseudo_label_path,
                    pseudo_confidence_weight=args.pu_pseudo_conf_weight,
                    pseudo_seg_weight=args.pu_pseudo_seg_weight,
                    pseudo_shape_weight=args.pu_pseudo_shape_weight,
                    **common_ds,
                )
                if args.mode == "train"
                else None
            )
            val_iter_ds = ZarrChunkBatchIterableDataset(
                val_records,
                batch_size=args.batch_size,
                shuffle=False,
                seed=args.seed,
                num_replicas=world_size if distributed and args.mode == "train" else 1,
                rank=rank if distributed and args.mode == "train" else 0,
                drop_last=False,
                shuffle_within_chunk=False,
                augment=False,
                load_eval_ignore_sources=args.mode == "eval" or int(args.detect_every) > 0,
                **common_ds,
            )

            def _zarr_iter_loader_kwargs():
                kwargs: dict[str, object] = {
                    "batch_size": None,
                    "num_workers": args.num_workers,
                    "collate_fn": zarr_passthrough_batch,
                    "pin_memory": bool(args.pin_memory),
                }
                if int(args.num_workers) > 0:
                    kwargs["persistent_workers"] = bool(args.persistent_workers)
                    kwargs["prefetch_factor"] = max(1, int(args.prefetch_factor))
                return kwargs

            train_sampler = None
            train_loader = DataLoader(train_iter_ds, **_zarr_iter_loader_kwargs()) if train_iter_ds is not None else None
            val_loader = DataLoader(val_iter_ds, **_zarr_iter_loader_kwargs())
            train_epoch_setter = train_iter_ds
            if train_iter_ds is not None and is_main:
                train_batches = len(train_iter_ds)
                if train_batches == 0:
                    raise ValueError(
                        "Zarr worker-owned train iterator produced 0 batches. "
                        f"batch_size={int(args.batch_size)}, zarr_drop_last={bool(args.zarr_drop_last)}, "
                        f"max_local_chunk_size={int(train_iter_ds.max_local_chunk_size())}. "
                        "Use batch_size <= the Zarr chunk tile count, disable --zarr-drop-last, "
                        "or regenerate Zarr with a larger --chunk-tiles/CHUNK_TILES."
                    )
                incomplete_train_batches = int(train_iter_ds.incomplete_batch_count())
                if incomplete_train_batches > 0:
                    message = (
                        f"WARNING: worker-owned Zarr chunk iterator will emit {incomplete_train_batches} "
                        f"incomplete train batch(es) on rank {rank}; batch-size changes can trigger "
                        "extra torch.compile recompilation pauses."
                    )
                    if (bool(args.sam_compile) or bool(args.sam_compile_mask_decoder)) and not bool(args.zarr_drop_last):
                        message += " Add --zarr-drop-last for compile-heavy runs."
                    print(message, flush=True)
            num_val_local = int(val_iter_ds.local_sample_count())
        elif use_zarr_chunk_batches:
            train_sampler = (
                ZarrChunkLocalBatchSampler(
                    train_records,
                    batch_size=args.batch_size,
                    shuffle=args.mode == "train",
                    seed=args.seed,
                    num_replicas=world_size if distributed and args.mode == "train" else 1,
                    rank=rank if distributed and args.mode == "train" else 0,
                    drop_last=bool(args.zarr_drop_last),
                    shuffle_within_chunk=bool(args.zarr_shuffle_within_chunk),
                    equalize_replicas=bool(distributed and args.mode == "train"),
                )
                if args.mode == "train"
                else None
            )
            val_batch_sampler = ZarrChunkLocalBatchSampler(
                val_records,
                batch_size=args.batch_size,
                shuffle=False,
                seed=args.seed,
                num_replicas=world_size if distributed and args.mode == "train" else 1,
                rank=rank if distributed and args.mode == "train" else 0,
                drop_last=False,
                shuffle_within_chunk=False,
            )

            def _zarr_loader_kwargs(batch_sampler):
                kwargs: dict[str, object] = {
                    "batch_sampler": batch_sampler,
                    "num_workers": args.num_workers,
                    "collate_fn": collate_cutouts,
                    "pin_memory": bool(args.pin_memory),
                }
                if int(args.num_workers) > 0:
                    kwargs["persistent_workers"] = bool(args.persistent_workers)
                    kwargs["prefetch_factor"] = max(1, int(args.prefetch_factor))
                return kwargs

            train_loader = DataLoader(train_ds, **_zarr_loader_kwargs(train_sampler)) if train_sampler is not None else None
            val_loader = DataLoader(val_ds, **_zarr_loader_kwargs(val_batch_sampler))
            if train_sampler is not None and is_main:
                train_batches = len(train_sampler)
                if train_batches == 0:
                    raise ValueError(
                        "Zarr chunk-local train sampler produced 0 batches. "
                        f"batch_size={int(args.batch_size)}, zarr_drop_last={bool(args.zarr_drop_last)}, "
                        f"max_local_chunk_size={int(train_sampler.max_local_chunk_size())}. "
                        "Use batch_size <= the Zarr chunk tile count, disable --zarr-drop-last, "
                        "or regenerate Zarr with a larger --chunk-tiles/CHUNK_TILES."
                    )
                incomplete_train_batches = sum(1 for batch in train_sampler if len(batch) < int(args.batch_size))
                if incomplete_train_batches > 0:
                    message = (
                        f"WARNING: Zarr chunk-local sampler will emit {incomplete_train_batches} "
                        f"incomplete train batch(es) on rank {rank}; batch-size changes can trigger "
                        "extra torch.compile recompilation pauses."
                    )
                    if (bool(args.sam_compile) or bool(args.sam_compile_mask_decoder)) and not bool(args.zarr_drop_last):
                        message += " Add --zarr-drop-last for compile-heavy runs."
                    print(message, flush=True)
            num_val_local = sum(len(batch) for batch in val_batch_sampler)
            train_epoch_setter = train_sampler
        else:
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
            num_val_local = len(val_loader_ds)
            train_epoch_setter = train_sampler

        if args.model_variant == "auto":
            model_variant = "fused_encoder" if len(args.bands) > 1 or args.enable_en_loss else "fused"
        else:
            model_variant = args.model_variant
        args.model_variant = model_variant
        if model_variant == "sam_per_band":
            args.seg_loss_weight = 0.0
            args.disable_ex_loss = True
            args.enable_en_loss = False
            args.use_en_postprocess = False
            args.use_ex_link_postprocess = False
            args.train_detect_ex_link = False
            args.enable_triplet = False
            args.enable_pu_self_training = False
            if (
                int(args.sam_lr_phase2_epoch) < 0
                and (
                    args.sam_encoder_lr_after is not None
                    or args.sam_head_lr_after is not None
                    or args.sam_decoder_lr_after is not None
                )
            ):
                raise ValueError(
                    "--sam-*-lr-after requires --sam-lr-phase2-epoch >= 0; otherwise the LR override never activates."
                )
        matcher_variant = model_variant in ("per_band", "fused_encoder", "sam_per_band")
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

        if model_variant == "sam_per_band":
            if args.sam_encoder_style_prompt is None:
                args.sam_encoder_style_prompt = False
                if args.checkpoint:
                    style_ckpt = torch.load(_expand_path(args.checkpoint), map_location="cpu")
                    style_state = style_ckpt.get("model", style_ckpt) if isinstance(style_ckpt, dict) else style_ckpt
                    if isinstance(style_state, dict):
                        args.sam_encoder_style_prompt = any(
                            "encoder.style_router." in str(key)
                            or "encoder.image_encoder.style_adapters." in str(key)
                            for key in style_state
                        )
                    del style_ckpt, style_state
            if (
                args.mode == "train"
                and bool(args.zarr_random_image_batches)
                and bool(args.sam_encoder_style_prompt)
            ):
                if is_main:
                    print(
                        "WARNING: disabling --sam-encoder-style-prompt for image-level Zarr training; "
                        "image-level batches contain one band per sample, so the multiband style router is incompatible."
                    )
                args.sam_encoder_style_prompt = False
            if args.sam_decoder_film is None:
                args.sam_decoder_film = False
                if args.checkpoint:
                    film_ckpt = torch.load(_expand_path(args.checkpoint), map_location="cpu")
                    film_state = film_ckpt.get("model", film_ckpt) if isinstance(film_ckpt, dict) else film_ckpt
                    if isinstance(film_state, dict):
                        args.sam_decoder_film = any(
                            "decoder.denoised_film." in str(key) for key in film_state
                        )
                    del film_ckpt, film_state
            if bool(args.sam_encoder_style_prompt) and args.sam_model_type != "vit_b":
                raise ValueError("--sam-encoder-style-prompt currently supports vit_b only")
            if bool(args.sam_encoder_style_prompt) and bool(args.sam_decoder_film):
                raise ValueError("--sam-encoder-style-prompt and --sam-decoder-film cannot be enabled together")
            weights = replace(
                weights,
                style_router_loss_weight=(
                    float(args.style_router_loss_weight) if bool(args.sam_encoder_style_prompt) else 0.0
                ),
            )
            model = build_sam_cellect2d(
                args.sam_model_type,
                checkpoint=_expand_path(args.sam_checkpoint) if args.sam_checkpoint else None,
                num_bands=len(args.bands),
                image_size=512,
                patch_size=16,
                seg_classes=args.seg_classes,
                confidence_levels=5,
                embedding_dim=args.embedding_dim,
                shape_channels=3,
                decoder_channels=(
                    args.base_channels * 8,
                    args.base_channels * 4,
                    args.base_channels * 2,
                    args.base_channels,
                ),
                use_cen=not args.disable_sam_cen,
                cen_input_image=True,
                cen_width=max(2, args.base_channels // 4),
                decoder_denoised_film=bool(args.sam_decoder_film),
                encoder_style_prompt=bool(args.sam_encoder_style_prompt),
                style_prompt_dim=int(args.style_prompt_dim),
                style_prompt_layers=(2, 5, 8),
                style_adapter_dim=int(args.style_adapter_dim),
                style_router_temperature=float(args.style_router_temperature),
                candidate_count=args.matcher_candidate_count,
                shape_feature_dim=6,
                enable_matchers=False,
                astro_preprocess_in_model=False,
                dynamic_image_size=bool(args.sam_dynamic_image_size),
            ).to(device)
        elif model_variant == "per_band":
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
            if model_variant == "sam_per_band" and (
                bool(args.sam_decoder_film) or bool(args.sam_encoder_style_prompt)
            ):
                incompatible = model.load_state_dict(state, strict=False)
                allowed_missing_prefixes = []
                if bool(args.sam_decoder_film):
                    allowed_missing_prefixes.append("decoder.denoised_film.")
                if bool(args.sam_encoder_style_prompt):
                    allowed_missing_prefixes.extend((
                        "encoder.style_router.",
                        "encoder.style_prompt_",
                        "encoder.image_encoder.style_adapters.",
                    ))
                invalid_missing = [
                    key for key in incompatible.missing_keys
                    if not str(key).startswith(tuple(allowed_missing_prefixes))
                ]
                if invalid_missing or incompatible.unexpected_keys:
                    raise RuntimeError(
                        "Checkpoint is incompatible with the requested SAM conditioning modules: "
                        f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
                    )
                if incompatible.missing_keys and is_main:
                    print(
                        f"Initialized requested SAM conditioning modules; checkpoint omitted "
                        f"{len(incompatible.missing_keys)} new parameter(s).",
                        flush=True,
                    )
            else:
                model.load_state_dict(state)
            if hasattr(model, "EX") and hasattr(model, "EN"):
                if isinstance(ckpt, dict) and ckpt.get("EX") is not None:
                    model.EX.load_state_dict(ckpt["EX"])
                if isinstance(ckpt, dict) and ckpt.get("EN") is not None:
                    model.EN.load_state_dict(ckpt["EN"])

        model = _compile_sam_mask_decoder_modules(model, args, is_main=is_main)
        model = _compile_sam_runtime_model(model, args, is_main=is_main)

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
                use_ordinal_expectation=bool(args.use_ordinal_expectation),
                debug_ordinal_expectation=bool(args.debug_ordinal_expectation),
                center_refinement=args.center_refinement,
                center_refinement_radius=args.center_refinement_radius,
                use_en_postprocess=en_postprocess_enabled,
                en_threshold=args.en_postprocess_threshold,
                use_ex_link_postprocess=ex_link_postprocess_enabled,
                ex_link_threshold=args.ex_link_threshold,
                band_names=args.bands,
                collect_candidate_stats=bool(args.debug_detection_metrics),
                ignore_mask_during_detection=bool(args.ignore_mask_during_detection),
                epoch_index=0,
                ellipse_sigma=float(args.ellipse_sigma),
                amp_dtype=amp_dtype,
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
                        use_ordinal_expectation=bool(args.use_ordinal_expectation),
                        debug_ordinal_expectation=bool(args.debug_ordinal_expectation),
                        center_refinement=args.center_refinement,
                        center_refinement_radius=args.center_refinement_radius,
                        match_radius=center_radius_px,
                        pixel_scale_arcsec=float(args.pixel_scale_arcsec),
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
                link_json = _expand_path(args.linking_metrics_json) if args.linking_metrics_json else out_dir / "linking_metrics.json"
                if _write_linking_metrics_json(link_json, det, epoch=None):
                    det["linking_metrics_json"] = str(link_json)
                print(json.dumps({"dense": dense, "detection": det}, indent=2))
            _sync_distributed()
            return

        if distributed:
            model = _wrap_ddp(model, args, device, local_rank)
            if is_main:
                print(
                    "DDP resolved: "
                    f"static_graph={bool(getattr(args, '_ddp_static_graph_resolved', False))}, "
                    f"find_unused_parameters={bool(getattr(args, '_ddp_find_unused_parameters_resolved', False))}"
                )
        # CELLECT style model
        sam_iteration_scheduler = None
        if model_variant != "sam_per_band":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
        else:
            total_steps = max(1, int(args.epochs) * max(1, len(train_loader)))
            warmup_steps = max(1, int(round(float(args.sam_warmup_ratio) * float(total_steps))))
            drop_steps = tuple(
                sorted(
                    {
                        min(total_steps, max(1, int(round(float(fraction) * float(total_steps)))))
                        for fraction in args.sam_lr_drop_fractions
                    }
                )
            )
            sam_param_groups = sam_optimizer_param_groups(
                model,
                head_lr=float(args.lr),
                encoder_lr=float(args.sam_encoder_lr),
                weight_decay=float(args.weight_decay),
            )
            use_fused_adamw = bool(args.sam_fused_adamw) and device.type == "cuda"
            if use_fused_adamw:
                try:
                    optimizer = torch.optim.AdamW(sam_param_groups, fused=True)
                except TypeError:
                    optimizer = torch.optim.AdamW(sam_param_groups)
                    use_fused_adamw = False
            else:
                optimizer = torch.optim.AdamW(sam_param_groups)
            sam_iteration_scheduler = WarmupStepIterationLR(
                optimizer,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                drop_steps=drop_steps,
                drop_gamma=float(args.sam_lr_drop_gamma),
            )
            scheduler = None
            sam_phase2_lr_overrides: dict[str, float] = {}
            if int(args.sam_lr_phase2_epoch) >= 0:
                if args.sam_head_lr_after is not None:
                    sam_phase2_lr_overrides["head"] = float(args.sam_head_lr_after)
                if args.sam_encoder_lr_after is not None:
                    sam_phase2_lr_overrides["encoder"] = float(args.sam_encoder_lr_after)
                if args.sam_decoder_lr_after is not None:
                    sam_phase2_lr_overrides["sam_decoder"] = float(args.sam_decoder_lr_after)
        if model_variant != "sam_per_band":
            sam_phase2_lr_overrides = {}

        triplet_loss_fn = HardTripletLoss(weights.triplet_margin)

        metadata = {
            "args": vars(args),
            "loss_weights": asdict(weights),
            "num_records": len(records),
            "num_train": len(train_records),
            "num_val": len(val_records),
            "num_val_local": num_val_local,
            "fixed_val_names": list(args.fixed_val_names),
            "val_record_names": [rec.name for rec in val_records],
            "train_patch_specs": sorted(train_patch_specs),
            "val_patch_specs": sorted(val_patch_specs),
            "center_radius_px": center_radius_px,
            "confidence_score": str(args.confidence_score),
            "use_ordinal_expectation": bool(args.use_ordinal_expectation),
            "debug_ordinal_expectation": bool(args.debug_ordinal_expectation),
            "debug_detection_metrics": bool(args.debug_detection_metrics),
            "center_refinement": str(args.center_refinement),
            "center_refinement_radius": int(args.center_refinement_radius),
            "targets_dir": str(targets_dir) if targets_dir is not None else None,
            "image_cache_dir": str(image_cache_dir) if image_cache_dir is not None else None,
            "band_reference_root": str(band_reference_root) if band_reference_root is not None else None,
            "model_variant": model_variant,
            "sam_model_type": str(args.sam_model_type) if model_variant == "sam_per_band" else None,
            "sam_checkpoint": str(_expand_path(args.sam_checkpoint)) if model_variant == "sam_per_band" and args.sam_checkpoint else None,
            "sam_cen_enabled": bool(model_variant == "sam_per_band" and not args.disable_sam_cen),
            "sam_decoder_film": bool(model_variant == "sam_per_band" and args.sam_decoder_film),
            "sam_encoder_style_prompt": bool(model_variant == "sam_per_band" and args.sam_encoder_style_prompt),
            "style_prompt_dim": int(args.style_prompt_dim),
            "style_adapter_dim": int(args.style_adapter_dim),
            "style_prompt_layers": [2, 5, 8] if bool(args.sam_encoder_style_prompt) else [],
            "style_router_temperature": float(args.style_router_temperature),
            "style_router_loss_weight": float(weights.style_router_loss_weight),
            "sam_astro_preprocess_in_model": bool(False) if model_variant == "sam_per_band" else None,
            "sam_compile": bool(model_variant == "sam_per_band" and args.sam_compile),
            "sam_compile_backend": str(args.sam_compile_backend) if model_variant == "sam_per_band" and args.sam_compile else None,
            "sam_compile_mode": str(args.sam_compile_mode) if model_variant == "sam_per_band" and args.sam_compile else None,
            "sam_compile_fullgraph": bool(args.sam_compile_fullgraph) if model_variant == "sam_per_band" and args.sam_compile else None,
            "sam_compile_mask_decoder": bool(model_variant == "sam_per_band" and args.sam_compile_mask_decoder),
            "mask_loss_interval": int(args.mask_loss_interval),
            "mask_loss_interval_scale": bool(args.mask_loss_interval_scale),
            "sam_lr_schedule": (
                {
                    "type": "iteration_warmup_step",
                    "head_lr": float(args.lr),
                    "encoder_lr": float(args.sam_encoder_lr),
                    "sam_decoder_lr": float(args.lr),
                    "total_steps": int(sam_iteration_scheduler.total_steps),
                    "warmup_steps": int(sam_iteration_scheduler.warmup_steps),
                    "drop_steps": [int(step) for step in sam_iteration_scheduler.drop_steps],
                    "drop_gamma": float(sam_iteration_scheduler.drop_gamma),
                    "phase2_epoch": int(args.sam_lr_phase2_epoch),
                    "phase2_lr_overrides": dict(sam_phase2_lr_overrides),
                }
                if sam_iteration_scheduler is not None
                else None
            ),
            "single_band_detector": bool(args.single_band_detector),
            "seg_classes": int(args.seg_classes),
            "distributed": distributed,
            "distributed_validation": bool(distributed and args.mode == "train"),
            "world_size": world_size,
            "ddp_progress_step_mode": str(args.ddp_progress_step_mode),
            "dist_backend": backend,
            "amp": str(args.amp),
            "amp_dtype": str(amp_dtype).replace("torch.", "") if amp_dtype is not None else None,
            "tf32_enabled": bool(device.type == "cuda" and not bool(args.disable_tf32)),
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
            run = _init_wandb_or_disable(args, out_dir=out_dir, metadata=metadata, is_main=is_main)
        _sync_distributed()

        best_val = float("inf")
        best_state_cpu: dict[str, torch.Tensor] | None = None
        global_step = 0

        def _wandb_iteration_log(metrics: dict[str, float], step: int) -> None:
            if run is None or not is_main:
                return
            payload: dict[str, float | int] = {}
            for key, value in metrics.items():
                if key == "lr":
                    payload["lr/iteration/default"] = value
                elif key.startswith("lr/"):
                    payload[f"lr/iteration/{key.removeprefix('lr/')}"] = value
                else:
                    payload[f"train/iteration/{key}"] = value
            payload["iteration"] = int(step)
            wandb.log(payload, step=int(step))

        wandb_iteration_log_fn = (
            _wandb_iteration_log
            if str(args.wandb_mode) != "disabled" and int(args.wandb_log_interval) > 0
            else None
        )

        sam_phase2_applied = False

        def _phase2_active(epoch: int) -> bool:
            return (
                model_variant == "sam_per_band"
                and int(args.sam_lr_phase2_epoch) >= 0
                and int(epoch) >= int(args.sam_lr_phase2_epoch)
            )

        def _phase2_group_lr(epoch: int, group_name: str) -> float | None:
            if not _phase2_active(epoch):
                return None
            if group_name not in sam_phase2_lr_overrides:
                return None
            return float(sam_phase2_lr_overrides[group_name])

        def _apply_phase2_lrs(epoch: int) -> bool:
            nonlocal sam_phase2_applied
            if (
                sam_iteration_scheduler is None
                or sam_phase2_applied
                or not _phase2_active(epoch)
                or not sam_phase2_lr_overrides
            ):
                return False
            sam_iteration_scheduler.set_base_lrs(sam_phase2_lr_overrides)
            sam_phase2_applied = True
            return True

        def _encoder_frozen(epoch: int) -> bool:
            lr_after = _phase2_group_lr(epoch, "encoder")
            return lr_after is not None and float(lr_after) <= 0.0

        def _proposal_frozen(epoch: int) -> bool:
            old_freeze = (
                model_variant == "sam_per_band"
                and int(args.freeze_proposal_after_epochs) >= 0
                and int(epoch) >= int(args.freeze_proposal_after_epochs)
            )
            lr_after = _phase2_group_lr(epoch, "head")
            return bool(old_freeze or (lr_after is not None and float(lr_after) <= 0.0))

        def _weights_for_epoch(epoch: int) -> LossWeights:
            if not _proposal_frozen(epoch):
                return weights
            return replace(
                weights,
                confidence_outer_weight=0.0,
                shape_outer_weight=0.0,
                center_position=0.0,
                small_shape_loss_weight=0.0,
                style_router_loss_weight=0.0,
                detach_mask_prompt_shapes=True,
            )

        def _wandb_epoch_loss_metrics(metrics: dict[str, float], active_weights: LossWeights) -> dict[str, float]:
            filtered: dict[str, float] = {
                "total": float(metrics["total"]),
            }
            if float(active_weights.confidence_outer_weight) > 0.0 and "confidence" in metrics:
                filtered["confidence"] = float(metrics["confidence"])
            if float(active_weights.segmentation_outer_weight) > 0.0 and "seg" in metrics:
                filtered["seg"] = float(metrics["seg"])
            if float(active_weights.shape_outer_weight) > 0.0 and "shape" in metrics:
                filtered["shape"] = float(metrics["shape"])
            if float(active_weights.center_position) > 0.0 and "center" in metrics:
                filtered["center"] = float(metrics["center"])
            if float(active_weights.small_shape_loss_weight) > 0.0 and "small_shape" in metrics:
                filtered["small_shape"] = float(metrics["small_shape"])
            if float(active_weights.style_router_loss_weight) > 0.0 and "style_router" in metrics:
                filtered["style_router"] = float(metrics["style_router"])
            if bool(args.enable_triplet) and float(active_weights.triplet_outer_weight) > 0.0 and "triplet" in metrics:
                filtered["triplet"] = float(metrics["triplet"])
            if bool(ex_enabled) and float(active_weights.matcher_outer_weight) > 0.0 and "ex_class" in metrics:
                filtered["ex_class"] = float(metrics["ex_class"])
            if bool(en_enabled) and float(active_weights.matcher_outer_weight) > 0.0 and "en_class" in metrics:
                filtered["en_class"] = float(metrics["en_class"])
            active_mask_keys = active_mask_loss_keys(active_weights)
            if float(active_weights.mask_outer_weight) > 0.0 and active_mask_keys and "mask" in metrics:
                filtered["mask"] = float(metrics["mask"])
                if "mask_gt" in metrics:
                    filtered["mask_gt"] = float(metrics["mask_gt"])
                if "mask_pred" in metrics:
                    filtered["mask_pred"] = float(metrics["mask_pred"])
                if "mask_prompts" in metrics:
                    filtered["mask_prompts"] = float(metrics["mask_prompts"])
                if "mask_gt_prompts" in metrics:
                    filtered["mask_gt_prompts"] = float(metrics["mask_gt_prompts"])
                if "mask_pred_prompts" in metrics:
                    filtered["mask_pred_prompts"] = float(metrics["mask_pred_prompts"])
                for key in active_mask_keys:
                    metric_key = f"mask_{key}"
                    if metric_key in metrics:
                        filtered[metric_key] = float(metrics[metric_key])
            return filtered

        for epoch in range(args.epochs):
            phase2_started = _apply_phase2_lrs(epoch)
            epoch_weights = _weights_for_epoch(epoch)
            proposal_frozen = _proposal_frozen(epoch)
            encoder_frozen = _encoder_frozen(epoch)
            if phase2_started and is_main:
                print(
                    f"SAM phase-2 LR overrides active from epoch {epoch}: "
                    f"{sam_phase2_lr_overrides}",
                    flush=True,
                )
            if (
                proposal_frozen
                and is_main
                and (
                    int(epoch) == int(args.freeze_proposal_after_epochs)
                    or (
                        _phase2_active(epoch)
                        and "head" in sam_phase2_lr_overrides
                        and float(sam_phase2_lr_overrides["head"]) <= 0.0
                        and int(epoch) == int(args.sam_lr_phase2_epoch)
                    )
                )
            ):
                print(
                    f"SAM proposal head frozen from epoch {epoch}: "
                    "confidence/shape/center losses disabled and prompt shapes detached.",
                    flush=True,
                )
            if (
                encoder_frozen
                and is_main
                and int(epoch) == int(args.sam_lr_phase2_epoch)
            ):
                print(
                    f"SAM encoder frozen from epoch {epoch}: encoder gradients are cleared before optimizer.step().",
                    flush=True,
                )
            if train_epoch_setter is not None:
                train_epoch_setter.set_epoch(epoch)
            display_step_scale = (
                world_size
                if distributed and str(args.ddp_progress_step_mode) == "global"
                else 1
            )
            train_metrics = run_epoch(
                model,
                train_loader,
                optimizer=optimizer,
                device=device,
                weights=epoch_weights,
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
                epoch_index=epoch,
                threshold=args.confidence_threshold,
                nms_radius=args.nms_radius,
                confidence_score=args.confidence_score,
                use_ordinal_expectation=bool(args.use_ordinal_expectation),
                debug_ordinal_expectation=bool(args.debug_ordinal_expectation),
                center_refinement=args.center_refinement,
                center_refinement_radius=args.center_refinement_radius,
                ellipse_sigma=float(args.ellipse_sigma),
                amp_dtype=amp_dtype,
                scheduler_step=sam_iteration_scheduler.step if sam_iteration_scheduler is not None else None,
                global_step_start=global_step,
                global_step_scale=display_step_scale,
                iteration_log_interval=int(args.wandb_log_interval),
                iteration_log_fn=wandb_iteration_log_fn,
                debug_batch_start=int(args.debug_batch_start),
                debug_batch_end=int(args.debug_batch_end),
                debug_batch_all_ranks=bool(args.debug_batch_all_ranks),
                distributed=distributed,
                show_progress=is_main,
                freeze_sam_proposal=proposal_frozen,
                freeze_sam_encoder=encoder_frozen,
                mask_loss_interval=max(1, int(args.mask_loss_interval)),
                mask_loss_interval_scale=bool(args.mask_loss_interval_scale),
            )
            global_step += len(train_loader) * display_step_scale

            eval_model = _runtime_unwrap_model(model)
            run_detect = int(args.detect_every) > 0 and ((epoch + 1) % int(args.detect_every) == 0)
            val_metrics, det_metrics = validate_epoch(
                eval_model,
                val_loader,
                device=device,
                weights=epoch_weights,
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
                use_ordinal_expectation=bool(args.use_ordinal_expectation),
                debug_ordinal_expectation=bool(args.debug_ordinal_expectation),
                center_refinement=args.center_refinement,
                center_refinement_radius=args.center_refinement_radius,
                use_en_postprocess=en_postprocess_enabled,
                en_threshold=args.en_postprocess_threshold,
                use_ex_link_postprocess=train_detect_ex_link_enabled,
                ex_link_threshold=args.ex_link_threshold,
                band_names=args.bands,
                collect_candidate_stats=bool(args.debug_detection_metrics),
                ignore_mask_during_detection=bool(args.ignore_mask_during_detection),
                epoch_index=epoch,
                ellipse_sigma=float(args.ellipse_sigma),
                amp_dtype=amp_dtype,
                distributed=distributed,
                show_progress=is_main,
            )
            if not run_detect:
                det_metrics = {"skipped": True, "detect_every": int(args.detect_every)}
            val_total = float(val_metrics["total"])

            val_total_tensor = torch.tensor([val_total], device=device, dtype=torch.float64)
            if distributed:
                dist.broadcast(val_total_tensor, src=0)
            if scheduler is not None:
                scheduler.step(float(val_total_tensor.item()))

            if is_main:
                lr_groups = {str(group.get("name", idx)): float(group["lr"]) for idx, group in enumerate(optimizer.param_groups)}
                effective_mask_outer_weight = (
                    0.0
                    if int(epoch) < max(0, int(epoch_weights.mask_loss_warmup_epochs))
                    else float(epoch_weights.mask_outer_weight)
                )
                log_line = {
                    "epoch": epoch,
                    "train": train_metrics,
                    "val": val_metrics,
                    "detection": det_metrics,
                    "lr": optimizer.param_groups[0]["lr"],
                    "lr_groups": lr_groups,
                    "mask_effective_outer_weight": effective_mask_outer_weight,
                    "proposal_frozen": proposal_frozen,
                    "encoder_frozen": encoder_frozen,
                    "sam_phase2_active": _phase2_active(epoch),
                    "sam_phase2_lr_overrides": dict(sam_phase2_lr_overrides),
                    "proposal_loss_weights": {
                        "confidence": float(epoch_weights.confidence_outer_weight),
                        "shape": float(epoch_weights.shape_outer_weight),
                        "center": float(epoch_weights.center_position),
                        "small_shape": float(epoch_weights.small_shape_loss_weight),
                    },
                }
                if sam_iteration_scheduler is not None:
                    log_line["lr_schedule"] = sam_iteration_scheduler.state_dict()
                if run_detect:
                    link_epoch_json = out_dir / f"linking_metrics_epoch_{int(epoch) + 1:04d}.json"
                    if _write_linking_metrics_json(link_epoch_json, det_metrics, epoch=int(epoch)):
                        link_latest_json = out_dir / "linking_metrics_latest.json"
                        _write_linking_metrics_json(link_latest_json, det_metrics, epoch=int(epoch))
                        det_metrics["linking_metrics_json"] = str(link_epoch_json)
                        det_metrics["linking_metrics_latest_json"] = str(link_latest_json)
                if run is not None:
                    epoch_payload = {
                        "epoch": int(epoch),
                        "iteration": int(global_step),
                        "prompt/predicted_ratio": float(
                            0.0
                            if int(epoch) < int(epoch_weights.mask_prompt_gt_epochs)
                            else (
                                1.0
                                if int(epoch) >= int(epoch_weights.mask_prompt_pred_epoch)
                                else float(int(epoch) - int(epoch_weights.mask_prompt_gt_epochs))
                                / max(
                                    float(
                                        int(epoch_weights.mask_prompt_pred_epoch)
                                        - int(epoch_weights.mask_prompt_gt_epochs)
                                    ),
                                    1.0,
                                )
                            )
                        ),
                        "mask/effective_outer_weight": float(
                            0.0
                            if int(epoch) < max(0, int(epoch_weights.mask_loss_warmup_epochs))
                            else float(epoch_weights.mask_outer_weight)
                        ),
                        "proposal/frozen": float(proposal_frozen),
                        "proposal/encoder_frozen": float(encoder_frozen),
                        "sam_phase2/active": float(_phase2_active(epoch)),
                        "proposal/confidence_loss_weight": float(epoch_weights.confidence_outer_weight),
                        "proposal/shape_loss_weight": float(epoch_weights.shape_outer_weight),
                        "proposal/center_loss_weight": float(epoch_weights.center_position),
                        "proposal/small_shape_loss_weight": float(epoch_weights.small_shape_loss_weight),
                    }
                    epoch_payload["prompt/gt_ratio"] = 1.0 - float(epoch_payload["prompt/predicted_ratio"])
                    epoch_payload.update(
                        {
                            f"train/epoch/{key}": float(value)
                            for key, value in _wandb_epoch_loss_metrics(train_metrics, epoch_weights).items()
                        }
                    )
                    epoch_payload.update(
                        {
                            f"val/epoch/{key}": float(value)
                            for key, value in _wandb_epoch_loss_metrics(val_metrics, epoch_weights).items()
                        }
                    )
                    epoch_payload["lr/epoch/default"] = float(optimizer.param_groups[0]["lr"])
                    epoch_payload.update({f"lr/epoch/{name}": float(value) for name, value in lr_groups.items()})
                    if sam_iteration_scheduler is not None:
                        sched_state = sam_iteration_scheduler.state_dict()
                        epoch_payload["lr_schedule/step_count"] = int(sched_state.get("step_count", 0))
                    wandb.log(epoch_payload, step=int(global_step))
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
                if args.ckpt_interval < 0:
                    torch.save(ckpt, out_dir / "last.pt")
                elif (epoch + 1) % int(args.ckpt_interval) == 0:
                    torch.save(ckpt, out_dir / f"epoch_{epoch + 1:04d}.pt")
                best_updated = False
                if val_metrics["total"] < best_val:
                    best_val = val_metrics["total"]
                    best_state_cpu = _state_dict_cpu_copy(eval_model)
                    torch.save(ckpt, out_dir / "best.pt")
                    best_updated = True
                log_line["best_updated"] = best_updated
                if (
                    model_variant != "sam_per_band"
                    and args.enable_pu_self_training
                    and int(args.pu_self_train_every) > 0
                    and ((epoch + 1) % int(args.pu_self_train_every) == 0)
                ):
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
                        use_ordinal_expectation=bool(args.use_ordinal_expectation),
                        debug_ordinal_expectation=bool(args.debug_ordinal_expectation),
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
                and model_variant != "sam_per_band"
                and pseudo_label_path is not None
                and args.mode == "train"
                and int(args.pu_self_train_every) > 0
                and ((epoch + 1) % int(args.pu_self_train_every) == 0)
            ):
                train_ds.reload_pseudo_labels(pseudo_label_path)
                train_loader = DataLoader(train_ds, **_loader_kwargs(args, shuffle=train_sampler is None, sampler=train_sampler))
    finally:
        if run is not None:
            wandb.finish()
        _cleanup_distributed()


if __name__ == "__main__":
    main()
