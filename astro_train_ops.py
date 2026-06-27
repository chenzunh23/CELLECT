"""Losses, post-processing, and epoch/evaluation helpers for AstroCELLECT training."""

from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from astro_cellect2d import AstroMatchNet2D, ordinal_confidence_loss
from astro_match_eval import matcher_classification_loss as vectorized_matcher_classification_loss
from sam_backbone.losses import mask_outer_weight_for_epoch, prompt_pred_ratio, sam_prompt_mask_losses
from utils.eval_metrics_utils import (
    _accumulate_link_metrics,
    _centers_from_confidence_target,
    _connected_components_bool,
    _ellipse_mask_np,
    _evaluate_link_components,
    _finalize_detection_totals,
    _init_candidate_stats_bucket,
    _init_detection_totals,
    _max_iou_with_labeled_mask,
    _merge_detection_totals,
    _min_distance_to_points,
    _update_candidate_stats_bucket,
)
from utils.train_ops_utils import (
    binary_segmentation_logits as _binary_segmentation_logits,
    cellect_confidence_smooth_2d as _cellect_confidence_smooth_2d,
    cellect_foreground_gate_2d as _cellect_foreground_gate_2d,
    confidence_detection_score as _confidence_detection_score,
    count_points_in_masks as _count_points_in_masks,
    filter_points_by_mask_with_count as _filter_points_by_mask_with_count,
    flatten_band_centers as _flatten_band_centers,
    flatten_band_masks as _flatten_band_masks,
    flatten_per_band_outputs as _flatten_per_band_outputs,
    nearest_candidate_indices as _nearest_candidate_indices,
    refine_peak_coordinates as _refine_peak_coordinates,
    sample_embeddings_at_centers as _sample_embeddings_at_centers,
    sample_map_at_centers as _sample_map_at_centers,
    shape_regression_loss_map,
)


def _autocast_context(device: torch.device, amp_dtype: Optional[torch.dtype]):
    if amp_dtype is not None and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the original model under DDP or the local DDP helper wrapper."""

    current = model
    while True:
        if hasattr(current, "module"):
            current = current.module  # type: ignore[assignment]
            continue
        if hasattr(current, "_ddp_wrapped_model"):
            current = current._ddp_wrapped_model  # type: ignore[attr-defined,assignment]
            continue
        return current


@dataclass(frozen=True)
class LossWeights:
    """CELLECT constants carried into the 2D astronomy training loop."""

    segmentation_binary: Tuple[float, float] = (1.0, 32.0)
    segmentation_outer_weight: float = 1.0
    segmentation_loss_stride: int = 1
    confidence_outer_weight: float = 1.0
    confidence_pos_weight: float = 32.0
    shape_outer_weight: float = 1.0
    center_position: float = 1.0
    shape_angle_weight: float = 4.0
    triplet_margin: float = 0.3
    triplet_outer_weight: float = 10.0
    matcher_outer_weight: float = 10.0
    mask_outer_weight: float = 0.0
    mask_loss_warmup_epochs: int = 0
    mask_dice: float = 0.0
    mask_bce: float = 0.0
    mask_centroid: float = 0.2
    mask_outside: float = 0.5
    mask_min_area: float = 0.1
    mask_max_area: float = 0.1
    mask_pred_iou: float = 0.1
    mask_stability: float = 0.1
    mask_unmatched_prompt: float = 0.2
    center_only_shape_factor: float = 0.2
    mask_min_area_px: float = 15.0
    mask_area_ratio_lower: float = 0.15
    mask_area_ratio_upper: float = 1.05
    mask_max_area_ratio: float = 0.5
    mask_pred_iou_thresh: float = 0.8
    mask_stability_score_thresh: float = 0.95
    mask_stability_score_offset: float = 1.0
    mask_stability_temperature: float = 10.0
    mask_prompt_gt_epochs: int = 5
    mask_prompt_pred_epoch: int = 30
    mask_max_gt_per_sample: int = 0
    mask_max_pred_per_sample: int = 0
    mask_prompt_chunk_size: int = 128
    mask_multimask: bool = True
    mask_selection: str = "pred_iou"
    detach_mask_prompt_shapes: bool = False
    mask_prompt_center_only: bool = False


def active_mask_loss_keys(weights: LossWeights) -> Tuple[str, ...]:
    """Mask loss component keys with positive weights."""

    pairs = (
        ("dice", float(weights.mask_dice)),
        ("bce", float(weights.mask_bce)),
        ("centroid", float(weights.mask_centroid)),
        ("outside", float(weights.mask_outside)),
        ("area", float(weights.mask_min_area)),
        ("max_area", float(weights.mask_max_area)),
        ("pred_iou", float(weights.mask_pred_iou)),
        ("stability", float(weights.mask_stability)),
    )
    return tuple(key for key, weight in pairs if weight > 0.0)


def _sam_proposal_modules(model: nn.Module) -> List[nn.Module]:
    modules: List[nn.Module] = []
    decoder = getattr(model, "decoder", None)
    if isinstance(decoder, nn.Module):
        modules.append(decoder)
    for name in ("confidence_head", "shape_refine", "shape_head", "cen", "CEN"):
        module = getattr(model, name, None)
        if isinstance(module, nn.Module):
            modules.append(module)
    return modules


def _clear_module_grads(module: nn.Module) -> int:
    cleared = 0
    seen: set[int] = set()
    for param in module.parameters(recurse=True):
        ident = id(param)
        if ident in seen:
            continue
        seen.add(ident)
        if param.grad is not None:
            param.grad = None
            cleared += 1
    return cleared


def clear_sam_proposal_grads(model: nn.Module) -> int:
    """Remove gradients from frozen SAM proposal decoder before optimizer.step()."""

    cleared = 0
    for module in _sam_proposal_modules(unwrap_model(model)):
        cleared += _clear_module_grads(module)
    return cleared


def clear_sam_encoder_grads(model: nn.Module) -> int:
    """Remove gradients from a frozen SAM encoder before optimizer.step()."""

    encoder = getattr(unwrap_model(model), "encoder", None)
    if not isinstance(encoder, nn.Module):
        return 0
    return _clear_module_grads(encoder)


def clear_named_optimizer_group_grads(optimizer: torch.optim.Optimizer, group_name: str) -> int:
    """Remove gradients from all parameters in a named optimizer group."""

    cleared = 0
    seen: set[int] = set()
    for group in optimizer.param_groups:
        if str(group.get("name", "")) != str(group_name):
            continue
        for param in group.get("params", []):
            if not isinstance(param, torch.Tensor):
                continue
            ident = id(param)
            if ident in seen:
                continue
            seen.add(ident)
            if param.grad is not None:
                param.grad = None
                cleared += 1
    return cleared


class HardTripletLoss(nn.Module):
    """CELLECT/Open-ReID style hard triplet loss with margin 0.3.

    Positives are repeated object IDs, usually the same source appearing in
    overlapping tiles. Negatives can be constrained with group labels; for the
    HSC single-band setup this keeps negatives inside the same 512x512 image.
    """

    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, features: Tensor, labels: Tensor, groups: Optional[Tensor] = None) -> Tensor:
        labels = labels.long()
        valid = labels > 0
        if groups is not None:
            groups = groups.long()
            groups = groups[valid]
        features = features[valid]
        labels = labels[valid]
        if features.shape[0] < 4 or torch.unique(labels).numel() < 2:
            return features.sum() * 0.0
        dist = torch.cdist(features, features, p=2)
        same = labels[:, None] == labels[None, :]
        eye = torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
        pos_mask = same & ~eye
        neg_mask = ~same
        if groups is not None:
            same_group = groups[:, None] == groups[None, :]
            neg_mask = neg_mask & same_group
        has_pos = pos_mask.any(dim=1)
        has_neg = neg_mask.any(dim=1)
        keep = has_pos & has_neg
        if not bool(keep.any()):
            return features.sum() * 0.0
        dist_ap = dist.masked_fill(~pos_mask, -1.0).max(dim=1).values[keep]
        dist_an = dist.masked_fill(~neg_mask, float("inf")).min(dim=1).values[keep]
        target = torch.ones_like(dist_an)
        return self.margin_loss(dist_an, dist_ap, target)


def dense_losses(
    outputs: Dict[str, Tensor],
    batch: Dict[str, object],
    *,
    weights: LossWeights,
    device: torch.device,
    center_radius_px: float,
) -> Dict[str, Tensor]:
    # Debug
    # torch.cuda.synchronize()
    # start_time = time.time()
    conf_target = batch["confidence"].to(device=device, dtype=torch.long)  # type: ignore[union-attr]
    conf_weight = batch.get("confidence_weight")
    if conf_weight is not None:
        # confidence_weight is the single source of truth: clean apertures keep
        # full weight, bright/strict-center-only sources can contribute with a
        # low weight, and ordinary ignore/background remain unconstrained.
        conf_weight = conf_weight.to(device=device, dtype=torch.float32)  # type: ignore[union-attr]
    else:
        clean_mask = batch.get("clean_mask")
        conf_weight = (
            clean_mask.to(device=device, dtype=torch.float32)  # type: ignore[union-attr]
            if clean_mask is not None
            else None
        )

    if float(weights.segmentation_outer_weight) <= 0.0 or "seg_logits" not in outputs:
        seg_loss = outputs["confidence"].sum() * 0.0
    else:
        seg_target = (batch["seg"].to(device=device, dtype=torch.long) > 0).long()  # type: ignore[union-attr]
        seg_loss_weight = batch.get("seg_loss_weight")
        if seg_loss_weight is not None:
            seg_loss_weight = seg_loss_weight.to(device=device, dtype=torch.float32)  # type: ignore[union-attr]
        seg_logits = _binary_segmentation_logits(outputs["seg_logits"])
        seg_stride = max(1, int(weights.segmentation_loss_stride))
        if seg_stride > 1:
            seg_logits = F.avg_pool2d(seg_logits, kernel_size=seg_stride, stride=seg_stride)
            seg_target = F.max_pool2d(
                seg_target.unsqueeze(1).to(dtype=torch.float32),
                kernel_size=seg_stride,
                stride=seg_stride,
            ).squeeze(1).to(dtype=torch.long)
            if seg_loss_weight is not None:
                seg_loss_weight = F.max_pool2d(
                    seg_loss_weight.unsqueeze(1),
                    kernel_size=seg_stride,
                    stride=seg_stride,
                ).squeeze(1)
        seg_weight = torch.tensor(weights.segmentation_binary[:2], device=device, dtype=torch.float32)
        per_pixel_seg = F.cross_entropy(seg_logits, seg_target, weight=seg_weight, reduction="none")
        if seg_loss_weight is not None:
            seg_loss_weight = seg_loss_weight.clamp_min(0.0)
            if bool((seg_loss_weight > 0).any()):
                seg_loss = (per_pixel_seg * seg_loss_weight).sum() / seg_loss_weight.sum().clamp_min(1.0)
            else:
                seg_loss = per_pixel_seg.sum() * 0.0
        else:
            seg_loss = per_pixel_seg.mean()
    # torch.cuda.synchronize()
    # seg_time = time.time()
    # print(f'[DEBUG] Segmentation loss computed in {seg_time - start_time:.3f} seconds.')
    if float(weights.confidence_outer_weight) <= 0.0:
        conf_loss = outputs["confidence"].sum() * 0.0
    else:
        conf_loss = ordinal_confidence_loss(
            outputs["confidence"],
            conf_target,
            pos_weight=weights.confidence_pos_weight,
            weight=conf_weight,
        )
    # torch.cuda.synchronize()
    # conf_time = time.time()
    # print(f'[DEBUG] Confidence loss computed in {conf_time - seg_time:.3f} seconds.')
    if float(weights.shape_outer_weight) <= 0.0:
        shape_loss = outputs["shape"].sum() * 0.0
    else:
        shape_target = batch["shape"].to(device=device, dtype=torch.float32)  # type: ignore[union-attr]
        shape_weight = batch["shape_weight"].to(device=device, dtype=torch.float32)  # type: ignore[union-attr]
        pseudo_mask = batch.get("pseudo_mask")
        pseudo_bool = (
            pseudo_mask.to(device=device, dtype=torch.bool)  # type: ignore[union-attr]
            if pseudo_mask is not None
            else None
        )
        ignore_mask = batch.get("ignore_mask")
        if ignore_mask is not None:
            ignore_bool = ignore_mask.to(device=device, dtype=torch.bool)  # type: ignore[union-attr]
            if pseudo_bool is not None:
                ignore_bool = ignore_bool & ~pseudo_bool
            shape_weight = shape_weight * (~ignore_bool).to(dtype=torch.float32)
        center_only_mask = batch.get("center_only_mask")
        if center_only_mask is not None:
            center_only_bool = center_only_mask.to(device=device, dtype=torch.bool)  # type: ignore[union-attr]
            if pseudo_bool is not None:
                center_only_bool = center_only_bool & ~pseudo_bool
            center_only_factor = shape_weight.new_tensor(float(weights.center_only_shape_factor)).clamp(0.0, 1.0)
            shape_weight = torch.where(center_only_bool, shape_weight * center_only_factor, shape_weight)
        per_pixel_shape = shape_regression_loss_map(
            outputs["shape"],
            shape_target,
            angle_weight=weights.shape_angle_weight,
        )
        if bool((shape_weight > 0).any()):
            shape_loss = (per_pixel_shape * shape_weight).sum() / shape_weight.sum().clamp_min(1.0)
        else:
            shape_loss = per_pixel_shape.mean() * 0.0
    # torch.cuda.synchronize()
    # shape_time = time.time()
    # print(f'[DEBUG] Shape loss computed in {shape_time - conf_time:.3f} seconds.')
    if weights.center_position > 0.0:
        center_loss = center_localization_loss(outputs, batch["centers"], radius_px=center_radius_px)
    else:
        center_loss = torch.tensor(0.0, device=device)
    # torch.cuda.synchronize()
    # center_time = time.time()
    # print(f'[DEBUG] Center localization loss computed in {center_time - shape_time:.3f} seconds.')
    total = (
        weights.segmentation_outer_weight * seg_loss
        + weights.confidence_outer_weight * conf_loss
        + weights.shape_outer_weight * shape_loss
        + weights.center_position * center_loss
    )
    return {"total": total, "seg": seg_loss, "confidence": conf_loss, "shape": shape_loss, "center": center_loss}


def dense_losses_any(
    outputs: Dict[str, Tensor],
    batch: Dict[str, object],
    *,
    weights: LossWeights,
    device: torch.device,
    center_radius_px: float,
) -> Dict[str, Tensor]:
    """Dense loss for both fused BCHW outputs and per-band B,C,CHW outputs."""

    per_band = outputs["confidence"].ndim == 5
    if not per_band:
        return dense_losses(outputs, batch, weights=weights, device=device, center_radius_px=center_radius_px)

    flat_outputs = _flatten_per_band_outputs(outputs)
    flat_batch = {
        "confidence": batch["band_confidence"].reshape(-1, *batch["band_confidence"].shape[2:]),  # type: ignore[union-attr]
        "shape": batch["band_shape"].reshape(-1, *batch["band_shape"].shape[2:]),  # type: ignore[union-attr]
        "shape_weight": batch["band_shape_weight"].reshape(-1, *batch["band_shape_weight"].shape[2:]),  # type: ignore[union-attr]
        "clean_mask": batch["band_clean_mask"].reshape(-1, *batch["band_clean_mask"].shape[2:]),  # type: ignore[union-attr]
        "confidence_weight": batch["band_confidence_weight"].reshape(
            -1, *batch["band_confidence_weight"].shape[2:]
        ),  # type: ignore[union-attr]
        "ignore_mask": batch["band_ignore_mask"].reshape(-1, *batch["band_ignore_mask"].shape[2:]),  # type: ignore[union-attr]
        "center_only_mask": batch["band_center_only_mask"].reshape(
            -1, *batch["band_center_only_mask"].shape[2:]
        ),  # type: ignore[union-attr]
        "pseudo_mask": batch["band_pseudo_mask"].reshape(-1, *batch["band_pseudo_mask"].shape[2:]),  # type: ignore[union-attr]
        "centers": _flatten_band_centers(batch["band_centers"]),  # type: ignore[arg-type]
    }
    if "seg_logits" in outputs:
        flat_batch["seg"] = batch["band_seg"].reshape(-1, *batch["band_seg"].shape[2:])  # type: ignore[union-attr]
        flat_batch["seg_loss_weight"] = batch["band_seg_loss_weight"].reshape(
            -1, *batch["band_seg_loss_weight"].shape[2:]
        )  # type: ignore[union-attr]
    return dense_losses(flat_outputs, flat_batch, weights=weights, device=device, center_radius_px=center_radius_px)


def center_localization_loss(outputs: Dict[str, Tensor], centers_list: Sequence[Tensor], *, radius_px: float) -> Tensor:
    """Differentiable center-position loss on the top confidence channel.

    The astronomy data has one 2D catalog center shared by all input bands. This
    differs from CELLECT microscopy, where the center is a point in 3D space. For
    each catalog center we take a local window on the highest confidence channel,
    compute a soft-argmax position, and penalize the offset normalized by the
    astrometric matching tolerance, by default 0.5 arcsec / 0.168 arcsec per px.
    """

    conf = outputs["confidence"][:, -1]
    h, w = conf.shape[-2:]
    radius = max(float(radius_px), 1e-6)
    window = max(1, int(math.ceil(radius)))
    losses: List[Tensor] = []
    for b, centers in enumerate(centers_list):
        if centers.numel() == 0:
            continue
        centers_f = centers.to(device=conf.device, dtype=conf.dtype)
        for center in centers_f:
            cx, cy = center[0], center[1]
            xi = int(round(float(cx.detach().cpu())))
            yi = int(round(float(cy.detach().cpu())))
            x0, x1 = max(0, xi - window), min(w, xi + window + 1)
            y0, y1 = max(0, yi - window), min(h, yi + window + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            patch = conf[b, y0:y1, x0:x1]
            prob = F.softmax(patch.reshape(-1), dim=0)
            xs = torch.arange(x0, x1, device=conf.device, dtype=conf.dtype)
            ys = torch.arange(y0, y1, device=conf.device, dtype=conf.dtype)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            pred_x = (prob * xx.reshape(-1)).sum()
            pred_y = (prob * yy.reshape(-1)).sum()
            pred_xy = torch.stack((pred_x, pred_y)) / radius
            target_xy = torch.stack((cx, cy)) / radius
            losses.append(F.smooth_l1_loss(pred_xy, target_xy, reduction="sum"))
    if not losses:
        return conf.sum() * 0.0
    return torch.stack(losses).mean()


def _confidence_score_at_centers(outputs: Dict[str, Tensor], centers: Tensor, *, batch_index: int = 0) -> Tensor:
    """Sample CELLECT-smoothed top confidence channel at detected centers."""

    if centers.numel() == 0:
        return outputs["confidence"].new_zeros((0,))
    confidence = outputs["confidence"]
    if confidence.ndim == 5:
        raise ValueError("_confidence_score_at_centers expects one dense band output")
    score_map = _cellect_confidence_smooth_2d(confidence)[batch_index, -1]
    h, w = score_map.shape[-2:]
    xy = torch.round(centers.to(device=score_map.device, dtype=torch.float32)).to(dtype=torch.long)
    x = xy[:, 0].clamp(0, w - 1)
    y = xy[:, 1].clamp(0, h - 1)
    return score_map[y, x]


ORDINAL_EXPECTATION_THRESHOLD = 2.0
ORDINAL_EXPECTATION_MERGE_RADIUS = 3.0


def _merge_close_centers_by_score(
    coords: Tensor,
    scores: Tensor,
    *,
    min_distance: float,
) -> Tuple[Tensor, Tensor]:
    if coords.numel() == 0 or scores.numel() == 0 or float(min_distance) <= 0.0:
        return coords, scores
    if coords.shape[0] != scores.shape[0]:
        raise ValueError("coords and scores must have matching lengths")

    dist2_thresh = float(min_distance) * float(min_distance)
    delta = coords[:, None, :2] - coords[None, :, :2]
    dist2 = (delta * delta).sum(dim=2)
    adjacency = dist2 < dist2_thresh

    visited = torch.zeros(coords.shape[0], dtype=torch.bool, device=coords.device)
    keep_indices: List[int] = []
    for start_idx in range(coords.shape[0]):
        if bool(visited[start_idx]):
            continue
        stack = [int(start_idx)]
        component: List[int] = []
        visited[start_idx] = True
        while stack:
            current = stack.pop()
            component.append(current)
            neighbors = torch.nonzero(adjacency[current] & ~visited, as_tuple=False).flatten()
            for neighbor_idx in neighbors.tolist():
                visited[neighbor_idx] = True
                stack.append(int(neighbor_idx))
        if len(component) == 1:
            keep_indices.append(component[0])
            continue
        comp_idx = torch.as_tensor(component, device=scores.device, dtype=torch.long)
        comp_scores = scores[comp_idx]
        best_local = int(torch.argmax(comp_scores).item())
        keep_indices.append(int(comp_idx[best_local].item()))

    keep = torch.as_tensor(sorted(keep_indices), device=coords.device, dtype=torch.long)
    return coords[keep], scores[keep]


def _compute_detection_peak_maps(
    outputs: Dict[str, Tensor],
    *,
    threshold: float,
    nms_radius: int,
    confidence_score: str,
    use_ordinal_expectation: bool = False,
    debug_ordinal_expectation: bool = False,
) -> Tuple[Tensor, Tensor, Dict[str, object]]:
    detection_score_mode = "ordinal_expectation" if use_ordinal_expectation else str(confidence_score)
    effective_threshold = float(ORDINAL_EXPECTATION_THRESHOLD if use_ordinal_expectation else threshold)
    ordinal_expectation_score: Optional[Tensor] = None
    if bool(use_ordinal_expectation) or bool(debug_ordinal_expectation) or detection_score_mode == "ordinal_expectation":
        ordinal_expectation_score = _confidence_detection_score(outputs, "ordinal_expectation")

    if detection_score_mode == "cellect":
        smoothed = _cellect_confidence_smooth_2d(outputs["confidence"])
        argmax_channel = smoothed.argmax(dim=1)
        local_score = smoothed.max(dim=1).values
        center_score = smoothed[:, -1]  # scores of layer 4
        pooled = F.max_pool2d(
            local_score.unsqueeze(1),
            kernel_size=2 * nms_radius + 1,
            stride=1,
            padding=nms_radius,
        ).squeeze(1)
        center_pooled = F.max_pool2d(
            center_score.unsqueeze(1),
            kernel_size=2 * nms_radius + 1,
            stride=1,
            padding=nms_radius,
        ).squeeze(1)
        foreground_gate = (
            _cellect_foreground_gate_2d(outputs["seg_logits"])
            if "seg_logits" in outputs
            else torch.ones_like(center_score, dtype=torch.bool)
        )
        top_channel_argmax = center_score == local_score
        center_pooled_candidates = center_pooled == local_score
        spatial_localmax = pooled == local_score
        seed_candidates = pooled == center_score
        threshold_pass = center_score > effective_threshold
        peaks = seed_candidates & foreground_gate & threshold_pass
        debug: Dict[str, object] = {
            "center_score": center_score,
            "argmax_channel": argmax_channel,
            "top_channel_argmax": top_channel_argmax,
            "center_pooled_candidates": center_pooled_candidates,
            "spatial_localmax": spatial_localmax,
            "seed_candidates": seed_candidates,
            "foreground_gate": foreground_gate,
            "threshold_pass": threshold_pass,
            "final_peaks": peaks,
            "foreground_gate_active": bool("seg_logits" in outputs),
            "detection_score_mode": detection_score_mode,
            "effective_threshold": effective_threshold,
            "ordinal_expectation_threshold": float(ORDINAL_EXPECTATION_THRESHOLD),
            "use_ordinal_expectation": bool(use_ordinal_expectation),
            "debug_ordinal_expectation": bool(debug_ordinal_expectation),
        }
        if ordinal_expectation_score is not None and bool(debug_ordinal_expectation):
            ordinal_expectation_pooled = F.max_pool2d(
                ordinal_expectation_score.unsqueeze(1),
                kernel_size=2 * nms_radius + 1,
                stride=1,
                padding=nms_radius,
            ).squeeze(1)
            ordinal_expectation_seed_candidates = ordinal_expectation_score == ordinal_expectation_pooled
            ordinal_expectation_threshold_pass = ordinal_expectation_score > float(ORDINAL_EXPECTATION_THRESHOLD)
            debug.update(
                {
                    "ordinal_expectation_score": ordinal_expectation_score,
                    "ordinal_expectation_seed_candidates": ordinal_expectation_seed_candidates,
                    "ordinal_expectation_threshold_pass": ordinal_expectation_threshold_pass,
                    "ordinal_expectation_final_peaks": (
                        ordinal_expectation_seed_candidates & foreground_gate & ordinal_expectation_threshold_pass
                    ),
                }
            )
        return center_score, peaks, debug

    center_score = (
        ordinal_expectation_score
        if detection_score_mode == "ordinal_expectation" and ordinal_expectation_score is not None
        else _confidence_detection_score(outputs, detection_score_mode)
    )
    argmax_channel = outputs["confidence"].argmax(dim=1)
    foreground_gate = (
        outputs["seg_logits"].argmax(dim=1) > 0
        if "seg_logits" in outputs
        else torch.ones_like(center_score, dtype=torch.bool)
    )
    pooled = F.max_pool2d(
        center_score.unsqueeze(1),
        kernel_size=2 * nms_radius + 1,
        stride=1,
        padding=nms_radius,
    ).squeeze(1)
    seed_candidates = center_score == pooled
    threshold_pass = center_score > effective_threshold
    peaks = seed_candidates & foreground_gate & threshold_pass
    debug = {
        "center_score": center_score,
        "argmax_channel": argmax_channel,
        "top_channel_argmax": None,
        "spatial_localmax": seed_candidates,
        "center_pooled_candidates": None,
        "seed_candidates": seed_candidates,
        "foreground_gate": foreground_gate,
        "threshold_pass": threshold_pass,
        "final_peaks": peaks,
        "foreground_gate_active": bool("seg_logits" in outputs),
        "detection_score_mode": detection_score_mode,
        "effective_threshold": effective_threshold,
        "ordinal_expectation_threshold": float(ORDINAL_EXPECTATION_THRESHOLD),
        "use_ordinal_expectation": bool(use_ordinal_expectation),
        "debug_ordinal_expectation": bool(debug_ordinal_expectation),
    }
    if ordinal_expectation_score is not None and bool(debug_ordinal_expectation):
        ordinal_expectation_pooled = F.max_pool2d(
            ordinal_expectation_score.unsqueeze(1),
            kernel_size=2 * nms_radius + 1,
            stride=1,
            padding=nms_radius,
        ).squeeze(1)
        ordinal_expectation_seed_candidates = ordinal_expectation_score == ordinal_expectation_pooled
        ordinal_expectation_threshold_pass = ordinal_expectation_score > float(ORDINAL_EXPECTATION_THRESHOLD)
        debug.update(
            {
                "ordinal_expectation_score": ordinal_expectation_score,
                "ordinal_expectation_seed_candidates": ordinal_expectation_seed_candidates,
                "ordinal_expectation_threshold_pass": ordinal_expectation_threshold_pass,
                "ordinal_expectation_final_peaks": (
                    ordinal_expectation_seed_candidates & foreground_gate & ordinal_expectation_threshold_pass
                ),
            }
        )
    return center_score, peaks, debug


@torch.no_grad()
def en_deduplicate_centers(
    en_net: AstroMatchNet2D,
    outputs: Dict[str, Tensor],
    centers_xy: np.ndarray,
    *,
    batch_index: int = 0,
    candidate_count: int = 5,
    offset_scale: float = 1.0,
    same_threshold: float = 0.6,
    strong_threshold: float = 0.9999,
    use_group_mean: bool = True,
) -> np.ndarray:
    """CELLECT-style intra-frame EN post-processing for detected centers.

    CELLECT does not stop at raw confidence local maxima.  During inference it
    samples features at detected centers, finds five nearest same-frame
    neighbors, applies EN, and groups centers when EN says they are the same
    object.  This helper implements the same idea for one 2D band.
    """

    centers = torch.as_tensor(np.asarray(centers_xy, dtype=np.float32), device=outputs["embedding"].device)
    if centers.ndim != 2 or centers.shape[0] <= 1:
        return np.asarray(centers_xy, dtype=np.float32)
    n = centers.shape[0]
    k = min(int(candidate_count), max(n - 1, 1))

    emb_map = outputs["embedding"][batch_index]
    shape_map = outputs["shape"][batch_index]
    features = _sample_map_at_centers(emb_map, centers)
    shapes = _sample_map_at_centers(shape_map, centers)
    center_scores = _confidence_score_at_centers(outputs, centers, batch_index=batch_index)

    dist = torch.cdist(centers, centers, p=2)
    dist.fill_diagonal_(float("inf"))
    nn_dist, nn_idx = torch.topk(dist, k=k, largest=False)
    if k < int(candidate_count):
        pad = int(candidate_count) - k
        nn_idx = torch.cat([nn_idx, nn_idx[:, -1:].repeat(1, pad)], dim=1)
        nn_dist = torch.cat([nn_dist, nn_dist[:, -1:].repeat(1, pad)], dim=1)

    cand_features = features[nn_idx]
    cand_xy = centers[nn_idx]
    offsets = (cand_xy - centers[:, None, :]) / max(float(offset_scale), 1e-6)
    cand_shapes = shapes[nn_idx]
    shape_features = torch.cat([shapes[:, None, :].expand(-1, int(candidate_count), -1), cand_shapes], dim=-1)
    logits = en_net(features, cand_features, offsets, shape_features)
    prob = F.softmax(logits, dim=1)
    cand_prob = prob[:, :-1]
    none_prob = prob[:, -1:]

    suppressed: set[int] = set()
    groups: List[List[int]] = []
    for i in range(n):
        if i in suppressed:
            continue
        group = [i]
        major = float(shapes[i, 0].detach().cpu()) if shapes.shape[1] > 0 else 1.0
        local_radius = max(5.0, 0.6 * max(abs(major), 1.0))
        for j in range(int(candidate_count)):
            candidate = int(nn_idx[i, j].item())
            if candidate == i or candidate in suppressed:
                continue
            same = bool(cand_prob[i, j] > none_prob[i, 0])
            confident = bool(cand_prob[i, j] > strong_threshold)
            moderate_close = bool(cand_prob[i, j] > same_threshold and nn_dist[i, min(j, nn_dist.shape[1] - 1)] < local_radius)
            if same and (confident or moderate_close):
                group.append(candidate)
                suppressed.add(candidate)
        groups.append(group)

    dedup: List[np.ndarray] = []
    centers_cpu = centers.detach().float().cpu().numpy().astype(np.float32)
    scores_cpu = center_scores.detach().float().cpu().numpy()
    for group in groups:
        if use_group_mean and len(group) > 1:
            dedup.append(centers_cpu[group].mean(axis=0))
        else:
            best = max(group, key=lambda idx: float(scores_cpu[idx]))
            dedup.append(centers_cpu[best])
    return np.stack(dedup, axis=0).astype(np.float32) if dedup else np.zeros((0, 2), dtype=np.float32)


@torch.no_grad()
def detect_centers_with_en(
    model: nn.Module,
    outputs: Dict[str, Tensor],
    *,
    threshold: float,
    nms_radius: int,
    confidence_score: str,
    use_ordinal_expectation: bool = False,
    debug_ordinal_expectation: bool = False,
    match_radius: float = 3.0,
    candidate_count: int = 5,
    en_threshold: float = 0.6,
    center_refinement: str = "integer",
    center_refinement_radius: int = 1,
) -> List[np.ndarray]:
    """Detect centers and optionally apply CELLECT-style EN deduplication."""

    model = unwrap_model(model)
    if outputs["seg_logits"].ndim == 5:
        batch, bands = outputs["seg_logits"].shape[:2]
        flat_outputs = _flatten_per_band_outputs(outputs)
        raw = detect_centers(
            flat_outputs,
            threshold=threshold,
            nms_radius=nms_radius,
            confidence_score=confidence_score,
            use_ordinal_expectation=use_ordinal_expectation,
            debug_ordinal_expectation=debug_ordinal_expectation,
            center_refinement=center_refinement,
            center_refinement_radius=center_refinement_radius,
        )
        if not hasattr(model, "EN"):
            return raw
        result: List[np.ndarray] = []
        for b in range(batch):
            for band in range(bands):
                idx = b * bands + band
                one = {key: value[b, band].unsqueeze(0) for key, value in outputs.items()}
                result.append(
                    en_deduplicate_centers(
                        model.EN,
                        one,
                        raw[idx],
                        batch_index=0,
                        candidate_count=candidate_count,
                        offset_scale=match_radius,
                        same_threshold=en_threshold,
                    )
                )
        return result

    raw = detect_centers(
        outputs,
        threshold=threshold,
        nms_radius=nms_radius,
        confidence_score=confidence_score,
        use_ordinal_expectation=use_ordinal_expectation,
        debug_ordinal_expectation=debug_ordinal_expectation,
        center_refinement=center_refinement,
        center_refinement_radius=center_refinement_radius,
    )
    if not hasattr(model, "EN"):
        return raw
    return [
        en_deduplicate_centers(
            model.EN,
            outputs,
            centers,
            batch_index=batch_idx,
            candidate_count=candidate_count,
            offset_scale=match_radius,
            same_threshold=en_threshold,
        )
        for batch_idx, centers in enumerate(raw)
    ]


@torch.no_grad()
def detect_centers_with_ex_link(
    model: nn.Module,
    outputs: Dict[str, Tensor],
    *,
    threshold: float,
    nms_radius: int,
    confidence_score: str,
    use_ordinal_expectation: bool = False,
    debug_ordinal_expectation: bool = False,
    match_radius: float = 3.0,
    candidate_count: int = 5,
    ex_threshold: float = 0.5,
    center_refinement: str = "integer",
    center_refinement_radius: int = 1,
    use_en_postprocess: bool = False,
    en_threshold: float = 0.6,
    max_distance_factor: float = 6.0,
    band_pairs: Optional[Sequence[Tuple[int, int]]] = None,
) -> Tuple[List[np.ndarray], List[List[Dict[str, object]]]]:
    """CELLECT-style cross-band linking after per-band center detection.

    CELLECT uses EX after center extraction: each anchor detection searches its
    five nearest detections in the next frame and the MLP scores candidate links
    using center embedding, position offset, and size/division features.  The
    astronomy variant has no division branch, so EX links detections across
    bands using embedding, position offset, and predicted shape/size.  The
    returned predictions are object-level fused centers, while ``components``
    keeps the per-band members for diagnostics and catalog construction.
    """

    model = unwrap_model(model)
    if outputs["seg_logits"].ndim != 5 or not hasattr(model, "EX"):
        if outputs["seg_logits"].ndim == 5:
            flat_outputs = _flatten_per_band_outputs(outputs)
            raw = detect_centers(
                flat_outputs,
                threshold=threshold,
                nms_radius=nms_radius,
                confidence_score=confidence_score,
                use_ordinal_expectation=use_ordinal_expectation,
                debug_ordinal_expectation=debug_ordinal_expectation,
                center_refinement=center_refinement,
                center_refinement_radius=center_refinement_radius,
            )
            return raw, []
        if use_en_postprocess and hasattr(model, "EN"):
            return (
                detect_centers_with_en(
                    model,
                    outputs,
                    threshold=threshold,
                    nms_radius=nms_radius,
                    confidence_score=confidence_score,
                    use_ordinal_expectation=use_ordinal_expectation,
                    debug_ordinal_expectation=debug_ordinal_expectation,
                    center_refinement=center_refinement,
                    center_refinement_radius=center_refinement_radius,
                    match_radius=match_radius,
                    candidate_count=candidate_count,
                    en_threshold=en_threshold,
                ),
                [],
            )
        return (
            detect_centers(
                outputs,
                threshold=threshold,
                nms_radius=nms_radius,
                confidence_score=confidence_score,
                use_ordinal_expectation=use_ordinal_expectation,
                debug_ordinal_expectation=debug_ordinal_expectation,
                center_refinement=center_refinement,
                center_refinement_radius=center_refinement_radius,
            ),
            [],
        )

    batch, bands = outputs["seg_logits"].shape[:2]
    flat_outputs = _flatten_per_band_outputs(outputs)
    raw = detect_centers(
        flat_outputs,
        threshold=threshold,
        nms_radius=nms_radius,
        confidence_score=confidence_score,
        use_ordinal_expectation=use_ordinal_expectation,
        debug_ordinal_expectation=debug_ordinal_expectation,
        center_refinement=center_refinement,
        center_refinement_radius=center_refinement_radius,
    )
    per_batch: List[List[np.ndarray]] = []
    for batch_idx in range(batch):
        item: List[np.ndarray] = []
        for band_idx in range(bands):
            flat_idx = batch_idx * bands + band_idx
            centers = raw[flat_idx]
            if use_en_postprocess and hasattr(model, "EN"):
                one = {key: value[batch_idx, band_idx].unsqueeze(0) for key, value in outputs.items()}
                centers = en_deduplicate_centers(
                    model.EN,
                    one,
                    centers,
                    batch_index=0,
                    candidate_count=candidate_count,
                    offset_scale=match_radius,
                    same_threshold=en_threshold,
                )
            item.append(np.asarray(centers, dtype=np.float32))
        per_batch.append(item)

    merged: List[np.ndarray] = []
    components_all: List[List[Dict[str, object]]] = []
    for batch_idx, centers_by_band in enumerate(per_batch):
        centers, components = _link_one_multiband_item(
            model.EX,
            outputs,
            centers_by_band,
            batch_index=batch_idx,
            candidate_count=candidate_count,
            offset_scale=match_radius,
            ex_threshold=ex_threshold,
            max_distance_factor=max_distance_factor,
            band_pairs=band_pairs,
        )
        merged.append(centers)
        components_all.append(components)
    return merged, components_all


@torch.no_grad()
def _link_one_multiband_item(
    ex_net: AstroMatchNet2D,
    outputs: Dict[str, Tensor],
    centers_by_band: Sequence[np.ndarray],
    *,
    batch_index: int,
    candidate_count: int,
    offset_scale: float,
    ex_threshold: float,
    max_distance_factor: float,
    band_pairs: Optional[Sequence[Tuple[int, int]]] = None,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    device = outputs["embedding"].device
    bands = len(centers_by_band)
    node_band: List[int] = []
    node_index: List[int] = []
    node_xy: List[Tensor] = []
    node_feature: List[Tensor] = []
    node_shape: List[Tensor] = []
    node_score: List[Tensor] = []

    for band_idx, centers_np in enumerate(centers_by_band):
        centers = torch.as_tensor(np.asarray(centers_np, dtype=np.float32), device=device)
        if centers.ndim != 2 or centers.shape[0] == 0:
            continue
        emb_map = outputs["embedding"][batch_index, band_idx]
        shape_map = outputs["shape"][batch_index, band_idx]
        one = {key: value[batch_index, band_idx].unsqueeze(0) for key, value in outputs.items()}
        features = _sample_map_at_centers(emb_map, centers)
        shapes = _sample_map_at_centers(shape_map, centers)
        scores = _confidence_score_at_centers(one, centers, batch_index=0)
        for det_idx in range(centers.shape[0]):
            node_band.append(band_idx)
            node_index.append(det_idx)
            node_xy.append(centers[det_idx])
            node_feature.append(features[det_idx])
            node_shape.append(shapes[det_idx])
            node_score.append(scores[det_idx])

    if not node_xy:
        return np.zeros((0, 2), dtype=np.float32), []
    if len(node_xy) == 1 or bands <= 1:
        center = node_xy[0].detach().cpu().numpy()[None].astype(np.float32)
        return center, [
            {
                "members": [(node_band[0], node_index[0])],
                "member_centers": [(node_band[0], node_xy[0].detach().cpu().numpy().astype(np.float32).tolist())],
                "score": float(node_score[0].detach().cpu()),
            }
        ]

    xy = torch.stack(node_xy)
    features = torch.stack(node_feature)
    shapes = torch.stack(node_shape)
    scores = torch.stack(node_score)
    node_band_t = torch.tensor(node_band, dtype=torch.long, device=device)
    k = int(candidate_count)
    edges: List[Tuple[float, float, int, int]] = []
    pair_map: Optional[Dict[int, List[int]]] = None
    if band_pairs is not None:
        pair_map = {}
        for src, dst in band_pairs:
            src_i = int(src)
            dst_i = int(dst)
            if src_i == dst_i:
                continue
            pair_map.setdefault(src_i, [])
            if dst_i not in pair_map[src_i]:
                pair_map[src_i].append(dst_i)

    for anchor in range(xy.shape[0]):
        anchor_band = int(node_band_t[anchor].item())
        if pair_map is None:
            dst_bands = [band for band in range(bands) if band != anchor_band]
        else:
            dst_bands = [band for band in pair_map.get(anchor_band, []) if 0 <= band < bands and band != anchor_band]
        for dst_band in dst_bands:
            other = node_band_t == int(dst_band)
            if not bool(other.any()):
                continue
            candidate_ids = torch.nonzero(other, as_tuple=False).flatten()
            dist = torch.linalg.norm(xy[candidate_ids] - xy[anchor][None, :], dim=1)
            order = torch.argsort(dist)[: min(k, candidate_ids.numel())]
            selected = candidate_ids[order]
            selected_dist = dist[order]
            if selected.numel() == 0:
                continue
            if selected.numel() < k:
                pad = k - int(selected.numel())
                selected = torch.cat([selected, selected[-1:].repeat(pad)], dim=0)
                selected_dist = torch.cat([selected_dist, selected_dist[-1:].repeat(pad)], dim=0)

            cand_features = features[selected].unsqueeze(0)
            cand_xy = xy[selected].unsqueeze(0)
            offsets = (cand_xy - xy[anchor].view(1, 1, 2)) / max(float(offset_scale), 1e-6)
            cand_shapes = shapes[selected].unsqueeze(0)
            shape_features = torch.cat([shapes[anchor].view(1, 1, -1).expand(1, k, -1), cand_shapes], dim=-1)
            ex_was_training = ex_net.training
            if ex_was_training:
                ex_net.eval()
            logits = ex_net(features[anchor].view(1, -1), cand_features, offsets, shape_features)
            if ex_was_training:
                ex_net.train()
            prob = F.softmax(logits, dim=1)[0]
            cand_prob = prob[:k]
            none_prob = prob[k]
            best = int(torch.argmax(cand_prob).item())
            best_node = int(selected[best].item())
            best_score = float(cand_prob[best].detach().cpu())
            best_dist = float(selected_dist[best].detach().cpu())
            major = max(float(abs(shapes[anchor, 0].detach().cpu())) if shapes.shape[1] > 0 else 1.0, 1.0)
            distance_limit = max(float(offset_scale), max_distance_factor * max(major, 1.0))
            if best_score > float(ex_threshold) and cand_prob[best] > none_prob and best_dist <= distance_limit:
                lo, hi = sorted((anchor, best_node))
                edges.append((best_score, best_dist, lo, hi))

    parent = list(range(len(node_xy)))
    component_bands: List[set[int]] = [{band} for band in node_band]

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for score, dist, a, b in sorted(edges, key=lambda item: (-item[0], item[1])):
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if component_bands[ra] & component_bands[rb]:
            continue
        parent[rb] = ra
        component_bands[ra] |= component_bands[rb]

    grouped: Dict[int, List[int]] = {}
    for idx in range(len(node_xy)):
        grouped.setdefault(find(idx), []).append(idx)

    components: List[Dict[str, object]] = []
    fused_centers: List[np.ndarray] = []
    for members in grouped.values():
        member_scores = scores[members].detach()
        weights = torch.clamp(member_scores, min=0)
        if not bool((weights > 0).any()):
            weights = torch.ones_like(weights)
        member_xy = xy[members]
        fused = (member_xy * weights[:, None]).sum(dim=0) / weights.sum().clamp_min(1e-6)
        fused_centers.append(fused.detach().cpu().numpy().astype(np.float32))
        components.append(
            {
                "members": [(int(node_band[i]), int(node_index[i])) for i in members],
                "member_centers": [
                    (int(node_band[i]), xy[i].detach().cpu().numpy().astype(np.float32).tolist()) for i in members
                ],
                "center": fused.detach().cpu().numpy().astype(np.float32).tolist(),
                "score": float(member_scores.max().detach().cpu()),
            }
        )

    order = np.argsort([item["score"] for item in components])[::-1] if components else []
    components = [components[int(i)] for i in order]
    fused_array = (
        np.stack([fused_centers[int(i)] for i in order], axis=0).astype(np.float32)
        if len(order) > 0
        else np.zeros((0, 2), dtype=np.float32)
    )
    return fused_array, components


def _sample_multiband_sources(
    outputs: Dict[str, Tensor],
    batch: Dict[str, object],
) -> List[List[Dict[str, Tensor]]]:
    """Sample per-source features for every item and band.

    Returns a nested list [batch][band] with ``xy``, ``ids``, ``features`` and
    ``shape`` tensors.  This is the training-time bridge from dense UNet maps to
    CELLECT-style EX/EN candidate classification.
    """

    emb = outputs["embedding"]
    shape = outputs["shape"]
    if emb.ndim != 5:
        raise ValueError("multiband source sampling requires outputs with shape [B,C,...]")
    nested_centers: Sequence[Sequence[Tensor]] = batch["band_centers"]  # type: ignore[assignment]
    nested_ids: Sequence[Sequence[Tensor]] = batch["band_ids"]  # type: ignore[assignment]
    sampled: List[List[Dict[str, Tensor]]] = []
    for b in range(emb.shape[0]):
        per_band: List[Dict[str, Tensor]] = []
        for band in range(emb.shape[1]):
            centers = nested_centers[b][band].to(device=emb.device, dtype=torch.float32)
            ids = nested_ids[b][band].to(device=emb.device, dtype=torch.long)
            per_band.append(
                {
                    "xy": centers,
                    "ids": ids,
                    "features": _sample_map_at_centers(emb[b, band], centers),
                    "shape": _sample_map_at_centers(shape[b, band], centers),
                }
            )
        sampled.append(per_band)
    return sampled


def _resolve_band_index(bands: Sequence[str], name: str) -> int:
    """Resolve full band names like HSC-I and short names like I."""

    query = str(name).strip()
    if not query:
        raise ValueError("empty band name")
    for idx, band in enumerate(bands):
        if str(band) == query:
            return idx
    query_upper = query.upper()
    for idx, band in enumerate(bands):
        band_upper = str(band).upper()
        if band_upper == query_upper or band_upper.split("-")[-1] == query_upper:
            return idx
    raise ValueError(f"band {name!r} is not in configured bands {list(bands)}")


def parse_ex_band_pairs(
    bands: Sequence[str],
    *,
    core_band: str,
    pair_specs: Optional[Sequence[str]] = None,
) -> Tuple[Tuple[int, int], ...]:
    """Build directed EX training pairs.

    Default EX pairs are adjacent bands in both directions.  Explicit specs use
    ``src:dst`` or ``src->dst``; ``all`` restores every directed cross-band
    pair and ``core`` uses the legacy core-band-to-all pattern.
    """

    if len(bands) <= 1:
        return tuple()
    specs = [part.strip() for item in (pair_specs or ()) for part in str(item).split(",") if part.strip()]
    if any(spec.lower() == "all" for spec in specs):
        return tuple((src, dst) for src in range(len(bands)) for dst in range(len(bands)) if src != dst)
    if any(spec.lower() == "core" for spec in specs):
        core_idx = _resolve_band_index(bands, core_band)
        return tuple((core_idx, dst) for dst in range(len(bands)) if dst != core_idx)
    pairs: List[Tuple[int, int]] = []
    if specs:
        for spec in specs:
            if "->" in spec:
                src_name, dst_name = spec.split("->", 1)
            elif ":" in spec:
                src_name, dst_name = spec.split(":", 1)
            else:
                raise ValueError(f"EX band pair {spec!r} must use src:dst or src->dst")
            src = _resolve_band_index(bands, src_name)
            dst = _resolve_band_index(bands, dst_name)
            if src == dst:
                raise ValueError(f"EX band pair {spec!r} links a band to itself")
            pair = (src, dst)
            if pair not in pairs:
                pairs.append(pair)
        return tuple(pairs)

    for src in range(len(bands) - 1):
        pairs.append((src, src + 1))
        pairs.append((src + 1, src))
    return tuple(pairs)


def matcher_classification_loss(
    matcher: AstroMatchNet2D,
    outputs: Dict[str, Tensor],
    batch: Dict[str, object],
    *,
    mode: str,
    candidate_count: int,
    offset_scale: float,
    band_pairs: Optional[Sequence[Tuple[int, int]]] = None,
) -> Tensor:
    """Cross-entropy loss for EX/EN candidate classification.

    ``mode="ex"`` samples candidates from other bands in the same cutout; the
    target is the candidate with the same source ``id``.  This trains EX to
    merge detections of the same astronomical source across bands.

    ``mode="en"`` samples candidates from the same band, excluding the anchor;
    positives are repeated ids if they exist, otherwise the none-of-above logit
    is the target.  This trains EN as an intra-band duplicate suppressor while
    remaining usable for single-band data.
    """

    if outputs["embedding"].ndim != 5:
        return outputs["embedding"].sum() * 0.0
    sampled = _sample_multiband_sources(outputs, batch)
    anchor_features: List[Tensor] = []
    candidate_features: List[Tensor] = []
    candidate_offsets: List[Tensor] = []
    candidate_shapes: List[Tensor] = []
    targets: List[int] = []
    k = int(candidate_count)
    scale = max(float(offset_scale), 1e-6)
    pair_map: Optional[Dict[int, List[int]]] = None
    if mode == "ex" and band_pairs is not None:
        pair_map = {}
        for src, dst in band_pairs:
            pair_map.setdefault(int(src), []).append(int(dst))

    for per_item in sampled:
        for band_idx, anchor_band in enumerate(per_item):
            if anchor_band["xy"].numel() == 0:
                continue
            if mode == "ex":
                if pair_map is None:
                    candidate_band_indices = [idx for idx in range(len(per_item)) if idx != band_idx]
                else:
                    candidate_band_indices = [idx for idx in pair_map.get(band_idx, []) if 0 <= idx < len(per_item)]
            elif mode == "en":
                candidate_band_indices = [band_idx]
            else:
                raise ValueError(f"unknown matcher mode: {mode}")
            if not candidate_band_indices:
                continue

            for anchor_idx in range(anchor_band["xy"].shape[0]):
                anchor_id = anchor_band["ids"][anchor_idx]
                anchor_xy = anchor_band["xy"][anchor_idx]
                anchor_shape = anchor_band["shape"][anchor_idx]
                for cand_band_idx in candidate_band_indices:
                    cand_band = per_item[cand_band_idx]
                    if cand_band["xy"].numel() == 0:
                        continue
                    positive = cand_band["ids"] == anchor_id
                    exclude_index = anchor_idx if mode == "en" and cand_band_idx == band_idx else None
                    if exclude_index is not None:
                        positive = positive.clone()
                        if 0 <= exclude_index < positive.shape[0]:
                            positive[exclude_index] = False
                    selected = _nearest_candidate_indices(
                        anchor_xy,
                        cand_band["xy"],
                        candidate_count=k,
                        exclude_index=exclude_index,
                        positive_mask=positive,
                    )
                    if selected.numel() == 0:
                        continue
                    pad = k - int(selected.numel())
                    if pad > 0:
                        selected = torch.cat([selected, selected[-1:].repeat(pad)], dim=0)
                    else:
                        selected = selected[:k]

                    cand_features = cand_band["features"][selected]
                    cand_xy = cand_band["xy"][selected]
                    cand_shape = cand_band["shape"][selected]
                    target_idx = k
                    same_selected = cand_band["ids"][selected] == anchor_id
                    if bool(same_selected.any()):
                        target_idx = int(torch.nonzero(same_selected, as_tuple=False)[0, 0].item())

                    anchor_features.append(anchor_band["features"][anchor_idx])
                    candidate_features.append(cand_features)
                    candidate_offsets.append((cand_xy - anchor_xy[None, :]) / scale)
                    candidate_shapes.append(torch.cat([anchor_shape[None, :].expand(k, -1), cand_shape], dim=1))
                    targets.append(target_idx)

    if not anchor_features:
        return outputs["embedding"].sum() * 0.0

    anchor_tensor = torch.stack(anchor_features)
    candidate_tensor = torch.stack(candidate_features)
    offset_tensor = torch.stack(candidate_offsets)
    shape_tensor = torch.stack(candidate_shapes)
    target_tensor = torch.tensor(targets, dtype=torch.long, device=anchor_tensor.device)
    logits = matcher(anchor_tensor, candidate_tensor, offset_tensor, shape_tensor)
    return F.cross_entropy(logits, target_tensor)


def embedding_triplet_loss(
    outputs: Dict[str, Tensor],
    batch: Dict[str, object],
    *,
    loss_fn: HardTripletLoss,
    max_sources_per_group: int = 0,
    negative_scope: str = "tile",
) -> Tensor:
    def _loss_with_optional_source_filter(features: Tensor, labels: Tensor, groups: Tensor) -> Tensor:
        if max_sources_per_group > 0:
            features, labels, groups = _filter_triplet_sources(
                features,
                labels,
                groups,
                max_sources_per_group=max_sources_per_group,
            )
            if features.shape[0] == 0:
                return outputs["embedding"].sum() * 0.0
        if negative_scope == "tile":
            loss_groups: Optional[Tensor] = groups
        elif negative_scope == "batch":
            loss_groups = None
        else:
            raise ValueError(f"unknown triplet negative scope: {negative_scope}")
        return loss_fn(features, labels, groups=loss_groups)

    if outputs["embedding"].ndim == 5:
        sampled = _sample_multiband_sources(outputs, batch)
        features: List[Tensor] = []
        labels: List[Tensor] = []
        groups: List[Tensor] = []
        for batch_idx, per_item in enumerate(sampled):
            for band in per_item:
                if band["features"].numel() == 0:
                    continue
                features.append(band["features"])
                labels.append(band["ids"])
                groups.append(torch.full((band["ids"].shape[0],), batch_idx, dtype=torch.long, device=band["ids"].device))
        if not features:
            return outputs["embedding"].sum() * 0.0
        return _loss_with_optional_source_filter(
            torch.cat(features, dim=0),
            torch.cat(labels, dim=0),
            torch.cat(groups, dim=0),
        )

    features, batch_ids = _sample_embeddings_at_centers(outputs, batch["centers"])  # type: ignore[arg-type]
    labels_list = [ids.to(device=features.device, dtype=torch.long) for ids in batch["ids"]]  # type: ignore[index]
    if not labels_list:
        return outputs["embedding"].sum() * 0.0
    labels = torch.cat(labels_list, dim=0)
    if labels.shape[0] != features.shape[0]:
        return outputs["embedding"].sum() * 0.0
    return _loss_with_optional_source_filter(features, labels, batch_ids)


def _filter_triplet_sources(
    features: Tensor,
    labels: Tensor,
    groups: Tensor,
    *,
    max_sources_per_group: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Limit triplet mining to a random subset of source IDs per image/group.

    The expensive part of hard triplet mining is the all-pairs ``cdist``.  This
    sampler keeps at most ``max_sources_per_group`` unique positive labels inside
    each group, while retaining every feature row for the selected labels.  For
    multi-band data that means G/R/I embeddings of the same source remain
    together, so cross-band positive pairs are not accidentally split.
    """

    if features.shape[0] == 0 or max_sources_per_group <= 0:
        return features, labels, groups
    labels = labels.long()
    groups = groups.long()
    keep = torch.zeros(labels.shape[0], dtype=torch.bool, device=labels.device)
    valid = labels > 0
    all_labels, all_counts = torch.unique(labels[valid], return_counts=True) if bool(valid.any()) else (labels[:0], labels[:0])
    repeated_labels = all_labels[all_counts > 1]
    for group in torch.unique(groups):
        in_group = valid & (groups == group)
        if not bool(in_group.any()):
            continue
        unique_labels = torch.unique(labels[in_group])
        limit = int(max_sources_per_group)
        if unique_labels.numel() > limit:
            priority = unique_labels[torch.isin(unique_labels, repeated_labels)]
            other = unique_labels[~torch.isin(unique_labels, repeated_labels)]
            if priority.numel() >= limit:
                order = torch.randperm(priority.numel(), device=labels.device)[:limit]
                unique_labels = priority[order]
            else:
                take_other = limit - int(priority.numel())
                if other.numel() > take_other:
                    other = other[torch.randperm(other.numel(), device=labels.device)[:take_other]]
                unique_labels = torch.cat([priority, other], dim=0)
        keep |= in_group & torch.isin(labels, unique_labels)
    return features[keep], labels[keep], groups[keep]


@torch.no_grad()
def detect_centers(
    outputs: Dict[str, Tensor],
    *,
    threshold: float = 0.0,
    nms_radius: int = 1,
    confidence_score: str = "cellect",
    use_ordinal_expectation: bool = False,
    debug_ordinal_expectation: bool = False,
    center_refinement: str = "integer",
    center_refinement_radius: int = 1,
) -> List[np.ndarray]:
    """Detect center candidates from confidence maps."""

    conf, peaks, _ = _compute_detection_peak_maps(
        outputs,
        threshold=threshold,
        nms_radius=nms_radius,
        confidence_score=confidence_score,
        use_ordinal_expectation=use_ordinal_expectation,
        debug_ordinal_expectation=debug_ordinal_expectation,
    )

    result: List[np.ndarray] = []
    merge_close = bool(use_ordinal_expectation or str(confidence_score) == "ordinal_expectation")
    for b in range(conf.shape[0]):
        y, x = torch.where(peaks[b])
        coords = _refine_peak_coordinates(
            conf[b],
            y,
            x,
            method=center_refinement,
            radius=center_refinement_radius,
        )
        if bool(merge_close) and coords.numel() > 0:
            peak_scores = conf[b, y, x]
            coords, _peak_scores = _merge_close_centers_by_score(
                coords,
                peak_scores,
                min_distance=float(ORDINAL_EXPECTATION_MERGE_RADIUS),
            )
        result.append(coords.detach().cpu().numpy().astype(np.float32))
    return result


@torch.no_grad()
def detect_centers_with_scores(
    outputs: Dict[str, Tensor],
    *,
    threshold: float = -float("inf"),
    nms_radius: int = 1,
    confidence_score: str = "cellect",
    use_ordinal_expectation: bool = False,
    debug_ordinal_expectation: bool = False,
    center_refinement: str = "integer",
    center_refinement_radius: int = 1,
) -> List[Dict[str, np.ndarray]]:
    """Detect centers and return ``xy`` plus scalar confidence scores."""

    conf, peaks, _ = _compute_detection_peak_maps(
        outputs,
        threshold=threshold,
        nms_radius=nms_radius,
        confidence_score=confidence_score,
        use_ordinal_expectation=use_ordinal_expectation,
        debug_ordinal_expectation=debug_ordinal_expectation,
    )

    result: List[Dict[str, np.ndarray]] = []
    merge_close = bool(use_ordinal_expectation or str(confidence_score) == "ordinal_expectation")
    for b in range(conf.shape[0]):
        y, x = torch.where(peaks[b])
        if x.numel() == 0:
            result.append({"xy": np.zeros((0, 2), dtype=np.float32), "score": np.zeros((0,), dtype=np.float32)})
            continue
        xy_t = _refine_peak_coordinates(
            conf[b],
            y,
            x,
            method=center_refinement,
            radius=center_refinement_radius,
        )
        peak_scores = conf[b, y, x]
        if bool(merge_close):
            xy_t, peak_scores = _merge_close_centers_by_score(
                xy_t,
                peak_scores,
                min_distance=float(ORDINAL_EXPECTATION_MERGE_RADIUS),
            )
        xy = xy_t.detach().cpu().numpy().astype(np.float32)
        score = peak_scores.detach().cpu().numpy().astype(np.float32)
        result.append({"xy": xy, "score": score})
    return result


@torch.no_grad()
def build_cellect_style_segmentation(
    outputs: Dict[str, Tensor],
    centers_list: Sequence[np.ndarray | Tensor],
    *,
    ellipse_sigma: float = 3.0,
    min_axis: float = 1.5,
) -> List[Dict[str, np.ndarray]]:
    """Build CELLECT-style dense ``seg`` and per-center ``seg_mask`` outputs.

    CELLECT first detects centers, samples a size estimate at each center, fills
    one instance region per center, and finally intersects that instance map
    with the coarse foreground segmentation.  This is the 2D astronomy version:

    - ``seg`` is the dense argmax class map from ``seg_logits``.
    - ``seg_mask`` stores instance ids 1..N filled from detected centers and
      predicted ellipses, then filtered by ``seg > 0``.

    The model's shape target stores unscaled semi-major/semi-minor axes, while
    the pseudo segmentation label uses ``ellipse_sigma`` times those axes.  The
    same scale is therefore used here to produce instance masks comparable to
    training-time masks.
    """

    seg = outputs["seg_logits"].argmax(dim=1).detach().cpu().numpy().astype(np.int16)
    shape = outputs["shape"].detach().cpu().numpy().astype(np.float32)
    results: List[Dict[str, np.ndarray]] = []

    for batch_idx, centers in enumerate(centers_list):
        if isinstance(centers, Tensor):
            centers_np = centers.detach().cpu().numpy().astype(np.float32)
        else:
            centers_np = np.asarray(centers, dtype=np.float32)
        if centers_np.ndim == 1:
            centers_np = centers_np.reshape((-1, 2)) if centers_np.size else np.zeros((0, 2), dtype=np.float32)

        seg_map = seg[batch_idx]
        h, w = seg_map.shape
        instance = np.zeros((h, w), dtype=np.int32)

        for instance_id, (cx, cy) in enumerate(centers_np[:, :2], start=1):
            if not np.isfinite(cx) or not np.isfinite(cy):
                continue
            xi = int(round(float(cx)))
            yi = int(round(float(cy)))
            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                continue

            major = float(shape[batch_idx, 0, yi, xi]) if shape.shape[1] >= 1 else min_axis
            minor = float(shape[batch_idx, 1, yi, xi]) if shape.shape[1] >= 2 else major
            theta = float(shape[batch_idx, 2, yi, xi]) if shape.shape[1] >= 3 else 0.0
            if not np.isfinite(major):
                major = min_axis
            if not np.isfinite(minor):
                minor = major
            if not np.isfinite(theta):
                theta = 0.0

            a = max(abs(major) * float(ellipse_sigma), float(min_axis))
            b = max(abs(minor) * float(ellipse_sigma), float(min_axis))
            radius = int(math.ceil(max(a, b))) + 2
            y0, y1 = max(0, yi - radius), min(h, yi + radius + 1)
            x0, x1 = max(0, xi - radius), min(w, xi + radius + 1)
            if x0 >= x1 or y0 >= y1:
                continue

            yy, xx = np.mgrid[y0:y1, x0:x1]
            dx = xx.astype(np.float32) - float(cx)
            dy = yy.astype(np.float32) - float(cy)
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            xr = cos_t * dx + sin_t * dy
            yr = -sin_t * dx + cos_t * dy
            ellipse = (xr / a) ** 2 + (yr / b) ** 2 <= 1.0
            patch = instance[y0:y1, x0:x1]
            patch[ellipse] = instance_id

        instance[seg_map <= 0] = 0
        results.append({"seg": seg_map.copy(), "seg_mask": instance})

    return results


def _as_center_array(obj: np.ndarray | Tensor) -> np.ndarray:
    if isinstance(obj, Tensor):
        arr = obj.detach().cpu().numpy()
    else:
        arr = np.asarray(obj)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return arr.reshape(-1, 2)


def build_adjacent_mutual_link_components(
    pred_by_band: Sequence[np.ndarray | Tensor],
    *,
    match_radius: float,
    band_names: Sequence[str],
) -> List[Dict[str, object]]:
    """Link adjacent-band detections by mutual nearest neighbors.

    Only edges between adjacent bands are considered, e.g. G->R->I->Z->Y.
    A pair is linked when it is within ``match_radius`` and each detection is
    the nearest neighbor of the other in the adjacent band.
    """

    centers_by_band = [_as_center_array(item) for item in pred_by_band]
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}

    def find(node: Tuple[int, int]) -> Tuple[int, int]:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: Tuple[int, int], b: Tuple[int, int]) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for band_idx, centers in enumerate(centers_by_band):
        for source_idx in range(int(centers.shape[0])):
            parent.setdefault((band_idx, source_idx), (band_idx, source_idx))

    radius = float(match_radius)
    for band_idx in range(max(0, len(centers_by_band) - 1)):
        left = centers_by_band[band_idx]
        right = centers_by_band[band_idx + 1]
        if left.shape[0] == 0 or right.shape[0] == 0:
            continue
        diff = left[:, None, :2] - right[None, :, :2]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        nearest_right = np.argmin(dist, axis=1)
        nearest_left = np.argmin(dist, axis=0)
        for left_idx, right_idx in enumerate(nearest_right.tolist()):
            if int(nearest_left[int(right_idx)]) != int(left_idx):
                continue
            if float(dist[int(left_idx), int(right_idx)]) <= radius:
                union((band_idx, int(left_idx)), (band_idx + 1, int(right_idx)))

    groups: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for node in sorted(parent):
        groups.setdefault(find(node), []).append(node)

    components: List[Dict[str, object]] = []
    for members in groups.values():
        members = sorted(members)
        member_centers = [
            [int(band_idx), centers_by_band[int(band_idx)][int(source_idx), :2].astype(float).tolist()]
            for band_idx, source_idx in members
        ]
        member_band_names = [
            str(band_names[int(band_idx)]) if int(band_idx) < len(band_names) else f"band{int(band_idx)}"
            for band_idx, _source_idx in members
        ]
        ref_band, ref_source = (
            next(((band_idx, source_idx) for band_idx, source_idx in members if int(band_idx) == 2), members[0])
        )
        ref_center = centers_by_band[int(ref_band)][int(ref_source), :2].astype(float).tolist()
        components.append(
            {
                "members": [[int(band_idx), int(source_idx)] for band_idx, source_idx in members],
                "member_centers": member_centers,
                "member_bands": member_band_names,
                "center": ref_center,
                "reference_band": int(ref_band),
                "link_method": "adjacent_mutual_nearest",
            }
        )
    return components


@torch.no_grad()
def generate_pu_pseudo_labels(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    output_path: Path,
    band_names: Sequence[str],
    score_percentile_start: float = 99.0,
    score_percentile_end: float = 60.0,
    min_center_distance_px: float = 3.0,
    clean_iou_threshold: float = 0.33,
    axis_ratio_min: float = 0.1,
    nms_radius: int = 1,
    confidence_score: str = "cellect",
    use_ordinal_expectation: bool = False,
    debug_ordinal_expectation: bool = False,
    ellipse_sigma: float = 2.0,
    max_pseudo_per_record_band: int = 512,
    show_progress: bool = True,
) -> Dict[str, object]:
    """Run detection and write high-confidence PU self-training pseudo labels.

    Candidates are thresholded by a percentile schedule from p99 to p60,
    are required to be farther than ``min_center_distance_px`` from existing
    clean/center_only centers, have IoU < ``clean_iou_threshold`` with the clean
    mask, and have ``minor / major > axis_ratio_min``.
    """

    model.eval()
    epoch_number = int(epoch) + 1
    progress = 0.0 if total_epochs <= 1 else min(max(float(epoch) / float(total_epochs - 1), 0.0), 1.0)
    percentile = float(score_percentile_start) + (float(score_percentile_end) - float(score_percentile_start)) * progress
    percentile = float(np.clip(percentile, 0.0, 100.0))
    raw_items: List[Dict[str, object]] = []
    all_scores: List[float] = []

    for batch in tqdm(loader, desc="pu-pseudo-detect", leave=False, disable=not show_progress):
        images = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)  # type: ignore[union-attr]
        outputs = model(images)
        if outputs["seg_logits"].ndim == 5:
            flat_outputs = _flatten_per_band_outputs(outputs)
            detection_items = detect_centers_with_scores(
                flat_outputs,
                threshold=-float("inf"),
                nms_radius=nms_radius,
                confidence_score=confidence_score,
                use_ordinal_expectation=use_ordinal_expectation,
                debug_ordinal_expectation=debug_ordinal_expectation,
            )
            batch_size = int(outputs["seg_logits"].shape[0])
            band_count = int(outputs["seg_logits"].shape[1])
            flat_shape = flat_outputs["shape"].detach().cpu()
            for flat_idx, item in enumerate(detection_items):
                rec_idx = flat_idx // band_count
                band_idx = flat_idx % band_count
                if rec_idx >= batch_size:
                    continue
                record_name = str(batch["name"][rec_idx])  # type: ignore[index]
                clean_mask = batch["band_clean_mask"][rec_idx][band_idx].detach().cpu().numpy().astype(bool)  # type: ignore[index]
                clean_label_map, clean_label_areas = _connected_components_bool(clean_mask)
                existing_centers = _centers_from_confidence_target(batch["band_confidence"][rec_idx][band_idx])  # type: ignore[index]
                h, w = clean_mask.shape
                xy = np.asarray(item["xy"], dtype=np.float32)
                score = np.asarray(item["score"], dtype=np.float32)
                for cand_idx, (center_xy, score_value) in enumerate(zip(xy, score)):
                    x, y = float(center_xy[0]), float(center_xy[1])
                    xi, yi = int(round(x)), int(round(y))
                    if xi < 0 or yi < 0 or xi >= w or yi >= h:
                        continue
                    major = float(flat_shape[flat_idx, 0, yi, xi]) if flat_shape.shape[1] >= 1 else 1.5
                    minor = float(flat_shape[flat_idx, 1, yi, xi]) if flat_shape.shape[1] >= 2 else major
                    theta = float(flat_shape[flat_idx, 2, yi, xi]) if flat_shape.shape[1] >= 3 else 0.0
                    axis_ratio = min(abs(minor), abs(major)) / max(abs(minor), abs(major), 1e-6)
                    if not np.isfinite(axis_ratio) or axis_ratio <= float(axis_ratio_min):
                        continue
                    min_dist = _min_distance_to_points(center_xy, existing_centers)
                    if min_dist <= float(min_center_distance_px):
                        continue
                    pred_mask = _ellipse_mask_np((h, w), x, y, major, minor, theta, ellipse_sigma=ellipse_sigma)
                    pred_area = int(pred_mask.sum())
                    if pred_area <= 0:
                        continue
                    clean_iou = _max_iou_with_labeled_mask(pred_mask, clean_label_map, clean_label_areas)
                    if clean_iou >= float(clean_iou_threshold):
                        continue
                    item_dict = {
                        "record": record_name,
                        "band": str(band_names[band_idx]) if band_idx < len(band_names) else f"band{band_idx}",
                        "band_index": int(band_idx),
                        "x": x,
                        "y": y,
                        "score": float(score_value),
                        "major": major,
                        "minor": minor,
                        "theta": theta,
                        "axis_ratio": float(axis_ratio),
                        "min_existing_center_distance_px": min_dist,
                        "clean_iou": clean_iou,
                        "source": "pu_self_training",
                        "epoch": epoch_number,
                    }
                    raw_items.append(item_dict)
                    all_scores.append(float(score_value))
        else:
            detection_items = detect_centers_with_scores(
                outputs,
                threshold=-float("inf"),
                nms_radius=nms_radius,
                confidence_score=confidence_score,
                use_ordinal_expectation=use_ordinal_expectation,
                debug_ordinal_expectation=debug_ordinal_expectation,
            )
            shape_cpu = outputs["shape"].detach().cpu()
            for rec_idx, item in enumerate(detection_items):
                record_name = str(batch["name"][rec_idx])  # type: ignore[index]
                clean_mask = batch["clean_mask"][rec_idx].detach().cpu().numpy().astype(bool)  # type: ignore[index]
                clean_label_map, clean_label_areas = _connected_components_bool(clean_mask)
                existing_centers = _centers_from_confidence_target(batch["confidence"][rec_idx])  # type: ignore[index]
                h, w = clean_mask.shape
                xy = np.asarray(item["xy"], dtype=np.float32)
                score = np.asarray(item["score"], dtype=np.float32)
                for center_xy, score_value in zip(xy, score):
                    x, y = float(center_xy[0]), float(center_xy[1])
                    xi, yi = int(round(x)), int(round(y))
                    if xi < 0 or yi < 0 or xi >= w or yi >= h:
                        continue
                    major = float(shape_cpu[rec_idx, 0, yi, xi]) if shape_cpu.shape[1] >= 1 else 1.5
                    minor = float(shape_cpu[rec_idx, 1, yi, xi]) if shape_cpu.shape[1] >= 2 else major
                    theta = float(shape_cpu[rec_idx, 2, yi, xi]) if shape_cpu.shape[1] >= 3 else 0.0
                    axis_ratio = min(abs(minor), abs(major)) / max(abs(minor), abs(major), 1e-6)
                    if not np.isfinite(axis_ratio) or axis_ratio <= float(axis_ratio_min):
                        continue
                    min_dist = _min_distance_to_points(center_xy, existing_centers)
                    if min_dist <= float(min_center_distance_px):
                        continue
                    pred_mask = _ellipse_mask_np((h, w), x, y, major, minor, theta, ellipse_sigma=ellipse_sigma)
                    pred_area = int(pred_mask.sum())
                    if pred_area <= 0:
                        continue
                    clean_iou = _max_iou_with_labeled_mask(pred_mask, clean_label_map, clean_label_areas)
                    if clean_iou >= float(clean_iou_threshold):
                        continue
                    item_dict = {
                        "record": record_name,
                        "band": "__primary__",
                        "band_index": -1,
                        "x": x,
                        "y": y,
                        "score": float(score_value),
                        "major": major,
                        "minor": minor,
                        "theta": theta,
                        "axis_ratio": float(axis_ratio),
                        "min_existing_center_distance_px": min_dist,
                        "clean_iou": clean_iou,
                        "source": "pu_self_training",
                        "epoch": epoch_number,
                    }
                    raw_items.append(item_dict)
                    all_scores.append(float(score_value))

    score_threshold = float(np.percentile(np.asarray(all_scores, dtype=np.float32), percentile)) if all_scores else float("inf")
    by_record: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
    kept = 0
    counts_by_record_band: Dict[Tuple[str, str], int] = {}
    for item in sorted(raw_items, key=lambda row: float(row["score"]), reverse=True):
        if float(item["score"]) < score_threshold:
            continue
        key = (str(item["record"]), str(item["band"]))
        count = counts_by_record_band.get(key, 0)
        if count >= int(max_pseudo_per_record_band):
            continue
        counts_by_record_band[key] = count + 1
        record_bucket = by_record.setdefault(str(item["record"]), {})
        band_bucket = record_bucket.setdefault(str(item["band"]), [])
        band_bucket.append(item)
        kept += 1

    payload = {
        "version": 1,
        "epoch": epoch_number,
        "percentile": percentile,
        "score_threshold": score_threshold,
        "raw_candidates": len(raw_items),
        "kept": kept,
        "filters": {
            "min_center_distance_px": float(min_center_distance_px),
            "clean_iou_threshold": float(clean_iou_threshold),
            "axis_ratio_min": float(axis_ratio_min),
            "ellipse_sigma": float(ellipse_sigma),
        },
        "bands": [str(band) for band in band_names],
        "by_record": by_record,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "path": str(output_path),
        "epoch": epoch_number,
        "percentile": percentile,
        "score_threshold": score_threshold,
        "raw_candidates": len(raw_items),
        "kept": kept,
    }


def match_points(pred_xy: np.ndarray, gt_xy: np.ndarray, radius: float) -> Tuple[int, int, int]:
    matched_pred, matched_gt = match_point_indices(pred_xy, gt_xy, radius)
    tp = len(matched_pred)
    fp = len(pred_xy) - tp
    fn = len(gt_xy) - tp
    return tp, fp, fn


def match_point_indices(pred_xy: np.ndarray, gt_xy: np.ndarray, radius: float) -> Tuple[set[int], set[int]]:
    if pred_xy.size == 0:
        return set(), set()
    if gt_xy.size == 0:
        return set(), set()
    dist = np.sqrt(((pred_xy[:, None, :] - gt_xy[None, :, :]) ** 2).sum(axis=2))
    pairs = []
    for i in range(dist.shape[0]):
        j = int(np.argmin(dist[i]))
        if dist[i, j] <= radius:
            pairs.append((float(dist[i, j]), i, j))
    pairs.sort()
    used_pred = set()
    used_gt = set()
    for _d, i, j in pairs:
        if i in used_pred or j in used_gt:
            continue
        used_pred.add(i)
        used_gt.add(j)
    return used_pred, used_gt


def match_clean_and_ordinary_ignore(
    pred_xy: np.ndarray,
    clean_gt_xy: np.ndarray,
    ordinary_ignore_xy: np.ndarray,
    radius: float,
    *,
    clean_region_masks: Sequence[Optional[Tensor]] = (),
) -> Dict[str, int]:
    clean_pred, clean_gt = match_point_indices(pred_xy, clean_gt_xy, radius)
    unmatched_indices = [idx for idx in range(len(pred_xy)) if idx not in clean_pred]
    if unmatched_indices:
        unmatched_pred = pred_xy[np.asarray(unmatched_indices, dtype=np.int64)]
    else:
        unmatched_pred = np.zeros((0, 2), dtype=np.float32)
    ordinary_pred_rel, ordinary_gt = match_point_indices(unmatched_pred, ordinary_ignore_xy, radius)
    ordinary_pred = {unmatched_indices[idx] for idx in ordinary_pred_rel}
    clean_tp = len(clean_pred)
    ordinary_tp = len(ordinary_pred)
    unmatched_after_all = [
        idx for idx in range(len(pred_xy))
        if idx not in clean_pred and idx not in ordinary_pred
    ]
    if unmatched_after_all:
        unmatched_after_all_xy = pred_xy[np.asarray(unmatched_after_all, dtype=np.int64)]
    else:
        unmatched_after_all_xy = np.zeros((0, 2), dtype=np.float32)
    clean_region_fp = _count_points_in_masks(unmatched_after_all_xy, clean_region_masks)
    return {
        "tp": clean_tp,
        "fp": int(len(pred_xy) - clean_tp - ordinary_tp),
        "clean_region_fp": int(clean_region_fp),
        "fn": int(len(clean_gt_xy) - len(clean_gt)),
        "ordinary_ignore_tp": ordinary_tp,
        "ordinary_ignore_fn": int(len(ordinary_ignore_xy) - len(ordinary_gt)),
        "ordinary_ignore_total": int(len(ordinary_ignore_xy)),
    }


def _update_detection_totals(
    totals: Dict[str, object],
    base_model: nn.Module,
    outputs: Dict[str, Tensor],
    batch: Dict[str, object],
    *,
    threshold: float,
    nms_radius: int,
    confidence_score: str,
    use_ordinal_expectation: bool,
    debug_ordinal_expectation: bool,
    center_refinement: str,
    center_refinement_radius: int,
    match_radius: float,
    use_en_postprocess: bool,
    en_candidate_count: int,
    en_threshold: float,
    use_ex_link_postprocess: bool,
    ex_link_threshold: float,
    ex_band_pairs: Optional[Sequence[Tuple[int, int]]],
    band_names: Sequence[str],
    collect_candidate_stats: bool = False,
    ignore_mask_during_detection: bool = True,
) -> None:
    per_band_outputs = outputs["confidence"].ndim == 5
    band_count = int(outputs["confidence"].shape[1]) if per_band_outputs else 0
    def _empty_center_list(n: int) -> List[Tensor]:
        return [outputs["confidence"].new_zeros((0, 2), dtype=torch.float32).cpu() for _ in range(n)]

    def _merge_center_lists(*lists: Sequence[Tensor]) -> List[Tensor]:
        max_len = max((len(items) for items in lists), default=0)
        merged: List[Tensor] = []
        for idx in range(max_len):
            pieces = [items[idx].reshape(-1, 2).cpu() for items in lists if idx < len(items) and items[idx].numel()]
            if not pieces:
                merged.append(outputs["confidence"].new_zeros((0, 2), dtype=torch.float32).cpu())
                continue
            centers = torch.cat(pieces, dim=0).to(dtype=torch.float32)
            merged.append(torch.unique(centers, dim=0))
        return merged

    candidate_debug: Dict[str, object] = {}

    if use_en_postprocess and hasattr(base_model, "EN"):
        if bool(collect_candidate_stats):
            candidate_outputs = _flatten_per_band_outputs(outputs) if per_band_outputs else outputs
            _, _, candidate_debug = _compute_detection_peak_maps(
                candidate_outputs,
                threshold=threshold,
                nms_radius=nms_radius,
                confidence_score=confidence_score,
                use_ordinal_expectation=use_ordinal_expectation,
                debug_ordinal_expectation=debug_ordinal_expectation,
            )
        pred_list = detect_centers_with_en(
            base_model,
            outputs,
            threshold=threshold,
            nms_radius=nms_radius,
            confidence_score=confidence_score,
            use_ordinal_expectation=use_ordinal_expectation,
            debug_ordinal_expectation=debug_ordinal_expectation,
            center_refinement=center_refinement,
            center_refinement_radius=center_refinement_radius,
            match_radius=match_radius,
            candidate_count=en_candidate_count,
            en_threshold=en_threshold,
        )
        gt_list = _flatten_band_centers(batch["band_centers"]) if per_band_outputs else batch["centers"]  # type: ignore[arg-type]
        band_indices = [idx % band_count for idx in range(len(pred_list))] if band_count else [None for _ in pred_list]
        clean_masks = (
            _flatten_band_masks(batch["band_clean_mask"])  # type: ignore[arg-type]
            if per_band_outputs and "band_clean_mask" in batch
            else ([item for item in batch["clean_mask"]] if "clean_mask" in batch else [None for _ in pred_list])  # type: ignore[index]
        )
        background_masks = (
            _flatten_band_masks(batch["band_background_mask"])  # type: ignore[arg-type]
            if per_band_outputs and "band_background_mask" in batch
            else ([item for item in batch["background_mask"]] if "background_mask" in batch else [None for _ in pred_list])  # type: ignore[index]
        )
        ordinary_base = (
            _flatten_band_centers(batch["band_ignore_centers"])  # type: ignore[arg-type]
            if per_band_outputs and "band_ignore_centers" in batch
            else (batch["ignore_centers"] if "ignore_centers" in batch else _empty_center_list(len(pred_list)))  # type: ignore[assignment]
        )
        strict_center_list = (
            _flatten_band_centers(batch["band_strict_center_only_centers"])  # type: ignore[arg-type]
            if per_band_outputs and "band_strict_center_only_centers" in batch
            else (batch["strict_center_only_centers"] if "strict_center_only_centers" in batch else _empty_center_list(len(pred_list)))  # type: ignore[assignment]
        )
        strict_ignore_list = (
            _flatten_band_centers(batch["band_strict_ignore_centers"])  # type: ignore[arg-type]
            if per_band_outputs and "band_strict_ignore_centers" in batch
            else (batch["strict_ignore_centers"] if "strict_ignore_centers" in batch else _empty_center_list(len(pred_list)))  # type: ignore[assignment]
        )
        ordinary_ignore_list = _merge_center_lists(ordinary_base, strict_center_list, strict_ignore_list)
    elif per_band_outputs:
        flat_outputs = _flatten_per_band_outputs(outputs)
        if bool(collect_candidate_stats):
            _, _, candidate_debug = _compute_detection_peak_maps(
                flat_outputs,
                threshold=threshold,
                nms_radius=nms_radius,
                confidence_score=confidence_score,
                use_ordinal_expectation=use_ordinal_expectation,
                debug_ordinal_expectation=debug_ordinal_expectation,
            )
        pred_list = detect_centers(
            flat_outputs,
            threshold=threshold,
            nms_radius=nms_radius,
            confidence_score=confidence_score,
            use_ordinal_expectation=use_ordinal_expectation,
            debug_ordinal_expectation=debug_ordinal_expectation,
            center_refinement=center_refinement,
            center_refinement_radius=center_refinement_radius,
        )
        gt_list = _flatten_band_centers(batch["band_centers"])  # type: ignore[arg-type]
        band_indices = [idx % band_count for idx in range(len(pred_list))]
        clean_masks = (
            _flatten_band_masks(batch["band_clean_mask"])  # type: ignore[arg-type]
            if "band_clean_mask" in batch
            else [None for _ in pred_list]
        )
        background_masks = (
            _flatten_band_masks(batch["band_background_mask"])  # type: ignore[arg-type]
            if "band_background_mask" in batch
            else [None for _ in pred_list]
        )
        ordinary_base = (
            _flatten_band_centers(batch["band_ignore_centers"])  # type: ignore[arg-type]
            if "band_ignore_centers" in batch
            else _empty_center_list(len(pred_list))
        )
        strict_center_list = (
            _flatten_band_centers(batch["band_strict_center_only_centers"])  # type: ignore[arg-type]
            if "band_strict_center_only_centers" in batch
            else _empty_center_list(len(pred_list))
        )
        strict_ignore_list = (
            _flatten_band_centers(batch["band_strict_ignore_centers"])  # type: ignore[arg-type]
            if "band_strict_ignore_centers" in batch
            else _empty_center_list(len(pred_list))
        )
        ordinary_ignore_list = _merge_center_lists(ordinary_base, strict_center_list, strict_ignore_list)
    else:
        if bool(collect_candidate_stats):
            _, _, candidate_debug = _compute_detection_peak_maps(
                outputs,
                threshold=threshold,
                nms_radius=nms_radius,
                confidence_score=confidence_score,
                use_ordinal_expectation=use_ordinal_expectation,
                debug_ordinal_expectation=debug_ordinal_expectation,
            )
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
        gt_list = batch["centers"]  # type: ignore[assignment]
        band_indices = [None for _ in pred_list]
        clean_masks = [item for item in batch["clean_mask"]] if "clean_mask" in batch else [None for _ in pred_list]  # type: ignore[index]
        background_masks = (
            [item for item in batch["background_mask"]] if "background_mask" in batch else [None for _ in pred_list]  # type: ignore[index]
        )
        ordinary_base = batch["ignore_centers"] if "ignore_centers" in batch else _empty_center_list(len(pred_list))  # type: ignore[assignment]
        strict_center_list = (
            batch["strict_center_only_centers"] if "strict_center_only_centers" in batch else _empty_center_list(len(pred_list))  # type: ignore[assignment]
        )
        strict_ignore_list = batch["strict_ignore_centers"] if "strict_ignore_centers" in batch else _empty_center_list(len(pred_list))  # type: ignore[assignment]
        ordinary_ignore_list = _merge_center_lists(ordinary_base, strict_center_list, strict_ignore_list)

    per_band_counts = totals["per_band_counts"]
    assert isinstance(per_band_counts, dict)
    candidate_stats_total = totals.get("candidate_stats") if bool(collect_candidate_stats) else None
    candidate_center_score = candidate_debug.get("center_score")
    candidate_argmax_channel = candidate_debug.get("argmax_channel")
    candidate_spatial_localmax = candidate_debug.get("spatial_localmax")
    candidate_seed = candidate_debug.get("seed_candidates")
    candidate_foreground_gate = candidate_debug.get("foreground_gate")
    candidate_threshold_pass = candidate_debug.get("threshold_pass")
    candidate_final = candidate_debug.get("final_peaks")
    candidate_top_channel = candidate_debug.get("top_channel_argmax")
    candidate_foreground_gate_active = bool(candidate_debug.get("foreground_gate_active", False))
    candidate_effective_threshold = float(candidate_debug.get("effective_threshold", threshold))
    if (
        isinstance(candidate_stats_total, dict)
        and isinstance(candidate_center_score, Tensor)
        and isinstance(candidate_argmax_channel, Tensor)
        and isinstance(candidate_spatial_localmax, Tensor)
        and isinstance(candidate_seed, Tensor)
        and isinstance(candidate_foreground_gate, Tensor)
        and isinstance(candidate_threshold_pass, Tensor)
        and isinstance(candidate_final, Tensor)
    ):
        for map_idx, band_idx in enumerate(band_indices):
            top_channel_map = candidate_top_channel[map_idx] if isinstance(candidate_top_channel, Tensor) else None
            _update_candidate_stats_bucket(
                candidate_stats_total,
                center_score=candidate_center_score[map_idx],
                argmax_channel=candidate_argmax_channel[map_idx],
                top_channel_argmax=top_channel_map,
                spatial_localmax=candidate_spatial_localmax[map_idx],
                seed_candidates=candidate_seed[map_idx],
                foreground_gate=candidate_foreground_gate[map_idx],
                threshold_pass=candidate_threshold_pass[map_idx],
                final_peaks=candidate_final[map_idx],
                foreground_gate_active=candidate_foreground_gate_active,
                threshold=candidate_effective_threshold,
            )
            if bool(collect_candidate_stats) and band_idx is not None and band_names:
                band_name = str(band_names[int(band_idx)])
                band_bucket = per_band_counts.setdefault(
                    band_name,
                    {
                        "tp": 0,
                        "fp": 0,
                        "clean_region_fp": 0,
                        "fn": 0,
                        "ordinary_ignore_tp": 0,
                        "ordinary_ignore_fn": 0,
                        "ordinary_ignore_total": 0,
                        "strict_ignored_pred": 0,
                        **({"candidate_stats": _init_candidate_stats_bucket()} if bool(collect_candidate_stats) else {}),
                    },
                )
                assert isinstance(band_bucket, dict)
                band_candidate_stats = band_bucket.get("candidate_stats")
                if isinstance(band_candidate_stats, dict):
                    _update_candidate_stats_bucket(
                        band_candidate_stats,
                        center_score=candidate_center_score[map_idx],
                        argmax_channel=candidate_argmax_channel[map_idx],
                        top_channel_argmax=top_channel_map,
                        spatial_localmax=candidate_spatial_localmax[map_idx],
                        seed_candidates=candidate_seed[map_idx],
                        foreground_gate=candidate_foreground_gate[map_idx],
                        threshold_pass=candidate_threshold_pass[map_idx],
                        final_peaks=candidate_final[map_idx],
                        foreground_gate_active=candidate_foreground_gate_active,
                        threshold=candidate_effective_threshold,
                    )
    for pred_xy, gt_xy, band_idx, clean_mask, background_mask, ordinary_ignore_xy in zip(
        pred_list, gt_list, band_indices, clean_masks, background_masks, ordinary_ignore_list
    ):
        strict_ignored = 0
        clean_gt_np = gt_xy.numpy().astype(np.float32)
        ordinary_np = ordinary_ignore_xy.numpy().astype(np.float32)
        counts_now = match_clean_and_ordinary_ignore(
            pred_xy,
            clean_gt_np,
            ordinary_np,
            match_radius,
            clean_region_masks=(clean_mask, background_mask),
        )
        totals["tp"] = int(totals["tp"]) + int(counts_now["tp"])
        totals["fp"] = int(totals["fp"]) + int(counts_now["fp"])
        totals["clean_region_fp"] = int(totals["clean_region_fp"]) + int(counts_now["clean_region_fp"])
        totals["fn"] = int(totals["fn"]) + int(counts_now["fn"])
        totals["ordinary_ignore_tp"] = int(totals["ordinary_ignore_tp"]) + int(counts_now["ordinary_ignore_tp"])
        totals["ordinary_ignore_fn"] = int(totals["ordinary_ignore_fn"]) + int(counts_now["ordinary_ignore_fn"])
        totals["ordinary_ignore_total"] = int(totals["ordinary_ignore_total"]) + int(counts_now["ordinary_ignore_total"])
        totals["strict_ignored_pred"] = int(totals["strict_ignored_pred"]) + int(strict_ignored)
        if band_idx is not None and band_names:
            band_name = str(band_names[int(band_idx)])
            counts = per_band_counts.setdefault(
                band_name,
                {
                    "tp": 0,
                    "fp": 0,
                    "clean_region_fp": 0,
                    "fn": 0,
                    "ordinary_ignore_tp": 0,
                    "ordinary_ignore_fn": 0,
                    "ordinary_ignore_total": 0,
                    "strict_ignored_pred": 0,
                    "candidate_stats": _init_candidate_stats_bucket(),
                },
            )
            assert isinstance(counts, dict)
            counts["tp"] = int(counts.get("tp", 0)) + int(counts_now["tp"])
            counts["fp"] = int(counts.get("fp", 0)) + int(counts_now["fp"])
            counts["clean_region_fp"] = int(counts.get("clean_region_fp", 0)) + int(counts_now["clean_region_fp"])
            counts["fn"] = int(counts.get("fn", 0)) + int(counts_now["fn"])
            counts["ordinary_ignore_tp"] = int(counts.get("ordinary_ignore_tp", 0)) + int(counts_now["ordinary_ignore_tp"])
            counts["ordinary_ignore_fn"] = int(counts.get("ordinary_ignore_fn", 0)) + int(counts_now["ordinary_ignore_fn"])
            counts["ordinary_ignore_total"] = int(counts.get("ordinary_ignore_total", 0)) + int(counts_now["ordinary_ignore_total"])
            counts["strict_ignored_pred"] = int(counts.get("strict_ignored_pred", 0)) + int(strict_ignored)

    ex_link_active = bool(use_ex_link_postprocess and per_band_outputs and hasattr(base_model, "EX"))
    if per_band_outputs and not ex_link_active:
        nested_band_centers: Sequence[Sequence[Tensor]] = batch["band_centers"]  # type: ignore[assignment]
        nested_band_ids: Sequence[Sequence[Tensor]] = batch["band_ids"]  # type: ignore[assignment]
        merged_centers_list: Sequence[Tensor] = batch["centers"]  # type: ignore[assignment]
        merged_ids_list: Sequence[Tensor] = batch["ids"]  # type: ignore[assignment]
        nested_band_rejected_ids: Optional[Sequence[Sequence[Tensor]]] = batch.get("band_rejected_ids")  # type: ignore[assignment,union-attr]
        link_metrics_total = totals["link_metrics_total"]
        assert isinstance(link_metrics_total, dict)
        for item_idx in range(int(outputs["confidence"].shape[0])):
            item_pred_by_band = [
                pred_list[item_idx * band_count + band_idx]
                for band_idx in range(band_count)
            ]
            components = build_adjacent_mutual_link_components(
                item_pred_by_band,
                match_radius=match_radius,
                band_names=band_names,
            )
            item_metrics = _evaluate_link_components(
                components,
                nested_band_centers[item_idx],
                nested_band_ids[item_idx],
                match_radius=match_radius,
                band_names=band_names,
                merged_centers=merged_centers_list[item_idx],
                merged_ids=merged_ids_list[item_idx],
                band_rejected_ids=(
                    nested_band_rejected_ids[item_idx] if nested_band_rejected_ids is not None else None
                ),
            )
            _accumulate_link_metrics(link_metrics_total, item_metrics)

    if ex_link_active:
        linked_pred_list, components_all = detect_centers_with_ex_link(
            base_model,
            outputs,
            threshold=threshold,
            nms_radius=nms_radius,
            confidence_score=confidence_score,
            use_ordinal_expectation=use_ordinal_expectation,
            debug_ordinal_expectation=debug_ordinal_expectation,
            center_refinement=center_refinement,
            center_refinement_radius=center_refinement_radius,
            match_radius=match_radius,
            candidate_count=en_candidate_count,
            ex_threshold=ex_link_threshold,
            use_en_postprocess=use_en_postprocess,
            en_threshold=en_threshold,
            band_pairs=ex_band_pairs,
        )
        linked_gt_list = batch["centers"]  # type: ignore[assignment]
        for pred_xy, gt_xy in zip(linked_pred_list, linked_gt_list):
            t, f, n = match_points(pred_xy, gt_xy.numpy().astype(np.float32), match_radius)
            totals["linked_tp"] = int(totals["linked_tp"]) + int(t)
            totals["linked_fp"] = int(totals["linked_fp"]) + int(f)
            totals["linked_fn"] = int(totals["linked_fn"]) + int(n)

        nested_band_centers: Sequence[Sequence[Tensor]] = batch["band_centers"]  # type: ignore[assignment]
        nested_band_ids: Sequence[Sequence[Tensor]] = batch["band_ids"]  # type: ignore[assignment]
        merged_centers_list: Sequence[Tensor] = batch["centers"]  # type: ignore[assignment]
        merged_ids_list: Sequence[Tensor] = batch["ids"]  # type: ignore[assignment]
        nested_band_rejected_ids: Optional[Sequence[Sequence[Tensor]]] = batch.get("band_rejected_ids")  # type: ignore[assignment,union-attr]
        link_metrics_total = totals["link_metrics_total"]
        assert isinstance(link_metrics_total, dict)
        for item_idx, components in enumerate(components_all):
            item_metrics = _evaluate_link_components(
                components,
                nested_band_centers[item_idx],
                nested_band_ids[item_idx],
                match_radius=match_radius,
                band_names=band_names,
                merged_centers=merged_centers_list[item_idx],
                merged_ids=merged_ids_list[item_idx],
                band_rejected_ids=(
                    nested_band_rejected_ids[item_idx] if nested_band_rejected_ids is not None else None
                ),
            )
            _accumulate_link_metrics(link_metrics_total, item_metrics)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    weights: LossWeights,
    triplet_loss_fn: HardTripletLoss,
    triplet_enabled: bool,
    ex_enabled: bool,
    en_enabled: bool,
    matcher_candidate_count: int,
    matcher_max_anchors_per_band: int,
    triplet_max_sources_per_group: int,
    triplet_negative_scope: str,
    ex_band_pairs: Optional[Sequence[Tuple[int, int]]],
    center_radius_px: float,
    epoch_index: int = 0,
    threshold: float = 2.0,
    nms_radius: int = 3,
    confidence_score: str = "smooth",
    use_ordinal_expectation: bool = False,
    debug_ordinal_expectation: bool = False,
    center_refinement: str = "integer",
    center_refinement_radius: int = 1,
    ellipse_sigma: float = 2.0,
    amp_dtype: Optional[torch.dtype] = None,
    scheduler_step: Optional[Callable[[], None]] = None,
    global_step_start: int = 0,
    iteration_log_interval: int = 0,
    iteration_log_fn: Optional[Callable[[Dict[str, float], int], None]] = None,
    debug_batch_start: int = -1,
    debug_batch_end: int = -1,
    debug_batch_all_ranks: bool = False,
    distributed: bool = False,
    show_progress: bool = True,
    freeze_sam_proposal: bool = False,
    freeze_sam_encoder: bool = False,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    base_model = unwrap_model(model)
    sums: Dict[str, float] = {
        "total": 0.0,
        "seg": 0.0,
        "confidence": 0.0,
        "shape": 0.0,
        "center": 0.0,
        "triplet": 0.0,
        "ex_class": 0.0,
        "en_class": 0.0,
        "mask": 0.0,
        "mask_gt": 0.0,
        "mask_pred": 0.0,
        "mask_dice": 0.0,
        "mask_bce": 0.0,
        "mask_centroid": 0.0,
        "mask_outside": 0.0,
        "mask_area": 0.0,
        "mask_max_area": 0.0,
        "mask_pred_iou": 0.0,
        "mask_stability": 0.0,
        "mask_prompts": 0.0,
        "mask_gt_prompts": 0.0,
        "mask_pred_prompts": 0.0,
    }
    count = 0
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    debug_start = int(debug_batch_start)
    debug_end = int(debug_batch_end) if int(debug_batch_end) >= 0 else debug_start
    debug_enabled = debug_start >= 0

    def _sync_debug(active: bool) -> None:
        if active and device.type == "cuda":
            torch.cuda.synchronize(device)

    def _mem_debug() -> str:
        if device.type != "cuda":
            return ""
        allocated = torch.cuda.memory_allocated(device) / (1024.0**3)
        reserved = torch.cuda.memory_reserved(device) / (1024.0**3)
        peak = torch.cuda.max_memory_allocated(device) / (1024.0**3)
        return f" mem_alloc={allocated:.2f}GiB mem_reserved={reserved:.2f}GiB mem_peak={peak:.2f}GiB"

    def _phase_debug(active: bool, label: str, seconds: float) -> None:
        if active:
            print(
                f"[debug-batch][rank={rank}] epoch={epoch_index} batch={batch_idx} "
                f"{label}={float(seconds):.4f}s{_mem_debug()}",
                flush=True,
            )

    def _batch_debug_summary(batch_obj: Dict[str, object]) -> str:
        names = batch_obj.get("name", [])
        tiles = batch_obj.get("tile_name", [])
        x0 = batch_obj.get("x0", [])
        y0 = batch_obj.get("y0", [])
        parts: List[str] = []
        if isinstance(names, Sequence):
            parts.append(f"names={list(names)}")
        if isinstance(tiles, Sequence):
            parts.append(f"tiles={list(tiles)}")
        if isinstance(x0, Sequence) and isinstance(y0, Sequence):
            parts.append(f"xy0={list(zip(list(x0), list(y0)))}")
        band_centers = batch_obj.get("band_centers")
        if isinstance(band_centers, Sequence):
            counts: List[List[int]] = []
            for item in band_centers:
                if isinstance(item, Sequence):
                    counts.append([int(getattr(centers, "shape", [0])[0]) for centers in item])
            parts.append(f"gt_band_counts={counts}")
        return " ".join(parts)

    data_wait_start = time.perf_counter()
    prompt_ratio = prompt_pred_ratio(epoch_index, weights)
    effective_mask_outer_weight = mask_outer_weight_for_epoch(epoch_index, weights)
    progress = tqdm(loader, desc="train" if training else "eval", leave=False, disable=not show_progress)
    for batch_idx, batch in enumerate(progress):
        debug_this_batch = debug_enabled and debug_start <= int(batch_idx) <= debug_end
        debug_print = debug_this_batch and (bool(debug_batch_all_ranks) or rank == 0)
        t_after_data = time.perf_counter()
        if debug_this_batch and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if debug_print:
            print(
                f"[debug-batch][rank={rank}] epoch={epoch_index} batch={batch_idx} "
                f"data_wait={t_after_data - data_wait_start:.4f}s {_batch_debug_summary(batch)}",
                flush=True,
            )
        if optimizer is not None and show_progress:
            head_group = next(
                (group for group in optimizer.param_groups if str(group.get("name", "")) == "head"),
                optimizer.param_groups[0],
            )
            progress.set_postfix(head_lr=f"{float(head_group['lr']) / 1e-4:.3g}*1e-4", refresh=False)
        t0 = time.perf_counter()
        image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        _sync_debug(debug_this_batch)
        t_transfer = time.perf_counter()
        _phase_debug(debug_print, "h2d", t_transfer - t0)
        with _autocast_context(device, amp_dtype):
            outputs = model(image)
            _sync_debug(debug_this_batch)
            t_forward = time.perf_counter()
            _phase_debug(debug_print, "forward", t_forward - t_transfer)
            losses = dense_losses_any(outputs, batch, weights=weights, device=device, center_radius_px=center_radius_px)
            _sync_debug(debug_this_batch)
            t_dense_loss = time.perf_counter()
            _phase_debug(debug_print, "dense_loss", t_dense_loss - t_forward)
            total = losses["total"]
            mask_losses = sam_prompt_mask_losses(
                base_model,
                outputs,
                batch,
                weights=weights,
                device=device,
                epoch_index=epoch_index,
                threshold=threshold,
                nms_radius=nms_radius,
                confidence_score=confidence_score,
                use_ordinal_expectation=use_ordinal_expectation,
                debug_ordinal_expectation=debug_ordinal_expectation,
                center_refinement=center_refinement,
                center_refinement_radius=center_refinement_radius,
                ellipse_sigma=ellipse_sigma,
                detect_centers_fn=detect_centers,
            )
            _sync_debug(debug_this_batch)
            t_mask_loss = time.perf_counter()
            _phase_debug(debug_print, "mask_loss", t_mask_loss - t_dense_loss)
            total = total + float(effective_mask_outer_weight) * mask_losses["total"]
            if training and "embedding" in outputs:
                # Keep the embedding branch visible to DDP when triplet/EX/EN losses are disabled.
                total = total + outputs["embedding"].sum() * 0.0

            triplet = total.new_tensor(0.0)
            if triplet_enabled:
                triplet = embedding_triplet_loss(
                    outputs,
                    batch,
                    loss_fn=triplet_loss_fn,
                    max_sources_per_group=triplet_max_sources_per_group,
                    negative_scope=triplet_negative_scope,
                )
                total = total + weights.triplet_outer_weight * triplet

            ex_class = total.new_tensor(0.0)
            if ex_enabled and hasattr(base_model, "EX") and image.shape[1] > 1:
                ex_class = vectorized_matcher_classification_loss(
                    base_model.EX,
                    outputs,
                    batch,
                    mode="ex",
                    candidate_count=matcher_candidate_count,
                    offset_scale=center_radius_px,
                    band_pairs=ex_band_pairs,
                    max_anchors_per_band=matcher_max_anchors_per_band,
                )
                total = total + weights.matcher_outer_weight * ex_class

            en_class = total.new_tensor(0.0)
            if en_enabled and hasattr(base_model, "EN"):
                en_class = vectorized_matcher_classification_loss(
                    base_model.EN,
                    outputs,
                    batch,
                    mode="en",
                    candidate_count=matcher_candidate_count,
                    offset_scale=center_radius_px,
                    max_anchors_per_band=matcher_max_anchors_per_band,
                )
                total = total + weights.matcher_outer_weight * en_class

        _sync_debug(debug_this_batch)
        t_extra_loss = time.perf_counter()
        _phase_debug(debug_print, "extra_loss", t_extra_loss - t_mask_loss)
        if training:
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            if freeze_sam_proposal:
                clear_sam_proposal_grads(base_model)
            if freeze_sam_encoder:
                clear_sam_encoder_grads(base_model)
            _sync_debug(debug_this_batch)
            t_backward = time.perf_counter()
            _phase_debug(debug_print, "backward", t_backward - t_extra_loss)
            optimizer.step()
            if scheduler_step is not None:
                scheduler_step()
            _sync_debug(debug_this_batch)
            t_step = time.perf_counter()
            _phase_debug(debug_print, "step", t_step - t_backward)
        else:
            t_backward = t_extra_loss
            t_step = t_extra_loss

        batch_size = int(image.shape[0])
        count += batch_size
        sums["total"] += float(total.detach()) * batch_size
        sums["seg"] += float(losses["seg"].detach()) * batch_size
        sums["confidence"] += float(losses["confidence"].detach()) * batch_size
        sums["shape"] += float(losses["shape"].detach()) * batch_size
        sums["center"] += float(losses["center"].detach()) * batch_size
        sums["triplet"] += float(triplet.detach()) * batch_size
        sums["ex_class"] += float(ex_class.detach()) * batch_size
        sums["en_class"] += float(en_class.detach()) * batch_size
        sums["mask"] += float(mask_losses["total"].detach()) * batch_size
        sums["mask_gt"] += float(mask_losses["gt_total"].detach()) * batch_size
        sums["mask_pred"] += float(mask_losses["pred_total"].detach()) * batch_size
        sums["mask_dice"] += float(mask_losses["dice"].detach()) * batch_size
        sums["mask_bce"] += float(mask_losses["bce"].detach()) * batch_size
        sums["mask_centroid"] += float(mask_losses["centroid"].detach()) * batch_size
        sums["mask_outside"] += float(mask_losses["outside"].detach()) * batch_size
        sums["mask_area"] += float(mask_losses["area"].detach()) * batch_size
        sums["mask_max_area"] += float(mask_losses["max_area"].detach()) * batch_size
        sums["mask_pred_iou"] += float(mask_losses["pred_iou"].detach()) * batch_size
        sums["mask_stability"] += float(mask_losses["stability"].detach()) * batch_size
        sums["mask_prompts"] += float(mask_losses["prompts"].detach()) * batch_size
        sums["mask_gt_prompts"] += float(mask_losses["gt_prompts"].detach()) * batch_size
        sums["mask_pred_prompts"] += float(mask_losses["pred_prompts"].detach()) * batch_size
        current_step = int(global_step_start) + int(batch_idx) + 1
        should_log_iteration = (
            training
            and iteration_log_fn is not None
            and int(iteration_log_interval) > 0
            and (current_step % int(iteration_log_interval) == 0)
        )
        if should_log_iteration:
            local_metrics = {
                "loss/total": float(total.detach()),
            }
            if float(weights.confidence_outer_weight) > 0.0:
                local_metrics["loss/confidence"] = float(losses["confidence"].detach())
            if float(weights.segmentation_outer_weight) > 0.0:
                local_metrics["loss/seg"] = float(losses["seg"].detach())
            if float(weights.shape_outer_weight) > 0.0:
                local_metrics["loss/shape"] = float(losses["shape"].detach())
            if float(weights.center_position) > 0.0:
                local_metrics["loss/center"] = float(losses["center"].detach())
            if triplet_enabled and float(weights.triplet_outer_weight) > 0.0:
                local_metrics["loss/triplet"] = float(triplet.detach())
            if ex_enabled and float(weights.matcher_outer_weight) > 0.0:
                local_metrics["loss/ex_class"] = float(ex_class.detach())
            if en_enabled and float(weights.matcher_outer_weight) > 0.0:
                local_metrics["loss/en_class"] = float(en_class.detach())
            active_mask_keys = active_mask_loss_keys(weights)
            local_metrics["loss/mask_outer_weight_effective"] = float(effective_mask_outer_weight)
            if float(effective_mask_outer_weight) > 0.0 and active_mask_keys:
                local_metrics["loss/mask"] = float(mask_losses["total"].detach())
                local_metrics["loss/mask_weighted"] = float((float(effective_mask_outer_weight) * mask_losses["total"]).detach())
                local_metrics["loss/mask_gt"] = float(mask_losses["gt_total"].detach())
                local_metrics["loss/mask_pred"] = float(mask_losses["pred_total"].detach())
                for key in active_mask_keys:
                    local_metrics[f"loss/mask_{key}"] = float(mask_losses[key].detach())
                local_metrics["loss/mask_prompts"] = float(mask_losses["prompts"].detach())
                local_metrics["loss/mask_gt_prompts"] = float(mask_losses["gt_prompts"].detach())
                local_metrics["loss/mask_pred_prompts"] = float(mask_losses["pred_prompts"].detach())
            if distributed and dist.is_available() and dist.is_initialized():
                metric_keys = list(local_metrics.keys())
                packed = torch.tensor(
                    [local_metrics[key] * float(batch_size) for key in metric_keys] + [float(batch_size)],
                    device=device,
                    dtype=torch.float64,
                )
                dist.all_reduce(packed, op=dist.ReduceOp.SUM)
                denom = float(packed[-1].item()) if float(packed[-1].item()) > 0.0 else 1.0
                local_metrics = {
                    key: float(packed[idx].item()) / denom
                    for idx, key in enumerate(metric_keys)
                }
            if optimizer is not None:
                for group_idx, group in enumerate(optimizer.param_groups):
                    name = str(group.get("name", group_idx))
                    local_metrics[f"lr/{name}"] = float(group["lr"])
                local_metrics["lr"] = float(optimizer.param_groups[0]["lr"])
            local_metrics["prompt/predicted_ratio"] = float(prompt_ratio)
            local_metrics["prompt/gt_ratio"] = float(1.0 - prompt_ratio)
            local_metrics["proposal/frozen"] = float(bool(freeze_sam_proposal))
            local_metrics["proposal/confidence_loss_weight"] = float(weights.confidence_outer_weight)
            local_metrics["proposal/shape_loss_weight"] = float(weights.shape_outer_weight)
            local_metrics["proposal/center_loss_weight"] = float(weights.center_position)
            local_metrics["epoch"] = float(epoch_index)
            local_metrics["iteration"] = float(current_step)
            iteration_log_fn(local_metrics, current_step)
        _sync_debug(debug_this_batch)
        t_done = time.perf_counter()
        if debug_print:
            print(
                f"[debug-batch][rank={rank}] epoch={epoch_index} batch={batch_idx} "
                f"h2d={t_transfer - t0:.4f}s forward={t_forward - t_transfer:.4f}s "
                f"dense_loss={t_dense_loss - t_forward:.4f}s mask_loss={t_mask_loss - t_dense_loss:.4f}s "
                f"extra_loss={t_extra_loss - t_mask_loss:.4f}s backward={t_backward - t_extra_loss:.4f}s "
                f"step={t_step - t_backward:.4f}s post={t_done - t_step:.4f}s total_iter={t_done - t_after_data:.4f}s "
                f"loss={float(total.detach()):.6g} mask={float(mask_losses['total'].detach()):.6g} "
                f"mask_prompts={float(mask_losses['prompts'].detach()):.1f}{_mem_debug()}",
                flush=True,
            )
        data_wait_start = time.perf_counter()
    if distributed and dist.is_available() and dist.is_initialized():
        keys = list(sums.keys())
        packed = torch.tensor([sums[key] for key in keys] + [float(count)], device=device, dtype=torch.float64)
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        sums = {key: float(packed[idx].item()) for idx, key in enumerate(keys)}
        count = int(packed[-1].item())
    # print(f"[DEBUG] Epoch completed in {time.time() - t_epoch_start:.3f} seconds.")
    return {key: val / max(count, 1) for key, val in sums.items()}


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    weights: LossWeights,
    triplet_loss_fn: HardTripletLoss,
    triplet_enabled: bool,
    ex_enabled: bool,
    en_enabled: bool,
    matcher_candidate_count: int,
    matcher_max_anchors_per_band: int,
    triplet_max_sources_per_group: int,
    triplet_negative_scope: str,
    ex_band_pairs: Optional[Sequence[Tuple[int, int]]],
    center_radius_px: float,
    compute_detection: bool,
    threshold: float,
    nms_radius: int,
    confidence_score: str,
    use_ordinal_expectation: bool,
    debug_ordinal_expectation: bool,
    center_refinement: str,
    center_refinement_radius: int,
    use_en_postprocess: bool,
    en_threshold: float,
    use_ex_link_postprocess: bool,
    ex_link_threshold: float,
    band_names: Sequence[str],
    collect_candidate_stats: bool = False,
    ignore_mask_during_detection: bool = True,
    epoch_index: int = 0,
    ellipse_sigma: float = 2.0,
    amp_dtype: Optional[torch.dtype] = None,
    distributed: bool = False,
    show_progress: bool = True,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    model.eval()
    base_model = unwrap_model(model)
    sums: Dict[str, float] = {
        "total": 0.0,
        "seg": 0.0,
        "confidence": 0.0,
        "shape": 0.0,
        "center": 0.0,
        "triplet": 0.0,
        "ex_class": 0.0,
        "en_class": 0.0,
        "mask": 0.0,
        "mask_gt": 0.0,
        "mask_pred": 0.0,
        "mask_dice": 0.0,
        "mask_bce": 0.0,
        "mask_centroid": 0.0,
        "mask_outside": 0.0,
        "mask_area": 0.0,
        "mask_max_area": 0.0,
        "mask_pred_iou": 0.0,
        "mask_stability": 0.0,
        "mask_prompts": 0.0,
        "mask_gt_prompts": 0.0,
        "mask_pred_prompts": 0.0,
    }
    count = 0
    det_totals = _init_detection_totals(band_names, collect_candidate_stats=collect_candidate_stats) if compute_detection else None
    effective_mask_outer_weight = mask_outer_weight_for_epoch(epoch_index, weights)
    desc = "val+detect" if compute_detection else "val"
    for batch in tqdm(loader, desc=desc, leave=False, disable=not show_progress):
        image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        with _autocast_context(device, amp_dtype):
            outputs = model(image)
            losses = dense_losses_any(outputs, batch, weights=weights, device=device, center_radius_px=center_radius_px)
            total = losses["total"]
            mask_losses = sam_prompt_mask_losses(
                base_model,
                outputs,
                batch,
                weights=weights,
                device=device,
                epoch_index=epoch_index,
                threshold=threshold,
                nms_radius=nms_radius,
                confidence_score=confidence_score,
                use_ordinal_expectation=use_ordinal_expectation,
                debug_ordinal_expectation=debug_ordinal_expectation,
                center_refinement=center_refinement,
                center_refinement_radius=center_refinement_radius,
                ellipse_sigma=ellipse_sigma,
                detect_centers_fn=detect_centers,
            )
            total = total + float(effective_mask_outer_weight) * mask_losses["total"]

            triplet = total.new_tensor(0.0)
            if triplet_enabled:
                triplet = embedding_triplet_loss(
                    outputs,
                    batch,
                    loss_fn=triplet_loss_fn,
                    max_sources_per_group=triplet_max_sources_per_group,
                    negative_scope=triplet_negative_scope,
                )
                total = total + weights.triplet_outer_weight * triplet

            ex_class = total.new_tensor(0.0)
            if ex_enabled and hasattr(base_model, "EX") and image.shape[1] > 1:
                ex_class = vectorized_matcher_classification_loss(
                    base_model.EX,
                    outputs,
                    batch,
                    mode="ex",
                    candidate_count=matcher_candidate_count,
                    offset_scale=center_radius_px,
                    band_pairs=ex_band_pairs,
                    max_anchors_per_band=matcher_max_anchors_per_band,
                )
                total = total + weights.matcher_outer_weight * ex_class

            en_class = total.new_tensor(0.0)
            if en_enabled and hasattr(base_model, "EN"):
                en_class = vectorized_matcher_classification_loss(
                    base_model.EN,
                    outputs,
                    batch,
                    mode="en",
                    candidate_count=matcher_candidate_count,
                    offset_scale=center_radius_px,
                    max_anchors_per_band=matcher_max_anchors_per_band,
                )
                total = total + weights.matcher_outer_weight * en_class

        if det_totals is not None:
            detection_outputs = {
                key: value.float() if torch.is_floating_point(value) else value
                for key, value in outputs.items()
            }
            _update_detection_totals(
                det_totals,
                base_model,
                detection_outputs,
                batch,
                threshold=threshold,
                nms_radius=nms_radius,
                confidence_score=confidence_score,
                use_ordinal_expectation=use_ordinal_expectation,
                debug_ordinal_expectation=debug_ordinal_expectation,
                center_refinement=center_refinement,
                center_refinement_radius=center_refinement_radius,
                match_radius=center_radius_px,
                use_en_postprocess=use_en_postprocess,
                en_candidate_count=matcher_candidate_count,
                en_threshold=en_threshold,
                use_ex_link_postprocess=use_ex_link_postprocess,
                ex_link_threshold=ex_link_threshold,
                ex_band_pairs=ex_band_pairs,
                band_names=band_names,
                collect_candidate_stats=collect_candidate_stats,
                ignore_mask_during_detection=ignore_mask_during_detection,
            )

        batch_size = int(image.shape[0])
        count += batch_size
        sums["total"] += float(total.detach()) * batch_size
        sums["seg"] += float(losses["seg"].detach()) * batch_size
        sums["confidence"] += float(losses["confidence"].detach()) * batch_size
        sums["shape"] += float(losses["shape"].detach()) * batch_size
        sums["center"] += float(losses["center"].detach()) * batch_size
        sums["triplet"] += float(triplet.detach()) * batch_size
        sums["ex_class"] += float(ex_class.detach()) * batch_size
        sums["en_class"] += float(en_class.detach()) * batch_size
        sums["mask"] += float(mask_losses["total"].detach()) * batch_size
        sums["mask_gt"] += float(mask_losses["gt_total"].detach()) * batch_size
        sums["mask_pred"] += float(mask_losses["pred_total"].detach()) * batch_size
        sums["mask_dice"] += float(mask_losses["dice"].detach()) * batch_size
        sums["mask_bce"] += float(mask_losses["bce"].detach()) * batch_size
        sums["mask_centroid"] += float(mask_losses["centroid"].detach()) * batch_size
        sums["mask_outside"] += float(mask_losses["outside"].detach()) * batch_size
        sums["mask_area"] += float(mask_losses["area"].detach()) * batch_size
        sums["mask_max_area"] += float(mask_losses["max_area"].detach()) * batch_size
        sums["mask_pred_iou"] += float(mask_losses["pred_iou"].detach()) * batch_size
        sums["mask_stability"] += float(mask_losses["stability"].detach()) * batch_size
        sums["mask_prompts"] += float(mask_losses["prompts"].detach()) * batch_size
        sums["mask_gt_prompts"] += float(mask_losses["gt_prompts"].detach()) * batch_size
        sums["mask_pred_prompts"] += float(mask_losses["pred_prompts"].detach()) * batch_size

    if distributed and dist.is_available() and dist.is_initialized():
        keys = list(sums.keys())
        packed = torch.tensor([sums[key] for key in keys] + [float(count)], device=device, dtype=torch.float64)
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        sums = {key: float(packed[idx].item()) for idx, key in enumerate(keys)}
        count = int(packed[-1].item())
        if det_totals is not None:
            gathered: List[Optional[Dict[str, object]]] = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered, det_totals)
            det_totals = _merge_detection_totals(
                [item for item in gathered if isinstance(item, dict)],
                band_names,
            )

    dense = {key: val / max(count, 1) for key, val in sums.items()}
    det = (
        _finalize_detection_totals(
            det_totals,
            band_names=band_names,
            use_ex_link_postprocess=use_ex_link_postprocess,
        )
        if det_totals is not None
        else {}
    )
    return dense, det


@torch.no_grad()
def evaluate_detection(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    threshold: float,
    nms_radius: int,
    confidence_score: str,
    use_ordinal_expectation: bool = False,
    debug_ordinal_expectation: bool = False,
    match_radius: float,
    center_refinement: str = "integer",
    center_refinement_radius: int = 1,
    use_en_postprocess: bool = False,
    en_candidate_count: int = 5,
    en_threshold: float = 0.6,
    use_ex_link_postprocess: bool = False,
    ex_link_threshold: float = 0.5,
    ex_band_pairs: Optional[Sequence[Tuple[int, int]]] = None,
    band_names: Sequence[str] = (),
    collect_candidate_stats: bool = False,
    show_progress: bool = True,
) -> Dict[str, object]:
    model.eval()
    base_model = unwrap_model(model)
    totals = _init_detection_totals(band_names, collect_candidate_stats=collect_candidate_stats)
    for batch in tqdm(loader, desc="detect", leave=False, disable=not show_progress):
        image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        outputs = model(image)
        _update_detection_totals(
            totals,
            base_model,
            outputs,
            batch,
            threshold=threshold,
            nms_radius=nms_radius,
            confidence_score=confidence_score,
            use_ordinal_expectation=use_ordinal_expectation,
            debug_ordinal_expectation=debug_ordinal_expectation,
            center_refinement=center_refinement,
            center_refinement_radius=center_refinement_radius,
            match_radius=match_radius,
            use_en_postprocess=use_en_postprocess,
            en_candidate_count=en_candidate_count,
            en_threshold=en_threshold,
            use_ex_link_postprocess=use_ex_link_postprocess,
            ex_link_threshold=ex_link_threshold,
            ex_band_pairs=ex_band_pairs,
            band_names=band_names,
            collect_candidate_stats=collect_candidate_stats,
        )
    return _finalize_detection_totals(
        totals,
        band_names=band_names,
        use_ex_link_postprocess=use_ex_link_postprocess,
    )
