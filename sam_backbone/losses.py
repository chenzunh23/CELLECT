from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from utils.train_ops_utils import flatten_per_band_outputs as _flatten_per_band_outputs


def prompt_pred_ratio(epoch_index: int, weights: object) -> float:
    start = int(getattr(weights, "mask_prompt_gt_epochs"))
    end = int(getattr(weights, "mask_prompt_pred_epoch"))
    if int(epoch_index) < start:
        return 0.0
    if int(epoch_index) >= end:
        return 1.0
    return float(int(epoch_index) - start) / max(float(end - start), 1.0)


def mask_outer_weight_for_epoch(epoch_index: int, weights: object) -> float:
    """Return the effective outer mask-loss weight for this epoch.

    ``mask_loss_warmup_epochs=N`` disables SAM mask decoder supervision for
    epochs ``0..N-1`` and enables the configured mask weight from epoch ``N``.
    This is intentionally epoch-derived so checkpoints do not need optimizer
    state to recover the transition behavior.
    """

    warmup_epochs = max(0, int(getattr(weights, "mask_loss_warmup_epochs", 0)))
    if int(epoch_index) < warmup_epochs:
        return 0.0
    return float(getattr(weights, "mask_outer_weight", 0.0))


def _prompt_sample_limit(candidate_count: int, max_per_sample: int, fraction: float) -> int:
    if int(candidate_count) <= 0 or float(fraction) <= 0.0:
        return 0
    base = int(candidate_count)
    if int(max_per_sample) > 0:
        base = min(base, int(max_per_sample))
    if float(fraction) >= 1.0:
        return base
    return max(1, min(base, int(round(float(base) * float(fraction)))))


def _random_subset_rows(tensor: Tensor, count: int) -> Tensor:
    if int(count) <= 0:
        return tensor[:0]
    if tensor.shape[0] <= int(count):
        return tensor
    indices = torch.randperm(tensor.shape[0], device=tensor.device)[: int(count)]
    return tensor[indices]


def _sample_chw_nearest(map_chw: Tensor, centers_xy: Tensor) -> Tensor:
    if centers_xy.numel() == 0:
        return map_chw.new_zeros((0, int(map_chw.shape[0])))
    h, w = int(map_chw.shape[-2]), int(map_chw.shape[-1])
    x = centers_xy[:, 0].round().long().clamp(0, w - 1)
    y = centers_xy[:, 1].round().long().clamp(0, h - 1)
    return map_chw[:, y, x].transpose(0, 1)


def _sample_hw_nearest(map_hw: Tensor, centers_xy: Tensor) -> Tensor:
    if centers_xy.numel() == 0:
        return map_hw.new_zeros((0,))
    h, w = int(map_hw.shape[-2]), int(map_hw.shape[-1])
    x = centers_xy[:, 0].round().long().clamp(0, w - 1)
    y = centers_xy[:, 1].round().long().clamp(0, h - 1)
    return map_hw[y, x]


def _boxes_from_centers_shapes(
    centers_xy: Tensor,
    shapes: Tensor,
    *,
    image_size: Tuple[int, int],
    ellipse_sigma: float,
) -> Tensor:
    if centers_xy.numel() == 0:
        return centers_xy.new_zeros((0, 4))
    h, w = int(image_size[0]), int(image_size[1])
    major = shapes[:, 0].abs().clamp_min(1.0) * float(ellipse_sigma)
    minor = shapes[:, 1].abs().clamp_min(1.0) * float(ellipse_sigma)
    theta = shapes[:, 2] if shapes.shape[1] >= 3 else shapes.new_zeros((shapes.shape[0],))
    cos_t = torch.cos(theta).abs()
    sin_t = torch.sin(theta).abs()
    half_w = torch.sqrt((major * cos_t) ** 2 + (minor * sin_t) ** 2).clamp_min(1.5)
    half_h = torch.sqrt((major * sin_t) ** 2 + (minor * cos_t) ** 2).clamp_min(1.5)
    x0 = (centers_xy[:, 0] - half_w).clamp(0, w - 1)
    y0 = (centers_xy[:, 1] - half_h).clamp(0, h - 1)
    x1 = (centers_xy[:, 0] + half_w).clamp(0, w - 1)
    y1 = (centers_xy[:, 1] + half_h).clamp(0, h - 1)
    return torch.stack([x0, y0, x1, y1], dim=1)


def _ellipse_targets_lowres(
    centers_xy: Tensor,
    shapes: Tensor,
    *,
    mask_hw: Tuple[int, int],
    image_hw: Tuple[int, int],
    ellipse_sigma: float,
) -> Tensor:
    n = int(centers_xy.shape[0])
    mh, mw = int(mask_hw[0]), int(mask_hw[1])
    if n == 0:
        return centers_xy.new_zeros((0, mh, mw))
    scale_x = float(mw) / float(image_hw[1])
    scale_y = float(mh) / float(image_hw[0])
    yy, xx = torch.meshgrid(
        torch.arange(mh, device=centers_xy.device, dtype=centers_xy.dtype),
        torch.arange(mw, device=centers_xy.device, dtype=centers_xy.dtype),
        indexing="ij",
    )
    cx = centers_xy[:, 0] * scale_x
    cy = centers_xy[:, 1] * scale_y
    major = shapes[:, 0].abs().clamp_min(1.0) * float(ellipse_sigma) * scale_x
    minor = shapes[:, 1].abs().clamp_min(1.0) * float(ellipse_sigma) * scale_y
    theta = shapes[:, 2] if shapes.shape[1] >= 3 else shapes.new_zeros((n,))
    dx = xx.unsqueeze(0) - cx[:, None, None]
    dy = yy.unsqueeze(0) - cy[:, None, None]
    cos_t = torch.cos(theta)[:, None, None]
    sin_t = torch.sin(theta)[:, None, None]
    xr = cos_t * dx + sin_t * dy
    yr = -sin_t * dx + cos_t * dy
    dist = (xr / major[:, None, None].clamp_min(1e-3)) ** 2 + (yr / minor[:, None, None].clamp_min(1e-3)) ** 2
    return (dist <= 1.0).to(dtype=centers_xy.dtype)


def _calculate_stability_score(logits: Tensor, *, mask_threshold: float = 0.0, threshold_offset: float = 1.0) -> Tensor:
    """SAM AMG stability score: IoU between high/low logit thresholds."""

    intersections = (logits > (float(mask_threshold) + float(threshold_offset))).sum(dim=(-1, -2))
    unions = (logits > (float(mask_threshold) - float(threshold_offset))).sum(dim=(-1, -2))
    return intersections.to(dtype=logits.dtype) / unions.to(dtype=logits.dtype).clamp_min(1.0)


def _soft_stability_score(
    logits: Tensor,
    *,
    mask_threshold: float = 0.0,
    threshold_offset: float = 1.0,
    temperature: float = 10.0,
) -> Tensor:
    """Differentiable surrogate for SAM AMG's stability score."""

    temp = float(temperature)
    high = torch.sigmoid(temp * (logits - (float(mask_threshold) + float(threshold_offset))))
    low = torch.sigmoid(temp * (logits - (float(mask_threshold) - float(threshold_offset))))
    return high.sum(dim=(-1, -2)) / low.sum(dim=(-1, -2)).clamp_min(1e-6)


def _zero_mask_losses(anchor: Tensor) -> Dict[str, Tensor]:
    zero = anchor.sum() * 0.0
    return {
        "total": zero,
        "gt_total": zero,
        "pred_total": zero,
        "dice": zero,
        "bce": zero,
        "centroid": zero,
        "outside": zero,
        "area": zero,
        "max_area": zero,
        "pred_iou": zero,
        "stability": zero,
        "prompts": zero,
        "gt_prompts": zero,
        "pred_prompts": zero,
    }


def _any_mask_component_enabled(weights: object) -> bool:
    return any(
        float(getattr(weights, name, default)) > 0.0
        for name, default in (
            ("mask_dice", 0.0),
            ("mask_bce", 0.0),
            ("mask_centroid", 0.0),
            ("mask_outside", 0.0),
            ("mask_min_area", 0.0),
            ("mask_max_area", 0.1),
            ("mask_pred_iou", 0.1),
            ("mask_stability", 0.1),
        )
    )


def _build_gt_prompt_tensors(
    outputs: Dict[str, Tensor],
    batch: Dict[str, object],
    *,
    device: torch.device,
    max_gt_per_sample: int,
    sample_fraction: float,
) -> Optional[Dict[str, Tensor]]:
    confidence = outputs["confidence"]
    if confidence.ndim != 5:
        return None
    batch_size, band_count = int(confidence.shape[0]), int(confidence.shape[1])
    prompt_batch: List[Tensor] = []
    centers_all: List[Tensor] = []
    shapes_all: List[Tensor] = []
    weights_all: List[Tensor] = []
    band_centers = batch["band_centers"]  # type: ignore[index]
    band_shape = batch["band_shape"].to(device=device, dtype=torch.float32)  # type: ignore[union-attr]
    band_center_only = batch["band_center_only_mask"].to(device=device, dtype=torch.bool)  # type: ignore[union-attr]
    for b in range(batch_size):
        for band in range(band_count):
            centers = band_centers[b][band].to(device=device, dtype=torch.float32)
            if centers.numel() == 0:
                continue
            keep_count = _prompt_sample_limit(centers.shape[0], int(max_gt_per_sample), float(sample_fraction))
            centers = _random_subset_rows(centers, keep_count)
            if centers.numel() == 0:
                continue
            flat_idx = b * band_count + band
            shapes = _sample_chw_nearest(band_shape[b, band], centers)
            center_only = _sample_hw_nearest(band_center_only[b, band], centers).to(dtype=torch.float32)
            weights = torch.where(
                center_only > 0,
                centers.new_full((centers.shape[0],), 0.2),
                centers.new_ones((centers.shape[0],)),
            )
            prompt_batch.append(torch.full((centers.shape[0],), flat_idx, device=device, dtype=torch.long))
            centers_all.append(centers)
            shapes_all.append(shapes)
            weights_all.append(weights)
    if not centers_all:
        return None
    centers = torch.cat(centers_all, dim=0)
    shapes = torch.cat(shapes_all, dim=0)
    return {
        "batch_indices": torch.cat(prompt_batch, dim=0),
        "centers": centers,
        "prompt_shapes": shapes,
        "target_shapes": shapes,
        "weights": torch.cat(weights_all, dim=0),
        "mask_target_weights": centers.new_ones((centers.shape[0],)),
    }


def _build_pred_prompt_tensors(
    outputs: Dict[str, Tensor],
    batch: Dict[str, object],
    *,
    device: torch.device,
    threshold: float,
    nms_radius: int,
    confidence_score: str,
    use_ordinal_expectation: bool,
    debug_ordinal_expectation: bool,
    center_refinement: str,
    center_refinement_radius: int,
    max_pred_per_sample: int,
    sample_fraction: float,
    unmatched_weight: float,
    detach_prompt_shapes: bool,
    detect_centers_fn: Callable[..., List[Tensor | np.ndarray]],
    debug_timer: Optional[Callable[[str], None]] = None,
) -> Optional[Dict[str, Tensor]]:
    if outputs["confidence"].ndim != 5:
        return None
    flat_outputs = _flatten_per_band_outputs({"confidence": outputs["confidence"], "shape": outputs["shape"]})
    pred_list = detect_centers_fn(
        flat_outputs,
        threshold=threshold,
        nms_radius=nms_radius,
        confidence_score=confidence_score,
        use_ordinal_expectation=use_ordinal_expectation,
        debug_ordinal_expectation=debug_ordinal_expectation,
        center_refinement=center_refinement,
        center_refinement_radius=center_refinement_radius,
        merge_close_centers=False,
    )
    if debug_timer is not None:
        debug_timer("mask.pred.detect_centers")
    flat_shape_pred = flat_outputs["shape"]
    flat_clean = batch["band_clean_mask"].reshape(-1, *batch["band_clean_mask"].shape[2:]).to(device=device, dtype=torch.bool)  # type: ignore[union-attr]
    flat_shape_target = batch["band_shape"].reshape(-1, *batch["band_shape"].shape[2:]).to(device=device, dtype=torch.float32)  # type: ignore[union-attr]
    prompt_batch: List[Tensor] = []
    centers_all: List[Tensor] = []
    prompt_shapes_all: List[Tensor] = []
    target_shapes_all: List[Tensor] = []
    weights_all: List[Tensor] = []
    mask_target_weights_all: List[Tensor] = []
    for flat_idx, pred_item in enumerate(pred_list):
        if isinstance(pred_item, Tensor):
            centers = pred_item.to(device=device, dtype=torch.float32).reshape(-1, 2)
        else:
            centers = torch.as_tensor(np.asarray(pred_item, dtype=np.float32), device=device).reshape(-1, 2)
        if centers.numel() == 0:
            continue
        keep_count = _prompt_sample_limit(centers.shape[0], int(max_pred_per_sample), float(sample_fraction))
        centers = _random_subset_rows(centers, keep_count)
        if centers.numel() == 0:
            continue
        prompt_shapes = _sample_chw_nearest(flat_shape_pred[flat_idx], centers)
        if bool(detach_prompt_shapes):
            prompt_shapes = prompt_shapes.detach()
        clean_at_center = _sample_hw_nearest(flat_clean[flat_idx], centers).to(dtype=torch.bool)
        target_shapes_clean = _sample_chw_nearest(flat_shape_target[flat_idx], centers)
        target_shapes = torch.where(clean_at_center[:, None], target_shapes_clean, prompt_shapes.detach())
        weights = torch.where(
            clean_at_center,
            centers.new_ones((centers.shape[0],)),
            centers.new_full((centers.shape[0],), float(unmatched_weight)),
        )
        prompt_batch.append(torch.full((centers.shape[0],), flat_idx, device=device, dtype=torch.long))
        centers_all.append(centers)
        prompt_shapes_all.append(prompt_shapes)
        target_shapes_all.append(target_shapes)
        weights_all.append(weights)
        mask_target_weights_all.append(clean_at_center.to(dtype=torch.float32))
    if not centers_all:
        return None
    if debug_timer is not None:
        debug_timer("mask.pred.pack_prompts")
    return {
        "batch_indices": torch.cat(prompt_batch, dim=0),
        "centers": torch.cat(centers_all, dim=0),
        "prompt_shapes": torch.cat(prompt_shapes_all, dim=0),
        "target_shapes": torch.cat(target_shapes_all, dim=0),
        "weights": torch.cat(weights_all, dim=0),
        "mask_target_weights": torch.cat(mask_target_weights_all, dim=0),
    }


def _sam_mask_loss_for_prompts(
    model: nn.Module,
    outputs: Dict[str, Tensor],
    prompts: Dict[str, Tensor],
    *,
    weights: object,
    image_hw: Tuple[int, int],
    ellipse_sigma: float,
    debug_timer: Optional[Callable[[str], None]] = None,
    debug_prefix: str = "mask",
) -> Dict[str, Tensor]:
    if prompts["centers"].numel() == 0:
        return _zero_mask_losses(outputs["confidence"])
    boxes = None
    if not bool(getattr(weights, "mask_prompt_center_only", False)):
        boxes = _boxes_from_centers_shapes(
            prompts["centers"],
            prompts["prompt_shapes"],
            image_size=image_hw,
            ellipse_sigma=ellipse_sigma,
        )
    if debug_timer is not None:
        debug_timer(f"{debug_prefix}.boxes")
    masks, iou_predictions = model.forward_sam_masks(
        outputs["image_embeddings"],
        prompts["batch_indices"],
        prompts["centers"],
        boxes,
        multimask_output=bool(getattr(weights, "mask_multimask")),
        chunk_size=int(getattr(weights, "mask_prompt_chunk_size")),
    )
    if debug_timer is not None:
        debug_timer(f"{debug_prefix}.sam_decoder")
    if masks.numel() == 0:
        return _zero_mask_losses(outputs["confidence"])
    n, _k, mh, mw = masks.shape
    logits = masks.float()
    iou_predictions = iou_predictions.to(device=logits.device, dtype=logits.dtype)
    if iou_predictions.ndim != 2 or iou_predictions.shape[0] != n:
        raise ValueError(f"SAM iou predictions must have shape [N, K], got {tuple(iou_predictions.shape)}")
    if iou_predictions.shape[1] != logits.shape[1]:
        raise ValueError(
            f"SAM iou predictions and masks disagree: iou={tuple(iou_predictions.shape)} masks={tuple(logits.shape)}"
        )

    weight_dice = float(getattr(weights, "mask_dice")) > 0.0
    weight_bce = float(getattr(weights, "mask_bce")) > 0.0
    weight_centroid = float(getattr(weights, "mask_centroid")) > 0.0
    weight_outside = float(getattr(weights, "mask_outside")) > 0.0
    weight_min_area = float(getattr(weights, "mask_min_area")) > 0.0
    weight_max_area = float(getattr(weights, "mask_max_area", 0.1)) > 0.0
    weight_pred_iou = float(getattr(weights, "mask_pred_iou", 0.1)) > 0.0
    weight_stability = float(getattr(weights, "mask_stability", 0.1)) > 0.0

    needs_prob = weight_dice or weight_centroid or weight_outside or weight_min_area or weight_max_area
    needs_target = weight_dice or weight_bce or weight_outside
    prob = torch.sigmoid(logits) if needs_prob else None
    if debug_timer is not None:
        debug_timer(f"{debug_prefix}.prob")
    prompt_weights = prompts["weights"].to(dtype=logits.dtype).clamp_min(0.0)
    mask_target_weights = prompts.get("mask_target_weights")
    if mask_target_weights is None:
        mask_target_weights = torch.ones_like(prompt_weights)
    else:
        mask_target_weights = mask_target_weights.to(device=logits.device, dtype=logits.dtype).clamp(0.0, 1.0)
    valid_prompt = prompt_weights > 0
    zero = logits.new_zeros(())
    target = (
        _ellipse_targets_lowres(
            prompts["centers"],
            prompts["target_shapes"].detach(),
            mask_hw=(mh, mw),
            image_hw=image_hw,
            ellipse_sigma=ellipse_sigma,
        )
        if needs_target
        else None
    )
    if debug_timer is not None:
        debug_timer(f"{debug_prefix}.target")

    if weight_bce:
        assert target is not None
        bce = F.binary_cross_entropy_with_logits(
            logits,
            target[:, None, :, :].expand(-1, int(logits.shape[1]), -1, -1),
            reduction="none",
        ).mean(dim=(-1, -2))
    else:
        bce = logits.new_zeros((n, int(logits.shape[1])))
    if debug_timer is not None:
        debug_timer(f"{debug_prefix}.bce")
    if weight_dice:
        assert target is not None and prob is not None
        target_expanded = target[:, None, :, :]
        inter = (prob * target_expanded).sum(dim=(-1, -2))
        denom = prob.sum(dim=(-1, -2)) + target_expanded.sum(dim=(-1, -2))
        dice = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)
    else:
        dice = logits.new_zeros((n, int(logits.shape[1])))
    if debug_timer is not None:
        debug_timer(f"{debug_prefix}.dice")
    if weight_outside:
        assert target is not None and prob is not None
        outside = (prob * (1.0 - target[:, None, :, :])).sum(dim=(-1, -2)) / prob.sum(dim=(-1, -2)).clamp_min(1e-6)
    else:
        outside = logits.new_zeros((n, int(logits.shape[1])))
    if debug_timer is not None:
        debug_timer(f"{debug_prefix}.outside")
    full_area_scale = float(image_hw[0] * image_hw[1]) / float(mh * mw)
    area_full = prob.sum(dim=(-1, -2)) * full_area_scale if needs_prob and prob is not None else None
    if weight_stability:
        soft_stability = _soft_stability_score(
            logits,
            mask_threshold=0.0,
            threshold_offset=float(getattr(weights, "mask_stability_score_offset", 1.0)),
            temperature=float(getattr(weights, "mask_stability_temperature", 10.0)),
        )
    else:
        soft_stability = None
    if debug_timer is not None:
        debug_timer(f"{debug_prefix}.stability_score")
    if weight_min_area:
        assert area_full is not None
        lower = float(getattr(weights, "mask_area_ratio_lower", 0.15))
        upper = float(getattr(weights, "mask_area_ratio_upper", 1.05))
        if lower < 0.0 or upper <= lower:
            raise ValueError("mask_area_ratio_lower must be >= 0 and mask_area_ratio_upper must be greater than lower.")
        target_shapes = prompts["target_shapes"].to(device=logits.device, dtype=logits.dtype).detach()
        shape_major = target_shapes[:, 0].abs().clamp_min(1.0) * float(ellipse_sigma)
        shape_minor = target_shapes[:, 1].abs().clamp_min(1.0) * float(ellipse_sigma)
        shape_area = math.pi * shape_major * shape_minor
        area_ratio = area_full / shape_area[:, None].clamp_min(1.0)
        area_loss = F.relu(1.0 / torch.clamp_min(area_ratio, 0.1) - 1 / lower) + F.relu(area_ratio - upper)
    else:
        area_loss = logits.new_zeros((n, int(logits.shape[1])))
    if debug_timer is not None:
        debug_timer(f"{debug_prefix}.area")

    if weight_centroid:
        assert prob is not None
        yy, xx = torch.meshgrid(
            torch.arange(mh, device=logits.device, dtype=logits.dtype),
            torch.arange(mw, device=logits.device, dtype=logits.dtype),
            indexing="ij",
        )
        area_lr = prob.sum(dim=(-1, -2))
        cx = (prob * xx).sum(dim=(-1, -2)) / area_lr.clamp_min(1e-6)
        cy = (prob * yy).sum(dim=(-1, -2)) / area_lr.clamp_min(1e-6)
        target_cx = prompts["centers"][:, 0].to(dtype=logits.dtype)[:, None] * (float(mw) / float(image_hw[1]))
        target_cy = prompts["centers"][:, 1].to(dtype=logits.dtype)[:, None] * (float(mh) / float(image_hw[0]))
        centroid = (torch.abs(cx - target_cx) + torch.abs(cy - target_cy)) # / max(float(mh + mw) * 0.5, 1.0)
    else:
        centroid = logits.new_zeros((n, int(logits.shape[1])))
    if debug_timer is not None:
        debug_timer(f"{debug_prefix}.centroid")

    max_area_ratio = float(getattr(weights, "mask_max_area_ratio", 0.5))
    if max_area_ratio <= 0.0:
        raise ValueError("mask_max_area_ratio must be in (0, 1]; use >=1 to disable large-mask filtering.")
    if weight_max_area and max_area_ratio < 1.0:
        assert area_full is not None
        max_area_px = float(image_hw[0] * image_hw[1]) * max_area_ratio
        max_area_loss = F.relu(area_full - max_area_px) / max(max_area_px, 1.0)
    else:
        max_area_loss = torch.zeros_like(area_loss)
    if debug_timer is not None:
        debug_timer(f"{debug_prefix}.max_area")
    if weight_pred_iou:
        pred_iou_thresh = float(getattr(weights, "mask_pred_iou_thresh", 0.8))
        pred_iou_loss = F.relu(pred_iou_thresh - iou_predictions) / max(pred_iou_thresh, 1e-6)
    else:
        pred_iou_loss = logits.new_zeros((n, int(logits.shape[1])))
    if debug_timer is not None:
        debug_timer(f"{debug_prefix}.pred_iou")
    if weight_stability:
        assert soft_stability is not None
        stability_thresh = float(getattr(weights, "mask_stability_score_thresh", 0.95))
        stability_loss = F.relu(stability_thresh - soft_stability) / max(stability_thresh, 1e-6)
    else:
        stability_loss = logits.new_zeros((n, int(logits.shape[1])))
    if debug_timer is not None:
        debug_timer(f"{debug_prefix}.stability_loss")

    selection = str(getattr(weights, "mask_selection", "pred_iou")).lower()
    if selection == "pred_iou":
        best = torch.argmax(iou_predictions.detach(), dim=1)
    elif selection == "loss":
        target_factor = mask_target_weights[:, None]
        selection_score = (
            (float(getattr(weights, "mask_dice")) * dice * target_factor if weight_dice else 0.0)
            + (float(getattr(weights, "mask_bce")) * bce * target_factor if weight_bce else 0.0)
            + (float(getattr(weights, "mask_centroid")) * centroid if weight_centroid else 0.0)
            + (float(getattr(weights, "mask_outside")) * outside if weight_outside else 0.0)
            + (float(getattr(weights, "mask_min_area")) * area_loss if weight_min_area else 0.0)
            + (float(getattr(weights, "mask_max_area", 0.1)) * max_area_loss if weight_max_area else 0.0)
            + (float(getattr(weights, "mask_pred_iou", 0.1)) * pred_iou_loss if weight_pred_iou else 0.0)
            + (float(getattr(weights, "mask_stability", 0.1)) * stability_loss if weight_stability else 0.0)
        )
        best = torch.argmin(selection_score.detach(), dim=1)
    else:
        raise ValueError(f"Unknown mask_selection={selection!r}; expected 'pred_iou' or 'loss'.")
    if debug_timer is not None:
        debug_timer(f"{debug_prefix}.selection")

    row = torch.arange(n, device=logits.device)
    selected_iou = iou_predictions[row, best]

    selected = {
        "dice": dice[row, best],
        "bce": bce[row, best],
        "centroid": centroid[row, best],
        "outside": outside[row, best],
        "area": area_loss[row, best],
        "max_area": max_area_loss[row, best],
        "pred_iou": pred_iou_loss[row, best],
        "stability": stability_loss[row, best],
    }

    geom_weights = prompt_weights * valid_prompt.to(dtype=prompt_weights.dtype)
    geom_denom = geom_weights.sum().clamp_min(1.0)
    mask_target_weights = mask_target_weights * valid_prompt.to(dtype=mask_target_weights.dtype)
    target_loss_weights = prompt_weights * mask_target_weights
    target_denom = target_loss_weights.sum().clamp_min(1.0)
    out = {
        "dice": (selected["dice"] * target_loss_weights).sum() / target_denom if weight_dice else zero,
        "bce": (selected["bce"] * target_loss_weights).sum() / target_denom if weight_bce else zero,
        "centroid": (selected["centroid"] * geom_weights).sum() / geom_denom if weight_centroid else zero,
        "outside": (selected["outside"] * geom_weights).sum() / geom_denom if weight_outside else zero,
        "area": (selected["area"] * geom_weights).sum() / geom_denom if weight_min_area else zero,
        "max_area": (selected["max_area"] * geom_weights).sum() / geom_denom if weight_max_area else zero,
        "pred_iou": (selected["pred_iou"] * geom_weights).sum() / geom_denom if weight_pred_iou else zero,
        "stability": (selected["stability"] * geom_weights).sum() / geom_denom if weight_stability else zero,
    }
    total = (
        (float(getattr(weights, "mask_dice")) * out["dice"] if weight_dice else zero)
        + (float(getattr(weights, "mask_bce")) * out["bce"] if weight_bce else zero)
        + (float(getattr(weights, "mask_centroid")) * out["centroid"] if weight_centroid else zero)
        + (float(getattr(weights, "mask_outside")) * out["outside"] if weight_outside else zero)
        + (float(getattr(weights, "mask_min_area")) * out["area"] if weight_min_area else zero)
        + (float(getattr(weights, "mask_max_area", 0.1)) * out["max_area"] if weight_max_area else zero)
        + (float(getattr(weights, "mask_pred_iou", 0.1)) * out["pred_iou"] if weight_pred_iou else zero)
        + (float(getattr(weights, "mask_stability", 0.1)) * out["stability"] if weight_stability else zero)
    )
    out["total"] = total
    out["prompts"] = logits.new_tensor(float(valid_prompt.sum().detach().item()))
    if debug_timer is not None:
        debug_timer(f"{debug_prefix}.reduce")
    return out


def sam_prompt_mask_losses(
    model: nn.Module,
    outputs: Dict[str, Tensor],
    batch: Dict[str, object],
    *,
    weights: object,
    device: torch.device,
    epoch_index: int,
    threshold: float,
    nms_radius: int,
    confidence_score: str,
    use_ordinal_expectation: bool,
    debug_ordinal_expectation: bool,
    center_refinement: str,
    center_refinement_radius: int,
    ellipse_sigma: float,
    detect_centers_fn: Callable[..., List[np.ndarray]],
    debug_timer: Optional[Callable[[str], None]] = None,
) -> Dict[str, Tensor]:
    if (
        mask_outer_weight_for_epoch(epoch_index, weights) <= 0.0
        or not _any_mask_component_enabled(weights)
        or not hasattr(model, "forward_sam_masks")
        or "image_embeddings" not in outputs
    ):
        return _zero_mask_losses(outputs["confidence"])
    image_hw = tuple(int(v) for v in outputs["confidence"].shape[-2:])
    pred_ratio = prompt_pred_ratio(epoch_index, weights)
    pieces: List[Tuple[float, Dict[str, Tensor]]] = []
    gt_loss: Optional[Dict[str, Tensor]] = None
    pred_loss: Optional[Dict[str, Tensor]] = None
    if pred_ratio < 1.0:
        gt_prompts = _build_gt_prompt_tensors(
            outputs,
            batch,
            device=device,
            max_gt_per_sample=int(getattr(weights, "mask_max_gt_per_sample", 0)),
            sample_fraction=float(1.0 - pred_ratio),
        )
        if debug_timer is not None:
            debug_timer("mask.gt.build_prompts")
        if gt_prompts is not None:
            gt_loss = _sam_mask_loss_for_prompts(
                model,
                outputs,
                gt_prompts,
                weights=weights,
                image_hw=image_hw,
                ellipse_sigma=ellipse_sigma,
                debug_timer=debug_timer,
                debug_prefix="mask.gt",
            )
            pieces.append((1.0 - pred_ratio, gt_loss))
    if pred_ratio > 0.0:
        pred_prompts = _build_pred_prompt_tensors(
            outputs,
            batch,
            device=device,
            threshold=threshold,
            nms_radius=nms_radius,
            confidence_score=confidence_score,
            use_ordinal_expectation=use_ordinal_expectation,
            debug_ordinal_expectation=debug_ordinal_expectation,
            center_refinement=center_refinement,
            center_refinement_radius=center_refinement_radius,
            max_pred_per_sample=int(getattr(weights, "mask_max_pred_per_sample")),
            sample_fraction=float(pred_ratio),
            unmatched_weight=float(getattr(weights, "mask_unmatched_prompt")),
            detach_prompt_shapes=bool(getattr(weights, "detach_mask_prompt_shapes", False)),
            detect_centers_fn=detect_centers_fn,
            debug_timer=debug_timer,
        )
        if debug_timer is not None:
            debug_timer("mask.pred.build_prompts_total")
        if pred_prompts is not None:
            pred_loss = _sam_mask_loss_for_prompts(
                model,
                outputs,
                pred_prompts,
                weights=weights,
                image_hw=image_hw,
                ellipse_sigma=ellipse_sigma,
                debug_timer=debug_timer,
                debug_prefix="mask.pred",
            )
            pieces.append((pred_ratio, pred_loss))
    if not pieces:
        return _zero_mask_losses(outputs["confidence"])
    keys = ("total", "dice", "bce", "centroid", "outside", "area", "max_area", "pred_iou", "stability", "prompts")
    result: Dict[str, Tensor] = {}
    for key in keys:
        if key == "prompts":
            result[key] = sum(loss[key] for _scale, loss in pieces)
        else:
            result[key] = sum(float(scale) * loss[key] for scale, loss in pieces)
    if debug_timer is not None:
        debug_timer("mask.combine")
    zero = result["total"].new_zeros(())
    result["gt_total"] = gt_loss["total"] if gt_loss is not None else zero
    result["pred_total"] = pred_loss["total"] if pred_loss is not None else zero
    result["gt_prompts"] = gt_loss["prompts"] if gt_loss is not None else zero
    result["pred_prompts"] = pred_loss["prompts"] if pred_loss is not None else zero
    return result


__all__ = ["mask_outer_weight_for_epoch", "prompt_pred_ratio", "sam_prompt_mask_losses"]
