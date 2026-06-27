from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def binary_segmentation_logits(seg_logits: Tensor) -> Tensor:
    """Return [B,2,H,W] background/foreground logits."""

    if seg_logits.ndim < 4:
        raise ValueError("seg_logits must have channel dimension")
    if seg_logits.shape[1] == 2:
        return seg_logits
    if seg_logits.shape[1] > 2:
        bg = seg_logits[:, :1]
        fg = torch.logsumexp(seg_logits[:, 1:], dim=1, keepdim=True)
        return torch.cat([bg, fg], dim=1)
    raise ValueError("binary segmentation requires at least two logits")


def shape_regression_loss_map(pred: Tensor, target: Tensor, *, angle_weight: float = 4.0) -> Tensor:
    """Per-pixel shape loss with periodic angular error for channel 2."""

    if pred.shape != target.shape:
        raise ValueError("pred and target shape tensors must have identical shapes")
    if pred.ndim < 3 or pred.shape[1] < 3:
        return F.mse_loss(pred, target, reduction="none").mean(dim=1)
    pred_a = pred[:, 0].clamp_min(1e-3)
    pred_b = pred[:, 1].clamp_min(1e-3)
    target_a = target[:, 0].clamp_min(1e-3)
    target_b = target[:, 1].clamp_min(1e-3)
    pred_area = pred_a * pred_b
    target_area = target_a * target_b
    area_loss = F.smooth_l1_loss(torch.log(pred_area), torch.log(target_area), reduction="none")
    pred_ratio = torch.log(pred_a / pred_b.clamp_min(1e-3))
    target_ratio = torch.log(target_a / target_b.clamp_min(1e-3))
    ratio_loss = F.smooth_l1_loss(pred_ratio, target_ratio, reduction="none")
    axes_loss = ratio_loss + area_loss
    angle_loss = 1.0 - torch.cos(2 * (pred[:, 2] - target[:, 2]))
    angle_weight_t = pred.new_tensor(float(angle_weight)).clamp_min(0.0)
    return (2.0 * axes_loss + angle_weight_t * angle_loss) / (2.0 + angle_weight_t)


def flatten_per_band_outputs(outputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Convert [B, C, ...] per-band outputs to [B*C, ...] dense-loss layout."""

    first = next((value for value in outputs.values() if torch.is_tensor(value)), None)
    if first is None or first.ndim != 5:
        return outputs
    return {key: value.reshape(value.shape[0] * value.shape[1], *value.shape[2:]) for key, value in outputs.items()}


def flatten_band_centers(nested: Sequence[Sequence[Tensor]]) -> List[Tensor]:
    return [centers for item in nested for centers in item]


def flatten_band_masks(mask: Tensor) -> List[Tensor]:
    if mask.ndim == 4:
        return [item for record in mask for item in record]
    if mask.ndim == 3:
        return [item for item in mask]
    return []


def filter_points_by_ignore_mask(pred_xy: np.ndarray, ignore_mask: Optional[Tensor]) -> np.ndarray:
    if ignore_mask is None or pred_xy.size == 0:
        return pred_xy
    mask_np = ignore_mask.detach().cpu().numpy().astype(bool)
    if mask_np.ndim != 2:
        return pred_xy
    h, w = mask_np.shape
    keep = []
    for xy in pred_xy:
        x = int(round(float(xy[0])))
        y = int(round(float(xy[1])))
        if x < 0 or y < 0 or x >= w or y >= h:
            keep.append(True)
        else:
            keep.append(not bool(mask_np[y, x]))
    return pred_xy[np.asarray(keep, dtype=bool)]


def filter_points_by_mask_with_count(pred_xy: np.ndarray, mask: Optional[Tensor]) -> Tuple[np.ndarray, int]:
    if mask is None or pred_xy.size == 0:
        return pred_xy, 0
    mask_np = mask.detach().cpu().numpy().astype(bool)
    if mask_np.ndim != 2:
        return pred_xy, 0
    h, w = mask_np.shape
    keep = []
    removed = 0
    for xy in pred_xy:
        x = int(round(float(xy[0])))
        y = int(round(float(xy[1])))
        inside = 0 <= x < w and 0 <= y < h and bool(mask_np[y, x])
        if inside:
            removed += 1
            keep.append(False)
        else:
            keep.append(True)
    return pred_xy[np.asarray(keep, dtype=bool)], removed


def point_in_mask(mask_np: np.ndarray, xy: np.ndarray) -> bool:
    if mask_np.ndim != 2:
        return False
    h, w = mask_np.shape
    x = int(round(float(xy[0])))
    y = int(round(float(xy[1])))
    return 0 <= x < w and 0 <= y < h and bool(mask_np[y, x])


def count_points_in_masks(pred_xy: np.ndarray, masks: Sequence[Optional[Tensor]]) -> int:
    if pred_xy.size == 0:
        return 0
    mask_arrays: List[np.ndarray] = []
    for mask in masks:
        if mask is None:
            continue
        arr = mask.detach().cpu().numpy().astype(bool)
        if arr.ndim == 2:
            mask_arrays.append(arr)
    if not mask_arrays:
        return int(len(pred_xy))
    count = 0
    for xy in pred_xy:
        if any(point_in_mask(mask_np, xy) for mask_np in mask_arrays):
            count += 1
    return count


def sample_embeddings_at_centers(outputs: Dict[str, Tensor], centers_list: Sequence[Tensor]) -> Tuple[Tensor, Tensor]:
    feats: List[Tensor] = []
    batch_ids: List[Tensor] = []
    emb = outputs["embedding"]
    h, w = emb.shape[-2:]
    for b, centers in enumerate(centers_list):
        if centers.numel() == 0:
            continue
        xy = centers.to(device=emb.device, dtype=torch.long)
        x = xy[:, 0].clamp(0, w - 1)
        y = xy[:, 1].clamp(0, h - 1)
        feats.append(emb[b, :, y, x].transpose(0, 1))
        batch_ids.append(torch.full((xy.shape[0],), b, dtype=torch.long, device=emb.device))
    if not feats:
        return emb.new_zeros((0, emb.shape[1])), emb.new_zeros((0,), dtype=torch.long)
    return torch.cat(feats, dim=0), torch.cat(batch_ids, dim=0)


def sample_map_at_centers(map_tensor: Tensor, centers: Tensor) -> Tensor:
    """Sample [C,H,W] tensor at x,y center coordinates -> [N,C]."""

    if centers.numel() == 0:
        return map_tensor.new_zeros((0, map_tensor.shape[0]))
    h, w = map_tensor.shape[-2:]
    xy = torch.round(centers.to(device=map_tensor.device, dtype=torch.float32)).to(dtype=torch.long)
    x = xy[:, 0].clamp(0, w - 1)
    y = xy[:, 1].clamp(0, h - 1)
    return map_tensor[:, y, x].transpose(0, 1)


def confidence_detection_score(outputs: Dict[str, Tensor], mode: str) -> Tensor:
    logits = outputs["confidence"]
    if mode == "raw":
        return logits[:, -1]
    if mode == "ordinal_prob":
        prev = logits[:, :-1].max(dim=1).values
        curr = logits[:, -1]
        return torch.softmax(torch.stack([prev, curr], dim=1), dim=1)[:, 1]
    if mode == "ordinal_expectation":
        prob = torch.softmax(logits.float(), dim=1)
        level_values = torch.arange(logits.shape[1], device=prob.device, dtype=prob.dtype).view(1, -1, 1, 1)
        return (prob * level_values).sum(dim=1).to(dtype=logits.dtype)
    raise ValueError(f"Unknown confidence score mode: {mode}")


def cellect_confidence_smooth_2d(logits: Tensor) -> Tensor:
    """Apply CELLECT's DK1 confidence smoothing kernel in 2D."""

    kernel = logits.new_tensor(
        [
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 2, 1, 0, 0],
            [0, 1, 2, 3, 2, 1, 0],
            [1, 2, 3, 4, 3, 2, 1],
            [0, 1, 2, 3, 2, 1, 0],
            [0, 0, 1, 2, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ]
    ).reshape(1, 1, 7, 7)
    weight = kernel.expand(logits.shape[1], 1, 7, 7)
    return F.conv2d(logits.float(), weight.float(), padding=3, groups=logits.shape[1]).to(dtype=logits.dtype)


def cellect_foreground_gate_2d(seg_logits: Tensor) -> Tensor:
    """Reject candidates touching predicted background, matching CELLECT's kflb gate."""

    background = seg_logits.argmax(dim=1) == 0
    background_near = F.max_pool2d(background.float().unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1) > 0
    return ~background_near


def refine_peak_coordinates(
    score_map: Tensor,
    y: Tensor,
    x: Tensor,
    *,
    method: str,
    radius: int,
) -> Tensor:
    """Return x,y peak coordinates, optionally refined to sub-pixel positions."""

    if x.numel() == 0:
        return score_map.new_zeros((0, 2), dtype=torch.float32)
    method = str(method)
    if method in ("none", "integer"):
        return torch.stack([x, y], dim=1).to(dtype=torch.float32)
    if method != "softargmax":
        raise ValueError(f"Unknown center refinement method: {method}")

    h, w = score_map.shape[-2:]
    window = max(0, int(radius))
    coords: List[Tensor] = []
    for yi, xi in zip(y.tolist(), x.tolist()):
        x0, x1 = max(0, int(xi) - window), min(w, int(xi) + window + 1)
        y0, y1 = max(0, int(yi) - window), min(h, int(yi) + window + 1)
        patch = score_map[y0:y1, x0:x1].float()
        if patch.numel() == 0 or not bool(torch.isfinite(patch).any()):
            coords.append(score_map.new_tensor([float(xi), float(yi)], dtype=torch.float32))
            continue
        patch = torch.nan_to_num(patch, nan=float("-inf"))
        weights = F.softmax((patch - patch.max()).reshape(-1), dim=0)
        xs = torch.arange(x0, x1, device=score_map.device, dtype=torch.float32)
        ys = torch.arange(y0, y1, device=score_map.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        refined_x = (weights * xx.reshape(-1)).sum()
        refined_y = (weights * yy.reshape(-1)).sum()
        coords.append(torch.stack((refined_x, refined_y)))
    return torch.stack(coords, dim=0).to(dtype=torch.float32)


def nearest_candidate_indices(
    anchor_xy: Tensor,
    candidate_xy: Tensor,
    *,
    candidate_count: int,
    exclude_index: Optional[int] = None,
    positive_mask: Optional[Tensor] = None,
) -> Tensor:
    """Return K candidate indices, forcing a positive inside K when present."""

    if candidate_xy.numel() == 0:
        return candidate_xy.new_empty((0,), dtype=torch.long)
    dist = torch.linalg.norm(candidate_xy - anchor_xy[None, :], dim=1)
    if exclude_index is not None and 0 <= exclude_index < dist.shape[0]:
        dist = dist.clone()
        dist[int(exclude_index)] = float("inf")
    finite = torch.isfinite(dist)
    if not bool(finite.any()):
        return candidate_xy.new_empty((0,), dtype=torch.long)
    order = torch.argsort(dist)
    order = order[torch.isfinite(dist[order])]
    selected = order[: int(candidate_count)]
    if positive_mask is not None and bool(positive_mask.any()):
        pos_order = torch.argsort(dist.masked_fill(~positive_mask, float("inf")))
        pos_order = pos_order[torch.isfinite(dist[pos_order])]
        if pos_order.numel() > 0 and not bool((selected == pos_order[0]).any()):
            if selected.numel() >= int(candidate_count):
                selected = torch.cat([selected[:-1], pos_order[:1]], dim=0)
            else:
                selected = torch.cat([selected, pos_order[:1]], dim=0)
    return selected
