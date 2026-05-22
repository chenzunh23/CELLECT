"""Losses, post-processing, and epoch/evaluation helpers for AstroCELLECT training."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from astro_cellect2d import AstroMatchNet2D, ordinal_confidence_loss
from astro_match_eval import matcher_classification_loss as vectorized_matcher_classification_loss


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

    segmentation_3cls: Tuple[float, float, float] = (1.0, 32.0, 1.0)
    confidence_pos_weight: float = 32.0
    center_position: float = 1.0
    triplet_margin: float = 0.3
    triplet_outer_weight: float = 10.0
    matcher_outer_weight: float = 10.0


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
    seg_target = batch["seg"].to(device=device, dtype=torch.long)  # type: ignore[union-attr]
    conf_target = batch["confidence"].to(device=device, dtype=torch.long)  # type: ignore[union-attr]
    shape_target = batch["shape"].to(device=device, dtype=torch.float32)  # type: ignore[union-attr]
    shape_weight = batch["shape_weight"].to(device=device, dtype=torch.float32)  # type: ignore[union-attr]

    seg_weight = torch.tensor(weights.segmentation_3cls, device=device, dtype=torch.float32)
    seg_loss = F.cross_entropy(outputs["seg_logits"], seg_target, weight=seg_weight)
    # torch.cuda.synchronize()
    # seg_time = time.time()
    # print(f'[DEBUG] Segmentation loss computed in {seg_time - start_time:.3f} seconds.')
    conf_loss = ordinal_confidence_loss(
        outputs["confidence"],
        conf_target,
        pos_weight=weights.confidence_pos_weight,
    )
    # torch.cuda.synchronize()
    # conf_time = time.time()
    # print(f'[DEBUG] Confidence loss computed in {conf_time - seg_time:.3f} seconds.')
    per_pixel_shape = F.mse_loss(outputs["shape"], shape_target, reduction="none").mean(dim=1)
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
    total = seg_loss + conf_loss + shape_loss + weights.center_position * center_loss
    return {"total": total, "seg": seg_loss, "confidence": conf_loss, "shape": shape_loss, "center": center_loss}


def _flatten_per_band_outputs(outputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Convert [B, C, ...] per-band outputs to [B*C, ...] dense-loss layout."""

    if outputs["seg_logits"].ndim != 5:
        return outputs
    return {key: value.reshape(value.shape[0] * value.shape[1], *value.shape[2:]) for key, value in outputs.items()}


def _flatten_band_centers(nested: Sequence[Sequence[Tensor]]) -> List[Tensor]:
    return [centers for item in nested for centers in item]


def dense_losses_any(
    outputs: Dict[str, Tensor],
    batch: Dict[str, object],
    *,
    weights: LossWeights,
    device: torch.device,
    center_radius_px: float,
) -> Dict[str, Tensor]:
    """Dense loss for both fused BCHW outputs and per-band B,C,CHW outputs."""

    if outputs["seg_logits"].ndim != 5:
        return dense_losses(outputs, batch, weights=weights, device=device, center_radius_px=center_radius_px)

    flat_outputs = _flatten_per_band_outputs(outputs)
    flat_batch = {
        "seg": batch["band_seg"].reshape(-1, *batch["band_seg"].shape[2:]),  # type: ignore[union-attr]
        "confidence": batch["band_confidence"].reshape(-1, *batch["band_confidence"].shape[2:]),  # type: ignore[union-attr]
        "shape": batch["band_shape"].reshape(-1, *batch["band_shape"].shape[2:]),  # type: ignore[union-attr]
        "shape_weight": batch["band_shape_weight"].reshape(-1, *batch["band_shape_weight"].shape[2:]),  # type: ignore[union-attr]
        "centers": _flatten_band_centers(batch["band_centers"]),  # type: ignore[arg-type]
    }
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


def _sample_embeddings_at_centers(outputs: Dict[str, Tensor], centers_list: Sequence[Tensor]) -> Tuple[Tensor, Tensor]:
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


def _sample_map_at_centers(map_tensor: Tensor, centers: Tensor) -> Tensor:
    """Sample [C,H,W] tensor at x,y center coordinates -> [N,C]."""

    if centers.numel() == 0:
        return map_tensor.new_zeros((0, map_tensor.shape[0]))
    h, w = map_tensor.shape[-2:]
    xy = centers.to(device=map_tensor.device, dtype=torch.long)
    x = xy[:, 0].clamp(0, w - 1)
    y = xy[:, 1].clamp(0, h - 1)
    return map_tensor[:, y, x].transpose(0, 1)


def _confidence_score_at_centers(outputs: Dict[str, Tensor], centers: Tensor, *, batch_index: int = 0) -> Tensor:
    """Sample CELLECT-smoothed top confidence channel at detected centers."""

    if centers.numel() == 0:
        return outputs["confidence"].new_zeros((0,))
    confidence = outputs["confidence"]
    if confidence.ndim == 5:
        raise ValueError("_confidence_score_at_centers expects one dense band output")
    score_map = _cellect_confidence_smooth_2d(confidence)[batch_index, -1]
    h, w = score_map.shape[-2:]
    xy = centers.to(device=score_map.device, dtype=torch.long)
    x = xy[:, 0].clamp(0, w - 1)
    y = xy[:, 1].clamp(0, h - 1)
    return score_map[y, x]


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
    centers_cpu = centers.detach().cpu().numpy().astype(np.float32)
    scores_cpu = center_scores.detach().cpu().numpy()
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
    match_radius: float,
    candidate_count: int,
    en_threshold: float,
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
    match_radius: float,
    candidate_count: int,
    ex_threshold: float,
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


def _nearest_candidate_indices(
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

    By default EX is trained from one core band to every other band, which keeps
    the CELLECT idea of one anchor frame while avoiding all pair permutations.
    Explicit specs use ``src:dst`` or ``src->dst``; ``all`` restores every
    directed cross-band pair.
    """

    if len(bands) <= 1:
        return tuple()
    specs = [part.strip() for item in (pair_specs or ()) for part in str(item).split(",") if part.strip()]
    if any(spec.lower() == "all" for spec in specs):
        return tuple((src, dst) for src in range(len(bands)) for dst in range(len(bands)) if src != dst)
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

    core_idx = _resolve_band_index(bands, core_band)
    return tuple((core_idx, dst) for dst in range(len(bands)) if dst != core_idx)


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
def _confidence_detection_score(outputs: Dict[str, Tensor], mode: str) -> Tensor:
    logits = outputs["confidence"]
    if mode == "raw":
        return logits[:, -1]
    if mode == "ordinal_prob":
        prev = logits[:, :-1].max(dim=1).values
        curr = logits[:, -1]
        return torch.softmax(torch.stack([prev, curr], dim=1), dim=1)[:, 1]
    raise ValueError(f"Unknown confidence score mode: {mode}")


def _cellect_confidence_smooth_2d(logits: Tensor) -> Tensor:
    """Apply CELLECT's DK1 confidence smoothing kernel in 2D.

    Original CELLECT applies a grouped 3D convolution with the 7x7x1 DK1
    diamond kernel before candidate extraction. For HSC 2D images we use the
    same XY kernel without normalization, matching the original score scale.
    """

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


def _cellect_foreground_gate_2d(seg_logits: Tensor) -> Tensor:
    """Reject candidates touching predicted background, matching CELLECT's kflb gate."""

    background = seg_logits.argmax(dim=1) == 0
    background_near = F.max_pool2d(background.float().unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1) > 0
    return ~background_near


def detect_centers(
    outputs: Dict[str, Tensor],
    *,
    threshold: float = 0.0,
    nms_radius: int = 1,
    confidence_score: str = "cellect",
) -> List[np.ndarray]:
    """Detect center candidates from confidence maps.

    ``confidence_score="cellect"`` follows the original CELLECT extraction:
    smooth all confidence channels with DK1, max-pool the channel-wise maximum
    with kernel size 3, and keep voxels/pixels where that local maximum equals
    channel 4 while the local segmentation neighborhood is non-background.
    """

    if confidence_score == "cellect":
        smoothed = _cellect_confidence_smooth_2d(outputs["confidence"])
        local_score = smoothed.max(dim=1).values
        center_score = smoothed[:, -1]
        pooled = F.max_pool2d(local_score.unsqueeze(1), kernel_size=2 * nms_radius + 1, stride=1, padding=1).squeeze(1)
        peaks = (pooled == center_score) & _cellect_foreground_gate_2d(outputs["seg_logits"]) & (center_score > threshold)
        conf = center_score
    else:
        conf = _confidence_detection_score(outputs, confidence_score)
        fg = outputs["seg_logits"].argmax(dim=1) > 0
        pooled = F.max_pool2d(
            conf.unsqueeze(1),
            kernel_size=2 * nms_radius + 1,
            stride=1,
            padding=nms_radius,
        ).squeeze(1)
        peaks = (conf == pooled) & fg & (conf > threshold)

    result: List[np.ndarray] = []
    for b in range(conf.shape[0]):
        y, x = torch.where(peaks[b])
        coords = torch.stack([x, y], dim=1).detach().cpu().numpy().astype(np.float32)
        result.append(coords)
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


def match_points(pred_xy: np.ndarray, gt_xy: np.ndarray, radius: float) -> Tuple[int, int, int]:
    if pred_xy.size == 0:
        return 0, 0, int(len(gt_xy))
    if gt_xy.size == 0:
        return 0, int(len(pred_xy)), 0
    dist = np.sqrt(((pred_xy[:, None, :] - gt_xy[None, :, :]) ** 2).sum(axis=2))
    pairs = []
    for i in range(dist.shape[0]):
        j = int(np.argmin(dist[i]))
        if dist[i, j] <= radius:
            pairs.append((float(dist[i, j]), i, j))
    pairs.sort()
    used_pred = set()
    used_gt = set()
    tp = 0
    for _d, i, j in pairs:
        if i in used_pred or j in used_gt:
            continue
        used_pred.add(i)
        used_gt.add(j)
        tp += 1
    fp = len(pred_xy) - tp
    fn = len(gt_xy) - tp
    return tp, fp, fn


def _band_label(band_idx: int, band_names: Sequence[str]) -> str:
    if 0 <= int(band_idx) < len(band_names):
        return str(band_names[int(band_idx)])
    return f"band{int(band_idx)}"


def _format_band_set(bands: set[int], band_names: Sequence[str]) -> str:
    if not bands:
        return "-"
    return ",".join(_band_label(idx, band_names) for idx in sorted(bands))


def _nearest_gt_id(
    center_xy: np.ndarray,
    gt_centers: Tensor,
    gt_ids: Tensor,
    *,
    radius: float,
) -> Optional[int]:
    if gt_centers.numel() == 0 or gt_ids.numel() == 0:
        return None
    centers = gt_centers.detach().cpu().numpy().astype(np.float32)
    ids = gt_ids.detach().cpu().numpy().astype(np.int64)
    if centers.ndim != 2 or centers.shape[0] == 0:
        return None
    dist = np.sqrt(((centers[:, :2] - np.asarray(center_xy, dtype=np.float32)[None, :2]) ** 2).sum(axis=1))
    best = int(np.argmin(dist))
    if float(dist[best]) <= float(radius):
        return int(ids[best])
    return None


def _evaluate_link_components(
    components: Sequence[Dict[str, object]],
    band_centers: Sequence[Tensor],
    band_ids: Sequence[Tensor],
    *,
    match_radius: float,
    band_names: Sequence[str],
    merged_centers: Optional[Tensor] = None,
    merged_ids: Optional[Tensor] = None,
    band_rejected_ids: Optional[Sequence[Tensor]] = None,
) -> Dict[str, object]:
    filtered_bands_by_id: Dict[int, set[int]] = {}
    for band_idx, ids in enumerate(band_ids):
        ids_np = ids.detach().cpu().numpy().astype(np.int64) if ids.numel() else np.zeros((0,), dtype=np.int64)
        for source_id in ids_np:
            filtered_bands_by_id.setdefault(int(source_id), set()).add(int(band_idx))

    gt_bands_by_id: Dict[int, set[int]] = {}
    use_merged_gt = merged_centers is not None and merged_ids is not None
    if use_merged_gt:
        ids_np = (
            merged_ids.detach().cpu().numpy().astype(np.int64)
            if merged_ids is not None and merged_ids.numel()
            else np.zeros((0,), dtype=np.int64)
        )
        rejected_ids: set[int] = set()
        for ids in band_rejected_ids or ():
            rejected_np = ids.detach().cpu().numpy().astype(np.int64) if ids.numel() else np.zeros((0,), dtype=np.int64)
            rejected_ids.update(int(source_id) for source_id in rejected_np)
        for source_id in ids_np:
            source_id_i = int(source_id)
            if source_id_i in rejected_ids:
                continue
            bands_for_source = filtered_bands_by_id.get(source_id_i, set())
            if bands_for_source:
                gt_bands_by_id[source_id_i] = set(bands_for_source)
    else:
        gt_bands_by_id = {source_id: set(bands) for source_id, bands in filtered_bands_by_id.items()}

    tp = fp = 0
    matched_gt_ids: set[int] = set()
    layer_counts: Dict[str, int] = {}
    reason_counts: Dict[str, int] = {}
    examples: List[Dict[str, object]] = []

    def add_layer(
        reason: str,
        matched_bands: set[int],
        fp_bands: set[int],
        fn_bands: set[int],
        *,
        member_bands: Optional[set[int]] = None,
        gt_bands: Optional[set[int]] = None,
        source_id: Optional[int] = None,
        matched_source_ids: Optional[Sequence[int]] = None,
    ) -> None:
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        key = (
            f"reason={reason}|matched={_format_band_set(matched_bands, band_names)}|"
            f"fp={_format_band_set(fp_bands, band_names)}|fn={_format_band_set(fn_bands, band_names)}"
        )
        layer_counts[key] = layer_counts.get(key, 0) + 1
        if len(examples) < 50:
            examples.append(
                {
                    "reason": reason,
                    "source_id": source_id,
                    "matched_source_ids": list(matched_source_ids or []),
                    "member_bands": [_band_label(idx, band_names) for idx in sorted(member_bands or set())],
                    "gt_bands": [_band_label(idx, band_names) for idx in sorted(gt_bands or set())],
                    "matched_bands": [_band_label(idx, band_names) for idx in sorted(matched_bands)],
                    "fp_bands": [_band_label(idx, band_names) for idx in sorted(fp_bands)],
                    "fn_bands": [_band_label(idx, band_names) for idx in sorted(fn_bands)],
                }
            )

    for component in components:
        member_centers = component.get("member_centers")
        if not isinstance(member_centers, list):
            member_centers = []
            for item in component.get("members", []):
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    member_centers.append((int(item[0]), None))

        member_bands: set[int] = set()
        matched_by_id: Dict[int, set[int]] = {}
        for item in member_centers:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            band_idx = int(item[0])
            member_bands.add(band_idx)
            center_xy = item[1]
            if center_xy is None or band_idx < 0 or band_idx >= len(band_centers):
                continue
            gt_centers = merged_centers if use_merged_gt and merged_centers is not None else band_centers[band_idx]
            gt_ids = merged_ids if use_merged_gt and merged_ids is not None else band_ids[band_idx]
            matched_id = _nearest_gt_id(
                np.asarray(center_xy, dtype=np.float32),
                gt_centers,
                gt_ids,
                radius=match_radius,
            )
            if matched_id is not None and int(matched_id) in gt_bands_by_id:
                matched_by_id.setdefault(matched_id, set()).add(band_idx)

        if not member_bands:
            fp += 1
            add_layer("empty_component", set(), set(), set())
            continue

        if not matched_by_id:
            fp += 1
            add_layer("no_member_matches_gt", set(), set(member_bands), set(), member_bands=set(member_bands))
            continue

        source_id, matched_bands = max(
            matched_by_id.items(),
            key=lambda item: (len(item[1]), -int(item[0])),
        )
        gt_bands = gt_bands_by_id.get(int(source_id), set())
        fn_bands = set(gt_bands) - set(matched_bands)
        fp_bands = (set(member_bands) - set(matched_bands)) | (set(member_bands) - set(gt_bands))
        complete = bool(gt_bands) and set(member_bands) == set(gt_bands) and set(matched_bands) == set(gt_bands)

        if complete and int(source_id) not in matched_gt_ids:
            tp += 1
            matched_gt_ids.add(int(source_id))
            continue

        fp += 1
        if complete:
            reason = "duplicate_complete_match"
        elif len(matched_by_id) > 1:
            reason = "mixed_source_ids"
        elif fp_bands and fn_bands:
            reason = "wrong_and_missing_bands"
        elif fp_bands:
            reason = "extra_or_wrong_bands"
        elif fn_bands:
            reason = "missing_bands"
        else:
            reason = "incomplete_match"
        add_layer(
            reason,
            set(matched_bands),
            fp_bands,
            fn_bands,
            member_bands=set(member_bands),
            gt_bands=set(gt_bands),
            source_id=int(source_id),
            matched_source_ids=sorted(int(source_id) for source_id in matched_by_id),
        )

    fn = len(set(gt_bands_by_id) - matched_gt_ids)
    for source_id in sorted(set(gt_bands_by_id) - matched_gt_ids):
        add_layer(
            "unmatched_gt_source",
            set(),
            set(),
            gt_bands_by_id[source_id],
            gt_bands=gt_bands_by_id[source_id],
            source_id=int(source_id),
        )

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": precision,
        "purity": precision,
        "recall": recall,
        "completeness": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "layers": dict(sorted(layer_counts.items(), key=lambda item: (-item[1], item[0]))),
        "reasons": dict(sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))),
        "examples": examples,
    }


def _accumulate_link_metrics(total: Dict[str, object], item: Dict[str, object]) -> None:
    for key in ("tp", "fp", "fn"):
        total[key] = float(total.get(key, 0.0)) + float(item.get(key, 0.0))
    for key in ("layers", "reasons"):
        merged = total.setdefault(key, {})
        assert isinstance(merged, dict)
        for name, count in item.get(key, {}).items():  # type: ignore[union-attr]
            merged[str(name)] = int(merged.get(str(name), 0)) + int(count)
    examples = total.setdefault("examples", [])
    assert isinstance(examples, list)
    for example in item.get("examples", []):  # type: ignore[union-attr]
        if len(examples) >= 50:
            break
        examples.append(example)


def _finalize_link_metrics(total: Dict[str, object]) -> Dict[str, object]:
    tp = float(total.get("tp", 0.0))
    fp = float(total.get("fp", 0.0))
    fn = float(total.get("fn", 0.0))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "purity": precision,
        "recall": recall,
        "completeness": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "gt_reference": total.get("gt_reference", "band_reference_union"),
        "layers": dict(sorted(total.get("layers", {}).items(), key=lambda item: (-item[1], item[0]))),  # type: ignore[union-attr]
        "reasons": dict(sorted(total.get("reasons", {}).items(), key=lambda item: (-item[1], item[0]))),  # type: ignore[union-attr]
        "examples": total.get("examples", []),
    }


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
    distributed: bool = False,
    show_progress: bool = True,
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
    }
    count = 0
    t_epoch_start = time.time()
    # torch.cuda.synchronize(device) if device.type == "cuda" else None
    # print('[DEBUG] Starting epoch, running initial evaluation loop to prime caches...')
    # t_batch = time.time()
    for batch in tqdm(loader, desc="train" if training else "eval", leave=False, disable=not show_progress):
        image = batch["image"].to(device=device, dtype=torch.float32)
        outputs = model(image)
        # torch.cuda.synchronize(device) if device.type == "cuda" else None
        # t_batch_model = time.time()
        # print(f'[DEBUG] Model forward pass done, time {t_batch_model - t_batch:.3f} seconds.')
        losses = dense_losses_any(outputs, batch, weights=weights, device=device, center_radius_px=center_radius_px)
        total = losses["total"]
        # torch.cuda.synchronize(device) if device.type == "cuda" else None
        # dense_time = time.time()
        # print(f'[DEBUG] Dense loss computation done, time {dense_time - t_batch_model:.3f} seconds.')

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
            # torch.cuda.synchronize(device) if device.type == "cuda" else None
            # t_triplet = time.time()
            # print(f'[DEBUG] Triplet loss computation done, time {t_triplet - dense_time:.3f} seconds.')
            # dense_time = t_triplet

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
            # torch.cuda.synchronize(device) if device.type == "cuda" else None
            # t_ex = time.time()
            # print(f'[DEBUG] EX classification loss computation done, time {t_ex - dense_time:.3f} seconds.')
            # dense_time = t_ex

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
            # torch.cuda.synchronize(device) if device.type == "cuda" else None
            # t_en = time.time()
            # print(f'[DEBUG] EN classification loss computation done, time {t_en - dense_time:.3f} seconds.')
            # dense_time = t_en

        # t_backward = dense_time
        if training:
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            # torch.cuda.synchronize(device) if device.type == "cuda" else None
            # t_backward = time.time()
            # print(f'[DEBUG] Optimizer step done, time {t_backward - dense_time:.3f} seconds.')

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
        t_batch = time.time()
    if distributed and dist.is_available() and dist.is_initialized():
        keys = list(sums.keys())
        packed = torch.tensor([sums[key] for key in keys] + [float(count)], device=device, dtype=torch.float64)
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        sums = {key: float(packed[idx].item()) for idx, key in enumerate(keys)}
        count = int(packed[-1].item())
    # print(f"[DEBUG] Epoch completed in {time.time() - t_epoch_start:.3f} seconds.")
    return {key: val / max(count, 1) for key, val in sums.items()}


@torch.no_grad()
def evaluate_detection(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    threshold: float,
    nms_radius: int,
    confidence_score: str,
    match_radius: float,
    use_en_postprocess: bool = False,
    en_candidate_count: int = 5,
    en_threshold: float = 0.6,
    use_ex_link_postprocess: bool = False,
    ex_link_threshold: float = 0.5,
    ex_band_pairs: Optional[Sequence[Tuple[int, int]]] = None,
    band_names: Sequence[str] = (),
    show_progress: bool = True,
) -> Dict[str, object]:
    model.eval()
    base_model = unwrap_model(model)
    tp = fp = fn = 0
    linked_tp = linked_fp = linked_fn = 0
    link_metrics_total: Dict[str, object] = {
        "tp": 0.0,
        "fp": 0.0,
        "fn": 0.0,
        "layers": {},
        "reasons": {},
        "examples": [],
        "gt_reference": "merged_reference_catalog_with_filtered_band_presence",
    }
    per_band_counts: Dict[str, Dict[str, int]] = {
        str(name): {"tp": 0, "fp": 0, "fn": 0} for name in band_names
    }
    for batch in tqdm(loader, desc="detect", leave=False, disable=not show_progress):
        image = batch["image"].to(device=device, dtype=torch.float32)
        outputs = model(image)
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
            )
            gt_list = _flatten_band_centers(batch["band_centers"]) if outputs["seg_logits"].ndim == 5 else batch["centers"]  # type: ignore[arg-type]
            band_indices = [idx % band_count for idx in range(len(pred_list))] if band_count else [None for _ in pred_list]
        elif outputs["seg_logits"].ndim == 5:
            flat_outputs = _flatten_per_band_outputs(outputs)
            pred_list = detect_centers(
                flat_outputs,
                threshold=threshold,
                nms_radius=nms_radius,
                confidence_score=confidence_score,
            )
            gt_list = _flatten_band_centers(batch["band_centers"])  # type: ignore[arg-type]
            band_indices = [idx % band_count for idx in range(len(pred_list))]
        else:
            pred_list = detect_centers(
                outputs,
                threshold=threshold,
                nms_radius=nms_radius,
                confidence_score=confidence_score,
            )
            gt_list = batch["centers"]  # type: ignore[assignment]
            band_indices = [None for _ in pred_list]
        for pred_xy, gt_xy, band_idx in zip(pred_list, gt_list, band_indices):
            t, f, n = match_points(pred_xy, gt_xy.numpy().astype(np.float32), match_radius)
            tp += t
            fp += f
            fn += n
            if band_idx is not None and band_names:
                band_name = str(band_names[int(band_idx)])
                counts = per_band_counts.setdefault(band_name, {"tp": 0, "fp": 0, "fn": 0})
                counts["tp"] += int(t)
                counts["fp"] += int(f)
                counts["fn"] += int(n)

        if use_ex_link_postprocess and outputs["seg_logits"].ndim == 5 and hasattr(base_model, "EX"):
            linked_pred_list, components_all = detect_centers_with_ex_link(
                base_model,
                outputs,
                threshold=threshold,
                nms_radius=nms_radius,
                confidence_score=confidence_score,
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
                linked_tp += t
                linked_fp += f
                linked_fn += n
            nested_band_centers: Sequence[Sequence[Tensor]] = batch["band_centers"]  # type: ignore[assignment]
            nested_band_ids: Sequence[Sequence[Tensor]] = batch["band_ids"]  # type: ignore[assignment]
            merged_centers_list: Sequence[Tensor] = batch["centers"]  # type: ignore[assignment]
            merged_ids_list: Sequence[Tensor] = batch["ids"]  # type: ignore[assignment]
            nested_band_rejected_ids: Optional[Sequence[Sequence[Tensor]]] = batch.get("band_rejected_ids")  # type: ignore[assignment,union-attr]
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
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    result: Dict[str, object] = {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": precision,
        "purity": precision,
        "recall": recall,
        "completeness": recall,
        "f1": f1,
    }
    if per_band_counts:
        per_band: Dict[str, Dict[str, float]] = {}
        for band_name, counts in per_band_counts.items():
            btp = int(counts["tp"])
            bfp = int(counts["fp"])
            bfn = int(counts["fn"])
            b_precision = btp / max(btp + bfp, 1)
            b_recall = btp / max(btp + bfn, 1)
            per_band[band_name] = {
                "tp": float(btp),
                "fp": float(bfp),
                "fn": float(bfn),
                "precision": b_precision,
                "purity": b_precision,
                "recall": b_recall,
                "completeness": b_recall,
                "f1": 2.0 * b_precision * b_recall / max(b_precision + b_recall, 1e-12),
            }
        result["per_band"] = per_band
    if use_ex_link_postprocess:
        linked_precision = linked_tp / max(linked_tp + linked_fp, 1)
        linked_recall = linked_tp / max(linked_tp + linked_fn, 1)
        result["linked"] = {
            "tp": float(linked_tp),
            "fp": float(linked_fp),
            "fn": float(linked_fn),
            "precision": linked_precision,
            "purity": linked_precision,
            "recall": linked_recall,
            "completeness": linked_recall,
            "f1": 2.0 * linked_precision * linked_recall / max(linked_precision + linked_recall, 1e-12),
        }
        result["link_metrics"] = _finalize_link_metrics(link_metrics_total)
    return result
