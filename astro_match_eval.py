"""Matcher utilities for AstroCELLECT EX/EN training and evaluation.

The original training script grew a large amount of EX/EN candidate-building
logic.  This module keeps the matcher-specific pieces together and uses
batched distance/top-k operations instead of per-source Python nearest-neighbor
loops.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from astro_cellect2d import AstroMatchNet2D


def _sample_map_at_centers(map_tensor: Tensor, centers: Tensor) -> Tensor:
    """Sample [C,H,W] tensor at x,y center coordinates -> [N,C]."""

    if centers.numel() == 0:
        return map_tensor.new_zeros((0, map_tensor.shape[0]))
    h, w = map_tensor.shape[-2:]
    xy = centers.to(device=map_tensor.device, dtype=torch.long)
    x = xy[:, 0].clamp(0, w - 1)
    y = xy[:, 1].clamp(0, h - 1)
    return map_tensor[:, y, x].transpose(0, 1)


def sample_multiband_sources(outputs: Dict[str, Tensor], batch: Dict[str, object]) -> List[List[Dict[str, Tensor]]]:
    """Sample per-source EX/EN features for each batch item and band."""

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

    Default: one core band anchors all other bands, e.g. HSC-I -> HSC-G/R.
    Explicit specs use ``src:dst`` or ``src->dst``.  ``all`` restores every
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


def _sample_anchor_indices(ids: Tensor, *, max_anchors: int, prefer_repeated: bool) -> Tensor:
    n = int(ids.shape[0])
    if max_anchors <= 0 or n <= max_anchors:
        return torch.arange(n, device=ids.device, dtype=torch.long)

    if not prefer_repeated:
        return torch.randperm(n, device=ids.device)[:max_anchors]

    unique_ids, counts = torch.unique(ids, return_counts=True)
    repeated_ids = unique_ids[counts > 1]
    repeated_mask = torch.isin(ids, repeated_ids)
    repeated = torch.nonzero(repeated_mask, as_tuple=False).flatten()
    other = torch.nonzero(~repeated_mask, as_tuple=False).flatten()
    selected_parts: List[Tensor] = []
    if repeated.numel() > 0:
        selected_parts.append(repeated[torch.randperm(repeated.numel(), device=ids.device)[:max_anchors]])
    selected = torch.cat(selected_parts, dim=0) if selected_parts else ids.new_empty((0,), dtype=torch.long)
    if selected.numel() < max_anchors and other.numel() > 0:
        take = min(max_anchors - int(selected.numel()), int(other.numel()))
        selected = torch.cat([selected, other[torch.randperm(other.numel(), device=ids.device)[:take]]], dim=0)
    return selected[:max_anchors]


def _topk_candidate_indices(
    anchor_xy: Tensor,
    candidate_xy: Tensor,
    *,
    k: int,
    anchor_ids: Tensor,
    candidate_ids: Tensor,
    exclude_indices: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Vectorized K-nearest candidates with positive forced into K when present."""

    if anchor_xy.numel() == 0 or candidate_xy.numel() == 0:
        empty_idx = anchor_ids.new_empty((0, k), dtype=torch.long)
        empty_valid = torch.zeros((0,), dtype=torch.bool, device=anchor_xy.device)
        empty_dist = anchor_xy.new_empty((0, k))
        return empty_idx, empty_valid, empty_dist

    dist = torch.cdist(anchor_xy.float(), candidate_xy.float(), p=2)
    if exclude_indices is not None and exclude_indices.numel() == dist.shape[0]:
        rows = torch.arange(dist.shape[0], device=dist.device)
        valid_exclude = (exclude_indices >= 0) & (exclude_indices < candidate_xy.shape[0])
        if bool(valid_exclude.any()):
            dist = dist.clone()
            dist[rows[valid_exclude], exclude_indices[valid_exclude]] = float("inf")

    k_eff = min(int(k), int(candidate_xy.shape[0]))
    selected_dist, selected = torch.topk(dist, k=k_eff, dim=1, largest=False)
    valid = torch.isfinite(selected_dist).any(dim=1)
    if k_eff < k:
        pad = k - k_eff
        selected = torch.cat([selected, selected[:, -1:].repeat(1, pad)], dim=1)
        selected_dist = torch.cat([selected_dist, selected_dist[:, -1:].repeat(1, pad)], dim=1)

    finite_selected = torch.isfinite(selected_dist)
    if bool(valid.any()) and not bool(finite_selected[valid].all()):
        first_valid_pos = finite_selected.float().argmax(dim=1)
        first_valid_idx = selected.gather(1, first_valid_pos[:, None])
        first_valid_dist = selected_dist.gather(1, first_valid_pos[:, None])
        selected = torch.where(finite_selected, selected, first_valid_idx.expand_as(selected))
        selected_dist = torch.where(finite_selected, selected_dist, first_valid_dist.expand_as(selected_dist))

    positive = candidate_ids.unsqueeze(0) == anchor_ids.unsqueeze(1)
    if exclude_indices is not None and exclude_indices.numel() == positive.shape[0]:
        rows = torch.arange(positive.shape[0], device=positive.device)
        valid_exclude = (exclude_indices >= 0) & (exclude_indices < candidate_ids.shape[0])
        if bool(valid_exclude.any()):
            positive = positive.clone()
            positive[rows[valid_exclude], exclude_indices[valid_exclude]] = False
    selected_positive = positive.gather(1, selected)
    has_selected_positive = selected_positive.any(dim=1)
    positive_dist = dist.masked_fill(~positive, float("inf"))
    nearest_positive_dist, nearest_positive = positive_dist.min(dim=1)
    has_positive = torch.isfinite(nearest_positive_dist)
    replace = has_positive & ~has_selected_positive
    if bool(replace.any()):
        selected = selected.clone()
        selected_dist = selected_dist.clone()
        selected[replace, -1] = nearest_positive[replace]
        selected_dist[replace, -1] = nearest_positive_dist[replace]

    return selected, valid, selected_dist


def matcher_classification_loss(
    matcher: AstroMatchNet2D,
    outputs: Dict[str, Tensor],
    batch: Dict[str, object],
    *,
    mode: str,
    candidate_count: int,
    offset_scale: float,
    band_pairs: Optional[Sequence[Tuple[int, int]]] = None,
    max_anchors_per_band: int = 0,
) -> Tensor:
    """Vectorized cross-entropy loss for EX/EN candidate classification."""

    if outputs["embedding"].ndim != 5:
        return outputs["embedding"].sum() * 0.0
    sampled = sample_multiband_sources(outputs, batch)
    anchor_features: List[Tensor] = []
    candidate_features: List[Tensor] = []
    candidate_offsets: List[Tensor] = []
    candidate_shapes: List[Tensor] = []
    targets: List[Tensor] = []
    k = int(candidate_count)
    scale = max(float(offset_scale), 1e-6)

    pair_map: Optional[Dict[int, List[int]]] = None
    if mode == "ex" and band_pairs is not None:
        pair_map = {}
        for src, dst in band_pairs:
            pair_map.setdefault(int(src), []).append(int(dst))

    for per_item in sampled:
        anchor_cache: Dict[int, Tensor] = {}
        for band_idx, anchor_band in enumerate(per_item):
            if anchor_band["xy"].numel() == 0:
                continue
            if mode == "ex":
                if pair_map is None:
                    candidate_band_indices = [idx for idx in range(len(per_item)) if idx != band_idx]
                else:
                    candidate_band_indices = [idx for idx in pair_map.get(band_idx, []) if 0 <= idx < len(per_item)]
                prefer_repeated = False
            elif mode == "en":
                candidate_band_indices = [band_idx]
                prefer_repeated = True
            else:
                raise ValueError(f"unknown matcher mode: {mode}")
            if not candidate_band_indices:
                continue

            anchor_idx_all = anchor_cache.get(band_idx)
            if anchor_idx_all is None:
                anchor_idx_all = _sample_anchor_indices(
                    anchor_band["ids"],
                    max_anchors=max_anchors_per_band,
                    prefer_repeated=prefer_repeated,
                )
                anchor_cache[band_idx] = anchor_idx_all
            if anchor_idx_all.numel() == 0:
                continue

            anchor_xy = anchor_band["xy"][anchor_idx_all]
            anchor_ids = anchor_band["ids"][anchor_idx_all]
            anchor_feat = anchor_band["features"][anchor_idx_all]
            anchor_shape = anchor_band["shape"][anchor_idx_all]

            for cand_band_idx in candidate_band_indices:
                cand_band = per_item[cand_band_idx]
                if cand_band["xy"].numel() == 0:
                    continue
                exclude = anchor_idx_all if mode == "en" and cand_band_idx == band_idx else None
                selected, valid, _selected_dist = _topk_candidate_indices(
                    anchor_xy,
                    cand_band["xy"],
                    k=k,
                    anchor_ids=anchor_ids,
                    candidate_ids=cand_band["ids"],
                    exclude_indices=exclude,
                )
                if not bool(valid.any()):
                    continue
                selected = selected[valid]
                valid_anchor_xy = anchor_xy[valid]
                valid_anchor_ids = anchor_ids[valid]
                valid_anchor_feat = anchor_feat[valid]
                valid_anchor_shape = anchor_shape[valid]

                cand_feat = cand_band["features"][selected]
                cand_xy = cand_band["xy"][selected]
                cand_shape = cand_band["shape"][selected]
                same_selected = cand_band["ids"][selected] == valid_anchor_ids.unsqueeze(1)
                has_positive = same_selected.any(dim=1)
                target = torch.full((selected.shape[0],), k, dtype=torch.long, device=valid_anchor_ids.device)
                target[has_positive] = same_selected.float().argmax(dim=1)[has_positive].long()

                anchor_features.append(valid_anchor_feat)
                candidate_features.append(cand_feat)
                candidate_offsets.append((cand_xy - valid_anchor_xy[:, None, :]) / scale)
                candidate_shapes.append(torch.cat([valid_anchor_shape[:, None, :].expand(-1, k, -1), cand_shape], dim=2))
                targets.append(target)

    if not anchor_features:
        return outputs["embedding"].sum() * 0.0

    anchor_tensor = torch.cat(anchor_features, dim=0)
    candidate_tensor = torch.cat(candidate_features, dim=0)
    offset_tensor = torch.cat(candidate_offsets, dim=0)
    shape_tensor = torch.cat(candidate_shapes, dim=0)
    target_tensor = torch.cat(targets, dim=0)
    logits = matcher(anchor_tensor, candidate_tensor, offset_tensor, shape_tensor)
    return F.cross_entropy(logits, target_tensor)
