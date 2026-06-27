from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

def _ellipse_mask_np(
    shape_hw: Tuple[int, int],
    cx: float,
    cy: float,
    major: float,
    minor: float,
    theta: float,
    *,
    ellipse_sigma: float,
    min_axis: float = 1.5,
) -> np.ndarray:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return np.zeros((h, w), dtype=bool)
    a = max(abs(float(major)) * float(ellipse_sigma), float(min_axis))
    b = max(abs(float(minor)) * float(ellipse_sigma), float(min_axis))
    if not np.isfinite(a) or not np.isfinite(b):
        return np.zeros((h, w), dtype=bool)
    xi = int(round(float(cx)))
    yi = int(round(float(cy)))
    radius = int(math.ceil(max(a, b))) + 2
    y0, y1 = max(0, yi - radius), min(h, yi + radius + 1)
    x0, x1 = max(0, xi - radius), min(w, xi + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return np.zeros((h, w), dtype=bool)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx = xx.astype(np.float32) - float(cx)
    dy = yy.astype(np.float32) - float(cy)
    cos_t = math.cos(float(theta)) if np.isfinite(theta) else 1.0
    sin_t = math.sin(float(theta)) if np.isfinite(theta) else 0.0
    xr = cos_t * dx + sin_t * dy
    yr = -sin_t * dx + cos_t * dy
    local = (xr / a) ** 2 + (yr / b) ** 2 <= 1.0
    out = np.zeros((h, w), dtype=bool)
    out[y0:y1, x0:x1] = local
    return out


def _connected_components_bool(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask_bool = np.asarray(mask, dtype=bool)
    if not bool(mask_bool.any()):
        return np.zeros(mask_bool.shape, dtype=np.int32), np.zeros((1,), dtype=np.int64)
    try:
        from scipy import ndimage

        labels, _num = ndimage.label(mask_bool, structure=np.ones((3, 3), dtype=np.uint8))
        areas = np.bincount(labels.ravel()).astype(np.int64)
        return labels.astype(np.int32, copy=False), areas
    except Exception:
        labels = np.zeros(mask_bool.shape, dtype=np.int32)
        h, w = mask_bool.shape
        current = 0
        areas = [0]
        for y in range(h):
            for x in range(w):
                if not mask_bool[y, x] or labels[y, x] != 0:
                    continue
                current += 1
                stack = [(y, x)]
                labels[y, x] = current
                area = 0
                while stack:
                    yy, xx = stack.pop()
                    area += 1
                    for ny in range(max(0, yy - 1), min(h, yy + 2)):
                        for nx in range(max(0, xx - 1), min(w, xx + 2)):
                            if mask_bool[ny, nx] and labels[ny, nx] == 0:
                                labels[ny, nx] = current
                                stack.append((ny, nx))
                areas.append(area)
        return labels, np.asarray(areas, dtype=np.int64)


def _max_iou_with_labeled_mask(pred_mask: np.ndarray, label_map: np.ndarray, label_areas: np.ndarray) -> float:
    pred_bool = np.asarray(pred_mask, dtype=bool)
    pred_area = int(pred_bool.sum())
    if pred_area <= 0 or label_areas.size <= 1:
        return 0.0
    pred_labels = np.asarray(label_map[pred_bool], dtype=np.int64)
    pred_labels = pred_labels[pred_labels > 0]
    if pred_labels.size == 0:
        return 0.0
    overlaps = np.bincount(pred_labels, minlength=int(label_areas.size))
    hit_labels = np.flatnonzero(overlaps > 0)
    max_iou = 0.0
    for label in hit_labels:
        inter = int(overlaps[label])
        clean_area = int(label_areas[label])
        union = pred_area + clean_area - inter
        if union > 0:
            max_iou = max(max_iou, float(inter) / float(union))
    return max_iou


def _centers_from_confidence_target(confidence: Tensor) -> np.ndarray:
    conf = confidence.detach().cpu()
    if conf.numel() == 0:
        return np.zeros((0, 2), dtype=np.float32)
    level = int(conf.max().item())
    if level <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    y, x = torch.where(conf == level)
    if x.numel() == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return torch.stack([x, y], dim=1).numpy().astype(np.float32)


def _min_distance_to_points(xy: np.ndarray, points: np.ndarray) -> float:
    if points.size == 0:
        return float("inf")
    delta = points[:, :2].astype(np.float32) - np.asarray(xy, dtype=np.float32)[None, :2]
    return float(np.sqrt(np.sum(delta * delta, axis=1)).min())


def _band_label(band_idx: int, band_names: Sequence[str]) -> str:
    if 0 <= int(band_idx) < len(band_names):
        return str(band_names[int(band_idx)])
    return f"band{int(band_idx)}"


def _format_band_set(bands: set[int], band_names: Sequence[str]) -> str:
    if not bands:
        return "-"
    return ",".join(_band_label(idx, band_names) for idx in sorted(bands))


def _short_band_pattern(bands: set[int], band_names: Sequence[str]) -> str:
    if not bands:
        return "-"
    labels: List[str] = []
    for idx in sorted(bands):
        label = _band_label(idx, band_names)
        labels.append(label.split("-")[-1] if "-" in label else label)
    return "".join(labels)


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
    partial_components = 0
    matched_gt_ids: set[int] = set()
    touched_gt_ids: set[int] = set()
    layer_counts: Dict[str, int] = {}
    reason_counts: Dict[str, int] = {}
    component_patterns: Dict[str, int] = {}
    gt_patterns: Dict[str, int] = {}
    tp_patterns: Dict[str, int] = {}
    partial_patterns: Dict[str, int] = {}
    fn_patterns: Dict[str, int] = {}
    examples: List[Dict[str, object]] = []

    for bands in gt_bands_by_id.values():
        pattern = _short_band_pattern(set(bands), band_names)
        gt_patterns[pattern] = gt_patterns.get(pattern, 0) + 1

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
            gt_centers = band_centers[band_idx]
            gt_ids = band_ids[band_idx]
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
        component_pattern = _short_band_pattern(set(member_bands), band_names)
        component_patterns[component_pattern] = component_patterns.get(component_pattern, 0) + 1

        if not matched_by_id:
            fp += 1
            add_layer("no_member_matches_gt", set(), set(member_bands), set(), member_bands=set(member_bands))
            continue

        for matched_source_id in matched_by_id:
            if int(matched_source_id) in gt_bands_by_id:
                touched_gt_ids.add(int(matched_source_id))

        complete_candidates = [
            (int(source_id), set(matched_bands), set(gt_bands_by_id.get(int(source_id), set())))
            for source_id, matched_bands in matched_by_id.items()
            if gt_bands_by_id.get(int(source_id), set()) and set(gt_bands_by_id[int(source_id)]).issubset(set(matched_bands))
        ]
        if complete_candidates:
            source_id, matched_bands, gt_bands = max(
                complete_candidates,
                key=lambda item: (len(item[2]), len(item[1]), -int(item[0])),
            )
            if int(source_id) not in matched_gt_ids:
                tp += 1
                matched_gt_ids.add(int(source_id))
                pattern = _short_band_pattern(set(member_bands), band_names)
                tp_patterns[pattern] = tp_patterns.get(pattern, 0) + 1
                continue
            fp += 1
            add_layer(
                "duplicate_complete_match",
                set(matched_bands),
                set(),
                set(),
                member_bands=set(member_bands),
                gt_bands=set(gt_bands),
                source_id=int(source_id),
                matched_source_ids=sorted(int(source_id) for source_id in matched_by_id),
            )
            continue

        source_id, matched_bands = max(
            matched_by_id.items(),
            key=lambda item: (len(item[1]), -int(item[0])),
        )
        gt_bands = gt_bands_by_id.get(int(source_id), set())
        fn_bands = set(gt_bands) - set(matched_bands)
        fp_bands = set()
        partial_components += 1
        if len(matched_by_id) > 1:
            reason = "mixed_source_ids"
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

    partial_gt_ids = (set(gt_bands_by_id) & touched_gt_ids) - matched_gt_ids
    true_fn_ids = set(gt_bands_by_id) - matched_gt_ids - partial_gt_ids
    for source_id in sorted(partial_gt_ids):
        pattern = _short_band_pattern(gt_bands_by_id[source_id], band_names)
        partial_patterns[pattern] = partial_patterns.get(pattern, 0) + 1
        add_layer(
            "partial_gt_source",
            set(),
            set(),
            gt_bands_by_id[source_id],
            gt_bands=gt_bands_by_id[source_id],
            source_id=int(source_id),
        )
    for source_id in sorted(true_fn_ids):
        pattern = _short_band_pattern(gt_bands_by_id[source_id], band_names)
        fn_patterns[pattern] = fn_patterns.get(pattern, 0) + 1
        add_layer(
            "unmatched_gt_source",
            set(),
            set(),
            gt_bands_by_id[source_id],
            gt_bands=gt_bands_by_id[source_id],
            source_id=int(source_id),
        )

    partial = len(partial_gt_ids)
    fn = len(true_fn_ids)
    precision = tp / max(tp + partial + fp, 1)
    recall = tp / max(tp + partial + fn, 1)
    recall_with_partial = (tp + partial) / max(tp + partial + fn, 1)
    return {
        "tp": float(tp),
        "partial": float(partial),
        "partial_components": float(partial_components),
        "fp": float(fp),
        "fn": float(fn),
        "true_fn": float(fn),
        "precision": precision,
        "purity": precision,
        "recall": recall,
        "completeness": recall,
        "recall_with_partial": recall_with_partial,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "component_patterns": dict(sorted(component_patterns.items(), key=lambda item: (-item[1], item[0]))),
        "gt_patterns": dict(sorted(gt_patterns.items(), key=lambda item: (-item[1], item[0]))),
        "tp_patterns": dict(sorted(tp_patterns.items(), key=lambda item: (-item[1], item[0]))),
        "partial_patterns": dict(sorted(partial_patterns.items(), key=lambda item: (-item[1], item[0]))),
        "fn_patterns": dict(sorted(fn_patterns.items(), key=lambda item: (-item[1], item[0]))),
        "layers": dict(sorted(layer_counts.items(), key=lambda item: (-item[1], item[0]))),
        "reasons": dict(sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))),
        "examples": examples,
    }


def _accumulate_link_metrics(total: Dict[str, object], item: Dict[str, object]) -> None:
    for key in ("tp", "partial", "partial_components", "fp", "fn", "true_fn"):
        total[key] = float(total.get(key, 0.0)) + float(item.get(key, 0.0))
    for key in ("layers", "reasons", "component_patterns", "gt_patterns", "tp_patterns", "partial_patterns", "fn_patterns"):
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
    partial = float(total.get("partial", 0.0))
    fp = float(total.get("fp", 0.0))
    fn = float(total.get("fn", 0.0))
    precision = tp / max(tp + partial + fp, 1)
    recall = tp / max(tp + partial + fn, 1)
    recall_with_partial = (tp + partial) / max(tp + partial + fn, 1)
    return {
        "tp": tp,
        "partial": partial,
        "partial_components": float(total.get("partial_components", 0.0)),
        "fp": fp,
        "fn": fn,
        "true_fn": float(total.get("true_fn", fn)),
        "precision": precision,
        "purity": precision,
        "recall": recall,
        "completeness": recall,
        "recall_with_partial": recall_with_partial,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "gt_reference": total.get("gt_reference", "band_reference_union"),
        "component_patterns": dict(sorted(total.get("component_patterns", {}).items(), key=lambda item: (-item[1], item[0]))),  # type: ignore[union-attr]
        "gt_patterns": dict(sorted(total.get("gt_patterns", {}).items(), key=lambda item: (-item[1], item[0]))),  # type: ignore[union-attr]
        "tp_patterns": dict(sorted(total.get("tp_patterns", {}).items(), key=lambda item: (-item[1], item[0]))),  # type: ignore[union-attr]
        "partial_patterns": dict(sorted(total.get("partial_patterns", {}).items(), key=lambda item: (-item[1], item[0]))),  # type: ignore[union-attr]
        "fn_patterns": dict(sorted(total.get("fn_patterns", {}).items(), key=lambda item: (-item[1], item[0]))),  # type: ignore[union-attr]
        "layers": dict(sorted(total.get("layers", {}).items(), key=lambda item: (-item[1], item[0]))),  # type: ignore[union-attr]
        "reasons": dict(sorted(total.get("reasons", {}).items(), key=lambda item: (-item[1], item[0]))),  # type: ignore[union-attr]
        "examples": total.get("examples", []),
    }


def _init_candidate_stats_bucket() -> Dict[str, object]:
    return {
        "maps": 0,
        "pixels": 0,
        "foreground_gate_active_maps": 0,
        "top_channel_argmax_pass": 0,
        "spatial_localmax_pass": 0,
        "seed_candidates": 0,
        "seed_after_foreground_gate": 0,
        "seed_after_threshold": 0,
        "final_peaks": 0,
        "foreground_gate_pass_pixels": 0,
        "threshold_pass_pixels": 0,
        "center_score_count": 0,
        "center_score_sum": 0.0,
        "center_score_sum_sq": 0.0,
        "center_score_min": float("inf"),
        "center_score_max": float("-inf"),
        "hist_threshold": None,
        "hist_binning": None,
        "hist_edges": [],
        "seed_hist": [],
        "after_foreground_gate_hist": [],
        "final_hist": [],
        "spatial_localmax_argmax_num_channels": 0,
        "spatial_localmax_argmax_hist": [],
    }


def _candidate_hist_edges(threshold: float) -> Tuple[str, List[float]]:
    if float(threshold) > 0.0:
        edges = [-float("inf"), 0.0, 0.5 * float(threshold), float(threshold), 1.5 * float(threshold), 2.0 * float(threshold), 3.0 * float(threshold), 5.0 * float(threshold), float("inf")]
        deduped: List[float] = []
        for edge in edges:
            if deduped and math.isfinite(edge) and math.isfinite(deduped[-1]) and abs(edge - deduped[-1]) < 1e-12:
                continue
            deduped.append(float(edge))
        if len(deduped) < 2:
            deduped = [-float("inf"), float("inf")]
        return "threshold_relative", deduped
    return "absolute_fallback", [-float("inf"), 0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, float("inf")]


def _ensure_candidate_hist_bucket(bucket: Dict[str, object], threshold: float) -> None:
    if bucket.get("hist_edges"):
        return
    mode, edges = _candidate_hist_edges(float(threshold))
    bucket["hist_threshold"] = float(threshold)
    bucket["hist_binning"] = mode
    bucket["hist_edges"] = list(edges)
    zeros = [0 for _ in range(max(len(edges) - 1, 0))]
    bucket["seed_hist"] = list(zeros)
    bucket["after_foreground_gate_hist"] = list(zeros)
    bucket["final_hist"] = list(zeros)


def _add_hist_counts(bucket: Dict[str, object], key: str, values: Tensor) -> None:
    edges_obj = bucket.get("hist_edges", [])
    counts_obj = bucket.get(key, [])
    if not isinstance(edges_obj, list) or len(edges_obj) < 2:
        return
    if not isinstance(counts_obj, list):
        return
    values_np = values.detach().reshape(-1).float().cpu().numpy()
    if values_np.size == 0:
        return
    finite = np.isfinite(values_np)
    if not bool(finite.any()):
        return
    hist, _ = np.histogram(values_np[finite], bins=np.asarray(edges_obj, dtype=np.float64))
    if len(counts_obj) != len(hist):
        return
    for idx, value in enumerate(hist.tolist()):
        counts_obj[idx] = int(counts_obj[idx]) + int(value)


def _ensure_channel_hist_bucket(bucket: Dict[str, object], num_channels: int) -> None:
    num_channels = max(int(num_channels), 0)
    current = int(bucket.get("spatial_localmax_argmax_num_channels", 0))
    if num_channels <= current:
        return
    hist = bucket.get("spatial_localmax_argmax_hist", [])
    if not isinstance(hist, list):
        hist = []
    hist.extend(0 for _ in range(num_channels - len(hist)))
    bucket["spatial_localmax_argmax_num_channels"] = num_channels
    bucket["spatial_localmax_argmax_hist"] = hist


def _update_candidate_stats_bucket(
    bucket: Dict[str, object],
    *,
    center_score: Tensor,
    argmax_channel: Optional[Tensor],
    top_channel_argmax: Optional[Tensor],
    spatial_localmax: Tensor,
    seed_candidates: Tensor,
    foreground_gate: Tensor,
    threshold_pass: Tensor,
    final_peaks: Tensor,
    foreground_gate_active: bool,
    threshold: float,
) -> None:
    _ensure_candidate_hist_bucket(bucket, threshold)

    bucket["maps"] = int(bucket.get("maps", 0)) + 1
    bucket["pixels"] = int(bucket.get("pixels", 0)) + int(center_score.numel())
    bucket["foreground_gate_active_maps"] = int(bucket.get("foreground_gate_active_maps", 0)) + int(bool(foreground_gate_active))
    if top_channel_argmax is not None:
        bucket["top_channel_argmax_pass"] = int(bucket.get("top_channel_argmax_pass", 0)) + int(top_channel_argmax.sum().item())
    bucket["spatial_localmax_pass"] = int(bucket.get("spatial_localmax_pass", 0)) + int(spatial_localmax.sum().item())
    bucket["seed_candidates"] = int(bucket.get("seed_candidates", 0)) + int(seed_candidates.sum().item())
    bucket["seed_after_foreground_gate"] = int(bucket.get("seed_after_foreground_gate", 0)) + int((seed_candidates & foreground_gate).sum().item())
    bucket["seed_after_threshold"] = int(bucket.get("seed_after_threshold", 0)) + int((seed_candidates & threshold_pass).sum().item())
    bucket["final_peaks"] = int(bucket.get("final_peaks", 0)) + int(final_peaks.sum().item())
    bucket["foreground_gate_pass_pixels"] = int(bucket.get("foreground_gate_pass_pixels", 0)) + int(foreground_gate.sum().item())
    bucket["threshold_pass_pixels"] = int(bucket.get("threshold_pass_pixels", 0)) + int(threshold_pass.sum().item())
    if argmax_channel is not None:
        _ensure_channel_hist_bucket(bucket, int(argmax_channel.max().item()) + 1 if argmax_channel.numel() > 0 else 0)
        hist_obj = bucket.get("spatial_localmax_argmax_hist", [])
        if isinstance(hist_obj, list) and argmax_channel.numel() > 0:
            localmax_channels = argmax_channel[spatial_localmax].detach().reshape(-1).to(dtype=torch.long)
            if localmax_channels.numel() > 0:
                bincount = torch.bincount(localmax_channels.cpu(), minlength=len(hist_obj))
                for idx, value in enumerate(bincount.tolist()):
                    hist_obj[idx] = int(hist_obj[idx]) + int(value)

    scores = center_score.detach().reshape(-1).float()
    finite = torch.isfinite(scores)
    if bool(finite.any()):
        finite_scores = scores[finite]
        bucket["center_score_count"] = int(bucket.get("center_score_count", 0)) + int(finite_scores.numel())
        bucket["center_score_sum"] = float(bucket.get("center_score_sum", 0.0)) + float(finite_scores.sum().item())
        bucket["center_score_sum_sq"] = float(bucket.get("center_score_sum_sq", 0.0)) + float((finite_scores * finite_scores).sum().item())
        bucket["center_score_min"] = min(float(bucket.get("center_score_min", float("inf"))), float(finite_scores.min().item()))
        bucket["center_score_max"] = max(float(bucket.get("center_score_max", -float("inf"))), float(finite_scores.max().item()))

    _add_hist_counts(bucket, "seed_hist", center_score[seed_candidates])
    _add_hist_counts(bucket, "after_foreground_gate_hist", center_score[seed_candidates & foreground_gate])
    _add_hist_counts(bucket, "final_hist", center_score[final_peaks])


def _merge_candidate_stats_bucket(dst: Dict[str, object], src: Dict[str, object]) -> None:
    for key in (
        "maps",
        "pixels",
        "foreground_gate_active_maps",
        "top_channel_argmax_pass",
        "spatial_localmax_pass",
        "seed_candidates",
        "seed_after_foreground_gate",
        "seed_after_threshold",
        "final_peaks",
        "foreground_gate_pass_pixels",
        "threshold_pass_pixels",
        "center_score_count",
    ):
        dst[key] = int(dst.get(key, 0)) + int(src.get(key, 0))
    for key in ("center_score_sum", "center_score_sum_sq"):
        dst[key] = float(dst.get(key, 0.0)) + float(src.get(key, 0.0))
    dst["center_score_min"] = min(float(dst.get("center_score_min", float("inf"))), float(src.get("center_score_min", float("inf"))))
    dst["center_score_max"] = max(float(dst.get("center_score_max", -float("inf"))), float(src.get("center_score_max", -float("inf"))))
    _ensure_channel_hist_bucket(dst, max(int(dst.get("spatial_localmax_argmax_num_channels", 0)), int(src.get("spatial_localmax_argmax_num_channels", 0))))

    src_edges = src.get("hist_edges", [])
    if src_edges and not dst.get("hist_edges"):
        dst["hist_threshold"] = src.get("hist_threshold")
        dst["hist_binning"] = src.get("hist_binning")
        dst["hist_edges"] = list(src_edges) if isinstance(src_edges, list) else []
        dst["seed_hist"] = [0 for _ in range(max(len(dst["hist_edges"]) - 1, 0))]  # type: ignore[arg-type]
        dst["after_foreground_gate_hist"] = [0 for _ in range(max(len(dst["hist_edges"]) - 1, 0))]  # type: ignore[arg-type]
        dst["final_hist"] = [0 for _ in range(max(len(dst["hist_edges"]) - 1, 0))]  # type: ignore[arg-type]
    for key in ("seed_hist", "after_foreground_gate_hist", "final_hist"):
        dst_hist = dst.get(key, [])
        src_hist = src.get(key, [])
        if not isinstance(dst_hist, list) or not isinstance(src_hist, list) or len(dst_hist) != len(src_hist):
            continue
        for idx, value in enumerate(src_hist):
            dst_hist[idx] = int(dst_hist[idx]) + int(value)
    dst_channel_hist = dst.get("spatial_localmax_argmax_hist", [])
    src_channel_hist = src.get("spatial_localmax_argmax_hist", [])
    if isinstance(dst_channel_hist, list) and isinstance(src_channel_hist, list):
        if len(dst_channel_hist) < len(src_channel_hist):
            dst_channel_hist.extend(0 for _ in range(len(src_channel_hist) - len(dst_channel_hist)))
        for idx, value in enumerate(src_channel_hist):
            dst_channel_hist[idx] = int(dst_channel_hist[idx]) + int(value)


def _format_candidate_hist_range(lo: float, hi: float) -> str:
    lo_text = "-inf" if not math.isfinite(float(lo)) else f"{float(lo):.3g}"
    hi_text = "inf" if not math.isfinite(float(hi)) else f"{float(hi):.3g}"
    return f"({lo_text}, {hi_text}]"


def _finalize_candidate_stats_bucket(bucket: Dict[str, object]) -> Dict[str, object]:
    pixels = int(bucket.get("pixels", 0))
    maps = int(bucket.get("maps", 0))
    count = int(bucket.get("center_score_count", 0))
    score_sum = float(bucket.get("center_score_sum", 0.0))
    score_sum_sq = float(bucket.get("center_score_sum_sq", 0.0))
    mean = score_sum / max(count, 1)
    var = max(score_sum_sq / max(count, 1) - mean * mean, 0.0)
    edges = bucket.get("hist_edges", [])
    seed_hist = bucket.get("seed_hist", [])
    after_fg_hist = bucket.get("after_foreground_gate_hist", [])
    final_hist = bucket.get("final_hist", [])
    spatial_localmax_argmax_hist = bucket.get("spatial_localmax_argmax_hist", [])
    hist_rows: List[Dict[str, object]] = []
    if isinstance(edges, list) and isinstance(seed_hist, list) and isinstance(after_fg_hist, list) and isinstance(final_hist, list):
        for idx in range(max(len(edges) - 1, 0)):
            hist_rows.append(
                {
                    "range": _format_candidate_hist_range(float(edges[idx]), float(edges[idx + 1])),
                    "seed_candidates": float(seed_hist[idx]) if idx < len(seed_hist) else 0.0,
                    "after_foreground_gate": float(after_fg_hist[idx]) if idx < len(after_fg_hist) else 0.0,
                    "final_peaks": float(final_hist[idx]) if idx < len(final_hist) else 0.0,
                }
            )
    seed_candidates = int(bucket.get("seed_candidates", 0))
    seed_after_foreground_gate = int(bucket.get("seed_after_foreground_gate", 0))
    seed_after_threshold = int(bucket.get("seed_after_threshold", 0))
    final_peaks = int(bucket.get("final_peaks", 0))
    channel_hist_rows: List[Dict[str, object]] = []
    if isinstance(spatial_localmax_argmax_hist, list):
        spatial_localmax_total = max(int(bucket.get("spatial_localmax_pass", 0)), 1)
        for idx, value in enumerate(spatial_localmax_argmax_hist):
            channel_hist_rows.append(
                {
                    "channel": float(idx),
                    "count": float(value),
                    "fraction": float(value) / float(spatial_localmax_total),
                }
            )
    return {
        "maps": float(maps),
        "pixels": float(pixels),
        "foreground_gate_active_maps": float(bucket.get("foreground_gate_active_maps", 0)),
        "foreground_gate_active_fraction": float(bucket.get("foreground_gate_active_maps", 0)) / max(maps, 1),
        "counts": {
            "top_channel_argmax_pass": float(bucket.get("top_channel_argmax_pass", 0)),
            "spatial_localmax_pass": float(bucket.get("spatial_localmax_pass", 0)),
            "seed_candidates": float(seed_candidates),
            "seed_after_foreground_gate": float(seed_after_foreground_gate),
            "seed_after_threshold": float(seed_after_threshold),
            "final_peaks": float(final_peaks),
            "foreground_gate_pass_pixels": float(bucket.get("foreground_gate_pass_pixels", 0)),
            "threshold_pass_pixels": float(bucket.get("threshold_pass_pixels", 0)),
        },
        "retention": {
            "seed_over_pixels": float(seed_candidates) / max(pixels, 1),
            "after_foreground_gate_over_seed": float(seed_after_foreground_gate) / max(seed_candidates, 1),
            "after_threshold_over_seed": float(seed_after_threshold) / max(seed_candidates, 1),
            "final_over_seed": float(final_peaks) / max(seed_candidates, 1),
            "final_over_after_foreground_gate": float(final_peaks) / max(seed_after_foreground_gate, 1),
            "final_over_after_threshold": float(final_peaks) / max(seed_after_threshold, 1),
        },
        "center_score_all_pixels": {
            "count": float(count),
            "mean": mean,
            "std": math.sqrt(var),
            "min": float(bucket.get("center_score_min", float("inf"))) if count > 0 else 0.0,
            "max": float(bucket.get("center_score_max", -float("inf"))) if count > 0 else 0.0,
        },
        "center_score_histogram": {
            "threshold": bucket.get("hist_threshold"),
            "binning": bucket.get("hist_binning"),
            "bins": hist_rows,
        },
        "spatial_localmax_argmax_channel_histogram": channel_hist_rows,
    }


def _init_detection_totals(
    band_names: Sequence[str],
    *,
    collect_candidate_stats: bool = False,
) -> Dict[str, object]:
    totals: Dict[str, object] = {
        "tp": 0,
        "fp": 0,
        "clean_region_fp": 0,
        "fn": 0,
        "ordinary_ignore_tp": 0,
        "ordinary_ignore_fn": 0,
        "ordinary_ignore_total": 0,
        "strict_ignored_pred": 0,
        "linked_tp": 0,
        "linked_fp": 0,
        "linked_fn": 0,
        "collect_candidate_stats": bool(collect_candidate_stats),
        "per_band_counts": {},
        "link_metrics_total": {
            "tp": 0.0,
            "partial": 0.0,
            "partial_components": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "true_fn": 0.0,
            "component_patterns": {},
            "gt_patterns": {},
            "tp_patterns": {},
            "partial_patterns": {},
            "fn_patterns": {},
            "layers": {},
            "reasons": {},
            "examples": [],
            "gt_reference": "band_reference_union_extra_predicted_bands_ignored",
        },
    }
    per_band_counts: Dict[str, Dict[str, object]] = {}
    for name in band_names:
        bucket: Dict[str, object] = {
            "tp": 0,
            "fp": 0,
            "clean_region_fp": 0,
            "fn": 0,
            "ordinary_ignore_tp": 0,
            "ordinary_ignore_fn": 0,
            "ordinary_ignore_total": 0,
            "strict_ignored_pred": 0,
        }
        if bool(collect_candidate_stats):
            bucket["candidate_stats"] = _init_candidate_stats_bucket()
        per_band_counts[str(name)] = bucket
    totals["per_band_counts"] = per_band_counts
    if bool(collect_candidate_stats):
        totals["candidate_stats"] = _init_candidate_stats_bucket()
    return totals


def _merge_detection_totals(items: Sequence[Dict[str, object]], band_names: Sequence[str]) -> Dict[str, object]:
    collect_candidate_stats = any(bool(item.get("collect_candidate_stats", False)) for item in items if isinstance(item, dict))
    merged = _init_detection_totals(band_names, collect_candidate_stats=collect_candidate_stats)
    for item in items:
        for key in (
            "tp",
            "fp",
            "clean_region_fp",
            "fn",
            "ordinary_ignore_tp",
            "ordinary_ignore_fn",
            "ordinary_ignore_total",
            "strict_ignored_pred",
            "linked_tp",
            "linked_fp",
            "linked_fn",
        ):
            merged[key] = int(merged.get(key, 0)) + int(item.get(key, 0))
        merged_candidate = merged.get("candidate_stats")
        item_candidate = item.get("candidate_stats")
        if bool(collect_candidate_stats) and isinstance(merged_candidate, dict) and isinstance(item_candidate, dict):
            _merge_candidate_stats_bucket(merged_candidate, item_candidate)
        merged_per_band = merged["per_band_counts"]
        item_per_band = item.get("per_band_counts", {})
        assert isinstance(merged_per_band, dict)
        if isinstance(item_per_band, dict):
            for band_name, counts_obj in item_per_band.items():
                counts = counts_obj if isinstance(counts_obj, dict) else {}
                bucket = merged_per_band.setdefault(
                    str(band_name),
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
                assert isinstance(bucket, dict)
                for count_key in (
                    "tp",
                    "fp",
                    "clean_region_fp",
                    "fn",
                    "ordinary_ignore_tp",
                    "ordinary_ignore_fn",
                    "ordinary_ignore_total",
                    "strict_ignored_pred",
                ):
                    bucket[count_key] = int(bucket.get(count_key, 0)) + int(counts.get(count_key, 0))
                bucket_candidate = bucket.get("candidate_stats")
                counts_candidate = counts.get("candidate_stats")
                if bool(collect_candidate_stats) and isinstance(bucket_candidate, dict) and isinstance(counts_candidate, dict):
                    _merge_candidate_stats_bucket(bucket_candidate, counts_candidate)
        merged_link = merged["link_metrics_total"]
        item_link = item.get("link_metrics_total")
        assert isinstance(merged_link, dict)
        if isinstance(item_link, dict):
            _accumulate_link_metrics(merged_link, item_link)
    return merged


def _finalize_detection_totals(
    totals: Dict[str, object],
    *,
    band_names: Sequence[str],
    use_ex_link_postprocess: bool,
) -> Dict[str, object]:
    collect_candidate_stats = bool(totals.get("collect_candidate_stats", False))
    tp = int(totals["tp"])
    fp = int(totals["fp"])
    clean_region_fp = int(totals.get("clean_region_fp", fp))
    fn = int(totals["fn"])
    ordinary_tp = int(totals.get("ordinary_ignore_tp", 0))
    ordinary_fn = int(totals.get("ordinary_ignore_fn", 0))
    ordinary_total = int(totals.get("ordinary_ignore_total", ordinary_tp + ordinary_fn))
    strict_ignored_pred = int(totals.get("strict_ignored_pred", 0))
    precision = tp / max(tp + fp, 1)
    clean_region_precision = tp / max(tp + clean_region_fp, 1)
    recall = tp / max(tp + fn, 1)
    combined_tp = tp + ordinary_tp
    combined_fn = fn + ordinary_fn
    combined_precision = combined_tp / max(combined_tp + fp, 1)
    combined_clean_region_precision = combined_tp / max(combined_tp + clean_region_fp, 1)
    combined_recall = combined_tp / max(combined_tp + combined_fn, 1)
    result: Dict[str, object] = {
        "tp": float(tp),
        "fp": float(fp),
        "clean_region_fp": float(clean_region_fp),
        "fn": float(fn),
        "precision": precision,
        "purity": precision,
        "clean_region_precision": clean_region_precision,
        "clean_region_purity": clean_region_precision,
        "recall": recall,
        "completeness": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "strict_ignored_pred": float(strict_ignored_pred),
        "ordinary_ignore": {
            "tp": float(ordinary_tp),
            "fn": float(ordinary_fn),
            "total": float(ordinary_total),
            "recall": ordinary_tp / max(ordinary_tp + ordinary_fn, 1),
            "completeness": ordinary_tp / max(ordinary_tp + ordinary_fn, 1),
        },
        "gt_plus_ordinary_ignore": {
            "tp": float(combined_tp),
            "fp": float(fp),
            "clean_region_fp": float(clean_region_fp),
            "fn": float(combined_fn),
            "precision": combined_precision,
            "purity": combined_precision,
            "clean_region_precision": combined_clean_region_precision,
            "clean_region_purity": combined_clean_region_precision,
            "recall": combined_recall,
            "completeness": combined_recall,
            "f1": 2.0 * combined_precision * combined_recall / max(combined_precision + combined_recall, 1e-12),
        },
    }
    candidate_stats = totals.get("candidate_stats")
    if bool(collect_candidate_stats) and isinstance(candidate_stats, dict):
        result["candidate_stats"] = _finalize_candidate_stats_bucket(candidate_stats)
    per_band_counts = totals.get("per_band_counts", {})
    if band_names and isinstance(per_band_counts, dict):
        per_band: Dict[str, Dict[str, object]] = {}
        for band_name, counts_obj in per_band_counts.items():
            counts = counts_obj if isinstance(counts_obj, dict) else {"tp": 0, "fp": 0, "fn": 0}
            btp = int(counts["tp"])
            bfp = int(counts["fp"])
            b_clean_region_fp = int(counts.get("clean_region_fp", bfp))
            bfn = int(counts["fn"])
            bord_tp = int(counts.get("ordinary_ignore_tp", 0))
            bord_fn = int(counts.get("ordinary_ignore_fn", 0))
            bord_total = int(counts.get("ordinary_ignore_total", bord_tp + bord_fn))
            b_strict_ignored = int(counts.get("strict_ignored_pred", 0))
            b_precision = btp / max(btp + bfp, 1)
            b_clean_region_precision = btp / max(btp + b_clean_region_fp, 1)
            b_recall = btp / max(btp + bfn, 1)
            b_combined_tp = btp + bord_tp
            b_combined_fn = bfn + bord_fn
            b_combined_precision = b_combined_tp / max(b_combined_tp + bfp, 1)
            b_combined_clean_region_precision = b_combined_tp / max(b_combined_tp + b_clean_region_fp, 1)
            b_combined_recall = b_combined_tp / max(b_combined_tp + b_combined_fn, 1)
            per_band[str(band_name)] = {
                "tp": float(btp),
                "fp": float(bfp),
                "clean_region_fp": float(b_clean_region_fp),
                "fn": float(bfn),
                "precision": b_precision,
                "purity": b_precision,
                "clean_region_precision": b_clean_region_precision,
                "clean_region_purity": b_clean_region_precision,
                "recall": b_recall,
                "completeness": b_recall,
                "f1": 2.0 * b_precision * b_recall / max(b_precision + b_recall, 1e-12),
                "strict_ignored_pred": float(b_strict_ignored),
                "ordinary_ignore": {
                    "tp": float(bord_tp),
                    "fn": float(bord_fn),
                    "total": float(bord_total),
                    "recall": bord_tp / max(bord_tp + bord_fn, 1),
                    "completeness": bord_tp / max(bord_tp + bord_fn, 1),
                },
                "gt_plus_ordinary_ignore": {
                    "tp": float(b_combined_tp),
                    "fp": float(bfp),
                    "clean_region_fp": float(b_clean_region_fp),
                    "fn": float(b_combined_fn),
                    "precision": b_combined_precision,
                    "purity": b_combined_precision,
                    "clean_region_precision": b_combined_clean_region_precision,
                    "clean_region_purity": b_combined_clean_region_precision,
                    "recall": b_combined_recall,
                    "completeness": b_combined_recall,
                    "f1": 2.0
                    * b_combined_precision
                    * b_combined_recall
                    / max(b_combined_precision + b_combined_recall, 1e-12),
                },
            }
            band_candidate_stats = counts.get("candidate_stats")
            if bool(collect_candidate_stats) and isinstance(band_candidate_stats, dict):
                per_band[str(band_name)]["candidate_stats"] = _finalize_candidate_stats_bucket(band_candidate_stats)
        result["per_band"] = per_band
    link_metrics_total = totals.get("link_metrics_total")
    if isinstance(link_metrics_total, dict):
        result["link_metrics"] = _finalize_link_metrics(link_metrics_total)
    if use_ex_link_postprocess:
        linked_tp = int(totals["linked_tp"])
        linked_fp = int(totals["linked_fp"])
        linked_fn = int(totals["linked_fn"])
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
    return result
