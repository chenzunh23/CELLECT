#!/usr/bin/env python3
"""Core external bright-source label logic without plotting or region output."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from astropy.io import fits
from scipy import ndimage


@dataclass(frozen=True)
class NoUpperBrightLabelConfig:
    cluster_iou_threshold: float = 1.0 / 3.0
    cluster_max_center_distance: float = 50.0
    cluster_max_area: float = 10000.0
    cluster_source_match_pixels: float = 6.0
    cluster_centroid_match_pixels: float = 10.0
    isolated_clean_area_max: float = 1000.0
    pixel_scale_arcsec: float = 0.168


def finite_float(value: object, default: float = float("nan")) -> float:
    if np.ma.is_masked(value):
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def mask_bits(header: fits.Header, names: Iterable[str]) -> dict[str, int]:
    return {str(name): int(header[f"MP_{name}"]) for name in names if f"MP_{name}" in header}


def source_center_in_bad_mask(
    source: dict[str, object],
    mask: np.ndarray,
    bits: dict[str, int],
) -> tuple[bool, str]:
    xi = int(round(float(source["x"])))
    yi = int(round(float(source["y"])))
    if not (0 <= xi < mask.shape[1] and 0 <= yi < mask.shape[0]):
        return False, ""
    value = int(mask[yi, xi])
    names = [name for name, bit in bits.items() if value & (1 << int(bit))]
    return bool(names), ",".join(names)


def ellipse_contains(row: dict[str, object], xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    x0 = float(row["x"])
    y0 = float(row["y"])
    major = max(float(row["major"]), 1.0e-6)
    minor = max(float(row["minor"]), 1.0e-6)
    theta = math.radians(float(row["theta_deg"]))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    dx = xs - x0
    dy = ys - y0
    xr = dx * cos_t + dy * sin_t
    yr = -dx * sin_t + dy * cos_t
    return (xr / major) ** 2 + (yr / minor) ** 2 <= 1.0


def approximate_ellipse_iou(a: dict[str, object], b: dict[str, object]) -> float:
    max_a = max(float(a["major"]), float(a["minor"]))
    max_b = max(float(b["major"]), float(b["minor"]))
    if math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"])) > max_a + max_b:
        return 0.0
    x0 = math.floor(min(float(a["x"]) - max_a, float(b["x"]) - max_b))
    x1 = math.ceil(max(float(a["x"]) + max_a, float(b["x"]) + max_b))
    y0 = math.floor(min(float(a["y"]) - max_a, float(b["y"]) - max_b))
    y1 = math.ceil(max(float(a["y"]) + max_a, float(b["y"]) + max_b))
    if x1 < x0 or y1 < y0:
        return 0.0
    step = max(1, int(math.ceil(max(x1 - x0 + 1, y1 - y0 + 1) / 512.0)))
    yy, xx = np.mgrid[y0 : y1 + 1 : step, x0 : x1 + 1 : step]
    ma = ellipse_contains(a, xx.astype(np.float32), yy.astype(np.float32))
    mb = ellipse_contains(b, xx.astype(np.float32), yy.astype(np.float32))
    union = int(np.count_nonzero(ma | mb))
    if union == 0:
        return 0.0
    return float(np.count_nonzero(ma & mb) / union)


class UnionFind:
    def __init__(self, items: list[int]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: int) -> int:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def groups(self) -> list[list[int]]:
        out: dict[int, list[int]] = defaultdict(list)
        for item in self.parent:
            out[self.find(item)].append(item)
        return [sorted(group) for group in out.values()]


def cluster_sources(
    indices: list[int],
    sources: list[dict[str, object]],
    *,
    iou_threshold: float,
    max_center_distance: float,
    max_area: float,
) -> list[list[int]]:
    if len(indices) <= 1:
        return [indices] if indices else []
    uf = UnionFind(indices)
    for pos, i in enumerate(indices):
        si = sources[i]
        if float(si.get("area", 0.0)) >= float(max_area):
            continue
        for j in indices[pos + 1 :]:
            sj = sources[j]
            if float(sj.get("area", 0.0)) >= float(max_area):
                continue
            if math.hypot(float(si["x"]) - float(sj["x"]), float(si["y"]) - float(sj["y"])) > float(max_center_distance):
                continue
            if approximate_ellipse_iou(si, sj) >= float(iou_threshold):
                uf.union(i, j)
    return uf.groups()


def source_intersects_any(
    source_idx: int,
    candidate_indices: list[int],
    sources: list[dict[str, object]],
) -> bool:
    for other_idx in candidate_indices:
        if int(other_idx) == int(source_idx):
            continue
        if approximate_ellipse_iou(sources[source_idx], sources[other_idx]) > 0.0:
            return True
    return False


def matching_gaia_rows_to_cluster(
    cluster: list[int],
    sources: list[dict[str, object]],
    gaia_rows: list[dict[str, object]],
    *,
    source_match_pixels: float,
    centroid_match_pixels: float,
) -> list[tuple[dict[str, object], str, float]]:
    if not cluster:
        return []
    centroid_x = float(np.mean([float(sources[idx]["x"]) for idx in cluster]))
    centroid_y = float(np.mean([float(sources[idx]["y"]) for idx in cluster]))
    matches: list[tuple[dict[str, object], str, float]] = []
    for gaia in gaia_rows:
        gx = float(gaia["x"])
        gy = float(gaia["y"])
        source_dist = min(math.hypot(float(sources[idx]["x"]) - gx, float(sources[idx]["y"]) - gy) for idx in cluster)
        centroid_dist = math.hypot(gx - centroid_x, gy - centroid_y)
        if source_dist <= float(source_match_pixels):
            matches.append((gaia, "source_center", source_dist))
        elif centroid_dist <= float(centroid_match_pixels):
            matches.append((gaia, "cluster_centroid", centroid_dist))
    return sorted(matches, key=lambda item: (finite_float(item[0].get("phot_g_mean_mag"), float("inf")), item[2]))


def synthetic_gaia_source(
    gaia: dict[str, object],
    *,
    comp: int,
    component_area: int,
    cluster_id: int,
    cluster_size: int,
    source_id_offset: int,
    pixel_scale_arcsec: float,
    match_mode: str,
    match_pixels: float,
) -> dict[str, object]:
    gsid = int(round(finite_float(gaia.get("source_id"), 0.0)))
    gmag = finite_float(gaia.get("phot_g_mean_mag"), float("nan"))
    x = float(gaia["x"])
    y = float(gaia["y"])
    return {
        "source_id": -gsid - int(source_id_offset),
        "row_index": -1,
        "x": x,
        "y": y,
        "output_x": x,
        "output_y": y,
        "major": 3.0,
        "minor": 3.0,
        "theta_deg": 0.0,
        "area": math.pi * 3.0 * 3.0,
        "axis_ratio": 1.0,
        "mag": gmag,
        "class": "gaia_star",
        "classification_extendedness": "",
        "existing_label": "external_gaia",
        "stage_status": "synthetic_bright_gaia",
        "final_label": "strict_center_only_external",
        "reason": "no_upper_gaia_direct_strict_center_only",
        "component_id": comp,
        "component_area": component_area,
        "cluster_id": cluster_id,
        "cluster_size": cluster_size,
        "center_in_bad_mask": "",
        "center_bad_mask": "",
        "cluster_has_bright_gaia": True,
        "gaia_source_id": gaia.get("source_id", ""),
        "gaia_g_mag": gaia.get("phot_g_mean_mag", ""),
        "gaia_match_arcsec": float(match_pixels) * float(pixel_scale_arcsec),
        "gaia_match_pixels": float(match_pixels),
        "gaia_match_mode": f"no_upper_{match_mode}",
        "measurement_surface": "gaia_direct",
    }


def draw_source_ellipse(mask: np.ndarray, row: dict[str, object]) -> None:
    x0 = float(row["x"])
    y0 = float(row["y"])
    major = max(float(row["major"]), 1.0e-6)
    minor = max(float(row["minor"]), 1.0e-6)
    theta = math.radians(float(row["theta_deg"]))
    radius = int(math.ceil(max(major, minor)))
    xi = int(round(x0))
    yi = int(round(y0))
    x_min = max(0, xi - radius)
    x_max = min(mask.shape[1], xi + radius + 1)
    y_min = max(0, yi - radius)
    y_max = min(mask.shape[0], yi + radius + 1)
    if x_max <= x_min or y_max <= y_min:
        return
    yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
    inside = ellipse_contains(row, xx.astype(np.float32), yy.astype(np.float32))
    mask[y_min:y_max, x_min:x_max] |= inside


def assign_no_upper_source_clusters(
    *,
    sources: list[dict[str, object]],
    gaia_rows: list[dict[str, object]],
    image_shape: tuple[int, int],
    mask: np.ndarray,
    mask_header: fits.Header,
    mask_names: list[str],
    config: NoUpperBrightLabelConfig = NoUpperBrightLabelConfig(),
) -> tuple[list[dict[str, object]], dict[int, dict[str, object]], list[dict[str, object]], np.ndarray, np.ndarray]:
    bits = mask_bits(mask_header, mask_names)
    for source in sources:
        in_bad, mask_hit = source_center_in_bad_mask(source, mask, bits)
        source["center_bad_mask"] = mask_hit
        source["center_in_bad_mask"] = in_bad

    bright_mask = np.zeros(image_shape, dtype=bool)
    clusters = cluster_sources(
        list(range(len(sources))),
        sources,
        iou_threshold=float(config.cluster_iou_threshold),
        max_center_distance=float(config.cluster_max_center_distance),
        max_area=float(config.cluster_max_area),
    )
    component_meta: dict[int, dict[str, object]] = {}
    cluster_rows: list[dict[str, object]] = []
    next_synthetic = 1
    for cluster_id, cluster in enumerate(clusters, start=1):
        cluster_area_sum = float(sum(float(sources[idx].get("area", 0.0)) for idx in cluster))
        for idx in cluster:
            sources[idx]["component_id"] = cluster_id
            sources[idx]["component_area"] = cluster_area_sum
            sources[idx]["cluster_id"] = cluster_id
            sources[idx]["cluster_size"] = len(cluster)
            sources[idx]["cluster_has_bright_gaia"] = False
        component_meta[cluster_id] = {
            "component_id": cluster_id,
            "component_area": cluster_area_sum,
            "hsc_source_count": len(cluster),
            "gaia_count": 0,
            "has_bright_gaia": False,
            "source_cluster_no_upper": True,
        }

        if len(cluster) == 1 and not source_intersects_any(cluster[0], list(range(len(sources))), sources):
            source = sources[cluster[0]]
            if float(source["area"]) < float(config.isolated_clean_area_max):
                source["final_label"] = "clean"
                source["reason"] = "no_upper_isolated_small_aperture_clean"
            else:
                source["final_label"] = "center_only_external"
                source["reason"] = "no_upper_isolated_large_aperture_center_only"
            cluster_rows.append(
                {
                    "cluster_id": cluster_id,
                    "component_id": cluster_id,
                    "component_area": cluster_area_sum,
                    "cluster_size": len(cluster),
                    "cluster_in_bad_mask": bool(source.get("center_in_bad_mask", False)),
                    "component_has_bright_gaia": False,
                    "chosen_source_id": source["source_id"],
                    "chosen_final_label": source["final_label"],
                    "chosen_reason": source["reason"],
                    "gaia_source_id": "",
                    "gaia_g_mag": "",
                    "gaia_match_arcsec": "",
                    "gaia_match_pixels": "",
                    "gaia_match_mode": "",
                    "source_ids": str(source["source_id"]),
                }
            )
            continue

        gaia_matches = matching_gaia_rows_to_cluster(
            cluster,
            sources,
            gaia_rows,
            source_match_pixels=float(config.cluster_source_match_pixels),
            centroid_match_pixels=float(config.cluster_centroid_match_pixels),
        )
        cluster_in_bad = any(bool(sources[idx].get("center_in_bad_mask", False)) for idx in cluster)
        if gaia_matches:
            for idx in cluster:
                sources[idx]["final_label"] = "restricted_bright_region"
                sources[idx]["reason"] = "no_upper_gaia_matched_cluster_hsc_fragment_restricted"
                draw_source_ellipse(bright_mask, sources[idx])
            synthetic_rows = []
            for gaia, mode, dist_pix in gaia_matches:
                synthetic_rows.append(
                    synthetic_gaia_source(
                        gaia,
                        comp=cluster_id,
                        component_area=int(round(cluster_area_sum)),
                        cluster_id=cluster_id,
                        cluster_size=len(cluster) + len(gaia_matches),
                        source_id_offset=next_synthetic,
                        pixel_scale_arcsec=float(config.pixel_scale_arcsec),
                        match_mode=mode,
                        match_pixels=dist_pix,
                    )
                )
                next_synthetic += 1
            sources.extend(synthetic_rows)
            component_meta[cluster_id]["gaia_count"] = len(gaia_matches)
            component_meta[cluster_id]["has_bright_gaia"] = True
            cluster_rows.append(
                {
                    "cluster_id": cluster_id,
                    "component_id": cluster_id,
                    "component_area": cluster_area_sum,
                    "cluster_size": len(cluster),
                    "cluster_in_bad_mask": cluster_in_bad,
                    "component_has_bright_gaia": True,
                    "chosen_source_id": " ".join(str(row["source_id"]) for row in synthetic_rows),
                    "chosen_final_label": "strict_center_only_external",
                    "chosen_reason": "no_upper_gaia_direct_strict_center_only",
                    "gaia_source_id": " ".join(str(row["gaia_source_id"]) for row in synthetic_rows),
                    "gaia_g_mag": " ".join(str(row["gaia_g_mag"]) for row in synthetic_rows),
                    "gaia_match_arcsec": " ".join(str(row["gaia_match_arcsec"]) for row in synthetic_rows),
                    "gaia_match_pixels": " ".join(str(row["gaia_match_pixels"]) for row in synthetic_rows),
                    "gaia_match_mode": "no_upper_gaia_direct",
                    "source_ids": " ".join(str(sources[idx]["source_id"]) for idx in cluster),
                }
            )
        else:
            for idx in cluster:
                sources[idx]["final_label"] = "ignore"
                sources[idx]["reason"] = "no_upper_unmatched_cluster_ignore"
            cluster_rows.append(
                {
                    "cluster_id": cluster_id,
                    "component_id": cluster_id,
                    "component_area": cluster_area_sum,
                    "cluster_size": len(cluster),
                    "cluster_in_bad_mask": cluster_in_bad,
                    "component_has_bright_gaia": False,
                    "chosen_source_id": "",
                    "chosen_final_label": "ignore",
                    "chosen_reason": "no_upper_unmatched_cluster_ignore",
                    "gaia_source_id": "",
                    "gaia_g_mag": "",
                    "gaia_match_arcsec": "",
                    "gaia_match_pixels": "",
                    "gaia_match_mode": "",
                    "source_ids": " ".join(str(sources[idx]["source_id"]) for idx in cluster),
                }
            )
    component_labels, _num = ndimage.label(bright_mask)
    return sources, component_meta, cluster_rows, bright_mask.astype(np.uint8), component_labels.astype(np.int32)
