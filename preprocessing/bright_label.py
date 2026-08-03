"""External bright-source labeling for preprocessing v3.

This module is the non-plotting core of
``data_filtering/build_external_bright_labels_v2.py``.  Image scaling and
bright-component construction live in :mod:`preprocessing.image_processing`;
this file only consumes component labels, refit source geometry, optional
SAT/BAD/EDGE masks, and Gaia rows.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from .image_processing import component_area_map, component_centroid_map
from .labels import SourceClass, SourceLabels
from .refit import RefitConfig, compute_kron_ellipse
from .utils.catalog import source_ids
from .utils.geometry import approximate_ellipse_iou, cluster_sources, component_at


@dataclass(frozen=True)
class BrightLabelConfig:
    isolated_area_max: float = 1000.0
    shape_max_area: float = 10000.0
    drop_area_max: float = 10000.0
    shape_axis_ratio_max: float = 5.0
    component_search_radius: int = 5
    cluster_iou_threshold: float = 1.0 / 3.0
    cluster_max_center_distance: float = 50.0
    cluster_max_area: float = 10000.0
    cluster_source_match_pixels: float = 6.0
    cluster_centroid_match_pixels: float = 10.0
    gaia_bright_mag_threshold: float = 18.0
    pixel_scale_arcsec: float = 0.168
    use_bad_mask_first_step: bool = False
    add_empty_large_bright_component_centers: bool = True
    empty_large_bright_component_area_min: float = 1000.0
    large_component_fast_center_only_source_min: int = 256
    add_unmatched_component_gaia_centers: bool = True


@dataclass
class BrightLabelResult:
    labels: SourceLabels
    strict_center_x: np.ndarray
    strict_center_y: np.ndarray
    strict_center_source_id: np.ndarray
    strict_center_reason: np.ndarray
    strict_center_component_id: np.ndarray
    restricted_source_mask: np.ndarray
    restricted_fallback_component_ids: np.ndarray
    source_rows: list[dict[str, object]]
    component_meta: dict[int, dict[str, object]]
    cluster_rows: list[dict[str, object]]


def finite_float(value: object, default: float = float("nan")) -> float:
    if np.ma.is_masked(value):
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def source_class_from_extendedness(value: float) -> str:
    if not math.isfinite(value):
        return "unknown"
    return "star" if value <= 0.5 else "galaxy"


def source_shape_usable(source: dict[str, object], *, max_area: float, max_axis_ratio: float) -> bool:
    return float(source["area"]) < float(max_area) and float(source["axis_ratio"]) <= float(max_axis_ratio)


def source_intersects_any(source_idx: int, candidate_indices: Sequence[int], sources: list[dict[str, object]]) -> bool:
    for other_idx in candidate_indices:
        if int(other_idx) == int(source_idx):
            continue
        if approximate_ellipse_iou(sources[source_idx], sources[int(other_idx)]) > 0.0:
            return True
    return False


def center_in_quality_mask(source: dict[str, object], quality_mask: np.ndarray | None) -> tuple[bool, str]:
    if quality_mask is None:
        return False, ""
    xi = int(round(float(source["x"])))
    yi = int(round(float(source["y"])))
    if not (0 <= xi < quality_mask.shape[1] and 0 <= yi < quality_mask.shape[0]):
        return False, ""
    return bool(np.asarray(quality_mask, dtype=bool)[yi, xi]), "SAT_BAD_EDGE"


def table_to_bright_sources(
    table: Table,
    candidate: np.ndarray,
    mag: np.ndarray,
    *,
    refit_config: RefitConfig = RefitConfig(),
) -> list[dict[str, object]]:
    geom = compute_kron_ellipse(table, refit_config)
    ids = source_ids(table)
    if "base_ClassificationExtendedness_value" in table.colnames:
        ext_values = np.asarray(table["base_ClassificationExtendedness_value"], dtype=np.float64)
    else:
        ext_values = np.full(len(table), np.nan, dtype=np.float64)
    rows: list[dict[str, object]] = []
    for idx in np.flatnonzero(np.asarray(candidate, dtype=bool)):
        if not (
            np.isfinite(geom.x[idx])
            and np.isfinite(geom.y[idx])
            and np.isfinite(geom.major[idx])
            and np.isfinite(geom.minor[idx])
            and np.isfinite(geom.theta[idx])
            and np.isfinite(geom.area[idx])
            and geom.major[idx] > 0
            and geom.minor[idx] > 0
        ):
            continue
        axis_ratio = max(float(geom.major[idx]), float(geom.minor[idx])) / max(min(float(geom.major[idx]), float(geom.minor[idx])), 1.0e-6)
        rows.append(
            {
                "source_id": int(ids[idx]),
                "table_index": int(idx),
                "row_index": int(idx),
                "x": float(geom.x[idx]),
                "y": float(geom.y[idx]),
                "output_x": float(geom.x[idx]),
                "output_y": float(geom.y[idx]),
                "major": float(geom.major[idx]),
                "minor": float(geom.minor[idx]),
                "theta_deg": math.degrees(float(geom.theta[idx])),
                "area": float(geom.area[idx]),
                "axis_ratio": float(axis_ratio),
                "mag": finite_float(mag[idx]),
                "class": source_class_from_extendedness(finite_float(ext_values[idx])),
                "classification_extendedness": finite_float(ext_values[idx]),
                "existing_label": "",
                "stage_status": "v3_meas_basic_bright",
                "final_label": "",
                "reason": "",
                "component_id": 0,
                "component_area": 0,
                "cluster_id": 0,
                "cluster_size": 0,
                "center_in_bad_mask": False,
                "center_bad_mask": "",
                "cluster_has_bright_gaia": False,
                "gaia_source_id": "",
                "gaia_g_mag": "",
                "gaia_match_arcsec": "",
                "gaia_match_pixels": "",
                "gaia_match_mode": "",
                "measurement_surface": "v3_refit",
            }
        )
    return rows


def project_gaia_rows(
    gaia_table: Table | None,
    *,
    image_shape: tuple[int, int] | None,
    image_header: fits.Header | None,
) -> list[dict[str, object]]:
    if gaia_table is None or len(gaia_table) == 0:
        return []
    if "x" in gaia_table.colnames and "y" in gaia_table.colnames:
        x = np.asarray(gaia_table["x"], dtype=np.float64)
        y = np.asarray(gaia_table["y"], dtype=np.float64)
    elif "ra" in gaia_table.colnames and "dec" in gaia_table.colnames and image_header is not None:
        wcs = WCS(image_header)
        x, y = wcs.all_world2pix(np.asarray(gaia_table["ra"], dtype=np.float64), np.asarray(gaia_table["dec"], dtype=np.float64), 1)
    else:
        return []
    if "source_id" in gaia_table.colnames:
        sid = np.asarray(gaia_table["source_id"], dtype=np.int64)
    else:
        sid = np.arange(len(gaia_table), dtype=np.int64)
    if "phot_g_mean_mag" in gaia_table.colnames:
        gmag = np.asarray(gaia_table["phot_g_mean_mag"], dtype=np.float64)
    else:
        gmag = np.full(len(gaia_table), np.nan, dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(y)
    if image_shape is not None:
        good &= (x >= 0) & (x < image_shape[1]) & (y >= 0) & (y < image_shape[0])
    rows = []
    for idx in np.flatnonzero(good):
        rows.append(
            {
                "source_id": int(sid[idx]),
                "phot_g_mean_mag": finite_float(gmag[idx]),
                "x": float(x[idx]),
                "y": float(y[idx]),
            }
        )
    return rows


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


def gaia_row_key(gaia: dict[str, object]) -> str:
    return str(gaia.get("source_id", f"{float(gaia['x']):.6f},{float(gaia['y']):.6f}"))


def assign_gaia_rows_to_clusters(
    clusters: list[list[int]],
    sources: list[dict[str, object]],
    gaia_rows: list[dict[str, object]],
    *,
    source_match_pixels: float,
    centroid_match_pixels: float,
) -> dict[int, list[tuple[dict[str, object], str, float]]]:
    candidates: list[tuple[float, float, int, dict[str, object], str, float]] = []
    for cluster_pos, cluster in enumerate(clusters):
        for gaia, mode, dist_pix in matching_gaia_rows_to_cluster(
            cluster,
            sources,
            gaia_rows,
            source_match_pixels=source_match_pixels,
            centroid_match_pixels=centroid_match_pixels,
        ):
            gmag = finite_float(gaia.get("phot_g_mean_mag"), float("inf"))
            candidates.append((float(dist_pix), float(gmag), int(cluster_pos), gaia, mode, float(dist_pix)))
    assigned: dict[int, list[tuple[dict[str, object], str, float]]] = defaultdict(list)
    used_gaia: set[str] = set()
    for _dist, _gmag, cluster_pos, gaia, mode, dist_pix in sorted(candidates, key=lambda item: (item[0], item[1])):
        gaia_key = gaia_row_key(gaia)
        if gaia_key in used_gaia:
            continue
        used_gaia.add(gaia_key)
        assigned[int(cluster_pos)].append((gaia, mode, dist_pix))
    return assigned


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
    reason: str = "gaia_direct_strict_center_only",
) -> dict[str, object]:
    gsid = int(round(finite_float(gaia.get("source_id"), 0.0)))
    gmag = finite_float(gaia.get("phot_g_mean_mag"), float("nan"))
    x = float(gaia["x"])
    y = float(gaia["y"])
    return {
        "source_id": -gsid - int(source_id_offset),
        "table_index": -1,
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
        "final_label": "strict_center_only",
        "reason": reason,
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
        "gaia_match_mode": match_mode,
        "measurement_surface": "gaia_direct",
        "strict_center_origin": "gaia",
    }


def synthetic_component_center_source(*, comp: int, component_area: int, x: float, y: float) -> dict[str, object]:
    return {
        "source_id": -(800000000000000000 + int(comp)),
        "table_index": -1,
        "row_index": -1,
        "x": float(x),
        "y": float(y),
        "output_x": float(x),
        "output_y": float(y),
        "major": 3.0,
        "minor": 3.0,
        "theta_deg": 0.0,
        "area": math.pi * 3.0 * 3.0,
        "axis_ratio": 1.0,
        "mag": "",
        "class": "added_bright_component_center",
        "classification_extendedness": "",
        "existing_label": "external_added",
        "stage_status": "synthetic_empty_large_bright_component",
        "final_label": "strict_center_only",
        "reason": "empty_large_bright_component_geometric_center",
        "component_id": int(comp),
        "component_area": int(component_area),
        "cluster_id": 0,
        "cluster_size": 1,
        "center_in_bad_mask": "",
        "center_bad_mask": "",
        "cluster_has_bright_gaia": False,
        "gaia_source_id": "",
        "gaia_g_mag": "",
        "gaia_match_arcsec": "",
        "gaia_match_pixels": "",
        "gaia_match_mode": "",
        "measurement_surface": "bright_component_geometric_center",
        "strict_center_origin": "added_component_center",
    }


def final_label_to_source_class(label: str) -> SourceClass:
    if label == "clean":
        return SourceClass.CLEAN
    if label in {"weak_shape", "center_only", "center_only_external"}:
        return SourceClass.WEAK_SHAPE
    if label == "strict_center_only":
        return SourceClass.STRICT_CENTER_ONLY
    if label == "restricted_bright_region":
        return SourceClass.RESTRICTED_BRIGHT_REGION
    if label == "strict_ignore":
        return SourceClass.STRICT_IGNORE
    return SourceClass.ORDINARY_IGNORE


def classify_no_upper(
    *,
    sources: list[dict[str, object]],
    gaia_rows: list[dict[str, object]],
    image_shape: tuple[int, int],
    quality_mask: np.ndarray | None,
    config: BrightLabelConfig,
) -> tuple[list[dict[str, object]], dict[int, dict[str, object]], list[dict[str, object]], np.ndarray, np.ndarray]:
    del image_shape
    for source in sources:
        in_bad, mask_hit = center_in_quality_mask(source, quality_mask)
        source["center_bad_mask"] = mask_hit
        source["center_in_bad_mask"] = in_bad
    clusters = cluster_sources(
        list(range(len(sources))),
        sources,
        iou_threshold=config.cluster_iou_threshold,
        max_center_distance=config.cluster_max_center_distance,
        max_area=config.cluster_max_area,
    )
    component_meta: dict[int, dict[str, object]] = {}
    cluster_rows: list[dict[str, object]] = []
    next_synthetic = 1
    bright_mask = np.zeros(quality_mask.shape if quality_mask is not None else (1, 1), dtype=bool)
    for cluster_id, cluster in enumerate(clusters, start=1):
        cluster_area_sum = float(sum(float(sources[idx].get("area", 0.0)) for idx in cluster))
        for idx in cluster:
            sources[idx]["component_id"] = cluster_id
            sources[idx]["component_area"] = cluster_area_sum
            sources[idx]["cluster_id"] = cluster_id
            sources[idx]["cluster_size"] = len(cluster)
        component_meta[cluster_id] = {
            "component_id": cluster_id,
            "component_area": cluster_area_sum,
            "hsc_source_count": len(cluster),
                "source_cluster_no_upper": True,
        }
        gaia_matches = matching_gaia_rows_to_cluster(
            cluster,
            sources,
            gaia_rows,
            source_match_pixels=config.cluster_source_match_pixels,
            centroid_match_pixels=config.cluster_centroid_match_pixels,
        )
        if gaia_matches:
            for idx in cluster:
                sources[idx]["final_label"] = "restricted_bright_region"
                sources[idx]["reason"] = "no_upper_gaia_matched_cluster_hsc_fragment_restricted"
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
                        pixel_scale_arcsec=config.pixel_scale_arcsec,
                        match_mode=f"no_upper_{mode}",
                        match_pixels=dist_pix,
                    )
                )
                next_synthetic += 1
            sources.extend(synthetic_rows)
            component_meta[cluster_id]["gaia_count"] = len(gaia_matches)
            component_meta[cluster_id]["has_bright_gaia"] = True
        elif len(cluster) == 1 and not source_intersects_any(cluster[0], list(range(len(sources))), sources):
            source = sources[cluster[0]]
            if float(source["area"]) < float(config.isolated_area_max):
                source["final_label"] = "clean"
                source["reason"] = "no_upper_isolated_small_aperture_clean"
            else:
                source["final_label"] = "center_only_external"
                source["reason"] = "no_upper_isolated_large_aperture_weak_shape"
        else:
            for idx in cluster:
                sources[idx]["final_label"] = "ignore"
                sources[idx]["reason"] = "no_upper_unmatched_cluster_ignore"
    return sources, component_meta, cluster_rows, bright_mask.astype(np.uint8), np.zeros_like(bright_mask, dtype=np.int32)


def classify_component_bright(
    *,
    sources: list[dict[str, object]],
    gaia_rows: list[dict[str, object]],
    component_labels: np.ndarray,
    quality_mask: np.ndarray | None,
    config: BrightLabelConfig,
) -> tuple[list[dict[str, object]], dict[int, dict[str, object]], list[dict[str, object]]]:
    component_areas = component_area_map(component_labels)
    component_centroids = component_centroid_map(component_labels)
    by_component: dict[int, list[int]] = defaultdict(list)
    gaia_by_component: dict[int, list[dict[str, object]]] = defaultdict(list)
    for idx, source in enumerate(sources):
        comp = component_at(component_labels, float(source["x"]), float(source["y"]), config.component_search_radius)
        source["component_id"] = comp
        source["component_area"] = component_areas.get(comp, 0)
        in_bad, mask_hit = center_in_quality_mask(source, quality_mask)
        source["center_bad_mask"] = mask_hit
        source["center_in_bad_mask"] = in_bad
        by_component[comp].append(idx)
    for gaia in gaia_rows:
        comp = component_at(component_labels, float(gaia["x"]), float(gaia["y"]), config.component_search_radius)
        if comp > 0:
            gaia_by_component[comp].append(gaia)
        else:
            gaia_by_component[0].append(gaia)

    component_meta: dict[int, dict[str, object]] = {}
    pending_by_component: dict[int, list[int]] = defaultdict(list)
    for idx, source in enumerate(sources):
        if float(source["area"]) >= float(config.drop_area_max):
            source["final_label"] = "ignore"
            source["reason"] = f"dropped_large_area_ge_{config.drop_area_max:g}"
            continue
        comp = int(source["component_id"])
        if comp <= 0:
            # A bright source whose center is outside the pixel-level bright
            # component should not be discarded here.  The bright mask is a
            # conservative saturated-region detector; ordinary bright galaxies
            # and compact bright sources often sit outside it.  Keep them in a
            # source-cluster branch and let isolation/Gaia decide the label.
            pending_by_component[0].append(idx)
            continue
        source_is_galaxy = str(source.get("class")) == "galaxy"
        has_bright_gaia = any(finite_float(row.get("phot_g_mean_mag"), float("inf")) <= config.gaia_bright_mag_threshold for row in gaia_by_component.get(comp, []))
        if config.use_bad_mask_first_step and bool(source["center_in_bad_mask"]) and not source_is_galaxy and not has_bright_gaia:
            source["final_label"] = "ignore"
            source["reason"] = f"bad_mask_non_galaxy_no_bright_gaia:{source['center_bad_mask']}"
            continue
        pending_by_component[comp].append(idx)

    cluster_rows: list[dict[str, object]] = []
    next_cluster_id = 1
    next_synthetic = 1

    def bright_gaia_rows(comp: int) -> list[dict[str, object]]:
        return [
            gaia
            for gaia in gaia_by_component.get(comp, [])
            if finite_float(gaia.get("phot_g_mean_mag"), float("inf")) <= float(config.gaia_bright_mag_threshold)
        ]

    def add_unmatched_gaia_centers_for_component(
        comp: int,
        *,
        cluster_id: int,
        cluster_size: int,
        used_gaia_keys: set[str],
        reason: str,
        match_mode: str,
    ) -> int:
        nonlocal next_synthetic
        if not config.add_unmatched_component_gaia_centers:
            return 0
        added = 0
        for gaia in sorted(
            bright_gaia_rows(comp),
            key=lambda row: (finite_float(row.get("phot_g_mean_mag"), float("inf")), finite_float(row.get("source_id"), 0.0)),
        ):
            key = gaia_row_key(gaia)
            if key in used_gaia_keys:
                continue
            sources.append(
                synthetic_gaia_source(
                    gaia,
                    comp=comp,
                    component_area=component_areas.get(comp, 0),
                    cluster_id=cluster_id,
                    cluster_size=cluster_size,
                    source_id_offset=next_synthetic,
                    pixel_scale_arcsec=config.pixel_scale_arcsec,
                    match_mode=match_mode,
                    match_pixels=0.0,
                    reason=reason,
                )
            )
            used_gaia_keys.add(key)
            next_synthetic += 1
            added += 1
        return added

    for comp in sorted(pending_by_component):
        component_meta[comp] = {
            "component_id": comp,
            "component_area": component_areas.get(comp, 0),
            "hsc_source_count": len(pending_by_component[comp]),
            "gaia_count": len(gaia_by_component.get(comp, [])),
        }
        if (
            comp > 0
            and config.large_component_fast_center_only_source_min > 0
            and len(pending_by_component[comp]) >= config.large_component_fast_center_only_source_min
        ):
            cluster_id = next_cluster_id
            next_cluster_id += 1
            added = add_unmatched_gaia_centers_for_component(
                comp,
                cluster_id=cluster_id,
                cluster_size=len(pending_by_component[comp]),
                used_gaia_keys=set(),
                reason="large_component_gaia_strict_center_only",
                match_mode="large_component_membership",
            )
            component_meta[comp]["unmatched_component_gaia_added"] = added
            for idx in pending_by_component[comp]:
                sources[idx]["cluster_id"] = cluster_id
                sources[idx]["cluster_size"] = len(pending_by_component[comp])
                sources[idx]["final_label"] = "restricted_bright_region"
                sources[idx]["reason"] = "large_bright_component_fast_restricted_fragment"
            continue
        clusters = cluster_sources(
            pending_by_component[comp],
            sources,
            iou_threshold=config.cluster_iou_threshold,
            max_center_distance=config.cluster_max_center_distance,
            max_area=config.cluster_max_area,
        )
        gaia_by_cluster = assign_gaia_rows_to_clusters(
            clusters,
            sources,
            gaia_by_component.get(comp, []),
            source_match_pixels=config.cluster_source_match_pixels,
            centroid_match_pixels=config.cluster_centroid_match_pixels,
        )
        used_gaia_keys = {
            gaia_row_key(gaia)
            for matches in gaia_by_cluster.values()
            for gaia, _mode, _dist_pix in matches
        }
        component_meta[comp]["cluster_count"] = len(clusters)
        component_is_singleton_cluster = comp > 0 and len(clusters) == 1 and len(clusters[0]) == 1
        for cluster_pos, cluster in enumerate(clusters):
            cluster_id = next_cluster_id
            next_cluster_id += 1
            for idx in cluster:
                sources[idx]["cluster_id"] = cluster_id
                sources[idx]["cluster_size"] = len(cluster)
            cluster_finalized = False
            matched_gaia = gaia_by_cluster.get(cluster_pos, [])
            if matched_gaia:
                for idx in cluster:
                    sources[idx]["final_label"] = "restricted_bright_region"
                    sources[idx]["reason"] = "gaia_matched_cluster_hsc_fragment_restricted"
                for gaia, mode, dist_pix in sorted(matched_gaia, key=lambda item: (finite_float(item[0].get("phot_g_mean_mag"), float("inf")), item[2])):
                    sources.append(
                        synthetic_gaia_source(
                            gaia,
                            comp=comp,
                            component_area=component_areas.get(comp, 0),
                            cluster_id=cluster_id,
                            cluster_size=len(cluster) + len(matched_gaia),
                            source_id_offset=next_synthetic,
                            pixel_scale_arcsec=config.pixel_scale_arcsec,
                            match_mode=f"cluster_{mode}",
                            match_pixels=dist_pix,
                            reason="gaia_matched_cluster_strict_center_only",
                        )
                    )
                    next_synthetic += 1
                cluster_finalized = True
            if not cluster_finalized and len(cluster) == 1 and (
                (comp > 0 and component_is_singleton_cluster)
                or (comp <= 0 and not source_intersects_any(cluster[0], pending_by_component[comp], sources))
            ):
                only = sources[cluster[0]]
                only_shape_ok = source_shape_usable(
                    only,
                    max_area=config.shape_max_area,
                    max_axis_ratio=config.shape_axis_ratio_max,
                )
                comp_area = int(component_areas.get(comp, 0))
                if comp <= 0:
                    comp_area = int(round(float(only.get("area", 0.0))))
                size_area = max(float(comp_area), float(only.get("area", 0.0)))
                if size_area < config.isolated_area_max and only_shape_ok:
                    only["final_label"] = "clean"
                    only["reason"] = "isolated_small_bright_source_clean" if comp <= 0 else "single_cluster_small_bright_component_clean"
                    cluster_finalized = True
                elif only_shape_ok:
                    only["final_label"] = "center_only_external"
                    only["reason"] = "isolated_bright_source_weak_shape" if comp <= 0 else "single_cluster_large_bright_component_weak_shape"
                    cluster_finalized = True
            if not cluster_finalized:
                for idx in cluster:
                    if comp > 0:
                        sources[idx]["final_label"] = "restricted_bright_region"
                        sources[idx]["reason"] = "bright_component_unmatched_cluster_restricted"
                    else:
                        sources[idx]["final_label"] = "ignore"
                        sources[idx]["reason"] = "bright_cluster_no_gaia_match_ignore"
            cluster_rows.append(
                {
                    "cluster_id": cluster_id,
                    "component_id": comp,
                    "component_area": component_areas.get(comp, 0),
                    "cluster_size": len(cluster),
                    "source_ids": " ".join(str(sources[idx]["source_id"]) for idx in cluster),
                }
            )
        if comp > 0:
            cluster_id = next_cluster_id
            added = add_unmatched_gaia_centers_for_component(
                comp,
                cluster_id=cluster_id,
                cluster_size=len(gaia_by_component.get(comp, [])),
                used_gaia_keys=used_gaia_keys,
                reason="component_unmatched_gaia_strict_center_only",
                match_mode="component_membership",
            )
            if added:
                next_cluster_id += 1
                component_meta[comp]["unmatched_component_gaia_added"] = added
    if config.add_empty_large_bright_component_centers:
        supervised = {
            int(source.get("component_id", 0))
            for source in sources
            if str(source.get("final_label", "")) in {"clean", "center_only_external", "strict_center_only"}
        }
        for comp, area in sorted(component_areas.items()):
            if comp <= 0 or area < config.empty_large_bright_component_area_min or comp in supervised:
                continue
            if comp not in component_centroids:
                continue
            x, y = component_centroids[comp]
            sources.append(synthetic_component_center_source(comp=comp, component_area=int(area), x=x, y=y))
            component_meta.setdefault(comp, {})["added_empty_large_center"] = True
    return sources, component_meta, cluster_rows


def apply_bright_rows_to_labels(
    sources: list[dict[str, object]],
    labels: SourceLabels,
    *,
    n_table_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    restricted = np.zeros(n_table_rows, dtype=bool)
    strict_x: list[float] = []
    strict_y: list[float] = []
    strict_id: list[int] = []
    strict_reason: list[str] = []
    strict_component_id: list[int] = []
    fallback_components: set[int] = set()
    for source in sources:
        label = str(source.get("final_label", "ignore"))
        source_class = final_label_to_source_class(label)
        table_index = int(source.get("table_index", -1))
        reason = str(source.get("reason", label))
        if table_index >= 0:
            mask = np.zeros(n_table_rows, dtype=bool)
            mask[table_index] = True
            labels.assign(mask, source_class, reason)
            if source_class == SourceClass.RESTRICTED_BRIGHT_REGION:
                restricted[table_index] = True
        elif source_class == SourceClass.STRICT_CENTER_ONLY:
            strict_x.append(float(source["output_x"]))
            strict_y.append(float(source["output_y"]))
            strict_id.append(int(source["source_id"]))
            strict_reason.append(reason)
            component_id = int(source.get("component_id", 0))
            strict_component_id.append(component_id)
            if component_id > 0:
                fallback_components.add(component_id)
    return (
        np.asarray(strict_x, dtype=np.float64),
        np.asarray(strict_y, dtype=np.float64),
        np.asarray(strict_id, dtype=np.int64),
        np.asarray(strict_reason, dtype=object),
        np.asarray(strict_component_id, dtype=np.int32),
        restricted,
        np.asarray(sorted(fallback_components), dtype=np.int32),
    )


def label_bright_sources(
    table: Table,
    candidate: np.ndarray,
    labels: SourceLabels,
    *,
    bright_region: np.ndarray | None,
    component_labels: np.ndarray | None = None,
    gaia_table: Table | None = None,
    gaia_rows: list[dict[str, object]] | None = None,
    image_header: fits.Header | None = None,
    quality_mask: np.ndarray | None = None,
    mag: np.ndarray | None = None,
    config: BrightLabelConfig = BrightLabelConfig(),
    refit_config: RefitConfig = RefitConfig(),
) -> BrightLabelResult:
    """Classify bright candidates with the v2 Gaia/component logic."""

    candidate = np.asarray(candidate, dtype=bool)
    if mag is None:
        mag = np.full(len(table), np.nan, dtype=np.float64)
    sources = table_to_bright_sources(table, candidate, np.asarray(mag, dtype=np.float64), refit_config=refit_config)
    image_shape = None
    if bright_region is not None:
        image_shape = np.asarray(bright_region).shape
    elif quality_mask is not None:
        image_shape = np.asarray(quality_mask).shape
    if gaia_rows is None:
        gaia_rows = project_gaia_rows(gaia_table, image_shape=image_shape, image_header=image_header)
    if component_labels is not None and np.asarray(component_labels).size and int(np.max(component_labels)) > 0:
        sources, component_meta, cluster_rows = classify_component_bright(
            sources=sources,
            gaia_rows=gaia_rows,
            component_labels=np.asarray(component_labels, dtype=np.int32),
            quality_mask=quality_mask,
            config=config,
        )
    else:
        sources, component_meta, cluster_rows, _mask, _labels = classify_no_upper(
            sources=sources,
            gaia_rows=gaia_rows,
            image_shape=image_shape or (1, 1),
            quality_mask=quality_mask,
            config=config,
        )
    strict_x, strict_y, strict_id, strict_reason, strict_component_id, restricted, fallback_components = apply_bright_rows_to_labels(
        sources,
        labels,
        n_table_rows=len(table),
    )
    return BrightLabelResult(
        labels=labels,
        strict_center_x=strict_x,
        strict_center_y=strict_y,
        strict_center_source_id=strict_id,
        strict_center_reason=strict_reason,
        strict_center_component_id=strict_component_id,
        restricted_source_mask=restricted,
        restricted_fallback_component_ids=fallback_components,
        source_rows=sources,
        component_meta=component_meta,
        cluster_rows=cluster_rows,
    )
