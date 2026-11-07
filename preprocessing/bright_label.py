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
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from .image_processing import component_area_map, component_centroid_map
from .utils.gaia import matching_radius_pixels, observation_time, propagated_gaia_radec, astrometric_evidence
from .labels import SourceClass, SourceLabels
from .refit import RefitConfig, compute_kron_ellipse
from .utils.catalog import source_ids
from .utils.geometry import approximate_ellipse_iou, cluster_sources, component_at, ellipse_bbox, ellipse_contains_dict


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
    gaia_bright_mag_threshold: float = 22.0
    pixel_scale_arcsec: float = 0.168
    gaia_source_match_arcsec: float | None = None
    gaia_centroid_match_arcsec: float | None = None
    use_bad_mask_first_step: bool = False
    add_empty_large_bright_component_centers: bool = True
    empty_large_bright_component_area_min: float = 1000.0
    empty_seeded_bright_component_area_min: float = 64.0
    blendedness_ignore_threshold: float = 0.5
    large_component_fast_center_only_source_min: int = 256
    add_unmatched_component_gaia_centers: bool = True
    add_nested_bright_component_centers: bool = True


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
    ordinary_ignore_component_ids: np.ndarray
    ordinary_ignore_source_mask: np.ndarray
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


def source_region_fully_inside_component(source: dict[str, object], component_labels: np.ndarray, comp: int) -> bool:
    if comp <= 0:
        return False
    if not all(np.isfinite(finite_float(source.get(key))) for key in ("x", "y", "major", "minor", "theta_deg")):
        return False
    ys, xs = ellipse_bbox(
        float(source["x"]),
        float(source["y"]),
        float(source["major"]),
        float(source["minor"]),
        math.radians(float(source["theta_deg"])),
        component_labels.shape,
    )
    if ys.stop <= ys.start or xs.stop <= xs.start:
        return False
    yy, xx = np.mgrid[ys, xs]
    inside_source = ellipse_contains_dict(source, xx.astype(np.float32), yy.astype(np.float32))
    if not bool(np.any(inside_source)):
        return False
    return bool(np.all(np.asarray(component_labels[ys, xs], dtype=np.int32)[inside_source] == int(comp)))


def component_touches_image_boundary(component_labels: np.ndarray, comp: int) -> bool:
    if comp <= 0:
        return False
    labels = np.asarray(component_labels, dtype=np.int32)
    if labels.size == 0:
        return False
    return bool(
        np.any(labels[0, :] == comp)
        or np.any(labels[-1, :] == comp)
        or np.any(labels[:, 0] == comp)
        or np.any(labels[:, -1] == comp)
    )


def source_blendedness(source: dict[str, object]) -> float:
    for key in ("blendedness_abs", "blendedness_raw"):
        value = finite_float(source.get(key))
        if math.isfinite(value):
            return value
    return float("nan")


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
    blended_abs = np.asarray(table["base_Blendedness_abs"], dtype=np.float64) if "base_Blendedness_abs" in table.colnames else np.full(len(table), np.nan, dtype=np.float64)
    blended_raw = np.asarray(table["base_Blendedness_raw"], dtype=np.float64) if "base_Blendedness_raw" in table.colnames else np.full(len(table), np.nan, dtype=np.float64)
    blended_old = np.asarray(table["base_Blendedness_old"], dtype=np.float64) if "base_Blendedness_old" in table.colnames else np.full(len(table), np.nan, dtype=np.float64)
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
                "blendedness_abs": finite_float(blended_abs[idx]),
                "blendedness_raw": finite_float(blended_raw[idx]),
                "blendedness_old": finite_float(blended_old[idx]),
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
    pixel_origin: int = 0,
    reference_epoch: float | None = None,
) -> list[dict[str, object]]:
    if gaia_table is None or len(gaia_table) == 0:
        return []
    if reference_epoch is not None and 'ref_epoch' not in gaia_table.colnames:
        gaia_table=gaia_table.copy();gaia_table['ref_epoch']=reference_epoch
    reliable=astrometric_evidence(gaia_table,allow_absolute_motion=False)
    moved=np.zeros(len(gaia_table),bool)
    if "x" in gaia_table.colnames and "y" in gaia_table.colnames:
        x = np.asarray(gaia_table["x"], dtype=np.float64)
        y = np.asarray(gaia_table["y"], dtype=np.float64)
    elif "ra" in gaia_table.colnames and "dec" in gaia_table.colnames and image_header is not None:
        wcs = WCS(image_header)
        ra, dec, moved = propagated_gaia_radec(gaia_table, observation_time(image_header))
        x, y = wcs.all_world2pix(ra, dec, pixel_origin)
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
                "phot_rp_mean_mag": finite_float(gaia_table['phot_rp_mean_mag'][idx]) if 'phot_rp_mean_mag' in gaia_table.colnames else float('nan'),
                "astrometric_evidence": bool(reliable[idx]),
                "position_propagated": bool(moved[idx]),
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
    gsid = int(gaia.get("source_id", 0))
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
        "blendedness_abs": "",
        "blendedness_raw": "",
        "blendedness_old": "",
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


def synthetic_component_center_source(
    *,
    comp: int,
    component_area: int,
    x: float,
    y: float,
    reason: str = "empty_large_bright_component_geometric_center",
) -> dict[str, object]:
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
        "blendedness_abs": "",
        "blendedness_raw": "",
        "blendedness_old": "",
        "existing_label": "external_added",
        "stage_status": "synthetic_empty_large_bright_component",
        "final_label": "strict_center_only",
        "reason": reason,
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
    if label in {"weak_shape", "center_only"}:
        return SourceClass.WEAK_SHAPE
    if label in {"strict_center_only", "center_only_external"}:
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
    # Gaia entries without stellar evidence are absent from label decisions.
    gaia_rows=[row for row in gaia_rows if bool(row.get('astrometric_evidence',False))
        and np.isfinite(finite_float(row.get('phot_g_mean_mag')))
        and finite_float(row.get('phot_g_mean_mag'))<config.gaia_bright_mag_threshold]
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
    *, sources: list[dict[str, object]], gaia_rows: list[dict[str, object]],
    component_labels: np.ndarray, quality_mask: np.ndarray | None,
    config: BrightLabelConfig, protected_component_ids=(),
) -> tuple[list[dict[str, object]], dict[int, dict[str, object]], list[dict[str, object]], np.ndarray]:
    """HSC isolated-source policy followed by authoritative Gaia overrides.

    Reliable means parallax or proper-motion SNR >=3. Gaia entries without
    this evidence do not participate and never veto existing HSC labels.
    High-blendedness candidates have no unrestricted geometric fallback.
    The nested clean/weak exception is evaluated on the final dense targets.
    """
    areas=component_area_map(component_labels)
    centroids=component_centroid_map(component_labels)
    by_component=defaultdict(list); gaia_by_component=defaultdict(list)
    original_count=len(sources); ambiguous=set(); ignored=set(); clusters_all=[]
    meta={}; cluster_rows=[]; added=[]; seen_gaia=set()
    for i,source in enumerate(sources):
        comp=component_at(component_labels,float(source['x']),float(source['y']),0)
        source.update(component_id=comp,component_area=areas.get(comp,0))
        hit,reason=center_in_quality_mask(source,quality_mask)
        source.update(center_in_bad_mask=hit,center_bad_mask=reason)
        by_component[comp].append(i)
        if source_blendedness(source)>config.blendedness_ignore_threshold:ambiguous.add(comp)
    gaia=[row for row in gaia_rows if np.isfinite(finite_float(row.get('phot_g_mean_mag')))
          and finite_float(row.get('phot_g_mean_mag'))<config.gaia_bright_mag_threshold
          and bool(row.get('astrometric_evidence',False))]
    for row in gaia:
        comp=component_at(component_labels,float(row['x']),float(row['y']),0)
        gaia_by_component[comp].append(row)
    # First preserve usable isolated galaxies/shapes. Gaia overrides run after
    # this decision, so a coincident stellar identification has priority.
    for comp,indices in sorted(by_component.items()):
        pending=[]
        for i in indices:
            if float(sources[i]['area'])>=config.drop_area_max:
                sources[i].update(final_label='ignore',reason='dropped_large_area')
            else:pending.append(i)
        groups=([pending] if comp>0 and config.large_component_fast_center_only_source_min>0
                and len(pending)>=config.large_component_fast_center_only_source_min else
                cluster_sources(pending,sources,iou_threshold=config.cluster_iou_threshold,
                    max_center_distance=config.cluster_max_center_distance,max_area=config.cluster_max_area))
        meta[comp]=dict(component_id=comp,component_area=areas.get(comp,0),hsc_source_count=len(indices),cluster_count=len(groups))
        for group in groups:
            cluster_id=len(clusters_all)+1;clusters_all.append(group)
            for i in group:
                sources[i].update(cluster_id=cluster_id,cluster_size=len(group),
                    final_label='ignore',reason='bright_cluster_no_reliable_gaia')
            if len(group)==1 and not source_intersects_any(group[0],pending,sources):
                source=sources[group[0]]
                usable=source_shape_usable(source,max_area=config.shape_max_area,max_axis_ratio=config.shape_axis_ratio_max)
                size=max(float(areas.get(comp,0)),float(source['area']))
                high_blend=source_blendedness(source)>config.blendedness_ignore_threshold
                if high_blend:
                    source.update(final_label='ignore',reason='high_blendedness_requires_gaia')
                elif usable:
                    bridged=comp>0 and len(groups)>1
                    if size<config.isolated_area_max and not bridged:
                        source.update(final_label='clean',reason='single_cluster_small_bright_component_clean' if comp>0 else 'isolated_small_bright_source_clean')
                    elif source.get('class','unknown')!='star':
                        source.update(final_label='weak_shape',reason='single_cluster_large_bright_component_weak_shape' if comp>0 else 'isolated_bright_galaxy_weak_shape')
                    else:
                        source.update(final_label='center_only_external',reason='isolated_bright_star_strict_center_only')
            cluster_rows.append(dict(cluster_id=cluster_id,component_id=comp,component_area=areas.get(comp,0),
                cluster_size=len(group),source_ids=' '.join(str(sources[i]['source_id']) for i in group)))
    reliable=defaultdict(list)
    for comp,rows in gaia_by_component.items():
        if comp<=0:continue
        for row in rows:
            reliable[comp].append((row,'component_membership',0.))
    # Gaia centers outside the bright mask retain the existing 6px/10px
    # source/cluster matching, across all components; each Gaia matches once.
    outside=gaia_by_component.get(0,[])
    matches=assign_gaia_rows_to_clusters(clusters_all,sources,outside,
        source_match_pixels=config.cluster_source_match_pixels,
        centroid_match_pixels=config.cluster_centroid_match_pixels)
    isolated_matches=[]
    for group_index,rows in matches.items():
        group=clusters_all[group_index];comp=int(sources[group[0]]['component_id'])
        for row,mode,distance in rows:
            if comp>0:
                reliable[comp].append((row,mode,distance))
            else:isolated_matches.append((group,row,mode,distance))
    def insert(row,comp,mode,distance):
        key=gaia_row_key(row)
        if key in seen_gaia:return
        seen_gaia.add(key)
        added.append(synthetic_gaia_source(row,comp=comp,component_area=areas.get(comp,0),
            cluster_id=0,cluster_size=len(by_component.get(comp,[])),source_id_offset=1,
            pixel_scale_arcsec=config.pixel_scale_arcsec,match_mode=mode,match_pixels=distance,
            reason='hsc_reliable_gaia_strict_center_only'))
    for comp in sorted(reliable):
        meta.setdefault(comp,dict(component_id=comp,component_area=areas.get(comp,0)))
        meta[comp].update(reliable_gaia_count=len(reliable[comp]))
        for i in by_component.get(comp,[]):
            sources[i].update(final_label='restricted_bright_region',reason='hsc_gaia_replaced_fragment')
        for row,mode,distance in reliable[comp]:insert(row,comp,mode,distance)
    for group,row,mode,distance in isolated_matches:
        for i in group:
            sources[i].update(final_label='restricted_bright_region',reason='hsc_outside_gaia_replaced_fragment')
        insert(row,0,mode,distance)
    occupied=set(protected_component_ids)
    occupied.update(int(row.get('component_id',0)) for row in sources+added
        if row.get('final_label') in ('clean','weak_shape','center_only_external','strict_center_only','restricted_bright_region'))
    if config.add_empty_large_bright_component_centers:
        for comp,area in sorted(areas.items()):
            if comp<=0 or comp in occupied or comp in ignored:continue
            # No guessed center for ambiguous blends.
            if comp in ambiguous:
                ignored.add(comp);continue
            if by_component.get(comp) and all(sources[i].get('class','unknown')!='star' for i in by_component[comp]):
                ignored.add(comp);continue
            if component_touches_image_boundary(component_labels,comp) or area<=config.empty_large_bright_component_area_min:
                ignored.add(comp);continue
            x,y=centroids[comp]
            added.append(synthetic_component_center_source(comp=comp,component_area=int(area),x=x,y=y,
                reason='hsc_unmatched_large_component_geometric_center'))
            meta.setdefault(comp,{})['added_empty_large_center']=True
    sources.extend(added)
    return sources,meta,cluster_rows,np.asarray(sorted(ignored),np.int32)


def restore_nested_bright_centers(labels, component_labels, *,
                                 config=BrightLabelConfig(), max_major_pixels=100.):
    """Protect real bright peaks nested in nonstellar HSC clean/weak targets.

    This exception restores confidence points only. It never changes dense
    pixels or parent shapes, and never uses a Gaia PSF as an enclosing parent.
    A complete detected component must lie inside one adopted HSC clean/weak
    ellipse and the final clean/weak raster, with no existing center in/near it.
    """
    from scipy import ndimage
    from scipy.spatial import cKDTree
    from .labels import DenseLabel
    radius=float(config.cluster_source_match_pixels)
    policy=dict(definition_id='hsc_nested_clean_weak_bright_centers_v1',
        parent='nonstellar HSC clean or weak shape; unknown follows galaxy; exclude Gaia PSF rows',
        containment='entire detected bright component inside one adopted parent ellipse and final clean/weak dense region',
        min_center_separation_pixels=radius,label='strict center only; no child shape or PSF',
        preserves='parent Kron and dense raster unchanged',enabled=config.add_nested_bright_component_centers,
        centers_added=0,centers_total=len(labels.table.meta.get('hsc_nested_bright_sources',[])))
    previous=list(labels.table.meta.get('hsc_nested_bright_sources',[]))
    labels.table.meta['hsc_nested_bright_policy']=policy
    labels.table.meta['hsc_nested_bright_sources']=previous
    if not config.add_nested_bright_component_centers:return labels
    components=np.asarray(component_labels)
    if components.shape!=labels.dense.shape:raise ValueError('Bright-component/dense shape mismatch')
    n=len(labels.label_classes)
    ext=(np.ma.asarray(labels.table['base_ClassificationExtendedness_value'],dtype=float).filled(np.nan)
         if 'base_ClassificationExtendedness_value' in labels.table.colnames else np.full(n,np.nan))
    model=(np.ma.asarray(labels.table['gaia_psf_model'],dtype=bool).filled(False)
           if 'gaia_psf_model' in labels.table.colnames else np.zeros(n,bool))
    valid=np.isfinite(labels.geom_x+labels.geom_y+labels.geom_major+labels.geom_minor+labels.geom_theta)
    valid &= (labels.geom_major>0)&(labels.geom_minor>0)
    parents=np.flatnonzero(np.isin(labels.label_classes,[SourceClass.CLEAN,SourceClass.WEAK_SHAPE])&valid&~model&(~np.isfinite(ext)|(ext>.5)))
    if not len(parents):return labels
    active=valid&np.isin(labels.label_classes,[SourceClass.CLEAN,SourceClass.WEAK_SHAPE,SourceClass.STRICT_CENTER_ONLY])
    centers=np.r_[np.c_[labels.geom_x[active],labels.geom_y[active]],np.c_[labels.strict_x,labels.strict_y]]
    centers=centers[np.isfinite(centers).all(axis=1)]
    tree=cKDTree(centers)
    occupied={component_at(components,float(x),float(y),0) for x,y in centers}
    occupied.discard(0)
    ids=set(map(int,labels.strict_ids))|set(map(int,labels.source_ids))
    boxes=ndimage.find_objects(components);done=set(occupied);added=[]
    height,width=components.shape
    for i in parents[np.argsort(labels.geom_major[parents]*labels.geom_minor[parents],kind='stable')]:
        a,b=float(labels.geom_major[i]),float(labels.geom_minor[i]);theta=float(labels.geom_theta[i])
        if max_major_pixels is not None:
            scale=min(1.,float(max_major_pixels)/max(a,b));a*=scale;b*=scale
        x,y=float(labels.geom_x[i]),float(labels.geom_y[i])
        ys,xs=ellipse_bbox(x,y,a,b,theta,components.shape)
        for cid in map(int,np.unique(components[ys,xs])):
            if cid<=0 or cid in done:continue
            sl=boxes[cid-1]
            if sl is None:continue
            if sl[0].start<ys.start or sl[0].stop>ys.stop or sl[1].start<xs.start or sl[1].stop>xs.stop:continue
            yy,xx=np.nonzero(components[sl]==cid);yy+=sl[0].start;xx+=sl[1].start
            if not np.all(np.isin(labels.dense[yy,xx],[DenseLabel.CLEAN,DenseLabel.WEAK_SHAPE])):
                done.add(cid);continue
            parent=dict(x=x,y=y,major=a,minor=b,theta_deg=math.degrees(theta))
            if not np.all(ellipse_contains_dict(parent,xx.astype(float),yy.astype(float))):continue
            cx,cy=float(xx.mean()),float(yy.mean());ix,iy=int(math.floor(cx+.5)),int(math.floor(cy+.5))
            if not (0<=ix<width and 0<=iy<height) or labels.dense[iy,ix] not in (DenseLabel.CLEAN,DenseLabel.WEAK_SHAPE):
                done.add(cid);continue
            distance=float(tree.query([cx,cy])[0])
            if distance<=radius or any(math.hypot(cx-r['x'],cy-r['y'])<=radius for r in added):
                done.add(cid);continue
            synthetic=synthetic_component_center_source(comp=cid,component_area=len(xx),x=cx,y=cy,
                reason='hsc_nested_bright_component_geometric_center')
            sid=int(synthetic['source_id'])
            if sid in ids:
                done.add(cid);continue
            added.append(dict(id=sid,x=cx,y=cy,component_id=cid,component_area_pixels=len(xx),
                parent_id=int(labels.source_ids[i]),parent_class='clean' if labels.label_classes[i]==SourceClass.CLEAN else 'weak shape',nearest_existing_center_pixels=distance,
                reason=synthetic['reason'],label='strict center only'))
            ids.add(sid);done.add(cid)
    if added:
        old_count=len(labels.strict_x)
        labels.strict_x=np.r_[labels.strict_x,np.array([r['x'] for r in added],np.float32)]
        labels.strict_y=np.r_[labels.strict_y,np.array([r['y'] for r in added],np.float32)]
        labels.strict_ids=np.r_[labels.strict_ids,np.array([r['id'] for r in added],np.int64)]
        prior=np.zeros(old_count,bool) if labels.strict_is_gaia is None else labels.strict_is_gaia
        labels.strict_is_gaia=np.r_[prior,np.zeros(len(added),bool)]
    labels.table.meta['hsc_nested_bright_sources']=previous+added
    policy['centers_added']=len(added)
    policy['centers_total']=len(previous)+len(added)
    print(f'[hsc-nested-bright] restored {len(added)} strict centers inside HSC clean/weak shapes',flush=True)
    return labels


def apply_bright_rows_to_labels(
    sources: list[dict[str, object]],
    labels: SourceLabels,
    *,
    n_table_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    restricted = np.zeros(n_table_rows, dtype=bool)
    ordinary_ignore_sources = np.zeros(n_table_rows, dtype=bool)
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
            if source_class == SourceClass.ORDINARY_IGNORE and bool(source.get("paint_ordinary_ignore", False)):
                ordinary_ignore_sources[table_index] = True
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
        ordinary_ignore_sources,
    )


def unsupervised_seeded_component_centers(
    table: Table,
    labels: SourceLabels,
    component_labels: np.ndarray | None,
    *,
    seed_component_ids: np.ndarray,
    catalog_component_ids: np.ndarray | None = None,
    existing_strict_component_ids: np.ndarray | None = None,
    min_area: float = 64.0,
    component_search_radius: int = 5,
    refit_config: RefitConfig = RefitConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return geometric centers for seeded bright components with no positives.

    ``seed_component_ids`` should come from the pre-AP2 bright candidates.  This
    lets AP2-rejected bright detections still mark their image component as a
    supervised bright object instead of silently becoming ordinary background.
    """

    if component_labels is None or not np.asarray(component_labels).size:
        empty_f = np.asarray([], dtype=np.float64)
        empty_i = np.asarray([], dtype=np.int64)
        empty_c = np.asarray([], dtype=np.int32)
        return empty_f, empty_f, empty_i, empty_c
    component_labels = np.asarray(component_labels, dtype=np.int32)
    if int(np.max(component_labels)) <= 0:
        empty_f = np.asarray([], dtype=np.float64)
        empty_i = np.asarray([], dtype=np.int64)
        empty_c = np.asarray([], dtype=np.int32)
        return empty_f, empty_f, empty_i, empty_c

    component_areas = component_area_map(component_labels)
    component_centroids = component_centroid_map(component_labels)
    seeded = {int(comp) for comp in np.asarray(seed_component_ids, dtype=np.int32).ravel() if int(comp) > 0}
    if not seeded:
        empty_f = np.asarray([], dtype=np.float64)
        empty_i = np.asarray([], dtype=np.int64)
        empty_c = np.asarray([], dtype=np.int32)
        return empty_f, empty_f, empty_i, empty_c

    supervised: set[int] = set()
    geom = compute_kron_ellipse(table, refit_config)
    positive = (
        labels.mask(SourceClass.CLEAN)
        | labels.mask(SourceClass.WEAK_SHAPE)
        | labels.mask(SourceClass.STRICT_CENTER_ONLY)
    )
    for idx in np.flatnonzero(positive & np.isfinite(geom.x) & np.isfinite(geom.y)):
        comp = component_at(
            component_labels,
            float(geom.x[idx]),
            float(geom.y[idx]),
            int(component_search_radius),
        )
        if comp > 0:
            supervised.add(int(comp))
    if existing_strict_component_ids is not None:
        supervised.update(int(comp) for comp in np.asarray(existing_strict_component_ids, dtype=np.int32).ravel() if int(comp) > 0)
    catalog_components = (
        {int(comp) for comp in np.asarray(catalog_component_ids, dtype=np.int32).ravel() if int(comp) > 0}
        if catalog_component_ids is not None
        else set()
    )

    xs: list[float] = []
    ys: list[float] = []
    ids: list[int] = []
    comps: list[int] = []
    for comp in sorted(seeded - supervised - catalog_components):
        area = float(component_areas.get(comp, 0))
        if area < float(min_area) or comp not in component_centroids or component_touches_image_boundary(component_labels, comp):
            continue
        x, y = component_centroids[comp]
        xs.append(float(x))
        ys.append(float(y))
        ids.append(-(900000000000000000 + int(comp)))
        comps.append(int(comp))
    return (
        np.asarray(xs, dtype=np.float64),
        np.asarray(ys, dtype=np.float64),
        np.asarray(ids, dtype=np.int64),
        np.asarray(comps, dtype=np.int32),
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
    """HSC bright policy with zero-based, epoch-propagated DR3 positions."""

    if not np.isfinite(config.pixel_scale_arcsec) or config.pixel_scale_arcsec <= 0:
        raise ValueError("pixel_scale_arcsec must be finite and positive")
    # Legacy pixel options denote distances on the HSC reference grid.
    config = replace(
        config,
        cluster_source_match_pixels=matching_radius_pixels(
            config.pixel_scale_arcsec, arcsec=config.gaia_source_match_arcsec,
            hsc_pixels=config.cluster_source_match_pixels),
        cluster_centroid_match_pixels=matching_radius_pixels(
            config.pixel_scale_arcsec, arcsec=config.gaia_centroid_match_arcsec,
            hsc_pixels=config.cluster_centroid_match_pixels),
    )
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
        gaia_rows = project_gaia_rows(gaia_table, image_shape=image_shape, image_header=image_header,
                                      pixel_origin=0,reference_epoch=2016.)
    if component_labels is not None and np.asarray(component_labels).size:
        geom=compute_kron_ellipse(table,refit_config)
        positive=np.isin(labels.source_class,[SourceClass.CLEAN,SourceClass.WEAK_SHAPE,SourceClass.STRICT_CENTER_ONLY]) & ~candidate
        protected={component_at(component_labels,geom.x[i],geom.y[i],0) for i in np.flatnonzero(positive & np.isfinite(geom.x+geom.y))}
        sources, component_meta, cluster_rows, ordinary_ignore_components = classify_component_bright(
            sources=sources,
            gaia_rows=gaia_rows,
            component_labels=np.asarray(component_labels, dtype=np.int32),
            quality_mask=quality_mask,
            config=config,
            protected_component_ids=protected,
        )
    else:
        sources, component_meta, cluster_rows, _mask, _labels = classify_no_upper(
            sources=sources,
            gaia_rows=gaia_rows,
            image_shape=image_shape or (1, 1),
            quality_mask=quality_mask,
            config=config,
        )
        ordinary_ignore_components = np.asarray([], dtype=np.int32)
    strict_x, strict_y, strict_id, strict_reason, strict_component_id, restricted, fallback_components, ordinary_ignore_sources = apply_bright_rows_to_labels(
        sources,
        labels,
        n_table_rows=len(table),
    )
    if len(ordinary_ignore_components):
        # Entire rejected components must stay ignore even when an ordinary
        # catalog source would otherwise repaint a positive dense shape there.
        geom=compute_kron_ellipse(table,refit_config)
        comps=np.array([component_at(component_labels,x,y,0) if np.isfinite(x+y) else 0
                        for x,y in zip(geom.x,geom.y)])
        veto=np.isin(comps,ordinary_ignore_components)&~np.isin(labels.source_class,[SourceClass.DROPPED,SourceClass.STRICT_IGNORE])
        labels.assign(veto,SourceClass.ORDINARY_IGNORE,'hsc_rejected_bright_component')
    return BrightLabelResult(
        labels=labels,
        strict_center_x=strict_x,
        strict_center_y=strict_y,
        strict_center_source_id=strict_id,
        strict_center_reason=strict_reason,
        strict_center_component_id=strict_component_id,
        restricted_source_mask=restricted,
        restricted_fallback_component_ids=fallback_components,
        ordinary_ignore_component_ids=ordinary_ignore_components,
        ordinary_ignore_source_mask=ordinary_ignore_sources,
        source_rows=sources,
        component_meta=component_meta,
        cluster_rows=cluster_rows,
    )
