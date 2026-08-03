#!/usr/bin/env python3
"""In-memory external bright-source relabeling for preprocessing.

This module mirrors the main branch of ``build_external_bright_labels_v2.py``
without writing diagnostic PNG/REG/CSV files.  It is intentionally conservative:
the goal is to keep production preprocessing labels aligned with the bright
source diagnostics that were tuned interactively.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from scipy import ndimage

from data_filtering import build_external_bright_labels as bright_base
from data_filtering import build_external_bright_labels_v2 as bright_v2
from data_filtering import diagnose_bright_filter_stages as stage_diag
from data_filtering import external_bright_labels_core as bright_core
from data_filtering.sam_input_scaling import (
    anscombe_single,
    build_bright_mask,
    current_sam_zscore,
    log_single,
    lupton_single,
    standardize_by_self,
)


def default_log_a_for_band(band: str, fallback: float = 1000.0) -> float:
    name = str(band).upper().replace("HSC-", "")
    if name == "NB1010":
        return 100.0
    if name == "NB0387":
        return 3000.0
    return float(fallback)


def _get(args: argparse.Namespace, *names: str, default: object = None) -> object:
    for name in names:
        if hasattr(args, name):
            return getattr(args, name)
    return default


def _stage_args(args: argparse.Namespace, *, band: str, mode: str) -> argparse.Namespace:
    no_upper = mode in {"zscore-no-upper", "zscore-unbounded"}
    return argparse.Namespace(
        mag_column=str(_get(args, "pu_mag_column", "mag_column", default="ext_photometryKron_KronFlux_instFlux")),
        mag_threshold=float(_get(args, "pu_ap2_kron_bright_mag_threshold", "ap2_kron_bright_mag_threshold", default=22.0)),
        zeropoint=float(_get(args, "pu_input_zeropoint", "zeropoint", default=27.0)),
        ellipse_sigma=float(_get(args, "ellipse_sigma", default=1.0)),
        min_axis=float(_get(args, "min_ellipse_axis", "min_axis", default=1.5)),
        a_area_max=float(_get(args, "pu_a_area_max", "a_area_max", default=10000.0)),
        a_faint_area_max=float(_get(args, "pu_a_faint_area_max", "a_faint_area_max", default=900.0)),
        a_faint_mag_min=float(_get(args, "pu_a_faint_mag_min", "a_faint_mag_min", default=28.0)),
        axis_ratio_max=float(_get(args, "pu_b_axis_ratio_max", "axis_ratio_max", default=5.0)),
        close_center_arcsec=float(_get(args, "pu_b_close_center_arcsec", "close_center_arcsec", default=0.5)),
        pixel_scale_arcsec=float(_get(args, "pixel_scale_arcsec", default=0.168)),
        ap2_flux_column=str(_get(args, "pu_ap2_flux_column", "ap2_flux_column", default="base_CircularApertureFlux_6_0_instFlux")),
        kron_flux_column=str(_get(args, "pu_ap2_kron_flux_column", "ap2_kron_flux_column", default="ext_photometryKron_KronFlux_instFlux")),
        include_ap2_filter=not no_upper,
        refined_bright_ap2=not no_upper,
        ap2_kron_abs_max=float(_get(args, "pu_ap2_kron_abs_max", "ap2_kron_abs_max", default=1.0)),
        ap2_kron_mid_max=float(_get(args, "pu_ap2_kron_bright_abs_max", "ap2_kron_bright_abs_max", default=2.0)),
        large_bright_region_area_min=float(
            _get(args, "pu_ap2_kron_large_bright_region_area_min", "ap2_kron_large_bright_region_area_min", default=1000.0)
        ),
        source_filter=str(_get(args, "source_filter", default="nchild0")),
        band=band,
    )


def _label_sets(*, clean: Table, center_only: Table, ignore: Table, strict_center_only: Table) -> dict[str, set[int]]:
    def ids(table: Table) -> set[int]:
        if len(table) == 0 or "id" not in table.colnames:
            return set()
        return {int(value) for value in np.asarray(table["id"], dtype=np.int64)}

    return {
        "clean": ids(clean),
        "center_only": ids(center_only),
        "strict_center_only": ids(strict_center_only),
        "ignore": ids(ignore),
        "strict_ignore": set(),
        "rejected": set(),
    }


def _existing_label(source_id: int, labels: dict[str, set[int]]) -> str:
    return bright_base.classify_existing_label(int(source_id), labels)


def _sources_from_stage(stage: dict[str, np.ndarray], meas: Table, labels: dict[str, set[int]]) -> list[dict[str, object]]:
    id_to_index = bright_v2.meas_index_by_source_id(meas)
    ext = np.full(len(meas), np.nan, dtype=np.float64)
    if "base_ClassificationExtendedness_value" in meas.colnames:
        ext = np.asarray(meas["base_ClassificationExtendedness_value"], dtype=np.float64)
    sources: list[dict[str, object]] = []
    status = np.asarray(stage["status"], dtype=object)
    ids = np.asarray(meas["id"], dtype=np.int64) if "id" in meas.colnames else np.arange(len(meas), dtype=np.int64)
    for idx in np.flatnonzero(np.asarray([str(value).startswith("remaining_") for value in status], dtype=bool)):
        sid = int(ids[idx])
        x = float(stage["x"][idx])
        y = float(stage["y"][idx])
        major = float(stage["a"][idx])
        minor = float(stage["b"][idx])
        theta = float(stage["theta"][idx])
        area = float(stage["area"][idx])
        mag = float(stage["mag"][idx])
        if not all(math.isfinite(v) for v in (x, y, major, minor, theta, area, mag)):
            continue
        if major <= 0.0 or minor <= 0.0:
            continue
        row_index = id_to_index.get(sid, int(idx))
        ext_value = float(ext[row_index]) if 0 <= row_index < len(ext) and math.isfinite(float(ext[row_index])) else float("nan")
        sources.append(
            {
                "source_id": sid,
                "row_index": row_index,
                "x": x,
                "y": y,
                "major": major,
                "minor": minor,
                "theta_deg": math.degrees(theta),
                "area": area,
                "axis_ratio": float(stage["axis_ratio"][idx]),
                "mag": mag,
                "class": bright_base.source_class_from_extendedness(ext_value),
                "classification_extendedness": ext_value,
                "existing_label": _existing_label(sid, labels),
                "measurement_surface": "integrated_stage_A_B_refined",
                "stage_status": str(status[idx]),
                "final_label": "",
                "reason": "",
                "component_id": 0,
                "cluster_id": 0,
                "cluster_size": 0,
                "cluster_is_stellar_mask": False,
                "gaia_source_id": "",
                "gaia_g_mag": "",
                "gaia_match_arcsec": "",
                "gaia_match_pixels": "",
                "gaia_match_mode": "",
                "output_x": x,
                "output_y": y,
            }
        )
    return sources


def _log_lupton_bright_components(
    image: np.ndarray,
    *,
    band: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    current_z, stats = current_sam_zscore(image)
    del current_z
    log_a = float(_get(args, "integrated_bright_log_a", default=float("nan")))
    if not math.isfinite(log_a):
        log_a = default_log_a_for_band(band, fallback=1000.0)
    log_map, _log_stats = log_single(
        image,
        minimum=float(stats["raw_min"]),
        high_pct=float(_get(args, "pu_bright_log_high_percentile", "bright_log_high_percentile", default=99.5)),
        a=log_a,
    )
    lupton_map, _lupton_stats = lupton_single(
        image,
        minimum=float(stats.get("zscore_median", stats["median"])),
        stretch=float(_get(args, "pu_bright_lupton_stretch", "bright_lupton_stretch", default=0.5)),
        q=float(_get(args, "pu_bright_lupton_q", "bright_lupton_q", default=20.0)),
    )
    _log_z, log_zclip, _log_zstats = standardize_by_self(log_map)
    _lupton_z, lupton_zclip, _lupton_zstats = standardize_by_self(lupton_map)
    bright = (log_zclip >= float(_get(args, "pu_bright_z_threshold", "bright_z_threshold", default=3.0))) & (
        lupton_zclip >= float(_get(args, "pu_bright_z_threshold", "bright_z_threshold", default=3.0))
    )
    dilation = int(_get(args, "pu_bright_mask_dilate", "bright_mask_dilate", default=2))
    if dilation > 0 and np.any(bright):
        bright = ndimage.binary_dilation(bright, iterations=dilation)
    labels, _num = ndimage.label(bright)
    return bright.astype(np.uint8), labels.astype(np.int32)


def build_integrated_bright_components(
    image: np.ndarray,
    *,
    band: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    mode = str(_get(args, "pu_bright_mask_mode", "bright_mask_mode", default="log-lupton")).strip().lower().replace("_", "-")
    if mode == "log-lupton":
        return _log_lupton_bright_components(image, band=band, args=args)
    bright = build_bright_mask(
        image,
        mode=mode,
        threshold=float(_get(args, "pu_bright_z_threshold", "bright_z_threshold", default=3.0)),
        dilation=int(_get(args, "pu_bright_mask_dilate", "bright_mask_dilate", default=2)),
        log_a=default_log_a_for_band(band, fallback=1000.0),
        log_high_percentile=float(_get(args, "pu_bright_log_high_percentile", "bright_log_high_percentile", default=99.5)),
        lupton_stretch=float(_get(args, "pu_bright_lupton_stretch", "bright_lupton_stretch", default=0.5)),
        lupton_q=float(_get(args, "pu_bright_lupton_q", "bright_lupton_q", default=20.0)),
        anscombe_scale=float(_get(args, "pu_bright_anscombe_scale", "bright_anscombe_scale", default=1000.0)),
    )
    labels, _num = ndimage.label(np.asarray(bright, dtype=bool))
    return np.asarray(bright, dtype=np.uint8), labels.astype(np.int32)


def _local_rows_to_physical(rows: list[dict[str, object]], origin_xy: tuple[int, int]) -> list[dict[str, object]]:
    ox, oy = int(origin_xy[0]), int(origin_xy[1])
    out: list[dict[str, object]] = []
    for row in rows:
        copied = dict(row)
        x = bright_base.finite_float(copied.get("output_x", copied.get("x")))
        y = bright_base.finite_float(copied.get("output_y", copied.get("y")))
        if math.isfinite(x):
            copied["output_x"] = x + ox
        if math.isfinite(y):
            copied["output_y"] = y + oy
        out.append(copied)
    return out


def _keep_integrated_override_row(row: dict[str, object], *, mode: str) -> bool:
    """Return whether an integrated bright row should override the PU label.

    For image-threshold bright modes, sources whose centers are outside every
    bright component have already been handled by the ordinary PU/AP2 flow.
    Emitting an ``ignore`` override for those rows would turn AP2-valid ordinary
    bright sources into ignore simply because they are not in the bright-region
    mask.  The zscore-no-upper path has no image bright components, so it keeps
    its source-cluster labels.
    """

    normalized = str(mode).strip().lower().replace("_", "-")
    if normalized in {"zscore-no-upper", "zscore-unbounded"}:
        return True
    try:
        component_id = int(round(float(row.get("component_id", 0))))
    except Exception:
        component_id = 0
    if component_id <= 0:
        return False
    return True


def build_integrated_bright_labels(
    *,
    pu_all: Table,
    clean: Table,
    center_only: Table,
    ignore: Table,
    strict_center_only: Table,
    image: np.ndarray,
    image_header: fits.Header,
    mask: np.ndarray,
    mask_header: fits.Header,
    image_origin: tuple[int, int],
    args: argparse.Namespace,
    band: str,
    patch: str,
) -> tuple[list[dict[str, object]], np.ndarray, dict[str, object]]:
    del patch
    mode = str(_get(args, "pu_bright_mask_mode", "bright_mask_mode", default="log-lupton")).strip().lower().replace("_", "-")
    labels = _label_sets(clean=clean, center_only=center_only, ignore=ignore, strict_center_only=strict_center_only)
    bright_mask: np.ndarray | None = None
    component_labels: np.ndarray | None = None
    if mode not in {"zscore-no-upper", "zscore-unbounded"}:
        bright_mask, component_labels = build_integrated_bright_components(image, band=band, args=args)
    stage = stage_diag.classify_stages(
        pu_all,
        _stage_args(args, band=band, mode=mode),
        (-float(image_origin[0]), -float(image_origin[1])),
        bright_region_mask=np.asarray(bright_mask, dtype=bool) if bright_mask is not None else None,
    )
    sources = _sources_from_stage(stage, pu_all, labels)
    gaia_path = Path(_get(args, "integrated_bright_gaia_fits", "bright_gaia_fits", default="output/gaia_dr3_cosmos.fits")).expanduser()
    if gaia_path.exists():
        gaia_rows = bright_base.load_gaia_for_patch(gaia_path, WCS(image_header), image.shape)
    else:
        gaia_rows = []

    if mode in {"zscore-no-upper", "zscore-unbounded"}:
        assigned, component_meta, cluster_rows, bright_mask_u8, component_labels = bright_core.assign_no_upper_source_clusters(
            sources=sources,
            gaia_rows=gaia_rows,
            image_shape=image.shape,
            mask=mask,
            mask_header=mask_header,
            mask_names=list(_get(args, "integrated_bright_mask_names", default=("SAT", "BAD", "EDGE"))),
            config=bright_core.NoUpperBrightLabelConfig(
                cluster_iou_threshold=float(_get(args, "integrated_bright_cluster_iou_threshold", default=1.0 / 3.0)),
                cluster_max_center_distance=float(_get(args, "integrated_bright_cluster_max_center_distance", default=50.0)),
                cluster_max_area=float(_get(args, "integrated_bright_cluster_max_area", default=10000.0)),
                cluster_source_match_pixels=float(_get(args, "integrated_bright_cluster_source_match_pixels", default=6.0)),
                cluster_centroid_match_pixels=float(_get(args, "integrated_bright_cluster_centroid_match_pixels", default=10.0)),
                isolated_clean_area_max=float(_get(args, "integrated_bright_isolated_clean_area_max", default=1000.0)),
                pixel_scale_arcsec=float(_get(args, "pixel_scale_arcsec", default=0.168)),
            ),
        )
        bright_mask = np.asarray(bright_mask_u8, dtype=bool)
    else:
        assert bright_mask is not None and component_labels is not None
        assigned, component_meta, cluster_rows = bright_v2.assign_labels_v2(
            sources=sources,
            gaia_rows=gaia_rows,
            component_labels=component_labels,
            component_areas=bright_v2.component_area_map(component_labels),
            component_centroids=bright_v2.component_centroid_map(component_labels),
            mask=mask,
            mask_header=mask_header,
            mask_names=list(_get(args, "integrated_bright_mask_names", default=("SAT", "BAD", "EDGE"))),
            component_search_radius=int(_get(args, "integrated_bright_component_search_radius", default=5)),
            match_radius_arcsec=float(_get(args, "integrated_bright_match_radius_arcsec", default=1.0)),
            gaia_bright_mag_threshold=float(_get(args, "integrated_bright_gaia_bright_mag_threshold", default=18.0)),
            cluster_iou_threshold=float(_get(args, "integrated_bright_cluster_iou_threshold", default=1.0 / 3.0)),
            cluster_max_center_distance=float(_get(args, "integrated_bright_cluster_max_center_distance", default=50.0)),
            cluster_max_area=float(_get(args, "integrated_bright_cluster_max_area", default=10000.0)),
            cluster_source_match_pixels=float(_get(args, "integrated_bright_cluster_source_match_pixels", default=6.0)),
            cluster_centroid_match_pixels=float(_get(args, "integrated_bright_cluster_centroid_match_pixels", default=10.0)),
            shape_max_area=float(_get(args, "integrated_bright_shape_max_area", default=10000.0)),
            shape_axis_ratio_max=float(_get(args, "integrated_bright_shape_axis_ratio_max", default=5.0)),
            drop_area_max=float(_get(args, "integrated_bright_drop_area_max", default=10000.0)),
            use_bad_mask_first_step=bool(_get(args, "integrated_bright_use_bad_mask_first_step", default=False)),
            use_bright_gaia_component_override=bool(_get(args, "integrated_bright_use_gaia_component_override", default=True)),
            add_empty_large_bright_component_centers=bool(_get(args, "integrated_bright_add_empty_large_component_centers", default=True)),
            empty_large_bright_component_area_min=float(_get(args, "integrated_bright_empty_large_component_area_min", default=1000.0)),
            large_component_fast_center_only_source_min=int(_get(args, "integrated_bright_large_component_fast_source_min", default=0)),
        )
    override_rows = [row for row in assigned if _keep_integrated_override_row(row, mode=mode)]
    physical_rows = _local_rows_to_physical(override_rows, image_origin)
    stats = {
        "integrated_bright_sources": len(sources),
        "integrated_bright_assigned": len(physical_rows),
        "integrated_bright_unassigned_outside_component": len(assigned) - len(override_rows),
        "integrated_bright_components": int(np.max(component_labels)) if component_labels is not None and component_labels.size else 0,
        "integrated_bright_clusters": len(cluster_rows),
        "integrated_bright_gaia": len(gaia_rows),
        "integrated_bright_component_meta": len(component_meta),
    }
    return physical_rows, np.asarray(bright_mask, dtype=bool), stats
