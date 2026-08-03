#!/usr/bin/env python3
"""Reusable PU filtering configuration helpers.

These helpers normalize CLI/runtime options and construct the catalog-level PU
classification arguments.  They intentionally do not crop FITS files or write
training products.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from typing import Dict, Optional, Sequence

import numpy as np


def build_pu_runtime_config(args: argparse.Namespace) -> SimpleNamespace:
    """Normalize legacy, direct-Zarr, and diagnostic preprocessing options."""

    if bool(getattr(args, "_shared_pu_runtime_config", False)):
        return args  # type: ignore[return-value]

    values = dict(vars(args))
    aliases = {
        "shape_source": ("target_shape_source", "kron"),
        "target_shape_source": ("target_shape_source", "kron"),
        "max_area_3sigma": (None, 400.0),
        "relaxed_area_3sigma": (None, 900.0),
        "area_filter_policy": (None, "max_area"),
        "drop_children": (None, False),
        "label_mode": (None, "pu"),
        "ellipse_sigma": ("ellipse_sigma", 1.0),
        "min_ellipse_axis": (None, 1.5),
        "pixel_scale_arcsec": ("pixel_scale_arcsec", 0.168),
        "no_clean_nonfinite": (None, False),
        "pu_a_flags": (None, ()),
        "pu_b_flags": ("b_flags", ("base_SdssShape_flag", "base_SdssCentroid_flag")),
        "pu_a_mode": (None, "any"),
        "pu_b_mode": (None, "any"),
        "pu_strict_flags": (None, ()),
        "pu_mag_column": ("mag_column", "ext_photometryKron_KronFlux_instFlux"),
        "pu_input_zeropoint": ("zeropoint", 27.0),
        "pu_require_kron_refit_match": ("require_kron_refit_match", True),
        "pu_kron_refit_csv": ("kron_refit_csv", None),
        "pu_kron_refit_radius_column": ("kron_refit_radius_column", "proxy_nan0_flux_aperture_radius"),
        "pu_kron_refit_good_column": ("kron_refit_good_column", "proxy_nan0_good"),
        "pu_a_area_max": ("a_area_max", 10000.0),
        "pu_a_faint_area_max": ("a_faint_area_max", 900.0),
        "pu_a_faint_mag_min": ("a_faint_mag_min", 28.0),
        "pu_center_only_fill_area_min": ("center_only_fill_area_min", 500.0),
        "pu_center_only_fill_ratio_max": ("center_only_fill_ratio_max", 0.3),
        "pu_b_mag_min": ("b_mag_min", 15.0),
        "pu_b_mag_max": ("b_mag_max", 35.0),
        "pu_use_band_limit_b_filter": ("use_band_limit_b_filter", False),
        "pu_band_limit_mags": ("band_limit_mags", None),
        "pu_band_limit_b_min_offset": (None, -5.0),
        "pu_band_limit_b_max_offset": (None, 0.0),
        "pu_ap2_kron_abs_max": ("ap2_kron_abs_max", 1.0),
        "pu_ap2_kron_bright_mag_threshold": ("ap2_kron_bright_mag_threshold", 22.0),
        "pu_ap2_kron_bright_abs_max": ("ap2_kron_bright_abs_max", 2.0),
        "pu_ap2_kron_bright_region_column": ("ap2_kron_bright_region_column", "pu_bright_region_center"),
        "pu_ap2_kron_bright_region_area_column": (
            "ap2_kron_bright_region_area_column",
            "pu_bright_region_component_area",
        ),
        "pu_ap2_kron_large_bright_region_area_min": ("ap2_kron_large_bright_region_area_min", 1000.0),
        "pu_ap2_flux_column": ("ap2_flux_column", "base_CircularApertureFlux_6_0_instFlux"),
        "pu_ap2_kron_flux_column": ("ap2_kron_flux_column", "ext_photometryKron_KronFlux_instFlux"),
        "pu_ap2_kron_small_bright_area_reject": ("ap2_kron_small_bright_area_reject", True),
        "pu_ap2_kron_small_bright_area_ratio_max": ("ap2_kron_small_bright_area_ratio_max", 1.0),
        "pu_ap2_kron_small_bright_area_abs_min": ("ap2_kron_small_bright_area_abs_min", 1.0),
        "pu_b_close_center_arcsec": ("close_center_arcsec", 0.5),
        "pu_overlap_iou_threshold": (None, 0.33),
        "pu_b_ellipse_area_max": (None, None),
        "pu_b_footprint_area_max": (None, None),
        "pu_b_axis_ratio_max": ("axis_ratio_max", 5.0),
        "pu_b_kron_radius_lt_sdss_major_ratio": (None, 0.5),
        "pu_drop_ellipse_area_min": ("drop_ellipse_area_min", 40000.0),
        "pu_ambiguous_area_max": (None, None),
        "pu_neighbor_radius": (None, 0.0),
        "pu_center_distance_factor": (None, 0.0),
        "pu_containment_threshold": ("containment_threshold", 0.80),
        "pu_mutual_overlap_threshold": (None, 0.0),
        "pu_overlap_sample_grid": (None, 16),
        "pu_ambiguous_mark": (None, "center_only"),
        "pu_keep_all_ab_clean": (None, True),
        "pu_enable_strict_bright_center_only": ("enable_strict_bright_center_only", True),
        "pu_strict_bright_center_only_mag_threshold": ("strict_bright_center_only_mag_threshold", None),
        "pu_strict_ignore_mag_threshold": (None, None),
        "pu_strict_bright_center_only_saturation_mags": ("strict_bright_center_only_saturation_mags", None),
        "pu_strict_ignore_saturation_mags": (None, None),
        "pu_strict_bright_center_only_radius_column": (
            "strict_bright_center_only_radius_column",
            "proxy_nan0_flux_aperture_radius",
        ),
        "pu_strict_bright_center_only_ellipse_sigma": ("strict_bright_center_only_ellipse_sigma", 1.0),
        "pu_remeasure_ap2_kron_outliers": ("remeasure_ap2_kron_outliers", True),
        "pu_remeasure_ap2_kron_threshold": (None, np.nan),
        "pu_remeasure_clean_abs_max": ("ap2_kron_abs_max", 1.0),
        "pu_remeasure_center_only_abs_max": ("remeasure_center_only_abs_max", 1.5),
        "pu_remeasure_small_footprint_fill_threshold": ("remeasure_small_footprint_fill_threshold", 0.2),
        "pu_remeasure_ignore_area_max": ("remeasure_ignore_area_max", 10000.0),
        "pu_remeasure_faint_mag_min": ("remeasure_faint_mag_min", 28.0),
        "pu_remeasure_faint_area_max": ("remeasure_faint_area_max", 900.0),
        "pu_remeasure_axis_ratio_max": ("remeasure_axis_ratio_max", 5.0),
        "pu_remeasure_containment_threshold": ("remeasure_containment_threshold", 0.80),
        "noncoadd_snr_use_source_mask": ("noncoadd_snr_use_source_mask", True),
        "noncoadd_snr_use_quality_mask": ("noncoadd_snr_use_quality_mask", True),
        "noncoadd_snr_mask_planes": (
            "noncoadd_snr_mask_planes",
            ("BRIGHT_OBJECT", "SAT", "BAD", "NO_DATA", "EDGE", "UNMASKEDNAN"),
        ),
        "pu_ignore_mask_planes": ("pu_ignore_mask_planes", ("SAT", "BAD", "EDGE")),
        "noncoadd_snr_exclude_self_source": ("noncoadd_snr_exclude_self_source", True),
    }
    for canonical, (alias, default) in aliases.items():
        if canonical in values:
            continue
        if alias is not None and alias in values:
            values[canonical] = values[alias]
        else:
            values[canonical] = default
    values["_shared_pu_runtime_config"] = True
    return SimpleNamespace(**values)


DEFAULT_PU_BAND_LIMIT_MAGS = {
    "HSC-G": 27.4,
    "HSC-R": 27.1,
    "HSC-I": 26.9,
    "HSC-Z": 26.3,
    "HSC-Y": 25.3,
}

DEFAULT_PU_STRICT_CENTER_ONLY_SATURATION_MAGS = {
    "HSC-G": 18.0,
    "HSC-R": 18.2,
    "HSC-I": 18.6,
    "HSC-Z": 17.7,
    "HSC-Y": 17.4,
    "HSC-NB0387": 14.8,
    "HSC-NB0816": 16.8,
    "HSC-NB0921": 16.9,
    "HSC-NB0924": 16.9,
    "HSC-NB1010": 14.8,
}


def normalize_band_name(name: str) -> str:
    text = str(name).strip().upper()
    if not text:
        return text
    if not text.startswith("HSC-"):
        text = f"HSC-{text}"
    return text


def parse_band_mags(values: Optional[Sequence[str]], defaults: Dict[str, float], *, label: str) -> Dict[str, float]:
    limits = dict(defaults)
    if not values:
        return limits
    for raw in values:
        for item in str(raw).replace(",", " ").split():
            if not item:
                continue
            if "=" not in item and ":" not in item:
                raise ValueError(f"{label} must be BAND=mag, got {item!r}")
            key, value = item.replace(":", "=", 1).split("=", 1)
            limits[normalize_band_name(key)] = float(value)
    return limits


def parse_band_limit_mags(values: Optional[Sequence[str]]) -> Dict[str, float]:
    return parse_band_mags(values, DEFAULT_PU_BAND_LIMIT_MAGS, label="band limit")


def parse_strict_center_only_saturation_mags(values: Optional[Sequence[str]]) -> Dict[str, float]:
    return parse_band_mags(
        values,
        DEFAULT_PU_STRICT_CENTER_ONLY_SATURATION_MAGS,
        label="strict center-only saturation magnitude",
    )


def strict_center_only_mag_threshold(args: argparse.Namespace, *, band: Optional[str]) -> float:
    override = getattr(args, "pu_strict_bright_center_only_mag_threshold", None)
    if override is None:
        override = getattr(args, "pu_strict_ignore_mag_threshold", None)
    if override is not None:
        return float(override)
    band_name = normalize_band_name(band or getattr(args, "catalog_band", ""))
    limits = parse_strict_center_only_saturation_mags(
        getattr(args, "pu_strict_bright_center_only_saturation_mags", None)
        or getattr(args, "pu_strict_ignore_saturation_mags", None)
    )
    if band_name not in limits:
        raise ValueError(f"No strict center-only saturation magnitude configured for {band_name!r}")
    return float(limits[band_name])


def pu_classify_args(args: argparse.Namespace, *, band: Optional[str] = None) -> argparse.Namespace:
    b_mag_min = float(args.pu_b_mag_min)
    b_mag_max = float(args.pu_b_mag_max)
    if bool(getattr(args, "pu_use_band_limit_b_filter", False)):
        band_name = normalize_band_name(band or getattr(args, "catalog_band", ""))
        limits = parse_band_limit_mags(getattr(args, "pu_band_limit_mags", None))
        if band_name not in limits:
            raise ValueError(f"No PU band limiting magnitude configured for {band_name!r}")
        limit = float(limits[band_name])
        b_mag_min = limit + float(args.pu_band_limit_b_min_offset)
        b_mag_max = limit + float(args.pu_band_limit_b_max_offset)
    return argparse.Namespace(
        source_filter=args.source_filter,
        a_flags=tuple(args.pu_a_flags),
        b_flags=tuple(args.pu_b_flags),
        a_mode=args.pu_a_mode,
        b_mode=args.pu_b_mode,
        strict_flags=args.pu_strict_flags,
        region_sigma=args.ellipse_sigma,
        min_axis=args.min_ellipse_axis,
        mag_column=args.pu_mag_column,
        input_zeropoint=args.pu_input_zeropoint,
        require_kron_refit_match=bool(args.pu_require_kron_refit_match),
        a_area_max=args.pu_a_area_max,
        a_faint_area_max=args.pu_a_faint_area_max,
        a_faint_mag_min=args.pu_a_faint_mag_min,
        center_only_fill_area_min=args.pu_center_only_fill_area_min,
        center_only_fill_ratio_max=args.pu_center_only_fill_ratio_max,
        b_mag_min=b_mag_min,
        b_mag_max=b_mag_max,
        ap2_kron_abs_max=args.pu_ap2_kron_abs_max,
        ap2_kron_bright_mag_threshold=args.pu_ap2_kron_bright_mag_threshold,
        ap2_kron_bright_abs_max=args.pu_ap2_kron_bright_abs_max,
        ap2_kron_bright_region_column=args.pu_ap2_kron_bright_region_column,
        ap2_kron_bright_region_area_column=args.pu_ap2_kron_bright_region_area_column,
        ap2_kron_large_bright_region_area_min=args.pu_ap2_kron_large_bright_region_area_min,
        ap2_flux_column=args.pu_ap2_flux_column,
        ap2_kron_flux_column=args.pu_ap2_kron_flux_column,
        ap2_kron_small_bright_area_reject=bool(args.pu_ap2_kron_small_bright_area_reject),
        ap2_kron_small_bright_area_ratio_max=args.pu_ap2_kron_small_bright_area_ratio_max,
        ap2_kron_small_bright_area_abs_min=args.pu_ap2_kron_small_bright_area_abs_min,
        pixel_scale_arcsec=args.pixel_scale_arcsec,
        b_close_center_arcsec=args.pu_b_close_center_arcsec,
        overlap_iou_threshold=args.pu_overlap_iou_threshold,
        b_ellipse_area_max=args.pu_b_ellipse_area_max,
        b_footprint_area_max=args.pu_b_footprint_area_max,
        b_axis_ratio_max=args.pu_b_axis_ratio_max,
        b_kron_radius_lt_sdss_major_ratio=args.pu_b_kron_radius_lt_sdss_major_ratio,
        drop_ellipse_area_min=args.pu_drop_ellipse_area_min,
        ambiguous_area_max=args.pu_ambiguous_area_max,
        neighbor_radius=args.pu_neighbor_radius,
        center_distance_factor=args.pu_center_distance_factor,
        containment_threshold=args.pu_containment_threshold,
        mutual_overlap_threshold=args.pu_mutual_overlap_threshold,
        overlap_sample_grid=args.pu_overlap_sample_grid,
        ambiguous_mark=args.pu_ambiguous_mark,
        keep_all_ab_clean=bool(getattr(args, "pu_keep_all_ab_clean", False)),
    )
