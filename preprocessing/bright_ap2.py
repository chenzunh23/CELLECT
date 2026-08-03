"""Bright-source AP2-Kron catalog filtering.

This module implements the bright-only AP2/Kron gate from the v3 workflow:
ordinary remeasurement is intentionally not used here.  The AP2 and Kron
magnitudes come directly from the meas/refit-attached table columns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.table import Table

from .image_processing import component_area_map
from .labels import SourceClass, SourceLabels
from .refit import RefitConfig, compute_kron_ellipse
from .utils.catalog import magnitude_from_flux
from .utils.geometry import component_at


@dataclass(frozen=True)
class BrightAp2Config:
    ap2_flux_column: str = "base_CircularApertureFlux_6_0_instFlux"
    kron_flux_column: str = "ext_photometryKron_KronFlux_instFlux"
    zeropoint: float = 27.0
    outside_bright_abs_max: float = 1.0
    small_bright_abs_max: float = 2.0
    large_bright_region_area_min: float = 1000.0
    component_search_radius: int = 5


@dataclass
class BrightAp2Result:
    candidate: np.ndarray
    labels: SourceLabels
    ap2_mag: np.ndarray
    kron_mag: np.ndarray
    diff: np.ndarray
    absdiff: np.ndarray
    component_id: np.ndarray
    component_area: np.ndarray
    center_in_bright_region: np.ndarray
    outside_bright_region: np.ndarray
    small_bright_region: np.ndarray
    large_bright_region: np.ndarray
    removed_outside: np.ndarray
    removed_small: np.ndarray
    removed_invalid: np.ndarray


def classify_bright_ap2(
    table: Table,
    candidate: np.ndarray,
    labels: SourceLabels,
    *,
    component_labels: np.ndarray | None,
    config: BrightAp2Config = BrightAp2Config(),
    refit_config: RefitConfig = RefitConfig(),
) -> BrightAp2Result:
    """Apply bright-aware AP2-Kron filtering without remeasurement.

    Rules for bright candidates:

    - center in large bright component (area >= threshold): keep, skip AP2.
    - center in small bright component: require ``abs(AP2-Kron) < 2``.
    - center outside bright components: require ``abs(AP2-Kron) < 1``.
    - invalid AP2/Kron diff fails for outside/small bright components.
    """

    candidate = np.asarray(candidate, dtype=bool)
    geom = compute_kron_ellipse(table, refit_config)
    ap2_mag = magnitude_from_flux(table, column=config.ap2_flux_column, zeropoint=config.zeropoint)
    kron_mag = magnitude_from_flux(table, column=config.kron_flux_column, zeropoint=config.zeropoint)
    diff = ap2_mag - kron_mag
    absdiff = np.abs(diff)
    valid_diff = np.isfinite(absdiff)

    component_id = np.zeros(len(table), dtype=np.int32)
    component_area = np.zeros(len(table), dtype=np.float32)
    if component_labels is not None and np.asarray(component_labels).size:
        component_labels = np.asarray(component_labels, dtype=np.int32)
        area_by_component = component_area_map(component_labels)
        for idx in np.flatnonzero(candidate & np.isfinite(geom.x) & np.isfinite(geom.y)):
            comp = component_at(
                component_labels,
                float(geom.x[idx]),
                float(geom.y[idx]),
                int(config.component_search_radius),
            )
            component_id[idx] = int(comp)
            component_area[idx] = float(area_by_component.get(int(comp), 0))

    center_in = candidate & (component_id > 0)
    large = center_in & (component_area >= float(config.large_bright_region_area_min))
    small = center_in & ~large
    outside = candidate & ~center_in

    removed_invalid = (outside | small) & ~valid_diff
    removed_outside = outside & valid_diff & (absdiff >= float(config.outside_bright_abs_max))
    removed_small = small & valid_diff & (absdiff >= float(config.small_bright_abs_max))
    removed = removed_invalid | removed_outside | removed_small

    labels.assign(removed_invalid, SourceClass.ORDINARY_IGNORE, "bright_ap2_kron_invalid")
    labels.assign(removed_outside, SourceClass.ORDINARY_IGNORE, "bright_ap2_outside_absdiff_ge1")
    labels.assign(removed_small, SourceClass.ORDINARY_IGNORE, "bright_ap2_small_component_absdiff_ge2")
    refined = candidate & ~removed
    return BrightAp2Result(
        candidate=refined,
        labels=labels,
        ap2_mag=ap2_mag,
        kron_mag=kron_mag,
        diff=diff,
        absdiff=absdiff,
        component_id=component_id,
        component_area=component_area,
        center_in_bright_region=center_in,
        outside_bright_region=outside,
        small_bright_region=small,
        large_bright_region=large,
        removed_outside=removed_outside,
        removed_small=removed_small,
        removed_invalid=removed_invalid,
    )
