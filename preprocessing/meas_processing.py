"""Initial meas-catalog filtering before bright/ordinary branch split."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.table import Table

from .labels import SourceClass, SourceLabels
from .refit import RefitConfig, compute_kron_ellipse
from .utils.catalog import magnitude_from_flux, source_filter_mask
from .utils.geometry import close_pair_dimmer_mask


@dataclass(frozen=True)
class MeasProcessingConfig:
    source_filter: str = "nchild0"
    flux_column: str = "ext_photometryKron_KronFlux_instFlux"
    zeropoint: float = 27.0
    bright_mag_threshold: float = 22.0
    a_area_max: float = 10000.0
    a_faint_area_max: float = 900.0
    a_faint_mag_min: float = 28.0
    b_mag_max: float = 30.0
    b_axis_ratio_max: float = 5.0
    close_center_arcsec: float = 0.5
    pixel_scale_arcsec: float = 0.168


@dataclass
class MeasProcessingResult:
    table: Table
    labels: SourceLabels
    mag: np.ndarray
    bright_candidate: np.ndarray
    ordinary_candidate: np.ndarray
    valid_geometry: np.ndarray
    source_ok: np.ndarray
    refit_valid: np.ndarray
    after_a: np.ndarray
    after_b_basic: np.ndarray
    a_large: np.ndarray
    a_faint_large: np.ndarray
    b_too_faint: np.ndarray
    b_bad_axis: np.ndarray
    b_close_dimmer: np.ndarray


def classify_meas_basics(
    table: Table,
    *,
    config: MeasProcessingConfig = MeasProcessingConfig(),
    refit_config: RefitConfig = RefitConfig(),
) -> MeasProcessingResult:
    """Apply refit-dependent A filter and B-filter basics.

    Bright sources only pass through the shared B basics here: mag <= 30, axis
    ratio <= threshold, and close-pair dimmer removal.  AP2, containment and SNR
    filters are intentionally left to branch-specific modules.
    """

    labels = SourceLabels.empty(len(table), default=SourceClass.DROPPED)
    geom = compute_kron_ellipse(table, refit_config)
    mag = magnitude_from_flux(table, column=config.flux_column, zeropoint=config.zeropoint)

    finite = geom.valid() & np.isfinite(mag)
    source_ok = source_filter_mask(table, config.source_filter)
    base = finite & source_ok

    labels.assign(~source_ok, SourceClass.DROPPED, "source_filter")
    labels.assign(source_ok & ~finite, SourceClass.ORDINARY_IGNORE, "invalid_refit_or_mag")

    a_large = base & (geom.area > config.a_area_max)
    a_faint_large = base & (geom.area > config.a_faint_area_max) & (mag > config.a_faint_mag_min)
    labels.assign(a_large, SourceClass.DROPPED, "A_area_gt_max")
    labels.assign(a_faint_large & ~a_large, SourceClass.ORDINARY_IGNORE, "A_faint_large")

    after_a = base & ~a_large & ~a_faint_large
    too_faint = after_a & (mag > config.b_mag_max)
    bad_axis = after_a & (geom.axis_ratio() > config.b_axis_ratio_max)
    close_dimmer = close_pair_dimmer_mask(
        geom.x,
        geom.y,
        mag,
        radius_pix=config.close_center_arcsec / config.pixel_scale_arcsec,
        candidate=after_a & ~too_faint & ~bad_axis,
    )

    labels.assign(too_faint, SourceClass.ORDINARY_IGNORE, "B_mag_gt_max")
    labels.assign(bad_axis & ~too_faint, SourceClass.ORDINARY_IGNORE, "B_axis_ratio")
    labels.assign(close_dimmer, SourceClass.ORDINARY_IGNORE, "B_close_pair_dimmer")

    kept = after_a & ~too_faint & ~bad_axis & ~close_dimmer
    bright_candidate = kept & (mag < config.bright_mag_threshold)
    ordinary_candidate = kept & ~bright_candidate
    return MeasProcessingResult(
        table=table,
        labels=labels,
        mag=mag,
        bright_candidate=bright_candidate,
        ordinary_candidate=ordinary_candidate,
        valid_geometry=finite,
        source_ok=source_ok,
        refit_valid=base,
        after_a=after_a,
        after_b_basic=kept,
        a_large=a_large,
        a_faint_large=a_faint_large,
        b_too_faint=too_faint,
        b_bad_axis=bad_axis,
        b_close_dimmer=close_dimmer,
    )
