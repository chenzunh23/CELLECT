#!/usr/bin/env python3
"""Shared source-filtering constants for CELLECT data diagnostics.

This module is intentionally limited to data-standard definitions.  It does
not create training products such as dense targets, PT tensors, or zarr stores.
"""

from __future__ import annotations

PIXEL_SCALE_ARCSEC = 0.168
SOURCE_FILTER = "nchild0"

BROAD_BANDS = ("HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y")
NARROW_BANDS = ("NB0387", "NB0816", "NB0921", "NB0924", "NB1010")

BAND_LIMIT_MAG = {
    "HSC-G": 27.4,
    "HSC-R": 27.1,
    "HSC-I": 26.9,
    "HSC-Z": 26.3,
    "HSC-Y": 25.3,
}

SATURATION_MAG = {
    "HSC-G": 18.0,
    "HSC-R": 18.2,
    "HSC-I": 18.6,
    "HSC-Z": 17.7,
    "HSC-Y": 17.4,
    "NB0387": 14.8,
    "NB0816": 16.8,
    "NB0921": 16.9,
    "NB0924": 16.9,
    "NB1010": 14.8,
}

COADD_FILTER_STANDARD = {
    "a_area_max": 10000.0,
    "a_faint_area_max": 900.0,
    "a_faint_mag_min": 28.0,
    "b_mag_min": 18.0,
    "b_mag_max": 30.0,
    "band_limit_min_offset": -5.0,
    "band_limit_max_offset": 0.0,
    "close_center_arcsec": 0.5,
    "axis_ratio_max": 5.0,
    "containment_threshold": 0.80,
    "ap2_kron_abs_max": 1.0,
    "ap2_kron_small_bright_area_reject": True,
    "ap2_kron_small_bright_area_ratio_max": 1.0,
    "ap2_kron_small_bright_area_abs_min": 1.0,
    "center_only_fill_area_min": 500.0,
    "center_only_fill_ratio_max": 0.3,
    "b_flags": ("base_SdssShape_flag", "base_SdssCentroid_flag"),
    "kron_refit_radius_column": "proxy_nan0_flux_aperture_radius",
    "kron_refit_good_column": "proxy_nan0_good",
}

COADD_AP2_SNR_POST_FILTER = {
    "broad_ignore_snr_max": 3.0,
    "broad_center_only_snr_max": 5.0,
    "narrow_ignore_snr_max": 5.0,
    "narrow_center_only_snr_max": 8.0,
    "area_center_only_min": 500.0,
    "area_center_only_snr_max": 8.0,
}

VARIANCE_SNR_DIAGNOSTIC = {
    "ap_radius": 6.0,
    "ignore_snr_max": 3.0,
    "center_only_snr_max": 5.0,
    "cap_t_max": 1.0,
    "scale_max_sources": 5000,
}

WEIGHT_RATIO_SNR_DIAGNOSTIC = {
    "ap_radius": 6.0,
    "ignore_snr_max": 3.0,
    "center_only_snr_max": 5.0,
    "cap_t_max": 1.0,
    "use_local_effective_count": True,
}

NONCOADD_IMAGE_SNR_FILTER = {
    "ap_radius": 6.0,
    "annulus_r_in": 10.0,
    "annulus_r_out": 15.0,
    "min_annulus_pixels": 50,
    "source_mask_ellipse_sigma": 1.0,
    "default_ignore_snr_thresh": 2.0,
    "default_center_only_snr_thresh": 3.0,
    "quality_mask_planes": ("BRIGHT_OBJECT", "SAT", "BAD", "NO_DATA", "EDGE", "UNMASKEDNAN"),
}
