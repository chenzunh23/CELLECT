"""Catalog-independent A filtering in HSC-equivalent pixel area."""

import numpy as np

HSC_PIXEL_SCALE = 0.168


def area_filter_masks(area, mag, *, pixel_scale_arcsec=HSC_PIXEL_SCALE,
                      area_max=10000.0, faint_area_max=900.0, faint_mag_min=28.0):
    """Return oversized and faint-oversized masks; both mean DROPPED.

    Areas are input-image pixels; thresholds are HSC pixels. Invalid geometry
    and magnitudes remain the responsibility of the catalog adapter.
    """
    if not np.isfinite(pixel_scale_arcsec) or pixel_scale_arcsec <= 0:
        raise ValueError("pixel_scale_arcsec must be finite and positive")
    equivalent = np.asarray(area, dtype=float) * (pixel_scale_arcsec / HSC_PIXEL_SCALE) ** 2
    valid = np.isfinite(equivalent) & (equivalent >= 0)
    # Scale conversion can move an exact boundary by a few floating-point ulps.
    def exceeds(limit):
        return (equivalent > limit) & ~np.isclose(equivalent, limit, rtol=1e-14, atol=0)
    return (valid & exceeds(area_max),
            valid & exceeds(faint_area_max) & (np.asarray(mag) > faint_mag_min))
