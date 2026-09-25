"""FITS-unit adaptation and finite-pixel repair; no source-label decisions."""

from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area
from scipy import ndimage

HSC_PIXEL_SCALE_ARCSEC = 0.168
HSC_ZEROPOINT = 27.0
# Keep the conversion convention used by the validated JWST diagnostics.
ARCSEC_PER_RADIAN = 206265.0
JY_AB_ZERO = 3631.0


def pixel_area_arcsec2(header=None, *, pixel_scale_arcsec=None) -> float:
    """Nominal pixel area, not a spatially varying distortion correction."""
    header = {} if header is None else header
    if pixel_scale_arcsec is not None:
        if not np.isfinite(pixel_scale_arcsec) or pixel_scale_arcsec <= 0:
            raise ValueError("pixel_scale_arcsec must be finite and positive")
        area = float(pixel_scale_arcsec) ** 2
    elif "PIXAR_A2" in header:
        area = float(header["PIXAR_A2"])
    elif "PIXAR_SR" in header:
        area = float(header["PIXAR_SR"]) * ARCSEC_PER_RADIAN ** 2
    else:
        wcs = WCS(header)
        if not wcs.has_celestial:
            raise ValueError("per-pixel flux requires pixel scale, PIXAR_A2/PIXAR_SR or celestial WCS")
        area = float(proj_plane_pixel_area(wcs.celestial)) * 3600.0 ** 2
    if not np.isfinite(area) or area <= 0:
        raise ValueError("pixel area must be finite and positive")
    return area


def hsc_surface_brightness_factor(header=None, *, input_unit="auto",
                                  pixel_scale_arcsec=None, input_zeropoint=None):
    """Convert to ZP27 flux in a 0.168-arcsec HSC pixel without resampling.

    MJy/sr is already a surface brightness. Jy (or Jy/pixel) is a per-pixel
    flux and requires the input pixel area. Explicit ``zp-flux`` additionally
    requires its AB zeropoint. ``native``/``hsc-zp27`` are identity adapters.
    """
    header = {} if header is None else header
    unit_text = str(input_unit).strip()
    if unit_text == "auto":
        unit_text = str(header.get("BUNIT", "")).strip()
        if not unit_text:
            raise ValueError("auto unit conversion requires FITS BUNIT; use hsc-zp27 for calibrated HSC input")
    target_area = HSC_PIXEL_SCALE_ARCSEC ** 2
    source_area = None
    if unit_text in {"native", "hsc-zp27"}:
        factor = 1.0
    elif unit_text == "zp-flux":
        if input_zeropoint is None or not np.isfinite(input_zeropoint):
            raise ValueError("zp-flux requires a finite input_zeropoint")
        source_area = pixel_area_arcsec2(header, pixel_scale_arcsec=pixel_scale_arcsec)
        factor = 10.0 ** (0.4 * (HSC_ZEROPOINT - input_zeropoint)) * target_area / source_area
    else:
        # Some input products redundantly annotate surface brightness /pixel.
        unit_text = unit_text.replace("/sr/pixel", "/sr").replace("/sr/pix", "/sr")
        try:
            unit = u.Unit(unit_text)
            if unit.is_equivalent(u.Jy / u.sr):
                factor = unit.to(u.Jy / u.sr) * target_area / ARCSEC_PER_RADIAN ** 2
            else:
                jy_per_pixel = unit.to(u.Jy) if unit.is_equivalent(u.Jy) else unit.to(u.Jy / u.pix)
                source_area = pixel_area_arcsec2(header, pixel_scale_arcsec=pixel_scale_arcsec)
                factor = jy_per_pixel * target_area / source_area
        except (ValueError, TypeError) as exc:
            raise ValueError(f"cannot convert FITS unit {unit_text!r} to HSC surface brightness: {exc}") from exc
        factor *= 10.0 ** (0.4 * HSC_ZEROPOINT) / JY_AB_ZERO
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("surface-brightness conversion factor must be finite and positive")
    return float(factor), {
        "input_unit": unit_text, "unit_factor": float(factor),
        "input_pixel_area_arcsec2": source_area, "input_zeropoint": input_zeropoint,
        "target_pixel_scale_arcsec": HSC_PIXEL_SCALE_ARCSEC,
        "target_zeropoint": HSC_ZEROPOINT, "resampled": False,
    }


def fill_nonfinite_hybrid(image: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """Reference JWST NaN fill: source rings -> nearest; other holes -> smooth.

    Gaussian sigma=32 pixels; an 8-pixel ring is source-like if its median
    exceeds background+3 sigma or its p90 exceeds background+5 sigma.
    Component rings use bounded ROIs, equivalent to full-image dilation.
    """
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("NaN filling requires a nonempty 2-D image")
    bad = ~np.isfinite(arr)
    stats = {"nan_pixels": int(bad.sum()), "nan_components": 0,
             "nan_pixels_nearest_fill": 0, "nan_pixels_background_fill": 0}
    if not bad.any():
        return arr.copy(), stats
    if bad.all():
        stats.update(nan_components=1, nan_pixels_background_fill=int(bad.size))
        return np.zeros_like(arr), stats
    good = ~bad
    values = arr[good].astype(np.float64)
    _, median, std = sigma_clipped_stats(values, sigma=3.0, maxiters=None)
    bg = float(median) if np.isfinite(median) else float(np.median(values))
    sigma = float(std) if np.isfinite(std) and std > 0 else float(np.std(values))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    weights = good.astype(np.float32)
    safe = np.where(good, arr, bg).astype(np.float32)
    numerator = ndimage.gaussian_filter(safe * weights, sigma=32.0, mode="nearest")
    denominator = ndimage.gaussian_filter(weights, sigma=32.0, mode="nearest")
    smooth = np.full_like(arr, bg)
    np.divide(numerator, denominator, out=smooth, where=denominator > 1e-4)
    nearest = arr[tuple(ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True))]
    filled = arr.copy()
    filled[bad] = smooth[bad]
    labels, count = ndimage.label(bad)
    stats["nan_components"] = int(count)
    for component, slices in enumerate(ndimage.find_objects(labels), 1):
        sy, sx = slices
        sy = slice(max(0, sy.start - 8), min(arr.shape[0], sy.stop + 8))
        sx = slice(max(0, sx.start - 8), min(arr.shape[1], sx.stop + 8))
        comp = labels[sy, sx] == component
        ring = ndimage.binary_dilation(comp, iterations=8) & good[sy, sx] & ~comp
        if ring.any():
            ring_values = arr[sy, sx][ring].astype(np.float64)
            if np.median(ring_values) > bg + 3 * sigma or np.percentile(ring_values, 90) > bg + 5 * sigma:
                filled[sy, sx][comp] = nearest[sy, sx][comp]
                stats["nan_pixels_nearest_fill"] += int(comp.sum())
    stats["nan_pixels_background_fill"] = stats["nan_pixels"] - stats["nan_pixels_nearest_fill"]
    return filled, stats
