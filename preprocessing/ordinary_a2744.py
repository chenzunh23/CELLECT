"""A2744: PSF size, close pairs, iterative flag-aware containment, then SNR."""

from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from .labels import SourceClass as C
from .ordinary_common import start_ordinary, retained, downgrade
from .utils.containment import nearby_pairs, area_fraction, pixel_iou, mask_sample_fraction
from .snr import apply_candidate_snr, apply_remeasured_aperture_snr, remeasure_aperture_snr


@dataclass(frozen=True)
class A2744OrdinaryConfig:
    psf_fwhm_pixels: float
    containment_fraction: float = 0.80
    iou_threshold: float = 0.80
    image_shape: tuple[int, int] | None = None
    strict_flags: tuple[int, ...] = (16, 17, 18, 19)
    aperture_mag_min: float | None = None
    aperture_radius: float = 10.0
    star_overlap_min: float = 0.35
    enable_containment: bool = True
    enable_catalog_snr: bool = True
    enable_aperture_snr: bool = True
    enable_star_mask: bool = True


def flag_priority(flag):
    return {0: 3, 2: 2, 1: 1, 3: 1}.get(int(flag), 0)


def source_containment(data, result, config):
    geom, flags = data.geometry, data.flags
    masks = {name: np.zeros(len(data.mag), bool) for name in
             ("high_iou_ignore", "containment_weak", "containment_strict", "containment_ignore")}
    def change(i, category, reason):
        mask = np.zeros(len(data.mag), bool)
        mask[i] = True
        if category != C.ORDINARY_IGNORE:
            if flags[i] in config.strict_flags:
                category = C.STRICT_CENTER_ONLY
            elif flags[i] not in (0, 1, 2, 3):
                category = C.WEAK_SHAPE
        return downgrade(result, mask, category, reason)

    # Every change lowers one finite rank; convergence is bounded by 3*N.
    for iteration in range(3 * len(data.mag) + 1):
        changed = False
        for i, j in nearby_pairs(geom, retained(result)):
            if result.labels.source_class[i] not in (C.CLEAN, C.WEAK_SHAPE, C.STRICT_CENTER_ONLY) or result.labels.source_class[j] not in (C.CLEAN, C.WEAK_SHAPE, C.STRICT_CENTER_ONLY):
                continue
            if pixel_iou(geom, i, j, config.image_shape) >= config.iou_threshold:
                loser = i if data.mag[i] >= data.mag[j] else j
                masks["high_iou_ignore"][loser] = True
                changed |= change(loser, C.ORDINARY_IGNORE, "a2744_high_iou")
                continue
            ij, ji = area_fraction(geom, i, j), area_fraction(geom, j, i)
            if ij >= config.containment_fraction and (geom.area[i] >= geom.area[j] or ji < config.containment_fraction):
                large, small = i, j
            elif ji >= config.containment_fraction:
                large, small = j, i
            else:
                continue
            lp, sp = flag_priority(flags[large]), flag_priority(flags[small])
            if lp < sp:
                masks["containment_ignore"][large] = True
                changed |= change(large, C.ORDINARY_IGNORE, "a2744_low_priority_parent")
            else:
                category = C.WEAK_SHAPE if lp > sp else C.STRICT_CENTER_ONLY
                masks["containment_weak" if lp > sp else "containment_strict"][large] = True
                masks["containment_weak"][small] = True
                changed |= change(large, category, "a2744_containment_candidate")
                changed |= change(small, C.WEAK_SHAPE, "a2744_containment_candidate")
        if not changed:
            break
    result.diagnostics.update(masks, containment_iterations=iteration + 1)


def label_ordinary_sources(data, candidate, labels, *, config: A2744OrdinaryConfig,
                           aperture_measurement=None, star_hit=None, image=None,
                           sky_mask=None, excluded_mask=None, star_footprint=None, valid_mask=None):
    if data.flags is None or data.snr is None:
        raise ValueError("A2744 requires flags and catalog SNR")
    if not np.isfinite(config.psf_fwhm_pixels) or config.psf_fwhm_pixels <= 0:
        raise ValueError("PSF FWHM must be finite and positive")
    if image is not None and aperture_measurement is not None:
        raise ValueError("provide image or precomputed aperture measurement, not both")
    if star_hit is not None and np.shape(star_hit) != data.mag.shape:
        raise ValueError("star_hit must have one value per source")
    if star_hit is not None and star_footprint is not None:
        raise ValueError("provide star_hit or star_footprint, not both")
    result = start_ordinary(data, candidate, labels, image=image, valid_mask=valid_mask)
    geom = data.geometry
    small = retained(result) & ((geom.major < config.psf_fwhm_pixels) | (geom.minor < config.psf_fwhm_pixels))
    downgrade(result, small, C.ORDINARY_IGNORE, "a2744_psf_size")
    result.snapshot("psf_size")
    idx = np.flatnonzero(retained(result))
    close = np.zeros(len(data.mag), bool)
    if len(idx) > 1:
        tree = cKDTree(np.column_stack((geom.x[idx], geom.y[idx])))
        for i, j in sorted(tree.query_pairs(config.psf_fwhm_pixels)):
            i, j = int(idx[i]), int(idx[j])
            if close[i] or close[j]:
                continue
            loser = i if data.mag[i] < data.mag[j] else j
            close[loser] = True
    downgrade(result, close, C.ORDINARY_IGNORE, "a2744_close_brighter")
    downgrade(result, np.isin(data.flags, config.strict_flags), C.STRICT_CENTER_ONLY, "a2744_flag_candidate")
    result.snapshot("close_pair_flags")
    if config.enable_containment:
        source_containment(data, result, config)
    result.snapshot("containment")
    if config.enable_catalog_snr:
        apply_candidate_snr(result, data.snr)
    else:
        result.snapshot("snr")
    if star_footprint is not None:
        if image is not None and np.shape(star_footprint) != np.shape(image):
            raise ValueError("star footprint/image shape mismatch")
        overlap, center_hit = mask_sample_fraction(geom, star_footprint)
        star_hit = center_hit | (overlap >= config.star_overlap_min)
        result.diagnostics['star_footprint_overlap'] = overlap
        excluded_mask = np.asarray(star_footprint, bool) if excluded_mask is None else np.asarray(excluded_mask, bool) | np.asarray(star_footprint, bool)
    if star_hit is not None and config.enable_star_mask:
        downgrade(result, np.asarray(star_hit, bool) & (data.mag >= 22), C.ORDINARY_IGNORE, "a2744_star_mask")
        result.snapshot("star_mask")
    if image is not None and config.enable_aperture_snr:
        from dataclasses import replace
        measure = retained(result)
        if config.aperture_mag_min is not None:
            measure &= np.isfinite(data.mag) & (data.mag > config.aperture_mag_min)
        aperture_measurement = dict(snr=np.full(len(data.mag), np.nan), trusted=np.zeros(len(data.mag), bool))
        if np.any(measure):
            measurement_geom = replace(geom, x=np.where(measure, geom.x, np.nan),
                                        y=np.where(measure, geom.y, np.nan))
            aperture_measurement = remeasure_aperture_snr(image, measurement_geom, sky_mask=sky_mask,
                excluded_mask=excluded_mask, radius=config.aperture_radius)
    if aperture_measurement is not None and config.enable_aperture_snr:
        apply_remeasured_aperture_snr(result, data.mag, aperture_measurement, mag_min=config.aperture_mag_min)
    elif config.enable_aperture_snr:
        pending = retained(result)
        if config.aperture_mag_min is not None:
            pending &= np.isfinite(data.mag) & (data.mag > config.aperture_mag_min)
        result.diagnostics['aperture_snr_pending'] = pending
    else:
        result.snapshot('aperture_snr')
        result.diagnostics['aperture_snr_skipped'] = True
    result.diagnostics.update(psf_small=small, close_brighter=close)
    return result
