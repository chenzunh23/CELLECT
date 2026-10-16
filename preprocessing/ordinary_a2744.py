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
    ignore_flags: tuple[int, ...] = (16, 17, 18, 19)
    aperture_mag_min: float | None = None
    aperture_radius: float = 10.0
    star_overlap_min: float = 0.35
    enable_containment: bool = True
    enable_catalog_snr: bool = True
    enable_aperture_snr: bool = True
    enable_star_mask: bool = True


def flag_priority(flag):
    return {0: 3, 2: 2, 1: 1, 3: 1}.get(int(flag), 0)


def protect_ignored_children(data, result, config):
    """Keep only center supervision for a usable raw Kron containing ignore.

    Include valid ignored catalog sources regardless of the rejection stage.
    Dropped sources and invalid centers cannot establish this relationship.
    """
    geom = data.geometry
    valid = geom.valid() & ~result.diagnostics.get('nan_center_ignore', np.zeros(len(data.mag), bool))
    children = valid & (result.labels.source_class == C.ORDINARY_IGNORE)
    parents = valid & result.candidate & np.isin(result.labels.source_class, [C.CLEAN, C.WEAK_SHAPE])
    changed = np.zeros(len(data.mag), bool)
    # Geometry is immutable across rejection stages; reuse containment tests
    # when SNR or bright filtering adds ignored children later in the pipeline.
    cache = result.diagnostics.setdefault('_ignore_containment_cache', {})
    if parents.any() and children.any():
        for i, j in nearby_pairs(geom, parents | children):
            if parents[i] and children[j]:
                outer, inner = i, j
            elif parents[j] and children[i]:
                outer, inner = j, i
            else:
                continue
            if changed[outer] or geom.area[outer] < geom.area[inner]:
                continue
            pair = (outer, inner)
            if pair not in cache:
                cache[pair] = area_fraction(geom, outer, inner) >= config.containment_fraction
            if cache[pair]:
                changed[outer] = True
        downgrade(result, changed, C.STRICT_CENTER_ONLY, 'a2744_contains_ignore')
    previous = result.diagnostics.get('ignore_child_parents', np.zeros(len(data.mag), bool))
    result.diagnostics['ignore_child_parents'] = previous | changed
    return changed


def source_containment(data, result, config):
    geom, flags = data.geometry, data.flags
    # Hard-ignored flag sources still participate geometrically. Other ignored
    # sources (invalid centers, PSF rejection, etc.) keep the existing exclusion.
    flag_ignore = result.diagnostics.get("flag_ignore", np.zeros(len(data.mag), bool))
    masks = {name: np.zeros(len(data.mag), bool) for name in
             ("high_iou_ignore", "containment_weak", "containment_strict", "containment_ignore")}
    usable = (C.CLEAN, C.WEAK_SHAPE, C.STRICT_CENTER_ONLY)
    def change(i, category, reason):
        mask = np.zeros(len(data.mag), bool)
        mask[i] = True
        if category != C.ORDINARY_IGNORE:
            if flag_ignore[i]:
                category = C.ORDINARY_IGNORE
            elif flags[i] not in (0, 1, 2, 3):
                category = C.WEAK_SHAPE
        return downgrade(result, mask, category, reason)

    # Every change lowers one finite rank; convergence is bounded by 3*N.
    for iteration in range(3 * len(data.mag) + 1):
        changed = False
        active = retained(result) | flag_ignore
        for i, j in nearby_pairs(geom, active):
            if not (result.labels.source_class[i] in usable or flag_ignore[i]) or not (result.labels.source_class[j] in usable or flag_ignore[j]):
                continue
            if flag_ignore[i] and flag_ignore[j]:
                continue
            if pixel_iou(geom, i, j, config.image_shape) >= config.iou_threshold:
                # A hard-ignored neighbor never suppresses a usable source,
                # including when it is brighter and nearly coincident.
                if flag_ignore[i] or flag_ignore[j]:
                    loser = i if flag_ignore[i] else j
                else:
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
            lp = -1 if flag_ignore[large] else flag_priority(flags[large])
            sp = -1 if flag_ignore[small] else flag_priority(flags[small])
            if lp < sp:
                masks["containment_ignore"][large] = True
                changed |= change(large, C.ORDINARY_IGNORE, "a2744_low_priority_parent")
            elif flag_ignore[small]:
                masks['containment_strict'][large] = True
                mask = np.zeros(len(data.mag), bool)
                mask[large] = True
                changed |= downgrade(result, mask, C.STRICT_CENTER_ONLY, 'a2744_contains_ignore')
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
    flag_ignore = result.candidate & geom.valid() & np.isin(data.flags, config.ignore_flags)
    downgrade(result, flag_ignore, C.ORDINARY_IGNORE, "a2744_flag_ignore")
    result.diagnostics['flag_ignore'] = flag_ignore
    result.snapshot('flag_ignore')
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
    result.snapshot("close_pair_flags")
    if config.enable_containment:
        protect_ignored_children(data, result, config)
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
    if config.enable_containment:
        protect_ignored_children(data, result, config)
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
    if aperture_measurement is not None and config.enable_aperture_snr:
        result.diagnostics['aperture_measurement'] = aperture_measurement
    if config.enable_containment:
        # Aperture rejection can create new ignored children. Newly strict
        # parents must also pass the existing strict aperture-SNR policy.
        while protect_ignored_children(data, result, config).any():
            if aperture_measurement is not None and config.enable_aperture_snr:
                apply_remeasured_aperture_snr(result, data.mag, aperture_measurement, mag_min=config.aperture_mag_min)
        result.snapshot('ignore_child_protection')
    result.diagnostics.update(psf_small=small, close_brighter=close)
    return result
