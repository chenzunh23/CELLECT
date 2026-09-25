"""COSMOS: warn categories -> star mask -> SE++ photometry -> containment."""

from dataclasses import dataclass
import numpy as np
from .labels import SourceClass as C
from .ordinary_common import start_ordinary, retained, downgrade
from .utils.containment import nearby_pairs, boundary_fraction
from .snr import apply_candidate_snr


@dataclass(frozen=True)
class CosmosOrdinaryConfig:
    warn_ignore: tuple[int, ...] = (1, 4, 5, 6)
    photometry_center_min: float = 1.0
    photometry_ignore_min: float = 2.0
    containment_fraction: float = 0.995
    fill_bright_mag_max: float = 25.5
    fill_ratio_min: float = 0.1


def label_ordinary_sources(data, candidate, labels, *, config=CosmosOrdinaryConfig(), image=None, valid_mask=None):
    for name in ("flags", "star_mask", "model_mag", "snr"):
        if getattr(data, name) is None:
            raise ValueError(f"COSMOS requires {name}")
    result = start_ordinary(data, candidate, labels, image=image, valid_mask=valid_mask)
    # warn_flag is an enumerated category, not a bit mask.
    warn = retained(result) & np.isin(data.flags, config.warn_ignore)
    downgrade(result, warn, C.ORDINARY_IGNORE, "cosmos_warn_1456")
    result.snapshot("warn")
    star = retained(result) & np.asarray(data.star_mask, dtype=bool)
    downgrade(result, star, C.STRICT_CENTER_ONLY, "cosmos_star_mask")
    result.snapshot("star_mask")
    valid = np.isfinite(data.mag) & (data.mag < 90) & np.isfinite(data.model_mag) & (data.model_mag < 90)
    diff = np.abs(data.mag - data.model_mag)
    phot_ignore = retained(result) & (~valid | (diff > config.photometry_ignore_min))
    downgrade(result, phot_ignore, C.ORDINARY_IGNORE, "cosmos_photometry_ignore")
    phot_center = retained(result) & (diff >= config.photometry_center_min)
    downgrade(result, phot_center, C.STRICT_CENTER_ONLY, "cosmos_photometry_center")
    result.snapshot("photometry")
    contained = np.zeros(len(data.mag), dtype=bool)
    # Low-SNR sources cannot demote a neighbor that survives the final SNR step.
    eligible = retained(result) & np.isfinite(data.snr) & (data.snr > 3)
    geom = data.geometry
    for i, j in nearby_pairs(geom, eligible):
        if contained[i] or contained[j]:
            continue
        small, large = (i, j) if geom.area[i] <= geom.area[j] else (j, i)
        if boundary_fraction(geom, large, small) >= config.containment_fraction:
            contained[large] = True
    downgrade(result, contained, C.STRICT_CENTER_ONLY, "cosmos_containment")
    result.snapshot("containment")
    fill = data.segmentation_fill_ratio
    low_fill = np.zeros(len(data.mag), dtype=bool)
    if fill is not None:
        low_fill = (retained(result) & np.isfinite(data.mag)
                    & (data.mag < config.fill_bright_mag_max)
                    & np.isfinite(fill) & (fill >= 0) & (fill < config.fill_ratio_min))
        downgrade(result, low_fill, C.STRICT_CENTER_ONLY, "cosmos_bright_low_fill")
    result.snapshot("bright_fill")
    apply_candidate_snr(result, data.snr, weak_class=C.STRICT_CENTER_ONLY, ignore_inclusive=True)
    result.diagnostics.update(bright_low_fill=low_fill, segmentation_fill_available=fill is not None)
    result.diagnostics.update(warn_ignore=warn, star_mask=star, photometry_ignore=phot_ignore,
                              photometry_center=phot_center, dmag=diff, containment_center=contained)
    return result
