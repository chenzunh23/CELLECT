"""Convert per-source labels into dense region and confidence targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.table import Table

from .labels import DenseLabel, SOURCE_TO_DENSE, SourceClass, SourceLabels
from .refit import RefitConfig, compute_kron_ellipse
from .utils.geometry import EllipseGeometry, paint_ellipse


@dataclass(frozen=True)
class RegionFillingConfig:
    """Dense-target priority, highest priority first.

    Full source ellipses exclude background; only clean/weak dense shapes are
    truncated, preserving their axis ratios. DROPPED sources have no coverage.
    """

    class_priority: tuple[DenseLabel, ...] = (
        DenseLabel.CLEAN,
        DenseLabel.WEAK_SHAPE,
        DenseLabel.RESTRICTED_BRIGHT_REGION,
        DenseLabel.STRICT_IGNORE,
        DenseLabel.BACKGROUND,
        DenseLabel.ORDINARY_IGNORE,
    )
    max_major_pixels: float | None = 100.0


def fill_dense_regions(
    table: Table,
    labels: SourceLabels,
    shape: tuple[int, int],
    *,
    background_mask: np.ndarray | None = None,
    quality_ignore_mask: np.ndarray | None = None,
    restricted_fallback_mask: np.ndarray | None = None,
    ordinary_ignore_mask: np.ndarray | None = None,
    ordinary_ignore_source_mask: np.ndarray | None = None,
    bright_region_mask: np.ndarray | None = None,
    geometry: EllipseGeometry | None = None,
    config: RegionFillingConfig = RegionFillingConfig(),
    refit_config: RefitConfig = RefitConfig(),
) -> np.ndarray:
    """Paint dense source regions using the v3 target priority.

    Paint ignore -> LSST background -> full source/bright ignore coverage ->
    retained bright regions -> truncated weak shapes -> truncated clean shapes.
    Center-only sources supply confidence points, not dense shape supervision.
    ``geometry`` permits JWST catalogs to reuse painting without HSC columns.
    The legacy ordinary_ignore_source_mask is accepted for API compatibility;
    all ordinary-ignore sources now exclude background, not just that subset.
    """

    cap = config.max_major_pixels
    if cap is not None and (not np.isfinite(cap) or cap <= 0):
        raise ValueError("max_major_pixels must be positive and finite, or None")
    geom = geometry if geometry is not None else compute_kron_ellipse(table, refit_config)
    if len(geom.x) != len(labels.source_class):
        raise ValueError("geometry and labels length mismatch")
    valid = geom.valid()
    dense = np.full(shape, int(DenseLabel.ORDINARY_IGNORE), dtype=np.uint8)
    if background_mask is not None:
        if np.shape(background_mask) != shape:
            raise ValueError("background mask shape mismatch")
        dense[np.asarray(background_mask, dtype=bool)] = int(DenseLabel.BACKGROUND)
    # All non-dropped shapes erase background at their ORIGINAL size. Clean and
    # weak cores are restored below; their truncated outskirts stay ignore.
    for idx in np.flatnonzero(valid & ~labels.mask(SourceClass.DROPPED)):
        paint_ellipse(dense, float(geom.x[idx]), float(geom.y[idx]),
                      float(geom.major[idx]), float(geom.minor[idx]), float(geom.theta[idx]),
                      int(DenseLabel.ORDINARY_IGNORE))
    for mask in (bright_region_mask, restricted_fallback_mask, ordinary_ignore_mask):
        if mask is not None:
            if np.shape(mask) != shape:
                raise ValueError("region mask shape mismatch")
            dense[np.asarray(mask, dtype=bool)] = int(DenseLabel.ORDINARY_IGNORE)
    if quality_ignore_mask is not None:
        # FITS SAT/BAD/EDGE/NO_DATA/UNMASKEDNAN masks beat background but not
        # trusted source regions painted below.
        dense[np.asarray(quality_ignore_mask, dtype=bool)] = int(DenseLabel.STRICT_IGNORE)
    class_by_label = {dense_label: source_class for source_class, dense_label in SOURCE_TO_DENSE.items()}
    paint_order = [label for label in reversed(config.class_priority) if label not in {DenseLabel.ORDINARY_IGNORE, DenseLabel.BACKGROUND}]
    for dense_label in paint_order:
        if dense_label == DenseLabel.RESTRICTED_BRIGHT_REGION and restricted_fallback_mask is not None:
            dense[np.asarray(restricted_fallback_mask, dtype=bool)] = int(dense_label)
        source_class = class_by_label.get(dense_label)
        if source_class is None:
            continue
        for idx in np.flatnonzero(labels.mask(source_class)):
            if not valid[idx]:
                continue
            factor = 1.0
            if cap is not None and source_class in (SourceClass.CLEAN, SourceClass.WEAK_SHAPE):
                factor = min(1.0, cap / max(geom.major[idx], geom.minor[idx]))
            paint_ellipse(
                dense,
                float(geom.x[idx]),
                float(geom.y[idx]),
                float(geom.major[idx]) * factor,
                float(geom.minor[idx]) * factor,
                float(geom.theta[idx]),
                int(dense_label),
            )
    return dense


def confidence_points(table: Table, labels: SourceLabels, *, refit_config: RefitConfig = RefitConfig()) -> np.ndarray:
    """Return point-like confidence supervision rows: x, y, source_class."""

    geom = compute_kron_ellipse(table, refit_config)
    trainable = (
        labels.mask(SourceClass.CLEAN)
        | labels.mask(SourceClass.WEAK_SHAPE)
        | labels.mask(SourceClass.STRICT_CENTER_ONLY)
    )
    valid = trainable & np.isfinite(geom.x) & np.isfinite(geom.y)
    return np.column_stack([geom.x[valid], geom.y[valid], labels.source_class[valid].astype(np.float64)]).astype(np.float32)
