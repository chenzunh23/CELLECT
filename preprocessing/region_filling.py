"""Convert per-source labels into dense region and confidence targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.table import Table

from .labels import DenseLabel, SOURCE_TO_DENSE, SourceClass, SourceLabels
from .refit import RefitConfig, compute_kron_ellipse
from .utils.geometry import paint_ellipse


@dataclass(frozen=True)
class RegionFillingConfig:
    """Dense-target priority, highest priority first.

    ``ordinary_ignore`` is the fallback complement and is not painted from
    source ellipses by default.  The actual overwrite order is the reverse of
    this tuple after initializing the full image to ordinary-ignore.
    """

    class_priority: tuple[DenseLabel, ...] = (
        DenseLabel.CLEAN,
        DenseLabel.WEAK_SHAPE,
        DenseLabel.STRICT_CENTER_ONLY,
        DenseLabel.RESTRICTED_BRIGHT_REGION,
        DenseLabel.STRICT_IGNORE,
        DenseLabel.BACKGROUND,
        DenseLabel.ORDINARY_IGNORE,
    )


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
    config: RegionFillingConfig = RegionFillingConfig(),
    refit_config: RefitConfig = RefitConfig(),
) -> np.ndarray:
    """Paint dense source regions using the v3 target priority.

    The dense image starts as ``ordinary_ignore`` everywhere.  Explicit LSST
    background, FITS quality masks and trusted source regions then overwrite it
    in increasing priority.  This makes ordinary-ignore the complement of all
    higher-priority regions, while keeping SAT/BAD/EDGE masks distinct from
    generic ignore.
    """

    dense = np.full(shape, int(DenseLabel.ORDINARY_IGNORE), dtype=np.uint8)
    if background_mask is not None:
        dense[np.asarray(background_mask, dtype=bool)] = int(DenseLabel.BACKGROUND)
    if quality_ignore_mask is not None:
        # FITS SAT/BAD/EDGE/NO_DATA/UNMASKEDNAN masks beat background but not
        # trusted source regions painted below.
        dense[np.asarray(quality_ignore_mask, dtype=bool)] = int(DenseLabel.STRICT_IGNORE)
    if restricted_fallback_mask is not None:
        fallback = np.asarray(restricted_fallback_mask, dtype=bool)
        replaceable = (
            (dense == int(DenseLabel.ORDINARY_IGNORE))
            | (dense == int(DenseLabel.BACKGROUND))
            | (dense == int(DenseLabel.STRICT_IGNORE))
        )
        dense[fallback & replaceable] = int(DenseLabel.RESTRICTED_BRIGHT_REGION)
    if ordinary_ignore_mask is not None:
        ignore = np.asarray(ordinary_ignore_mask, dtype=bool)
        replaceable = (
            (dense == int(DenseLabel.ORDINARY_IGNORE))
            | (dense == int(DenseLabel.BACKGROUND))
            | (dense == int(DenseLabel.RESTRICTED_BRIGHT_REGION))
        )
        dense[ignore & replaceable] = int(DenseLabel.ORDINARY_IGNORE)
    geom = compute_kron_ellipse(table, refit_config)
    if ordinary_ignore_source_mask is not None:
        source_mask = np.asarray(ordinary_ignore_source_mask, dtype=bool)
        source_mask &= labels.mask(SourceClass.ORDINARY_IGNORE)
        for idx in np.flatnonzero(source_mask):
            if not np.isfinite(geom.major[idx]):
                continue
            paint_ellipse(
                dense,
                float(geom.x[idx]),
                float(geom.y[idx]),
                float(geom.major[idx]),
                float(geom.minor[idx]),
                float(geom.theta[idx]),
                int(DenseLabel.ORDINARY_IGNORE),
            )
    class_by_label = {dense_label: source_class for source_class, dense_label in SOURCE_TO_DENSE.items()}
    paint_order = [label for label in reversed(config.class_priority) if label not in {DenseLabel.ORDINARY_IGNORE, DenseLabel.BACKGROUND}]
    for dense_label in paint_order:
        source_class = class_by_label.get(dense_label)
        if source_class is None:
            continue
        for idx in np.flatnonzero(labels.mask(source_class)):
            if not np.isfinite(geom.major[idx]):
                continue
            paint_ellipse(
                dense,
                float(geom.x[idx]),
                float(geom.y[idx]),
                float(geom.major[idx]),
                float(geom.minor[idx]),
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
