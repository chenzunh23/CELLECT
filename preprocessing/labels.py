"""Shared source and dense-target label definitions for preprocessing v3."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class DenseLabel(IntEnum):
    """Pixel-level region labels written to zarr targets."""

    UNLABELED = 0
    CLEAN = 1
    WEAK_SHAPE = 2
    ORDINARY_IGNORE = 3
    BACKGROUND = 4
    STRICT_CENTER_ONLY = 5
    RESTRICTED_BRIGHT_REGION = 6
    STRICT_IGNORE = 7


class SourceClass(IntEnum):
    """Per-source supervision class after all source filters."""

    DROPPED = 0
    CLEAN = 1
    WEAK_SHAPE = 2
    ORDINARY_IGNORE = 3
    STRICT_CENTER_ONLY = 4
    RESTRICTED_BRIGHT_REGION = 5
    STRICT_IGNORE = 6


SOURCE_TO_DENSE = {
    SourceClass.CLEAN: DenseLabel.CLEAN,
    SourceClass.WEAK_SHAPE: DenseLabel.WEAK_SHAPE,
    SourceClass.ORDINARY_IGNORE: DenseLabel.ORDINARY_IGNORE,
    SourceClass.STRICT_CENTER_ONLY: DenseLabel.STRICT_CENTER_ONLY,
    SourceClass.RESTRICTED_BRIGHT_REGION: DenseLabel.RESTRICTED_BRIGHT_REGION,
    SourceClass.STRICT_IGNORE: DenseLabel.STRICT_IGNORE,
}


@dataclass(frozen=True)
class LabelWeights:
    """Default training weights for dense regions.

    The defaults intentionally make ``background``, ``strict_center_only`` and
    ``restricted_bright_region`` train like negative/background regions.  Their
    class ids remain separate so later fine tuning can assign different weights.
    """

    clean: float = 1.0
    weak_shape: float = 1.0
    ordinary_ignore: float = 0.0
    background: float = 1.0
    strict_center_only: float = 1.0
    restricted_bright_region: float = 1.0
    strict_ignore: float = 0.0

    def as_array(self) -> np.ndarray:
        out = np.zeros(max(int(label) for label in DenseLabel) + 1, dtype=np.float32)
        out[DenseLabel.CLEAN] = self.clean
        out[DenseLabel.WEAK_SHAPE] = self.weak_shape
        out[DenseLabel.ORDINARY_IGNORE] = self.ordinary_ignore
        out[DenseLabel.BACKGROUND] = self.background
        out[DenseLabel.STRICT_CENTER_ONLY] = self.strict_center_only
        out[DenseLabel.RESTRICTED_BRIGHT_REGION] = self.restricted_bright_region
        out[DenseLabel.STRICT_IGNORE] = self.strict_ignore
        return out


@dataclass
class SourceLabels:
    """Vectorized per-source labels plus human-readable reasons."""

    source_class: np.ndarray
    reason: np.ndarray

    @classmethod
    def empty(cls, n: int, *, default: SourceClass = SourceClass.DROPPED) -> "SourceLabels":
        return cls(
            source_class=np.full(n, int(default), dtype=np.int16),
            reason=np.full(n, "unassigned", dtype=object),
        )

    def assign(self, mask: np.ndarray, source_class: SourceClass, reason: str) -> None:
        mask = np.asarray(mask, dtype=bool)
        self.source_class[mask] = int(source_class)
        self.reason[mask] = str(reason)

    def mask(self, source_class: SourceClass) -> np.ndarray:
        return self.source_class == int(source_class)

