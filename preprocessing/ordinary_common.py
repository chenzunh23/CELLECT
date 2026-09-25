"""Ordinary interface: pixel/radian geometry and explicit band measurements.

Stage snapshots contain candidates, not permission to bypass later SNR rules.
"""

from dataclasses import dataclass, field
import numpy as np
from .labels import SourceClass, SourceLabels
from .utils.geometry import EllipseGeometry


@dataclass
class OrdinaryInput:
    geometry: EllipseGeometry
    mag: np.ndarray
    snr: np.ndarray | None = None
    flags: np.ndarray | None = None
    star_mask: np.ndarray | None = None
    model_mag: np.ndarray | None = None
    table: object = None
    segmentation_fill_ratio: np.ndarray | None = None
    center_invalid: np.ndarray | None = None

    def __post_init__(self):
        n = len(self.geometry.x)
        for name in ("mag", "snr", "flags", "star_mask", "model_mag", "segmentation_fill_ratio", "center_invalid"):
            value = getattr(self, name)
            if value is not None:
                value = np.asarray(value)
                if value.shape != (n,):
                    raise ValueError(f"{name} must have one value per source")
                setattr(self, name, value)
        for name in ("y", "major", "minor", "theta", "area"):
            if np.shape(getattr(self.geometry, name)) != (n,):
                raise ValueError(f"geometry.{name} length mismatch")


@dataclass
class DatasetOrdinaryResult:
    labels: SourceLabels
    candidate: np.ndarray
    stages: dict[str, np.ndarray] = field(default_factory=dict)
    diagnostics: dict[str, object] = field(default_factory=dict)

    def snapshot(self, name):
        self.stages[name] = self.labels.source_class.copy()

    def mask(self, category):
        return self.candidate & self.labels.mask(category)

    @property
    def clean(self):
        return self.mask(SourceClass.CLEAN)

    @property
    def weak_shape(self):
        return self.mask(SourceClass.WEAK_SHAPE)

    @property
    def strict_center_only(self):
        return self.mask(SourceClass.STRICT_CENTER_ONLY)

    @property
    def ordinary_ignore(self):
        return self.mask(SourceClass.ORDINARY_IGNORE)


def start_ordinary(data, candidate, labels, *, image=None, valid_mask=None):
    candidate = np.asarray(candidate, dtype=bool)
    if candidate.shape != data.mag.shape or labels.source_class.shape != candidate.shape:
        raise ValueError("candidate/labels length mismatch")
    from .utils.no_data import center_ignore_mask, apply_center_ignore
    nan_ignore=center_ignore_mask(data.geometry,image=image,valid_mask=valid_mask,
        center_invalid=data.center_invalid)
    apply_center_ignore(labels,nan_ignore)
    blocked = np.isin(labels.source_class, [SourceClass.DROPPED, SourceClass.ORDINARY_IGNORE,
                                          SourceClass.STRICT_IGNORE]) & (labels.reason != "unassigned")
    active = candidate & ~blocked
    labels.assign(active & (labels.reason == "unassigned"), SourceClass.CLEAN, "ordinary_candidate")
    labels.assign(active & ~data.geometry.valid(), SourceClass.ORDINARY_IGNORE, "ordinary_invalid_geometry")
    result=DatasetOrdinaryResult(labels, active)
    result.diagnostics['nan_center_ignore']=nan_ignore
    result.snapshot('center_validity')
    return result


def retained(result):
    return result.candidate & np.isin(result.labels.source_class, [
        SourceClass.CLEAN, SourceClass.WEAK_SHAPE, SourceClass.STRICT_CENTER_ONLY])


def downgrade(result, mask, category, reason):
    rank = {SourceClass.CLEAN: 3, SourceClass.WEAK_SHAPE: 2,
            SourceClass.STRICT_CENTER_ONLY: 1, SourceClass.ORDINARY_IGNORE: 0}
    eligible = np.zeros(result.candidate.shape, dtype=bool)
    for current, value in rank.items():
        if value > rank[category]:
            eligible |= result.labels.mask(current)
    changed = result.candidate & np.asarray(mask, dtype=bool) & eligible
    result.labels.assign(changed, category, reason)
    return bool(changed.any())


def classify_ordinary_sources(data, candidate, labels, *, dataset, **kwargs):
    if dataset == "hsc":
        from .ordinary_hsc import label_ordinary_sources
        if data.table is None:
            raise ValueError("HSC ordinary rules require the original meas table")
        legacy = label_ordinary_sources(data.table, candidate, labels, snr=data.snr, **kwargs)
        result = DatasetOrdinaryResult(legacy.labels, np.asarray(candidate, dtype=bool))
        result.snapshot("hsc_ordinary")
        result.diagnostics["legacy"] = legacy
        return result
    if dataset == "cosmos":
        from .ordinary_cosmos import label_ordinary_sources
    elif dataset == "a2744":
        from .ordinary_a2744 import label_ordinary_sources
    else:
        raise ValueError(f"unknown ordinary dataset: {dataset!r}")
    return label_ordinary_sources(data, candidate, labels, **kwargs)
