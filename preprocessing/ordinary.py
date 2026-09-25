"""Backward-compatible HSC entry points and dataset-independent dispatch."""

from .ordinary_hsc import (
    OrdinaryConfig, OrdinaryResult, ap2_kron_keep_mask, aperture_fill_ratio,
    b_flag_removed_mask, label_ordinary_sources, source_containment_removed_mask,
)
from .ordinary_common import OrdinaryInput, DatasetOrdinaryResult, classify_ordinary_sources
