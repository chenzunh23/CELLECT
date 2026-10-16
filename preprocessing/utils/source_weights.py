"""Size weights use adopted, uncropped GT Kron axes in native image pixels."""
from functools import lru_cache
from pathlib import Path
import json

import numpy as np


@lru_cache(maxsize=1)
def source_area_weight_policy():
    path = Path(__file__).resolve().parents[1] / 'assets/source_area_weights_v1.json'
    policy = json.loads(path.read_text())
    cuts = np.asarray(policy['thresholds_exclusive'], float)
    weights = np.asarray(policy['weights'], float)
    if (cuts.ndim != 1 or len(weights) != len(cuts) + 1
            or not np.isfinite(cuts).all() or not np.all(np.diff(cuts) > 0)
            or not np.isfinite(weights).all() or not np.all(weights > 0)):
        raise ValueError('Invalid source-area weight policy')
    return policy


def kron_area_weights(major, minor):
    major, minor = np.broadcast_arrays(np.asarray(major, float), np.asarray(minor, float))
    result = np.ones(major.shape, np.float32)
    policy = source_area_weight_policy()
    if policy['enabled']:
        valid = np.isfinite(major) & np.isfinite(minor) & (major > 0) & (minor > 0)
        area = np.pi * major[valid] * minor[valid]
        # Strictly greater than each boundary: 2500 -> 1, 10000 -> 2.
        index = np.searchsorted(policy['thresholds_exclusive'], area, side='left')
        result[valid] = np.asarray(policy['weights'], np.float32)[index]
    return result
