"""Geometry helpers for source filtering and dense target painting."""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import math

import numpy as np


@dataclass
class EllipseGeometry:
    x: np.ndarray
    y: np.ndarray
    major: np.ndarray
    minor: np.ndarray
    theta: np.ndarray
    area: np.ndarray

    def valid(self) -> np.ndarray:
        return (
            np.isfinite(self.x)
            & np.isfinite(self.y)
            & np.isfinite(self.major)
            & np.isfinite(self.minor)
            & np.isfinite(self.theta)
            & (self.major > 0.0)
            & (self.minor > 0.0)
            & np.isfinite(self.area)
            & (self.area > 0.0)
        )

    def axis_ratio(self) -> np.ndarray:
        denom = np.maximum(np.minimum(self.major, self.minor), 1e-6)
        return np.maximum(self.major, self.minor) / denom


def ellipse_axes_from_moments(xx: np.ndarray, yy: np.ndarray, xy: np.ndarray) -> dict[str, np.ndarray]:
    """Return SDSS-style ellipse axes from second moments.

    ``a`` and ``b`` are one-sigma semi-axes in pixels. ``determinant_radius`` is
    sqrt(a*b), matching the radius normalization used by LSST KronFlux and by
    ``batch_heavyfp_kron_refit.py``.
    """

    xx = np.asarray(xx, dtype=np.float64)
    yy = np.asarray(yy, dtype=np.float64)
    xy = np.asarray(xy, dtype=np.float64)
    trace = xx + yy
    delta = np.sqrt(np.maximum((xx - yy) ** 2 + 4.0 * xy * xy, 0.0))
    lambda_major = 0.5 * (trace + delta)
    lambda_minor = 0.5 * (trace - delta)
    valid = (
        np.isfinite(lambda_major)
        & np.isfinite(lambda_minor)
        & (lambda_major > 0.0)
        & (lambda_minor > 0.0)
    )
    a = np.full(xx.shape, np.nan, dtype=np.float64)
    b = np.full(xx.shape, np.nan, dtype=np.float64)
    theta_rad = np.full(xx.shape, np.nan, dtype=np.float64)
    determinant_radius = np.full(xx.shape, np.nan, dtype=np.float64)
    a[valid] = np.sqrt(lambda_major[valid])
    b[valid] = np.sqrt(lambda_minor[valid])
    theta_rad[valid] = 0.5 * np.arctan2(2.0 * xy[valid], xx[valid] - yy[valid])
    determinant_radius[valid] = np.sqrt(a[valid] * b[valid])
    return {
        "a": a,
        "b": b,
        "theta_rad": theta_rad,
        "theta_deg": np.degrees(theta_rad),
        "determinant_radius": determinant_radius,
        "valid": valid,
    }


def close_pair_dimmer_mask(x: np.ndarray, y: np.ndarray, mag: np.ndarray, radius_pix: float, candidate: np.ndarray) -> np.ndarray:
    """Return sources to drop from close pairs by removing the dimmer member."""

    candidate = np.asarray(candidate, dtype=bool)
    out = np.zeros(len(x), dtype=bool)
    idx = np.flatnonzero(candidate & np.isfinite(x) & np.isfinite(y) & np.isfinite(mag))
    radius2 = float(radius_pix) ** 2
    for pos, i in enumerate(idx):
        if out[i]:
            continue
        rest = idx[pos + 1 :]
        if len(rest) == 0:
            continue
        dist2 = (x[rest] - x[i]) ** 2 + (y[rest] - y[i]) ** 2
        for j in rest[dist2 < radius2]:
            if not np.isfinite(mag[j]):
                out[j] = True
            elif not np.isfinite(mag[i]):
                out[i] = True
            elif mag[i] >= mag[j]:
                out[i] = True
            else:
                out[j] = True
    return out


def ellipse_bbox(x: float, y: float, major: float, minor: float, theta: float, shape: tuple[int, int]) -> tuple[slice, slice]:
    radius = int(np.ceil(max(float(major), float(minor)))) + 2
    h, w = shape
    x0 = max(0, int(np.floor(x)) - radius)
    x1 = min(w, int(np.floor(x)) + radius + 1)
    y0 = max(0, int(np.floor(y)) - radius)
    y1 = min(h, int(np.floor(y)) + radius + 1)
    return slice(y0, y1), slice(x0, x1)


def paint_ellipse(mask: np.ndarray, x: float, y: float, major: float, minor: float, theta: float, value: int | bool = True) -> None:
    ys, xs = ellipse_bbox(x, y, major, minor, theta, mask.shape)
    if ys.stop <= ys.start or xs.stop <= xs.start:
        return
    yy, xx = np.mgrid[ys, xs]
    dx = xx.astype(np.float64) - float(x)
    dy = yy.astype(np.float64) - float(y)
    c = np.cos(float(theta))
    s = np.sin(float(theta))
    xp = c * dx + s * dy
    yp = -s * dx + c * dy
    inside = (xp / max(float(major), 1e-6)) ** 2 + (yp / max(float(minor), 1e-6)) ** 2 <= 1.0
    mask[ys, xs][inside] = value


def component_at(labels: np.ndarray, x: float, y: float, search_radius: int) -> int:
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    if 0 <= xi < labels.shape[1] and 0 <= yi < labels.shape[0]:
        value = int(labels[yi, xi])
        if value > 0:
            return value
    r = int(search_radius)
    if r <= 0:
        return 0
    x0 = max(0, xi - r)
    x1 = min(labels.shape[1], xi + r + 1)
    y0 = max(0, yi - r)
    y1 = min(labels.shape[0], yi + r + 1)
    window = np.asarray(labels[y0:y1, x0:x1], dtype=np.int64)
    values = window[window > 0]
    if values.size == 0:
        return 0
    return int(np.bincount(values.ravel()).argmax())


def ellipse_contains_dict(row: dict[str, object], xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    x0 = float(row["x"])
    y0 = float(row["y"])
    major = max(float(row["major"]), 1.0e-6)
    minor = max(float(row["minor"]), 1.0e-6)
    theta = math.radians(float(row["theta_deg"]))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    dx = xs - x0
    dy = ys - y0
    xr = dx * cos_t + dy * sin_t
    yr = -dx * sin_t + dy * cos_t
    return (xr / major) ** 2 + (yr / minor) ** 2 <= 1.0


def approximate_ellipse_iou(a: dict[str, object], b: dict[str, object]) -> float:
    max_a = max(float(a["major"]), float(a["minor"]))
    max_b = max(float(b["major"]), float(b["minor"]))
    if math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"])) > max_a + max_b:
        return 0.0
    x0 = math.floor(min(float(a["x"]) - max_a, float(b["x"]) - max_b))
    x1 = math.ceil(max(float(a["x"]) + max_a, float(b["x"]) + max_b))
    y0 = math.floor(min(float(a["y"]) - max_a, float(b["y"]) - max_b))
    y1 = math.ceil(max(float(a["y"]) + max_a, float(b["y"]) + max_b))
    if x1 < x0 or y1 < y0:
        return 0.0
    step = max(1, int(math.ceil(max(x1 - x0 + 1, y1 - y0 + 1) / 512.0)))
    yy, xx = np.mgrid[y0 : y1 + 1 : step, x0 : x1 + 1 : step]
    ma = ellipse_contains_dict(a, xx.astype(np.float32), yy.astype(np.float32))
    mb = ellipse_contains_dict(b, xx.astype(np.float32), yy.astype(np.float32))
    union = int(np.count_nonzero(ma | mb))
    if union == 0:
        return 0.0
    return float(np.count_nonzero(ma & mb) / union)


class UnionFind:
    def __init__(self, items: list[int]) -> None:
        self.parent = {int(item): int(item) for item in items}

    def find(self, item: int) -> int:
        parent = self.parent[int(item)]
        if parent != item:
            self.parent[int(item)] = self.find(parent)
        return self.parent[int(item)]

    def union(self, a: int, b: int) -> None:
        ra = self.find(int(a))
        rb = self.find(int(b))
        if ra != rb:
            self.parent[rb] = ra


def cluster_sources(
    indices: list[int],
    sources: list[dict[str, object]],
    *,
    iou_threshold: float,
    max_center_distance: float,
    max_area: float,
) -> list[list[int]]:
    if not indices:
        return []
    uf = UnionFind(indices)
    sorted_indices = sorted(indices, key=lambda idx: float(sources[idx]["x"]))
    area = {idx: max(float(sources[idx]["area"]), 1.0e-6) for idx in sorted_indices}
    for pos, idx_a in enumerate(sorted_indices):
        src_a = sources[idx_a]
        xa = float(src_a["x"])
        ya = float(src_a["y"])
        if area[idx_a] >= float(max_area):
            continue
        for idx_b in sorted_indices[pos + 1 :]:
            dx = float(sources[idx_b]["x"]) - xa
            if dx > float(max_center_distance):
                break
            if area[idx_b] >= float(max_area):
                continue
            dy = float(sources[idx_b]["y"]) - ya
            if dx * dx + dy * dy > float(max_center_distance) ** 2:
                continue
            if min(area[idx_a], area[idx_b]) / max(area[idx_a], area[idx_b]) < float(iou_threshold):
                continue
            if approximate_ellipse_iou(src_a, sources[idx_b]) >= float(iou_threshold):
                uf.union(idx_a, idx_b)
    groups: dict[int, list[int]] = defaultdict(list)
    for idx in indices:
        groups[uf.find(idx)].append(idx)
    return sorted(groups.values(), key=lambda group: min(group))
