"""Ellipse neighborhood and overlap measurements in pixels/radians."""

import math
import numpy as np
from scipy.spatial import cKDTree


def nearby_pairs(geom, candidate):
    idx = np.flatnonzero(np.asarray(candidate, dtype=bool) & geom.valid())
    if len(idx) < 2:
        return
    xy = np.column_stack((geom.x[idx], geom.y[idx]))
    radius = np.maximum(geom.major[idx], geom.minor[idx])
    tree = cKDTree(xy)
    largest = float(radius.max())
    for li, i in enumerate(idx):
        for lj in tree.query_ball_point(xy[li], radius[li] + largest):
            if lj > li and np.hypot(*(xy[li] - xy[lj])) <= radius[li] + radius[lj]:
                yield int(i), int(idx[lj])


def inside(geom, i, x, y, tol=0):
    c, s = np.cos(geom.theta[i]), np.sin(geom.theta[i])
    dx, dy = x - geom.x[i], y - geom.y[i]
    return ((c * dx + s * dy) / geom.major[i])**2 + ((-s * dx + c * dy) / geom.minor[i])**2 <= 1 + tol


def boundary_fraction(geom, outer, inner, samples=72):
    t = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    c, s = np.cos(geom.theta[inner]), np.sin(geom.theta[inner])
    dx, dy = geom.major[inner] * np.cos(t), geom.minor[inner] * np.sin(t)
    x = np.r_[geom.x[inner] + c * dx - s * dy, geom.x[inner]]
    y = np.r_[geom.y[inner] + s * dx + c * dy, geom.y[inner]]
    return float(inside(geom, outer, x, y, tol=1e-3).mean())


def area_fraction(geom, outer, inner):
    """HSC containment: fraction of inner ellipse area inside outer ellipse.

    Uses the same bounded 384-pixel sampling grid as HSC, without image clipping.
    """
    from .geometry import ellipse_contains_dict
    radius = max(float(geom.major[inner]), float(geom.minor[inner]))
    outer_radius = max(float(geom.major[outer]), float(geom.minor[outer]))
    if np.hypot(geom.x[inner]-geom.x[outer], geom.y[inner]-geom.y[outer]) > radius+outer_radius:
        return 0.0
    x0, x1 = math.floor(geom.x[inner]-radius), math.ceil(geom.x[inner]+radius)
    y0, y1 = math.floor(geom.y[inner]-radius), math.ceil(geom.y[inner]+radius)
    step = max(1, int(math.ceil(max(x1-x0+1, y1-y0+1)/384.0)))
    yy, xx = np.mgrid[y0:y1+1:step, x0:x1+1:step]
    def ellipse(i):
        return dict(x=geom.x[i], y=geom.y[i], major=geom.major[i], minor=geom.minor[i],
                    theta_deg=np.degrees(geom.theta[i]))
    a = ellipse_contains_dict(ellipse(inner), xx.astype(np.float32), yy.astype(np.float32))
    n = np.count_nonzero(a)
    if not n:
        return 0.0
    b = ellipse_contains_dict(ellipse(outer), xx.astype(np.float32), yy.astype(np.float32))
    return float(np.count_nonzero(a & b)/n)


def pixel_iou(geom, i, j, shape=None):
    radius = np.maximum(geom.major, geom.minor)
    x0 = math.floor(min(geom.x[i] - radius[i], geom.x[j] - radius[j]))
    x1 = math.ceil(max(geom.x[i] + radius[i], geom.x[j] + radius[j]))
    y0 = math.floor(min(geom.y[i] - radius[i], geom.y[j] - radius[j]))
    y1 = math.ceil(max(geom.y[i] + radius[i], geom.y[j] + radius[j]))
    if shape is not None:
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(shape[1] - 1, x1), min(shape[0] - 1, y1)
    intersection = union = 0
    # Exact counts without a full giant source-pair mesh.
    for start in range(y0, y1 + 1, 64):
        yy, xx = np.mgrid[start:min(start + 64, y1 + 1), x0:x1 + 1]
        a, b = inside(geom, i, xx, yy), inside(geom, j, xx, yy)
        intersection += np.count_nonzero(a & b)
        union += np.count_nonzero(a | b)
    return float(intersection / union) if union else 0.0


def mask_sample_fraction(geom, mask):
    """Reference mask overlap: 48 angles at normalized radii 0, 0.5 and 1."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("footprint mask must be 2-D")
    fraction = np.zeros(len(geom.x), dtype=float)
    center_hit = np.zeros(len(geom.x), dtype=bool)
    rr, tt = np.meshgrid([0., .5, 1.], np.linspace(0, 2*np.pi, 48, endpoint=False), indexing='ij')
    for i in np.flatnonzero(geom.valid()):
        c, s = np.cos(geom.theta[i]), np.sin(geom.theta[i])
        dx, dy = geom.major[i]*rr.ravel()*np.cos(tt.ravel()), geom.minor[i]*rr.ravel()*np.sin(tt.ravel())
        x = np.rint(geom.x[i] + c*dx - s*dy).astype(int)
        y = np.rint(geom.y[i] + s*dx + c*dy).astype(int)
        valid = (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
        if valid.any():
            fraction[i] = mask[y[valid], x[valid]].mean()
        xi, yi = int(round(geom.x[i])), int(round(geom.y[i]))
        if 0 <= xi < mask.shape[1] and 0 <= yi < mask.shape[0]:
            center_hit[i] = mask[yi, xi]
    return fraction, center_hit
