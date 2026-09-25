"""Ordinary catalog center validity, evaluated before filling image NaNs.

Gaia sources are inserted later and never pass through this initial filter.
Coordinates are zero-based pixel centers; origin locates a local image within
the coordinate system of the supplied geometry.
"""
import numpy as np

from ..labels import SourceClass


def center_no_data(geometry, *, image=None, valid_mask=None, origin=(0, 0)):
    if image is None and valid_mask is None:
        return np.zeros(len(geometry.x), bool)
    valid = np.asarray(valid_mask, bool) if valid_mask is not None else np.isfinite(image)
    if valid.ndim != 2:
        raise ValueError('center validity must be a 2-D image mask')
    if image is not None:
        if np.shape(image) != valid.shape:
            raise ValueError('image/validity shape mismatch')
        valid = valid & np.isfinite(image)
    x = np.asarray(geometry.x, float) - origin[0]
    y = np.asarray(geometry.y, float) - origin[1]
    finite = np.isfinite(x) & np.isfinite(y)
    ix = np.zeros(len(x), np.int64)
    iy = np.zeros(len(y), np.int64)
    # Do not clip outside positions onto valid edge pixels or cast NaN to int.
    inside = (finite & (x >= -.5) & (x < valid.shape[1] - .5)
              & (y >= -.5) & (y < valid.shape[0] - .5))
    ix[inside] = np.floor(x[inside] + .5).astype(np.int64)
    iy[inside] = np.floor(y[inside] + .5).astype(np.int64)
    bad = np.ones(len(x), bool)
    bad[inside] = ~valid[iy[inside], ix[inside]]
    return bad


def center_ignore_mask(geometry, *, image=None, valid_mask=None,
                       center_invalid=None, origin=(0, 0)):
    bad = center_no_data(geometry, image=image, valid_mask=valid_mask, origin=origin)
    if center_invalid is not None:
        if np.shape(center_invalid) != bad.shape:
            raise ValueError('center_invalid length mismatch')
        bad |= np.asarray(center_invalid, bool)
    return bad


def apply_center_ignore(labels, mask):
    """Keep NaN-center catalog sources as ignore ellipses, not dropped holes."""
    labels.assign(np.asarray(mask, bool), SourceClass.ORDINARY_IGNORE, 'center_no_data')
