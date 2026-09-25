"""Sparse sidecars for overlapping masks; ordinary instances stay in the ID plane.

Rows: sample, band, source_id, parent_source_id (-1 if absent), x0, y0, h, w.
Coordinates are zero-based within a training tile. Packed bits are row-major,
little-endian, independently padded to a byte boundary for each instance.
"""
import numpy as np

MASK_COLUMNS = ['sample', 'band', 'source_id', 'parent_source_id', 'x0', 'y0', 'height', 'width']
MASK_ENCODING = 'bbox_packbits_little_v1'


def crop_overlap_masks(masks, x0, y0, size, valid=None):
    """Crop only the sparse parent sidecars; input coordinates are parent-local."""
    out = []
    for row in masks:
        mask = row['mask']; y, x = row['y0'], row['x0']
        xa, ya = max(x, x0), max(y, y0)
        xb, yb = min(x+mask.shape[1], x0+size), min(y+mask.shape[0], y0+size)
        if xa >= xb or ya >= yb:
            continue
        cut = mask[ya-y:yb-y, xa-x:xb-x].copy()
        if valid is not None:
            cut &= valid[ya-y0:yb-y0, xa-x0:xb-x0]
        if not cut.any():
            continue
        out.append(dict(row, x0=xa-x0, y0=ya-y0, mask=cut))
    return out


def pack_overlap_masks(rows):
    """Pack rows that already contain sample/band indices. Empty is valid."""
    meta, data, weights, offsets = [], [], [], [0]
    for row in rows:
        mask = np.asarray(row['mask'], bool)
        if mask.ndim != 2 or not mask.any():
            raise ValueError('sidecar mask must be a nonempty 2D mask')
        h, w = mask.shape
        meta.append([row['sample'], row.get('band', 0), row['source_id'],
                     -1 if row.get('parent_source_id') is None else row['parent_source_id'],
                     row['x0'], row['y0'], h, w])
        bits = np.packbits(mask.ravel(), bitorder='little')
        data.append(bits); offsets.append(offsets[-1]+len(bits))
        weights.append(row['weight'])
    return dict(meta=np.asarray(meta, np.int64).reshape(-1, 8),
                data=np.concatenate(data) if data else np.empty(0, np.uint8),
                offsets=np.asarray(offsets, np.int64), weights=np.asarray(weights, np.float32))


def validate_overlap_masks(packed, n, bands, h, w):
    """Validate before creating/overwriting the Zarr store."""
    if set(packed) != {'meta', 'data', 'offsets', 'weights'}:
        raise ValueError('invalid overlapping segmentation sidecar keys')
    meta, data, offsets, weights = (np.asarray(packed[k]) for k in ('meta','data','offsets','weights'))
    if meta.ndim != 2 or meta.shape[1] != 8 or not np.issubdtype(meta.dtype, np.integer):
        raise ValueError('invalid sidecar metadata')
    m = len(meta)
    if (data.ndim != 1 or data.dtype != np.uint8 or offsets.shape != (m+1,)
            or not np.issubdtype(offsets.dtype, np.integer) or weights.shape != (m,)):
        raise ValueError('invalid sidecar arrays')
    if offsets[0] != 0 or offsets[-1] != len(data) or np.any(np.diff(offsets) < 0):
        raise ValueError('invalid sidecar offsets')
    if not np.all(np.isfinite(weights) & (weights > 0) & (weights <= 1)):
        raise ValueError('invalid sidecar weights')
    if not m:
        return
    sample, band, sid, parent, x, y, mh, mw = meta.T
    if (np.any((sample < 0) | (sample >= n) | (band < 0) | (band >= bands))
            or np.any((sid <= 0) | (parent < -1) | (parent == 0) | (sid == parent))
            or np.any((x < 0) | (y < 0) | (mh <= 0) | (mw <= 0) | (x+mw > w) | (y+mh > h))
            or not np.array_equal(np.diff(offsets), (mh*mw+7)//8)
            or len(np.unique(meta[:,:3], axis=0)) != m):
        raise ValueError('invalid sidecar instance or bounding box')


def decode_overlap_mask(packed, index):
    """Return one bbox mask from arrays (also works with Zarr array handles)."""
    row = np.asarray(packed['meta'][index], np.int64)
    start, stop = np.asarray(packed['offsets'][index:index+2], np.int64)
    mask = np.unpackbits(np.asarray(packed['data'][start:stop]), bitorder='little',
                         count=int(row[6]*row[7])).reshape(int(row[6]), int(row[7])).astype(bool)
    return row, mask


def iter_training_segmentation_masks(group, sample, band=0):
    """Yield independent full-tile masks, overriding only overlapping parents.

    Positive-only supervision is unchanged: pixels outside each mask are unknown.
    Consumers must opt into this helper (or implement the documented schema).
    """
    ids = np.asarray(group['band_segmentation_ids'][sample, band])
    weights = np.asarray(group['band_segmentation_weight'][sample, band])
    overrides = {}
    if 'segmentation_overlap_meta' in group:
        packed = {k: group['segmentation_overlap_'+k] for k in ('meta','data','offsets','weights')}
        meta = np.asarray(packed['meta'][:])
        for idx in np.flatnonzero((meta[:,0] == sample) & (meta[:,1] == band)):
            row, cut = decode_overlap_mask(packed, int(idx))
            mask = np.zeros(ids.shape, bool)
            x, y, h, w = map(int, row[4:])
            mask[y:y+h, x:x+w] = cut
            if 'band_valid_mask' in group:
                mask &= np.asarray(group['band_valid_mask'][sample, band], bool)
            overrides[int(row[2])] = (mask, float(packed['weights'][idx]))
    source_ids = (set(map(int, np.unique(ids))) - {0}) | set(overrides)
    for sid in sorted(source_ids):
        if sid in overrides:
            mask, weight = overrides[sid]
            weight_map = np.where(mask, weight, 0).astype(np.float32)
        else:
            mask = (ids == sid) & (weights > 0)
            weight_map = np.where(mask, weights, 0).astype(np.float32)
        if mask.any():
            yield dict(source_id=sid, mask=mask, positive_weight=weight_map)
