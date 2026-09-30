"""Read sparse, independently overlapping instance targets from training Zarr.

Use the production reader, including its truncated final-chunk convention.
Only bbox masks are retained in CPU batches; do not expand every instance to
an H x W plane until the selected prompts are evaluated.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import find_objects


def read_instance_targets(reader, sample, band_pu, image):
    bands, h, w = band_pu.shape
    valid = (
        torch.from_numpy(reader.read_first_axis('band_valid_mask', sample).astype(bool))
        if reader.has_array('band_valid_mask')
        else torch.isfinite(image).all(dim=1) if image.ndim == 4 else torch.isfinite(image)
    )
    # Both pipelines write their accepted background into dense class 4.
    # The backend controls how it was measured, not whether SAM may use it.
    # Keep unknown provenance opt-out for old stores with ambiguous labels.
    methods = ' '.join(str(reader.attrs.get(k, '')).lower() for k in (
        'background_method', 'training_background', 'hsc_background_method'))
    trusted_backend = 'sextractor' in methods or 'lsst' in methods
    trusted = (band_pu == 4) & valid if trusted_backend else torch.zeros_like(valid)
    result = [[] for _ in range(bands)]
    if not reader.has_array('band_segmentation_ids'):
        return result, valid, trusted
    ids = reader.read_first_axis('band_segmentation_ids', sample)
    weights = reader.read_first_axis('band_segmentation_weight', sample)
    for b in range(bands):
        positive = (ids[b] > 0) & (weights[b] > 0) & valid[b].numpy()
        trusted[b] &= ~torch.from_numpy(positive)
        source_ids, inverse = np.unique(np.where(positive, ids[b], 0), return_inverse=True)
        # find_objects expects zero for background and small positive integers.
        labels = inverse.reshape(h, w).astype(np.int32) + 1
        labels[~positive] = 0
        for index, box in enumerate(find_objects(labels), start=1):
            if box is None:
                continue
            sid = int(source_ids[index - 1])
            y, x = box
            mask = positive[box] & (ids[b][box] == sid)
            result[b].append(dict(source_id=sid, x0=x.start, y0=y.start,
                mask=torch.from_numpy(mask.copy()), weight=float(weights[b][box][mask].mean())))
    if reader.has_array('segmentation_overlap_meta'):
        meta = reader.read_full_small('segmentation_overlap_meta')
        offsets = reader.read_full_small('segmentation_overlap_offsets')
        data = reader.read_full_small('segmentation_overlap_data')
        overlap_weights = reader.read_full_small('segmentation_overlap_weights')
        for j in np.flatnonzero(meta[:, 0] == sample):
            _, b, sid, parent, x, y, mh, mw = map(int, meta[j])
            bits = data[int(offsets[j]):int(offsets[j + 1])]
            mask = np.unpackbits(bits, bitorder='little', count=mh*mw).reshape(mh, mw).astype(bool)
            # Clip defensively, including old stores with an unclipped bbox.
            xa, ya, xb, yb = max(0, x), max(0, y), min(w, x+mw), min(h, y+mh)
            result[b] = [r for r in result[b] if r['source_id'] != sid]
            if xa >= xb or ya >= yb:
                continue
            mask = mask[ya-y:yb-y, xa-x:xb-x].copy() & valid[b, ya:yb, xa:xb].numpy()
            if not mask.any():
                continue
            cut = torch.from_numpy(mask)
            trusted[b, ya:yb, xa:xb] &= ~cut
            result[b].append(dict(source_id=sid, parent_source_id=parent,
                x0=xa, y0=ya, mask=cut, weight=float(overlap_weights[j])))
    return result, valid, trusted


def flip_instance_targets(instances, width):
    return [[dict(row, x0=width-row['x0']-row['mask'].shape[1],
                  mask=torch.flip(row['mask'], (-1,))) for row in band] for band in instances]
