"""HSC training tile quality: Aug-27 weighted score, including EDGE weight 0.1.

Keep SAT/BAD label protection separate from the NaN/NO_DATA coverage gate.
JWST does not use this score. Existing all-finite/all-BAD compatibility applies.
"""
import numpy as np
from data_filtering.calexp_quality import read_mask_plane_for_score, bad_score_map
from .large_sources import crop_padded

WEIGHTS={'NO_DATA':1.,'UNMASKEDNAN':1.,'INTRP':.3,'BAD':.5,'EDGE':.1}
THRESHOLD=.13


def hsc_training_tile_quality(path,image,origin,specs):
    mask,planes,ignored=read_mask_plane_for_score(path)
    if mask.shape!=image.shape:raise ValueError(f'HSC mask/image mismatch: {path}')
    valid=np.isfinite(image)
    for name in ('NO_DATA','UNMASKEDNAN'):
        if name in planes:valid &= (mask & (1<<planes[name]))==0
    score=bad_score_map(mask,planes,WEIGHTS,ignored_planes=ignored)
    keep=[];rows=[]
    for spec in specs:
        cut=crop_padded(score,spec.x0,spec.y0,origin,spec.size,fill=1.)
        fraction=float(np.mean(cut));invalid=1-float(crop_padded(valid,spec.x0,spec.y0,origin,spec.size,fill=False).mean())
        accepted=fraction<THRESHOLD and invalid<=.10
        rows.append(dict(name=spec.name,bad_score=fraction,invalid_fraction=invalid,accepted=accepted))
        if accepted:keep.append(spec)
    return valid,keep,dict(policy='0827_bad_score_edge0p1',weights=WEIGHTS,threshold=THRESHOLD,
                           invalid_threshold=.10,ignored_planes=sorted(ignored),tiles=rows)
