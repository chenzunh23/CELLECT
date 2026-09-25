"""Frozen-label centered stamps matching the 2026-09-09 JWST diagnostics."""
import numpy as np
from .inputs import TileSpec
from ..labels import SourceClass


def large_source_specs(labels, dataset, *, origin=(0,0), size=512, selected=None):
    a,b=np.asarray(labels.geom_major),np.asarray(labels.geom_minor)
    cls=np.asarray(labels.label_classes)
    if dataset in ('abell','a2744'):
        keep=np.isin(cls,[SourceClass.CLEAN,SourceClass.WEAK_SHAPE]) & (a>100)
    elif dataset=='cosmos':
        keep=(cls==SourceClass.CLEAN) & ((a>100)|(a>150)|(np.sqrt(a*b)>100))
    else:raise ValueError(f'No large-source policy for {dataset}')
    keep &= np.isfinite(labels.geom_x)&np.isfinite(labels.geom_y)&np.isfinite(a)&np.isfinite(b)&(a>0)&(b>0)
    if selected is not None:keep &= np.asarray(selected,bool)
    specs=[]
    for i in np.flatnonzero(keep):
        x=int(np.floor(labels.geom_x[i]+origin[0]+.5))-size//2
        y=int(np.floor(labels.geom_y[i]+origin[1]+.5))-size//2
        specs.append(TileSpec(f'large_id{int(labels.source_ids[i])}_x{x}_y{y}',x,y,size,kind='large_source'))
    return specs


def crop_padded(array, x0, y0, origin, size, *, fill=0):
    """Never shift an edge source away from the requested center."""
    a=np.asarray(array);x=int(x0-origin[0]);y=int(y0-origin[1])
    out=np.full(a.shape[:-2]+(size,size),fill,dtype=a.dtype)
    xa,ya=max(0,x),max(0,y);xb,yb=min(a.shape[-1],x+size),min(a.shape[-2],y+size)
    if xa<xb and ya<yb:out[...,ya-y:yb-y,xa-x:xb-x]=a[...,ya:yb,xa:xb]
    return out


def valid_tile_specs(specs, valid, origin, max_invalid_fraction):
    if not 0<=max_invalid_fraction<=1:raise ValueError('invalid fraction must be in [0,1]')
    return [s for s in specs if
            np.count_nonzero(crop_padded(valid,s.x0,s.y0,origin,s.size,fill=False))
            >= (1-max_invalid_fraction)*s.size*s.size]
