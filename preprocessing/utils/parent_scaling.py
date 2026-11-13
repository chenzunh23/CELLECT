"""Fit RGB statistics once on a parent and apply unchanged to centered stamps."""
import numpy as np
from data_filtering.sam_input_scaling import (current_sam_zscore, log_single, lupton_single,
    unclipped_zscore_input, SCALING_VERSION, scaling_contract)


def fit_parent_rgb(image, *, clip=5., log_a=1000., high_percentile=99.5, q=20., stretch=.5):
    _,z=current_sam_zscore(image,clip_sigma=3.,z_clip=(-clip,clip))
    transform_input=unclipped_zscore_input(image,z)
    log,lp=log_single(transform_input,minimum=float(np.nanmin(transform_input)),high_pct=high_percentile,a=log_a)
    lup,_=lupton_single(transform_input,minimum=0.,stretch=stretch,q=q)
    valid=np.isfinite(image)
    def stats(a):
        v=a[valid & np.isfinite(a)];return [float(np.mean(v)),float(np.std(v)) or 1.]
    return dict(z=z,log=lp,log_stats=stats(log),lupton_stats=stats(lup),q=q,stretch=stretch,clip=clip,
                scaling_version=SCALING_VERSION,scaling=scaling_contract(scope='parent',q=q,stretch=stretch,
                    log_a=log_a,clip=clip,high_percentile=high_percentile))


def apply_parent_rgb(image, params):
    p=params;z=p['z'];lp=p['log']
    if p.get('scaling_version') != SCALING_VERSION:
        raise ValueError('Parent scaling definition changed; refit raw-first/cached parameters with fit_parent_rgb')
    valid=np.isfinite(image)
    image=np.nan_to_num(image,nan=z['zscore_median'],posinf=z['zscore_median'],neginf=z['zscore_median'])
    zz=(np.minimum(image,z['clip_hi'])-z['zscore_median'])/z['std']
    transform_input=unclipped_zscore_input(image,z)
    log=np.log1p(lp['a']*np.clip((transform_input-lp['minimum'])/(lp['hi']-lp['minimum']),0,1))/np.log(lp['a'])
    log=(log-p['log_stats'][0])/p['log_stats'][1]
    lup,_=lupton_single(transform_input,minimum=0.,stretch=p['stretch'],q=p['q'])
    lup=(lup-p['lupton_stats'][0])/p['lupton_stats'][1]
    result=np.clip(np.stack((zz,log,lup)), -p['clip'],p['clip']).astype(np.float32)
    result[:,~valid]=0
    return result
