"""Fit RGB statistics once on a parent and apply unchanged to centered stamps."""
import numpy as np
from data_filtering.sam_input_scaling import current_sam_zscore, log_single, lupton_single


def fit_parent_rgb(image, *, clip=5., log_a=1000., high_percentile=99.5, q=20., stretch=.5):
    _,z=current_sam_zscore(image,clip_sigma=3.,z_clip=(-clip,clip))
    log,lp=log_single(image,minimum=z['raw_min'],high_pct=high_percentile,a=log_a)
    lup,_=lupton_single(image,minimum=z['zscore_median'],stretch=stretch,q=q)
    def stats(a):
        v=a[np.isfinite(a)];return [float(np.mean(v)),float(np.std(v)) or 1.]
    return dict(z=z,log=lp,log_stats=stats(log),lupton_stats=stats(lup),q=q,stretch=stretch,clip=clip)


def apply_parent_rgb(image, params):
    p=params;z=p['z'];lp=p['log']
    image=np.nan_to_num(image,nan=z['zscore_median'],posinf=z['zscore_median'],neginf=z['zscore_median'])
    zz=(np.minimum(image,z['clip_hi'])-z['zscore_median'])/z['std']
    log=np.log1p(lp['a']*np.clip((image-lp['minimum'])/(lp['hi']-lp['minimum']),0,1))/np.log(lp['a'])
    log=(log-p['log_stats'][0])/p['log_stats'][1]
    lup,_=lupton_single(image,minimum=z['zscore_median'],stretch=p['stretch'],q=p['q'])
    lup=(lup-p['lupton_stats'][0])/p['lupton_stats'][1]
    return np.clip(np.stack((zz,log,lup)), -p['clip'],p['clip']).astype(np.float32)
