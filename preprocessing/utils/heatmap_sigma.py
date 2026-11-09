"""Equivalent Gaussian sigma from existing PSF EE calibration, in native pixels."""
from functools import lru_cache
import json
import math
from pathlib import Path
import numpy as np


@lru_cache(maxsize=2048)
def resolve_sigma(metadata_json, band, root='', minimum=2/3, maximum=8/3):
    meta=json.loads(metadata_json)
    config=meta.get('confidence_config') or {}
    if 'level_radii_pixels' in config and config.get('mode','psf-ee')=='psf-ee':
        radii=config['level_radii_pixels']
        fractions=config.get('ee_fractions',{'4':.1,'3':.35,'2':.6,'1':.7})
        values=[radii[str(k)]/math.sqrt(-2*math.log1p(-float(fractions[str(k)]))) for k in (4,3,2,1)]
    elif band.startswith(('HSC-','NB')):
        path=Path(root)/band/(str(meta['patch'])+'.json')
        if not root or not path.is_file():
            raise ValueError(f'PSF-EE heatmap needs HSC calibration {path}; supply --heatmap-psf-ee-root (no nominal fallback)')
        saved=json.loads(path.read_text())
        return resolve_sigma(json.dumps({'confidence_config':saved}),band,'',minimum,maximum)
    else:
        from .confidence import confidence_asset,_pixel_scale
        config,_=confidence_asset('psf-ee')
        scale=(meta.get('confidence_config') or {}).get('pixel_scale_arcsec')
        if scale is None:scale=_pixel_scale(None,meta.get('image_fits'))
        item=config['filters'][band.upper()]['radii_arcsec']
        values=[item[f'r{percent}']/scale/math.sqrt(-2*math.log1p(-percent/100)) for percent in (10,35,60,70)]
    if not np.isfinite(values).all() or min(values)<=0:raise ValueError('Invalid PSF EE calibration')
    # A non-Gaussian PSF gives four different equivalent sigmas; robust median
    # avoids treating the long diffraction wings as a second-moment width.
    return float(np.clip(np.median(values),minimum,maximum))
