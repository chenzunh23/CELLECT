"""Asset-defined JWST EE targets, legacy FWHM targets, and HSC Manhattan mode."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ASSETS = Path(__file__).resolve().parents[1] / 'assets'
EE_CONFIG = ASSETS / 'jwst_confidence_ee10_35_60_70_v1.json'
FWHM_CONFIG = ASSETS / 'jwst_confidence_fwhm_v1.json'


def validate_confidence(mode, minimum, maximum, fwhm_pixels=None):
    if mode not in ('auto', 'manhattan', 'psf-matched', 'psf-ee'):
        raise ValueError(f'Unknown confidence mode: {mode}')
    if not (np.isfinite(minimum) and np.isfinite(maximum) and 0 < minimum <= maximum):
        raise ValueError('Confidence FWHM limits must be finite and 0 < min <= max')
    if fwhm_pixels is not None and not (np.isfinite(fwhm_pixels) and fwhm_pixels > 0):
        raise ValueError('Confidence FWHM override must be positive finite output pixels')
    if mode == 'psf-ee' and fwhm_pixels is not None:
        raise ValueError('FWHM override requires --confidence-mode psf-matched')


def confidence_asset(mode, config_path=None):
    """Read once per image, never per source or pixel; no large FITS I/O."""
    path = Path(config_path) if config_path else (EE_CONFIG if mode == 'psf-ee' else FWHM_CONFIG)
    raw = path.read_bytes()
    cfg = json.loads(raw)
    if cfg.get('schema_version') != 1:
        raise ValueError(f'Unsupported confidence asset schema: {path}')
    if mode == 'psf-ee':
        try:
            bounds = cfg['boundaries']
            fractions = np.array([b['ee_fraction'] for b in bounds], float)
            valid = ([b['level'] for b in bounds] == [4,3,2,1]
                     and np.isfinite(fractions).all() and 0 < fractions[0]
                     and fractions[-1] <= 1 and np.all(np.diff(fractions) > 0)
                     and all(b['outer_inclusive'] for b in bounds)
                     and all(b['radius_key'] == f"r{100*b['ee_fraction']:g}" for b in bounds)
                     and cfg['psf']['extension'] == 'OVERSAMP'
                     and cfg['psf']['normalization']['mode'] == 'original_unit'
                     and cfg['psf']['normalization']['denominator'] == 1.0
                     and cfg['psf']['normalization']['renormalize_by_stamp_sum'] is False
                     and cfg['rasterization']['fallback']['nearest_xy'] == 'floor(source_xy + 0.5)'
                     and cfg['rasterization']['fallback']['retain_multiple_natural_level4_pixels'] is True
                     and cfg['rasterization']['fwhm_clipping'] is None)
            if not valid: raise ValueError('Unsupported EE definition')
            for item in cfg['filters'].values():
                radii = np.array([item['radii_arcsec'][b['radius_key']] for b in bounds], float)
                if not (np.isfinite(radii).all() and radii[0] > 0 and np.all(np.diff(radii)>0)):
                    raise ValueError('EE radii must be positive, finite and strictly increasing')
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'Invalid or unsupported EE asset: {path}: {exc}') from exc
    elif cfg.get('mode') != 'psf-matched' or cfg.get('level_radius_factors') != {'3':.5,'2':.75,'1':1.25}:
        raise ValueError(f'Invalid or unsupported FWHM asset: {path}')
    return cfg, {'asset_path': str(path.resolve()), 'asset_sha256': hashlib.sha256(raw).hexdigest(),
                 'definition_id': cfg['definition_id']}


def _pixel_scale(image_wcs, image_path):
    if image_wcs is None:
        if image_path is None:
            raise ValueError('PSF confidence requires output WCS or image path')
        from ..dataset_inputs import header_info
        from astropy.wcs import WCS
        header, _, _ = header_info(image_path)
        image_wcs = WCS(header).celestial
    from astropy.wcs.utils import proj_plane_pixel_scales
    scales = np.asarray(proj_plane_pixel_scales(image_wcs.celestial), float) * 3600
    if scales.shape != (2,) or not np.all(np.isfinite(scales) & (scales > 0)):
        raise ValueError('Invalid celestial WCS pixel scale for confidence')
    if not np.isclose(scales[0], scales[1], rtol=.01):
        raise ValueError('Circular pixel PSF targets require a square-pixel output grid')
    return float(scales.mean())


def resolve_confidence(task, *, image_wcs=None, image_path=None):
    """Resolve radii once per output image; returned metadata is JSON serializable."""
    requested = task.confidence_mode
    minimum, maximum = task.confidence_fwhm_min, task.confidence_fwhm_max
    override = task.confidence_fwhm_pixels
    validate_confidence(requested, minimum, maximum, override)
    band = str(task.band).upper()
    mode = ('manhattan' if band.startswith(('HSC-', 'NB')) else 'psf-ee') if requested == 'auto' else requested
    meta = dict(mode=mode, requested_mode=requested, version='confidence_v1', positive_levels=[1,2,3,4])
    if mode == 'manhattan':
        return {**meta, 'distance':'manhattan', 'positive_support':'distance < 4', 'center_rounding':'round'}
    if mode == 'psf-ee' and band.startswith(('HSC-', 'NB')):
        path = getattr(task, 'confidence_config_path', None)
        if path is None:
            raise ValueError('HSC PSF-EE requires a measured per-patch config; use --training-batch')
        cfg = json.loads(Path(path).read_text())
        if cfg.get('definition_id') != 'hsc_coaddpsf_ee10_35_60_70_v1':
            raise ValueError('Not an HSC PSF-EE measurement')
        scale = _pixel_scale(image_wcs, image_path)
        radii = cfg['level_radii_arcsec']
        values = np.array([radii[str(k)] for k in (4,3,2,1)], float)
        if not (np.isfinite(values).all() and values[0]>0 and np.all(np.diff(values)>0)):
            raise ValueError('Invalid HSC EE radii')
        return {**cfg, **meta, 'distance':'euclidean', 'measurement_path':str(path),
                'pixel_scale_arcsec':scale, 'level_radii_pixels':{k:v/scale for k,v in radii.items()}}
    cfg, asset = confidence_asset(mode, getattr(task, 'confidence_config_path', None))
    if mode == 'psf-ee':
        if override is not None:
            raise ValueError('JWST auto now uses EE: use psf-matched for a FWHM override')
        if band not in cfg['filters']:
            raise ValueError(f'{band}: no EE PSF radii in {asset["asset_path"]}')
        pixel_scale = _pixel_scale(image_wcs, image_path)
        item = cfg['filters'][band]
        radii = {str(b['level']): item['radii_arcsec'][b['radius_key']] for b in cfg['boundaries']}
        return {**meta, **asset, 'version':cfg['definition_id'], 'distance':'euclidean',
                'pixel_scale_arcsec':pixel_scale, 'level_radii_arcsec':radii,
                'level_radii_pixels':{k:v/pixel_scale for k,v in radii.items()},
                'ee_fractions':{str(b['level']):b['ee_fraction'] for b in cfg['boundaries']},
                'normalization':cfg['psf']['normalization'], 'psf_path':item['path'],
                'psf_sha256':item['sha256'], 'psf_extension':item['extension'],
                'psf_stamp_sum':item['stamp_sum'], 'fallback':cfg['rasterization']['fallback'],
                'level4':'all pixel centers inside r10; one nearest center if empty',
                'center_rounding':'floor(center + 0.5)', 'merge':'pixelwise maximum',
                'fwhm_clip_pixels':None}
    pixel_scale = arcsec = None
    if override is not None:
        raw = float(override)
        source = 'explicit output-pixel FWHM override'
    else:
        if band not in cfg['nominal_fwhm_arcsec']:
            raise ValueError(f'{band}: no nominal FWHM; provide --confidence-fwhm-pixels')
        pixel_scale = _pixel_scale(image_wcs, image_path)
        arcsec = float(cfg['nominal_fwhm_arcsec'][band])
        raw = arcsec / pixel_scale
        source = 'nominal NIRCam FWHM asset / output WCS pixel scale'
    width = float(np.clip(raw, minimum, maximum))
    return {**meta, **asset, 'distance':'euclidean', 'center_rounding':'floor(center + 0.5)',
            'fwhm_source':source, 'nominal_fwhm_arcsec':arcsec, 'pixel_scale_arcsec':pixel_scale,
            'fwhm_pixels_raw':raw, 'fwhm_pixels_used':width, 'fwhm_clip_pixels':[minimum,maximum],
            'level_radii_pixels':{k:v*width for k,v in cfg['level_radius_factors'].items()},
            'level4':'one nearest pixel', 'merge':'pixelwise maximum'}


def paint_ee_confidence(conf, weight, centers, *, level_radii_pixels, value_weight=1.):
    """Pixel-center circular EE rings. Overlapping sources merge by maximum."""
    radii = np.array([level_radii_pixels[str(k)] for k in (4,3,2,1)], float)
    if not (np.isfinite(radii).all() and radii[0]>0 and np.all(np.diff(radii)>0)):
        raise ValueError('EE radii must be positive, finite and strictly increasing')
    if conf.ndim != 2 or weight.shape != conf.shape:
        raise ValueError('Confidence and weights must share a 2-D shape')
    h,w = conf.shape
    radius = int(math.ceil(radii[-1])) + 1
    for cx,cy in np.asarray(centers,dtype=np.float64).reshape(-1,2):
        if not (np.isfinite(cx) and np.isfinite(cy) and 0<=cx<w and 0<=cy<h):
            continue
        # At the image edge choose the closest existing center; interior ties round up.
        ix,iy = min(w-1,math.floor(cx+.5)),min(h-1,math.floor(cy+.5))
        y0,y1=max(0,iy-radius),min(h,iy+radius+1)
        x0,x1=max(0,ix-radius),min(w,ix+radius+1)
        yy,xx=np.mgrid[y0:y1,x0:x1]
        dist=np.hypot(xx-cx,yy-cy)
        values=np.zeros(dist.shape,np.uint8)
        for level,r in zip((1,2,3,4),radii[::-1]): values[dist<=r]=level
        if not np.any(values==4): values[iy-y0,ix-x0]=4
        local=conf[y0:y1,x0:x1]
        np.maximum(local,values,out=local)
        local_weight=weight[y0:y1,x0:x1];positive=values>0
        local_weight[positive]=np.maximum(local_weight[positive],value_weight)


def paint_psf_confidence(conf, weight, centers, *, fwhm_pixels,
                         minimum=1.6, maximum=8., value_weight=1.):
    """09-08 Euclidean rings and nearest-pixel rule; configurable FWHM limits.

    Invalid pixels are zero-weighted by the writer after targets are painted.
    """
    validate_confidence('psf-matched', minimum, maximum, fwhm_pixels)
    width = float(np.clip(fwhm_pixels, minimum, maximum))
    radius = int(math.ceil(1.25*width)) + 1
    if conf.ndim != 2 or weight.shape != conf.shape:
        raise ValueError('Confidence and weights must share a 2-D shape')
    h, w = conf.shape
    for cx, cy in np.asarray(centers, dtype=np.float64).reshape(-1, 2):
        cx, cy = float(cx), float(cy)
        if not np.isfinite(cx+cy):
            continue
        ix, iy = math.floor(cx+.5), math.floor(cy+.5)
        if not (0 <= ix < w and 0 <= iy < h):
            continue
        y0, y1 = max(0, iy-radius), min(h, iy+radius+1)
        x0, x1 = max(0, ix-radius), min(w, ix+radius+1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dist = np.hypot(xx.astype(np.float32)-cx, yy.astype(np.float32)-cy)
        values = np.zeros(dist.shape, np.uint8)
        values[(dist > 0) & (dist <= .5*width)] = 3
        values[(dist > .5*width) & (dist <= .75*width)] = 2
        values[(dist > .75*width) & (dist <= 1.25*width)] = 1
        values[iy-y0, ix-x0] = 4
        local = conf[y0:y1, x0:x1]
        np.maximum(local, values, out=local)
        local_weight = weight[y0:y1, x0:x1]
        positive = values > 0
        local_weight[positive] = np.maximum(local_weight[positive], value_weight)
