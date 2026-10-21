"""Native-grid AUTO background, PSF-scale residual detection, PSF RMS grow.

This is an opt-in training-background recipe. It never modifies science pixels
or the background used to classify sources on the full reference coadd.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy import ndimage
from .jwst_background_sextractor import AggressiveSkyConfig, aggressive_sextractor_background
from ..dataset_inputs import effective_band

METHOD = 'sextractor_native_auto128_psf_smoothing_v2'


@dataclass(frozen=True)
class LocalSkyConfig:
    psf_rms_path: str
    psf_rms_sha256: str
    implementation_sha256: str
    background_mesh: int = 128
    background_filter_size: int = 3
    grow_factor: float = 2.4
    # fixed mode remains available for explicitly requested legacy experiments.
    smoothing_mode: str = 'psf'
    smoothing_sigma_floor_pixels: float = 1.
    factors: tuple = (1,)
    smoothing_sigma_pixels: tuple = (1.,)  # used only in fixed mode
    thresholds_sigma: tuple = (1.5,) #1.8
    minarea_native_pixels: tuple = (1,)

    def __post_init__(self):
        if self.background_mesh < 16 or self.background_filter_size < 1 or self.grow_factor <= 0:
            raise ValueError('Invalid local background configuration')
        if self.smoothing_mode not in {'psf','fixed'} or not np.isfinite(self.smoothing_sigma_floor_pixels) or self.smoothing_sigma_floor_pixels<=0:
            raise ValueError('Invalid smoothing mode or sigma floor')
        if self.smoothing_mode=='psf' and tuple(self.factors)!=(1,):
            raise ValueError('PSF smoothing uses one native-resolution pass; use fixed mode for multiscale')
        self.detection_config(psf_sigma=1.)

    def detection_config(self, *, psf_sigma=None):
        sigmas=self.smoothing_sigma_pixels
        if self.smoothing_mode=='psf':
            if psf_sigma is None or not np.isfinite(psf_sigma) or psf_sigma<=0:
                raise ValueError('A positive science-grid PSF sigma is required')
            sigmas=(max(self.smoothing_sigma_floor_pixels,float(psf_sigma)),)
        return AggressiveSkyConfig(factors=self.factors,
            smoothing_sigma_pixels=sigmas,
            thresholds_sigma=self.thresholds_sigma,
            minarea_native_pixels=self.minarea_native_pixels,
            grow_native_pixels=(0,) * len(self.factors))


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def implementation_sha256():
    h = hashlib.sha256()
    for path in (Path(__file__), Path(__file__).with_name('jwst_background_sextractor.py')):
        h.update(path.read_bytes())
    return h.hexdigest()


def validate_psf_asset(config, band):
    if file_sha256(config.psf_rms_path) != config.psf_rms_sha256:
        raise ValueError('PSF RMS asset changed after planning')
    row = json.loads(Path(config.psf_rms_path).read_text())[band]
    psf = Path(row['path']);stamp = psf.stat()
    if stamp.st_size != row['source_bytes'] or stamp.st_mtime_ns != row['source_mtime_ns']:
        raise ValueError(f'Stale PSF RMS measurement for {band}: {psf}')
    return row


def fit_native_background(image, valid, config, work, sex, log):
    """Exactly the native AUTO fit used by the 2026-10-13 comparison."""
    fits.writeto(work/'science.fits', np.where(valid,image,0).astype('f4'))
    fits.writeto(work/'weight.fits',valid.astype('f4'))
    (work/'params').write_text('NUMBER\nX_IMAGE\nY_IMAGE\n')
    opts=dict(CATALOG_NAME=str(work/'catalog.cat'),CATALOG_TYPE='ASCII_HEAD',
        PARAMETERS_NAME=str(work/'params'),DETECT_TYPE='CCD',THRESH_TYPE='ABSOLUTE',
        DETECT_THRESH=1.,ANALYSIS_THRESH=1.,DETECT_MINAREA=16,FILTER='N',
        DEBLEND_NTHRESH=1,DEBLEND_MINCONT=1,CLEAN='N',BACK_TYPE='AUTO',
        BACK_SIZE=config.background_mesh,BACK_FILTERSIZE=config.background_filter_size,
        WEIGHT_TYPE='MAP_WEIGHT',WEIGHT_IMAGE=str(work/'weight.fits'),WEIGHT_THRESH=.5,
        WEIGHT_GAIN='N',CHECKIMAGE_TYPE='BACKGROUND',CHECKIMAGE_NAME=str(work/'background.fits'),
        MEMORY_OBJSTACK=65536,MEMORY_PIXSTACK=10000000,
        MEMORY_BUFSIZE=max(1024,image.shape[1]),VERBOSE_TYPE='NORMAL')
    (work/'config.sex').write_text('\n'.join(f'{k} {v}' for k,v in opts.items())+'\n')
    with log.open('w') as stream:
        subprocess.run([sex,str(work/'science.fits'),'-c',str(work/'config.sex')],
                       check=True,stdout=stream,stderr=subprocess.STDOUT)
    return np.array(fits.getdata(work/'background.fits'),dtype='f4')


def local_sextractor_background(image, header, output, *, config, source='',
                               executable=None, scratch_dir=None):
    image=np.asarray(image,dtype='f4')
    if image.ndim != 2:raise ValueError('Expected a two-dimensional science image')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    valid=np.isfinite(image);band=effective_band(header)
    psf=validate_psf_asset(config,band)
    scale=float(np.sqrt(np.prod(proj_plane_pixel_scales(WCS(header).celestial)))*3600)
    if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid science pixel scale')
    psf_sigma=psf['lsst_adaptive_sigma_arcsec']/scale
    detection=config.detection_config(psf_sigma=psf_sigma)
    radius=int(config.grow_factor*psf_sigma+.5)
    sig=dict(method=METHOD,config=json.loads(json.dumps(asdict(config))),source=str(source),
        shape=list(image.shape),image_sha256=hashlib.sha256(np.ascontiguousarray(image).view('u1')).hexdigest(),
        band=band,pixel_scale_arcsec=scale,grow_radius_pixels=radius,psf=psf,
        psf_sigma_native_pixels=psf_sigma,resolved_detection_config=json.loads(json.dumps(asdict(detection))))
    meta=output.with_suffix('.json')
    if output.exists() and meta.exists() and json.loads(meta.read_text()).get('signature')==sig:
        with np.load(output) as z:sky=np.array(z['background_mask'],dtype=bool)
        if sky.shape==image.shape:return sky&valid
    sex=executable or shutil.which('source-extractor') or shutil.which('sex')
    if sex is None:raise FileNotFoundError('SExtractor executable not found')
    scratch=Path(scratch_dir) if scratch_dir else output.parent
    scratch.mkdir(parents=True,exist_ok=True)
    if ',' in str(scratch):raise ValueError('SExtractor scratch directory cannot contain commas')
    passes=[];model_percentiles=[];sky=valid.copy()
    if valid.any():
        with tempfile.TemporaryDirectory(prefix='local-auto-',dir=scratch) as tmp:
            work=Path(tmp);model=fit_native_background(image,valid,config,work,sex,output.with_suffix('.fit.log'))
            model_percentiles=np.percentile(model[valid],[1,50,99]).tolist()
            residual=image-model
            sky0=aggressive_sextractor_background(residual,header,work/'residual_sky.npz',
                config=detection,source=f'{source}#native-AUTO={config.background_mesh}',executable=sex)
            passes=json.loads((work/'residual_sky.json').read_text())['passes']
            shutil.copyfile(work/'residual_sky.log',output.with_suffix('.log'))
            found=valid&~sky0
            if radius:
                yy,xx=np.mgrid[-radius:radius+1,-radius:radius+1]
                found=ndimage.binary_dilation(found,structure=xx*xx+yy*yy<=radius*radius)
            sky=valid&~found
    with tempfile.NamedTemporaryFile(prefix=output.stem+'.',suffix='.tmp.npz',dir=output.parent,delete=False) as stream:
        tmp=Path(stream.name);np.savez_compressed(stream,background_mask=sky)
    try:tmp.replace(output)
    finally:tmp.unlink(missing_ok=True)
    doc=dict(signature=sig,finite_pixels=int(valid.sum()),background_pixels=int(sky.sum()),
        passes=passes,background_model_percentiles=model_percentiles,
        threshold_noise='sigma-clipped std of each smoothed residual',science_pixels_modified=False)
    meta.write_text(json.dumps(doc,indent=2)+'\n')
    return sky
