"""SExtractor segmentation-derived training sky on the training image itself."""
from dataclasses import asdict, dataclass
import json
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import numpy as np
from astropy.io import fits
from scipy import ndimage


@dataclass(frozen=True)
class SExtractorBackgroundConfig:
    detect_thresh: float = 1.5
    detect_minarea: int = 5
    back_size: int = 64
    back_filtersize: int = 3
    deblend_nthresh: int = 32
    deblend_mincont: float = .005
    grow_pixels: int = 0

    def __post_init__(self):
        if not np.isfinite(self.detect_thresh) or self.detect_thresh<=0 or self.detect_minarea<1 or self.back_size<1 or self.grow_pixels<0:
            raise ValueError('Invalid SExtractor background parameters')


def sextractor_background(image, header, output, *, config=SExtractorBackgroundConfig(), executable=None, source=''):
    """Return finite AND segmentation==0, with optional source-mask dilation.

    This is a Boolean training-label mask, not subtraction of SExtractor's
    numerical background model. Temporary science/weight/segmentation FITS
    are removed on completion; the mask, parameters and log are retained.
    """
    image=np.asarray(image,np.float32);valid=np.isfinite(image)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    signature=dict(config=asdict(config),source=str(source),shape=list(image.shape),method='sextractor_segmentation',
                   image_sha256=hashlib.sha256(image.tobytes()).hexdigest())
    meta=output.with_suffix('.json')
    if output.exists() and meta.exists() and json.loads(meta.read_text()).get('signature')==signature:
        with np.load(output) as z:return np.asarray(z['background_mask'],bool)&valid
    executable=executable or shutil.which('source-extractor') or shutil.which('sex')
    if not executable:raise FileNotFoundError('SExtractor executable not found')
    if valid.any():
        with tempfile.TemporaryDirectory(prefix='sextractor-',dir=output.parent) as temp:
            temp=Path(temp)
            fits.writeto(temp/'image.fits',np.where(valid,image,np.median(image[valid])),header,overwrite=True)
            fits.writeto(temp/'weight.fits',valid.astype(np.float32),overwrite=True)
            (temp/'params').write_text('NUMBER\nX_IMAGE\nY_IMAGE\n')
            (temp/'filter.conv').write_text('CONV NORM\n1 2 1\n2 4 2\n1 2 1\n')
            default=subprocess.run([executable,'-d'],check=True,capture_output=True,text=True).stdout
            (temp/'default.sex').write_text(default)
            options=dict(CATALOG_NAME=str(temp/'catalog.cat'),CATALOG_TYPE='ASCII_HEAD',PARAMETERS_NAME=str(temp/'params'),
                FILTER='Y',FILTER_NAME=str(temp/'filter.conv'),DETECT_THRESH=config.detect_thresh,ANALYSIS_THRESH=config.detect_thresh,
                DETECT_MINAREA=config.detect_minarea,BACK_SIZE=config.back_size,BACK_FILTERSIZE=config.back_filtersize,
                DEBLEND_NTHRESH=config.deblend_nthresh,DEBLEND_MINCONT=config.deblend_mincont,
                WEIGHT_TYPE='MAP_WEIGHT',WEIGHT_IMAGE=str(temp/'weight.fits'),WEIGHT_THRESH=.5,WEIGHT_GAIN='N',
                CHECKIMAGE_TYPE='SEGMENTATION',CHECKIMAGE_NAME=str(temp/'segmentation.fits'),VERBOSE_TYPE='NORMAL')
            command=[executable,str(temp/'image.fits'),'-c',str(temp/'default.sex')]
            for key,value in options.items():command.extend(['-'+key,str(value)])
            with output.with_suffix('.log').open('w') as log:
                subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
            detected=fits.getdata(temp/'segmentation.fits')>0
        if config.grow_pixels:detected=ndimage.binary_dilation(detected,iterations=config.grow_pixels)
        sky=valid&~detected
    else:sky=np.zeros(image.shape,bool)
    tmp=output.with_suffix('.tmp.npz');np.savez_compressed(tmp,background_mask=sky);tmp.replace(output)
    meta.write_text(json.dumps(dict(signature=signature,background_pixels=int(sky.sum()),finite_pixels=int(valid.sum())),indent=2)+'\n')
    return sky
