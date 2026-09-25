"""LSST static detection for a bounded JWST reference cutout.

Run as a script in the LSST environment. Parameters match the existing JWST
background-mask batch: threshold=3, minPixels=15, nSigmaToGrow=1; no preliminary
bright detection or temporary background subtraction.
"""
import argparse
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

# This utility also runs directly in the separate LSST environment.
if __package__:
    from .psf import PSF_FWHM_ARCSEC
else:
    from psf import PSF_FWHM_ARCSEC


def detect(path, output, band):
    import lsst.afw.image as afwImage
    import lsst.afw.table as afwTable
    import lsst.meas.algorithms as measAlg
    from lsst.pipe.tasks.multiBand import DetectCoaddSourcesTask
    raw,h=fits.getdata(path,header=True);valid=np.isfinite(raw)
    if not valid.any():
        np.savez_compressed(output,background_mask=np.zeros(raw.shape,bool));return
    values=raw[valid];sample=values[::max(1,len(values)//1000000)]
    median=float(np.median(sample));sigma=1.4826*float(np.median(np.abs(sample-median)))
    if not np.isfinite(sigma) or sigma<=0:sigma=float(np.std(sample)) or 1.
    ny,nx=raw.shape;exposure=afwImage.ExposureF(nx,ny)
    exposure.image.array[:]=np.where(valid,raw,median)
    exposure.variance.array[:]=sigma*sigma
    exposure.mask.array[~valid] |= exposure.mask.getPlaneBitMask(['BAD','NO_DATA'])
    scale=float(np.mean(proj_plane_pixel_scales(WCS(h).celestial))*3600)
    psigma=PSF_FWHM_ARCSEC[band.upper()]/scale/2.354820045
    n=max(7,2*int(np.ceil(psigma*4))+1)
    exposure.setPsf(measAlg.SingleGaussianPsf(n,n,psigma))
    config=DetectCoaddSourcesTask.ConfigClass();config.detection.retarget(measAlg.SourceDetectionTask)
    for key,value in dict(minPixels=15,thresholdValue=3.,nSigmaToGrow=1.,
                          doBrightPrelimDetection=False,doTempLocalBackground=False,doTempWideBackground=False).items():
        if hasattr(config.detection,key):setattr(config.detection,key,value)
    task=DetectCoaddSourcesTask(config=config);run=task.detection.run
    def run_sigma(table,exposure,*args,**kwargs):
        kwargs.setdefault('sigma',psigma);return run(table,exposure,*args,**kwargs)
    task.detection.run=run_sigma
    result=task.run(exposure=exposure,idFactory=afwTable.IdFactory.makeSimple(),expId=0)
    background=valid.copy()
    for row in result.outputSources:
        for span in row.getFootprint().getSpans():
            y=span.getY();x0=max(0,span.getX0());x1=min(nx,span.getX1()+1)
            if 0<=y<ny:background[y,x0:x1]=False
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    tmp=output.with_suffix('.tmp.npz');np.savez_compressed(tmp,background_mask=background);tmp.replace(output)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('output');p.add_argument('band')
    a=p.parse_args();detect(a.input,a.output,a.band)
