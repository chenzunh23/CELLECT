"""LSST-only subprocess: sample persisted PSF without loading science pixels."""
import json
import sys
import numpy as np
import lsst.afw.image as afw
import lsst.geom as geom
reader=afw.ExposureFitsReader(sys.argv[1]);psf=reader.readPsf();bbox=reader.readBBox();wcs=reader.readWcs()
stamps={};positions={};errors=[]
requested=(json.load(open(sys.argv[3])) if len(sys.argv)>3 else {
    f'psf_{i}_{j}':[bbox.getMinX()+fx*(bbox.getWidth()-1),bbox.getMinY()+fy*(bbox.getHeight()-1)]
    for j,fy in enumerate([.2,.5,.8]) for i,fx in enumerate([.2,.5,.8])})
for key,(x,y) in requested.items():
    point=geom.Point2D(float(x),float(y))
    try:
        a=np.array(psf.computeKernelImage(point).array,dtype=float)
        if not np.isfinite(a).all() or not np.maximum(a,0).sum()>0:raise ValueError('Invalid PSF pixels')
        stamps[key]=a;positions[key]=[point.getX(),point.getY()]
    except Exception as exc:errors.append(dict(key=key,error=str(exc)))
if not stamps and len(sys.argv)<=3:
    point=psf.getAveragePosition()
    stamps['psf_average']=np.array(psf.computeKernelImage(point).array,dtype=float)
    positions['psf_average']=[point.getX(),point.getY()]
stamps['positions_json']=json.dumps(positions);stamps['errors_json']=json.dumps(errors)
stamps['pixel_scale']=wcs.getPixelScale().asArcseconds()
np.savez_compressed(sys.argv[2],**stamps)
