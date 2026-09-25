"""Bounded-memory, common-WCS tiling for surface-brightness images."""
from dataclasses import dataclass
import math
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates


@dataclass(frozen=True)
class SkyTile:
    ix: int
    iy: int
    x0: int
    y0: int
    size: int = 4096

    @property
    def name(self):
        return f'x{self.ix:+03d}_y{self.iy:+03d}'


def edge_points(x0, y0, width, height, n=33):
    t = np.linspace(0, 1, n)
    return np.vstack((np.c_[x0+t*width, np.full(n,y0)],
                      np.c_[np.full(n,x0+width), y0+t*height],
                      np.c_[x0+width-t*width, np.full(n,y0+height)],
                      np.c_[np.full(n,x0), y0+height-t*height]))


def plan_sky_tiles(input_header, reference_header, *, anchor=(8064,19102), size=4096, overlap=128):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError('require size > overlap >= 0')
    step = size-overlap
    native = WCS(input_header).celestial
    ref = WCS(reference_header).celestial
    edge = edge_points(-.5,-.5,input_header['NAXIS1'],input_header['NAXIS2'])
    xy = ref.all_world2pix(native.all_pix2world(edge,0),0)
    lo, hi = np.nanmin(xy,axis=0), np.nanmax(xy,axis=0)
    ranges = [range(math.floor((lo[k]-anchor[k]-size)/step)+1,
                    math.floor((hi[k]-anchor[k])/step)+1) for k in (0,1)]
    return [SkyTile(ix,iy,anchor[0]+ix*step,anchor[1]+iy*step,size)
            for iy in ranges[1] for ix in ranges[0]]


def tile_header(reference_header, tile, science_header=None):
    # Strip native WCS by constructing solely from the reference celestial WCS.
    w = WCS(reference_header).celestial.deepcopy()
    w.wcs.crpix -= [tile.x0,tile.y0]
    h = w.to_header(relax=True)
    if science_header is not None:
        for k in ('BUNIT','FILTER','PUPIL','TELESCOP','INSTRUME','MJD-AVG','DATE-AVG','DATE-OBS','MJD-OBS'):
            if k in science_header: h[k] = science_header[k]
    h['NAXIS']=2;h['NAXIS1']=tile.size;h['NAXIS2']=tile.size
    h['GRID_I']=tile.ix;h['GRID_J']=tile.iy
    h['GRID_X0']=tile.x0;h['GRID_Y0']=tile.y0
    return h


def reproject_tile(data, native_wcs, output_header, *, block_rows=128, categorical=False):
    """Bilinear SCI, normalizing finite support; nearest neighbour for masks.

    Returns NaN where there is no finite interpolation support. Reads only a
    bounded input rectangle and maps rows in chunks. Units remain MJy/sr.
    """
    n = int(output_header['NAXIS1']); m = int(output_header['NAXIS2'])
    out_wcs = WCS(output_header).celestial
    edge = edge_points(-.5,-.5,n,m)
    uv = native_wcs.all_world2pix(out_wcs.all_pix2world(edge,0),0)
    if not np.isfinite(uv).all(): raise ValueError('Nonfinite WCS boundary')
    x0,y0 = np.maximum(np.floor(uv.min(axis=0)).astype(int)-3,0)
    x1,y1 = np.minimum(np.ceil(uv.max(axis=0)).astype(int)+4,[data.shape[1],data.shape[0]])
    output = np.full((m,n),np.nan,np.float32)
    if x1<=x0 or y1<=y0: return output
    cut = np.array(data[y0:y1,x0:x1],dtype=np.float32,copy=True)
    valid = np.isfinite(cut)
    if not valid.any(): return output
    values = np.where(valid,cut,0).astype(np.float32)
    support = valid.astype(np.float32)
    for ya in range(0,m,block_rows):
        yb = min(ya+block_rows,m)
        xx,yy = np.meshgrid(np.arange(n),np.arange(ya,yb))
        ra,dec = out_wcs.all_pix2world(xx,yy,0)
        x,y = native_wcs.all_world2pix(ra,dec,0)
        coords = [y-y0,x-x0]
        order = 0 if categorical else 1
        weight = map_coordinates(support,coords,order=order,mode='grid-constant',cval=0,prefilter=False)
        num = map_coordinates(values,coords,order=order,mode='grid-constant',cval=0,prefilter=False)
        np.divide(num,weight,out=output[ya:yb],where=weight>1e-6)
    return output


def write_tile(path, data, header, *, source):
    """Atomic FITS publication with provenance and exact finite count."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    h=header.copy();h['ORIGFILE']=str(source);h['NVALID']=int(np.isfinite(data).sum())
    h['HISTORY']='Common F444W WCS, F250M-center anchored; finite-normalized bilinear SCI'
    temp=path.with_suffix('.fits.tmp')
    fits.PrimaryHDU(np.asarray(data,np.float32),h).writeto(temp,overwrite=True,output_verify='silentfix')
    temp.replace(path)
