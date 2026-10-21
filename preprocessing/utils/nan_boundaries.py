"""Original-FITS connectivity for NaN repair, independent of crop boundaries.

Label in bounded tiles and join their borders; never load a full float mosaic
or full-size integer label image. Persist only a packed edge-connected mask.
"""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy import ndimage

POLICY = 'original-fits-edge-4connected-v1'


@lru_cache(maxsize=128)
def _sampled_background(path, size, mtime_ns):
    """Stable source-hole threshold reference, independent of crop extent.

    Sample at most 512x512 original SCI pixels; no filled/no-data pixels enter
    these statistics. Cache by original file identity, not by view window.
    """
    from astropy.stats import sigma_clipped_stats
    from ..dataset_inputs import image_hdu
    with fits.open(path, memmap=False) as hd:
        idx = image_hdu(hd)
        ny, nx = hd[idx].shape
        sy, sx = max(1, (ny+511)//512), max(1, (nx+511)//512)
        sample = np.asarray(hd[idx].section[::sy, ::sx], dtype=np.float64)
    values = sample[np.isfinite(sample)]
    if not values.size:
        return None
    _, median, sigma = sigma_clipped_stats(values, sigma=3., maxiters=None)
    median = float(median) if np.isfinite(median) else float(np.median(values))
    sigma = float(sigma) if np.isfinite(sigma) and sigma > 0 else float(np.std(values))
    return median, sigma if np.isfinite(sigma) and sigma > 0 else 1.


def source_nan_background(source_path):
    source = Path(source_path).expanduser().resolve()
    stamp = source.stat()
    return _sampled_background(str(source), stamp.st_size, stamp.st_mtime_ns)


def edge_connected_nonfinite(image):
    bad = ~np.isfinite(image)
    seed = np.zeros(bad.shape, bool)
    seed[0] = bad[0]; seed[-1] = bad[-1]
    seed[:, 0] = bad[:, 0]; seed[:, -1] = bad[:, -1]
    return ndimage.binary_propagation(seed, mask=bad)


def _packed_edges(data, work, tile_size=1024):
    ny, nx = data.shape
    if tile_size % 8:
        raise ValueError('tile_size must be a multiple of eight')
    parent = [0]; outer = [False]; records = []; previous = {}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    def join(a, b):
        pairs = np.unique(np.c_[a, b], axis=0)
        for a, b in pairs:
            if not a or not b: continue
            ra, rb = root(int(a)), root(int(b))
            if ra != rb:
                parent[rb] = ra; outer[ra] |= outer[rb]

    for y in range(0, ny, tile_size):
        left = None; following = {}
        for x in range(0, nx, tile_size):
            bad = ~np.isfinite(data[y:min(y+tile_size, ny), x:min(x+tile_size, nx)])
            labels, n = ndimage.label(bad)
            ids = np.unique(np.r_[labels[0], labels[-1], labels[:,0], labels[:,-1]])
            ids = ids[ids != 0]
            lookup = np.zeros(n+1, np.uint32)
            for i in ids:
                lookup[i] = len(parent); parent.append(len(parent)); outer.append(False)
            mapped = lookup[labels]
            if left is not None: join(left, mapped[:,0])
            if x in previous: join(previous[x], mapped[0])
            for border, active in ((mapped[0], y == 0), (mapped[-1], y+tile_size >= ny),
                                   (mapped[:,0], x == 0), (mapped[:,-1], x+tile_size >= nx)):
                if active:
                    for i in np.unique(border):
                        if i: outer[root(int(i))] = True
            left = mapped[:,-1].copy(); following[x] = mapped[-1].copy()
            if ids.size:
                path = Path(work)/f'{x}_{y}.npz'
                np.savez_compressed(path, labels=mapped)
                records.append((x, y, path))
        previous = following
    selected = np.array([outer[root(i)] if i else False for i in range(len(parent))])
    packed = np.zeros((ny, (nx+7)//8), np.uint8)
    for x, y, path in records:
        with np.load(path) as doc: mask = selected[doc['labels']]
        bits = np.packbits(mask, axis=1)
        packed[y:y+mask.shape[0], x//8:x//8+bits.shape[1]] = bits
    return packed


@lru_cache(maxsize=2)
def _source_edges(path, size, mtime_ns):
    from ..dataset_inputs import image_hdu
    key = hashlib.sha256(json.dumps([POLICY,path,size,mtime_ns]).encode()).hexdigest()
    cache = Path(os.environ.get('CELLECT_NAN_CACHE', Path.home()/'.cache/cellect/nan_boundaries'))
    cache.mkdir(parents=True, exist_ok=True)
    output = cache/f'{key}.npz'
    # Several simultaneous frames can request the same source at startup.
    import fcntl
    with (cache/f'{key}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with fits.open(path, memmap=False) as hd:
            idx = image_hdu(hd)
            header = hd[0].header.copy(); header.update(hd[idx].header)
            shape = hd[idx].shape
            if output.exists():
                with np.load(output) as doc: packed = doc['packed']
            else:
                class Sections:
                    def __init__(self): self.shape = shape
                    def __getitem__(self, sl): return hd[idx].section[sl]
                with tempfile.TemporaryDirectory(prefix='nan-edges-',dir=cache) as tmp:
                    packed = _packed_edges(Sections(), tmp)
                with tempfile.NamedTemporaryFile(dir=cache,suffix='.npz',delete=False) as stream:
                    tmp = Path(stream.name); np.savez_compressed(stream,packed=packed)
                os.replace(tmp, output)
    packed.flags.writeable = False
    return packed, shape, header


def original_fits_edge_mask(image, header, source_path):
    """Return forbidden-to-fill pixels, including padding outside source SCI.

    WCS maps only nonfinite pixels. Four bilinear-support pixels are inspected
    for reprojected parents; interior holes cut by a crop remain eligible.
    """
    source = Path(source_path).expanduser().resolve()
    stamp = source.stat()
    packed, (ny,nx), source_header = _source_edges(str(source),stamp.st_size,stamp.st_mtime_ns)
    native = WCS(source_header).celestial; target = WCS(header).celestial
    if not native.has_celestial or not target.has_celestial:
        raise ValueError('Original FITS NaN connectivity requires celestial WCS')
    out = np.zeros(image.shape,bool)
    ys,xs = np.nonzero(~np.isfinite(image))
    for start in range(0,len(xs),65536):
        x,y = xs[start:start+65536],ys[start:start+65536]
        u,v = native.world_to_pixel_values(*target.pixel_to_world_values(x,y))
        finite = np.isfinite(u)&np.isfinite(v)
        outside = ~finite | (u < -.5)|(u > nx-.5)|(v < -.5)|(v > ny-.5)
        # Round tiny WCS floating errors before choosing bilinear support.
        u=np.where(finite,np.round(u,6),-1);v=np.where(finite,np.round(v,6),-1)
        ix=np.floor(u).astype(int);iy=np.floor(v).astype(int)
        hit=outside.copy()
        for dx,dy in ((0,0),(1,0),(0,1),(1,1)):
            xx,yy=ix+dx,iy+dy
            inside=(xx>=0)&(xx<nx)&(yy>=0)&(yy<ny)
            # A zero-weight neighbour is not part of the interpolation.
            weight=(1-(u-ix) if dx==0 else u-ix)*(1-(v-iy) if dy==0 else v-iy)
            use=inside&(weight>1e-8)
            hit[use] |= (packed[yy[use],xx[use]//8] & (128 >> (xx[use]%8))) != 0
        out[y,x]=hit
    return out
