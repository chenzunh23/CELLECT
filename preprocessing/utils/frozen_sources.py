"""Reuse a filtered source snapshot on another exposure without running filters."""
from dataclasses import dataclass
import csv
import json
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from preprocessing.labels import SourceClass, SourceLabels
from .geometry import EllipseGeometry
from .segmentation import transform_ellipses


@dataclass
class FrozenSources:
    source_ids: np.ndarray
    geometry: EllipseGeometry
    labels: SourceLabels
    magnitude: np.ndarray
    inserted_ids: np.ndarray
    inserted_xy: np.ndarray
    metadata: dict


def load_frozen_sources(snapshot_dir, target_wcs):
    """Load preview sources.csv/inserted.csv/summary.json; preserve every class.

    Geometry is mapped from snapshot cutout coordinates through the original
    FITS WCS. target_wcs must describe the actual destination array, including
    its cutout origin. No filtering, center insertion or label promotion occurs.
    """
    root = Path(snapshot_dir)
    meta = json.loads((root/'summary.json').read_text())
    with (root/'sources.csv').open() as f:
        rows = list(csv.DictReader(f))
    ids = np.array([int(r['id']) for r in rows], dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError('duplicate frozen source IDs')
    names = {'drop':0, 'clean':1, 'weak_shape':2, 'ignore':3,
             'strict_center_only':4, 'restricted':5, 'strict_ignore':6}
    classes = np.array([names[r['final']] for r in rows], dtype=np.int16)
    values = lambda key: np.array([float(r[key]) for r in rows])
    ox, oy = meta.get('origin', (0,0))
    with fits.open(meta['raw_fits'], memmap=True) as hs:
        header = next(h.header for h in hs if h.header.get('NAXIS') == 2)
        source_wcs = WCS(header).celestial
    mapped = transform_ellipses(source_wcs, target_wcs, values('x')+ox, values('y')+oy,
                                values('a'), values('b'), np.deg2rad(values('theta_deg')))
    x,y,a,b,theta = mapped
    geom = EllipseGeometry(x,y,a,b,theta,np.pi*a*b)
    if np.any(~geom.valid() & (classes != SourceClass.DROPPED)):
        raise ValueError('invalid mapped geometry for a non-dropped source')
    with (root/'inserted.csv').open() as f:
        inserted = list(csv.DictReader(f))
    ix=np.array([float(r['x'])+ox for r in inserted]);iy=np.array([float(r['y'])+oy for r in inserted])
    ra,dec=source_wcs.all_pix2world(ix,iy,0)
    tx,ty=target_wcs.all_world2pix(ra,dec,0)
    return FrozenSources(ids,geom,SourceLabels(classes,np.array([r['reason'] for r in rows],object)),
        values('mag'),np.array([int(r['id']) for r in inserted],np.int64),np.column_stack((tx,ty)),
        dict(snapshot=str(root),source_fits=meta['raw_fits'],source_origin=[ox,oy]))


def prepare_frozen_variant(snapshot_dir, image_fits, background_npz, *, origin=(0,0),
                           shape=(4096,4096), bright_config=None):
    """Prepare new image/bright/LSST-background arrays, retaining frozen sources.

    The background NPZ must have a sibling summary.json from the LSST batch
    tool identifying this exact FITS. Source/SNR/bright-label filtering is NEVER
    rerun here; returned components are image artifacts, not new source labels.
    """
    from preprocessing.image_processing import prepare_image,build_bright_components,BrightRegionConfig
    image_fits=Path(image_fits);background_npz=Path(background_npz)
    bgmeta=json.loads((background_npz.parent/'summary.json').read_text())
    if bgmeta['status']!='ok' or Path(bgmeta['input']).resolve()!=image_fits.resolve():
        raise ValueError('background must be successfully generated for the new FITS')
    if bgmeta['origin_xy'] != [0,0]:
        raise ValueError('background must use full-image coordinates')
    x0,y0=map(int,origin);h,w=map(int,shape)
    with fits.open(image_fits,memmap=True) as hs:
        image=next(hh for hh in hs if hh.header.get('NAXIS')==2)
        if min(x0,y0)<0 or min(h,w)<=0 or y0+h>image.shape[0] or x0+w>image.shape[1]:
            raise ValueError('requested cutout is outside the new image')
        if list(image.shape)!=bgmeta['shape_yx']:
            raise ValueError('background metadata/image shape mismatch')
        raw=np.array(image.section[y0:y0+h,x0:x0+w],dtype=np.float32)
        header=image.header.copy()
        target_wcs=WCS(header).celestial.slice((slice(y0,y0+h),slice(x0,x0+w)))
        header.update(target_wcs.to_header())
    with np.load(background_npz) as z:
        bg=z['background_mask']
        if list(bg.shape)!=bgmeta['shape_yx']:
            raise ValueError('background array shape mismatch')
        sky=bg[y0:y0+h,x0:x0+w].astype(bool)
    sources=load_frozen_sources(snapshot_dir,target_wcs)
    prepared=prepare_image(raw,header=header)
    config=bright_config or BrightRegionConfig(threshold=5,clip_threshold=5,statistics_clip_sigma=5)
    bright,components=build_bright_components(prepared,config=config)
    return dict(sources=sources,prepared=prepared,image_header=header,
                lsst_background_mask=sky & prepared.finite_mask,
                bright_region_mask=bright,component_labels=components,
                metadata=dict(image_fits=str(image_fits),background_npz=str(background_npz),
                              origin=[x0,y0],source_filters='frozen',new_centers_inserted=0))
