"""Shared image discovery/loading for HSC, COSMOS and A2744.

Training pixels and catalog-selection reference pixels are separate inputs.
No catalog classification or image resampling is performed by this module.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
import json
from pathlib import Path
import re

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

DEFAULT_PATHS = Path(__file__).with_name('dataset_paths.json')


def load_paths(path=None):
    """Partial JSON overrides; relative paths are relative to the JSON file."""
    config = json.loads(DEFAULT_PATHS.read_text())
    if path is not None:
        path = Path(path).expanduser().resolve()
        overrides = json.loads(path.read_text())
        for dataset, values in overrides.items():
            if dataset not in config or not isinstance(values, dict):
                raise ValueError(f'Unknown/invalid dataset configuration: {dataset}')
            unknown = set(values) - set(config[dataset])
            if unknown:
                raise ValueError(f'Unknown {dataset} path keys: {sorted(unknown)}')
            for key, value in values.items():
                if value is not None:
                    p = Path(value).expanduser()
                    value = str(p if p.is_absolute() else path.parent / p)
                config[dataset][key] = value
    return config


@dataclass(frozen=True)
class ImageInput:
    dataset: str
    sample_name: str
    band: str
    patch: str
    image_fits: Path
    reference_fits: Path
    dataset_source: str = 'coadd'
    proposal: int | None = None
    quality_mask_npz: Path | None = None
    catalog_root: Path | None = None
    split_group: str = ''

    def to_dict(self):
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}


def image_hdu(hdus, preferred=1):
    """Honor explicit HDUs; default SCI/1 falls back to a 2-D primary image."""
    if isinstance(preferred, str) and preferred.upper() != 'SCI':
        h = hdus[preferred]
        if h.header.get('NAXIS') != 2:
            raise ValueError(f'{preferred} is not a 2-D image')
        return hdus.index_of(preferred)
    if preferred in (1, 'SCI', None) and 'SCI' in hdus:
        return hdus.index_of('SCI')
    if isinstance(preferred, int) and 0 <= preferred < len(hdus):
        if hdus[preferred].header.get('NAXIS') == 2:
            return preferred
    for i, h in enumerate(hdus):
        if h.header.get('NAXIS') == 2 and isinstance(h, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)):
            return i
    raise ValueError('No 2-D science image HDU')


def header_info(path):
    with fits.open(path, memmap=True) as h:
        idx = image_hdu(h)
        header = h[0].header.copy(); header.update(h[idx].header)
        return header, tuple(h[idx].shape), idx


def effective_band(header):
    pupil = str(header.get('PUPIL', '')).upper()
    return pupil if re.fullmatch(r'F\d{3}[WMN]', pupil) else str(header.get('FILTER', '')).upper()


def discover_cosmos(config, *, proposals=None, bands=None, pointings=None):
    path = Path(config['manifest']).expanduser()
    doc = json.loads(path.read_text())
    selected = []; seen = set()
    for row in doc['items']:
        prop = int(row['proposal']); band = row['band'].upper(); point = int(row['pointing'])
        if proposals and prop not in proposals: continue
        if bands and band not in {b.upper() for b in bands}: continue
        if pointings and point not in pointings: continue
        key = (prop, point, band)
        if key in seen: raise ValueError(f'Duplicate COSMOS input: {key}')
        seen.add(key)
        def resolve(value):
            p = Path(value).expanduser()
            return p if p.is_absolute() else path.parent / p
        image = resolve(row['image_fits'])
        quality = resolve(row['quality_mask_npz'])
        if not image.is_file() or not quality.is_file():
            raise FileNotFoundError(f'Missing COSMOS image/quality mask: {image}, {quality}')
        selected.append(ImageInput('cosmos', row['sample_name'], band, f'{point:04d}', image,
            image, proposal=prop, quality_mask_npz=quality,
            catalog_root=Path(config['catalog_root']), split_group=f'COSMOS_Pointing_{point:04d}'))
    return selected


def discover_abell(config, *, bands=None):
    def index(root):
        result = {}
        for p in sorted(Path(root).glob('*_i2d_mbkg.fits')):
            header, _, _ = header_info(p); band = effective_band(header)
            if not re.fullmatch(r'F\d{3}[WMN]', band):
                raise ValueError(f'Unrecognized filter/pupil in {p}')
            if band in result: raise ValueError(f'Ambiguous Abell band {band}: {p}')
            result[band] = p
        if not result: raise FileNotFoundError(f'No Abell coadds in {root}')
        return result
    training = index(config['image_root']); reference = index(config['reference_root'])
    selected = []
    for band, path in sorted(training.items()):
        if bands and band not in {b.upper() for b in bands}: continue
        if band not in reference: raise FileNotFoundError(f'No full-coadd reference for Abell {band}')
        if path.resolve() == reference[band].resolve():
            raise ValueError('Abell half-coadd and full-coadd reference must be different files')
        selected.append(ImageInput('abell', f'a2744_half_{band.lower()}', band,
            'field_ra3p573_dec-30p376', path, reference[band], dataset_source='half_coadd',
            catalog_root=Path(config['catalog_root']), split_group='Abell2744'))
    return selected


def hsc_input(task):
    """Keep existing HSC official/half/variant path resolution unchanged."""
    from .utils.image_level import _coadd_image_path, _variant_image_path
    reference = _coadd_image_path(task.coadd_fits_root, task.band, task.tract, task.patch)
    image = reference if task.dataset_source == 'coadd' else _variant_image_path(
        task.denoised_fits_root, task.patch, task.group, task.band, task.dataset_source, task.tract)
    return ImageInput('hsc', f'hsc_{task.tract}_{task.patch}_{task.band}_{task.dataset_source}_{task.group}',
        task.band, task.patch, image, reference, dataset_source=task.dataset_source,
        catalog_root=task.data_root, split_group=f'HSC_{task.tract}_{task.patch}')


@dataclass
class LoadedImage:
    image: np.ndarray
    bad: np.ndarray
    header: fits.Header
    full_header: fits.Header
    origin: tuple[int, int]
    path: Path


def load_image(source: ImageInput, *, role='training', origin=(0, 0), shape=None):
    """Read raw linear pixels plus validity. origin=(x,y), shape=(height,width).

    A reference read uses its own grid and never applies the training-grid mask.
    The returned header describes the cutout; full_header retains the full WCS.
    """
    if role not in {'training', 'reference'}: raise ValueError('role must be training or reference')
    path = source.image_fits if role == 'training' else source.reference_fits
    x, y = map(int, origin)
    with fits.open(path, memmap=False) as hd:
        idx = image_hdu(hd); ny, nx = hd[idx].shape
        height, width = (ny-y, nx-x) if shape is None else tuple(map(int, shape))
        if x < 0 or y < 0 or min(height, width) <= 0 or x+width > nx or y+height > ny:
            raise ValueError(f'Cutout outside {path}: origin={origin}, shape={shape}, full={(ny,nx)}')
        sl = (slice(y,y+height), slice(x,x+width))
        data = np.array(hd[idx].section[sl], dtype=np.float32, copy=True)
        bad = ~np.isfinite(data)
        full_header = hd[0].header.copy(); full_header.update(hd[idx].header)
        for name in ['WHT', 'TRAIN_BAD', 'DQ', 'MASK']:
            if name not in hd: continue
            if hd[name].shape != (ny,nx): raise ValueError(f'{name} shape differs from SCI')
            a = hd[name].section[sl]
            if name == 'WHT': bad |= ~np.isfinite(a) | (a <= 0)
            elif name == 'TRAIN_BAD': bad |= a != 0
            elif name == 'DQ': bad |= (a.astype(np.uint64) & 1) != 0
            else:
                from .utils.image_level import _mask_plane_bits
                bits = _mask_plane_bits(hd[name].header)
                for plane in ('SAT','BAD','EDGE','NO_DATA','UNMASKEDNAN'):
                    if plane in bits: bad |= (a.astype(np.uint64) & (1 << bits[plane])) != 0
    if role == 'training' and source.quality_mask_npz is not None:
        from .utils.npz_window import mask_window
        with np.load(source.quality_mask_npz) as z:
            keys = [k for k in ('bad','sat','edge','BAD','SAT','EDGE') if k in z.files]
            if not keys: raise ValueError('Quality NPZ has no supported mask keys')
        for key in keys:
            bad |= mask_window(source.quality_mask_npz, key, (ny,nx), x, y, width, height)
    header = full_header.copy()
    if (x,y) != (0,0) or (height,width) != (ny,nx):
        # Translation only: retain the original CD/PC and SIP convention.
        # Merging WCS.to_header() into a CD header could leave two matrices.
        crpix = WCS(full_header).wcs.crpix
        header['CRPIX1'] = float(crpix[0]-x)
        header['CRPIX2'] = float(crpix[1]-y)
        header['NAXIS1'] = width; header['NAXIS2'] = height
    return LoadedImage(data,bad,header,full_header,(x,y),path)


def reference_to_training(source, x, y):
    """Map zero-based catalog/full-coadd pixels onto the training image WCS."""
    ref, _, _ = header_info(source.reference_fits)
    target, _, _ = header_info(source.image_fits)
    ra, dec = WCS(ref).celestial.all_pix2world(x,y,0)
    return WCS(target).celestial.all_world2pix(ra,dec,0)


def bind_task(task, source):
    """Namespace existing classified-patch writer outputs; HSC stays unchanged."""
    if source.dataset == 'hsc': return task
    root = task.output_root / source.dataset
    if source.proposal is not None: root = root / f'proposal_{source.proposal}'
    return replace(task, output_root=root, patch=source.patch, band=source.band,
                   tract=(f'COSMOS_{source.proposal}' if source.dataset=='cosmos' else 'Abell2744'),
                   dataset_source=source.dataset_source, group='half' if source.dataset=='abell' else '')
