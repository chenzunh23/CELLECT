"""Resolve LSST backgrounds for the exact HSC half/noisy training image."""
from pathlib import Path
import json
import numpy as np


def resolve_lsst_background(root, row, tract):
    """Return mask and optional receipt; never fall back to a full coadd."""
    from .image_level import _variant_background_dirs
    if not root:
        raise ValueError('--hsc-background-method lsst requires --variant-lsst-background-root')
    variant = {'hsc_half': 'half_coadd', 'hsc_noisy': 'noisy'}[row['kind']]
    directories = _variant_background_dirs(
        Path(root), variant, tract, row['patch'], row['group'], row['band'])
    # Historical half-coadd LSST jobs call their single stack "coadd".
    # This is an alias within half_coadd/, not the original full coadd.
    alias = Path(root) / variant / str(tract) / row['patch'] / 'coadd' / row['band']
    if variant == 'half_coadd' and alias not in directories:
        directories.append(alias)
    for directory in directories:
        mask = directory / 'background_mask.npz'
        candidates = [mask] if mask.is_file() else sorted(directory.glob('det-*.fits'))
        if not candidates:
            continue
        receipt = directory / 'summary.json'
        if receipt.exists():
            doc = json.loads(receipt.read_text())
            if doc.get('status') not in (None, 'ok', 'done'):
                raise ValueError(f'Incomplete LSST background: {receipt}')
            source = doc.get('input')
            if source and Path(source).resolve() != Path(row['source']).resolve():
                raise ValueError(f'LSST background belongs to {source}, not {row["source"]}: {receipt}')
            if directory == alias and not source:
                raise ValueError(f'Half-coadd alias needs an input path in {receipt}')
        elif directory == alias:
            raise ValueError(f'Cannot verify half-coadd alias without {receipt}')
        return candidates[0], receipt if receipt.exists() else None
    raise FileNotFoundError('Matching HSC LSST background not found; tried: ' +
                            ', '.join(str(d) for d in directories))


def read_lsst_background(path, shape, origin, receipt=None):
    """Read existing NPZ/det products, checking shape and known grid origin."""
    path = Path(path)
    if receipt is not None:
        doc = json.loads(Path(receipt).read_text())
        if 'shape_yx' in doc and tuple(doc['shape_yx']) != tuple(shape):
            raise ValueError(f'LSST background shape mismatch: {receipt}')
        if 'origin_xy' in doc and tuple(doc['origin_xy']) != tuple(origin):
            raise ValueError(f'LSST background pixel origin mismatch: {receipt}')
    if path.suffix == '.npz':
        with np.load(path) as z:
            key = next((k for k in ('background', 'background_mask', 'lsst_background') if k in z), None)
            if key is None:
                raise ValueError(f'No background mask array in {path}')
            mask = np.asarray(z[key], dtype=bool)
    else:
        from .inputs import _read_det_background_mask
        mask = _read_det_background_mask(path, shape, origin)
    if mask.shape != tuple(shape):
        raise ValueError(f'LSST background shape mismatch: {path}: {mask.shape} != {shape}')
    return mask


def resolve_training_lsst_background(args, row):
    """Explicitly select matching-input or full-coadd sky for HSC half images."""
    mode=getattr(args,'hsc_half_lsst_background_source','training')
    if row['kind']!='hsc_half' or mode=='training':
        return resolve_lsst_background(args.variant_lsst_background_root,row,args.tract)
    if mode!='coadd':raise ValueError(f'Unknown HSC half background source: {mode}')
    from .image_level import _coadd_image_path
    from ..dataset_inputs import header_info
    from astropy.wcs import WCS
    reference=_coadd_image_path(Path(args.data_root),row['band'],args.tract,row['patch'])
    folder=Path(args.variant_lsst_background_root)/'coadd'/str(args.tract)/row['patch']/'coadd'/row['band']
    mask=folder/'background_mask.npz';receipt=folder/'summary.json'
    if not mask.is_file() or not receipt.is_file():
        raise FileNotFoundError(f'Missing full-coadd HSC background or receipt: {folder}')
    doc=json.loads(receipt.read_text())
    if doc.get('status') not in ('ok','done') or Path(doc.get('input','')).resolve()!=reference.resolve():
        raise ValueError(f'Full-coadd background provenance mismatch: {receipt}')
    half_header,half_shape,_=header_info(row['source'])
    full_header,full_shape,_=header_info(reference)
    if half_shape!=full_shape or tuple(doc['shape_yx'])!=half_shape:
        raise ValueError(f'Half/full-coadd background grid shape mismatch: {folder}')
    for key in ('LTV1','LTV2'):
        if not np.isclose(half_header.get(key,0),full_header.get(key,0)):
            raise ValueError(f'Half/full-coadd pixel origin mismatch: {folder}')
    h,w=half_shape;xs=np.array([0,w-1,0,w-1,(w-1)/2]);ys=np.array([0,0,h-1,h-1,(h-1)/2])
    sky=WCS(half_header).celestial.pixel_to_world_values(xs,ys)
    fx,fy=WCS(full_header).celestial.world_to_pixel_values(*sky)
    error=np.hypot(fx-xs,fy-ys)
    if not np.isfinite(error).all() or np.max(error)>.01:
        raise ValueError(f'Half/full-coadd WCS mismatch: {folder}; max error={np.max(error)} px')
    return mask,receipt
