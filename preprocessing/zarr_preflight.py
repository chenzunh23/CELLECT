"""Metadata preflight for run_generate_zarr.sh; never classifies or writes Zarr.

Validate the selected jobs against their actual training-image backgrounds,
catalog/reference inputs and Gaia/PSF assets before starting worker processes.
This intentionally reuses completed background products without rebuilding them.
"""
import argparse
from collections import Counter
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path

from .build_image_level_zarr import parse_args
from .training_batch import plan, precomputed_hsc_background
from .labels import LabelWeights
from .utils.background_tiles import check_source
from .utils.confidence import confidence_asset
from .utils.image_level import _coadd_image_path, _refit_csv_path
from .utils.inputs import _band_catalog_path
from .utils.jwst_background_local import METHOD
from .utils.jwst_star_psf import ASSET, star_asset, star_policy
from .utils.source_weights import source_area_weight_policy


def required(path):
    path = Path(path)
    if not path.is_file() or not path.stat().st_size:
        raise FileNotFoundError(f'Missing or empty input: {path}')
    return path


@lru_cache(maxsize=None)
def document(path):
    return json.loads(required(path).read_text())


def background_receipt(path, source, *, allowed_thresholds=(1.5,)):
    doc = document(str(path))
    if doc.get('status') != 'done':
        raise ValueError(f'Background is not complete: {path}')
    sig = doc['signature']
    check_source(sig['source'], source)
    if sig['method'] != METHOD:
        raise ValueError(f'Unexpected background method: {path}')
    thresholds=sig['config'].get('thresholds_sigma')
    if not any(thresholds==[value] for value in allowed_thresholds):
        raise ValueError(f'Expected background threshold in {allowed_thresholds}: {path}')
    expected = dict(background_mesh=128, background_filter_size=3,
        thresholds_sigma=thresholds, minarea_native_pixels=[1], factors=[1],
        smoothing_mode='psf', smoothing_sigma_floor_pixels=1., grow_factor=2.4)
    if any(sig['config'].get(k) != v for k, v in expected.items()):
        raise ValueError(f'Expected AUTO128 / PSF smoothing / minarea 1: {path}')
    return doc


def check(args):
    if not args.training_batch:
        raise ValueError('This preflight requires --training-batch')
    if args.workers < 1 or args.chunk_tiles < 1 or args.tile_size != 512:
        raise ValueError('Require workers/chunk-tiles >= 1 and tile-size=512')
    recipe = document(str(Path(args.background_root)/'recipe.json'))
    if recipe.get('method') != METHOD or recipe.get('input') not in ('training half coadd','training half coadd and noisy'):
        raise ValueError('Background root must contain the completed training-image recipe')
    rows = plan(args)
    counts = Counter(r['kind'] for r in rows)
    if not rows:
        raise ValueError('No training jobs selected')
    if not args.job_limit:
        missing = set(args.training_kinds) - set(counts)
        if missing:
            raise ValueError(f'No jobs discovered for requested families: {sorted(missing)}')
    checked = set()
    jwst_bands = set()
    for row in rows:
        kind = row['kind']
        if kind == 'cosmos':
            source = row['source']
            jwst_bands.add(source['band'])
            required(source['image_fits']); required(source['quality_mask_npz'])
            required(Path(source['catalog_root'])/'COSMOSWeb_mastercatalog_v1.1.fits')
            required(args.paths['cosmos']['gaia_fits'])
            manifest = Path(row['background'])
            if manifest not in checked:
                doc = document(str(manifest))
                # COSMOS may use the reviewed threshold=2 recipe; other
                # families retain their existing threshold=1.5 validation.
                receipt = background_receipt(manifest.parent/'complete.json', source['image_fits'],
                                             allowed_thresholds=(1.5,2.))
                check_source(doc['source'], source['image_fits'])
                if doc['method'] != METHOD or doc['config'] != receipt['signature']['config']:
                    raise ValueError(f'COSMOS manifest/receipt recipe mismatch: {manifest}')
                if doc['config']['thresholds_sigma'] != recipe.get('thresholds_sigma'):
                    raise ValueError(f'COSMOS background root/manifest threshold mismatch: {manifest}')
                for tile in doc['tiles']:
                    required(tile['mask'])
                checked.add(manifest)
        elif kind == 'abell':
            job = row['job']
            jwst_bands.add(job['band'])
            folder = Path(job['output'])
            background_receipt(folder/'complete.json', job['source'])
            required(folder/'background_mask.npz'); required(folder/'training_half_4096.fits')
            required(job['grid_reference']); required(args.abell_catalog); required(args.abell_gaia)
            checked.add(folder/'complete.json')
        else:
            required(row['source']); required(args.gaia_fits)
            reference = required(_coadd_image_path(Path(args.data_root),row['band'],args.tract,row['patch']))
            required(_refit_csv_path(Path(args.refit_root),args.tract,row['band'],row['patch']))
            required(_band_catalog_path(Path(args.data_root),row['band'],args.tract,row['patch']))
            if args.hsc_background_method == 'sextractor-precomputed':
                mask, receipt = precomputed_hsc_background(args,row)
                background_receipt(receipt,row['source']); required(mask)
                checked.add(receipt)
            elif args.hsc_background_method == 'lsst':
                from .utils.hsc_background import resolve_training_lsst_background
                mask, _ = resolve_training_lsst_background(args,row)
                required(mask)
            else:
                raise ValueError('Use sextractor-precomputed or explicitly select lsst; no silent background regeneration')
            # HSC confidence radii are measured from persisted PSFs in LSST.
            from .utils.hsc_psf_ee import ASSET as hsc_asset
            required(document(str(hsc_asset))['lsst_setup'])
    if jwst_bands:
        config, _ = star_asset()
        ee, _ = confidence_asset('psf-ee')
        for band in sorted(jwst_bands):
            required(config['filters'][band]['path'])
            required(ee['filters'][band]['path'])
        if config['policy']['star_wings_confidence_weight'] != LabelWeights().star_wings:
            raise ValueError('STAR_WINGS asset and written confidence weights disagree')
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.rglob('*.py')) + sorted((Path(__file__).parent/'assets').glob('*.json')):
        if 'tests' not in path.parts:
            digest.update(path.read_bytes())
    return dict(arguments=vars(args), parent_jobs=dict(counts), selected_jobs=len(rows),
        checked_background_products=len(checked), background_recipe=recipe,
        dense_region_weights=asdict(LabelWeights()), source_area_weights=source_area_weight_policy(),
        weak_shape_training='source_center shape weight = area weight * training --center-only-shape-factor (default 0.2); not baked into Zarr',
        confidence='positive centers override regional negatives; CE class weights are applied by training',
        jwst_star_protection=star_policy() if jwst_bands else None,
        preprocessing_sha256=digest.hexdigest(), star_asset=str(ASSET),
        grid='HSC/Abell stride 368; COSMOS grid stride 512 in 4096 parents',
        output_root=str(Path(args.output_root).resolve()),
        note='Metadata checks only; no science pixel reads, labels, PSF fitting or Zarr generation')


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--report',required=True)
    options, rest = parser.parse_known_args()
    args = parse_args(rest)
    result = check(args)
    output = Path(options.report)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,default=str)+'\n')
    print('Preflight passed:',result['parent_jobs'],flush=True)
    print('Dense region weights:',result['dense_region_weights'],flush=True)
    print('Output:',result['output_root'],flush=True)


if __name__ == '__main__':
    main()
