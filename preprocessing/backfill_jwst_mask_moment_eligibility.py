"""Backfill JWST moment eligibility without regenerating images/dense labels.

Default: dry-run, catalogs/saved audits only. --replay-missing explicitly permits
label/SNR replay of Abell windows with missing evidence. Replays write only to
temporary files and a separate cache, NEVER the production preprocessing tree.
Stop generation/training on the selected Zarr stores before --apply.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from functools import lru_cache
import gc
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from astro_train_zarr_data import PatchZarrReader
from preprocessing.backfill_mask_moment_eligibility import ARRAY, POLICY, snapshot, commit_flags
from preprocessing.utils.mask_moments import JWST_POLICY, jwst_moment_snr


class Inputs:
    def __init__(self):
        self.stamps = {}

    def add(self, path):
        path = Path(path).expanduser().resolve()
        st = path.stat()
        stamp = dict(path=str(path), bytes=st.st_size, mtime_ns=st.st_mtime_ns)
        old = self.stamps.setdefault(str(path), stamp)
        if old != stamp:
            raise ValueError(f'Input changed during backfill: {path}')
        return path

    def check(self):
        for path in list(self.stamps):
            self.add(path)


def _json(path, inputs):
    return json.loads(inputs.add(path).read_text())


def read_sparse(reader):
    names = ['shape_source_ids', 'shape_source_classes', 'shape_source_values',
             'shape_source_centers', 'shape_source_offsets', 'tile_x0', 'tile_y0']
    d = {name.removeprefix('shape_source_'): reader.read_full_small(name) for name in names}
    n = len(d['ids'])
    if d['ids'].dtype != np.dtype('int64'):
        raise ValueError('Source IDs must be int64, never float')
    if (d['classes'].shape != (n,) or d['centers'].shape != (n, 2)
            or d['values'].shape != (n, 3)):
        raise ValueError('Sparse arrays are not ID-aligned')
    offsets = d['offsets']
    if offsets.shape != (len(d['tile_x0']), 2) or len(d['tile_y0']) != len(offsets):
        raise ValueError('Only single-band image-level stores are supported')
    cursor = 0
    for start, stop in offsets:
        if start != cursor or not start <= stop <= n:
            raise ValueError('Sparse offsets are not contiguous')
        if len(np.unique(d['ids'][start:stop])) != stop - start:
            raise ValueError('Duplicate source IDs within one tile')
        cursor = int(stop)
    if cursor != n:
        raise ValueError('Sparse offsets do not cover all sources')
    d['valid'] = (np.isfinite(d['centers']).all(1) & np.isfinite(d['values']).all(1)
                  & (d['values'][:, :2] > 0).all(1))
    return d


def blocks_for(reader):
    n = len(reader.read_full_small('shape_source_offsets'))
    blocks = reader.attrs.get('sample_blocks')
    if blocks is None:
        if reader.attrs.get('sample_kind') == 'mixed':
            raise ValueError('Mixed store lacks sample_blocks; cannot infer window provenance')
        blocks = [dict(start=0, count=n, **{k:v for k,v in reader.attrs.items()
                  if k in ('sample_kind', 'label_window', 'stamp_bounds', 'jwst_star_sources')})]
    cursor = 0
    for block in blocks:
        if block['start'] != cursor or block['count'] < 0:
            raise ValueError('Invalid sample_blocks offsets')
        cursor += block['count']
    if cursor != n:
        raise ValueError('sample_blocks do not cover the store')
    return blocks


def block_rows(d, block):
    start, stop = block['start'], block['start'] + block['count']
    if start == stop:
        return np.empty(0, np.int64), np.empty((0, 2))
    rows = np.arange(d['offsets'][start, 0], d['offsets'][stop-1, 1])
    xy = d['centers'][rows].astype(float).copy()
    shift = block.get('coordinate_shift', [0, 0])
    lo = int(rows[0]) if len(rows) else 0
    for i in range(start, stop):
        a, b = d['offsets'][i] - lo
        xy[a:b] += [d['tile_x0'][i] - shift[0], d['tile_y0'][i] - shift[1]]
    return rows, xy


def lookup(ids, catalog_ids):
    """Exact int64 search; repeated IDs across tiles are intentional."""
    ids = np.asarray(ids)
    catalog_ids = np.asarray(catalog_ids)
    if ids.dtype != np.dtype('int64') or catalog_ids.dtype != np.dtype('int64'):
        raise ValueError('ID lookup requires int64 arrays')
    order = np.argsort(catalog_ids)
    sorted_ids = catalog_ids[order]
    if len(sorted_ids) > 1 and (sorted_ids[1:] == sorted_ids[:-1]).any():
        raise ValueError('Measurement table has duplicate IDs')
    pos = np.searchsorted(sorted_ids, ids)
    good = pos < len(order)
    good[good] &= sorted_ids[pos[good]] == ids[good]
    if not good.all():
        raise ValueError(f'Missing native source IDs in measurement: {ids[~good][:5].tolist()}')
    return order[pos]


def validate_measurement(d, rows, xy, measurement):
    j = lookup(d['ids'][rows], measurement['ids'])
    if 'classes' in measurement and not np.array_equal(d['classes'][rows], measurement['classes'][j]):
        raise ValueError('Saved classes disagree with same-window measurement; refusing backfill')
    if not np.allclose(xy, measurement['xy'][j], rtol=0, atol=.003):
        raise ValueError('Source centers disagree with measurement/window coordinates')
    if not np.allclose(d['values'][rows, :2], measurement['axes'][j], rtol=1e-5, atol=.003):
        raise ValueError('Source axes disagree with measurement/window')
    if 'theta' in measurement:
        angle = (d['values'][rows, 2] - measurement['theta'][j] + np.pi/2) % np.pi - np.pi/2
        if not np.all(np.abs(angle) < .001):
            raise ValueError('Source angles disagree with measurement/window')
    return measurement['snr'][j]


@lru_cache(maxsize=1)
def cosmos_catalog(path, size, mtime, band):
    names = ['id', 'ra', 'dec', 'kron2_a', 'kron2_b', 'theta_world', f'snr_{band.lower()}']
    with fits.open(path, memmap=True) as hd:
        return {k:np.array((hd[7].data if k in hd[7].columns.names else hd[1].data)[k]) for k in names}


def cosmos_measurement(attrs, ids, inputs):
    path = inputs.add(attrs['source_catalog']);st = path.stat();band = attrs['bands'][0]
    cat = cosmos_catalog(str(path), st.st_size, st.st_mtime_ns, band)
    j = lookup(ids, cat['id'].astype(np.int64))
    # sky_wcs_header in COSMOS stores describes the FULL training image.
    if attrs.get('sky_wcs_header'):
        header = fits.Header.fromstring(attrs['sky_wcs_header'], sep='\n')
    else:
        from preprocessing.dataset_inputs import header_info
        header = header_info(inputs.add(attrs['image_fits']))[0]
    w = WCS(header).celestial
    scale = float(np.mean(proj_plane_pixel_scales(w))*3600)
    x, y = w.world_to_pixel_values(cat['ra'][j], cat['dec'][j])
    # Select unique IDs to avoid duplicate-index ambiguity in repeated tiles.
    _, first = np.unique(ids, return_index=True)
    return dict(ids=ids[first], xy=np.column_stack([x, y])[first],
                axes=np.column_stack([cat['kron2_a'][j], cat['kron2_b'][j]])[first]/scale,
                theta=np.deg2rad(cat['theta_world'][j][first]),
                snr=cat[f'snr_{band.lower()}'][j][first])


def audit_dir(store, attrs, data_root):
    band = attrs['bands'][0];patch = attrs.get('parent_id', attrs['patch'])
    candidates = [data_root/'abell/labels'/band/patch, data_root/'labels'/band/patch]
    for p in store.parents:
        candidates.extend([p/'abell/labels'/band/patch, p/'labels'/band/patch])
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(f'No Abell audit directory for {band}/{patch}; use --data-root')


def parent_measurement(audit, inputs):
    path = inputs.add(audit/'sources.fits')
    table = Table.read(path, hdu=1)
    required = ['NUMBER', 'final_class', 'parent_x', 'parent_y', 'original_kron_a', 'original_kron_b']
    if any(k not in table.colnames for k in required):
        raise ValueError('Parent audit lacks ID/class/adopted-geometry columns')
    snr, source = jwst_moment_snr(table)
    if source is None:
        raise ValueError('Parent audit lacks measured classification SNR')
    def plain(name):
        return np.asarray(np.ma.filled(table[name], np.nan), dtype=float)
    # Strip MaskedColumn subclasses BEFORE stacking (numpy/astropy mixed
    # array_function dispatch can otherwise produce a broken _mask attribute).
    return dict(ids=np.asarray(table['NUMBER'], np.int64), classes=np.asarray(table['final_class']),
                xy=np.column_stack([plain('parent_x'), plain('parent_y')]),
                axes=np.column_stack([plain('original_kron_a'), plain('original_kron_b')]), snr=snr)


@lru_cache(maxsize=1)
def code_digest():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for p in sorted(root.rglob('*.py')) + sorted((root/'assets').glob('*')):
        if p.is_file() and 'tests' not in p.parts:
            digest.update(p.read_bytes())
    for p in sorted((root.parent/'utils').glob('*snr*.py')):
        digest.update(p.read_bytes())
    digest.update((root.parent/'data_filtering/sam_input_scaling.py').read_bytes())
    return digest.hexdigest()


def check_generation_inputs(audit, attrs, inputs):
    """Check original catalog/reference/Gaia/assets, not a changed config default."""
    doc = _json(audit/'complete.json', inputs)
    sig = dict(doc['signature'])
    for key, time_key in [('image_fits','training_mtime_ns'), ('reference_fits','reference_mtime_ns'),
                          ('reference_catalog','catalog_mtime_ns')]:
        if key in attrs and time_key in sig:
            p = inputs.add(attrs[key])
            if p.stat().st_mtime_ns != sig[time_key]:
                raise ValueError(f'Generation input changed: {p}')
    for stamp in attrs.get('uncover_policy', {}).get('assets', []):
        p = inputs.add(stamp['path']);st = p.stat()
        if st.st_size != stamp['size'] or st.st_mtime_ns != stamp['mtime_ns']:
            raise ValueError(f'UNCOVER generation asset changed: {p}')
    # Mixed generation receipts retain the actual Gaia path, unlike the
    # parent receipt's mtime-only field. Never assume a new CLI default.
    for root in audit.parents:
        receipt=root/'receipts'/f'abell_{attrs["bands"][0]}_{attrs.get("parent_id",attrs["patch"])}.json'
        if not receipt.exists():continue
        outer=_json(receipt,inputs)
        for stamp in outer['signature']['inputs']:
            p=inputs.add(stamp['path']);st=p.stat()
            if st.st_size!=stamp['bytes'] or st.st_mtime_ns!=stamp['mtime_ns']:
                raise ValueError(f'Generation receipt input changed: {p}')
            if 'gaia' in p.name.lower():sig['_gaia_path']=str(p)
        break
    return sig


def replay_abell(attrs, block, audit, data_root, cache_dir, inputs):
    """Replay a missing window, saving compact measurements outside production."""
    from preprocessing.build_image_level_zarr import parse_args
    from preprocessing.dataset_inputs import image_hdu
    from preprocessing.utils.sky_tiles import reproject_tile
    from preprocessing.utils.background_tiles import abell_region
    from preprocessing.abell_zarr import ensure_background, uncover_labels_for_window
    from preprocessing.image_processing import prepare_image
    sig = check_generation_inputs(audit, attrs, inputs)
    band = attrs['bands'][0]
    training = inputs.add(attrs['image_fits']);reference = inputs.add(attrs['reference_fits'])
    h = fits.getheader(training);rh = fits.getheader(reference)
    independent = block.get('sample_kind') == 'large_source'
    prefix = 'reference'
    if independent:
        rows = _json(audit/'large_sources.json', inputs)
        matches = [r for r in rows if r.get('sample_index') == block['start'] and r.get('status') == 'kept']
        if len(matches) != 1 or block['count'] != 1:
            raise ValueError('Cannot identify independent-window audit prefix')
        prefix = matches[0]['name'].split('_')[1]
        full = inputs.add(rh['ORIGFILE']);half = inputs.add(h['ORIGFILE'])
        bounds = block.get('label_window');parent = attrs.get('parent_bounds')
        if bounds is None or parent is None:
            raise ValueError('Independent Abell block lacks label-window bounds')
        dx, dy = bounds[0]-parent[0], bounds[1]-parent[1]
        if list(block.get('coordinate_shift', [])) != [dx, dy]:
            raise ValueError('Independent label_window and coordinate_shift disagree')
        h['CRPIX1']-=dx;h['CRPIX2']-=dy
        h['NAXIS1']=bounds[2]-bounds[0];h['NAXIS2']=bounds[3]-bounds[1]
        rh = h.copy();rh['ORIGFILE']=str(full)
        sky_name=prefix+'_reference_sky'
    else:
        half=training;full=reference
        bounds=attrs['parent_bounds'];sky_name='reference_snr_background'
    background_root = Path(sig['background_root'])
    for p in sorted((background_root/'abell'/band).glob('*/complete.json')):
        inputs.add(p);inputs.add(p.parent/'background_mask.npz')
    for suffix in ['.npz','.json']:
        p=audit/(sky_name+suffix)
        if p.exists():inputs.add(p)
    args=parse_args(['--output-root','/tmp/jwst-moment-replay-not-used'])
    args.abell_catalog=str(inputs.add(attrs['reference_catalog']))
    args.uncover_catalog=args.abell_catalog
    args.background_root=str(background_root)
    cuts=attrs['snr_thresholds'];args.abell_snr_ignore_threshold=cuts['ignore_below'];args.abell_snr_weak_threshold=cuts['weak_below']
    args.use_uncover=bool(attrs.get('use_uncover'))
    if args.use_uncover:
        policy=attrs['uncover_policy'];assets=policy['assets']
        args.uncover_detection=assets[1]['path'];args.uncover_segmentation=assets[2]['path']
        args.uncover_model_root=str(Path(assets[3]['path']).parent)
        for key,attr in [('no_kron_snr','uncover_no_kron_snr'),('risk_no_kron_snr','uncover_risk_no_kron_snr'),
                         ('risk_bright_mag','uncover_risk_bright_mag'),('risk_sigma','uncover_risk_sigma'),
                         ('reject_untrusted','uncover_reject_untrusted'),('kron_other_fraction_max','uncover_kron_other_fraction')]:
            setattr(args,attr,policy[key])
        review=policy['native_psf_review'];args.uncover_native_psf_review=review['enabled']
        args.uncover_native_psf_snr=review['snr_min'];args.uncover_native_psf_moment_max=review['moment_max_arcsec']
        compact=policy['compact_centers'];args.uncover_compact_root=compact['root']
        args.uncover_blue_catalog=compact['blue_catalog'];args.uncover_compact_red_snr=compact['red_min_snr']
        args.uncover_compact_dedup_pixels=compact['dedup_pixels']
        for stamp in compact.get('assets',[]):
            p=inputs.add(stamp['path'])
            if p.stat().st_size!=stamp['size'] or p.stat().st_mtime_ns!=stamp['mtime_ns']:
                raise ValueError(f'Compact catalog changed: {p}')
    # Gaia is not a native science catalog; verify its recorded generation mtime.
    gaia=inputs.add(sig.get('_gaia_path',args.abell_gaia))
    if gaia.stat().st_mtime_ns!=sig['gaia_mtime_ns']:
        raise ValueError('Gaia generation input changed')
    original_half=inputs.add(h.get('ORIGFILE',str(half)))
    signature=dict(version=1,attrs=attrs,block=block,code=code_digest(),inputs=list(inputs.stamps.values()))
    key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    cache_dir.mkdir(parents=True,exist_ok=True);cache=cache_dir/(key+'.npz')
    if cache.exists():
        with np.load(cache,allow_pickle=False) as z:result={k:z[k].copy() for k in z.files}
        inputs.check()
        return result, 'replay_cache'
    print(f'[jwst-moment] replay {band}/{attrs["parent_id"]}/{prefix}',flush=True)
    with tempfile.TemporaryDirectory(prefix='jwst-moment-replay-') as td:
        scratch=Path(td);args.output_root=str(scratch)
        for suffix in ['.npz','.json']:
            src=audit/(sky_name+suffix)
            if src.exists():shutil.copy2(src,scratch/(sky_name+suffix))
        if independent:
            with fits.open(half,memmap=True) as hd,fits.open(full,memmap=True) as rd:
                hi=image_hdu(hd);ri=image_hdu(rd)
                raw=reproject_tile(hd[hi].data,WCS(hd[hi].header).celestial,h)
                if 'BUNIT' in rd[ri].header:rh['BUNIT']=rd[ri].header['BUNIT']
                ref=reproject_tile(rd[ri].data,WCS(rd[ri].header).celestial,rh)
        else:
            raw=fits.getdata(training);ref=fits.getdata(reference)
        background=ensure_background(full,scratch/(sky_name+'.npz'),image=ref,header=rh)
        sky,_=abell_region(background_root,band,original_half,*bounds[:2],h['NAXIS1'],h['NAXIS2'])
        catalog=Table.read(args.abell_catalog,hdu=1);gaia_table=Table.read(gaia)
        prepared=prepare_image(raw,header=h)
        if args.use_uncover:
            labels,_,_=uncover_labels_for_window(args,ref,rh,catalog,gaia_table,background,
                                                band,sky,prepared,h,scratch,prefix=prefix)
        else:
            from preprocessing.utils.abell_labels import classify_reference
            from preprocessing.snr import SnrThresholds
            labels,_=classify_reference(ref,rh,catalog,gaia_table,background,band,
                training_background=sky,bright_image=prepared,bright_header=h,
                snr_thresholds=SnrThresholds(cuts['ignore_below'],cuts['weak_below']))
        snr,source=jwst_moment_snr(labels.table)
        if source is None:raise ValueError('Replay did not produce measured SNR')
        result=dict(ids=np.asarray(labels.source_ids,np.int64),classes=np.asarray(labels.label_classes),
                    xy=np.column_stack([labels.geom_x,labels.geom_y]),
                    axes=np.column_stack([labels.geom_major,labels.geom_minor]),
                    theta=np.asarray(labels.geom_theta),snr=snr)
        inputs.check()
        fd,name=tempfile.mkstemp(prefix='.'+key,dir=cache_dir)
        try:
            with os.fdopen(fd,'wb') as out:np.savez_compressed(out,**result)
            os.replace(name,cache)
        finally:
            if Path(name).exists():Path(name).unlink()
    return result,'replay'


def build_flags(store, data_root, cache_dir, replay_missing):
    reader=PatchZarrReader(store);attrs=reader.attrs;d=read_sparse(reader)
    flags=((d['classes']==1)&d['valid']).astype(np.uint8)
    threshold=float(attrs['snr_thresholds']['weak_below'])
    if not np.isfinite(threshold) or threshold<=0:raise ValueError('Invalid saved SNR threshold')
    inputs=Inputs();methods={};pending=[]
    for block in blocks_for(reader):
        rows,xy=block_rows(d,block)
        weak=(d['classes'][rows]==2)&d['valid'][rows]
        # These appended PSF-model rows have masked numeric SNR even in the
        # original classifier. They cannot qualify as known high-SNR weak.
        stars=block.get('jwst_star_sources',attrs.get('jwst_star_sources',[])) or []
        known_psf={int(s['gaia_id']) for s in stars}
        selected=weak&~np.isin(d['ids'][rows],list(known_psf))
        if not selected.any():continue
        rows=rows[selected];xy=xy[selected]
        dataset=attrs['dataset']
        if dataset=='cosmos':
            # COSMOS SNR/ellipse are catalog quantities, independent of window.
            measurement=cosmos_measurement(attrs,d['ids'][rows],inputs);method='catalog'
        elif dataset=='abell':
            audit=audit_dir(store,attrs,data_root)
            if block.get('sample_kind','grid')=='grid':
                check_generation_inputs(audit,attrs,inputs)
                try:measurement=parent_measurement(audit,inputs);method='parent_audit'
                except (FileNotFoundError,ValueError):
                    if not replay_missing:
                        pending.append(dict(start=block['start'],kind='grid',weak=len(rows)));continue
                    measurement,method=replay_abell(attrs,block,audit,data_root,cache_dir,inputs)
            else:
                if not replay_missing:
                    pending.append(dict(start=block['start'],kind='large_source',weak=len(rows)));continue
                measurement,method=replay_abell(attrs,block,audit,data_root,cache_dir,inputs)
        else:raise ValueError(f'Not a COSMOS/Abell store: {dataset}')
        snr=validate_measurement(d,rows,xy,measurement)
        flags[rows]=(np.isfinite(snr)&(snr>=threshold)).astype(np.uint8)
        methods[method]=methods.get(method,0)+1
    if pending:
        raise ValueError(f'Missing same-window SNR for {pending}; retry with --replay-missing (no Zarr regeneration)')
    inputs.check()
    return flags,dict(mask_moment_weak_snr_threshold=threshold,
        mask_moment_snr_source='ID-aligned catalog / same-window measured aperture SNR',
        mask_moment_backfill_inputs=list(inputs.stamps.values())),dict(methods=methods,
        eligible_clean=int(((d['classes']==1)&(flags>0)).sum()),
        eligible_weak=int(((d['classes']==2)&(flags>0)).sum()))


def process_one(job):
    store,options=job;store=Path(store);started=time.monotonic()
    try:
        reader=PatchZarrReader(store);policy=reader.attrs.get('mask_moment_policy')
        if policy==JWST_POLICY and reader.has_array(ARRAY):
            flags=reader.read_full_small(ARRAY)
            if flags.shape!=reader.meta('shape_source_ids').shape or not np.isin(flags,[0,1]).all():
                raise ValueError('Existing eligibility array is invalid')
            return dict(store=str(store),status='already_done')
        if policy not in (None,POLICY,JWST_POLICY):raise ValueError('Unknown existing moment policy')
        before=snapshot(store)
        flags,attrs,stats=build_flags(store,Path(options['data_root']),Path(options['cache_dir']),options['replay_missing'])
        if snapshot(store)!=before:raise ValueError('Store changed during validation; stop its writer')
        if reader.has_array(ARRAY) and not np.array_equal(reader.read_full_small(ARRAY),flags) and not options['replace_legacy']:
            raise ValueError('Existing legacy eligibility differs; inspect then use --replace-legacy to back it up')
        if options['apply']:
            commit_flags(store,flags,before,policy=JWST_POLICY,attrs_update=attrs,
                         method='validated_jwst_snr_backfill',replace_legacy=options['replace_legacy'])
        return dict(store=str(store),status='written' if options['apply'] else 'dry_run_ok',
                    count=len(flags),eligible=int(flags.sum()),seconds=round(time.monotonic()-started,2),**stats)
    except Exception as exc:
        return dict(store=str(store),status='error',error=f'{type(exc).__name__}: {exc}')
    finally:gc.collect()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--data-root',type=Path,help='Original generation root containing abell/labels, if moved')
    parser.add_argument('--datasets',nargs='+',choices=['cosmos','abell'],default=['cosmos','abell'])
    parser.add_argument('--bands',nargs='+')
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--cache-dir',type=Path,default=Path('/home/czh23/.cache/cellect/jwst_moment_backfill'))
    parser.add_argument('--replay-missing',action='store_true',help='Permit missing Abell window label/SNR replays; cached outside Zarr')
    parser.add_argument('--replace-legacy',action='store_true',help='Back up and replace a differing legacy eligibility array only')
    parser.add_argument('--apply',action='store_true',help='Write metadata; otherwise validate only')
    parser.add_argument('--report',type=Path,help='Optional JSONL report (also written in dry-run)')
    args=parser.parse_args(argv)
    root=args.root.expanduser().resolve()
    if not root.is_dir() or args.workers<1 or args.limit<0:parser.error('Require existing root, positive workers and nonnegative limit')
    data_root=(args.data_root or root).expanduser().resolve()
    if args.data_root is None:
        for p in [root,*root.parents]:
            if (p/'queue.json').is_file() or (p/'labels').is_dir():data_root=p;break
    stores=[]
    for p in sorted([root] if root.suffix=='.zarr' else root.rglob('*.zarr')):
        a=json.loads((p/'.zattrs').read_text())
        if a.get('dataset') in args.datasets and len(a.get('bands',[]))==1 and (not args.bands or a['bands'][0] in args.bands):
            stores.append(p)
    if args.limit:stores=stores[:args.limit]
    if not stores:parser.error('No matching COSMOS/Abell stores')
    options=dict(data_root=str(data_root),cache_dir=str(args.cache_dir.expanduser().resolve()),
                 replay_missing=args.replay_missing,replace_legacy=args.replace_legacy,apply=args.apply)
    failed=0
    with ExitStack() as stack:
        # Same locks as both generation entry points, opened read-only. Never
        # truncate a live lock or send anything to an existing training pane.
        import fcntl
        for p in [data_root/'logs/generation.lock',data_root/'.scheduler.lock']:
            if p.exists():
                h=stack.enter_context(p.open('rb'))
                try:fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:parser.error(f'Generation is active: {p}')
        report=None
        if args.report:
            args.report.parent.mkdir(parents=True,exist_ok=True);report=stack.enter_context(args.report.open('a'))
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            jobs={pool.submit(process_one,(str(p),options)):p for p in stores}
            for future in as_completed(jobs):
                result=future.result();line=json.dumps(result);print(line,flush=True)
                if report:report.write(line+'\n');report.flush()
                failed+=result['status']=='error'
        print(json.dumps(dict(selected=len(stores),failed=failed,apply=args.apply)),flush=True)
    return int(failed>0)


if __name__=='__main__':
    raise SystemExit(main())
