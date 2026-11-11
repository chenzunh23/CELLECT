"""Replay HSC classification once, validate sparse targets, add eligibility only.

Dry-run is the default. Stop generation/training on the selected stores before
--apply. The original generation command is parsed, NEVER executed. Coordinate
corrections require full regeneration, not this metadata-only backfill.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields, replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import tempfile
import time

import numpy as np

from astro_train_zarr_data import PatchZarrReader
from direct_zarr_preprocessing.zarr_writer import ZarrArrayWriter
from preprocessing.utils.geometry import EllipseGeometry
from preprocessing.utils.image_level import StoreTask
from preprocessing.utils.mask_moments import POLICY, moment_eligibility

ARRAY = 'shape_source_moment_eligible'
SPARSE = ('shape_source_ids', 'shape_source_classes', 'shape_source_centers',
          'shape_source_values', 'shape_source_offsets', 'tile_x0', 'tile_y0')


def snapshot(store):
    """Fingerprint only existing metadata/sparse arrays; never read RGB chunks."""
    digest = hashlib.sha256((store / '.zattrs').read_bytes())
    for name in SPARSE:
        for path in sorted((store / name).iterdir()):
            if path.is_file():
                digest.update(name.encode() + path.name.encode() + path.read_bytes())
    return digest.hexdigest()


def aligned_flags(reader, labels, origin):
    """Validate int64 IDs, classes, local centers and ellipse values before mapping."""
    ids = reader.read_full_small('shape_source_ids')
    classes = reader.read_full_small('shape_source_classes')
    centers = reader.read_full_small('shape_source_centers')
    values = reader.read_full_small('shape_source_values')
    offsets = reader.read_full_small('shape_source_offsets')
    if ids.dtype != np.dtype('int64') or len(np.unique(labels.source_ids)) != len(labels.source_ids):
        raise ValueError('Source IDs must be unique int64 in the replay catalog')
    if offsets.ndim != 2 or offsets.shape[1] != 2:
        raise ValueError('Only single-band HSC stores are supported')
    if (classes.shape != ids.shape or centers.shape != (len(ids), 2)
            or values.shape != (len(ids), 3)):
        raise ValueError('Saved sparse arrays are not aligned')
    # Search on int64, never via float: HSC/Gaia IDs exceed 2**53.
    order = np.argsort(labels.source_ids)
    sorted_ids = labels.source_ids[order]
    pos = np.searchsorted(sorted_ids, ids)
    if len(ids) and (np.any(pos >= len(order)) or np.any(sorted_ids[pos] != ids)):
        raise ValueError('Saved source IDs absent from replay; regenerate this store')
    rows = order[pos]
    if not np.array_equal(classes, labels.label_classes[rows]):
        raise ValueError('Source classifications changed; regenerate this store')
    expected = np.column_stack([labels.geom_major[rows], labels.geom_minor[rows], labels.geom_theta[rows]])
    if not np.allclose(values[:, :2], expected[:, :2], rtol=1e-5, atol=1e-3):
        raise ValueError('Ellipse axes changed; regenerate this store')
    angles = (values[:, 2] - expected[:, 2] + np.pi / 2) % np.pi - np.pi / 2
    if not np.all(np.abs(angles) < 1e-3):
        raise ValueError('Ellipse angles changed; regenerate this store')
    x0 = reader.read_full_small('tile_x0') - origin[0]
    y0 = reader.read_full_small('tile_y0') - origin[1]
    if len(offsets) != len(x0) or len(y0) != len(x0):
        raise ValueError('Tile offsets do not align')
    cursor = 0
    for i, (start, stop) in enumerate(offsets):
        if start != cursor or not start <= stop <= len(ids):
            raise ValueError('Sparse offsets are not contiguous')
        r = rows[start:stop]
        xy = np.column_stack([labels.geom_x[r] - x0[i], labels.geom_y[r] - y0[i]])
        if not np.allclose(centers[start:stop], xy, rtol=0, atol=1e-3):
            raise ValueError('Center coordinates changed (possibly old +1 shift); regenerate this store')
        cursor = stop
    if cursor != len(ids):
        raise ValueError('Sparse offsets do not cover all sources')
    geom = EllipseGeometry(labels.geom_x, labels.geom_y, labels.geom_major,
                           labels.geom_minor, labels.geom_theta,
                           np.pi * labels.geom_major * labels.geom_minor)
    return moment_eligibility(labels.table, labels.label_classes, geom)[rows].astype(np.uint8)


def commit_flags(store, flags, expected_snapshot, *, policy=POLICY, attrs_update=None,
                 method='validated_classification_replay', replace_legacy=False):
    """Publish only the new array and policy; preserve original attrs for recovery.

    If interrupted between array and policy publication, rerunning verifies the
    existing array and completes the policy. Readers should be stopped meanwhile.
    """
    with (store / '.mask_moment_backfill.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if snapshot(store) != expected_snapshot:
            raise RuntimeError('Store changed during replay; stop its writer and retry')
        if (store / '.zmetadata').exists():
            raise ValueError('Consolidated metadata not supported; no data changed')
        flags = np.asarray(flags)
        reader = PatchZarrReader(store)
        if flags.shape != reader.meta('shape_source_ids').shape or not np.isin(flags, [0, 1]).all():
            raise ValueError('Eligibility must be an ID-aligned 0/1 vector')
        # Validate replacement authority BEFORE publishing anything. Backups
        # are renamed directories, preserving shared/hardlinked chunk bytes.
        replace = False
        if (store / ARRAY).exists():
            if not reader.has_array(ARRAY) or not np.array_equal(reader.read_full_small(ARRAY), flags):
                from preprocessing.utils.mask_moments import JWST_POLICY
                replace = (replace_legacy and policy == JWST_POLICY
                           and reader.attrs.get('mask_moment_policy') in (None, POLICY))
                if not replace:
                    raise ValueError('Existing eligibility differs; refusing to overwrite')
                if (store / ('.' + ARRAY + '.before_jwst_moment_backfill')).exists():
                    raise ValueError('Previous eligibility backup exists; inspect before replacement')
        if not (store / ARRAY).exists() or replace:
            with tempfile.TemporaryDirectory(prefix='.moment-stage-', dir=store) as td:
                writer = ZarrArrayWriter(Path(td), ARRAY, shape=flags.shape,
                                         chunks=(max(1, len(flags)),), dtype=np.uint8)
                writer.write_full(flags)
                if replace:
                    (store / ARRAY).rename(store / ('.' + ARRAY + '.before_jwst_moment_backfill'))
                (Path(td) / ARRAY).rename(store / ARRAY)
        original = (store / '.zattrs').read_bytes()
        backup = store / '.zattrs.before_mask_moment_backfill'
        if not backup.exists():
            with backup.open('xb') as handle:
                handle.write(original)
        attrs = json.loads(original)
        attrs.update(attrs_update or {})
        attrs['mask_moment_policy'] = policy
        attrs['mask_moment_backfill'] = dict(version=1, method=method,
                                            eligible=int(flags.sum()), count=len(flags))
        fd, name = tempfile.mkstemp(prefix='.zattrs-moment-', dir=store)
        try:
            with os.fdopen(fd, 'w') as handle:
                json.dump(attrs, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, store / '.zattrs')
        finally:
            if Path(name).exists():
                Path(name).unlink()


def generation_args(command_file):
    from preprocessing.build_image_level_zarr import parse_args
    tokens = shlex.split(Path(command_file).read_text())
    module = 'preprocessing.build_image_level_zarr'
    i = tokens.index('-m')
    if tokens[i + 1] != module:
        raise ValueError(f'Expected a saved python -m {module} command')
    return parse_args(tokens[i + 2:])


def replay(store, args, scratch):
    from preprocessing.build_image_level_zarr import _classify_patch
    from preprocessing.training_batch import task_for
    from preprocessing.utils.image_level import _read_image_header_origin
    from preprocessing.utils.hsc_background import read_lsst_background
    attrs = PatchZarrReader(store).attrs
    band = attrs['bands'][0]
    source = attrs['dataset_source']
    kind = {'half_coadd': 'hsc_half', 'noisy': 'hsc_noisy'}.get(source)
    if kind is None:
        raise ValueError('Only training-batch HSC half_coadd/noisy supported')
    receipt = Path(args.output_root) / 'receipts' / f'{kind}_{band}_{attrs["patch"]}_{attrs["group"]}.json'
    doc = json.loads(receipt.read_text())
    if str(store.resolve()) not in [str(Path(p).resolve()) for p in doc['result']['outputs']]:
        raise ValueError(f'Store is not covered by generation receipt: {receipt}')
    # Missing/changed inputs must not silently produce a new eligibility policy.
    def check_inputs():
        for stamp in doc['signature']['inputs']:
            st = Path(stamp['path']).stat()
            if (st.st_size, st.st_mtime_ns) != (stamp['bytes'], stamp['mtime_ns']):
                raise ValueError(f'Generation input changed: {stamp["path"]}')
    check_inputs()
    if Path(doc['signature']['job']['source']).resolve() != Path(attrs['image_fits']).resolve():
        raise ValueError('Receipt and stored input disagree')
    options = {f.name: attrs[f.name] for f in fields(StoreTask)
               if f.name in attrs}
    options.update(output_root=scratch, coadd_fits_root=Path(args.data_root),
                   denoised_fits_root=Path(args.hsc_products_root), band=band,
                   tract=int(attrs['tract']), dataset_source=source)
    # Saved paths are strings, StoreTask uses Paths.
    for name in ('coadd_lsst_background_root', 'variant_lsst_background_root', 'bright_object_mask_root'):
        if name in options:
            options[name] = Path(options[name]) if options[name] else None
    options['gaia_fits'] = Path(attrs['hsc_gaia_catalog'])
    task = task_for(args, **options)
    thresholds = attrs['snr_thresholds']
    task = replace(task, snr_ignore_threshold=thresholds['ignore_below'],
                   snr_weak_threshold=thresholds['weak_below'],
                   narrow_snr_ignore_threshold=thresholds['narrow_ignore_below'],
                   narrow_snr_weak_threshold=thresholds['narrow_weak_below'])
    # Reuse persisted PSFs without writing to the production cache, even in dry-run.
    original_cache = Path(args.output_root) / 'zarr/hsc/hsc_star_psf' / band / attrs['patch']
    if original_cache.is_dir():
        shutil.copytree(original_cache, scratch / 'hsc_star_psf' / band / attrs['patch'])
    image, header, origin = _read_image_header_origin(Path(attrs['image_fits']))
    sky = read_lsst_background(attrs['background_mask'], image.shape, origin) & np.isfinite(image)
    labels = _classify_patch(task, image, header, origin, Path(attrs['reference_fits']), background_override=sky)
    if labels.star_protection_policy and labels.star_protection_policy.get('psf_error'):
        raise ValueError('PSF replay failed; refusing partial backfill')
    check_inputs()
    return aligned_flags(PatchZarrReader(store), labels, origin)


def process_one(job):
    store_name, command_file, apply = job
    store = Path(store_name)
    started = time.monotonic()
    try:
        reader = PatchZarrReader(store)
        if reader.attrs.get('mask_moment_policy') == POLICY and reader.has_array(ARRAY):
            flag = reader.read_full_small(ARRAY)
            if flag.shape != reader.meta('shape_source_ids').shape or not np.isin(flag, [0, 1]).all():
                raise ValueError('Existing eligibility is invalid')
            return dict(store=str(store), status='already_done')
        if reader.attrs.get('hsc_refit_center_policy') != 'fits1_to_numpy0_v1':
            raise ValueError('Missing fixed-coordinate policy; regenerate before backfill')
        if reader.attrs.get('mask_moment_policy') not in (None, POLICY):
            raise ValueError('Unknown moment policy; refusing overwrite')
        before = snapshot(store)
        args = generation_args(command_file)
        with tempfile.TemporaryDirectory(prefix='hsc-moment-replay-') as td:
            flags = replay(store, args, Path(td))
        if snapshot(store) != before:
            raise RuntimeError('Store changed during replay')
        if apply:
            commit_flags(store, flags, before)
        return dict(store=str(store), status='written' if apply else 'dry_run_ok',
                    count=len(flags), eligible=int(flags.sum()), seconds=round(time.monotonic() - started, 2))
    except Exception as exc:
        return dict(store=str(store), status='error', error=f'{type(exc).__name__}: {exc}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--generation-command', required=True, type=Path)
    parser.add_argument('--dataset-sources', nargs='+', choices=['noisy', 'half_coadd'], default=['noisy', 'half_coadd'])
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--limit', type=int, default=0, help='0 means all selected stores')
    parser.add_argument('--apply', action='store_true', help='Actually add arrays; otherwise dry-run')
    args = parser.parse_args(argv)
    if args.workers < 1 or args.limit < 0:
        parser.error('workers must be positive; limit must be nonnegative')
    generation_args(args.generation_command)  # fail early on invalid command
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f'Root does not exist: {root}')
    stores = []
    for store in sorted([root] if root.suffix == '.zarr' else root.rglob('*.zarr')):
        attrs = json.loads((store / '.zattrs').read_text())
        bands = attrs.get('bands', [])
        if (len(bands) == 1 and bands[0].startswith(('HSC-', 'NB'))
                and attrs.get('dataset_source') in args.dataset_sources):
            stores.append(store)
    if args.limit:
        stores = stores[:args.limit]
    if not stores:
        parser.error('No matching HSC stores')
    jobs = [(str(p), str(args.generation_command.resolve()), args.apply) for p in stores]
    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(process_one, jobs):
            print(json.dumps(result), flush=True)
            failed += result['status'] == 'error'
    print(json.dumps(dict(selected=len(stores), failed=failed, apply=args.apply)), flush=True)
    return int(failed > 0)


if __name__ == '__main__':
    raise SystemExit(main())
