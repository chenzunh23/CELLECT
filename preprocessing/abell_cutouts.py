"""Align full/half Abell coadds on an anchored grid, without loading mosaics."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import time
import warnings
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from .dataset_inputs import discover_abell, header_info, image_hdu, load_paths
from .utils.sky_tiles import plan_sky_tiles, tile_header, reproject_tile, write_tile


def progress(message):
    print(f'[{datetime.now().astimezone().isoformat(timespec="seconds")}] '
          f'[pid={os.getpid()}] {message}',flush=True)


def configure_progress_warnings():
    # Keep the first copy of each metadata correction; retain other warnings.
    from astropy.wcs import FITSFixedWarning
    warnings.filterwarnings('once',message="'(obsfix|datfix)' made the change",category=FITSFixedWarning)


def completed_tasks(futures, stage, *, heartbeat_seconds=30):
    """Yield completed futures while reporting long gaps without completion."""
    pending=set(futures);total=len(pending);finished=0;started=time.monotonic()
    while pending:
        done,pending=wait(pending,timeout=heartbeat_seconds,return_when=FIRST_COMPLETED)
        if not done:
            progress(f'[{stage}] WAIT completed={finished}/{total} '
                     f'unfinished={len(pending)} elapsed={time.monotonic()-started:.1f}s')
        for future in done:
            finished+=1
            yield future
            progress(f'[{stage}] PROGRESS completed={finished}/{total} '
                     f'unfinished={len(pending)} elapsed={time.monotonic()-started:.1f}s')


def cut_band(source, reference_header, output_root, anchor, size, overlap, overwrite=False,uncover_detection=None):
    configure_progress_warnings();started=time.monotonic()
    progress(f'[abell-cutouts] START band={source.band}')
    try:
        rows=_cut_band(source,reference_header,output_root,anchor,size,overlap,overwrite,uncover_detection)
    except Exception as exc:
        progress(f'[abell-cutouts] FAILED band={source.band} elapsed={time.monotonic()-started:.1f}s error={exc!r}')
        raise
    counts={s:sum(r['status']==s for r in rows) for s in ('kept','empty','outside_uncover')}
    progress(f'[abell-cutouts] END band={source.band} elapsed={time.monotonic()-started:.1f}s counts={counts}')
    return rows


def _cut_band(source, reference_header, output_root, anchor, size, overlap, overwrite=False,uncover_detection=None):
    root=Path(output_root); rows=[]
    ih,_,_=header_info(source.image_fits)
    rh,_,_=header_info(source.reference_fits)
    native=WCS(ih).celestial
    plans=plan_sky_tiles(ih,reference_header,anchor=anchor,size=size,overlap=overlap)
    progress(f'[abell-cutouts] {source.band} planned_parents={len(plans)}; opening HALF/FULL')
    with fits.open(source.image_fits,memmap=True) as half, fits.open(source.reference_fits,memmap=True) as full:
        data=half[image_hdu(half)].data; reference=full[image_hdu(full)].data
        for index,tile in enumerate(plans,1):
            started=time.monotonic()
            progress(f'[abell-cutouts] START parent={source.band}/{tile.name} index={index}/{len(plans)} stage=coverage')
            header=tile_header(reference_header,tile,ih)
            train_path=root/'parents'/source.band/f'{tile.name}_half.fits'
            ref_path=root/'parents'/source.band/f'{tile.name}_full.fits'
            record=dict(band=source.band,**asdict(tile),patch=tile.name,
                        training_fits=str(train_path),reference_fits=str(ref_path),
                        original_training=str(source.image_fits),original_reference=str(source.reference_fits),
                        grid_anchor=list(anchor),parent_overlap=overlap)
            if uncover_detection is not None:
                from .utils.uncover_labels import coverage_on_grid
                coverage=coverage_on_grid(uncover_detection,header,(size,size),cache_root=root/'uncover_coverage')
                record['uncover_coverage_fraction']=float(coverage.mean())
                if not coverage.any():
                    record.update(status='outside_uncover',finite_pixels=None)
                    progress(f'[abell-cutouts] SKIP parent={source.band}/{tile.name} reason=outside_uncover elapsed={time.monotonic()-started:.1f}s')
                    rows.append(record);continue
            if not overwrite and train_path.exists() and ref_path.exists():
                saved=fits.getheader(train_path)
                for key,value in [('GRID_X0',tile.x0),('GRID_Y0',tile.y0),('NAXIS1',size),('NAXIS2',size)]:
                    if saved.get(key)!=value: raise ValueError(f'Stale grid: {train_path} {key}')
                if not np.allclose(WCS(saved).wcs.crval,WCS(header).wcs.crval,rtol=0,atol=1e-10):
                    raise ValueError(f'Stale WCS: {train_path}')
                count=int(saved['NVALID'])
            else:
                progress(f'[abell-cutouts] {source.band}/{tile.name} stage=reproject_HALF')
                image=reproject_tile(data,native,header)
                count=int(np.isfinite(image).sum())
                if not count:
                    record.update(status='empty',finite_pixels=0)
                    progress(f'[abell-cutouts] SKIP parent={source.band}/{tile.name} reason=empty elapsed={time.monotonic()-started:.1f}s')
                    rows.append(record);continue
                write_tile(train_path,image,header,source=source.image_fits)
                del image
                ref_header=tile_header(reference_header,tile,rh)
                progress(f'[abell-cutouts] {source.band}/{tile.name} stage=reproject_FULL')
                ref_image=reproject_tile(reference,WCS(rh).celestial,ref_header)
                write_tile(ref_path,ref_image,ref_header,source=source.reference_fits)
                del ref_image
            record.update(status='kept',finite_pixels=count,finite_fraction=count/(size*size))
            rows.append(record)
            progress(f'[abell-cutouts] END parent={source.band}/{tile.name} valid={count/(size*size):.5f} elapsed={time.monotonic()-started:.1f}s')
    p=root/'manifests'/f'{source.band}.json';p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix('.json.tmp');temp.write_text(json.dumps(rows,indent=2)+'\n');temp.replace(p)
    gc.collect()
    return rows


def run_cutouts(args):
    configure_progress_warnings();started=time.monotonic()
    progress(f'[abell-cutouts] START workers={args.workers} output={args.output_root} stage=discover_inputs')
    config=load_paths(args.paths_config)['abell']
    sources=discover_abell(config,bands=args.jwst_bands)
    progress(f'[abell-cutouts] discovered_bands={len(sources)} stage=validate_assets_and_plan')
    if getattr(args,'use_uncover',False) and not args.abell_plan_only:
        from .utils.uncover_labels import model_for_band
        for p in [args.uncover_catalog,args.uncover_detection,args.uncover_segmentation]:
            if not Path(p).is_file():raise FileNotFoundError(p)
        for source in sources:model_for_band(args.uncover_model_root,source.band)
    ref_path=next(Path(config['reference_root']).glob('*_f444w_clear_i2d_mbkg.fits'))
    ref_header,_,_=header_info(ref_path)
    root=Path(args.output_root).expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
    anchor=tuple(args.abell_anchor)
    plans={s.band:[asdict(t) for t in plan_sky_tiles(header_info(s.image_fits)[0],ref_header,
        anchor=anchor,size=args.parent_size,overlap=args.parent_overlap)] for s in sources}
    meta=dict(reference_fits=str(ref_path),anchor=list(anchor),size=args.parent_size,
        overlap=args.parent_overlap,stride=args.parent_size-args.parent_overlap,
        keep_parent='at least one finite training SCI pixel',tile_max_invalid=args.jwst_max_invalid_fraction,
        interpolation='finite-normalized bilinear surface brightness; no flux area multiplication',plans=plans)
    meta.update(use_uncover=getattr(args,'use_uncover',False),
        uncover_detection=getattr(args,'uncover_detection',None) if getattr(args,'use_uncover',False) else None,
        tile_invalid_union='NaN/no-data OR outside finite UNCOVER detection' if getattr(args,'use_uncover',False) else 'NaN/no-data')
    (root/'grid_plan.json').write_text(json.dumps(meta,indent=2)+'\n')
    if args.abell_plan_only:
        print(json.dumps({b:len(t) for b,t in plans.items()}),flush=True);return []
    results=[];failures=[]
    progress(f'[abell-cutouts] SUBMIT bands={len(sources)} workers={args.workers}; each task aligns HALF and FULL parents')
    with ProcessPoolExecutor(max_workers=args.workers,max_tasks_per_child=1) as pool:
        futures={pool.submit(cut_band,s,ref_header,root,anchor,args.parent_size,args.parent_overlap,args.overwrite,
            args.uncover_detection if getattr(args,'use_uncover',False) else None):s for s in sources}
        for f in completed_tasks(futures,'abell-cutouts'):
            try:results.extend(f.result())
            except Exception as e:
                failures.append(dict(band=futures[f].band,error=repr(e)))
                print(f'[abell-cutouts] FAILED {futures[f].band}: {e}',flush=True)
    (root/'manifest.json').write_text(json.dumps(dict(config=meta,rows=results,failures=failures),indent=2)+'\n')
    if failures:raise RuntimeError(f'{len(failures)} bands failed; see manifest.json')
    progress(f'[abell-cutouts] END kept={sum(r["status"]=="kept" for r in results)}/{len(results)} elapsed={time.monotonic()-started:.1f}s')
    return results
