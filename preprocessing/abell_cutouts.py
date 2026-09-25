"""Align full/half Abell coadds on an anchored grid, without loading mosaics."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import gc
import json
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from .dataset_inputs import discover_abell, header_info, image_hdu, load_paths
from .utils.sky_tiles import plan_sky_tiles, tile_header, reproject_tile, write_tile


def cut_band(source, reference_header, output_root, anchor, size, overlap, overwrite=False):
    root=Path(output_root); rows=[]
    ih,_,_=header_info(source.image_fits)
    rh,_,_=header_info(source.reference_fits)
    native=WCS(ih).celestial
    plans=plan_sky_tiles(ih,reference_header,anchor=anchor,size=size,overlap=overlap)
    with fits.open(source.image_fits,memmap=True) as half, fits.open(source.reference_fits,memmap=True) as full:
        data=half[image_hdu(half)].data; reference=full[image_hdu(full)].data
        for tile in plans:
            header=tile_header(reference_header,tile,ih)
            train_path=root/'parents'/source.band/f'{tile.name}_half.fits'
            ref_path=root/'parents'/source.band/f'{tile.name}_full.fits'
            record=dict(band=source.band,**asdict(tile),patch=tile.name,
                        training_fits=str(train_path),reference_fits=str(ref_path),
                        original_training=str(source.image_fits),original_reference=str(source.reference_fits),
                        grid_anchor=list(anchor),parent_overlap=overlap)
            if not overwrite and train_path.exists() and ref_path.exists():
                saved=fits.getheader(train_path)
                for key,value in [('GRID_X0',tile.x0),('GRID_Y0',tile.y0),('NAXIS1',size),('NAXIS2',size)]:
                    if saved.get(key)!=value: raise ValueError(f'Stale grid: {train_path} {key}')
                if not np.allclose(WCS(saved).wcs.crval,WCS(header).wcs.crval,rtol=0,atol=1e-10):
                    raise ValueError(f'Stale WCS: {train_path}')
                count=int(saved['NVALID'])
            else:
                image=reproject_tile(data,native,header)
                count=int(np.isfinite(image).sum())
                if not count:
                    record.update(status='empty',finite_pixels=0)
                    rows.append(record);continue
                write_tile(train_path,image,header,source=source.image_fits)
                del image
                ref_header=tile_header(reference_header,tile,rh)
                ref_image=reproject_tile(reference,WCS(rh).celestial,ref_header)
                write_tile(ref_path,ref_image,ref_header,source=source.reference_fits)
                del ref_image
            record.update(status='kept',finite_pixels=count,finite_fraction=count/(size*size))
            rows.append(record)
            print(f'[abell-cutouts] {source.band} {tile.name} valid={count/(size*size):.5f}',flush=True)
    p=root/'manifests'/f'{source.band}.json';p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix('.json.tmp');temp.write_text(json.dumps(rows,indent=2)+'\n');temp.replace(p)
    gc.collect()
    return rows


def run_cutouts(args):
    config=load_paths(args.paths_config)['abell']
    sources=discover_abell(config,bands=args.jwst_bands)
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
    (root/'grid_plan.json').write_text(json.dumps(meta,indent=2)+'\n')
    if args.abell_plan_only:
        print(json.dumps({b:len(t) for b,t in plans.items()}),flush=True);return []
    results=[];failures=[]
    with ProcessPoolExecutor(max_workers=args.workers,max_tasks_per_child=1) as pool:
        futures={pool.submit(cut_band,s,ref_header,root,anchor,args.parent_size,args.parent_overlap,args.overwrite):s for s in sources}
        for f in as_completed(futures):
            try:results.extend(f.result())
            except Exception as e:
                failures.append(dict(band=futures[f].band,error=repr(e)))
                print(f'[abell-cutouts] FAILED {futures[f].band}: {e}',flush=True)
    (root/'manifest.json').write_text(json.dumps(dict(config=meta,rows=results,failures=failures),indent=2)+'\n')
    if failures:raise RuntimeError(f'{len(failures)} bands failed; see manifest.json')
    print(f'[abell-cutouts] kept {sum(r["status"]=="kept" for r in results)} / {len(results)}',flush=True)
    return results
