"""Abell full-reference labels and independent source-centered large stamps."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields, replace
import gc
import json
import os
from pathlib import Path
import time
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from .utils.image_level import StoreTask
from .utils.inputs import make_tile_specs, TileSpec
from .utils.large_sources import large_source_specs, valid_tile_specs, crop_padded
from .utils.parent_scaling import fit_parent_rgb, apply_parent_rgb
from .utils.sky_tiles import reproject_tile
from .utils.abell_labels import classify_reference
from .image_processing import prepare_image
from .labels import DenseLabel
from .dataset_inputs import header_info, image_hdu
from .utils.sextractor_background import SExtractorBackgroundConfig, sextractor_background
from .utils.jwst_background_sextractor import aggressive_sextractor_background, METHOD as AGGRESSIVE_BACKGROUND_METHOD


def store_task(args, row):
    # Explicit JWST task: none of the HSC catalog/refit paths are consumed.
    values={f.name:getattr(args,f.name,None) for f in fields(StoreTask)}
    values.update(output_root=Path(getattr(args,'zarr_output_root',Path(args.output_root)/'zarr')),data_root=Path('.'),
        coadd_fits_root=Path('.'),refit_root=Path('.'),denoised_fits_root=Path('.'),coadd_weight_root=Path('.'),
        tract='Abell2744',patch=row['patch'],band=row['band'],dataset_source='half_coadd',group='half',
        tile_size=512,stride=368,max_tiles=args.max_tiles,overwrite=True,image_scaling_scope='patch',image_scaling_mode='zscore-log-lupton-rgb',
        image_log_a=1000.,bright_log_a=1000.,clip_threshold=5.,gaia_fits=Path(args.abell_gaia),
        coadd_lsst_background_root=None,variant_lsst_background_root=None,bright_object_mask_root=None,
        subtract_bright_object_from_background=False)
    return StoreTask(**values)


def ensure_background(reference, dest, *, image, header):
    """SExtractor source-exclusion mask on the full reference image grid."""
    return aggressive_sextractor_background(image, header, dest, source=str(reference))


def build_parent(args, row):
    from .build_image_level_zarr import write_classified_patch
    root=Path(args.output_root);audit=root/'labels'/row['band']/row['patch'];audit.mkdir(parents=True,exist_ok=True)
    receipt=audit/'complete.json'
    from .utils.confidence import confidence_asset
    mode = 'psf-ee' if args.confidence_mode == 'auto' else args.confidence_mode
    asset = confidence_asset(mode, getattr(args, 'confidence_config_path', None))[1] if mode != 'manhattan' else None
    signature=dict(confidence_asset=asset,max_invalid_fraction=args.jwst_max_invalid_fraction,
        stride=368,max_tiles=args.max_tiles,large_only=getattr(args,'large_only',False),
        large_limit=getattr(args,'large_limit',0),cross_boundary_only=getattr(args,'cross_boundary_only',False),
        background_root=args.background_root,sex_detect_thresh=args.sex_detect_thresh,
        sex_minarea=args.sex_minarea,sex_back_size=args.sex_back_size,sex_grow=args.sex_grow,
        training_mtime_ns=Path(row['training_fits']).stat().st_mtime_ns,
        reference_mtime_ns=Path(row['reference_fits']).stat().st_mtime_ns,
        catalog_mtime_ns=Path(args.abell_catalog).stat().st_mtime_ns,gaia_mtime_ns=Path(args.abell_gaia).stat().st_mtime_ns,
        confidence_mode=args.confidence_mode,confidence_fwhm_min=args.confidence_fwhm_min,
        confidence_fwhm_max=args.confidence_fwhm_max,confidence_fwhm_pixels=args.confidence_fwhm_pixels,policy_version=7,chunk_tiles=args.chunk_tiles,parent_packed_stamps=True,reference_background_method=AGGRESSIVE_BACKGROUND_METHOD)
    if receipt.exists() and not args.overwrite:
        previous=json.loads(receipt.read_text())
        if previous.get('max_invalid_fraction')!=args.jwst_max_invalid_fraction:
            raise ValueError(f'Existing products have different invalid threshold: {receipt}')
        if previous.get('signature')==signature:
            outputs=previous.get('outputs',[])
            if all(Path(p).exists() and Path(p+'_manifest.json').exists() for p in outputs):return previous
    t=time.monotonic();task=store_task(args,row)
    from .utils.parent_zarr import ParentStoreAssembly
    assembly=ParentStoreAssembly(task,audit/"_parts")
    part_task=assembly.task(task)
    raw,header=fits.getdata(row['training_fits'],header=True)
    valid=np.isfinite(raw);shape=raw.shape
    grid=make_tile_specs(parent_origin=(0,0),image_shape=(shape[1],shape[0]),tile_size=512,stride=368,compare_origin=None)
    grid=valid_tile_specs(grid,valid,(0,0),args.jwst_max_invalid_fraction)
    if valid.sum()<(1-args.jwst_max_invalid_fraction)*512**2:
        result=dict(band=row['band'],patch=row['patch'],status='no_possible_valid_512',grid_samples=0,large_samples=0,
                    max_invalid_fraction=args.jwst_max_invalid_fraction,signature=signature,outputs=[])
        receipt.write_text(json.dumps(result,indent=2)+'\n');return result
    print(f'[abell-zarr] {row["band"]} {row["patch"]}: reference labels',flush=True)
    reference,rheader=fits.getdata(row['reference_fits'],header=True)
    background=ensure_background(Path(row['reference_fits']),audit/'background.npz',
                                 image=reference,header=rheader)
    from .utils.background_tiles import abell_region
    training_sky, background_provenance = abell_region(args.background_root, row['band'],
        row['original_training'], row['x0'], row['y0'], row['size'], row['size'])
    catalog=Table.read(args.abell_catalog,hdu=1);gaia=Table.read(args.abell_gaia)
    labels,summary=classify_reference(reference,rheader,catalog,gaia,background,row['band'],training_background=training_sky)
    del reference,background
    # Keep frozen original geometry and final labels for audits/centered views.
    labels.table['final_class']=labels.label_classes
    labels.table['parent_x']=labels.geom_x;labels.table['parent_y']=labels.geom_y
    labels.table['original_kron_a']=labels.geom_major;labels.table['original_kron_b']=labels.geom_minor
    labels.table.write(audit/'sources.fits',overwrite=True)
    prepared=prepare_image(raw,header=header)
    params=fit_parent_rgb(prepared.image)
    (audit/'scaling.json').write_text(json.dumps(params,indent=2)+'\n')
    scaled=apply_parent_rgb(prepared.image,params)
    provenance=dict(dataset='abell',split_group='Abell2744',parent_id=row['patch'],
        parent_grid=[row['ix'],row['iy']],parent_bounds=[row['x0'],row['y0'],row['x0']+row['size'],row['y0']+row['size']],
        parent_overlap=row['parent_overlap'],image_fits=row['training_fits'],reference_fits=row['reference_fits'],
        reference_catalog=args.abell_catalog,sky_wcs_header=header.tostring(sep='\n'),
        source_selection_reference='full coadd',background_method='SExtractor segmentation on aligned half-coadd parent',
        training_background='sextractor',reference_snr_background=AGGRESSIVE_BACKGROUND_METHOD,
        scaling_statistics='4096 half-coadd parent',sample_kind='grid')
    # Source classes remain those measured from the FULL reference image.
    result=write_classified_patch(part_task,prepared.image,labels,tile_specs=[] if getattr(args,'large_only',False) else grid,valid_mask=valid,
        max_invalid_fraction=args.jwst_max_invalid_fraction,scaled_image_chw=scaled,provenance=provenance,
        image_wcs=WCS(header).celestial)
    assembly.add(result)
    del scaled,prepared,raw
    # One owner per source/band: nearest parent center on the anchored lattice.
    step=row['size']-row['parent_overlap'];anchor=row['grid_anchor']
    owner_x=np.floor((labels.geom_x+row['x0']-anchor[0]-row['size']/2)/step+.5).astype(int)
    owner_y=np.floor((labels.geom_y+row['y0']-anchor[1]-row['size']/2)/step+.5).astype(int)
    extra=large_source_specs(labels,'abell',selected=(owner_x==row['ix'])&(owner_y==row['iy']))
    if getattr(args,'cross_boundary_only',False):
        extra=[s for s in extra if s.x0<0 or s.y0<0 or s.x1>row['size'] or s.y1>row['size']]
    if getattr(args,'large_limit',0):extra=extra[:args.large_limit]
    extra_rows=[]
    with fits.open(row['original_training'],memmap=True) as hd, fits.open(row['original_reference'],memmap=True) as refhd:
        ih=image_hdu(hd);native=WCS(hd[ih].header).celestial;data=hd[ih].data
        ri=image_hdu(refhd);refwcs=WCS(refhd[ri].header).celestial
        for spec in extra:
            # Regenerate ALL labels on an independent 4096 source-centered window.
            # The source's eligibility/owner was established on its full reference parent.
            cx,cy=spec.x0-1792,spec.y0-1792
            ch=header.copy();ch['CRPIX1']-=cx;ch['CRPIX2']-=cy;ch['NAXIS1']=ch['NAXIS2']=4096
            context=reproject_tile(data,native,ch)
            ok=np.isfinite(context);fraction=float(ok[1792:2304,1792:2304].mean())
            meta=dict(name=spec.name,x0=spec.x0+row['x0'],y0=spec.y0+row['y0'],valid_fraction=fraction,status='invalid')
            if fraction<1-args.jwst_max_invalid_fraction:
                extra_rows.append(meta);continue
            rch=ch.copy()
            if 'BUNIT' in refhd[ri].header:rch['BUNIT']=refhd[ri].header['BUNIT']
            refcontext=reproject_tile(refhd[ri].data,refwcs,rch)
            sky,bgmeta=abell_region(args.background_root,row['band'],row['original_training'],
                row['x0']+cx,row['y0']+cy,4096,4096)
            source_id=spec.name.split('_')[1]
            refsky=ensure_background(Path(row['original_reference']),audit/(source_id+'_reference_sky.npz'),
                                    image=refcontext,header=rch)
            lab,_=classify_reference(refcontext,rch,catalog,gaia,refsky,row['band'],training_background=sky)
            extra_task=replace(part_task,patch=row['patch']+'_'+source_id)
            prep=prepare_image(context,header=ch)
            one=TileSpec(spec.name,1792,1792,512,kind='large_source')
            extra_result=write_classified_patch(extra_task,prep.image,lab,tile_specs=[one],valid_mask=ok,
                image_wcs=WCS(ch).celestial,
                max_invalid_fraction=args.jwst_max_invalid_fraction,scaled_image_chw=apply_parent_rgb(prep.image,params),
                provenance={**provenance,'sample_kind':'large_source','large_source_policy':'final clean/weak AND original a>100',
                    'patch':row['patch'], 'labels_generated_independently':True,
                    'label_window':[row['x0']+cx,row['y0']+cy,row['x0']+cx+4096,row['y0']+cy+4096],
                    'background_tiles':abell_region(args.background_root,row['band'],row['original_training'],meta['x0'],meta['y0'],512,512)[1],
                    'stamp_bounds':[meta['x0'],meta['y0'],meta['x0']+512,meta['y0']+512],
                    'sky_wcs_header':ch.tostring(sep='\n')})
            assembly.add(extra_result,shift=(spec.x0-1792,spec.y0-1792))
            meta.update(status='kept',output=extra_result.get('output'));extra_rows.append(meta)
            del context,refcontext,lab,prep,sky,refsky;gc.collect()
    output=assembly.finish()
    sample_cursor=result.get('samples',0)
    for item in extra_rows:
        if item['status']=='kept':
            item.update(output=output,sample_index=sample_cursor);sample_cursor+=1
    (audit/'large_sources.json').write_text(json.dumps(extra_rows,indent=2)+'\n')
    done=dict(band=row['band'],patch=row['patch'],status='complete',grid_samples=result['samples'],
              training_background='sextractor',
              large_samples=sum(r['status']=='kept' for r in extra_rows),labels=summary,
              max_invalid_fraction=args.jwst_max_invalid_fraction,seconds=time.monotonic()-t,signature=signature,
              outputs=[output] if output else [])
    receipt.write_text(json.dumps(done,indent=2)+'\n');gc.collect()
    print(f'[abell-zarr] complete {done}',flush=True)
    return done


def run_abell_zarr(args):
    root=Path(args.output_root)
    if (root/'manifest.json').exists():manifest=json.loads((root/'manifest.json').read_text())
    else:
        # Allows an explicitly selected pilot after that band's cutouts finish.
        if not getattr(args,'abell_parent',None):raise ValueError('Wait for cutout manifest before batch Zarr')
        manifest=dict(rows=[r for p in (root/'manifests').glob('*.json') for r in json.loads(p.read_text())],failures=[])
    if manifest['failures']:raise ValueError('Cutout stage is incomplete')
    rows=[r for r in manifest['rows'] if r['status']=='kept' and (not args.jwst_bands or r['band'] in {b.upper() for b in args.jwst_bands})]
    if getattr(args,'abell_parent',None):rows=[r for r in rows if r['patch'] in args.abell_parent]
    results=[];failures=[]
    with ProcessPoolExecutor(max_workers=args.workers,max_tasks_per_child=1) as pool:
        fs={pool.submit(build_parent,args,r):r for r in rows}
        for f in as_completed(fs):
            try:results.append(f.result())
            except Exception as e:
                failures.append(dict(band=fs[f]['band'],patch=fs[f]['patch'],error=repr(e)))
                import traceback;traceback.print_exc()
    summary=dict(results=results,failures=failures)
    (root/'zarr_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if failures:raise RuntimeError(f'{len(failures)} parent Zarr failures; see zarr_summary.json')
    return summary
