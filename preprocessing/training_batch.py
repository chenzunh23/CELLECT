"""Four-family production Zarr jobs; one process lifetime per parent image.

Only precomputed sky masks are combined across parent boundaries. Centered
stamps get a fresh catalog/segmentation evaluation on a 4096 context window.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields, replace
import gc
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from .dataset_inputs import discover_cosmos, header_info, load_image, image_hdu
from .utils.image_level import StoreTask, attach_cosmos_segmentation
from .utils.inputs import TileSpec, make_tile_specs
from .utils.large_sources import large_source_specs, crop_padded, valid_tile_specs
from .utils.parent_scaling import fit_parent_rgb, apply_parent_rgb
from .utils.background_tiles import cosmos_region, abell_region, read_mask, check_source
from .image_processing import prepare_image


def task_for(args, **kwargs):
    values={f.name:getattr(args,f.name,None) for f in fields(StoreTask)}
    for name in ('data_root','coadd_fits_root','output_root','refit_root','denoised_fits_root','coadd_weight_root','gaia_fits'):
        value=values.get(name)
        values[name]=Path(value) if value else None
    values.update(tile_size=512,stride=368,overwrite=True,max_tiles=args.max_tiles,tract=args.tract,
                  group='',patch='',band='',dataset_source='coadd',confidence_mode='psf-ee')
    values.update(kwargs)
    return StoreTask(**values)


def plan(args):
    rows=[];root=Path(args.background_root)
    if 'cosmos' in args.training_kinds:
        for source in discover_cosmos(args.paths['cosmos'], proposals=args.cosmos_proposals,
                                     bands=args.jwst_bands, pointings=args.cosmos_pointings):
            manifest=root/'cosmos'/f'proposal_{source.proposal}'/f'Pointing_{source.patch}'/source.band/'tiles.json'
            doc=json.loads(manifest.read_text());check_source(doc['source'],source.image_fits)
            height,width=doc['shape_yx']
            # Include low-coverage parents. Empty parents are the only ones omitted.
            occupied={tuple(r['full_cell_xyxy'][:2]) for r in doc['tiles']}
            for y in range(0,height,4096):
                for x in range(0,width,4096):
                    if (x,y) not in occupied:continue
                    if args.parent_xy and (x,y)!=tuple(args.parent_xy):continue
                    name=f'cosmos_p{source.proposal}_p{source.patch}_{source.band}_x{x}_y{y}'
                    rows.append(dict(kind='cosmos',name=name,source=source.to_dict(),x=x,y=y,background=str(manifest)))
    if 'abell' in args.training_kinds:
        for receipt in sorted((root/'abell').glob('*/*/complete.json')):
            doc=json.loads(receipt.read_text());job=doc['job']
            if doc['status']!='done':continue
            if args.jwst_bands and job['band'] not in args.jwst_bands:continue
            if args.abell_parent and job['name'] not in args.abell_parent:continue
            check_source(doc['signature']['source'],job['source'])
            rows.append(dict(kind='abell',name=f'abell_{job["band"]}_{job["name"]}',job=job))
    products=Path(args.hsc_products_root)
    for kind,source_kind in [('hsc_half','half_coadd'),('hsc_noisy','noisy')]:
        if kind not in args.training_kinds:continue
        for path in sorted((products/source_kind/str(args.tract)).glob('*/*/warp*.fits')):
            band,patch=path.parent.parent.name,path.parent.name
            if band not in args.bands:continue
            if args.patches!=['all'] and patch not in args.patches:continue
            group=path.stem.rsplit('-',1)[-1]
            if args.groups!=['all'] and group not in args.groups:continue
            rows.append(dict(kind=kind,name=f'{kind}_{band}_{patch}_{group}',source=str(path),
                             band=band,patch=patch,group=group))
    if args.job_limit:rows=rows[:args.job_limit]
    return rows


def restore_source(row):
    from .dataset_inputs import ImageInput
    d=dict(row)
    for k in ('image_fits','reference_fits','quality_mask_npz','catalog_root'):
        if d.get(k):d[k]=Path(d[k])
    return ImageInput(**d)


def read_window(source, x, y, size):
    """Bounded FITS window, padded without shifting a source at an image edge."""
    full, (h,w), _=header_info(source.image_fits)
    xa,ya=max(0,x),max(0,y);xb,yb=min(w,x+size),min(h,y+size)
    raw=np.full((size,size),np.nan,np.float32)
    valid=np.zeros((size,size),bool)
    if xa<xb and ya<yb:
        loaded=load_image(source,origin=(xa,ya),shape=(yb-ya,xb-xa))
        raw[ya-y:yb-y,xa-x:xb-x]=loaded.image
        valid[ya-y:yb-y,xa-x:xb-x]=~loaded.bad
    raw[~valid]=np.nan
    header=full.copy();header['CRPIX1']-=x;header['CRPIX2']-=y
    header['NAXIS1']=header['NAXIS2']=size
    return raw,valid,header


def cosmos_parent(args,row):
    from .build_image_level_zarr import write_classified_patch
    from .utils.cosmos_labels import load_catalog, classify_window
    source=restore_source(row['source']);px,py=row['x'],row['y']
    root=Path(args.output_root);audit=root/'audit'/row['name'];audit.mkdir(parents=True,exist_ok=True)
    catalog_path=source.catalog_root/'COSMOSWeb_mastercatalog_v1.1.fits'
    catalog=load_catalog(catalog_path,source.band);gaia=Table.read(args.paths['cosmos']['gaia_fits'])
    # A halo provides neighbors and complete segmentation masks across parent edges.
    halo=256;ox,oy=px-halo,py-halo;size=4096+2*halo
    raw,valid,header=read_window(source,ox,oy,size)
    sky,bgmeta=cosmos_region(row['background'],source.image_fits,ox,oy,size,size)
    labels=classify_window(raw,header,catalog,catalog_path,gaia,sky,source.band)
    core=raw[halo:halo+4096,halo:halo+4096]
    if not np.isfinite(core).any():return dict(status='empty',outputs=[])
    prep=prepare_image(raw,header=header)
    params=fit_parent_rgb(prepare_image(core,header=header).image)
    (audit/'scaling.json').write_text(json.dumps(params,indent=2))
    patch=f'P{source.patch}_x{px:05d}_y{py:05d}'
    task=task_for(args,output_root=root/'zarr'/'cosmos'/f'proposal_{source.proposal}',
        tract=f'COSMOS_{source.proposal}',patch=patch,band=source.band,stride=512,
        image_log_a=1000.,bright_log_a=1000.,clip_threshold=5.)
    provenance=dict(dataset='cosmos',proposal=source.proposal,pointing=source.patch,
        split_group=source.split_group,parent_id=f'P{source.patch}',parent_bounds=[px,py,px+4096,py+4096],
        image_fits=str(source.image_fits),quality_mask_npz=str(source.quality_mask_npz),
        sky_wcs_header=header_info(source.image_fits)[0].tostring(sep='\n'),
        source_catalog=str(catalog_path),background_method='precomputed aggressive SExtractor',
        background_manifest=row['background'],scaling_statistics='4096 training parent',
        label_window=[ox,oy,ox+size,oy+size],sample_kind='grid')
    outputs=[];grid_samples=0
    if not args.large_only:
        specs=[TileSpec(f'x{x}_y{y}',x,y,512) for y in range(py,py+4096,512) for x in range(px,px+4096,512)]
        result=write_classified_patch(task,prep.image,labels,origin=(ox,oy),tile_specs=specs,valid_mask=valid,
            max_invalid_fraction=args.jwst_max_invalid_fraction,scaled_image_chw=apply_parent_rgb(prep.image,params),
            cosmos_catalog=catalog_path,image_wcs=WCS(header_info(source.image_fits)[0]).celestial,provenance=provenance)
        if result.get('output'):outputs.append(result['output'])
        grid_samples=result['samples']
    owned=(labels.geom_x>=halo)&(labels.geom_x<halo+4096)&(labels.geom_y>=halo)&(labels.geom_y<halo+4096)
    specs=large_source_specs(labels,'cosmos',origin=(ox,oy),selected=owned)
    if args.cross_boundary_only:
        specs=[s for s in specs if s.x0<px or s.y0<py or s.x1>px+4096 or s.y1>py+4096]
    if args.large_limit:specs=specs[:args.large_limit]
    rows=[]
    del prep,raw,sky,labels;gc.collect()
    for spec in specs:
        # Fresh, source-centered context: all labels and segmentation regenerated.
        cx,cy=spec.x0-1792,spec.y0-1792
        raw,valid,ch=read_window(source,cx,cy,4096)
        if np.count_nonzero(valid[1792:2304,1792:2304])<(1-args.jwst_max_invalid_fraction)*512**2:continue
        sky,bgmeta=cosmos_region(row['background'],source.image_fits,cx,cy,4096,4096)
        lab=classify_window(raw,ch,catalog,catalog_path,gaia,sky,source.band)
        stask=replace(task,patch=patch+'_'+spec.name.split('_')[1])
        metadata={**provenance,'sample_kind':'large_source','stamp_bounds':[spec.x0,spec.y0,spec.x1,spec.y1],
            'label_window':[cx,cy,cx+4096,cy+4096], 'labels_generated_independently':True,
            'background_tiles':cosmos_region(row['background'],source.image_fits,spec.x0,spec.y0,512,512)[1],
            'large_source_policy':'final clean and original Kron a > 100 pixels'}
        prepared=prepare_image(raw,header=ch)
        result=write_classified_patch(stask,prepared.image,lab,origin=(cx,cy),tile_specs=[spec],valid_mask=valid,
            max_invalid_fraction=args.jwst_max_invalid_fraction,scaled_image_chw=apply_parent_rgb(prepared.image,params),
            cosmos_catalog=catalog_path,image_wcs=WCS(header_info(source.image_fits)[0]).celestial,provenance=metadata)
        if result.get('output'):
            outputs.append(result['output'])
            info=dict(name=spec.name,output=result['output'],bounds=metadata['stamp_bounds'],background=metadata['background_tiles'])
            if args.diagnostic_dir:
                diagnostic(args,row['name'],raw,lab,spec,(cx,cy),task,source,info)
            rows.append(info)
        del raw,valid,sky,lab,prepared;gc.collect()
    (audit/'large_sources.json').write_text(json.dumps(rows,indent=2))
    return dict(status='complete',grid_samples=grid_samples,large_samples=len(rows),outputs=outputs)


def diagnostic(args,name,raw,labels,spec,origin,task,source,info):
    """Inspect the exact labels and reread saved Zarr, including neighbor sources."""
    from .utils.image_level import _tile_targets
    from .utils.confidence import resolve_confidence
    from .labels import SourceClass
    import zarr
    config=resolve_confidence(task,image_wcs=WCS(header_info(source.image_fits)[0]).celestial)
    target=_tile_targets(labels,spec,origin,confidence=config)
    stamp=crop_padded(raw,spec.x0,spec.y0,origin,512,fill=np.nan)
    z=zarr.open_group(info['output'],mode='r')
    # Saved arrays, not a separate visualization-only label implementation.
    saved=z['band_pu_class_mask'][0,0]
    info['zarr_array_keys']=list(z.array_keys())
    if saved is not None:info['saved_dense_equal']=bool(np.array_equal(saved,target['pu']))
    info['saved_confidence_equal']=bool(np.array_equal(z['band_confidence'][0,0],target['confidence']))
    info['saved_source_ids_equal']=bool(np.array_equal(z['source_ids'][:],target['source_ids']))
    info['saved_source_centers_equal']=bool(np.array_equal(z['source_centers'][:],target['source_centers']))
    sx=labels.geom_x+origin[0]-spec.x0;sy=labels.geom_y+origin[1]-spec.y0
    ids=np.flatnonzero((sx>=0)&(sy>=0)&(sx<512)&(sy<512)&(labels.label_classes!=SourceClass.DROPPED))
    geometry=np.column_stack([sx[ids],sy[ids],labels.geom_major[ids],labels.geom_minor[ids],
                              labels.geom_theta[ids],labels.label_classes[ids]])
    from .utils.diagnostic_overlay import plot_diagnostic
    dest=Path(args.diagnostic_dir);dest.mkdir(parents=True,exist_ok=True)
    plot_diagnostic(dest/f'{name}_{spec.name}.png',name,spec.name,stamp,
                    saved,z['band_confidence'][0,0],geometry,[spec.x0,spec.y0,spec.x1,spec.y1])
    info.update(source_count=len(ids),dense_counts={str(k):int(v) for k,v in zip(*np.unique(target['pu'],return_counts=True))},
        confidence_counts={str(k):int(v) for k,v in zip(*np.unique(target['confidence'],return_counts=True))})
    (dest/f'{name}_{spec.name}.json').write_text(json.dumps(info,indent=2))


def hsc_job(args,row):
    from .build_image_level_zarr import _classify_patch,write_classified_patch
    from .utils.image_level import _coadd_image_path,_read_image_header_origin,_read_fits_quality_mask
    from .utils.jwst_background_sextractor import aggressive_sextractor_background,AggressiveSkyConfig
    from .utils.hsc_psf_ee import measure_psf
    root=Path(args.output_root);band=row['band'];patch=row['patch'];path=Path(row['source'])
    source_kind='half_coadd' if row['kind']=='hsc_half' else 'noisy'
    reference=_coadd_image_path(Path(args.data_root),band,args.tract,patch)
    task=task_for(args,output_root=root/'zarr'/'hsc',coadd_fits_root=Path(args.data_root),
        denoised_fits_root=Path(args.hsc_products_root),band=band,patch=patch,group=row['group'],dataset_source=source_kind)
    image,header,origin=_read_image_header_origin(path)
    if source_kind=='half_coadd':
        dest=root/'background'/'hsc_half'/band/patch/path.stem/'background_mask.npz'
        sky=aggressive_sextractor_background(image,header,dest,source=str(path),
            config=AggressiveSkyConfig(thresholds_sigma=(2.5,3.,3.5),grow_native_pixels=(4,10,20)),
            scratch_dir=root/'_sex_scratch')
        bgpath=dest
    else:
        folder=Path(args.background_root)/'hsc'/'noisy'/band/patch/path.stem
        receipt=json.loads((folder/'complete.json').read_text())
        check_source(receipt['signature']['source'],path)
        bgpath=folder/'background_mask.npz';sky=read_mask(bgpath)
    psfpath=root/'psf_ee'/band/f'{patch}.json'
    measure_psf(reference,psfpath)
    task=replace(task,confidence_config_path=str(psfpath))
    labels=_classify_patch(task,image,header,origin,reference,background_override=sky)
    valid=np.isfinite(image)&~_read_fits_quality_mask(path,image.shape)
    return write_classified_patch(task,image,labels,origin,valid_mask=valid,max_invalid_fraction=.10,
        provenance=dict(dataset='hsc',image_fits=str(path),reference_fits=str(reference),
            background_mask=str(bgpath),background_method='aggressive SExtractor on matching training input',
            split_group=f'HSC_{args.tract}_{patch}'))


def abell_job(args,row):
    from .abell_zarr import build_parent
    from .utils.sky_tiles import tile_header,SkyTile,reproject_tile,write_tile
    job=row['job'];tile=job['tile'];band=job['band'];root=Path(args.output_root)/'abell'
    grid,_,_=header_info(job['grid_reference'])
    # Catalog selection uses the FULL coadd, aligned to the half-coadd parent WCS.
    from .dataset_inputs import discover_abell
    source=next(s for s in discover_abell(args.paths['abell'],bands=[band]))
    h,_,_=header_info(source.reference_fits)
    rh=tile_header(grid,SkyTile(**tile),h)
    reference=root/'references'/band/(job['name']+'_full.fits')
    if not reference.exists():
        with fits.open(source.reference_fits,memmap=True) as hd:
            values=reproject_tile(hd[image_hdu(hd)].data,WCS(h).celestial,rh)
        write_tile(reference,values,rh,source=source.reference_fits)
    from copy import copy
    aa=copy(args);aa.output_root=str(root);aa.confidence_mode='psf-ee'
    aa.zarr_output_root=str(Path(args.output_root)/'zarr'/'abell')
    aa.overwrite=True  # The outer receipt validates code and inputs for this runner.
    r=dict(tile,band=band,patch=job['name'],training_fits=str(Path(job['output'])/'training_half_4096.fits'),
        reference_fits=str(reference),original_training=str(source.image_fits),original_reference=str(source.reference_fits),
        grid_anchor=args.abell_anchor,parent_overlap=128)
    return build_parent(aa,r)


def run_job(args,row):
    start=time.monotonic();root=Path(args.output_root);receipt=root/'receipts'/(row['name']+'.json')
    # Signature tracks inputs/settings and all preprocessing code/assets, not just output existence.
    import hashlib
    digest=hashlib.sha256()
    for p in sorted(Path(__file__).parent.rglob('*.py'))+sorted((Path(__file__).parent/'assets').glob('*.json')):
        if 'tests' not in p.parts:digest.update(p.read_bytes())
    paths=[]
    if row['kind']=='cosmos':
        paths=[row['source']['image_fits'],row['source']['quality_mask_npz'],row['background'],
               str(Path(row['source']['catalog_root'])/'COSMOSWeb_mastercatalog_v1.1.fits'),args.paths['cosmos']['gaia_fits']]
    elif row['kind']=='abell':
        job=row['job']
        paths=[job['source'],job['grid_reference'],str(Path(job['output'])/'background_mask.npz'),args.abell_catalog,args.abell_gaia]
    else:
        from .utils.image_level import _coadd_image_path,_refit_csv_path
        from .utils.inputs import _band_catalog_path
        paths=[row['source'],str(_coadd_image_path(Path(args.data_root),row['band'],args.tract,row['patch'])),
               str(_refit_csv_path(Path(args.refit_root),args.tract,row['band'],row['patch'])),
               str(_band_catalog_path(Path(args.data_root),row['band'],args.tract,row['patch'])),args.gaia_fits]
        if row['kind']=='hsc_noisy':
            paths.append(str(Path(args.background_root)/'hsc/noisy'/row['band']/row['patch']/Path(row['source']).stem/'background_mask.npz'))
    stamps=[dict(path=str(Path(p).resolve()),bytes=Path(p).stat().st_size,mtime_ns=Path(p).stat().st_mtime_ns) for p in paths if p]
    signature=dict(job=row,inputs=stamps,code=digest.hexdigest(),max_tiles=args.max_tiles,large_only=args.large_only,
        cross_boundary_only=args.cross_boundary_only,large_limit=args.large_limit,
        max_invalid_fraction=args.jwst_max_invalid_fraction)
    if receipt.exists() and not args.overwrite:
        previous=json.loads(receipt.read_text())
        if previous.get('signature')==signature and all(Path(p).exists() and Path(p+'_manifest.json').exists() for p in previous['result'].get('outputs',[])):
            return dict(name=row['name'],status='cached')
    print(f'[start] {row["name"]}',flush=True)
    result={'cosmos':cosmos_parent,'abell':abell_job,'hsc_half':hsc_job,'hsc_noisy':hsc_job}[row['kind']](args,row)
    if result.get('output'):result['outputs']=[result['output']]
    result.update(name=row['name'],seconds=time.monotonic()-start)
    receipt.parent.mkdir(parents=True,exist_ok=True)
    receipt.write_text(json.dumps(dict(signature=signature,result=result),indent=2))
    print(f'[done] {row["name"]} {result["seconds"]:.1f}s',flush=True);gc.collect()
    return result


def run(args):
    if args.tile_size!=512:raise ValueError('This training recipe requires 512-pixel stamps')
    root=Path(args.output_root);root.mkdir(parents=True,exist_ok=True)
    rows=plan(args);(root/'queue.json').write_text(json.dumps(rows,indent=2))
    print(f'{len(rows)} jobs; kinds {sorted(set(r["kind"] for r in rows))}',flush=True)
    if args.plan_only:return 0
    results=[];failures=[]
    # Spawn + recycle isolates large arrays and FITS/Zarr caches after every parent.
    with ProcessPoolExecutor(max_workers=args.workers,max_tasks_per_child=1) as pool:
        fs={pool.submit(run_job,args,row):row for row in rows}
        for f in as_completed(fs):
            try:results.append(f.result())
            except Exception as exc:
                failures.append(dict(name=fs[f]['name'],error=repr(exc)));traceback.print_exc()
            (root/'summary.json').write_text(json.dumps(dict(results=results,failures=failures),indent=2))
    return 1 if failures else 0
