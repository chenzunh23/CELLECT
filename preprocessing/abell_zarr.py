"""Abell full-reference labels and independent source-centered large stamps."""
from concurrent.futures import ProcessPoolExecutor
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
from .image_processing import prepare_image, jwst_bright_config
from .labels import DenseLabel
from .snr import thresholds_from_args, snr_policy_from_args
from .dataset_inputs import header_info, image_hdu
from .utils.sextractor_background import SExtractorBackgroundConfig, sextractor_background
from .utils.jwst_background_sextractor import aggressive_sextractor_background, METHOD as AGGRESSIVE_BACKGROUND_METHOD
from .abell_cutouts import progress, configure_progress_warnings, completed_tasks


def uncover_signature(args, band):
    """Settings/assets participate in both direct and mixed-run receipts."""
    if not getattr(args,'use_uncover',False):return dict(use_uncover=False)
    from .utils.uncover_labels import model_for_band,POLICY_VERSION
    paths=[Path(args.uncover_catalog),Path(args.uncover_detection),Path(args.uncover_segmentation),
           model_for_band(args.uncover_model_root,band)]
    compact_root=getattr(args,'uncover_compact_root',None)
    blue_catalog=getattr(args,'uncover_blue_catalog',None)
    if blue_catalog:paths.append(Path(blue_catalog))
    compact_files=[] if not compact_root else sorted(Path(compact_root).glob(f'*/{band}/blue_center_selection.fits'))
    return dict(use_uncover=True,policy=POLICY_VERSION,snr_thresholds=snr_policy_from_args(args,'abell'),
        assets=[dict(path=str(p.resolve()),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in paths],
        no_kron_snr=args.uncover_no_kron_snr,risk_no_kron_snr=args.uncover_risk_no_kron_snr,
        risk_bright_mag=args.uncover_risk_bright_mag,risk_sigma=args.uncover_risk_sigma,
        reject_untrusted=args.uncover_reject_untrusted,
        kron_other_fraction_max=getattr(args,'uncover_kron_other_fraction',.30),
        halo_no_kron=dict(enabled=True,region='current-band official bCG model risk',
            order='ignore native missing-Kron rows before BLUE dedup/insertion',preserve_independent_blue_gaia=True),
        native_psf_review=dict(enabled=getattr(args,'uncover_native_psf_review',True),
            snr_min=getattr(args,'uncover_native_psf_snr',4.),
            moment_max_arcsec=getattr(args,'uncover_native_psf_moment_max',.10)),
        compact_centers=dict(root=compact_root,red_min_snr=getattr(args,'uncover_compact_red_snr',4.),
            blue_catalog=str(blue_catalog) if blue_catalog else None,
            dedup_pixels=getattr(args,'uncover_compact_dedup_pixels',2.),
            assets=[dict(path=str(p.resolve()),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in compact_files]),
        model_use='current-band risk region only; raw FULL flux/noise/trust; HALF input/bright/Gaia unchanged',
        coverage_gate='fraction((not observed) OR (not finite native detection)) per 512 stamp',
        measurement_background='original raw FULL multiscale SExtractor sky minus HALF bright/Gaia footprint; no residual background')


def uncover_labels_for_window(args, reference, rheader, catalog, gaia, background, band,
                              training_sky, prepared, header, audit, prefix='reference'):
    """Share identical native policy for grid and independent centered stamps."""
    from .utils.uncover_labels import (project_product,coverage_on_grid,model_for_band,
        classify_reference as classify_uncover,UncoverSnrConfig)
    from .utils.image import hsc_surface_brightness_factor
    tag=f'{band}/{Path(audit).name}/{prefix}'
    progress(f'[abell-uncover] {tag} stage=project_coverage_and_segmentation')
    coverage=coverage_on_grid(args.uncover_detection,header,reference.shape,
                              cache_root=Path(args.output_root)/'uncover_coverage')
    segments,native_header=project_product(args.uncover_segmentation,header,reference.shape,mode='labels')
    modelpath=model_for_band(args.uncover_model_root,band)
    progress(f'[abell-uncover] {tag} stage=project_band_model file={modelpath.name}')
    model,mh=project_product(modelpath,header,reference.shape,mode='surface')
    from .dataset_inputs import effective_band
    if 'FILTER' in mh and effective_band(mh)!=band.upper():raise ValueError(f'Model FILTER/PUPIL does not match {band}: {modelpath}')
    # Match units for the risk comparison only. The classifier measures the
    # original FULL once, with its existing bright/Gaia exclusions and sky.
    factor=hsc_surface_brightness_factor({'BUNIT':'MJy/sr'})[0]
    normalized_model=model*factor
    progress(f'[abell-uncover] {tag} stage=raw_FULL_aperture_SNR; model=risk_only; no subtraction')
    config=UncoverSnrConfig(args.uncover_no_kron_snr,args.uncover_risk_no_kron_snr,
        args.uncover_reject_untrusted,args.uncover_risk_bright_mag,
        thresholds=thresholds_from_args(args, 'abell'),
        kron_other_fraction_max=getattr(args,'uncover_kron_other_fraction',.30),
        native_psf_review=getattr(args,'uncover_native_psf_review',True),
        native_psf_snr_min=getattr(args,'uncover_native_psf_snr',4.),
        native_psf_moment_max_arcsec=getattr(args,'uncover_native_psf_moment_max',.10))
    labels,summary=classify_uncover(reference,rheader,catalog,gaia,background,band,
        native_header=native_header,training_background=training_sky,
        bright_image=prepared,bright_header=header,native_segmentation=segments,
        bright_config=jwst_bright_config(args),
        snr_config=config,snr_risk_model=normalized_model,
        snr_risk_sigma=args.uncover_risk_sigma,snr_risk_coverage=coverage,
        progress=lambda stage,n=0:print(f'[abell-uncover] {band} {prefix}: {stage} n={n}',flush=True))
    compact_root=getattr(args,'uncover_compact_root',None)
    candidates=None;compact_path=None;ready_measurement=False
    rms=None;noise_factor=None;noise_audit={}
    if compact_root:
        parent=Path(audit).name
        compact_path=Path(compact_root)/parent/band/'blue_center_selection.fits'
        if compact_path.is_file():
            candidates=Table.read(compact_path)
            if candidates.meta.get('PSFBAND')!=band:
                raise ValueError(f'Compact measurement band does not match {band}: {compact_path}')
            ready_measurement=True
            # Project RA/Dec into this exact window, including large centered
            # stamps; never reuse the parent-local x/y in another cutout.
            candidates['x'],candidates['y']=WCS(header).celestial.world_to_pixel_values(candidates['ra'],candidates['dec'])
        else:
            progress(f'[abell-uncover] {tag} no premeasured compact catalog; try verified sky catalog')
    blue_catalog=getattr(args,'uncover_blue_catalog',None)
    if candidates is None and blue_catalog:
        from .utils.uncover_labels import blue_candidates_on_window,direct_blue_detection
        compact_path=Path(blue_catalog)
        candidates=blue_candidates_on_window(compact_path,header,reference.shape)
        candidates['fixed_psf_core_snr']=np.full(len(candidates),np.nan)
        progress(f'[abell-uncover] {tag} stage=verified_BLUE_projection candidates={len(candidates)} catalog={compact_path}')
        needs_psf=np.asarray(candidates['blue_candidate'],bool)&~direct_blue_detection(candidates,band)
        progress(f'[abell-uncover] {tag} BLUE_band_gate direct_current_band={int((~needs_psf).sum())} '
            f'fixed_PSF_required={int(needs_psf.sum())} threshold={getattr(args,"uncover_compact_red_snr",4.)}')
        if needs_psf.any():
            from .utils.jwst_star_psf import empirical_core_rms,calibrate_core_noise,measure_fixed_psf_catalog
            progress(f'[abell-uncover] {tag} stage=BLUE_current_band_fixed_PSF')
            sky=np.asarray(background,bool)&(segments==0)&np.isfinite(reference)
            rms=empirical_core_rms(reference,sky)
            noise_factor,noise_audit=calibrate_core_noise(reference,rms,band,sky)
            snr,measurements=measure_fixed_psf_catalog(reference,rms,band,candidates,
                noise_factor=noise_factor,selection=needs_psf,
                progress=lambda stage,n:progress(f'[abell-uncover] {tag} {stage} n={n}'))
            candidates['fixed_psf_core_snr']=snr
            measurements.write(Path(audit)/(prefix+'_blue_psf_measurements.fits'),overwrite=True)
            candidates.meta['NOISEFAC']=noise_factor
    if candidates is not None:
        candidates.meta.update(PSFBAND=band,PSFREF='RAW_FULL',PSFMODE='premeasured' if ready_measurement else 'current_window',
            DEDUP=getattr(args,'uncover_compact_dedup_pixels',2.))
    if config.native_psf_review or candidates is not None:
        from .utils.uncover_labels import refine_uncover_compact_sources
        progress(f'[abell-uncover] {tag} stage=native_compact_PSF_review_and_BLUE_recenter')
        updated=refine_uncover_compact_sources(labels,reference,rheader,band,config=config,
            candidates=candidates,core_snr=None if candidates is None else candidates['fixed_psf_core_snr'],
            red_min_snr=getattr(args,'uncover_compact_red_snr',4.),
            dedup_pixels=getattr(args,'uncover_compact_dedup_pixels',2.),scope=Path(audit).name,
            sky_mask=np.asarray(background,bool)&(segments==0),
            rms=rms,noise_factor=noise_factor,
            native_pixel_scale=float(np.sqrt(abs(np.linalg.det(WCS(native_header).celestial.pixel_scale_matrix)))*3600),
            progress=lambda stage,n:progress(f'[abell-uncover] {tag} {stage} n={n}'))
        summary.update(updated)
        if noise_audit:summary['native_psf_review']['noise_audit']=noise_audit
        if candidates is not None:
            compact=labels.compact_center_audit
            summary['compact_centers'].update(path=str(compact_path),premeasured=ready_measurement,
                candidates_in_window=len(candidates))
            candidates['supplement_status']=compact['status']
            candidates['supplement_source_id']=compact['source_id']
            candidates['matched_native_index']=compact['matched_native_index']
            candidates.write(Path(audit)/(prefix+'_blue_center_selection.fits'),overwrite=True)
        Table(dict(source_id=labels.strict_ids,x=labels.strict_x,y=labels.strict_y,
            is_gaia=labels.strict_is_gaia)).write(Path(audit)/(prefix+'_extra_strict_centers.fits'),overwrite=True)
        progress(f'[abell-uncover] {tag} END postprocess '
            f'PSF_tested={updated["native_psf_review"]["tested"]} PSF_ignored={updated["native_psf_review"]["ignored"]} '
            f'BLUE_recentered={0 if candidates is None else compact["recentered"]} '
            f'BLUE_added={0 if candidates is None else compact["added"]}')
    summary.update(use_uncover=True,model=str(modelpath),model_band=band,
        model_use='risk region only',model_subtraction='none',
        measurement_reference='original raw FULL',model_noise='original raw FULL blank-aperture noise and trust',
        coverage_fraction=float(coverage.mean()),
        current_band_photometry=f'f_{band.lower()}/e_{band.lower()}',
        model_to_prepared_factor=float(factor),
        measurement_flux_unit='sum of ZP27 surface brightness per 0.168-arcsec reference pixel')
    (Path(audit)/(prefix+'_uncover.json')).write_text(json.dumps(summary,indent=2,default=str)+'\n')
    return labels,summary,coverage


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
    configure_progress_warnings();started=time.monotonic()
    tag=f'{row["band"]}/{row["patch"]}'
    progress(f'[abell-zarr] START parent={tag}')
    try:
        result=_build_parent(args,row)
    except Exception as exc:
        progress(f'[abell-zarr] FAILED parent={tag} elapsed={time.monotonic()-started:.1f}s error={exc!r}')
        raise
    progress(f'[abell-zarr] END parent={tag} status={result.get("status")} '
             f'grid_samples={result.get("grid_samples",0)} large_samples={result.get("large_samples",0)} '
             f'elapsed={time.monotonic()-started:.1f}s')
    return result


def _build_parent(args, row):
    from .build_image_level_zarr import write_classified_patch
    root=Path(args.output_root);audit=root/'labels'/row['band']/row['patch'];audit.mkdir(parents=True,exist_ok=True)
    receipt=audit/'complete.json'
    from .utils.confidence import confidence_asset
    mode = 'psf-ee' if args.confidence_mode == 'auto' else args.confidence_mode
    asset = confidence_asset(mode, getattr(args, 'confidence_config_path', None))[1] if mode != 'manhattan' else None
    from .utils.source_weights import source_area_weight_policy
    from .utils.jwst_star_psf import star_policy
    use_uncover=getattr(args,'use_uncover',False)
    catalog_path=args.uncover_catalog if use_uncover else args.abell_catalog
    from .image_processing import generation_scaling
    signature=dict(scaling=generation_scaling(args, image_log_a=1000., bright_log_a=1000.),
        snr_thresholds=snr_policy_from_args(args,'abell'),uncover=uncover_signature(args,row['band']),jwst_star_protection=star_policy(),source_area_weight_policy=source_area_weight_policy(),confidence_asset=asset,max_invalid_fraction=args.jwst_max_invalid_fraction,
        stride=368,max_tiles=args.max_tiles,large_only=getattr(args,'large_only',False),
        large_limit=getattr(args,'large_limit',0),cross_boundary_only=getattr(args,'cross_boundary_only',False),
        background_root=args.background_root,sex_detect_thresh=args.sex_detect_thresh,
        sex_minarea=args.sex_minarea,sex_back_size=args.sex_back_size,sex_grow=args.sex_grow,
        training_mtime_ns=Path(row['training_fits']).stat().st_mtime_ns,
        reference_mtime_ns=Path(row['reference_fits']).stat().st_mtime_ns,
        catalog_mtime_ns=Path(catalog_path).stat().st_mtime_ns,gaia_mtime_ns=Path(args.abell_gaia).stat().st_mtime_ns,
        confidence_mode=args.confidence_mode,confidence_fwhm_min=args.confidence_fwhm_min,
        confidence_fwhm_max=args.confidence_fwhm_max,confidence_fwhm_pixels=args.confidence_fwhm_pixels,policy_version=13,
        bright_source='training_half_coadd',
        nan_fill_policy='enclosed_source_ring_only_v2',
        nan_input_policy='repaired_context_positive_centers_v1',scaling_nan_policy='finite_values_at_every_transform_v1',
        abell_ignore_flags=[16,17,18,19],ignore_child_parent_class='strict_center_only',chunk_tiles=args.chunk_tiles,parent_packed_stamps=True,reference_background_method=AGGRESSIVE_BACKGROUND_METHOD)
    if receipt.exists() and not args.overwrite:
        previous=json.loads(receipt.read_text())
        if previous.get('max_invalid_fraction')!=args.jwst_max_invalid_fraction:
            raise ValueError(f'Existing products have different invalid threshold: {receipt}')
        if previous.get('signature')==signature:
            outputs=previous.get('outputs',[])
            if all(Path(p).exists() and Path(p+'_manifest.json').exists() for p in outputs):
                progress(f'[abell-zarr] CACHE_HIT parent={row["band"]}/{row["patch"]}')
                return previous
    t=time.monotonic();task=store_task(args,row)
    from .utils.parent_zarr import ParentStoreAssembly
    assembly=ParentStoreAssembly(task,audit/"_parts")
    part_task=assembly.task(task)
    progress(f'[abell-zarr] {row["band"]}/{row["patch"]} stage=read_HALF_and_coverage')
    raw,header=fits.getdata(row['training_fits'],header=True)
    valid=np.isfinite(raw);shape=raw.shape;domain=None
    if use_uncover:
        from .utils.uncover_labels import coverage_on_grid,training_coverage
        domain=coverage_on_grid(args.uncover_detection,header,shape,cache_root=root/'uncover_coverage')
        valid=training_coverage(valid,domain)
    grid=make_tile_specs(parent_origin=(0,0),image_shape=(shape[1],shape[0]),tile_size=512,stride=368,compare_origin=None)
    grid=valid_tile_specs(grid,valid,(0,0),args.jwst_max_invalid_fraction)
    if valid.sum()<(1-args.jwst_max_invalid_fraction)*512**2:
        result=dict(band=row['band'],patch=row['patch'],status='no_possible_valid_512',grid_samples=0,large_samples=0,
                    max_invalid_fraction=args.jwst_max_invalid_fraction,signature=signature,outputs=[])
        assembly.finish()
        receipt.write_text(json.dumps(result,indent=2)+'\n');return result
    print(f'[abell-zarr] {row["band"]} {row["patch"]}: reference labels',flush=True)
    progress(f'[abell-zarr] {row["band"]}/{row["patch"]} stage=read_FULL_and_reference_background')
    reference,rheader=fits.getdata(row['reference_fits'],header=True)
    background=ensure_background(Path(row['reference_fits']),audit/'reference_snr_background.npz',
                                 image=reference,header=rheader)
    from .utils.background_tiles import abell_region
    progress(f'[abell-zarr] {row["band"]}/{row["patch"]} stage=training_background')
    training_sky, background_provenance = abell_region(args.background_root, row['band'],
        row['original_training'], row['x0'], row['y0'], row['size'], row['size'])
    catalog=Table.read(catalog_path,hdu=1);gaia=Table.read(args.abell_gaia)
    prepared=prepare_image(raw,header=header)
    if use_uncover:
        labels,summary,domain=uncover_labels_for_window(args,reference,rheader,catalog,gaia,background,
            row['band'],training_sky,prepared,header,audit)
    else:
        labels,summary=classify_reference(reference,rheader,catalog,gaia,background,row['band'],training_background=training_sky,
                                          bright_image=prepared,bright_header=header,
                                          snr_thresholds=thresholds_from_args(args,'abell'),
                                          bright_config=jwst_bright_config(args))
    del reference,background
    # Keep frozen original geometry and final labels for audits/centered views.
    labels.table['final_class']=labels.label_classes
    labels.table['parent_x']=labels.geom_x;labels.table['parent_y']=labels.geom_y
    labels.table['original_kron_a']=labels.geom_major;labels.table['original_kron_b']=labels.geom_minor
    labels.table.write(audit/'sources.fits',overwrite=True)
    progress(f'[abell-zarr] {row["band"]}/{row["patch"]} stage=scaling_and_grid_Zarr candidates={len(grid)}')
    params=fit_parent_rgb(prepared.image,q=task.image_lupton_q,stretch=task.image_lupton_stretch,
                          high_percentile=task.image_log_high_percentile)
    (audit/'scaling.json').write_text(json.dumps(params,indent=2)+'\n')
    scaled=apply_parent_rgb(prepared.image,params)
    provenance=dict(dataset='abell',snr_thresholds=snr_policy_from_args(args,'abell'),split_group='Abell2744',parent_id=row['patch'],
        parent_grid=[row['ix'],row['iy']],parent_bounds=[row['x0'],row['y0'],row['x0']+row['size'],row['y0']+row['size']],
        parent_overlap=row['parent_overlap'],image_fits=row['training_fits'],reference_fits=row['reference_fits'],
        reference_catalog=catalog_path,sky_wcs_header=header.tostring(sep='\n'),
        use_uncover=use_uncover,uncover_policy=signature['uncover'],
        source_selection_reference='full coadd',background_method='SExtractor segmentation on aligned half-coadd parent',
        bright_source='training_half_coadd',bright_image_fits=row['training_fits'],
        training_background='sextractor',reference_snr_background=AGGRESSIVE_BACKGROUND_METHOD,
        scaling_statistics='4096 half-coadd parent',sample_kind='grid')
    # Ordinary-source measurements use FULL; bright decisions use training HALF.
    result=write_classified_patch(part_task,prepared.image,labels,tile_specs=[] if getattr(args,'large_only',False) else grid,valid_mask=valid,
        input_valid_mask=prepared.input_valid_mask,supervision_mask=domain,
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
    progress(f'[abell-zarr] {row["band"]}/{row["patch"]} stage=independent_large_sources candidates={len(extra)}')
    with fits.open(row['original_training'],memmap=True) as hd, fits.open(row['original_reference'],memmap=True) as refhd:
        ih=image_hdu(hd);native=WCS(hd[ih].header).celestial;data=hd[ih].data
        ri=image_hdu(refhd);refwcs=WCS(refhd[ri].header).celestial
        for index,spec in enumerate(extra,1):
            progress(f'[abell-zarr] {row["band"]}/{row["patch"]} large_source={spec.name} index={index}/{len(extra)}')
            # Regenerate ALL labels on an independent 4096 source-centered window.
            # The source's eligibility/owner was established on its full reference parent.
            cx,cy=spec.x0-1792,spec.y0-1792
            ch=header.copy();ch['CRPIX1']-=cx;ch['CRPIX2']-=cy;ch['NAXIS1']=ch['NAXIS2']=4096
            context=reproject_tile(data,native,ch)
            ok=np.isfinite(context);center_domain=None
            if use_uncover:
                center_domain=coverage_on_grid(args.uncover_detection,ch,context.shape,cache_root=root/'uncover_coverage')
                ok=training_coverage(ok,center_domain)
            fraction=float(ok[1792:2304,1792:2304].mean())
            meta=dict(name=spec.name,x0=spec.x0+row['x0'],y0=spec.y0+row['y0'],valid_fraction=fraction,status='invalid')
            if fraction<1-args.jwst_max_invalid_fraction:
                extra_rows.append(meta);continue
            rch=ch.copy()
            rch['ORIGFILE']=str(row['original_reference'])
            if 'BUNIT' in refhd[ri].header:rch['BUNIT']=refhd[ri].header['BUNIT']
            refcontext=reproject_tile(refhd[ri].data,refwcs,rch)
            sky,bgmeta=abell_region(args.background_root,row['band'],row['original_training'],
                row['x0']+cx,row['y0']+cy,4096,4096)
            source_id=spec.name.split('_')[1]
            refsky=ensure_background(Path(row['original_reference']),audit/(source_id+'_reference_sky.npz'),
                                    image=refcontext,header=rch)
            prep=prepare_image(context,header=ch)
            if use_uncover:
                lab,_,center_domain=uncover_labels_for_window(args,refcontext,rch,catalog,gaia,refsky,
                    row['band'],sky,prep,ch,audit,prefix=source_id)
            else:
                lab,_=classify_reference(refcontext,rch,catalog,gaia,refsky,row['band'],training_background=sky,
                                         bright_image=prep,bright_header=ch,
                                         snr_thresholds=thresholds_from_args(args,'abell'),
                                         bright_config=jwst_bright_config(args))
            extra_task=replace(part_task,patch=row['patch']+'_'+source_id)
            one=TileSpec(spec.name,1792,1792,512,kind='large_source')
            extra_result=write_classified_patch(extra_task,prep.image,lab,tile_specs=[one],valid_mask=ok,
                input_valid_mask=prep.input_valid_mask,supervision_mask=center_domain,
                image_wcs=WCS(ch).celestial,
                max_invalid_fraction=args.jwst_max_invalid_fraction,scaled_image_chw=apply_parent_rgb(prep.image,params),
                provenance={**provenance,'sample_kind':'large_source','large_source_policy':'final clean/weak AND original a>100',
                    'patch':row['patch'], 'labels_generated_independently':True,
                    'bright_image_fits':row['original_training'], 'bright_window_origin':[cx,cy],
                    'label_window':[row['x0']+cx,row['y0']+cy,row['x0']+cx+4096,row['y0']+cy+4096],
                    'background_tiles':abell_region(args.background_root,row['band'],row['original_training'],meta['x0'],meta['y0'],512,512)[1],
                    'stamp_bounds':[meta['x0'],meta['y0'],meta['x0']+512,meta['y0']+512],
                    'sky_wcs_header':ch.tostring(sep='\n')})
            assembly.add(extra_result,shift=(spec.x0-1792,spec.y0-1792))
            meta.update(status='kept',output=extra_result.get('output'));extra_rows.append(meta)
            del context,refcontext,lab,prep,sky,refsky;gc.collect()
    progress(f'[abell-zarr] {row["band"]}/{row["patch"]} stage=pack_and_publish')
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
    return done


def run_abell_zarr(args):
    configure_progress_warnings();started=time.monotonic()
    progress(f'[abell-zarr] START workers={args.workers} use_uncover={getattr(args,"use_uncover",False)} '
             f'background_root={args.background_root} output={args.output_root} stage=read_manifest')
    root=Path(args.output_root)
    if (root/'manifest.json').exists():manifest=json.loads((root/'manifest.json').read_text())
    else:
        # Allows an explicitly selected pilot after that band's cutouts finish.
        if not getattr(args,'abell_parent',None):raise ValueError('Wait for cutout manifest before batch Zarr')
        manifest=dict(rows=[r for p in (root/'manifests').glob('*.json') for r in json.loads(p.read_text())],failures=[])
    if manifest['failures']:raise ValueError('Cutout stage is incomplete')
    rows=[r for r in manifest['rows'] if r['status']=='kept' and (not args.jwst_bands or r['band'] in {b.upper() for b in args.jwst_bands})]
    if getattr(args,'abell_parent',None):rows=[r for r in rows if r['patch'] in args.abell_parent]
    progress(f'[abell-zarr] planned_parents={len(rows)} stage=coverage_preflight')
    outside=[]
    if getattr(args,'use_uncover',False):
        from .utils.uncover_labels import coverage_on_grid
        selected=[]
        for index,row in enumerate(rows,1):
            h=fits.getheader(row['training_fits'])
            mask=coverage_on_grid(args.uncover_detection,h,(h['NAXIS2'],h['NAXIS1']),cache_root=root/'uncover_coverage')
            if mask.any():selected.append(row)
            else:outside.append(dict(band=row['band'],patch=row['patch'],status='outside_uncover'))
            if index%20==0 or index==len(rows):
                progress(f'[abell-zarr] coverage_preflight={index}/{len(rows)} selected={len(selected)} outside={len(outside)}')
        rows=selected
    results=[];failures=[]
    progress(f'[abell-zarr] SUBMIT parents={len(rows)} workers={args.workers}')
    with ProcessPoolExecutor(max_workers=args.workers,max_tasks_per_child=1) as pool:
        fs={pool.submit(build_parent,args,r):r for r in rows}
        for f in completed_tasks(fs,'abell-zarr'):
            try:results.append(f.result())
            except Exception as e:
                failures.append(dict(band=fs[f]['band'],patch=fs[f]['patch'],error=repr(e)))
                import traceback;traceback.print_exc()
    summary=dict(results=results,failures=failures,skipped=outside,use_uncover=getattr(args,'use_uncover',False))
    (root/'zarr_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if failures:raise RuntimeError(f'{len(failures)} parent Zarr failures; see zarr_summary.json')
    progress(f'[abell-zarr] END parents={len(results)} skipped={len(outside)} '
             f'grid_samples={sum(r.get("grid_samples",0) for r in results)} '
             f'large_samples={sum(r.get("large_samples",0) for r in results)} elapsed={time.monotonic()-started:.1f}s')
    return summary
