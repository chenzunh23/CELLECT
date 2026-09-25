"""End-to-end source-only diagnostics; stop before dense targets and Zarr writes.

Run with python -m preprocessing.tests.run_source_pipeline_preview --dataset all.
Reference analysis modules only supply catalog/cutout adapters and old labels.
"""
import argparse
import csv
import dataclasses
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-cellect')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import ZScaleInterval
from scipy.spatial import cKDTree
import zarr

from preprocessing import build_image_level_zarr as b
from preprocessing.labels import SourceClass as C
from preprocessing.image_processing import prepare_image, build_bright_components, BrightRegionConfig
from preprocessing.meas_processing import classify_catalog_basics
from preprocessing.ordinary_common import OrdinaryInput, classify_ordinary_sources
from preprocessing.ordinary_a2744 import A2744OrdinaryConfig
from preprocessing.bright_label_jwst import JWSTBrightConfig, label_bright_sources
from preprocessing.utils.geometry import EllipseGeometry
from preprocessing.tests.compare_ordinary_references import module, A_REF, C_REF, CLASSES

BASE = Path('/home/czh23/analysis/2026-09')
HREF = Path('/data/czh23/direct_zarr_v4_lupton/image_level/coadd/HSC-I/4,5.zarr')
GAIA = Path('/home/czh23/CELLECT/output/gaia_dr3_cosmos_full.fits')
NAMES = {0:'drop', 1:'clean', 2:'weak_shape', 3:'ignore', 4:'strict_center_only', 5:'restricted', 6:'strict_ignore', 7:'not_exported', 8:'pending', 9:'unchanged'}
COLORS = {0:'#8656bd', 1:'#00d060', 2:'#00cfff', 3:'#ff4040', 4:'#ee00dd', 5:'#ffb000', 6:'#9933cc', 7:'#888888', 8:'#ffff00', 9:'#888888'}


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    rows = list(rows)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def full_header(path):
    with fits.open(path, memmap=True) as hdus:
        header = hdus[0].header.copy()
        hdu = next(h for h in hdus if h.header.get('NAXIS') == 2)
        header.update(hdu.header)
        return header


def counts(values):
    return {NAMES[int(k)]: int(v) for k, v in zip(*np.unique(values, return_counts=True))}


def preview(path, image, geom, states, inserted, *, origin=(0, 0), size=None):
    x0, y0 = origin
    height, width = image.shape if size is None else (size, size)
    cut = image[y0:y0+height, x0:x0+width]
    ds = max(1, int(np.ceil(max(cut.shape)/1600)))
    display = np.asarray(cut[::ds, ::ds], np.float32)
    finite = display[np.isfinite(display)]
    lo, hi = ZScaleInterval().get_limits(finite) if finite.size else (0, 1)
    fig, axes = plt.subplots(1, len(states), figsize=(7*len(states), 7), dpi=150, squeeze=False)
    select = geom.valid() & (geom.x >= x0) & (geom.x < x0+width) & (geom.y >= y0) & (geom.y < y0+height)
    for ax, (name, values) in zip(axes[0], states.items()):
        ax.imshow(display, origin='lower', cmap='gray', vmin=lo, vmax=hi,
                  extent=(x0, x0+cut.shape[1], y0, y0+cut.shape[0]), interpolation='nearest')
        for cls in np.unique(values[select]):
            use = select & (values == cls)
            ax.add_collection(EllipseCollection(2*geom.major[use], 2*geom.minor[use], np.degrees(geom.theta[use]),
                units='xy', offsets=np.column_stack((geom.x[use], geom.y[use])), transOffset=ax.transData,
                facecolors='none', edgecolors=COLORS[int(cls)], linewidths=.35))
            ax.plot([], [], color=COLORS[int(cls)], label=f'{NAMES[int(cls)]} ({use.sum()})')
        if name == 'final' and len(inserted):
            use = (inserted[:,0] >= x0) & (inserted[:,0] < x0+width) & (inserted[:,1] >= y0) & (inserted[:,1] < y0+height)
            ax.scatter(*inserted[use].T, s=28, marker='+', c='yellow', linewidths=.6, label=f'inserted ({use.sum()})')
        ax.set_xlim(x0, x0+cut.shape[1]); ax.set_ylim(y0, y0+cut.shape[0]); ax.set_aspect('equal')
        ax.set_title(name); ax.set_axis_off(); ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def emit(out, name, image, geom, ids, mag, reference, stages, bright, old_inserted, metadata, snr=None, zooms=()):
    dest = out/name; dest.mkdir(parents=True, exist_ok=True)
    final = bright.labels.source_class.copy()
    stages['final'] = final
    inserted = np.column_stack((bright.strict_center_x, bright.strict_center_y))
    old_inserted = np.asarray(old_inserted, float).reshape(-1, 2)
    compare = np.where(np.isin(final, [1,2,4]), final, 7) if metadata.get('reference_membership_only') else final
    changed = compare != reference
    write_csv(dest/'sources.csv', (dict(id=int(ids[i]), x=float(geom.x[i]), y=float(geom.y[i]),
        a=float(geom.major[i]), b=float(geom.minor[i]), theta_deg=float(np.degrees(geom.theta[i])), mag=float(mag[i]),
        snr_value=float(snr[i]) if snr is not None else '', reference=NAMES[int(reference[i])],
        **{stage:NAMES[int(value[i])] for stage,value in stages.items()},
        changed=bool(changed[i]), reason=str(bright.labels.reason[i])) for i in range(len(ids))))
    write_csv(dest/'inserted.csv', (dict(id=int(sid), x=float(x), y=float(y), reason=str(reason), component=int(comp))
        for sid,x,y,reason,comp in zip(bright.strict_center_source_id, bright.strict_center_x, bright.strict_center_y,
                                     bright.strict_center_reason, bright.strict_center_component_id)))
    for label in ('complete', 'clean', 'weak_shape', 'strict_center_only', 'ignore', 'drop'):
        with (dest/f'{label}.reg').open('w') as f:
            f.write('# Region file format: DS9 version 4.1\nglobal width=1\nimage\n')
            for i in np.flatnonzero(geom.valid()):
                cls = int(final[i])
                if label != 'complete' and NAMES[cls] != label:
                    continue
                f.write(f'ellipse({geom.x[i]+1:.5f},{geom.y[i]+1:.5f},{geom.major[i]:.5f},{geom.minor[i]:.5f},{np.degrees(geom.theta[i]):.5f}) # color={COLORS[cls]} text={{id={int(ids[i])} mag={mag[i]:.2f} {NAMES[cls]}}}\n')
            if label in ('complete', 'strict_center_only'):
                for x,y in inserted:
                    f.write(f'point({x+1:.5f},{y+1:.5f}) # point=cross color=yellow\n')
    distances = cKDTree(old_inserted).query(inserted)[0] if len(old_inserted) and len(inserted) else np.full(len(inserted), np.inf)
    olddist = cKDTree(inserted).query(old_inserted)[0] if len(old_inserted) and len(inserted) else np.full(len(old_inserted), np.inf)
    summary = dict(name=name, sources=len(ids), reference_counts=counts(reference), final_counts=counts(final),
                   changed_sources=int(changed.sum()), reference_inserted=len(old_inserted), inserted=len(inserted),
                   new_centers_without_reference_within_1pixel=int((distances>1).sum()),
                   reference_centers_without_new_within_1pixel=int((olddist>1).sum()), **metadata)
    (dest/'summary.json').write_text(json.dumps(summary, indent=2, default=str))
    write_csv(dest/'stage_counts.csv', (dict(stage=k, **counts(v)) for k,v in stages.items()))
    np.savez_compressed(dest/'source_stages.npz', ids=ids, reference=reference, **stages,
                        inserted=inserted, reference_inserted=old_inserted)
    preview(dest/'comparison.png', image, geom, {'reference':reference, 'final':final,
            'changed':np.where(changed, final, 9)}, inserted)
    stage_view = {k:stages[k] for k in ('A', 'ordinary', 'final') if k in stages}
    for zoom, x0,y0,size in zooms:
        preview(dest/f'{zoom}_stages.png', image, geom, stage_view, inserted, origin=(x0,y0), size=size)
    print(json.dumps(summary, default=str), flush=True)
    return summary


def hsc(out):
    z = zarr.open_group(str(HREF), mode='r'); attrs = dict(z.attrs)
    argv = sys.argv
    try:
        sys.argv = ['build', '--output-root', str(out/'unused'), '--coadd-fits-root', '/data/czh23/Subaru_products/half_coadd',
                    '--patches', '4,5', '--bands', 'HSC-I', '--dataset-sources', 'coadd', '--gaia-fits', str(GAIA)]
        task = b._make_tasks(b.parse_args())[0]
    finally:
        sys.argv = argv
    task = dataclasses.replace(task, **{k:v for k,v in attrs.items() if k in task.__dataclass_fields__ and
        not k.endswith('root') and k not in ('tract','patch','bands','dataset_source','group')})
    path = b._coadd_image_path(task.coadd_fits_root, 'HSC-I', 9813, '4,5')
    image, header, origin = b._read_image_header_origin(path)
    epoch_path = BASE/'2026-09-08/shared_a_gaia_lupton_comparison/hsc_exposure_epochs.csv'
    epochs = read_csv(epoch_path)
    header['MJD-AVG'] = np.average([float(r['mjd']) for r in epochs], weights=[float(r['weight']) for r in epochs])
    table = b.attach_refit_geometry(Table.read(b._band_catalog_path(task.data_root,task.band,task.tract,task.patch)),
                                   b._refit_csv_path(task.refit_root,task.tract,task.band,task.patch), b.RefitConfig())
    mask, components = b.build_bright_components(image, config=BrightRegionConfig(mode=task.bright_mask_mode,
        threshold=task.bright_threshold, clip_threshold=task.clip_threshold, dilation=task.bright_dilation, log_a=b._bright_log_a(task)))
    stage = b.classify_meas_basics(table)
    stages = {'A':np.where(stage.after_a, 1, stage.labels.source_class), 'B_basic':np.where(stage.after_b_basic,8,stage.labels.source_class)}
    snr = b.compute_snr_for_sample(table, dataset_source='coadd', is_narrow_band=False)
    ordinary = b.label_ordinary_sources(table, stage.ordinary_candidate, stage.labels, is_narrow_band=False, snr=snr.snr)
    stages['ordinary'] = np.where(stage.bright_candidate & (ordinary.labels.reason == 'unassigned'),8,ordinary.labels.source_class)
    ap = b.classify_bright_ap2(table, stage.bright_candidate, ordinary.labels, component_labels=components)
    stages['bright_ap2'] = ap.labels.source_class.copy()
    gaia = Table.read(GAIA)
    if 'ref_epoch' not in gaia.colnames: gaia['ref_epoch'] = np.full(len(gaia),2016.)
    bright = b.label_bright_sources(table, ap.candidate, ap.labels, bright_region=mask, component_labels=components,
        gaia_table=gaia, image_header=header, quality_mask=b._read_fits_quality_mask(path,image.shape), mag=stage.mag,
        config=b.BrightLabelConfig(cluster_source_match_pixels=task.cluster_source_match_pixels,
            cluster_centroid_match_pixels=task.cluster_centroid_match_pixels, gaia_bright_mag_threshold=task.gaia_bright_mag_threshold))
    stages['bright'] = bright.labels.source_class.copy()
    seeds = ap.component_id[stage.bright_candidate]
    fx,fy,fi,fc = b.unsupervised_seeded_component_centers(table, bright.labels, components,
        seed_component_ids=seeds, catalog_component_ids=seeds, existing_strict_component_ids=bright.strict_center_component_id,
        min_area=b.BrightLabelConfig().empty_seeded_bright_component_area_min,
        component_search_radius=b.BrightAp2Config().component_search_radius)
    for key,values in [('strict_center_x',fx),('strict_center_y',fy),('strict_center_source_id',fi),('strict_center_component_id',fc),
                      ('strict_center_reason',np.full(len(fx),'seeded_bright_component_no_supervised_center',object))]:
        setattr(bright,key,np.concatenate([getattr(bright,key),values]))
    geom = b.compute_kron_ellipse(table); ids = b.source_ids(table)
    mapping = {}
    for sid,cls in zip(z['shape_source_ids'][:],z['shape_source_classes'][:]): mapping[int(sid)] = int(cls)
    for sid in z['strict_center_only_ids'][:]:
        if sid >= 0: mapping[int(sid)] = 4
    reference = np.array([mapping.get(int(sid),7) for sid in ids])
    oldins = []
    for tile,(start,end) in enumerate(z['strict_center_only_offsets'][:]):
        sid = z['strict_center_only_ids'][start:end]; xy = z['strict_center_only_centers'][start:end]
        oldins.extend((xy[sid<0]+[z['tile_x0'][tile]-origin[0],z['tile_y0'][tile]-origin[1]]).tolist())
    oldins = np.unique(np.round(np.asarray(oldins).reshape(-1,2),4),axis=0)
    return emit(out,'hsc_i_4_5',image,geom,ids,stage.mag,reference,stages,bright,oldins,
        dict(reference=str(HREF),raw_fits=str(path),reference_membership_only=True,origin=origin,
             epoch_mjd=float(header['MJD-AVG']),epoch_source=str(epoch_path),gaia=str(GAIA),
             scaling=attrs['image_scaling_mode'],bright_threshold=task.bright_threshold), snr.snr,
        zooms=[(f'tile{i:03d}',int(z['tile_x0'][i]-origin[0]),int(z['tile_y0'][i]-origin[1]),512) for i in (59,100)])


def cosmos(out):
    ref = module(C_REF/'jwst_no_morph_snr_phot_containment_three_blocks.py','cosmos_source_reference')
    p = ref.load_pipeline(); ref.set_context(p); cache = p.load_catalog()
    path = p.raw_fits_path('f444w'); image,wcs,scale = p.image_hdu(path)
    x,y = wcs.all_world2pix(cache['ra'],cache['dec'],0)
    idx = np.flatnonzero(np.isfinite(x+y)&(x>=0)&(y>=0)&(x<image.shape[1])&(y<image.shape[0]))
    sel = dict(idx=idx,x=x[idx],y=y[idx])
    a,bb = cache['kron2_a'][idx]/scale,cache['kron2_b'][idx]/scale
    geom = EllipseGeometry(x[idx],y[idx],a,bb,np.deg2rad(cache['theta_world'][idx]),np.pi*a*bb)
    data = OrdinaryInput(geom, cache['mag_auto_f444w'][idx], snr=cache['snr_f444w'][idx],
                        flags=cache['warn_flag'][idx],star_mask=cache['flag_star'][idx],model_mag=cache['mag_model_f444w'][idx])
    old_nan_drop = p.nan_component_center_mask(image,sel['x'],sel['y'])
    basics = classify_catalog_basics(geom,data.mag,dataset='cosmos',pixel_scale_arcsec=scale,image=image)
    nan_ignore = basics.nan_center_ignore
    data.center_invalid = nan_ignore
    stages = {'A':np.where(basics.after_a & ~nan_ignore,1,basics.labels.source_class)}
    from preprocessing.utils.segmentation import cosmos_fill_ratios
    data.segmentation_fill_ratio = cosmos_fill_ratios(geom, cache['id'][idx], wcs,
        '/data/shared/jwst_foundation/catalog/COSMOS_1727_1837_5893/COSMOSWeb_mastercatalog_v1.1.fits',
        selected=basics.after_b_basic & ~nan_ignore & (data.mag < 25.5))
    ordinary = classify_ordinary_sources(data,basics.after_b_basic & ~nan_ignore,basics.labels,dataset='cosmos')
    stages.update(ordinary.stages); stages['ordinary'] = ordinary.labels.source_class.copy()
    bright = label_bright_sources(data,ordinary.labels,image_shape=image.shape,image_header=full_header(path),gaia_table=Table.read(GAIA),
        source_ids=cache['id'][idx],config=JWSTBrightConfig('cosmos',scale,.145,gaia_reference_epoch=2016.))
    drop = (cache['kron2_area'][idx] > p.AREA_IGNORE_MAX_PIX)|old_nan_drop
    oldbase = dict(a_ignore=drop,stage1_ignore=~drop&np.isin(data.flags,p.WARN_IGNORE_VALUES),mag_auto=data.mag,
                   mag_model=data.model_mag,dmag=np.abs(data.mag-data.model_mag),snr=data.snr,area=cache['kron2_area'][idx])
    old = ref.ordered_classify(p,cache,sel,oldbase,scale)
    reference = np.array([CLASSES[v] for v in old['final']])
    policy = read_csv(BASE/'2026-09-07/jwst_gaia_insertion_policy/pointing0019_f444w_gaia_policy_gaia_policy.csv')
    oldins = [[float(r['gaia_x']),float(r['gaia_y'])] for r in policy if r['insert_gaia_strict_center']=='True']
    ny,nx = image.shape
    return emit(out,'cosmos19_f444w',image,geom,cache['id'][idx],data.mag,reference,stages,bright,oldins,
        dict(reference=str(C_REF),raw_fits=str(path),gaia=str(GAIA),pixel_scale=scale,nan_center_ignored=int(nan_ignore.sum()),old_nan_drop=int(old_nan_drop.sum()),
             scaling='source filters independent of scaling; COSMOS Gaia does not use bright mask'),data.snr,
        zooms=[('left_top',0,ny-4096,4096),('center',nx//2-2048,ny//2-2048,4096),('top_col3',8192,ny-4096,4096)])


def abell(out, bands):
    ref = module(A_REF/'a2744_sextractor_hsc_style_filter.py','a2744_source_reference')
    catalog = ref.load_catalog(); f444 = ref.image_hdu(ref.raw_fits_path('f444w'))
    gaia = Table.read(ref.GAIA_CATALOG)
    summaries = []
    for band in bands:
        info = ref.image_hdu(ref.raw_fits_path(band))
        bgdir = Path('/data/czh23/JWST/lsst_background_masks/jwst/default/Abell2744/group_00')/band
        bgmeta = json.loads((bgdir/'summary.json').read_text())
        if bgmeta['status'] != 'ok' or Path(bgmeta['input']).resolve() != Path(info.path).resolve():
            raise ValueError('LSST sky mask is incomplete or belongs to a different image')
        if bgmeta['origin_xy'] != [0, 0]:
            raise ValueError('LSST sky mask must use full-image coordinates')
        with np.load(bgdir/'background_mask.npz') as z:
            full_sky = z['background_mask'].astype(bool)
        if list(full_sky.shape) != bgmeta['shape_yx']:
            raise ValueError('LSST sky mask shape mismatch')
        for region in ref.read_regions():
            name = f'a2744_{region.name}_{band}'
            print(f'[run] {name}: preparing image',flush=True)
            raw,x0,y0 = ref.cutout_for_region(info,region,f444)
            sel = ref.select_sources(catalog,info,x0,y0); idx = sel['idx']
            aa,bb,theta,_ = ref.ellipse_params(catalog)
            geom = EllipseGeometry(sel['x'],sel['y'],aa[idx],bb[idx],np.deg2rad(theta[idx]),np.pi*aa[idx]*bb[idx])
            data = OrdinaryInput(geom,ref.finite_mag(catalog['FLUX_AUTO'])[idx],snr=ref.snr_values(catalog)[idx],flags=catalog['FLAGS'][idx])
            prepared = prepare_image(raw,header=full_header(info.path))
            mask,components = build_bright_components(prepared,config=BrightRegionConfig(threshold=5,clip_threshold=5,statistics_clip_sigma=5))
            with np.load(A_REF/f'{name}_stage_masks.npz') as z:
                star = z['star_footprint'].astype(bool); oldins = z['inserted_xy'].copy()
                mask_diff = int(np.count_nonzero(mask.astype(bool) != z['bright'].astype(bool)))
            basics = classify_catalog_basics(geom,data.mag,dataset='a2744',pixel_scale_arcsec=info.pixscale,valid_mask=prepared.finite_mask)
            data.center_invalid = basics.nan_center_ignore
            stages = {'A':np.where(basics.after_a,1,basics.labels.source_class)}
            linear = prepared.image.copy(); linear[~prepared.finite_mask] = np.nan
            print(f'[run] {name}: ordinary and aperture SNR',flush=True)
            ordinary = classify_ordinary_sources(data,basics.after_b_basic,basics.labels,dataset='a2744',
                config=A2744OrdinaryConfig(ref.PSF_FWHM_ARCSEC[band]/info.pixscale,image_shape=raw.shape),
                image=linear,excluded_mask=mask.astype(bool)|star,star_footprint=star,
                sky_mask=full_sky[y0:y0+raw.shape[0],x0:x0+raw.shape[1]])
            diagdir = out/name; diagdir.mkdir(parents=True, exist_ok=True)
            write_csv(diagdir/'aperture_snr.csv', (dict(id=int(catalog['NUMBER'][idx[i]]),
                snr=float(ordinary.diagnostics['aperture_snr'][i]),
                trusted=bool(ordinary.diagnostics['aperture_snr_trusted'][i]),
                eligible=bool(ordinary.diagnostics['aperture_snr_tested'][i]),
                ignored=bool(ordinary.diagnostics['aperture_snr_ignore'][i]),
                center_only=bool(ordinary.diagnostics['aperture_snr_center'][i])) for i in range(len(idx))))
            stages.update(ordinary.stages); stages['ordinary'] = ordinary.labels.source_class.copy()
            header = full_header(info.path); header.update(info.wcs.slice((slice(y0,y0+raw.shape[0]),slice(x0,x0+raw.shape[1]))).to_header())
            bright = label_bright_sources(data,ordinary.labels,image_shape=raw.shape,bright_region=mask,component_labels=components,
                gaia_table=gaia,image_header=header,source_ids=catalog['NUMBER'][idx],
                config=JWSTBrightConfig('a2744',info.pixscale,gaia_reference_epoch=2016.))
            old = read_csv(A_REF/f'{name}_source_stages.csv')
            if not np.array_equal([int(r['catalog_index']) for r in old],idx): raise ValueError('reference row alignment mismatch')
            summaries.append(emit(out,name,raw,geom,catalog['NUMBER'][idx],data.mag,np.array([CLASSES[r['final']] for r in old]),
                stages,bright,oldins,dict(reference=str(A_REF),raw_fits=str(info.path),origin=[x0,y0],gaia=str(ref.GAIA_CATALOG),
                    pixel_scale=info.pixscale,bright_mask_changed_pixels=mask_diff,star_mask='cached reference footprint',
                    snr_background=str(bgdir/'background_mask.npz'),aperture_radius_pixels=10,
                    snr_background_box=128,min_sky_apertures=8,
                    aperture_mag_min=None,aperture_selection='all retained candidates',
                    scaling='local log-lupton intersection, HSC surface brightness, hybrid fill'),
                ordinary.diagnostics['aperture_snr'],zooms=[('full_stages',0,0,4096)]))
    return summaries


def redraw_comparisons(out):
    """Re-render existing CSV results without re-running any source filters."""
    for dest in sorted(out.iterdir()):
        if not dest.is_dir() or not (dest/'sources.csv').exists():
            continue
        summary = json.loads((dest/'summary.json').read_text())
        rows = read_csv(dest/'sources.csv')
        def col(key):
            return np.array([float(r[key]) for r in rows])
        geom = EllipseGeometry(col('x'),col('y'),col('a'),col('b'),np.deg2rad(col('theta_deg')),np.pi*col('a')*col('b'))
        codes = {v:k for k,v in NAMES.items()}
        reference = np.array([codes[r['reference']] for r in rows])
        final = np.array([codes[r['final']] for r in rows])
        changed = np.array([r['changed']=='True' for r in rows])
        inserted_rows = read_csv(dest/'inserted.csv')
        inserted = np.array([[float(r['x']),float(r['y'])] for r in inserted_rows]).reshape(-1,2)
        with fits.open(summary['raw_fits'],memmap=True) as hdus:
            hdu = next(h for h in hdus if h.header.get('NAXIS') == 2)
            if dest.name.startswith('a2744'):
                x0,y0 = summary['origin']
                image = hdu.section[y0:y0+4096,x0:x0+4096]
            else:
                image = hdu.data
            preview(dest/'comparison.png', image, geom,
                    {'reference':reference,'final':final,'changed':np.where(changed,final,9)},inserted)
        print(f'[redraw] {dest.name}: unchanged={int((~changed).sum())}',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['all','hsc','cosmos','abell'],default='all')
    parser.add_argument('--output-root',type=Path,default=BASE/'2026-09-09/source_pipeline_before_dense')
    parser.add_argument('--abell-bands',nargs='+',choices=['f444w','f070w'],default=['f444w'])
    parser.add_argument('--redraw-only',action='store_true')
    args = parser.parse_args(); args.output_root.mkdir(parents=True,exist_ok=True)
    if args.redraw_only:
        redraw_comparisons(args.output_root)
        return
    summaries = []
    for name,fn in [('hsc',hsc),('cosmos',cosmos),('abell',lambda out:abell(out,args.abell_bands))]:
        if args.dataset in ('all',name):
            print(f'[run] {name}',flush=True)
            result = fn(args.output_root)
            summaries.extend(result if isinstance(result,list) else [result])
    (args.output_root/f'{args.dataset}_summary.json').write_text(json.dumps(summaries,indent=2,default=str))


if __name__ == '__main__':
    main()
