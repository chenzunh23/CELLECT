"""Real-catalog dense-only checks against the approved sequential paint rules."""
import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import ZScaleInterval
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import zarr

from preprocessing.tests import run_source_pipeline_preview as source
from preprocessing.tests.visualize_jwst_psf_confidence_dense import bright_component_centers
from preprocessing.labels import DenseLabel as D, SourceClass as C, SourceLabels
from preprocessing.region_filling import fill_dense_regions, RegionFillingConfig
from preprocessing.refit import compute_kron_ellipse
from preprocessing.utils.geometry import EllipseGeometry, paint_ellipse
from preprocessing.image_processing import prepare_image, build_bright_components, BrightRegionConfig


def coverage(geom, labels, shape):
    mask = np.zeros(shape, bool)
    for i in np.flatnonzero(geom.valid() & ~labels.mask(C.DROPPED)):
        paint_ellipse(mask, geom.x[i], geom.y[i], geom.major[i], geom.minor[i], geom.theta[i], True)
    return mask


def check_and_plot(out, name, image, geom, labels, kwargs, *, comparison_dense=None):
    dense = fill_dense_regions(Table(), labels, image.shape, geometry=geom, **kwargs)
    uncapped = fill_dense_regions(Table(), labels, image.shape, geometry=geom,
                                 config=RegionFillingConfig(max_major_pixels=None), **kwargs)
    union = coverage(geom, labels, image.shape)
    # Independent sequential painter, following the Sep 8 reference. Unlike
    # its cached background, use the actual LSST mask supplied to both paths.
    expected = np.full(image.shape, int(D.ORDINARY_IGNORE), np.uint8)
    expected[kwargs['background_mask']] = D.BACKGROUND
    expected[union] = D.ORDINARY_IGNORE
    for key in ('bright_region_mask', 'ordinary_ignore_mask'):
        if kwargs.get(key) is not None:
            expected[np.asarray(kwargs[key], bool)] = D.ORDINARY_IGNORE
    quality = kwargs.get('quality_ignore_mask')
    if quality is not None:
        expected[quality] = D.STRICT_IGNORE
    for cls, value in ((C.STRICT_IGNORE, D.STRICT_IGNORE), (C.RESTRICTED_BRIGHT_REGION, D.RESTRICTED_BRIGHT_REGION),
                       (C.WEAK_SHAPE, D.WEAK_SHAPE), (C.CLEAN, D.CLEAN)):
        if value == D.RESTRICTED_BRIGHT_REGION and kwargs.get('restricted_fallback_mask') is not None:
            expected[kwargs['restricted_fallback_mask']] = value
        for i in np.flatnonzero(geom.valid() & labels.mask(cls)):
            scale = min(1., 100 / max(geom.major[i], geom.minor[i])) if cls in (C.CLEAN, C.WEAK_SHAPE) else 1.
            paint_ellipse(expected, geom.x[i], geom.y[i], geom.major[i]*scale,
                          geom.minor[i]*scale, geom.theta[i], int(value))
    differences = int(np.count_nonzero(expected != dense))
    leakage = int(np.count_nonzero((dense == D.BACKGROUND) & union))
    assert differences == leakage == 0, (name, differences, leakage)
    selected = geom.valid() & np.isin(labels.source_class, [C.CLEAN, C.WEAK_SHAPE])
    selected &= (geom.x >= 0) & (geom.x < image.shape[1]) & (geom.y >= 0) & (geom.y < image.shape[0])
    result = dict(name=name, clipped_clean=int(np.count_nonzero(selected & labels.mask(C.CLEAN) & (geom.major > 100))),
                  clipped_weak=int(np.count_nonzero(selected & labels.mask(C.WEAK_SHAPE) & (geom.major > 100))),
                  changed_pixels=int(np.count_nonzero(dense != uncapped)), reference_painter_differences=differences,
                  background_source_overlap_pixels=leakage,
                  counts={D(int(v)).name:int(n) for v,n in zip(*np.unique(dense,return_counts=True))})
    np.savez_compressed(out/f'{name}_dense.npz', dense=dense)
    colors = {D.CLEAN:'#00df60', D.WEAK_SHAPE:'#00dfff', D.RESTRICTED_BRIGHT_REGION:'#ffb000',
              D.BACKGROUND:'#245bff', D.ORDINARY_IGNORE:'#ff4040', D.STRICT_IGNORE:'#9933cc'}
    from matplotlib.colors import to_rgba
    ds = max(1, int(np.ceil(max(image.shape)/1800)))
    raw = image[::ds,::ds]; finite = raw[np.isfinite(raw)]
    lo,hi = ZScaleInterval().get_limits(finite)
    fig,axes = plt.subplots(1,2,figsize=(16,8),dpi=150)
    before = uncapped if comparison_dense is None else comparison_dense
    titles = ('Uncapped','Semi-major <= 100 px') if comparison_dense is None else ('Previous dense target','New LSST background, cap 100 px')
    for ax,values,title in zip(axes,(before,dense),titles):
        ax.imshow(raw,origin='lower',cmap='gray',vmin=lo,vmax=hi,interpolation='nearest')
        small=values[::ds,::ds]; rgba=np.zeros((*small.shape,4),np.float32)
        for key,color in colors.items(): rgba[small==key]=to_rgba(color,.30)
        ax.imshow(rgba,origin='lower',interpolation='nearest'); ax.set_axis_off(); ax.set_title(title)
    axes[-1].legend(handles=[Patch(color=c,label=k.name.lower(),alpha=.5) for k,c in colors.items()],fontsize=9)
    fig.tight_layout(); fig.savefig(out/f'{name}_overlay.png'); plt.close(fig)
    print(json.dumps(result),flush=True)
    return result


def hsc(out):
    b=source.b; attrs=dict(zarr.open_group(str(source.HREF),mode='r').attrs)
    argv=sys.argv
    try:
        sys.argv=['build','--output-root',str(out/'unused'),'--coadd-fits-root','/data/czh23/Subaru_products/half_coadd',
                  '--patches','4,5','--bands','HSC-I','--dataset-sources','coadd','--gaia-fits',str(source.GAIA),
                  '--coadd-lsst-background-root','/data/czh23/Subaru_products/lsst_background_masks']
        task=b._make_tasks(b.parse_args())[0]
    finally: sys.argv=argv
    task=dataclasses.replace(task,**{k:v for k,v in attrs.items() if k in task.__dataclass_fields__ and
        not k.endswith('root') and k not in ('tract','patch','bands','dataset_source','group')})
    path=b._coadd_image_path(task.coadd_fits_root,'HSC-I',9813,'4,5')
    image,header,origin=b._read_image_header_origin(path)
    rows=source.read_csv(source.BASE/'2026-09-08/shared_a_gaia_lupton_comparison/hsc_exposure_epochs.csv')
    header['MJD-AVG']=np.average([float(r['mjd']) for r in rows],weights=[float(r['weight']) for r in rows])
    reports=[]; original=b.fill_dense_regions
    def capture(table,labels,shape,**kwargs):
        # Pin the correct half-coadd LSST background, not the official coadd.
        bg=Path('/data/czh23/Subaru_products/lsst_background_masks/half_coadd/9813/4,5/coadd/HSC-I/background_mask.npz')
        with np.load(bg) as z: kwargs['background_mask']=z['background_mask'].astype(bool)
        kwargs.pop('refit_config',None)
        reports.append(check_and_plot(out,'hsc_i_4_5',image,compute_kron_ellipse(table),labels,kwargs))
        return original(table,labels,shape,**kwargs)
    b.fill_dense_regions=capture
    try: b._classify_patch(task,image,header,origin,path)
    finally: b.fill_dense_regions=original
    return reports


def cosmos(out):
    dest=source.BASE/'2026-09-09/source_pipeline_before_dense/cosmos19_f444w'
    rows=source.read_csv(dest/'sources.csv'); meta=json.loads((dest/'summary.json').read_text())
    col=lambda k:np.array([float(r[k]) for r in rows])
    codes={v:k for k,v in source.NAMES.items()}
    labels=SourceLabels(np.array([codes[r['final']] for r in rows]),np.full(len(rows),'cached',object))
    x,y,a,b,theta=col('x'),col('y'),col('a'),col('b'),np.deg2rad(col('theta_deg'))
    ins=source.read_csv(dest/'inserted.csv'); centers=np.array([[float(r['x']),float(r['y'])] for r in ins])
    bgpath=Path('/data/czh23/JWST/lsst_background_masks/jwst/default/0019/group_00/f444w/background_mask.npz')
    with np.load(bgpath) as z: background=z['background_mask']
    results=[]
    with fits.open(meta['raw_fits'],memmap=True) as hdus:
        hdu=next(h for h in hdus if h.header.get('NAXIS')==2); ny,nx=hdu.shape
        for name,x0,y0 in [('left_top',0,ny-4096),('center',nx//2-2048,ny//2-2048),('top_col3',8192,ny-4096)]:
            print('[dense] cosmos '+name,flush=True)
            raw=np.asarray(hdu.section[y0:y0+4096,x0:x0+4096],np.float32)
            geom=EllipseGeometry(x-x0,y-y0,a,b,theta,np.pi*a*b)
            prepared=prepare_image(raw,header=hdu.header)
            bright,_=build_bright_components(prepared,config=BrightRegionConfig(
                threshold=5, clip_threshold=5, statistics_clip_sigma=5))
            # Preserve the Sep 6 COSMOS component policy; this test changes
            # filling only, not Gaia insertion or source classifications.
            union=coverage(geom,labels,raw.shape)
            _,small,_=bright_component_centers(bright,union,min_area=1000,protected_centers=centers-[x0,y0])
            result=check_and_plot(out,'cosmos19_f444w_'+name,raw,geom,labels,
                dict(background_mask=background[y0:y0+4096,x0:x0+4096],bright_region_mask=bright,
                     restricted_fallback_mask=bright.astype(bool)&~small,ordinary_ignore_mask=~np.isfinite(raw)))
            result['bright_policy']='Sep 6 component helper; current local scaling, not cached global scaling'
            result['scaling']={'threshold':5,'clip_threshold':5,'statistics_clip_sigma':5,'scope':'local'}
            results.append(result)
    return results


def abell(out):
    bgdir=Path('/data/czh23/JWST/lsst_background_masks/jwst/default/Abell2744/group_00/f444w')
    bgmeta=json.loads((bgdir/'summary.json').read_text())
    if bgmeta['status'] != 'ok': raise ValueError('LSST background incomplete')
    for key,value in {'threshold_value':3.,'n_sigma_to_grow':1.,'min_pixels':15,
                      'static_detection':True,'disable_bright_prelim':True,'disable_temp_backgrounds':True}.items():
        if bgmeta[key] != value: raise ValueError(f'LSST configuration differs from COSMOS: {key}')
    with np.load(bgdir/'background_mask.npz') as z: full_bg=z['background_mask'].astype(bool)
    if list(full_bg.shape) != bgmeta['shape_yx'] or bgmeta['origin_xy'] != [0,0]:
        raise ValueError('LSST background grid mismatch')
    reference=source.BASE/'2026-09-08/a2744_dense_confidence_overlay_from_strict_snr'
    results=[]
    for region in ('f250m_center','f115w_valid_center'):
        name=f'a2744_{region}_f444w'
        print('[dense] '+name,flush=True)
        dest=source.BASE/'2026-09-09/source_pipeline_before_dense'/name
        meta=json.loads((dest/'summary.json').read_text());rows=source.read_csv(dest/'sources.csv')
        if Path(meta['raw_fits']).resolve() != Path(bgmeta['input']).resolve():
            raise ValueError('Background and source selection use different FITS')
        x0,y0=meta['origin']; col=lambda k:np.array([float(r[k]) for r in rows])
        geom=EllipseGeometry(col('x'),col('y'),col('a'),col('b'),np.deg2rad(col('theta_deg')),np.pi*col('a')*col('b'))
        codes={v:k for k,v in source.NAMES.items()}
        labels=SourceLabels(np.array([codes[r['final']] for r in rows]),np.full(len(rows),'cached',object))
        with fits.open(meta['raw_fits'],memmap=True) as hdus:
            hdu=next(h for h in hdus if h.header.get('NAXIS')==2)
            if hdu.shape != full_bg.shape: raise ValueError('FITS/background shape mismatch')
            raw=np.asarray(hdu.section[y0:y0+4096,x0:x0+4096],np.float32)
            prepared=prepare_image(raw,header=hdu.header)
        bright,components=build_bright_components(prepared,config=BrightRegionConfig(
            threshold=5,clip_threshold=5,statistics_clip_sigma=5))
        areas=np.bincount(components.ravel());keep=areas>=1000;keep[0]=False
        big=keep[components]
        with np.load(reference/f'{name}_dense_confidence_masks.npz') as z:
            old_dense=z['dense'];old_bg=z['raw_lsst_background'].astype(bool);star=z['star_footprint'].astype(bool)
        bg=full_bg[y0:y0+4096,x0:x0+4096]
        kwargs=dict(background_mask=bg,bright_region_mask=bright,restricted_fallback_mask=big,
                    ordinary_ignore_mask=star|~prepared.finite_mask)
        result=check_and_plot(out,name,raw,geom,labels,kwargs,comparison_dense=old_dense)
        same_old_bg=fill_dense_regions(Table(),labels,raw.shape,geometry=geom,**{**kwargs,'background_mask':old_bg})
        with np.load(out/f'{name}_dense.npz') as z: dense=z['dense']
        assert not np.any((dense==D.BACKGROUND)&((bright>0)|star|~prepared.finite_mask))
        result.update(background_path=str(bgdir/'background_mask.npz'),source_labels=str(dest/'sources.csv'),
            source_filters_rerun=False,origin=[x0,y0],scaling=prepared.metadata,
            threshold=5,clip_threshold=5,statistics_clip_sigma=5,statistics_scope='local',
            raw_bright_pixels=int(np.count_nonzero(bright)),retained_bright_pixels=int(big.sum()),
            raw_background_fraction_before=float(old_bg.mean()),raw_background_fraction_after=float(bg.mean()),
            background_only_changed_pixels=int(np.count_nonzero(same_old_bg!=dense)),
            background_only_gained_pixels=int(np.count_nonzero((dense==D.BACKGROUND)&(same_old_bg!=D.BACKGROUND))),
            background_only_removed_pixels=int(np.count_nonzero((dense!=D.BACKGROUND)&(same_old_bg==D.BACKGROUND))),
            previous_dense_changed_pixels=int(np.count_nonzero(dense!=old_dense)))
        results.append(result)
        print(json.dumps(result),flush=True)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=source.BASE/'2026-09-09/dense_filling_alignment')
    parser.add_argument('--dataset',choices=['hsc','cosmos','abell','all'],default='all')
    args=parser.parse_args();args.output_root.mkdir(parents=True,exist_ok=True)
    reports=[]
    for name,fn in [('hsc',hsc),('cosmos',cosmos),('abell',abell)]:
        if args.dataset in (name,'all'): reports.extend(fn(args.output_root))
    (args.output_root/f'{args.dataset}_summary.json').write_text(json.dumps(reports,indent=2))


if __name__=='__main__': main()
