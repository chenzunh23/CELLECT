"""Full COSMOS filtering and native 512-pixel large-clean-source view audit."""
import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

from preprocessing.tests import run_source_pipeline_preview as common
from preprocessing.tests.visualize_jwst_psf_confidence_dense import SCALING
from preprocessing.meas_processing import classify_catalog_basics
from preprocessing.ordinary_common import OrdinaryInput, classify_ordinary_sources
from preprocessing.bright_label_jwst import JWSTBrightConfig, label_bright_sources
from preprocessing.labels import SourceClass as C
from preprocessing.utils.geometry import EllipseGeometry
from preprocessing.utils.image import hsc_surface_brightness_factor, fill_nonfinite_hybrid

CASES = ((17, 'f150w', .050), (19, 'f444w', .145), (4, 'f277w', .091), (25, 'f115w', .040))
RAW = Path('/data/shared/jwst_foundation/raw/COSMOS_1727_1837_5893')


def global_rgb(flux, params, scaling):
    z = (np.minimum(flux, params['clip_hi']) - params['zscore_median']) / params['zscore_std']
    log = (scaling.log_map(flux, params['raw_min'], params['log_hi']) - params['log_mean']) / params['log_std']
    lup = (scaling.lupton_map(flux, params['zscore_median'], 20.) - params['lupton_mean']) / params['lupton_std']
    return ((np.clip(np.stack([z, log, lup], axis=-1), -5, 5) + 5) / 10).astype(np.float32)


def crop512(image, x0, y0):
    out = np.full((512, 512), np.nan, np.float32)
    ny, nx = image.shape
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(nx, x0+512), min(ny, y0+512)
    if xa < xb and ya < yb:
        out[ya-y0:yb-y0, xa-x0:xb-x0] = image[ya:yb, xa:xb]
    return out


def run_case(out, pipe, cache, gaia, scaling, pointing, band, psf):
    name = f'pointing{pointing:04d}_{band}'
    dest = out/name; dest.mkdir(parents=True, exist_ok=True)
    path = RAW/f'Pointing_{pointing:04d}'/f'COSMOS_pointing_{pointing:04d}_{band.upper()}_detector_p001.fits'
    print(f'[{name}] catalog/WCS and full source filtering', flush=True)
    with fits.open(path, memmap=True) as hdus:
        hdu = next(h for h in hdus if h.header.get('NAXIS') == 2)
        image, header = hdu.data, hdu.header.copy()
        header_all = hdus[0].header.copy(); header_all.update(header)
        wcs = WCS(header).celestial
        pixscale = float(np.mean(proj_plane_pixel_scales(wcs))*3600)
        # Match the existing COSMOS catalog adapter on these north-up mosaics.
        matrix = wcs.pixel_scale_matrix
        if not np.allclose(matrix[[0,1],[1,0]], 0., atol=1e-12):
            raise ValueError('Rotated mosaic requires a WCS-transformed ellipse adapter')
        x,y = wcs.all_world2pix(cache['ra'],cache['dec'],0)
        ny,nx = image.shape
        idx = np.flatnonzero(np.isfinite(x+y)&(x>=0)&(y>=0)&(x<nx)&(y<ny))
        a,b = cache['kron2_a'][idx]/pixscale, cache['kron2_b'][idx]/pixscale
        geom = EllipseGeometry(x[idx],y[idx],a,b,np.deg2rad(cache['theta_world'][idx]),np.pi*a*b)
        data = OrdinaryInput(geom,cache[f'mag_auto_{band}'][idx],snr=cache[f'snr_{band}'][idx],
            flags=cache['warn_flag'][idx],star_mask=cache['flag_star'][idx],model_mag=cache[f'mag_model_{band}'][idx])
        basics = classify_catalog_basics(geom,data.mag,dataset='cosmos',pixel_scale_arcsec=pixscale,image=image)
        nan_ignore = basics.nan_center_ignore
        data.center_invalid = nan_ignore
        stages = {'A':np.where(basics.after_a & ~nan_ignore, int(C.CLEAN), basics.labels.source_class)}
        from preprocessing.utils.segmentation import cosmos_fill_ratios
        data.segmentation_fill_ratio = cosmos_fill_ratios(geom, cache['id'][idx], wcs,
            '/data/shared/jwst_foundation/catalog/COSMOS_1727_1837_5893/COSMOSWeb_mastercatalog_v1.1.fits',
            selected=basics.after_b_basic & ~nan_ignore & (data.mag < 25.5))
        ordinary = classify_ordinary_sources(data,basics.after_b_basic&~nan_ignore,basics.labels,dataset='cosmos')
        stages.update(ordinary.stages)
        bright = label_bright_sources(data,ordinary.labels,image_shape=image.shape,image_header=header_all,
            gaia_table=gaia,source_ids=cache['id'][idx],config=JWSTBrightConfig('cosmos',pixscale,psf,gaia_reference_epoch=2016.))
        final = bright.labels.source_class
        stages['final'] = final.copy()
        ids = cache['id'][idx]
        clean = geom.valid() & (final == C.CLEAN)
        criteria = {'a_gt_100':clean&(a>100), 'a_gt_150':clean&(a>150),
                    'sqrt_ab_gt_100':clean&(np.sqrt(a*b)>100)}
        candidates = np.flatnonzero(np.logical_or.reduce(list(criteria.values())))
        common.write_csv(dest/'sources.csv', (dict(id=int(ids[i]),ra=float(cache['ra'][idx[i]]),dec=float(cache['dec'][idx[i]]),
            x=float(geom.x[i]),y=float(geom.y[i]),a=float(a[i]),b=float(b[i]),theta_deg=float(np.degrees(geom.theta[i])),
            mag=float(data.mag[i]),model_mag=float(data.model_mag[i]),snr_value=float(data.snr[i]),warn_flag=int(data.flags[i]),
            catalog_star_mask=bool(data.star_mask[i]),**{s:common.NAMES[int(v[i])] for s,v in stages.items()},
            reason=str(bright.labels.reason[i])) for i in range(len(idx))))
        np.savez_compressed(dest/'source_stages.npz',ids=ids,**stages)
        common.write_csv(dest/'gaia_inserted.csv',(dict(id=int(s),x=float(x),y=float(y),reason=str(r))
            for s,x,y,r in zip(bright.strict_center_source_id,bright.strict_center_x,bright.strict_center_y,bright.strict_center_reason)))
        print(f'[{name}] sources={len(idx)} clean={int(clean.sum())} large={len(candidates)}; global RGB statistics',flush=True)
        factor,unit_meta = hsc_surface_brightness_factor(header)
        params = scaling.global_scaling_params(image, factor, 20.)
        (dest/'global_scaling.json').write_text(json.dumps(dict(params=params,units=unit_meta,
            clip=[-5,5],channels=['zscore','log','lupton'],Q=20,statistics_sigma=scaling.ZSCORE_CLIP_SIGMA),indent=2))
        view_rows=[]
        with (dest/'large_clean_views.reg').open('w') as reg:
            reg.write('# Region file format: DS9 version 4.1\nglobal color=green width=1\nimage\n')
            for i in candidates:
                cx,cy = float(geom.x[i]),float(geom.y[i])
                x0,y0 = int(np.floor(cx+.5))-256,int(np.floor(cy+.5))-256
                ct,st = np.cos(geom.theta[i]),np.sin(geom.theta[i])
                hx,hy = float(np.hypot(a[i]*ct,b[i]*st)),float(np.hypot(a[i]*st,b[i]*ct))
                def fits_box(xstart,ystart):
                    return cx-hx>=xstart-.5 and cx+hx<=xstart+511.5 and cy-hy>=ystart-.5 and cy+hy<=ystart+511.5
                centered_fits = fits_box(x0,y0)
                grid_fits = fits_box(int(np.floor(cx/512))*512,int(np.floor(cy/512))*512)
                raw = crop512(image,x0,y0)
                filled,fill_stats = fill_nonfinite_hybrid(raw*np.float32(factor))
                rgb = global_rgb(filled,params,scaling)
                rgb[~np.isfinite(raw)]=0
                zdisp = scaling.zscale_display(raw)
                prefix = f'id{int(ids[i])}'
                plt.imsave(dest/f'{prefix}_local_zscale.png',zdisp,origin='lower',cmap='gray',vmin=0,vmax=1)
                plt.imsave(dest/f'{prefix}_global_rgb.png',rgb,origin='lower')
                fig,axes=plt.subplots(1,2,figsize=(10.5,5.6),dpi=150)
                for ax,disp,title in zip(axes,(zdisp,rgb),('Local ZScale','Global zscore / log / Lupton')):
                    ax.imshow(disp,origin='lower',cmap='gray',vmin=0,vmax=1,interpolation='nearest')
                    ax.add_patch(Ellipse((cx-x0,cy-y0),2*a[i],2*b[i],angle=np.degrees(geom.theta[i]),
                        fill=False,edgecolor='lime',linewidth=.6))
                    ax.plot(cx-x0,cy-y0,'+',color='lime',ms=5,mew=.6)
                    ax.set_xlim(-.5,511.5);ax.set_ylim(-.5,511.5);ax.set_axis_off();ax.set_title(title,fontsize=12)
                fig.suptitle(f'{name} ID {int(ids[i])}  mag={data.mag[i]:.2f}  a={a[i]:.1f} b={b[i]:.1f} px\n'
                             f'ellipse inside 512: {centered_fits}; valid pixels: {np.isfinite(raw).mean():.1%}',fontsize=12)
                fig.tight_layout();fig.savefig(dest/f'{prefix}_comparison.png');plt.close(fig)
                reg.write(f'ellipse({cx+1:.5f},{cy+1:.5f},{a[i]:.5f},{b[i]:.5f},{np.degrees(geom.theta[i]):.5f}) # text={{ID {int(ids[i])}}}\n')
                reg.write(f'box({x0+256.5},{y0+256.5},512,512,0) # color=cyan\n')
                view_rows.append(dict(id=int(ids[i]),mag=float(data.mag[i]),a=float(a[i]),b=float(b[i]),sqrt_ab=float(np.sqrt(a[i]*b[i])),
                    x=cx,y=cy,x0=x0,y0=y0,ellipse_fits_centered=centered_fits,ellipse_fits_grid=grid_fits,
                    rescued_from_grid=bool(centered_fits and not grid_fits),valid_fraction=float(np.isfinite(raw).mean()),
                    no_data_le_10pct=bool(np.isfinite(raw).mean()>=.9),
                    **{k:bool(v[i]) for k,v in criteria.items()}))
        common.write_csv(dest/'large_clean_views.csv',view_rows)
        # Full-field overview only: no resampling is applied to any 512 view.
        ds=8; overview=np.asarray(image[::ds,::ds],np.float32)
        valid=np.isfinite(overview)
        flux=np.where(valid,overview*factor,params['zscore_median']).astype(np.float32)
        rgb=global_rgb(flux,params,scaling);rgb[~valid]=0
        plt.imsave(dest/'full_global_rgb_ds8.png',rgb,origin='lower')
        fig,ax=plt.subplots(figsize=(12,12),dpi=180)
        ax.imshow(rgb,origin='lower',interpolation='nearest')
        for row in view_rows:
            ax.add_patch(Rectangle((row['x0']/ds,row['y0']/ds),512/ds,512/ds,fill=False,edgecolor='lime',linewidth=.5))
            ax.text(row['x']/ds,row['y']/ds,str(row['id']),color='lime',fontsize=7)
        ax.set_axis_off();fig.tight_layout();fig.savefig(dest/'full_selected_views_ds8.png');plt.close(fig)
        report=dict(name=name,sources=len(idx),final_counts=common.counts(final),gaia_inserted=len(bright.strict_center_x),
            threshold_counts={k:int(v.sum()) for k,v in criteria.items()},unique_views=len(view_rows),
            centered_ellipse_fits=sum(r['ellipse_fits_centered'] for r in view_rows),
            rescued_from_grid=sum(r['rescued_from_grid'] for r in view_rows),
            no_data_le_10pct=sum(r['no_data_le_10pct'] for r in view_rows),
            per_threshold={k:dict(views=sum(r[k] for r in view_rows),fits=sum(r[k] and r['ellipse_fits_centered'] for r in view_rows),
                rescued=sum(r[k] and r['rescued_from_grid'] for r in view_rows)) for k in criteria},
            raw_fits=str(path),pixel_scale=pixscale,nan_center_ignored=int(nan_ignore.sum()),
            grid_definition='origin (0,0), stride 512; continuous ellipse bounds; centered view without shifting at image edge',
            units=unit_meta,global_rgb_stats_sigma=scaling.ZSCORE_CLIP_SIGMA,clip=[-5,5])
        (dest/'summary.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
        return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=common.BASE/'2026-09-09/cosmos_large_clean_centered_views')
    args=parser.parse_args();args.output_root.mkdir(parents=True,exist_ok=True)
    pipe=common.module(common.C_REF/'jwst_no_morph_snr_phot_containment_three_blocks.py','large_view_ref').load_pipeline()
    pipe.BANDS=tuple(b for _,b,_ in CASES)
    cache=pipe.load_catalog();gaia=Table.read(common.GAIA)
    scaling=common.module(SCALING,'large_view_global_scaling')
    reports=[]
    for pointing,band,psf in CASES:
        reports.append(run_case(args.output_root,pipe,cache,gaia,scaling,pointing,band,psf))
        (args.output_root/'summary.json').write_text(json.dumps(reports,indent=2))


if __name__=='__main__': main()
