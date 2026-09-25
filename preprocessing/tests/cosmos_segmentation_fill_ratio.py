"""Read-only segmentation/Kron coverage audit for four filtered COSMOS fields."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt

from preprocessing.tests import run_source_pipeline_preview as common

CATROOT=Path('/data/shared/jwst_foundation/catalog/COSMOS_1727_1837_5893')
INPUT=common.BASE/'2026-09-09/cosmos_large_clean_centered_views'
CASES=('pointing0017_f150w','pointing0019_f444w','pointing0004_f277w','pointing0025_f115w')
THRESHOLDS=(.05,.1,.15,.2,.3,.4,.5)


from preprocessing.utils.segmentation import measure, transform_ellipses


def plot(name, rows, out, population):
    select=np.array([r['population_after_a'] and (population=='after_A' or r['final']=='clean') for r in rows])
    mag=np.array([r['mag'] for r in rows]);own=np.array([r['own_fill'] for r in rows]);anyfill=np.array([r['any_fill'] for r in rows])
    large=np.array([r['a']>100 for r in rows]); valid=select&np.isfinite(own)
    fig,axes=plt.subplots(2,3,figsize=(16,9),dpi=160)
    summaries=[]
    for row,(title,subset) in enumerate([('Bright: mag < 25.5',mag<25.5),('Faint: mag >= 25.5',mag>=25.5)]):
        use=valid&subset; values=own[use]; other=anyfill[use]
        bins=np.linspace(0,1,51); weights=np.ones(len(values))/max(len(values),1)
        ax=axes[row,0]
        ax.hist(values,bins=bins,weights=weights,histtype='step',color='#1261b5',lw=1.5,label=f'Own segment (n={len(values)})')
        ax.hist(other,bins=bins,weights=weights,histtype='step',color='#b25b00',lw=1.2,label='Any segment')
        ax.set_xlabel('Fill ratio');ax.set_ylabel('Fraction per bin');ax.legend(fontsize=10);ax.set_title(title)
        ax=axes[row,1]
        for mask,color,label in [(use,'#1261b5','All'),(use&large,'#b51f5a','a > 100 px')]:
            v=np.sort(own[mask]);ax.step(v,np.arange(1,len(v)+1)/max(len(v),1),where='post',color=color,label=f'{label} (n={len(v)})')
        ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_xlabel('Own-segment fill ratio');ax.set_ylabel('Cumulative fraction');ax.legend(fontsize=10)
        ax=axes[row,2]
        h=ax.hexbin(mag[use],own[use],gridsize=(60,40),extent=(18,32,0,1),bins='log',mincnt=1,cmap='viridis')
        ax.scatter(mag[use&large],own[use&large],s=14,facecolors='none',edgecolors='#ed2878',linewidths=.7)
        ax.set_xlim(18,32);ax.set_ylim(0,1);ax.set_xticks(np.arange(18,33,2));ax.set_xlabel('Kron magnitude');ax.set_ylabel('Own-segment fill ratio')
        fig.colorbar(h,ax=ax,label='Source count')
        info=dict(name=name,population=population,brightness='bright' if row==0 else 'faint',count=int(use.sum()),
            zero_own=int(np.count_nonzero(use&(own==0))),large_count=int(np.count_nonzero(use&large)),
            quantiles=dict(zip(['p05','p25','p50','p75','p95'],map(float,np.quantile(values,[.05,.25,.5,.75,.95])))) if len(values) else {})
        for t in THRESHOLDS:
            info[f'own_below_{t}']=int(np.count_nonzero(use&(own<t)))
            info[f'any_below_{t}']=int(np.count_nonzero(use&(anyfill<t)))
            info[f'large_own_below_{t}']=int(np.count_nonzero(use&large&(own<t)))
        summaries.append(info)
    fig.suptitle(f'{name}: {population}, original Kron ellipse, no fill-ratio filtering',fontsize=13)
    fig.tight_layout();fig.savefig(out/f'{name}_{population}_fill_ratio.png');plt.close(fig)
    return summaries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=common.BASE/'2026-09-09/cosmos_segmentation_fill_ratio')
    args=parser.parse_args();out=args.output_root;out.mkdir(parents=True,exist_ok=True)
    with fits.open(CATROOT/'COSMOSWeb_mastercatalog_v1.1.fits',memmap=True) as hs:
        tab=hs[1].data
        ids=np.array(tab['id']);order=np.argsort(ids); sorted_ids=ids[order]
        segids=np.array(tab['segment-id']);tiles=np.array(tab['tile']).astype(str);segarea=np.array(tab['seg_area'])
    summaries=[];audits=[]
    for name in CASES:
        print(f'[{name}] reading frozen source classifications',flush=True)
        source_rows=common.read_csv(INPUT/name/'sources.csv');meta=json.loads((INPUT/name/'summary.json').read_text())
        sid=np.array([int(r['id']) for r in source_rows]);ci=order[np.searchsorted(sorted_ids,sid)]
        assert np.array_equal(ids[ci],sid)
        with fits.open(meta['raw_fits'],memmap=True) as hs:
            h=next(h for h in hs if h.header.get('NAXIS')==2);wcs=WCS(h.header).celestial
        rows=[]
        for j,r in enumerate(source_rows):
            rows.append(dict(id=int(r['id']),tile=str(tiles[ci[j]]),segment_id=int(segids[ci[j]]),
                mag=float(r['mag']),a=float(r['a']),b=float(r['b']),theta=float(r['theta_deg']),x=float(r['x']),y=float(r['y']),
                final=r['final'],warn_flag=int(r['warn_flag']),catalog_seg_area=float(segarea[ci[j]]),
                population_after_a=r['A']=='clean' and np.isfinite(float(r['mag'])) and float(r['mag'])<90,
                own_fill=np.nan,any_fill=np.nan,own_fill_analytic=np.nan,coverage_fraction=np.nan))
        nmeasured=0
        with ExitStack() as stack:
            for tile in sorted(set(r['tile'] for r in rows)):
                inds=[i for i,r in enumerate(rows) if r['tile']==tile and r['population_after_a']]
                if not inds: continue
                path=CATROOT/'segmentation_maps'/f'detection_chi2pos_SWLW_{tile}_segmap_v1.3.fits'
                hs=stack.enter_context(fits.open(path,memmap=True,do_not_scale_image_data=True))
                h=next(h for h in hs if h.header.get('NAXIS')==2)
                get=lambda k:np.array([rows[i][k] for i in inds])
                mapped=transform_ellipses(wcs,WCS(h.header).celestial,get('x'),get('y'),get('a'),get('b'),np.deg2rad(get('theta')))
                for j,i in enumerate(inds):
                    cx,cy,a,b,theta=(v[j] for v in mapped)
                    if not np.all(np.isfinite([cx,cy,a,b,theta])) or min(a,b)<=0: continue
                    result=measure(h.data,h.header,cx,cy,a,b,theta,rows[i]['segment_id'])
                    rows[i].update(result);nmeasured+=1
                print(f'[{name}] {tile}: {len(inds)} sources',flush=True)
        common.write_csv(out/f'{name}_source_fill_ratios.csv',rows)
        common.write_csv(out/f'{name}_large_clean_fill_ratios.csv',[r for r in rows if r['final']=='clean' and r['a']>100])
        for pop in ('clean','after_A'): summaries.extend(plot(name,rows,out,pop))
        valid=[r for r in rows if np.isfinite(r['own_fill'])]
        audit=dict(name=name,sources=len(rows),measured=nmeasured,fully_covered=len(valid),
            center_matches=sum(r.get('center_segment')==r['segment_id'] for r in valid),
            own_segment_present=sum(r['own_pixels']>0 for r in valid),
            missing_or_partial_coverage=nmeasured-len(valid))
        audits.append(audit);print(json.dumps(audit),flush=True)
        (out/'summary.json').write_text(json.dumps(dict(distributions=summaries,map_audits=audits),indent=2))
        common.write_csv(out/'distribution_summary.csv',summaries)


if __name__=='__main__': main()
