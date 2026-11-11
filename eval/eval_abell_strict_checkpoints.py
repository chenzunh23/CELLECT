#!/usr/bin/env python3
"""Evaluate all checkpoints against one common Abell Zarr validation selection."""
import argparse,csv,json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import torch
from torch.utils.data import DataLoader
from collections import Counter,defaultdict
from astro_train_zarr_data import discover_zarr_image_records,ZarrCutoutDataset
from astro_train_data import collate_cutouts
from utils.train_eval_utils import parse_patch_specs,filter_records_by_patches
from eval.eval_utils import load_cellect_model,infer_cellect,select_band_outputs,detection_rows
from eval.strict_center_metrics import point_metrics,finalize


def curves(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[json.loads(p.read_text()) for p in sorted(output.glob('*/*/metrics.json'))]
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for run in sorted({r['run'] for r in rows}):
        selected=sorted((r for r in rows if r['run']==run),key=lambda r:r['epoch'])
        for ax,key in zip(axes.flat,('strict_completeness','inserted_completeness','combined_completeness','combined_purity')):
            ax.plot([r['epoch'] for r in selected],[r['overall'][key] if r['overall'][key] is not None else np.nan for r in selected],'.-',label=run)
            ax.set(title=key,xlabel='checkpoint epoch',ylim=(0,1));ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(output/'completeness_purity.png',dpi=170);plt.close(fig)
    bands=sorted({b for r in rows for b in r['per_band']})
    for metric in ('strict_completeness','inserted_completeness','combined_completeness','combined_purity'):
        fig,axes=plt.subplots((len(bands)+3)//4,4,figsize=(16,3*((len(bands)+3)//4)),squeeze=False,layout='constrained')
        for ax,b in zip(axes.flat,bands):
            for run in sorted({r['run'] for r in rows}):
                selected=sorted((r for r in rows if r['run']==run and b in r['per_band']),key=lambda r:r['epoch'])
                ax.plot([r['epoch'] for r in selected],[r['per_band'][b][metric] if r['per_band'][b][metric] is not None else np.nan for r in selected],'.-',label=run)
            ax.set(title=b,ylim=(0,1),xlabel='epoch');ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(bands):]:ax.set_visible(False)
        axes.flat[0].legend(fontsize=6);fig.suptitle(metric);fig.savefig(output/(metric+'_per_band.png'),dpi=150);plt.close(fig)
    flat=[dict(run=r['run'],epoch=r['epoch'],**r['overall']) for r in rows]
    if flat:
        with (output/'summary.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-dir',action='append',required=True,type=Path)
    p.add_argument('--root',type=Path,help='Common NEW-label Zarr root for ALL models; default first run config root')
    p.add_argument('--selectors',type=Path,default=Path('/home/czh23/CELLECT/patches/val_zarr_sky.txt'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--amp',choices=['none','bf16'],default='bf16');p.add_argument('--match-radius',type=float,default=3)
    p.add_argument('--epochs',type=int,nargs='*');p.add_argument('--max-records',type=int)
    p.add_argument('--nms-radius',type=int,help='Override checkpoint config; browser currently uses 2')
    p.add_argument('--center-refinement',choices=['integer','softargmax'],help='Override checkpoint config')
    p.add_argument('--preview-per-checkpoint',type=int,default=6);p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    configs={d:json.loads((d/'run_config.json').read_text())['args'] for d in a.checkpoint_dir}
    root=(a.root or Path(configs[a.checkpoint_dir[0]]['root'])).resolve()
    records=discover_zarr_image_records(root/'abell' if (root/'abell').is_dir() else root,bands=[])
    records=[r for r in records if r.tract=='Abell2744' or 'Abell2744/' in r.name]
    records=filter_records_by_patches(records,parse_patch_specs([],str(a.selectors)),root)
    if a.max_records:records=records[:a.max_records]
    if not records:raise ValueError('No exact Abell validation records selected')
    plan=dict(root=str(root),selectors=str(a.selectors),records=len(records),match_radius=a.match_radius,
        gt_policy='same new-label root for all models; base=shape-supervised centers; strict=sparse strict centers; inserted=compact-added ID namespace',
        matching='global distance-sorted one-to-one union match; no prediction counted twice; base-only legacy score also reported',
        purity='catalog matching purity; unmatched predictions are not independently verified false sources',
        sample_policy='metrics per selected Zarr cutout, including overlapping/large-source special cutouts',
        selection_names=[r.name for r in records])
    selection_sha=hashlib.sha256(json.dumps([plan,a.nms_radius,a.center_refinement],sort_keys=True).encode()).hexdigest()
    plan['selection_sha256']=selection_sha
    (a.out/'selection.json').write_text(json.dumps(plan,indent=2))
    checkpoints=[]
    for directory in a.checkpoint_dir:
        for ck in sorted(directory.glob('epoch_*.pt')):
            epoch=int(ck.stem.split('_')[-1])
            if a.epochs is None or epoch in a.epochs:checkpoints.append((directory,ck,epoch))
    print(json.dumps(dict(records=len(records),checkpoints=len(checkpoints),root=str(root))),flush=True)
    if a.dry_run:return
    ds=ZarrCutoutDataset(records,augment=False)
    loader=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=a.workers,collate_fn=collate_cutouts,
        multiprocessing_context='spawn' if a.workers else None,pin_memory=str(a.device).startswith('cuda'))
    for directory,ck,epoch in checkpoints:
        dest=a.out/directory.name/ck.stem;dest.mkdir(parents=True,exist_ok=True)
        done=dest/'metrics.json'
        if done.exists():
            if json.loads(done.read_text()).get('selection_sha256')!=selection_sha:raise ValueError('Existing metrics use a different selection/config; choose another --out')
            print('skip',ck,flush=True);continue
        cfg=configs[directory];model,_=load_cellect_model(ck,None,torch.device(a.device),['F115W'])
        totals=Counter();bands=defaultdict(Counter);per=[];sample_index=0;previews=0
        for batch in loader:
            outputs=infer_cellect(model=model,image_tensor=batch['image'],device=torch.device(a.device),amp=a.amp)
            for i in range(len(batch['image'])):
                rec=records[sample_index];sample_index+=1
                out={k:v[i:i+1] for k,v in outputs.items() if torch.is_tensor(v) and v.shape[0]==len(batch['image'])}
                out=select_band_outputs(out,0);heat='offset' in out
                rows=detection_rows(out,threshold=.5 if heat else float(cfg.get('confidence_threshold',2)),nms_radius=1 if heat else (a.nms_radius if a.nms_radius is not None else int(cfg.get('nms_radius',1))),
                    confidence_score='heatmap' if heat else cfg.get('confidence_score','ordinal_expectation'),center_refinement=a.center_refinement or cfg.get('center_refinement','integer'),
                    center_refinement_radius=int(cfg.get('center_refinement_radius',1)),width=batch['image'].shape[-1],height=batch['image'].shape[-2],
                    valid_mask=batch['band_valid_mask'][i,0].numpy())
                pred=np.array([[r['x'],r['y']] for r in rows]).reshape(-1,2)
                base=batch['band_centers'][i][0].numpy();strict=batch['band_strict_center_only_centers'][i][0].numpy();ids=batch['band_strict_center_only_ids'][i][0].numpy()
                count,matches,gt=point_metrics(pred,base,strict,ids,a.match_radius)
                totals.update(count);band=batch['band_names'][i][0];bands[band].update(count)
                per.append(dict(name=rec.name,band=band,**finalize(count)))
                if previews<a.preview_per_checkpoint and count['strict_gt']:
                    import matplotlib.pyplot as plt
                    image=batch['image'][i,0].numpy().mean(0);hit={g for _,g,_ in matches}
                    fig,ax=plt.subplots(figsize=(8,8));ax.imshow(image,origin='lower',cmap='gray')
                    if len(pred):ax.scatter(*pred.T,s=20,facecolors='none',edgecolors='cyan',label='predictions')
                    for j,q in enumerate(gt):ax.plot(*q,marker='+' if j<len(base) else 'x',color='lime' if j in hit else 'red',ms=7)
                    ax.set(title=rec.name+'\nGreen=matched, red=missed; +=base, x=strict');fig.savefig(dest/f'preview_{previews:02d}.png',dpi=140);plt.close(fig);previews+=1
            print(f'{directory.name} {ck.stem}: {sample_index}/{len(records)}',flush=True)
        result=dict(run=directory.name,epoch=epoch,checkpoint=str(ck),root=str(root),selection_sha256=selection_sha,overall=finalize(totals),per_band={b:finalize(c) for b,c in bands.items()})
        (dest/'samples.json').write_text(json.dumps(per,indent=2));temp=dest/'metrics.tmp';temp.write_text(json.dumps(result,indent=2));temp.replace(done)
        del model
        if torch.cuda.is_available():torch.cuda.empty_cache()
        curves(a.out)
if __name__=='__main__':main()
