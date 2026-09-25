#!/usr/bin/env python
"""Prepare reproducible physical-field splits; metadata only, no image processing."""
import argparse
from collections import defaultdict, Counter
import json
from pathlib import Path
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing.make_zarr_patch_splits import scan_stores, build_candidates, validation_exclusions, _write_selectors


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20260925)
    p.add_argument('--val-fraction',type=float,default=.1)
    args=p.parse_args()
    if not 0 < args.val_fraction < 1: p.error('val-fraction must be between 0 and 1')
    stores=scan_stores(args.root,[],[])
    if not stores: raise RuntimeError('No completed stores')
    modes=Counter(json.loads((s.path/'.zattrs').read_text()).get('confidence_config',{}).get('mode','missing') for s in stores)
    if set(modes)!={'psf-ee'}: raise RuntimeError(f'Expected PSF-EE targets in every store: {modes}')
    candidates=build_candidates(stores,requested_bands=[],narrow_bands=[],narrow_weight_floor=1.,
        narrow_weight_power=1.,narrow_weight_mode='sum-tiles',disable_narrow_downweight=True,
        min_bands=1,require_all_bands=False,include_patches=set(),exclude_patches=set())
    groups=defaultdict(set)
    for c in candidates: groups[c.dataset].add(c.physical_patch)
    rng=random.Random(args.seed);chosen=set()
    for dataset,fields in sorted(groups.items()):
        fields=sorted(fields)
        chosen.update(rng.sample(fields,max(1,round(len(fields)*args.val_fraction))))
    val=[c for c in candidates if c.physical_patch in chosen]
    blocked=validation_exclusions(candidates,val)
    train=[c for c in candidates if c.patch not in blocked]
    if not train or not val: raise RuntimeError('Empty split')
    assert not {c.physical_patch for c in train}&{c.physical_patch for c in val}
    summary={'root':str(args.root.resolve()),'seed':args.seed,'val_fraction_groups':args.val_fraction,
             'confidence_modes':dict(modes),'val_physical_groups':sorted(chosen),
             'policy':'Tie HSC half/noisy by patch; tie COSMOS proposals/centered views by pointing; Abell overlap guard 256px on both sides.'}
    for name,items in [('train',train),('val',val)]:
        bands=Counter();datasets=Counter()
        for c in items:
            bands.update(c.samples_by_band);datasets[c.dataset]+=c.sample_count
        summary[name]={'selectors':len(items),'samples':sum(bands.values()),'samples_by_band':dict(sorted(bands.items())),
            'samples_by_dataset':dict(datasets),'physical_groups':len({c.physical_patch for c in items})}
    summary['excluded_guard_selectors']=len(candidates)-len(train)-len(val)
    summary['val_missing_bands']=sorted({s.band for s in stores}-set(summary['val']['samples_by_band']))
    args.out_dir.mkdir(parents=True,exist_ok=True)
    _write_selectors(args.out_dir/'train.txt',train);_write_selectors(args.out_dir/'val.txt',val)
    (args.out_dir/'split_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
