"""Resolve fixed gradient probes without changing any stored training sample."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from functools import lru_cache

import numpy as np


def field(record, key, default=''):
    return record.get(key, default) if isinstance(record, dict) else getattr(record, key, default)


def record_band(record):
    bands = field(record, 'bands', ())
    if len(bands) == 1:
        return str(bands[0]).upper()
    text = str(field(record, 'relative_root')) + '/' + str(field(record, 'name'))
    matches = re.findall(r'(?:^|/)(F\d{3}[WM]|HSC-[A-Z]|NB\d{4})(?=/|$)', text.upper())
    return matches[0] if matches else ''


def record_description(record, index):
    height,width=record_hw(record)
    return dict(index=int(index), name=str(field(record,'name')), band=record_band(record),
                patch=str(field(record,'patch')), dataset_source=str(field(record,'dataset_source')),
                x0=int(field(record,'x0',0)), y0=int(field(record,'y0',0)),
                width=width,height=height,
                image_paths=list(field(record,'image_paths',())))


@lru_cache(maxsize=4096)
def _store_hw(uri):
    path=Path(str(uri).removeprefix('zarr://').rsplit('#',1)[0])/'band_confidence'/'.zarray'
    if path.is_file():
        shape=json.loads(path.read_text())['shape']
        return int(shape[-2]),int(shape[-1])
    return 512,512


def record_hw(record):
    paths=field(record,'image_paths',())
    return _store_hw(str(paths[0]).rsplit('#',1)[0]) if paths else (512,512)


def resolve_probe_config(records, groups, config_path, *, seed, default_samples):
    raw = Path(config_path).expanduser().read_bytes()
    config = json.loads(raw)
    if config.get('version',1) != 1:
        raise ValueError('Unsupported gradient probe config version')
    allowed = {'version','random_samples_per_dataset','datasets','report_cohorts'}
    if set(config)-allowed:
        raise ValueError(f'Unknown gradient probe config keys: {sorted(set(config)-allowed)}')
    specs = config.get('datasets',{})
    if set(specs)-set(groups):
        raise ValueError(f'Probe datasets are absent from the training pool: {sorted(set(specs)-set(groups))}')
    fallback = int(config.get('random_samples_per_dataset',default_samples))
    if fallback < 0:
        raise ValueError('random_samples_per_dataset must be >=0')
    rng = np.random.default_rng(seed)
    indices={};cohorts={};selected_by={};cases=[];ids=set()
    for family,pool in sorted(groups.items()):
        spec=specs.get(family,{})
        if set(spec)-{'random_samples','probes'}:
            raise ValueError(f'Unknown probe dataset options for {family}')
        count=int(spec.get('random_samples',fallback))
        if count < 0:
            raise ValueError('random_samples must be >=0; zero means no random supplement')
        chosen=[]
        for case in spec.get('probes',[]):
            allowed={'id','description','cohort','name','patch','band','dataset_source','group',
                     'grid','grid_size','x0','y0','size','selection','max_tiles','min_overlap','include_large_source'}
            if set(case)-allowed:
                raise ValueError(f'Unknown probe selector keys: {sorted(set(case)-allowed)}')
            case_id=str(case.get('id',''))
            if not re.fullmatch(r'[A-Za-z0-9_.-]+',case_id) or case_id in ids:
                raise ValueError(f'Probe id must be unique and path-safe: {case_id!r}')
            ids.add(case_id)
            matches=[]
            for i in pool:
                rec=records[i]
                if case.get('name') is not None and field(rec,'name') != case['name']:continue
                if case.get('patch') is not None and field(rec,'patch') != case['patch']:continue
                if case.get('band') is not None and record_band(rec) != case['band'].upper():continue
                if case.get('dataset_source') is not None and field(rec,'dataset_source') != case['dataset_source']:continue
                if case.get('group') is not None:
                    paths=' '.join(field(rec,'image_paths',()))
                    group=str(case['group'])
                    if not re.search(r'__'+re.escape(group)+r'\.zarr(?:#|$)',paths):continue
                if not case.get('include_large_source',False) and not case.get('name'):
                    name=str(field(rec,'tile_name') or field(rec,'name')).lower()
                    if 'large' in name or 'centered' in name:continue
                matches.append(i)
            has_region='grid' in case or 'x0' in case or 'y0' in case
            if 'grid' in case and ('x0' in case or 'y0' in case):
                raise ValueError(f'{case_id}: use grid or x0/y0, not both')
            bounds=None;scores={}
            if has_region:
                if not case.get('patch') or not case.get('band'):
                    raise ValueError(f'{case_id}: spatial selectors require patch and band')
                if 'grid' in case:
                    grid=case['grid'];grid_size=int(case.get('grid_size',512))
                    if len(grid)!=2 or any(int(v)!=v or v<0 for v in grid) or grid_size<=0:
                        raise ValueError(f'{case_id}: grid must be [x,y] nonnegative integers')
                    x,y=int(grid[0])*grid_size,int(grid[1])*grid_size
                    size=int(case.get('size',grid_size))
                else:
                    if 'x0' not in case or 'y0' not in case:
                        raise ValueError(f'{case_id}: both x0 and y0 are required')
                    x,y=int(case['x0']),int(case['y0']);size=int(case.get('size',512))
                if size<=0:raise ValueError(f'{case_id}: size must be positive')
                bounds=[x,y,x+size,y+size]
                mode=case.get('selection','overlap')
                if mode not in ('overlap','exact'):
                    raise ValueError(f'{case_id}: selection must be overlap or exact')
                minimum=float(case.get('min_overlap',.25))
                if not 0<minimum<=1:raise ValueError(f'{case_id}: min_overlap must be in (0,1]')
                eligible=[]
                for i in matches:
                    rec=records[i];rx,ry=int(field(rec,'x0',0)),int(field(rec,'y0',0))
                    height,width=record_hw(rec)
                    overlap=max(0,min(x+size,rx+width)-max(x,rx))*max(0,min(y+size,ry+height)-max(y,ry))
                    fraction=overlap/(size*size)
                    if mode=='exact' and (rx,ry,width,height)!=(x,y,size,size):continue
                    if fraction<minimum:continue
                    scores[i]=fraction;eligible.append(i)
                matches=sorted(eligible,key=lambda i:(-scores[i],
                    (int(field(records[i],'x0',0))+record_hw(records[i])[1]/2-(x+size/2))**2+
                    (int(field(records[i],'y0',0))+record_hw(records[i])[0]/2-(y+size/2))**2,str(field(records[i],'name'))))
                limit=int(case.get('max_tiles',1))
                if limit<1:raise ValueError(f'{case_id}: max_tiles must be >=1')
                matches=matches[:limit]
            elif case.get('name'):
                if len(matches)>1:raise ValueError(f'{case_id}: sample name is ambiguous')
            else:
                raise ValueError(f'{case_id}: specify a saved name or spatial window')
            if not matches:
                raise ValueError(f'No training Zarr samples match probe {case_id}: {case}')
            cohort=case.get('cohort')
            if cohort:
                if not re.fullmatch(r'[A-Za-z0-9_.-]+',cohort) or '__' in cohort or cohort in groups or cohort=='full':
                    raise ValueError(f'Invalid/reserved probe cohort: {cohort!r}')
                cohorts.setdefault(cohort,[]).extend(matches)
            descriptions=[]
            for i in matches:
                selection=dict(id=case_id,cohort=cohort,description=case.get('description',''),
                               requested_bounds=bounds,selection=case.get('selection','saved_name'),
                               overlap_fraction=scores.get(i,1.))
                selected_by.setdefault(i,[]).append(selection)
                descriptions.append(dict(record_description(records[i],i),**selection))
            cases.append(dict(id=case_id,dataset=family,matches=descriptions))
            chosen.extend(matches)
        chosen=list(dict.fromkeys(chosen))
        chosen_set=set(chosen)
        remaining=[i for i in pool if i not in chosen_set]
        if count:
            chosen.extend(sorted(rng.choice(remaining,size=min(count,len(remaining)),replace=False).tolist()))
        if not chosen:
            raise ValueError(f'No gradient probes selected for {family}; full mixture would be undefined')
        indices[family]=chosen
    cohorts={k:list(dict.fromkeys(v)) for k,v in cohorts.items()} if config.get('report_cohorts',True) else {}
    manifest=dict(version=1,config_path=str(Path(config_path).expanduser().resolve()),
                  config_sha256=hashlib.sha256(raw).hexdigest(),
                  coordinate_note='grid [x,y] uses grid_size pixels; overlap selects original stored tiles, never synthesizes labels',
                  case_matches=cases,cohorts={k:[record_description(records[i],i) for i in v] for k,v in cohorts.items()},
                  dataset_probes={k:[record_description(records[i],i) for i in v] for k,v in indices.items()})
    return indices,cohorts,selected_by,manifest


def main():
    parser=argparse.ArgumentParser(description='Resolve gradient probe config without loading a model or starting training')
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--train-file',required=True)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=7)
    args=parser.parse_args()
    from astro_train_zarr_data import discover_zarr_image_records
    from utils.train_eval_utils import filter_records_by_patches,parse_patch_specs
    from utils.gradient_debug import dataset_family
    from collections import defaultdict
    records=discover_zarr_image_records(args.root,bands=[])
    records=filter_records_by_patches(records,parse_patch_specs([],args.train_file),args.root,seed=args.seed)
    groups=defaultdict(list)
    for i,rec in enumerate(records):groups[dataset_family(rec)].append(i)
    indices,cohorts,_,manifest=resolve_probe_config(records,groups,args.config,seed=args.seed,default_samples=2)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps(dict(probes={k:len(v) for k,v in indices.items()},
                          cohorts={k:len(v) for k,v in cohorts.items()},output=str(args.output)),ensure_ascii=False))


if __name__=='__main__':main()
