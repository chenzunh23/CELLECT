"""Read-only, exact gradient geometry on fixed dataset probes.

The full mixture is a record-count-weighted sum of per-dataset probe means.
samples_per_dataset=0 traverses every training record at fixed model weights.
No random projections, optimizer steps, or writes to Parameter.grad are used.
"""
from __future__ import annotations

import copy
import itertools
import json
import math
import random
import re
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from utils.gradient_debug_probes import resolve_probe_config


def dataset_family(record):
    def value(key):
        return record.get(key, '') if isinstance(record, dict) else getattr(record, key, '')
    def classify(text):
        # Match dataset components, never substrings of an experiment directory
        # such as training_zarr_cosmos2_gaia24, which also contains HSC stores.
        tokens = re.split(r"[^a-z0-9_-]+", str(text).lower())
        for token in tokens:
            if token in ('abell', 'abell2744', 'a2744'):
                return 'abell'
            if token in ('cosmos', 'proposal_1727', 'proposal_5893') or re.fullmatch(r'cosmos_\d+', token):
                return 'cosmos'
            if token in ('hsc', 'subaru') or token.startswith(('hsc-', 'nb0', 'nb1')):
                return 'hsc'
        return None
    # Relative metadata identifies the record independently of its parent path.
    for key in ('relative_root', 'name', 'bands', 'image_paths'):
        family = classify(value(key))
        if family is not None:
            return family
    return 'other'


def parameter_part(name):
    if name.startswith('encoder.'):
        return 'encoder'
    if name.startswith('decoder.'):
        return 'proposal_decoder'
    if name.startswith(('mask_decoder.', 'prompt_encoder.')):
        return 'mask'
    return 'other'


@contextmanager
def preserve_state(model, device, seed):
    """Eval-mode gradients with deterministic prompts; preserve caller RNG/modes."""
    modes = [(m, m.training) for m in model.modules()]
    py_state, np_state = random.getstate(), np.random.get_state()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed);np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            if devices:torch.cuda.manual_seed(seed)  # Current GPU only; never seed/init other ranks' GPUs.
            model.eval()
            with torch.enable_grad():
                yield
    finally:
        for module, training in modes:
            module.training = training
        random.setstate(py_state);np.random.set_state(np_state)


def gradient_grams(vectors, parts):
    """Exact CPU dot products, accumulated by disjoint parameter blocks."""
    count = len(vectors)
    grams = {part: np.zeros((count,count),np.float64) for part in ('encoder','proposal_decoder','mask','other')}
    for name, part in parts.items():
        active = [(i, v[name].reshape(-1)) for i,v in enumerate(vectors) if name in v]
        for offset,(i,a) in enumerate(active):
            for j,b in active[offset:]:
                dot = float(torch.dot(a,b))
                grams[part][i,j] += dot
                if i != j:
                    grams[part][j,i] += dot
    grams['proposal'] = grams['encoder'] + grams['proposal_decoder']
    grams['full_model'] = sum(grams[k] for k in ('encoder','proposal_decoder','mask','other'))
    return grams


def geometry_metrics(grams, datasets, proportions):
    n = len(datasets);coefficients = {}
    for i,dataset in enumerate(datasets):
        a=np.zeros(2*n);a[2*i]=1
        b=np.zeros(2*n);b[2*i+1]=1
        coefficients[dataset]={'proposal_loss':a,'mask_loss':b,'total':a+b}
    coefficients['full']={key:sum(proportions[d]*coefficients[d][key] for d in datasets)
                          for key in ('proposal_loss','mask_loss','total')}
    result={}
    def norm(g,a):return math.sqrt(max(0.,float(a@g@a)))
    def angle(g,a,b,prefix):
        na,nb=norm(g,a),norm(g,b)
        valid=na>1e-20 and nb>1e-20 and np.isfinite(na+nb)
        result[prefix+'/defined']=float(valid)
        if valid:
            cosine=float(np.clip((a@g@b)/(na*nb),-1,1))
            result[prefix+'/cosine']=cosine
            result[prefix+'/angle_deg']=math.degrees(math.acos(cosine))
    for scope in ('mask','proposal','proposal_decoder','encoder','full_model'):
        g=grams[scope]
        for component in ('proposal_loss','mask_loss','total'):
            for dataset,values in coefficients.items():
                result[f'{scope}/{component}/norm/{dataset}']=norm(g,values[component])
            for a,b in itertools.combinations(coefficients,2):
                angle(g,coefficients[a][component],coefficients[b][component],f'{scope}/{component}/{a}__{b}')
        for dataset,values in coefficients.items():
            angle(g,values['proposal_loss'],values['mask_loss'],f'{scope}/mask_vs_proposal/{dataset}')
    return result


class GradientDebugger:
    def __init__(self, dataset, records, collate_fn, output_dir, *, interval=500,
                 samples_per_dataset=2, batch_size=1, seed=7, log_fn=None,
                 hang_timeout_seconds=120, probe_config=None):
        if interval<1 or samples_per_dataset<0 or batch_size<1:
            raise ValueError('Gradient debug requires interval>=1, samples>=0, batch_size>=1')
        self.dataset=dataset;self.collate_fn=collate_fn;self.output_dir=Path(output_dir)
        self.interval=interval;self.batch_size=batch_size;self.seed=seed;self.log_fn=log_fn
        self.hang_timeout_seconds=hang_timeout_seconds
        self.watchdog=None
        self.groups=defaultdict(list)
        for index,record in enumerate(records):self.groups[dataset_family(record)].append(index)
        total=sum(map(len,self.groups.values()))
        self.proportions={k:len(v)/max(total,1) for k,v in self.groups.items()}
        rng=np.random.default_rng(seed)
        self.indices={key:(sorted(rng.choice(values,size=min(samples_per_dataset,len(values)),replace=False).tolist())
                           if samples_per_dataset else values) for key,values in sorted(self.groups.items())}
        self.cohorts={};self.selected_by={};self.probe_manifest=None
        if probe_config:
            if hasattr(dataset,'augment'):
                self.dataset=copy.copy(dataset)
                self.dataset.augment=False
            self.indices,self.cohorts,self.selected_by,self.probe_manifest=resolve_probe_config(
                records,self.groups,probe_config,seed=seed,default_samples=samples_per_dataset)
            if self.cohorts and batch_size != 1:
                raise ValueError('Named probe cohorts require gradient-debug-batch-size=1 for exact per-cohort gradients')
            if not dist.is_available() or not dist.is_initialized() or dist.get_rank()==0:
                self.output_dir.mkdir(parents=True,exist_ok=True)
                (self.output_dir/'probes_resolved.json').write_text(
                    json.dumps(self.probe_manifest,ensure_ascii=False,indent=2))
        self.membership=defaultdict(list)
        for cohort,indices in self.cohorts.items():
            for index in indices:self.membership[index].append(cohort)
        self.cache={} if probe_config or samples_per_dataset else None
        self.full_pass=not probe_config and samples_per_dataset==0

    def maybe_probe(self, model, optimizer, *, optimizer_step, iteration, epoch,
                    device, loss_fn, frozen_prefixes=()):
        if optimizer_step % self.interval:
            return None
        distributed=dist.is_available() and dist.is_initialized()
        if distributed:dist.barrier()
        result=None
        if not distributed or dist.get_rank()==0:
            try:
                print(f'[gradient_debug] start iteration={iteration} full_pass={self.full_pass} '
                      f'probe_samples={ {k:len(v) for k,v in self.indices.items()} }',flush=True)
                result=self.measure(model,optimizer,device=device,loss_fn=loss_fn,
                                    frozen_prefixes=frozen_prefixes)
                result.update(iteration=int(iteration),optimizer_step=int(optimizer_step),epoch=int(epoch))
                self.output_dir.mkdir(parents=True,exist_ok=True)
                with (self.output_dir/'metrics.jsonl').open('a') as handle:
                    handle.write(json.dumps(result,allow_nan=False)+'\n')
                if self.log_fn:self.log_fn(result['metrics'],iteration,epoch)
                print(f'[gradient_debug] iteration={iteration} datasets={list(self.indices)} '
                      f'probe_samples={result["probe_samples"]} seconds={result["seconds"]:.2f}',flush=True)
            except Exception as exc:
                # Diagnostics must not strand the other ranks at the next barrier.
                import traceback
                traceback.print_exc()
                self.output_dir.mkdir(parents=True,exist_ok=True)
                with (self.output_dir/'errors.jsonl').open('a') as handle:
                    handle.write(json.dumps(dict(iteration=iteration,error=repr(exc)))+'\n')
                print(f'[gradient_debug] failed: {exc}',flush=True)
        if distributed:dist.barrier()
        return result

    def measure(self, model, optimizer, *, device, loss_fn, frozen_prefixes=()):
        started=time.perf_counter()
        active={id(p) for group in optimizer.param_groups if float(group['lr'])>0 for p in group['params']}
        named=[(name,p) for name,p in model.named_parameters()
               if p.requires_grad and id(p) in active and not name.startswith(tuple(frozen_prefixes))]
        names=[n for n,_ in named];parameters=[p for _,p in named]
        vectors=[];losses={};metadata={};samples={};probe_stats={}
        extra_grad={c:{'proposal_loss':{},'mask_loss':{}} for c in self.cohorts}
        extra_losses={c:defaultdict(float) for c in self.cohorts}
        extra_stats={c:defaultdict(float) for c in self.cohorts}
        extra_metadata={c:[] for c in self.cohorts}
        with preserve_state(model,device,self.seed):
            for family,indices in self.indices.items():
                gradients={'proposal_loss':{},'mask_loss':{}}
                loss_means={'proposal_loss':0.,'mask_loss':0.};items=[];sample_metadata=[];stats=defaultdict(float)
                for start in range(0,len(indices),self.batch_size):
                    chosen=indices[start:start+self.batch_size];items=[]
                    for index in chosen:
                        if self.cache is not None:
                            if index not in self.cache:self.cache[index]=copy.deepcopy(self.dataset[index])
                            item=self.cache[index]
                        else:item=self.dataset[index]
                        items.append(item)
                        if not self.full_pass or len(sample_metadata)<8:
                            info={k:str(item.get(k,'')) for k in ('name','relative_root','dataset_source','band_names')}
                            if index in self.selected_by:info['probe_selections']=self.selected_by[index]
                            sample_metadata.append(info)
                            for cohort in self.membership[index]:extra_metadata[cohort].append(info)
                    batch=self.collate_fn(items)
                    parts=loss_fn(model,batch)
                    factor=len(chosen)/len(indices)
                    for key,value in parts.get('stats',{}).items():stats[key]+=float(value)*factor
                    cohort_factors={c:1/len(self.cohorts[c]) for c in self.membership[chosen[0]]} if self.cohorts else {}
                    for cohort,cf in cohort_factors.items():
                        for key,value in parts.get('stats',{}).items():extra_stats[cohort][key]+=float(value)*cf
                    for component in ('proposal_loss','mask_loss'):
                        loss=parts[component]
                        loss_means[component]+=float(loss.detach())*factor
                        for cohort,cf in cohort_factors.items():extra_losses[cohort][component]+=float(loss.detach())*cf
                        if not torch.isfinite(loss):raise ValueError(f'Nonfinite {family}/{component} loss')
                        if not loss.requires_grad or not parameters:continue
                        grads=torch.autograd.grad(loss,parameters,
                            retain_graph=component=='proposal_loss' and parts['mask_loss'].requires_grad,
                            allow_unused=True)
                        for name,g in zip(names,grads):
                            if g is None:continue
                            cpu=g.detach().to(device='cpu',dtype=torch.float32)
                            if not torch.isfinite(cpu).all():raise ValueError(f'Nonfinite gradient: {family}/{component}/{name}')
                            targets=[(gradients[component],factor)] + [
                                (extra_grad[cohort][component],cf) for cohort,cf in cohort_factors.items()]
                            for target,coefficient in targets:
                                if name in target:target[name].add_(cpu,alpha=coefficient)
                                else:target[name]=cpu.mul(coefficient)
                        del grads
                    del parts,batch,items
                vectors.extend([gradients['proposal_loss'],gradients['mask_loss']])
                losses[family]=loss_means;samples[family]=len(indices)
                probe_stats[family]=dict(stats)
                metadata[family]=sample_metadata if not self.full_pass else sample_metadata[:8]
            for cohort in self.cohorts:
                vectors.extend([extra_grad[cohort]['proposal_loss'],extra_grad[cohort]['mask_loss']])
                losses[cohort]=dict(extra_losses[cohort]);samples[cohort]=len(self.cohorts[cohort])
                probe_stats[cohort]=dict(extra_stats[cohort]);metadata[cohort]=extra_metadata[cohort]
            original_order=list(self.indices)+list(self.cohorts)
            names_for_geometry=list(self.cohorts)+list(self.indices)
            vectors=[vectors[2*original_order.index(name)+component]
                     for name in names_for_geometry for component in (0,1)]
            grams=gradient_grams(vectors,{name:parameter_part(name) for name in names})
        mixture={key:self.proportions.get(key,0.) for key in names_for_geometry}
        metrics=geometry_metrics(grams,names_for_geometry,mixture)
        for family,values in losses.items():
            for component,value in values.items():metrics[f'loss/{family}/{component}']=value
            metrics[f'data/{family}/probe_samples']=samples[family]
            metrics[f'data/{family}/training_records']=len(self.groups.get(family,[]))
            metrics[f'data/{family}/mixture_weight']=mixture[family]
            metrics[f'data/{family}/is_cohort']=int(family in self.cohorts)
            for key,value in probe_stats[family].items():metrics[f'data/{family}/{key}']=value
        metrics['probe/full_pass']=int(self.full_pass)
        metrics['probe/after_optimizer_update']=1
        metrics['probe/batch_size']=self.batch_size
        metrics['probe/seconds']=time.perf_counter()-started
        for scope in ('encoder','proposal_decoder','mask','other'):
            metrics[f'parameters/{scope}']=sum(p.numel() for name,p in named if parameter_part(name)==scope)
        # autograd.grad never accumulates into .grad; only read active weights.
        return dict(metrics=metrics,probe_samples={k:samples[k] for k in self.indices},
            cohort_samples={k:samples[k] for k in self.cohorts},seconds=time.perf_counter()-started,
            probe_config=self.probe_manifest and {k:self.probe_manifest[k] for k in ('config_path','config_sha256')},
            metadata=metadata,mode='eval; deterministic prompts; training loss weights/curriculum',
            full_definition='training-record-count-weighted mixture of dataset gradient means; '
                + ('all training records' if self.full_pass else 'fixed sampled probes, not a full-corpus gradient'),
            angle_definition='degrees acos(dot/(L2*L2)); exact parameter vectors, no projection; zero norms have defined=0',
            parameter_definition='requires_grad, positive optimizer LR, excluding proposal/encoder frozen by the training schedule',
            frozen_prefixes=list(frozen_prefixes))
