"""Isolated old/new moment loss benchmark; never loads a model or dataset."""
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from sam_backbone.mask_moments import native_ellipticity,probability_ellipticity
from sam_backbone.mask_supervision import partial_targets


def legacy_native(logits,coverage,image_hw):
    def evaluate(z,c):
        p=F.interpolate(z.float(),size=image_hw,mode='bilinear',align_corners=False).sigmoid()*c[:,None]
        return probability_ellipticity(p,image_hw)
    return torch.cat([checkpoint(evaluate,logits[s:s+16],coverage[s:s+16],use_reentrant=False)
                      for s in range(0,len(logits),16)],0)


def measure(fn,z,repeats):
    for _ in range(2):
        z.grad=None;fn().backward()
    torch.cuda.synchronize();times=[];peaks=[]
    for _ in range(repeats):
        z.grad=None;torch.cuda.synchronize();baseline=torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record();value=fn();value.backward();end.record();end.synchronize()
        times.append(start.elapsed_time(end));peaks.append((torch.cuda.max_memory_allocated()-baseline)/2**20)
    return dict(forward_backward_ms=statistics.median(times),incremental_peak_mib=max(peaks))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cuda:0');p.add_argument('--prompts',type=int,default=512)
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--output',type=Path);args=p.parse_args()
    device=torch.device(args.device);torch.manual_seed(7)
    n=args.prompts;native=(512,512);b=20
    z=torch.randn(n,3,128,128,device=device,requires_grad=True)
    coverage=torch.ones(b,*native,device=device,dtype=torch.bool);coverage[:,0,:]=False
    flat=torch.arange(n,device=device)%b;best=torch.arange(n,device=device)%3;row=torch.arange(n,device=device)
    target=torch.randn(n,2,device=device)*.2
    def old():
        e=legacy_native(z,coverage[flat],native)
        return F.smooth_l1_loss(e[row,best],target)
    def new():
        e=native_ellipticity(z[row,best][:,None],coverage,native,coverage_indices=flat)
        return F.smooth_l1_loss(e[:,0],target)
    z.grad=None;v0=old();v0.backward();g0=z.grad.clone()
    z.grad=None;v1=new();v1.backward();g1=z.grad
    error=dict(value_abs=float((v0-v1).abs().detach()),gradient_max_abs=float((g0-g1).abs().max()))
    torch.testing.assert_close(v0,v1,rtol=1e-5,atol=1e-7)
    torch.testing.assert_close(g0,g1,rtol=2e-4,atol=2e-8)
    del v0,v1,g0,g1;z.grad=None
    report=dict(device=torch.cuda.get_device_name(device),prompts=n,shape=[n,3,128,128],native_hw=native,error=error)
    report['old_moment']=measure(old,z,args.repeats);print(json.dumps(dict(old_moment=report['old_moment'])),flush=True)
    report['new_moment']=measure(new,z,args.repeats)
    report['moment_speedup']=report['old_moment']['forward_backward_ms']/report['new_moment']['forward_backward_ms']
    # Target construction has no gradient, but matters to the whole mask step.
    batch=dict(band_valid_mask=coverage[:,None],band_trusted_background=coverage[:,None])
    prompts=dict(centers=torch.zeros(n,2,device=device),batch_indices=flat,instance_masks=[None]*n)
    def legacy_targets():
        positive=torch.zeros(n,*native,device=device)
        valid=coverage[flat].float();negative=coverage[flat].float()
        positive*=valid;negative*=valid*(positive==0)
        positive,negative,valid=[F.interpolate(x[:,None],size=(128,128),mode='area')[:,0]
                                 for x in (positive,negative,valid)]
        return positive,(negative>=1-1e-6).float(),valid
    def new_targets():return partial_targets(prompts,batch,image_hw=native,mask_hw=(128,128),device=device)
    expected,actual=legacy_targets(),new_targets()
    for x,y in zip(expected,actual[:3]):torch.testing.assert_close(x,y,rtol=0,atol=0)
    del expected,actual
    for name,fn in [('old_targets',legacy_targets),('new_targets',new_targets)]:
        times=[];peaks=[]
        for _ in range(args.repeats+2):
            torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record();result=fn();end.record();end.synchronize()
            times.append(start.elapsed_time(end));peaks.append((torch.cuda.max_memory_allocated()-base)/2**20)
            del result
        report[name]=dict(forward_ms=statistics.median(times[2:]),incremental_peak_mib=max(peaks[2:]))
    print(json.dumps(report,indent=2),flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
