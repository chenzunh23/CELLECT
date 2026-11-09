"""Native-pixel center targets, local owned regression and heatmap decoding.

No predicted-peak matching is used during target construction. Coordinates are
zero-based pixel centers. Shape and offset are read at the ORIGINAL peak cell.
"""
from dataclasses import dataclass
import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class HeatmapConfig:
    sigma: float = 1.5
    truncate: float = 4.
    boundary_tolerance: float = .05
    offset_size: int = 5
    shape_size: int = 3
    tie_epsilon: float = 1e-4  # squared native pixels

    def __post_init__(self):
        if not math.isfinite(self.sigma) or self.sigma <= 0 or self.truncate <= 0:
            raise ValueError('Gaussian sigma/truncation must be positive')
        if not 0 <= self.boundary_tolerance < .5:
            raise ValueError('Pixel boundary tolerance must be in [0,.5)')
        if any(n < 1 or n % 2 != 1 for n in (self.offset_size,self.shape_size)):
            raise ValueError('Local regression windows must be positive odd integers')


class PredictionTower(nn.Sequential):
    def __init__(self, channels, outputs, *, bias=0.):
        groups=next(g for g in (8,4,2,1) if channels % g == 0)
        super().__init__(nn.Conv2d(channels,channels,3,padding=1,bias=False),
            nn.GroupNorm(groups,channels),nn.GELU(),
            nn.Conv2d(channels,outputs,1))
        nn.init.constant_(self[-1].bias,bias)


class ImagePyramid(nn.Module):
    """Optional /1,/2,/4 image features for ordinal and heatmap source decoders."""
    def __init__(self,stage_channels,width=16):
        super().__init__()
        def block(a,b,stride=1):
            return nn.Sequential(nn.Conv2d(a,b,3,stride,padding=1,bias=False),nn.GroupNorm(4,b),nn.GELU())
        self.full=block(3,width);self.half_stage=block(width,width,2);self.quarter=block(width,width,2)
        self.project=nn.ModuleDict({k:nn.Conv2d(width,c,1) for k,c in stage_channels.items()})

    def forward(self,image):
        if image.shape[1]==1:image=image.expand(-1,3,-1,-1)
        a=self.full(image);b=self.half_stage(a);c=self.quarter(b)
        return dict(up2=c,up3=b,up4=a)

    def fuse(self,stage,feature,pyramid):
        source=F.interpolate(pyramid[stage],size=feature.shape[-2:],mode='bilinear',align_corners=False)
        return feature+.1*self.project[stage](source)


@torch.no_grad()
def choose_anchors(centers, image, valid, tolerance=.05):
    """Nearest cell, with image-brightness tie breaking near pixel boundaries."""
    h,w=image.shape[-2:]
    luminance=image.float().mean(0) if image.ndim==3 else image.float()
    lower=torch.floor(centers).long()
    steps=lower.new_tensor([[0,0],[1,0],[0,1],[1,1]])
    candidates=lower[:,None]+steps[None]
    delta=candidates.float()-centers[:,None]
    nearest=torch.floor(centers+.5).long()
    near_boundary=(centers-lower.float()-.5).abs()<=tolerance
    allowed=((candidates==nearest[:,None])|near_boundary[:,None]).all(-1)
    x,y=candidates.unbind(-1)
    allowed &= (x>=0)&(x<w)&(y>=0)&(y<h)
    xx,yy=x.clamp(0,w-1),y.clamp(0,h-1)
    allowed &= valid[yy,xx]
    brightness=luminance[yy,xx]
    allowed &= torch.isfinite(brightness)
    brightness=brightness.masked_fill(~allowed,-torch.inf)
    best=brightness.amax(1,keepdim=True)
    tied=allowed & (brightness==best)
    distance=delta.square().sum(-1).masked_fill(~tied,torch.inf)
    selected=distance.argmin(1)
    return candidates[torch.arange(len(centers),device=centers.device),selected],allowed.any(1)


@torch.no_grad()
def make_targets(centers, image, base_weight, valid, *, sigma=None, config=HeatmapConfig(),
                 source_weights=None):
    """Return dense heatmap plus sparse regression samples owned by nearest GT.

    All trusted centers are supplied, including centers lacking shape labels.
    A tie skips regression only. Gaussian support overrides regional ignore,
    but never known-invalid input support. Source identity stays explicit.
    """
    h,w=base_weight.shape
    device=base_weight.device
    centers=centers.to(device=device,dtype=torch.float32).reshape(-1,2)
    finite=torch.isfinite(centers).all(1)&(centers[:,0]>=0)&(centers[:,0]<w)&(centers[:,1]>=0)&(centers[:,1]<h)
    original=torch.arange(len(centers),device=device)[finite]
    centers=centers[finite]
    anchors,good=choose_anchors(centers,image,valid,config.boundary_tolerance)
    centers,anchors,original=centers[good],anchors[good],original[good]
    target=torch.zeros((h*w,),device=device,dtype=torch.float32)
    positive_weight=torch.zeros_like(target)
    sigma=float(config.sigma if sigma is None else sigma)
    if not math.isfinite(sigma) or sigma<=0:raise ValueError('Invalid per-image heatmap sigma')
    radius=int(math.ceil(config.truncate*sigma))
    offsets=torch.stack(torch.meshgrid(torch.arange(-radius,radius+1,device=device),
        torch.arange(-radius,radius+1,device=device),indexing='ij'),-1).flip(-1).reshape(-1,2)
    gaussian=torch.exp(-offsets.float().square().sum(-1)/(2*sigma*sigma))
    support=offsets.square().sum(-1)<=(config.truncate*sigma)**2
    points=anchors[:,None]+offsets[None]
    x,y=points.unbind(-1)
    inside=(x>=0)&(x<w)&(y>=0)&(y<h)&support[None]
    xx,yy=x.clamp(0,w-1),y.clamp(0,h-1)
    inside &= valid[yy,xx]
    index=(yy*w+xx)[inside]
    values=gaussian.expand(len(centers),-1)[inside]
    target.scatter_reduce_(0,index,values,reduce='amax',include_self=True)
    sw=(torch.ones(len(centers),device=device) if source_weights is None
        else source_weights.to(device=device,dtype=torch.float32)[original])
    positive_weight.scatter_reduce_(0,index,sw[:,None].expand_as(inside)[inside],reduce='amax',include_self=True)
    weight=torch.where(positive_weight.reshape(h,w)>0,positive_weight.reshape(h,w),base_weight.float())
    weight=weight*valid
    # Vectorized GT-local candidate points; no detection or CPU matching.
    r=max(config.offset_size,config.shape_size)//2
    local=torch.stack(torch.meshgrid(torch.arange(-r,r+1,device=device),
        torch.arange(-r,r+1,device=device),indexing='ij'),-1).flip(-1).reshape(-1,2)
    points=anchors[:,None]+local[None]
    x,y=points.unbind(-1)
    good_points=(x>=0)&(x<w)&(y>=0)&(y<h)
    good_points &= valid[y.clamp(0,h-1),x.clamp(0,w-1)]
    owner=torch.arange(len(centers),device=device)[:,None].expand_as(good_points)
    p=points[good_points];expected=owner[good_points]
    accepted=[]
    # Chunking bounds temporary memory for crowded images; no per-source loop.
    for block in p.split(1024):
        distance=(block.float()[:,None]-centers[None]).square().sum(-1)
        if len(centers)>1:
            d,idx=distance.topk(2,largest=False,dim=1)
            accepted.append(torch.where(d[:,1]-d[:,0]>config.tie_epsilon,idx[:,0],-1))
        else:
            accepted.append(torch.zeros(len(block),device=device,dtype=torch.long))
    nearest=torch.cat(accepted) if accepted else expected.new_empty(0)
    keep=nearest==expected
    p,expected=p[keep],expected[keep]
    local_delta=p-anchors[expected]
    return dict(heatmap=target.reshape(h,w),weight=weight,anchors=anchors,
        source_indices=original,reg_xy=p,reg_source=original[expected],
        offset_target=centers[expected]-p.float(),
        offset_valid=local_delta.abs().amax(1)<=config.offset_size//2,
        shape_valid=local_delta.abs().amax(1)<=config.shape_size//2)


def center_loss(logits,target,weight,mode='soft_bce'):
    logits=logits.float();target=target.float();weight=weight.float()
    if mode=='soft_bce':
        loss=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
        return (loss*weight).sum()/weight.sum().clamp_min(1.)
    if mode!='focal':raise ValueError(f'Unknown heatmap loss: {mode}')
    prob=logits.sigmoid()
    positive=target==1
    loss=torch.where(positive,-F.logsigmoid(logits)*(1-prob).square(),
        -F.logsigmoid(-logits)*prob.square()*(1-target).pow(4))
    return (loss*weight).sum()/((weight*positive).sum().clamp_min(1.))


def source_mean_loss(point_loss,source_indices,weights,source_count):
    sums=point_loss.new_zeros(source_count).scatter_add_(0,source_indices,point_loss)
    counts=point_loss.new_zeros(source_count).scatter_add_(0,source_indices,torch.ones_like(point_loss))
    effective=weights.to(point_loss)*(counts>0)
    return ((sums/counts.clamp_min(1))*effective).sum()/effective.sum().clamp_min(1.)


@torch.no_grad()
def decode(outputs, threshold=.5, nms_radius=1):
    """Merge equal-valued 8-connected peak plateaus after offset correction.

    Shape sampling keeps one actual peak anchor: the valid vote nearest the
    corrected centroid, with row-major ties. Other peaks retain their anchors.
    """
    probability=outputs['confidence'].float().sigmoid()[:,0]
    maxima=F.max_pool2d(probability[:,None],2*nms_radius+1,1,nms_radius)[:,0]
    peaks=(probability>threshold)&(probability==maxima)
    results=[]
    for b in range(len(probability)):
        y,x=torch.where(peaks[b]);anchors=torch.stack([x,y],1).float()
        delta=outputs['offset'][b,:,y,x].T.float()
        xy=anchors+delta
        h,w=probability.shape[-2:]
        valid=torch.isfinite(xy).all(1)&(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
        scores=probability[b,y,x]
        groups,count=_equal_peak_plateaus(anchors,scores)
        if count==len(anchors):
            results.append(dict(xy=xy[valid],anchor_xy=anchors[valid],score=scores[valid]))
            continue
        groups=torch.as_tensor(groups,device=xy.device,dtype=torch.long)
        counts=xy.new_zeros(count).scatter_add_(0,groups[valid],torch.ones_like(scores[valid]))
        sums=xy.new_zeros((count,2)).index_add_(0,groups[valid],xy[valid])
        centroids=sums/counts.clamp_min(1)[:,None]
        distance=(xy-centroids[groups]).square().sum(1).masked_fill(~valid,float('inf'))
        minimum=distance.new_full((count,),float('inf')).scatter_reduce_(0,groups,distance,reduce='amin')
        indices=torch.arange(len(xy),device=xy.device)
        candidates=torch.where(valid&(distance==minimum[groups]),indices,len(xy))
        representative=indices.new_full((count,),len(xy)).scatter_reduce_(0,groups,candidates,reduce='amin')
        keep=counts>0
        representative=representative[keep]
        results.append(dict(xy=centroids[keep],anchor_xy=anchors[representative],score=scores[representative]))
    return results


def _equal_peak_plateaus(anchors,scores):
    """Sparse union-find; only adjacent peaks with exactly equal scores join."""
    coordinates=anchors.detach().cpu().numpy().astype(np.int64,copy=False)
    values=scores.detach().cpu().numpy()
    parent=list(range(len(coordinates)));seen={}
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,(x,y) in enumerate(coordinates):
        for dx,dy in ((-1,0),(-1,-1),(0,-1),(1,-1)):
            j=seen.get((x+dx,y+dy))
            if j is not None and values[i]==values[j]:
                a,b=root(i),root(j)
                parent[max(a,b)]=min(a,b)
        seen[(x,y)]=i
    roots=np.fromiter((root(i) for i in range(len(parent))),dtype=np.int64,count=len(parent))
    unique,groups=np.unique(roots,return_inverse=True)
    return groups,len(unique)


def proposal_kwargs_from_state_dict(state):
    prefix='decoder.center_head.'
    heatmap=any(str(k).removeprefix('module.').startswith(prefix) for k in state)
    fusion=any(str(k).removeprefix('module.').startswith('decoder.heatmap_image_pyramid.') for k in state)
    return {'proposal_mode':'heatmap' if heatmap else 'ordinal','heatmap_image_fusion':fusion}


def dense_losses(outputs,batch,weights):
    """Rebuild targets from augmented sparse GT, retaining regional PU weights."""
    from utils.train_ops_utils import shape_regression_loss_map
    from preprocessing.labels import LabelWeights
    config=HeatmapConfig(sigma=weights.heatmap_sigma,
        boundary_tolerance=weights.heatmap_boundary_tolerance,
        offset_size=weights.offset_center_size,shape_size=weights.shape_center_size)
    logits=outputs['confidence'][:,0]
    zero=outputs['shape'].sum()*0.+outputs['offset'].sum()*0.+logits.sum()*0.
    losses=[];offsets=[];shapes=[]
    for b in range(len(logits)):
        device=logits.device
        centers=torch.cat([batch['centers'][b],batch['strict_centers'][b]],0).to(device=device,dtype=torch.float32)
        ids=torch.cat([batch['ids'][b],batch['strict_ids'][b]],0).to(device=device,dtype=torch.long)
        shape_ids=batch['shape_source_ids'][b].to(device=device,dtype=torch.long)
        values=batch['shape_source_values'][b].to(device=device,dtype=torch.float32)
        classes=batch['shape_source_classes'][b].to(device=device)
        shape_weights=(batch['shape_source_weights'][b].to(device=device,dtype=torch.float32)
            if 'shape_source_weights' in batch else torch.ones(len(values),device=device))
        target_shape=torch.zeros((len(centers),3),device=device)
        shape_valid=torch.zeros(len(centers),device=device,dtype=torch.bool)
        sw=torch.ones(len(centers),device=device)
        reliability=torch.ones_like(sw)
        if len(shape_ids):
            ordered,order=shape_ids.sort();positions=torch.searchsorted(ordered,ids).clamp_max(len(ordered)-1)
            matched=ordered[positions]==ids
            index=order[positions]
            target_shape[matched]=values[index[matched]]
            shape_valid=matched & torch.isfinite(target_shape).all(1)&(target_shape[:,:2]>0).all(1)
            sw[matched]=shape_weights[index[matched]]
            reliability[matched]=torch.where(classes[index[matched]]==2,
                reliability.new_tensor(weights.center_only_shape_factor),reliability.new_tensor(1.))
        pu=batch['pu'][b].to(device=device).long()
        lookup=torch.as_tensor(LabelWeights().as_array(),device=device)
        base=lookup[pu.clamp(0,len(lookup)-1)]
        observed=batch['valid'][b].to(device=device,dtype=torch.bool)
        old_positive=(batch['confidence'][b].to(device)>0)&(batch['confidence_weight'][b].to(device)>0)
        valid=observed|old_positive
        sigma=weights.heatmap_sigma
        if weights.heatmap_sigma_mode=='psf-ee':
            from preprocessing.utils.heatmap_sigma import resolve_sigma
            import json
            metadata=batch['proposal_metadata'][b]
            sigma=resolve_sigma(json.dumps(metadata,sort_keys=True),batch['band_names'][b],
                weights.heatmap_psf_ee_root,weights.heatmap_sigma_min,weights.heatmap_sigma_max)
        target=make_targets(centers,batch['image'][b].to(device),base,valid,
            sigma=sigma,source_weights=sw,config=config)
        losses.append(center_loss(logits[b],target['heatmap'],target['weight'],weights.heatmap_loss))
        p=target['reg_xy'];source=target['reg_source'];x,y=p.unbind(1)
        pred=outputs['offset'][b,:,y,x].T.float()
        point_loss=F.smooth_l1_loss(pred,target['offset_target'],reduction='none').mean(1)
        select=target['offset_valid']
        offsets.append(source_mean_loss(point_loss[select],source[select],sw,len(centers)))
        select=target['shape_valid'] & shape_valid[source] & observed[y,x]
        pred=outputs['shape'][b,:,y[select],x[select]].T.float()
        desired=target_shape[source[select]]
        point_loss=shape_regression_loss_map(pred[:,:,None],desired[:,:,None],
            angle_weight=weights.shape_angle_weight,geometry_loss=weights.shape_geometry_loss).reshape(-1)
        shapes.append(source_mean_loss(point_loss,source[select],sw*reliability,len(centers)))
    hm=torch.stack(losses).mean() if losses else zero
    offset=torch.stack(offsets).mean() if offsets else zero
    shape=torch.stack(shapes).mean() if shapes else zero
    total=weights.confidence_outer_weight*hm+weights.offset_loss_weight*offset+weights.shape_outer_weight*shape+zero
    return dict(total=total,confidence=hm,shape=shape,center=offset,offset=offset,
                seg=zero,small_shape=zero)
