"""ID-aligned, scale-free mask moments; no runtime bright-source lookup."""
import warnings
import torch
import torch.nn.functional as F
from preprocessing.utils.mask_moments import JWST_POLICY


def catalog_targets(prompts, batch, *, bands, match_radius=3., require_eligibility=True, match_non_hsc=True):
    """GT uses exact IDs; predicted prompts use unique nearest catalog matches.

    Match to *all* source classes before applying eligibility, so a weak-SNR
    source can never acquire a nearby clean source's target. Only predicted
    centers need the batched distance calculation; eligibility is read from Zarr.
    Legacy JWST stores can safely supervise clean sources, but weak sources
    require the new high-SNR eligibility policy (not the old HSC bright rule).
    """
    centers=prompts['centers'];device=centers.device;n=len(centers)
    result=dict(shapes=prompts['target_shapes'].detach().float().clone(),
                centers=centers.detach().float().clone(),
                eligible=torch.zeros(n,device=device,dtype=torch.bool),
                matched=torch.zeros(n,device=device,dtype=torch.bool),
                is_hsc=torch.zeros(n,device=device,dtype=torch.bool))
    if batch is None:return result
    names=batch.get('band_names',[]);metadata=batch.get('proposal_metadata',[])
    kinds=prompts.get('source_kind',torch.zeros(n,device=device,dtype=torch.long))
    prompt_ids=prompts.get('source_ids',torch.full((n,),-1,device=device,dtype=torch.long))
    prompt_ids_cpu=prompt_ids.detach().cpu().tolist()
    kinds_cpu=kinds.detach().cpu().tolist()
    groups={}
    for i,flat in enumerate(prompts['batch_indices'].detach().cpu().tolist()):
        groups.setdefault(int(flat),[]).append(i)
    # Copy the small prompt metadata once, not nonzero()/tolist() from CUDA
    # once per image. Matching semantics and source-ID precision are unchanged.
    for flat,indices in sorted(groups.items()):
        b,band=divmod(int(flat),bands);ii=torch.tensor(indices,device=device,dtype=torch.long)
        name=(names[b][band] if b<len(names) and names[b] is not None and band<len(names[b]) else '')
        is_hsc=str(name).startswith(('HSC-','NB')) or (b<len(metadata) and metadata[b].get('dataset')=='hsc')
        result['is_hsc'][ii]=is_hsc
        if not is_hsc and not match_non_hsc:continue
        if 'band_shape_source_ids' not in batch:continue
        ids=batch['band_shape_source_ids'][b][band].tolist()
        if not ids:continue
        if len(set(ids))!=len(ids):raise ValueError('Mask moment targets require unique source IDs per image/band')
        xy=batch['band_shape_source_centers'][b][band].to(device=device,dtype=torch.float32)
        shapes=batch['band_shape_source_values'][b][band].to(device=device,dtype=torch.float32)
        if 'band_shape_source_classes' not in batch and is_hsc and require_eligibility:
            raise ValueError('HSC mask moments require per-source classes and shape_source_moment_eligible in Zarr')
        classes=(batch['band_shape_source_classes'][b][band].to(device=device)
                 if 'band_shape_source_classes' in batch else torch.zeros(len(ids),device=device,dtype=torch.long))
        flags=batch.get('band_shape_source_moment_eligible')
        flags=None if flags is None else flags[b][band]
        if flags is None:
            if is_hsc and require_eligibility and bool((classes==2).any()):
                raise ValueError('HSC Zarr lacks shape_source_moment_eligible: regenerate HSC labels '
                                 'or backfill this ID-aligned array; weak-shape reasons cannot be guessed. '
                                 'Use --mask-moment-weight 0 for the legacy loss.')
            eligible=classes==1
        else:
            flags=flags.to(device=device,dtype=torch.bool)
            if flags.shape!=classes.shape:raise ValueError('Mask moment eligibility/source IDs disagree')
            eligible=flags & ((classes==1)|(classes==2))
        if not is_hsc:
            policy = metadata[b].get('mask_moment_policy') if b < len(metadata) else None
            if require_eligibility and (policy != JWST_POLICY or flags is None) and bool((classes == 2).any()):
                warnings.warn('JWST Zarr lacks verified high-SNR weak moment eligibility: '
                              'only clean sources receive moment loss; regenerate or backfill '
                              'shape_source_moment_eligible with jwst_clean_or_high_snr_weak_v1 '
                              'to include high-SNR weak sources.', RuntimeWarning, stacklevel=2)
            eligible = (classes == 1) | ((classes == 2) & eligible & (policy == JWST_POLICY))
        finite=torch.isfinite(xy).all(1)&torch.isfinite(shapes).all(1)&(shapes[:,:2]>0).all(1)
        eligible &= finite
        by_id={int(s):j for j,s in enumerate(ids)};matches=[]
        for i in indices:
            if kinds_cpu[i]==1:continue
            j=by_id.get(int(prompt_ids_cpu[i]))
            if j is not None:matches.append((i,j))
        pi=torch.tensor([i for i in indices if kinds_cpu[i]==1],device=device,dtype=torch.long)
        if len(pi):
            distance=torch.cdist(centers[pi].detach().float(),torch.nan_to_num(xy))
            distance[:,~finite]=float('inf')
            values,nearest=distance.min(1);used=set()
            nearest_cpu=nearest.tolist();values_cpu=values.tolist()
            for k in torch.argsort(values,stable=True).tolist():
                j=int(nearest_cpu[k])
                if values_cpu[k]<=float(match_radius) and j not in used:
                    matches.append((int(pi[k]),j));used.add(j)
        if matches:
            ip=torch.tensor([i for i,j in matches],device=device)
            jp=torch.tensor([j for i,j in matches],device=device)
            result['shapes'][ip]=shapes[jp].detach();result['centers'][ip]=xy[jp].detach()
            result['eligible'][ip]=eligible[jp];result['matched'][ip]=finite[jp]
    return result


def ellipse_ellipticity(shapes):
    shapes=torch.nan_to_num(shapes.float(),nan=0.,posinf=0.,neginf=0.)
    a2=shapes[:,0].square();b2=shapes[:,1].square()
    amplitude=(a2-b2)/(a2+b2).clamp_min(1e-6);angle=2*shapes[:,2]
    return torch.stack((amplitude*angle.cos(),amplitude*angle.sin()),-1)


def probability_ellipticity(prob, image_hw):
    """Central population moments in native pixel units, also for H!=W.

    Cell centers use align_corners=False coordinates. Work in FP32 under AMP;
    translation does not affect the covariance. Missing coverage is already
    removed from prob by the caller. No gradients enter catalog targets.
    """
    prob=prob.float();mh,mw=prob.shape[-2:];h,w=image_hw
    x=(torch.arange(mw,device=prob.device,dtype=torch.float32)+.5)*(w/mw)-.5
    y=(torch.arange(mh,device=prob.device,dtype=torch.float32)+.5)*(h/mh)-.5
    # Recenter coordinates before accumulation to reduce cancellation.
    x=x-(w-1)/2;y=y-(h-1)/2
    mass=prob.sum((-1,-2)).clamp_min(1e-6)
    px=prob.sum(-2);py=prob.sum(-1)
    cx=(px*x).sum(-1)/mass;cy=(py*y).sum(-1)/mass
    dx=x-cx[...,None];dy=y-cy[...,None]
    xx=(px*dx.square()).sum(-1)/mass
    yy=(py*dy.square()).sum(-1)/mass
    xy=((prob*dx[...,None,:]).sum(-1)*dy).sum(-1)/mass
    trace=(xx+yy).clamp_min(1e-6)
    return torch.stack(((xx-yy)/trace,2*xy/trace),-1)


def native_ellipticity(logits, coverage, image_hw, *, chunk_size=64, coverage_indices=None):
    """Match inference resize-before-sigmoid, without retaining full-size masks.

    Checkpoint only the new moment branch. This recomputes inexpensive image
    interpolation during backward instead of retaining N*K*H*W probabilities.
    """
    from torch.utils.checkpoint import checkpoint
    def evaluate(work,observed,indices):
        if indices is not None:
            observed=observed[indices]
        if tuple(work.shape[-2:])!=tuple(image_hw):
            work=F.interpolate(work.float(),size=image_hw,mode='bilinear',align_corners=False)
        return probability_ellipticity(work.float().sigmoid()*observed[:,None],image_hw)
    if len(logits)==0:
        return logits.new_empty((*logits.shape[:2],2),dtype=torch.float32)
    # Keep native temporary tensors bounded also for loss-based selection
    # (all K candidates), rectangular inputs and future larger image sizes.
    chunk_size=min(chunk_size,max(1,(64*512*512)//(logits.shape[1]*image_hw[0]*image_hw[1])))
    parts=[]
    for start in range(0,len(logits),chunk_size):
        z=logits[start:start+chunk_size]
        if coverage_indices is not None:
            # Gather INSIDE checkpoint. Saving a gathered mask as each
            # checkpoint input would retain N*H*W masks until backward again.
            valid=coverage;indices=coverage_indices[start:start+chunk_size]
        else:
            valid=coverage[start:start+chunk_size];indices=None
        parts.append(checkpoint(evaluate,z,valid,indices,use_reentrant=False)
                     if torch.is_grad_enabled() and z.requires_grad else evaluate(z,valid,indices))
    return torch.cat(parts,0)


@torch.no_grad()
def prepare_moment_targets(targets, *, image_hw, valid, labelled, prompts, batch):
    """Gate targets BEFORE full-resolution prediction work; no loss semantics change.

    Partial/truncated catalog ellipses are excluded; a cropped mask cannot
    constrain the full catalog shape. JWST GT is evaluated on its observed pixels.
    """
    h,w=image_hw;shapes=torch.nan_to_num(targets['shapes']);centers=torch.nan_to_num(targets['centers'])
    a,b=shapes[:,:2].abs().clamp_min(1).unbind(1);t=shapes[:,2]
    rx=((a*t.cos()).square()+(b*t.sin()).square()).sqrt()
    ry=((a*t.sin()).square()+(b*t.cos()).square()).sqrt()
    clipped=(centers[:,0]-rx<0)|(centers[:,0]+rx>w-1)|(centers[:,1]-ry<0)|(centers[:,1]+ry>h-1)
    active=targets['eligible'] & (~clipped|labelled)
    desired=ellipse_ellipticity(shapes).detach()
    # Native-pixel GT moments: never threshold an area-downsampled tiny mask.
    rows=prompts.get('instance_masks',[None]*len(centers))
    for i in torch.where(labelled & active)[0].tolist():
        row=rows[i]
        if row is None:active[i]=False;continue
        y0,x0=int(row['y0']),int(row['x0']);mask=row['mask'].to(device=shapes.device,dtype=torch.bool)
        yy,xx=torch.where(mask);xx=xx+x0;yy=yy+y0
        keep=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
        xx,yy=xx[keep],yy[keep]
        if batch is not None and 'band_valid_mask' in batch:
            observed=batch['band_valid_mask'].to(shapes.device).reshape(-1,h,w)[prompts['batch_indices'][i]]
            keep=observed[yy,xx].bool();xx,yy=xx[keep],yy[keep]
        if len(xx)<2:active[i]=False;continue
        xx=xx.float();yy=yy.float();xx-=xx.mean();yy-=yy.mean()
        vx=xx.square().mean();vy=yy.square().mean();vxy=(xx*yy).mean();trace=(vx+vy).clamp_min(1e-6)
        desired[i]=torch.stack(((vx-vy)/trace,2*vxy/trace))
    # Holes in the catalog ellipse also make its visible moments incomplete.
    check=torch.where(active & ~labelled & (valid<1-1e-6).flatten(1).any(1))[0]
    if len(check):
        mh,mw=valid.shape[-2:]
        x=(torch.arange(mw,device=shapes.device)+.5)*(w/mw)-.5
        y=(torch.arange(mh,device=shapes.device)+.5)*(h/mh)-.5
        for ii in check.split(64):
            dx=x[None,None,:]-centers[ii,0,None,None];dy=y[None,:,None]-centers[ii,1,None,None]
            cs=t[ii].cos()[:,None,None];sn=t[ii].sin()[:,None,None]
            inside=((dx*cs+dy*sn)/a[ii,None,None]).square()
            inside+=((-dx*sn+dy*cs)/b[ii,None,None]).square()
            holes=((inside<=1)&(valid[ii]<1-1e-6)).any((-1,-2))
            active[ii] &= ~holes
    return desired,active


def moment_loss(prob, targets, *, image_hw, valid, labelled, prompts, batch, prediction_e=None, prepared=None):
    """Use native instance GT first, otherwise an eligible catalog shape prior."""
    desired,active=(prepare_moment_targets(targets,image_hw=image_hw,valid=valid,labelled=labelled,
                                         prompts=prompts,batch=batch) if prepared is None else prepared)
    if prediction_e is None:prediction_e=probability_ellipticity(prob,image_hw)
    values=F.smooth_l1_loss(prediction_e,desired[:,None].expand(-1,prob.shape[1],-1),reduction='none').mean(-1)
    return values*active[:,None],active
