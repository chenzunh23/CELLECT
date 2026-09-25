from types import SimpleNamespace
import numpy as np
import unittest
import torch
from sam_backbone.mask_supervision import (area_ratio_penalty, partial_targets,
    partial_bce_dice, instance_prompts, combine_prompts)
from sam_backbone.losses import _sam_mask_loss_for_prompts, sam_prompt_mask_losses
from utils.instance_mask_data import read_instance_targets, flip_instance_targets


def weights(**kwargs):
    d=dict(mask_dice=1.,mask_bce=1.,mask_supervision_weight=.2,mask_centroid=0.,
        mask_outside=0.,mask_outside_kron_scale=1.5,mask_min_area=0.,mask_max_area=0.,
        mask_pred_iou=0.,mask_stability=0.,mask_max_area_ratio=.5,mask_selection='loss',
        mask_multimask=False,mask_prompt_chunk_size=8,mask_prompt_center_only=True,
        mask_area_ratio_lower=.05,mask_area_ratio_upper=2.,mask_outer_weight=1.,
        mask_loss_warmup_epochs=0,mask_prompt_gt_epochs=5,mask_prompt_pred_epoch=30,
        mask_max_gt_per_sample=128,mask_max_pred_per_sample=128,mask_unmatched_prompt=.2)
    return SimpleNamespace(**(d|kwargs))


def prompts(n=1, center=(8.,8.), shape=(4.,4.,0.)):
    return dict(batch_indices=torch.zeros(n,dtype=torch.long),
        centers=torch.tensor([center]*n),prompt_shapes=torch.tensor([shape]*n),
        target_shapes=torch.tensor([shape]*n),weights=torch.ones(n),
        mask_target_weights=torch.ones(n),source_ids=torch.arange(n))


class Model(torch.nn.Module):
    dynamic_image_size=True
    def __init__(self, logits):
        super().__init__();self.logits=torch.nn.Parameter(logits.clone())
    def forward_sam_masks(self, emb, indices, centers, boxes, **kwargs):
        self.count=len(indices)
        masks=self.logits.expand(len(indices),1,*self.logits.shape[-2:])
        return masks, masks.new_zeros((len(indices),1))


def run(logits,p,w,batch=None):
    model=Model(logits)
    out=_sam_mask_loss_for_prompts(model,{'confidence':torch.zeros(1),'image_embeddings':torch.zeros(1)},p,
        weights=w,image_hw=logits.shape[-2:],ellipse_sigma=1.,batch=batch)
    return out,model


def test_area_lower_has_gradient_below_old_clamp_and_bounds():
    r=torch.tensor([.01,.049,.05,1.,2.,2.01],requires_grad=True)
    loss=area_ratio_penalty(r,.05,2.)
    assert torch.equal(loss[2:5],torch.zeros(3))
    loss.sum().backward()
    assert r.grad[0]<0 and r.grad[1]<0 and r.grad[-1]>0
    with unittest.TestCase().assertRaises(ValueError):area_ratio_penalty(r,0.,2.)


def test_bce_negative_only_sky_unknown_zero_gradient_dice_only_labelled():
    logits=torch.zeros((2,1,4,4),requires_grad=True)
    pos=torch.zeros(2,4,4);pos[0,1,1]=1
    neg=torch.zeros_like(pos);neg[:,3,3]=1
    bce,dice,eligible=partial_bce_dice(logits,pos,neg,torch.tensor([True,False]))
    assert dice[1].item()==0 and eligible.all()
    (bce.sum()+dice.sum()).backward()
    assert logits.grad[0,0,1,1]<0 and logits.grad[0,0,3,3]>0
    assert logits.grad[1,0,3,3]>0
    assert torch.all(logits.grad[:,:,0,0]==0)
    assert logits.grad[1,0,1,1]==0


def test_clipping_overlap_and_invalid_pixels_preserves_tiny_masks():
    p=prompts(2)
    p['instance_masks']=[dict(x0=-1,y0=0,mask=torch.ones(3,3,dtype=torch.bool),weight=.25),
                         dict(x0=0,y0=0,mask=torch.ones(1,1,dtype=torch.bool),weight=.25)]
    valid=torch.ones(1,1,4,4,dtype=torch.bool);valid[0,0,2,1]=False
    sky=torch.ones_like(valid)
    pos,neg,v,q,label=partial_targets(p,{'band_valid_mask':valid,'band_trusted_background':sky},
        image_hw=(4,4),mask_hw=(4,4),device='cpu')
    assert pos[0].sum()==5 and pos[1].sum()==1
    assert pos[0,0,0] and pos[1,0,0]  # independently overlapping parent and child
    assert neg[0,2,1]==0 and neg[0,0,0]==0
    low=partial_targets(p,{'band_valid_mask':valid,'band_trusted_background':sky},
        image_hw=(4,4),mask_hw=(2,2),device='cpu')
    assert low[0][1,0,0]==.25 and low[4].all()
    logits=torch.full((2,1,2,2),-30.);logits[:,0,0,0]=30
    _,dice,_=partial_bce_dice(logits,low[0],torch.zeros_like(low[1]),low[4])
    assert dice[1]<1e-6  # the unknown remainder of the cell is not negative


def test_geometry_ignores_invalid_pixels_and_clips_area_reference():
    # A Kron ellipse centered on a cutout edge has only its visible half here.
    p=prompts(center=(0.,8.),shape=(6.,6.,0.))
    yy,xx=torch.meshgrid(torch.arange(16),torch.arange(16),indexing='ij')
    visible=(xx**2+(yy-8)**2 <= 36)
    logits=torch.where(visible,30.,-30.)[None,None]
    out,_=run(logits,p,weights(mask_dice=0.,mask_bce=0.,mask_min_area=1.,
        mask_area_ratio_lower=.95,mask_area_ratio_upper=1.05))
    assert out['area']<1e-5
    valid=torch.ones(1,1,16,16,dtype=torch.bool);valid[:,:,0,:]=False
    a,_=run(logits,p,weights(mask_dice=0.,mask_bce=0.,mask_centroid=1.,mask_outside=1.),
        {'band_valid_mask':valid})
    logits[:,:,0,:]=30
    b,_=run(logits,p,weights(mask_dice=0.,mask_bce=0.,mask_centroid=1.,mask_outside=1.),
        {'band_valid_mask':valid})
    assert torch.allclose(a['centroid'],b['centroid']) and torch.allclose(a['outside'],b['outside'])


def test_instance_reliability_not_cancelled_by_normalization():
    p=prompts();p['instance_masks']=[dict(x0=7,y0=7,mask=torch.ones(2,2,dtype=torch.bool),weight=.25)]
    b=dict(band_valid_mask=torch.ones(1,1,16,16,dtype=torch.bool),
           band_trusted_background=torch.ones(1,1,16,16,dtype=torch.bool))
    small,_=run(torch.zeros(1,1,16,16),p,weights(),b)
    p['instance_masks'][0]['weight']=1.
    full,_=run(torch.zeros(1,1,16,16),p,weights(),b)
    assert torch.allclose(small['dice']*4,full['dice'])
    assert torch.allclose(small['bce']*4,full['bce'])


def test_outside_uses_1p5_kron_axes_not_original_ellipse():
    logits=torch.full((1,1,24,24),-40.);logits[0,0,12,18]=40
    p=prompts(center=(12.,12.),shape=(5.,5.,0.))
    loose,_=run(logits,p,weights(mask_dice=0.,mask_bce=0.,mask_outside=1.))
    tight,_=run(logits,p,weights(mask_dice=0.,mask_bce=0.,mask_outside=1.,mask_outside_kron_scale=1.))
    assert loose['outside']<1e-5 and tight['outside']>.99


def test_centroid_euclidean_offset_over_kron_geometric_mean():
    logits=torch.full((1,1,32,32),-40.);logits[0,0,19,20]=40
    p=prompts(center=(16.,16.),shape=(4.,9.,0.))
    out,_=run(logits,p,weights(mask_dice=0.,mask_bce=0.,mask_centroid=1.))
    assert torch.allclose(out['centroid'],torch.tensor(5/6),atol=1e-5)


def test_all_labelled_sources_survive_cap_and_predicted_phase():
    m=prompts(140);m['source_ids']=torch.arange(140);m['instance_masks']=[{'source_id':i} for i in range(140)]
    gt=prompts(200);pred=prompts(200);pred['centers']+=20
    for ratio in [0.,.5,1.]:
        out=combine_prompts(m,gt if ratio<1 else None,pred if ratio>0 else None,pred_ratio=ratio)
        assert len(out['centers'])==140
        assert (out['source_kind']==2).all()
    small={k:(v[:10] if isinstance(v,(torch.Tensor,list)) else v) for k,v in m.items()}
    out=combine_prompts(small,gt,None,pred_ratio=0)
    assert len(out['centers'])==128 and sum(x is not None for x in out['instance_masks'])==10


class Reader:
    def __init__(self,data):
        self.data=data;self.attrs={'background_method':'aggressive SExtractor'}
    def has_array(self,k):return k in self.data
    def read_first_axis(self,k,i):return self.data[k][i]
    def read_full_small(self,k):return self.data[k]


def test_loader_parent_override_sparse_masks_and_flip():
    ids=np.zeros((1,1,8,8),np.int32);ids[0,0,1:5,1:5]=10;ids[0,0,2,2]=20
    parent=np.ones((4,4),bool)
    data=dict(band_valid_mask=np.ones((1,1,8,8),bool),band_segmentation_ids=ids,
        band_segmentation_weight=(ids>0).astype(np.float32)*.25,
        segmentation_overlap_meta=np.array([[0,0,10,-1,1,1,4,4]]),
        segmentation_overlap_offsets=np.array([0,2]),segmentation_overlap_data=np.packbits(parent.ravel(),bitorder='little'),
        segmentation_overlap_weights=np.array([.25]))
    rows,valid,sky=read_instance_targets(Reader(data),0,torch.full((1,8,8),4),torch.ones(1,3,8,8))
    byid={r['source_id']:r for r in rows[0]}
    assert byid[10]['mask'].sum()==16 and byid[20]['mask'].sum()==1
    assert not sky[0,2,2] and sky[0,7,7]
    flipped={r['source_id']:r for r in flip_instance_targets(rows,8)[0]}
    assert flipped[10]['x0']==3 and flipped[20]['x0']==5
    # Unknown provenance must not silently turn non-SExtractor background into negatives.
    reader=Reader(data);reader.attrs={}
    assert not read_instance_targets(reader,0,torch.full((1,8,8),4),torch.ones(1,3,8,8))[2].any()


def test_source_ids_beat_dense_shape_in_nested_region_and_epoch30_keeps_gt():
    c=torch.tensor([[8.,8.],[8.,9.]])
    s=torch.tensor([[4.,4.,0.],[2.,2.,0.]])
    rows=[dict(source_id=10,x0=5,y0=5,mask=torch.ones(6,6,dtype=torch.bool),weight=.25),
          dict(source_id=20,x0=8,y0=8,mask=torch.ones(2,2,dtype=torch.bool),weight=.25)]
    batch=dict(band_mask_instances=[[rows]],band_shape_source_ids=[[torch.tensor([10,20])]],
        band_shape_source_centers=[[c]],band_shape_source_values=[[s]],
        band_valid_mask=torch.ones(1,1,16,16,dtype=torch.bool),
        band_trusted_background=torch.zeros(1,1,16,16,dtype=torch.bool),
        band_clean_mask=torch.zeros(1,1,16,16),band_shape=torch.zeros(1,1,3,16,16))
    outputs=dict(confidence=torch.zeros(1,1,5,16,16),shape=torch.ones(1,1,3,16,16),
        image_embeddings=torch.zeros(1,1,1,1,1))
    p=instance_prompts(batch,outputs,'cpu');assert torch.equal(p['target_shapes'],s)
    model=Model(torch.zeros(1,1,16,16))
    out=sam_prompt_mask_losses(model,outputs,batch,weights=weights(),device=torch.device('cpu'),
        epoch_index=30,threshold=.5,nms_radius=2,confidence_score='ordinal_expectation',
        use_ordinal_expectation=True,debug_ordinal_expectation=False,center_refinement='none',
        center_refinement_radius=1,ellipse_sigma=1.,detect_centers_fn=lambda *a,**k:[torch.empty(0,2)])
    assert out['supervised_prompts']==2 and out['gt_prompts']==2
    assert out['dice']>0;out['total'].backward();assert model.logits.grad.abs().sum()>0


if __name__ == "__main__":
    suite = unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in list(globals().items()) if name.startswith("test_") and callable(value))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
