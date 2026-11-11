"""Optimized mask moments preserve candidate selection, targets and gradients."""
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sam_backbone.losses import _sam_mask_loss_for_prompts
from sam_backbone.mask_moments import (catalog_targets, moment_loss, native_ellipticity,
                                      probability_ellipticity)
from sam_backbone.mask_supervision import partial_targets
from tests.test_instance_mask_training import weights


def naive_targets(prompts,batch,image_hw,mask_hw):
    n=len(prompts['centers']);h,w=image_hw;flat=prompts['batch_indices']
    pos=torch.zeros(n,h,w)
    valid=(torch.ones_like(pos) if batch is None or 'band_valid_mask' not in batch else
           batch['band_valid_mask'].reshape(-1,h,w)[flat].float())
    neg=(torch.zeros_like(pos) if batch is None or 'band_trusted_background' not in batch else
         batch['band_trusted_background'].reshape(-1,h,w)[flat].float())
    pos*=valid;neg*=valid*(pos==0)
    if image_hw!=mask_hw:
        pos,neg,valid=[F.interpolate(x[:,None],size=mask_hw,mode='area')[:,0] for x in (pos,neg,valid)]
        neg=(neg>=1-1e-6).float()
    return pos,neg,valid


class Multi(torch.nn.Module):
    dynamic_image_size=False
    def __init__(self,z,iou):
        super().__init__();self.logits=torch.nn.Parameter(z.clone());self.iou=iou
    def forward_sam_masks(self,*args,**kwargs):return self.logits,self.iou


class OptimizedMomentTests(unittest.TestCase):
    def test_unlabelled_pool_before_gather_matches_reference(self):
        torch.manual_seed(3)
        for image_hw,mask_hw in [((16,24),(4,6)),((16,24),(16,24)),((8,12),(16,24))]:
            h,w=image_hw
            p=dict(centers=torch.zeros(7,2),batch_indices=torch.tensor([2,0,1,2,2,0,1]))
            valid=torch.rand(3,1,h,w)>.2;sky=torch.rand(3,1,h,w)>.2
            for b in [None,dict(band_valid_mask=valid),dict(band_trusted_background=sky),
                      dict(band_valid_mask=valid,band_trusted_background=sky)]:
                with self.subTest(image_hw=image_hw,mask_hw=mask_hw,fields=[] if b is None else list(b)):
                    expected=naive_targets(p,b,image_hw,mask_hw)
                    got=partial_targets(p,b,image_hw=image_hw,mask_hw=mask_hw,device='cpu')
                    for a,c in zip(expected,got[:3]):torch.testing.assert_close(a,c,rtol=0,atol=0)
                    self.assertTrue((got[3]==1).all());self.assertFalse(got[4].any())

    def test_image_coverage_indices_match_materialized_coverage_and_gradient(self):
        torch.manual_seed(11)
        z=torch.randn(7,3,4,6,requires_grad=True);indices=torch.tensor([1,0,2,2,1,0,2])
        valid=torch.rand(3,16,24)>.1
        expected=probability_ellipticity(F.interpolate(z,size=(16,24),mode='bilinear',align_corners=False).sigmoid()
                                         *valid[indices,None],(16,24))
        got=native_ellipticity(z,valid,(16,24),coverage_indices=indices,chunk_size=3)
        torch.testing.assert_close(got,expected)
        grad0=torch.autograd.grad(expected.square().sum(),z)[0]
        grad1=torch.autograd.grad(got.square().sum(),z)[0]
        torch.testing.assert_close(grad0,grad1,rtol=1e-5,atol=1e-7)

    def test_checkpoint_shares_image_coverage_not_per_prompt_copies(self):
        z=torch.randn(9,1,4,6,requires_grad=True)
        coverage=torch.ones(2,16,24,dtype=torch.bool);indices=torch.arange(9)%2;saved=[]
        def pack(t):
            if t.dtype==torch.bool:saved.append(t)
            return t
        with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
            e=native_ellipticity(z,coverage,(16,24),coverage_indices=indices,chunk_size=2)
        self.assertTrue(saved)
        self.assertTrue(all(t.untyped_storage().data_ptr()==coverage.untyped_storage().data_ptr() for t in saved))
        e.square().sum().backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_entire_loss_matches_all_candidate_reference_with_and_without_gt(self):
        for mode in ['pred_iou','loss']:
            for with_gt in [False,True]:
                with self.subTest(mode=mode,with_gt=with_gt):
                    torch.manual_seed(19);n=5;native=(16,24)
                    ids=torch.arange(2**55,2**55+n,dtype=torch.long)
                    centers=torch.tensor([[8.,8.],[9.,8.],[10.,8.],[1.,1.],[18.,8.]])
                    shapes=torch.tensor([[4.,2.,.3]]*n)
                    p=dict(centers=centers,batch_indices=torch.zeros(n,dtype=torch.long),
                        prompt_shapes=shapes,target_shapes=shapes,weights=torch.tensor([1.,.2,1.,1.,1.]),
                        source_ids=ids,source_kind=torch.zeros(n,dtype=torch.long),instance_masks=[None]*n)
                    # Even a zero-weight prompt must keep its diagnostic
                    # candidate choice when selection is loss-based.
                    if mode=='loss':p['weights'][2]=0.
                    if with_gt:p['instance_masks'][0]=dict(x0=7,y0=7,mask=torch.ones(2,5,dtype=torch.bool),weight=.25)
                    coverage=torch.ones(1,1,*native,dtype=torch.bool);coverage[0,0,8,18]=False
                    b=dict(band_names=[['HSC-I']],band_shape_source_ids=[[ids]],
                        band_shape_source_centers=[[centers]],band_shape_source_values=[[shapes]],
                        band_shape_source_classes=[[torch.tensor([1,2,2,1,1])]],
                        band_shape_source_moment_eligible=[[torch.tensor([True,False,True,True,True])]],
                        band_valid_mask=coverage)
                    z=torch.randn(n,3,4,6);iou=torch.tensor([[3.,2.,1.],[1.,2.,3.],[1.,3.,2.],[1.,2.,3.],[1.,2.,3.]])
                    model=Multi(z,iou);w=weights(mask_dice=0.,mask_bce=0.,mask_moment=.3,mask_selection=mode)
                    out=dict(confidence=torch.zeros(1,1,5,*native),image_embeddings=torch.zeros(1))
                    with patch('sam_backbone.losses.native_ellipticity',wraps=native_ellipticity) as spy:
                        loss=_sam_mask_loss_for_prompts(model,out,p,weights=w,image_hw=native,
                            ellipse_sigma=1.,batch=b,return_selection=True)
                    self.assertEqual(spy.call_count,1)
                    self.assertEqual(spy.call_args.args[0].shape[:2],(2,1 if mode=='pred_iou' else 3))
                    # Original all-three-candidates, full native interpolation.
                    ref=z.clone().requires_grad_()
                    pos,neg,valid,quality,labelled=partial_targets(p,b,image_hw=native,mask_hw=(4,6),device='cpu')
                    target=catalog_targets(p,b,bands=1)
                    e=probability_ellipticity(F.interpolate(ref,size=native,mode='bilinear',align_corners=False).sigmoid()
                                              *coverage.reshape(1,*native)[:,None],native)
                    values,active=moment_loss(ref.sigmoid()*valid[:,None],target,image_hw=native,
                        valid=valid,labelled=labelled,prompts=p,batch=b,prediction_e=e)
                    best=iou.argmax(1) if mode=='pred_iou' else values.argmin(1)
                    selected=values[torch.arange(n),best]
                    q=torch.where(labelled,quality,torch.ones_like(quality))
                    denom=(p['weights']*active).sum().clamp_min(1.)
                    expected=.3*(selected*p['weights']*active*q).sum()/denom
                    torch.testing.assert_close(loss['selected_indices'],best)
                    torch.testing.assert_close(loss['total'],expected,rtol=1e-5,atol=1e-7)
                    loss['total'].backward();expected.backward()
                    torch.testing.assert_close(model.logits.grad,ref.grad,rtol=2e-5,atol=1e-7)

    def test_all_ineligible_skips_native_interpolation_but_keeps_zero_gradient(self):
        from tests.test_mask_moments import batch
        from tests.test_instance_mask_training import prompts,run
        b=batch((2,),(False,));p=prompts();p['source_ids']=b['band_shape_source_ids'][0][0]
        with patch('sam_backbone.losses.native_ellipticity',wraps=native_ellipticity) as spy:
            out,model=run(torch.randn(1,1,16,16),p,weights(mask_dice=0.,mask_bce=0.,mask_moment=.3),b)
        spy.assert_not_called();out['total'].backward()
        self.assertEqual(float(out['total'].detach()),0.)
        self.assertTrue((model.logits.grad==0).all())


if __name__=='__main__':unittest.main()
