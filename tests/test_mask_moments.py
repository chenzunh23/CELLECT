"""Moment correctness, source eligibility, ID association and loss integration."""
import unittest
import numpy as np
import torch
from astropy.table import Table

from preprocessing.utils.geometry import EllipseGeometry
from preprocessing.utils.mask_moments import moment_eligibility, JWST_POLICY
from sam_backbone.mask_moments import (probability_ellipticity,ellipse_ellipticity,
                                      catalog_targets,moment_loss,native_ellipticity)
from sam_backbone.mask_supervision import combine_prompts
from sam_backbone.mask_metrics import GROUP_STAT_KEYS,finish_group_metrics
from tests.test_instance_mask_training import prompts,weights,run


def batch(classes=(1,2,2),flags=(True,False,True),name='HSC-I'):
    n=len(classes);ids=torch.arange(43158185032286339,43158185032286339+n,dtype=torch.long)
    return dict(band_names=[[name]],band_shape_source_ids=[[ids]],
        band_shape_source_centers=[[torch.tensor([[8.+i*8,8.] for i in range(n)])]],
        band_shape_source_values=[[torch.tensor([[4.,2.,0.]]*n)]],
        band_shape_source_classes=[[torch.tensor(classes)]],
        band_shape_source_moment_eligible=[[None if flags is None else torch.tensor(flags)]])


class MaskMomentTests(unittest.TestCase):
    def test_native_moments_match_direct_numpy_covariance(self):
        rng=np.random.default_rng(7);p=rng.random((7,11));xy=np.array([(x,y) for y in range(7) for x in range(11)],float)
        xy*=np.array([44/11,14/7]);xy+=np.array([44/11/2-.5,14/7/2-.5])
        mean=np.average(xy,axis=0,weights=p.ravel());d=xy-mean
        c=np.einsum('n,ni,nj->ij',p.ravel(),d,d)/p.sum()
        expected=np.array([c[0,0]-c[1,1],2*c[0,1]])/np.trace(c)
        got=probability_ellipticity(torch.tensor(p,dtype=torch.float32)[None,None],(14,44))
        np.testing.assert_allclose(got.numpy()[0,0],expected,atol=1e-6)

    def test_scale_and_pi_rotation_invariance_and_finite_gradients(self):
        s=torch.tensor([[4.,2.,.7],[12.,6.,.7+np.pi]])
        torch.testing.assert_close(ellipse_ellipticity(s)[0],ellipse_ellipticity(s)[1])
        p=torch.rand(2,3,16,24,requires_grad=True)
        probability_ellipticity(p,(32,96)).square().sum().backward()
        self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(float(p.grad.abs().sum()),0)
        empty=torch.zeros(1,1,4,4,requires_grad=True)
        probability_ellipticity(empty,(4,4)).sum().backward()
        self.assertTrue(torch.isfinite(empty.grad).all())

    def test_lowres_prediction_uses_native_resize_before_sigmoid(self):
        z=torch.randn(2,3,4,6,requires_grad=True);coverage=torch.ones(2,16,24,dtype=torch.bool)
        coverage[:,:2]=False
        expected=probability_ellipticity(torch.nn.functional.interpolate(z,size=(16,24),
            mode='bilinear',align_corners=False).sigmoid()*coverage[:,None],(16,24))
        result=native_ellipticity(z,coverage,(16,24),chunk_size=1)
        torch.testing.assert_close(result,expected)
        result.square().sum().backward();self.assertTrue(torch.isfinite(z.grad).all())

    def test_source_policy_excludes_snr_weak_and_blends(self):
        reasons=['ordinary_snr_clean','ordinary_snr_weak_shape',
                 'single_cluster_large_bright_component_weak_shape','isolated_bright_galaxy_weak_shape',
                 'high_blendedness_requires_gaia','isolated_bright_star_strict_center_only']
        classes=np.array([1,2,2,2,2,4]);x=np.arange(6)*20.+10
        g=EllipseGeometry(x,x,np.ones(6)*3,np.ones(6)*2,np.zeros(6),np.ones(6)*6*np.pi)
        t=Table(dict(classification_reason=reasons))
        np.testing.assert_array_equal(moment_eligibility(t,classes,g),[True,False,True,True,False,False])

    def test_gaia_psf_weak_must_be_nonoverlapping(self):
        t=Table(dict(classification_reason=['hsc_gaia_psf_sat_within_ee95','ordinary_snr_clean'],gaia_psf_model=[True,False]))
        x=np.array([10.,12.]);g=EllipseGeometry(x,x,np.ones(2)*3,np.ones(2)*3,np.zeros(2),np.ones(2)*9*np.pi)
        self.assertFalse(moment_eligibility(t,np.array([2,1]),g)[0])
        g.x[1]=30;g.y[1]=30
        self.assertTrue(moment_eligibility(t,np.array([2,1]),g)[0])

    def test_jwst_all_clean_and_only_numerically_high_snr_weak(self):
        classes=np.array([1,2,2,2,2,2,4,3])
        t=Table(dict(classification_snr=[np.nan,6.99,7.,20.,np.nan,np.inf,100.,100.]))
        x=np.arange(len(classes))*10.+10
        g=EllipseGeometry(x,x,np.full(len(x),3.),np.full(len(x),2.),
                          np.zeros(len(x)),np.full(len(x),6*np.pi))
        for dataset in ['cosmos','abell']:
            np.testing.assert_array_equal(moment_eligibility(t,classes,g,dataset=dataset),
                                          [True,False,True,True,False,False,False,False])
        np.testing.assert_array_equal(moment_eligibility(t,classes,g,dataset='abell',weak_snr_threshold=20),
                                      [True,False,False,True,False,False,False,False])
        g.major[0]=np.nan
        self.assertFalse(moment_eligibility(t,classes,g,dataset='cosmos')[0])

    def test_jwst_uses_measured_snr_and_never_catalog_fallback_for_bad_rows(self):
        t=Table(masked=True,data=dict(remeasured_aperture_snr=[np.nan,6.,7.,100.],
                                     classification_snr=[100.]*4,snr_f444w=[100.]*4))
        t['remeasured_aperture_snr'].mask=[False,False,False,True]
        x=np.arange(4)*10.+10
        g=EllipseGeometry(x,x,np.full(4,3.),np.full(4,2.),np.zeros(4),np.full(4,6*np.pi))
        np.testing.assert_array_equal(moment_eligibility(t,np.full(4,2),g,dataset='abell'),
                                      [False,False,True,False])
        t=Table(dict(snr_f444w=[8.,6.,7.,np.nan],snr_f115w=[100.]*4))
        np.testing.assert_array_equal(moment_eligibility(t,np.full(4,2),g,dataset='cosmos',band='F444W'),
                                      [True,False,True,False])

    def test_jwst_catalog_fallback_is_active_and_backpropagates(self):
        for cls,flag,enabled in [(1,False,True),(2,True,True),(2,False,False),(4,True,False)]:
            b=batch((cls,),(flag,),name='F444W')
            b['proposal_metadata']=[dict(dataset='cosmos',mask_moment_policy=JWST_POLICY)]
            p=prompts();p['source_ids']=b['band_shape_source_ids'][0][0]
            z=torch.full((1,1,16,16),-8.);z[:,:,6:10,7:9]=5.
            out,model=run(z,p,weights(mask_dice=0.,mask_bce=0.,mask_moment=.3),b)
            out['total'].backward()
            self.assertEqual(float(out['moment'])>0,enabled)
            self.assertEqual(float(model.logits.grad.abs().sum())>0,enabled)
            self.assertTrue(torch.isfinite(model.logits.grad).all())

    def test_jwst_legacy_weak_flags_are_not_proof_of_high_snr(self):
        b=batch((1,2),(False,True),name='F444W')
        b['proposal_metadata']=[dict(mask_moment_policy='clean_or_isolated_bright_weak_v1')]
        p=prompts(2);p['source_ids']=b['band_shape_source_ids'][0][0]
        with self.assertWarnsRegex(RuntimeWarning,'high-SNR weak'):
            target=catalog_targets(p,b,bands=1)
        self.assertEqual(target['eligible'].tolist(),[True,False])

    def test_jwst_predictions_match_nearest_before_applying_weak_eligibility(self):
        b=batch(name='F444W');b['proposal_metadata']=[dict(dataset='abell',mask_moment_policy=JWST_POLICY)]
        p=prompts(3);p['source_kind']=torch.ones(3,dtype=torch.long)
        p['centers']=torch.tensor([[8.1,8.],[16.1,8.],[24.1,8.]])
        target=catalog_targets(p,b,bands=1)
        self.assertEqual(target['matched'].tolist(),[True,True,True])
        self.assertEqual(target['eligible'].tolist(),[True,False,True])

    def test_exact_large_ids_and_unique_predicted_matches(self):
        b=batch();p=prompts(3);p['source_ids']=b['band_shape_source_ids'][0][0].clone()
        # GT source IDs, not location or dense overlap, determine targets.
        p['centers'][:]=99
        t=catalog_targets(p,b,bands=1)
        torch.testing.assert_close(t['eligible'],torch.tensor([True,False,True]))
        torch.testing.assert_close(t['centers'],b['band_shape_source_centers'][0][0])
        p['source_kind']=torch.ones(3,dtype=torch.long)
        p['centers']=torch.tensor([[8.2,8.],[8.1,8.],[16.1,8.]])
        t=catalog_targets(p,b,bands=1)
        self.assertEqual(t['matched'].tolist(),[False,True,True])
        self.assertEqual(t['eligible'].tolist(),[False,True,False])

    def test_missing_weak_metadata_fails_instead_of_guessing(self):
        b=batch(flags=None);p=prompts();p['source_ids']=b['band_shape_source_ids'][0][0][:1]
        with self.assertRaisesRegex(ValueError,'shape_source_moment_eligible'):
            catalog_targets(p,b,bands=1)
        t=catalog_targets(p,b,bands=1,require_eligibility=False)
        self.assertTrue(t['eligible'][0])

    def test_prompt_mix_retains_gt_ids_without_narrowing_int64(self):
        b=batch();p=prompts();p['source_ids']=b['band_shape_source_ids'][0][0][:1]
        out=combine_prompts(None,p,None,pred_ratio=0)
        self.assertEqual(int(out['source_ids'][0]),43158185032286339)

    def test_hsc_area_bounds_do_not_change_jwst_bounds(self):
        z=torch.full((1,1,16,16),-30.);z[:,:,7:9,7:9]=30.
        p=prompts();w=weights(mask_dice=0.,mask_bce=0.,mask_min_area=1.)
        h,_=run(z,p,w,dict(band_names=[['HSC-I']]))
        j,_=run(z,p,w,dict(band_names=[['F444W']]))
        self.assertGreater(float(h['area']),1.)
        self.assertLess(float(j['area']),1e-5)

    def test_moment_gradient_is_gated_by_sparse_eligibility(self):
        for cls,flag,enabled in [(1,True,True),(2,False,False),(2,True,True)]:
            b=batch((cls,),(flag,));p=prompts();p['source_ids']=b['band_shape_source_ids'][0][0]
            z=torch.full((1,1,16,16),-8.);z[:,:,6:10,7:9]=5.
            out,m=run(z,p,weights(mask_dice=0.,mask_bce=0.,mask_moment=.3),b)
            out['total'].backward()
            self.assertEqual(float(out['moment'])>0,enabled)
            self.assertEqual(float(m.logits.grad.abs().sum())>0,enabled)
            self.assertTrue(torch.isfinite(m.logits.grad).all())

    def test_gt_instance_moments_are_not_catalog_ellipse_moments(self):
        b=batch((1,),(True,),name='F444W');p=prompts();p['source_ids']=b['band_shape_source_ids'][0][0]
        p['instance_masks']=[dict(x0=6,y0=7,mask=torch.ones(2,4,dtype=torch.bool),weight=1.)]
        z=torch.full((1,1,16,16),-40.);z[:,:,7:9,6:10]=40.
        out,_=run(z,p,weights(mask_dice=0.,mask_bce=0.,mask_moment=1.),b)
        self.assertLess(float(out['moment']),1e-7)

    def test_incomplete_catalog_source_does_not_train_full_shape(self):
        p=prompts();t=dict(shapes=torch.tensor([[4.,2.,0.]]),centers=torch.tensor([[1.,1.]]),
                         eligible=torch.tensor([True]),is_hsc=torch.tensor([True]))
        values,active=moment_loss(torch.rand(1,1,16,16),t,image_hw=(16,16),
            valid=torch.ones(1,16,16),labelled=torch.tensor([False]),prompts=p,batch=None)
        self.assertFalse(active[0]);self.assertEqual(float(values.sum()),0)

    def test_jwst_fallback_excludes_catalog_clipping_and_coverage_holes(self):
        p=prompts();t=dict(shapes=torch.tensor([[4.,2.,0.]]),centers=torch.tensor([[1.,1.]]),
                         eligible=torch.tensor([True]),is_hsc=torch.tensor([False]))
        p['instance_masks']=[None]
        for clipped in [True,False]:
            valid=torch.ones(1,16,16)
            if not clipped:
                t['centers']=torch.tensor([[8.,8.]]);valid[:,8,8]=0
            values,active=moment_loss(torch.rand(1,1,16,16),t,image_hw=(16,16),
                valid=valid,labelled=torch.tensor([False]),prompts=p,batch=None)
            self.assertFalse(active[0]);self.assertEqual(float(values.sum()),0)

    def test_jwst_high_snr_weak_instance_target_has_priority_over_catalog(self):
        b=batch((2,),(True,),name='F444W')
        b['proposal_metadata']=[dict(dataset='abell',mask_moment_policy=JWST_POLICY)]
        p=prompts();p['source_ids']=b['band_shape_source_ids'][0][0]
        # Catalog ellipse is horizontal; GT and prediction are vertical.
        p['instance_masks']=[dict(x0=7,y0=6,mask=torch.ones(4,2,dtype=torch.bool),weight=1.)]
        z=torch.full((1,1,16,16),-40.);z[:,:,6:10,7:9]=40.
        out,_=run(z,p,weights(mask_dice=0.,mask_bce=0.,mask_moment=1.),b)
        self.assertLess(float(out['moment']),1e-7)

    def test_moment_diagnostics_use_eligible_denominator(self):
        from sam_backbone.mask_metrics import group_statistics
        selected=dict(moment=torch.tensor([2.,0.]))
        stats=group_statistics(selected,torch.tensor([0,0]),torch.ones(2),torch.zeros(2,dtype=torch.bool),
            torch.zeros(2,dtype=torch.bool),torch.ones(2),1.,moment_active=torch.tensor([True,False]))
        metrics=finish_group_metrics(dict(zip(GROUP_STAT_KEYS,stats.tolist())),weights(mask_moment=.3))
        self.assertEqual(metrics['mask_gt_mean_moment'],2.)
        self.assertEqual(metrics['mask_gt_moment_count'],1)
        self.assertEqual(metrics['mask_gt_catalog_moment_count'],1)
        self.assertEqual(metrics['mask_gt_instance_moment_count'],0)


if __name__=='__main__':unittest.main()
