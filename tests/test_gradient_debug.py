"""Numerical and state-isolation checks for training gradient diagnostics."""
import copy
import json
import random
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from utils.gradient_debug import GradientDebugger, dataset_family, gradient_grams, geometry_metrics


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Linear(1,1,bias=False)
        self.decoder=nn.Linear(1,1,bias=False)
        self.mask_decoder=nn.Linear(1,1,bias=False)
        self.prompt_encoder=nn.Linear(1,1,bias=False)
        self.dropout=nn.Dropout(.5)
        for p in self.parameters():nn.init.constant_(p,1.)

    def forward(self,x):
        encoded=self.encoder(x)
        source=self.decoder(encoded)
        mask=self.mask_decoder(source)+self.prompt_encoder(encoded)
        return source,mask


def make_data():
    records=[SimpleNamespace(relative_root=p,name=p,bands=('F115W',)) for p in
             ('abell/half','cosmos/coadd','hsc/coadd','hsc/noisy')]
    samples=[dict(image=torch.tensor([float(i+1)]),name=r.name,relative_root=r.relative_root,
                  dataset_source='half',band_names=['F115W']) for i,r in enumerate(records)]
    return records,samples


def collate(items):return dict(image=torch.stack([v['image'] for v in items]))


def loss_fn(model,batch):
    a,b=model(batch['image'])
    return dict(proposal_loss=a.mean(),mask_loss=-2*b.mean())


class GradientDebugTests(unittest.TestCase):
    def test_dataset_names(self):
        self.assertEqual(dataset_family(dict(relative_root='cosmos/proposal_1727/image_level')),'cosmos')
        self.assertEqual(dataset_family(dict(name='COSMOS_5893/P0019')),'cosmos')
        self.assertEqual(dataset_family(dict(image_paths=['/data/A2744/image.fits'])),'abell')
        self.assertEqual(dataset_family(dict(relative_root='abell/image_level')),'abell')
        self.assertEqual(dataset_family(dict(bands=['HSC-I'])),'hsc')
        self.assertEqual(dataset_family(dict(bands=['NB0921'])),'hsc')

    def test_dataset_groups_ignore_experiment_directory_names(self):
        root='/data/training_zarr_cosmos2_gaia24/zarr'
        records=[SimpleNamespace(relative_root=rel, name=name,
                    image_paths=(f'zarr://{root}/{rel}/parent.zarr#0',))
                 for rel,name in (
                     ('hsc/image_level/half_coadd/HSC-I','9813/half_coadd/0,0/0/HSC-I/grid'),
                     ('cosmos/proposal_1727/image_level/coadd/F444W','COSMOS_1727/coadd/parent'),
                     ('abell/image_level/half_coadd/F115W','Abell2744/half_coadd/parent'))]
        self.assertEqual([dataset_family(r) for r in records],['hsc','cosmos','abell'])
        # Path-only fallback also handles copied stores without relative metadata.
        self.assertEqual(dataset_family(dict(image_paths=[root+'/hsc/image.zarr#0'])),'hsc')
        self.assertEqual(dataset_family(dict(image_paths=[root+'/unknown/image.zarr#0'])),'other')
        with tempfile.TemporaryDirectory() as tmp:
            debugger=GradientDebugger([],records,collate,tmp,samples_per_dataset=4)
            self.assertEqual({k:len(v) for k,v in debugger.groups.items()},
                             {'hsc':1,'cosmos':1,'abell':1})

    def test_geometry_exact_and_zero_angles_undefined(self):
        vectors=[{'decoder.a':torch.tensor([1.,0.])},{'decoder.a':torch.tensor([0.,1.])},
                 {'decoder.a':torch.tensor([-1.,0.])},{}]
        grams=gradient_grams(vectors,{'decoder.a':'proposal_decoder'})
        metrics=geometry_metrics(grams,['abell','cosmos'],{'abell':.25,'cosmos':.75})
        self.assertAlmostEqual(metrics['proposal/proposal_loss/abell__cosmos/angle_deg'],180.)
        self.assertAlmostEqual(metrics['proposal/mask_vs_proposal/abell/angle_deg'],90.)
        self.assertAlmostEqual(metrics['proposal/total/norm/full'],np.linalg.norm([-.5,.25]))
        self.assertEqual(metrics['mask/total/abell__cosmos/defined'],0.)
        self.assertNotIn('mask/total/abell__cosmos/angle_deg',metrics)

    def test_measure_preserves_weights_gradients_modes_and_rng(self):
        records,data=make_data();model=TinyModel().train();model.mask_decoder.eval()
        optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        for p in model.parameters():p.grad=torch.ones_like(p)*17
        expected=[(p.detach().clone(),p.grad.clone()) for p in model.parameters()]
        modes=[m.training for m in model.modules()]
        random.seed(99);np.random.seed(99);torch.manual_seed(99)
        py=random.getstate();npstate=np.random.get_state();rng=torch.random.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as tmp:
            debugger=GradientDebugger(data,records,collate,tmp,samples_per_dataset=0)
            result=debugger.measure(model,optimizer,device=torch.device('cpu'),loss_fn=loss_fn)
        self.assertEqual(result['probe_samples'],{'abell':1,'cosmos':1,'hsc':2})
        metrics=result['metrics']
        self.assertAlmostEqual(metrics['proposal_decoder/proposal_loss/norm/full'],2.5)
        self.assertAlmostEqual(metrics['proposal_decoder/mask_loss/norm/full'],5.)
        self.assertAlmostEqual(metrics['proposal_decoder/total/norm/full'],2.5)
        self.assertAlmostEqual(metrics['proposal_decoder/mask_vs_proposal/full/angle_deg'],180.)
        self.assertEqual([m.training for m in model.modules()],modes)
        for p,(weight,grad) in zip(model.parameters(),expected):
            torch.testing.assert_close(p,weight);torch.testing.assert_close(p.grad,grad)
        self.assertEqual(random.getstate(),py)
        np.testing.assert_equal(np.random.get_state(),npstate)
        torch.testing.assert_close(torch.random.get_rng_state(),rng,rtol=0,atol=0)

    def test_interval_files_frozen_and_lr_zero(self):
        records,data=make_data();model=TinyModel()
        optimizer=torch.optim.SGD([{'params':model.encoder.parameters(),'lr':0},
                                  {'params':list(model.decoder.parameters())+list(model.mask_decoder.parameters())+list(model.prompt_encoder.parameters()),'lr':.1}])
        with tempfile.TemporaryDirectory() as tmp:
            logs=[];debugger=GradientDebugger(data,records,collate,tmp,interval=2,log_fn=lambda *x:logs.append(x))
            kwargs=dict(model=model,optimizer=optimizer,device=torch.device('cpu'),loss_fn=loss_fn,
                        epoch=3,iteration=10,frozen_prefixes=('decoder.',))
            self.assertIsNone(debugger.maybe_probe(optimizer_step=1,**kwargs))
            result=debugger.maybe_probe(optimizer_step=2,**kwargs)
            self.assertEqual(result['metrics']['parameters/encoder'],0)
            self.assertEqual(result['metrics']['parameters/proposal_decoder'],0)
            self.assertEqual(len(logs),1)
            saved=json.loads((Path(tmp)/'metrics.jsonl').read_text())
            self.assertEqual(saved['iteration'],10)
            self.assertIn('fixed sampled',saved['full_definition'])

    def test_zero_mask_loss_and_sample_cache(self):
        records,data=make_data();model=TinyModel();optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        with tempfile.TemporaryDirectory() as tmp:
            debugger=GradientDebugger(data,records,collate,tmp,samples_per_dataset=1)
            def no_mask(model,batch):
                return dict(proposal_loss=model(batch['image'])[0].mean(),mask_loss=torch.tensor(0.))
            result=debugger.measure(model,optimizer,device=torch.device('cpu'),loss_fn=no_mask)
            self.assertEqual(len(debugger.cache),3)
            self.assertEqual(result['metrics']['mask/total/norm/full'],0)
            self.assertEqual(result['metrics']['proposal/mask_vs_proposal/full/defined'],0)

    def test_failure_is_logged_and_nonfatal(self):
        records,data=make_data();model=TinyModel();optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        with tempfile.TemporaryDirectory() as tmp:
            debugger=GradientDebugger(data,records,collate,tmp,interval=1)
            def fail(*args):raise ValueError('diagnostic failure')
            result=debugger.maybe_probe(model,optimizer,optimizer_step=1,iteration=1,epoch=0,
                device=torch.device('cpu'),loss_fn=fail)
            self.assertIsNone(result)
            self.assertIn('diagnostic failure',(Path(tmp)/'errors.jsonl').read_text())

    def test_training_updates_unchanged_with_real_probe_hook(self):
        import astro_train_ops as ops
        from sam_backbone.losses import _zero_mask_losses
        records,data=make_data();plain=TinyModel();debug=copy.deepcopy(plain)
        weights=ops.LossWeights(mask_outer_weight=2.,mask_loss_warmup_epochs=0)
        def forward(model,image,batch):
            source,mask=model(image)
            return dict(confidence=source,mask_value=mask)
        def dense(outputs,*args,**kwargs):
            z=outputs['confidence'].sum()*0
            return {key:(outputs['confidence'].square().mean() if key in ('total','confidence') else z)
                    for key in ('total','seg','confidence','shape','center','small_shape','style_router')}
        def masks(model,outputs,batch,**kwargs):
            result=_zero_mask_losses(outputs['confidence'])
            result['total']=outputs['mask_value'].square().mean()
            return result
        common=dict(device=torch.device('cpu'),weights=weights,triplet_loss_fn=None,
            triplet_enabled=False,ex_enabled=False,en_enabled=False,matcher_candidate_count=1,
            matcher_max_anchors_per_band=1,triplet_max_sources_per_group=1,
            triplet_negative_scope='batch',ex_band_pairs=None,center_radius_px=3,
            show_progress=False,mask_loss_interval=2,mask_loss_interval_scale=True)
        with tempfile.TemporaryDirectory() as tmp:
            probe=GradientDebugger(data,records,collate,tmp,interval=1)
            with patch.object(ops,'model_forward_with_batch_context',forward), \
                 patch.object(ops,'dense_losses_any',dense),patch.object(ops,'sam_prompt_mask_losses',masks):
                for model,debugger in ((plain,None),(debug,probe)):
                    ops.run_epoch(model,[dict(image=torch.tensor([[1.]])),dict(image=torch.tensor([[2.]]))],
                        optimizer=torch.optim.SGD(model.parameters(),lr=.01),gradient_debugger=debugger,**common)
            results=[json.loads(row) for row in (Path(tmp)/'metrics.jsonl').read_text().splitlines()]
            self.assertEqual(len(results),2)
            self.assertEqual(results[0]['metrics']['data/abell/mask_interval_scale'],2.)
            self.assertEqual(results[1]['metrics']['mask/total/norm/full'],0.)
        for a,b in zip(plain.parameters(),debug.parameters()):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            if a.grad is None:self.assertIsNone(b.grad)
            else:torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
