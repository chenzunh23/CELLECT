"""Fixed probe resolution and exact cohort gradient aggregation."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from tests.test_gradient_debug import TinyModel,make_data,collate,loss_fn
from utils.gradient_debug import GradientDebugger
from utils.gradient_debug_probes import resolve_probe_config


class AugmentedList(list):
    augment=True


class ProbeConfigTests(unittest.TestCase):
    def write_config(self,folder,config):
        path=Path(folder)/'probes.json';path.write_text(json.dumps(config));return path

    def test_evaluation_grid_uses_512_not_training_stride(self):
        records=[SimpleNamespace(name=f'Abell2744/half_coadd/p/F444W/tile{i}',patch='p',
                    relative_root='abell/image_level/half_coadd/F444W',dataset_source='half_coadd',
                    x0=x,y0=y,tile_name=f'grid{i}',image_paths=())
                 for i,(x,y) in enumerate(((1472,736),(1840,1104),(2208,1104),(2048,1024)))]
        config=dict(datasets={'abell':dict(random_samples=0,probes=[dict(id='halo',patch='p',
            band='F444W',grid=[4,2],selection='exact',cohort='red_halo')])})
        with tempfile.TemporaryDirectory() as tmp:
            path=self.write_config(tmp,config)
            indices,cohorts,by,manifest=resolve_probe_config(records,{'abell':list(range(4))},path,seed=7,default_samples=2)
        self.assertEqual(indices,{'abell':[3]})
        self.assertEqual(cohorts,{'red_halo':[3]})
        self.assertEqual(by[3][0]['requested_bounds'],[2048,1024,2560,1536])
        self.assertEqual(manifest['case_matches'][0]['matches'][0]['overlap_fraction'],1)

    def test_overlap_selection_is_ranked_deduplicated_and_strict(self):
        records=[dict(name=f'tile{i}',patch='p',bands=['F444W'],dataset_source='half_coadd',
                      x0=x,y0=y,tile_name='grid',image_paths=())
                 for i,(x,y) in enumerate(((1840,1104),(2208,1104),(1472,736)))]
        case=dict(id='halo',patch='p',band='F444W',grid=[4,2],max_tiles=2,cohort='halo')
        config=dict(datasets={'abell':dict(random_samples=0,probes=[case,dict(case,id='duplicate')])})
        with tempfile.TemporaryDirectory() as tmp:
            path=self.write_config(tmp,config)
            indices,cohorts,by,_=resolve_probe_config(records,{'abell':[0,1,2]},path,seed=7,default_samples=2)
            self.assertEqual(indices['abell'],[1,0])
            self.assertEqual(cohorts['halo'],[1,0])
            self.assertEqual(len(by[1]),2)
            config['datasets']['abell']['probes']=[dict(case,selection='exact')]
            path=self.write_config(tmp,config)
            with self.assertRaisesRegex(ValueError,'No training Zarr'):
                resolve_probe_config(records,{'abell':[0,1,2]},path,seed=7,default_samples=2)

    def test_cohorts_reuse_gradients_and_do_not_change_full_mixture(self):
        records,data=make_data();model=TinyModel();optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        data=AugmentedList(data)
        config=dict(random_samples_per_dataset=2,datasets={
            'abell':dict(random_samples=0,probes=[dict(id='halo',name=records[0].name,cohort='abell_red_halo')]),
            'hsc':dict(random_samples=0,probes=[dict(id=f'hsc{i}',name=records[i].name,cohort='hsc_examples') for i in (2,3)])})
        with tempfile.TemporaryDirectory() as tmp:
            path=self.write_config(tmp,config)
            original=GradientDebugger(data,records,collate,Path(tmp)/'base',samples_per_dataset=0)
            expected=original.measure(model,optimizer,device=torch.device('cpu'),loss_fn=loss_fn)
            configured=GradientDebugger(data,records,collate,Path(tmp)/'configured',probe_config=path)
            self.assertFalse(configured.dataset.augment)
            self.assertTrue(data.augment)
            calls=[]
            def counted(model,batch):calls.append(1);return loss_fn(model,batch)
            actual=configured.measure(model,optimizer,device=torch.device('cpu'),loss_fn=counted)
            self.assertEqual(len(calls),len(records))
            for key,value in expected['metrics'].items():
                if key.startswith(('mask/','proposal/','encoder/','proposal_decoder/','full_model/')):
                    self.assertAlmostEqual(actual['metrics'][key],value,places=6,msg=key)
            self.assertEqual(actual['cohort_samples'],{'abell_red_halo':1,'hsc_examples':2})
            self.assertEqual(actual['metrics']['data/abell_red_halo/mixture_weight'],0)
            self.assertAlmostEqual(actual['metrics']['proposal/total/abell_red_halo__abell/cosine'],1)
            self.assertAlmostEqual(actual['metrics']['proposal/total/norm/hsc_examples'],
                                   actual['metrics']['proposal/total/norm/hsc'])
            self.assertIn('probe_selections',actual['metadata']['abell'][0])
            self.assertTrue((Path(tmp)/'configured/probes_resolved.json').exists())
            with self.assertRaisesRegex(ValueError,'batch-size=1'):
                GradientDebugger(data,records,collate,tmp,probe_config=path,batch_size=2)


if __name__=='__main__':unittest.main()
