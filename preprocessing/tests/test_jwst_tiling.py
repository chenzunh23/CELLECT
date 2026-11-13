from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import zarr
from astropy.table import Table
from preprocessing.utils.large_sources import large_source_specs, valid_tile_specs, crop_padded
from preprocessing.utils.inputs import TileSpec
from preprocessing.utils.parent_scaling import fit_parent_rgb, apply_parent_rgb
from preprocessing.tests.test_segmentation_zarr import task, labels
from preprocessing.build_image_level_zarr import write_classified_patch
from preprocessing.make_zarr_patch_splits import scan_stores, build_candidates, validation_exclusions
from data_filtering.sam_input_scaling import scale_training_image


class JWSTTilingTests(unittest.TestCase):
    def test_nonlinear_input_is_unclipped_zscore_with_control_parameters(self):
        from data_filtering import sam_input_scaling as scaling
        image=np.random.default_rng(51).normal(27.,.03,(64,64)).astype('f4')
        image[12:15,20:23]+=10
        image[0,0]=np.nan
        _,stats=scaling.current_sam_zscore(image)
        expected=(image-stats['zscore_median'])/stats['std']
        self.assertGreater(np.nanmax(expected),100.)
        with patch.object(scaling,'lupton_single',wraps=scaling.lupton_single) as lupton:
            rgb=scaling.scale_training_image(image,mode='zscore-log-lupton-rgb',clip_threshold=5,log_a=1000)
        np.testing.assert_allclose(lupton.call_args.args[0],expected,equal_nan=True)
        self.assertEqual(lupton.call_args.kwargs,dict(minimum=0.,stretch=.5,q=20.))
        plane,_=scaling.lupton_single(expected,minimum=0.,stretch=.5,q=20.)
        _,want,_=scaling.standardize_by_self(plane,clip_threshold=5)
        np.testing.assert_array_equal(rgb[2],want)
        with patch.object(scaling,'lupton_single',wraps=scaling.lupton_single) as lupton:
            bright=scaling.build_bright_mask(image,threshold=5,clip_threshold=5,log_a=1000,dilation=0)
        np.testing.assert_allclose(lupton.call_args.args[0],expected,equal_nan=True)
        np.testing.assert_array_equal(bright,(rgb[1]>=5)&(rgb[2]>=5)&np.isfinite(image))

    def test_new_scaling_is_invariant_to_intensity_units(self):
        image=np.random.default_rng(8).normal(2.,.7,(64,64)).astype('f4')
        image[10:13,20:23]+=40
        a=scale_training_image(image,mode='zscore-log-lupton-rgb',clip_threshold=5)
        b=scale_training_image(image*8+16,mode='zscore-log-lupton-rgb',clip_threshold=5)
        np.testing.assert_allclose(a,b,atol=2e-5)

    def test_stale_parent_scaling_cannot_be_silently_reused(self):
        image=np.random.default_rng(10).normal(size=(32,32)).astype('f4')
        params=fit_parent_rgb(image)
        self.assertEqual(params.pop('scaling_version'),'zscore_first_v1')
        with self.assertRaisesRegex(ValueError,'refit'):
            apply_parent_rgb(image,params)

    def test_training_log_reports_saved_legacy_and_new_definitions(self):
        import contextlib
        import io
        from types import SimpleNamespace
        from astro_train_zarr_data import log_zarr_scaling
        from data_filtering.sam_input_scaling import scaling_contract
        legacy=dict(image_scaling_mode='zscore-log-lupton-rgb',image_lupton_q=20.,
                    image_lupton_stretch=.5,bands=['F115W'])
        new={**legacy,'image_scaling_contract':scaling_contract()}
        rows=[(Path('/old.zarr'),SimpleNamespace(attrs=legacy),Path('/')),
              (Path('/new.zarr'),SimpleNamespace(attrs=new),Path('/'))]
        text=io.StringIO()
        with patch('astro_train_zarr_data._image_store_readers',return_value=iter(rows)),contextlib.redirect_stdout(text):
            summary=log_zarr_scaling(Path('/unused'),bands=['F115W'])
        self.assertFalse(summary['runtime_rescaling'])
        self.assertEqual(len(summary['definitions']),2)
        self.assertIn('legacy_raw_first_unversioned',text.getvalue())
        self.assertIn('zscore_first_v1',text.getvalue())
        self.assertIn('"lupton_q": 20.0',text.getvalue())

    def test_generation_banner_precedes_inventory_and_uses_control_defaults(self):
        import contextlib
        import io
        from preprocessing import build_image_level_zarr as entry
        args=entry.parse_args(['--list-inputs','--output-root','/tmp/cellect-scaling-unused'])
        text=io.StringIO()
        with patch.object(entry,'parse_args',return_value=args), \
             patch.object(entry,'write_input_inventory') as inventory,contextlib.redirect_stdout(text):
            self.assertEqual(entry.main(),0)
        inventory.assert_called_once_with(args)
        line=text.getvalue().splitlines()[0]
        self.assertTrue(line.startswith('[scaling] Zarr generation '))
        definition=json.loads(line.split('Zarr generation ',1)[1])
        self.assertEqual(definition['image']['version'],'zscore_first_v1')
        self.assertEqual(definition['image']['lupton_q'],20.)
        self.assertEqual(definition['image']['lupton_stretch'],.5)

    def test_training_banner_precedes_model_and_gpu_setup(self):
        import contextlib
        import io
        from types import SimpleNamespace
        import astro_train_eval as entry
        args=SimpleNamespace(data_format='zarr',root='/unused',hsc_zarr_root=None,
            space_zarr_root=None,bands=['F115W'],zarr_random_image_batches=True,mask_moment_weight=-1.)
        parser=SimpleNamespace(parse_args=lambda:args)
        text=io.StringIO()
        with patch.object(entry,'build_arg_parser',return_value=parser), \
             patch('astro_train_zarr_data._image_store_readers',return_value=iter([])), \
             patch.object(entry,'_setup_distributed') as gpu, \
             patch.dict('os.environ',{'RANK':'0'}),contextlib.redirect_stdout(text):
            with self.assertRaisesRegex(ValueError,'mask-moment-weight'):
                entry.main()
        gpu.assert_not_called()
        self.assertTrue(text.getvalue().startswith('[scaling] Training input=stored_zarr_passthrough'))

    def test_invalid_threshold_before_fill(self):
        spec=TileSpec('test',0,0,512);valid=np.ones((512,512),bool)
        valid.flat[:26214]=False
        self.assertEqual(len(valid_tile_specs([spec],valid,(0,0),.10)),1)
        valid.flat[26214]=False
        self.assertEqual(len(valid_tile_specs([spec],valid,(0,0),.10)),0)

    def test_large_policy_and_center_padding(self):
        lab=labels();lab.geom_major=np.array([101.,151.,99.,200.]);lab.geom_minor=np.array([20.,20.,20.,20.])
        lab.label_classes=np.array([1,2,1,3]);lab.geom_x=np.array([1.2,20.,4.,6.]);lab.geom_y=np.array([2.8,20.,4.,6.])
        lab.source_ids=np.arange(4)
        a=large_source_specs(lab,'abell');c=large_source_specs(lab,'cosmos')
        self.assertEqual(len(a),2);self.assertEqual(len(c),1)
        self.assertEqual((a[0].x0,a[0].y0),(-255,-253))
        arr=crop_padded(np.ones((32,32)),a[0].x0,a[0].y0,(0,0),512,fill=np.nan)
        self.assertTrue(np.isnan(arr[0,0]));self.assertEqual(arr[253,255],1)

    def test_parent_scaling_reused_exactly(self):
        image=np.random.default_rng(5).normal(size=(64,64)).astype(np.float32)
        image[12:16,20:25]+=20
        p=fit_parent_rgb(image)
        expected=scale_training_image(image,mode='zscore-log-lupton-rgb',clip_threshold=5,log_a=1000)
        got=apply_parent_rgb(image,p)
        np.testing.assert_allclose(got,expected,atol=2e-6)
        np.testing.assert_allclose(apply_parent_rgb(image[8:24,10:26],p),got[:,8:24,10:26],atol=2e-6)

    def test_written_validity_and_zero_loss(self):
        lab=labels();image=np.ones((32,32),np.float32);valid=np.ones((32,32),bool)
        valid[:16,:16]=False;valid[20,20]=False
        with tempfile.TemporaryDirectory() as tmp,patch('preprocessing.build_image_level_zarr._scale_image_chw',side_effect=lambda a,t:a[None]):
            configured=replace(task(tmp),image_lupton_q=20.,image_lupton_stretch=.5)
            r=write_classified_patch(configured,image,lab,valid_mask=valid,max_invalid_fraction=.1)
            self.assertEqual(r['samples'],3)
            g=zarr.open_group(r['output'],mode='r')
            self.assertEqual(g.attrs['image_scaling_version'],'zscore_first_v1')
            self.assertEqual(g.attrs['image_scaling_contract']['lupton_q'],20.)
            self.assertEqual(g.attrs['image_scaling_contract']['lupton_stretch'],.5)
            v=g['band_valid_mask'][:].astype(bool)
            self.assertEqual(np.count_nonzero(~v),1)
            self.assertTrue(np.all(g['band_conf_weight'][:][~v]==0))
            self.assertTrue(np.all(g['band_shape_weight'][:][~v]==0))
            self.assertTrue(np.all(g['images'][:][:,:,0][~v]==0))
            r=write_classified_patch(task(tmp),image,lab,valid_mask=np.zeros_like(valid),max_invalid_fraction=.1)
            self.assertEqual(r['status'],'no_valid_tiles')

    def test_split_parent_large_and_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name,ix,n,kind in [('p0',0,10,'grid'),('p0_id7',0,1,'large_source'),('p1',1,4,'grid'),('p3',3,4,'grid')]:
                p=root/'abell/image_level/half_coadd/F444W'/f'{name}__half.zarr';p.mkdir(parents=True)
                p.with_name(p.name+'_manifest.json').write_text('{}')
                (p/'.zattrs').write_text(json.dumps(dict(format='cellect_direct_patch_zarr',image_level_training=True,
                    dataset='abell',tract='Abell2744',dataset_source='half_coadd',bands=['F444W'],patch=f'p{ix}',parent_id=f'p{ix}',
                    group='half',num_samples=n,sample_kind=kind,parent_bounds=[ix*3968,0,ix*3968+4096,4096])))
            stores=scan_stores(root,['half_coadd'],[])
            candidates=build_candidates(stores,requested_bands=[],narrow_bands=[],narrow_weight_floor=.25,
                narrow_weight_power=1,narrow_weight_mode='sum-tiles',disable_narrow_downweight=False,min_bands=1,
                require_all_bands=False,include_patches=set(),exclude_patches=set())
            self.assertEqual(len(candidates),3)
            c=next(c for c in candidates if c.patch=='Abell2744/p0');self.assertEqual(c.sample_count,11)
            blocked=validation_exclusions(candidates,[c])
            self.assertEqual(blocked,{'Abell2744/p0','Abell2744/p1'})

    def test_no_legacy_dependency(self):
        self.assertNotIn('astro_data_preprocessing',sys.modules)

if __name__=='__main__':unittest.main()
