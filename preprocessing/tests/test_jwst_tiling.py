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
            r=write_classified_patch(task(tmp),image,lab,valid_mask=valid,max_invalid_fraction=.1)
            self.assertEqual(r['samples'],3)
            g=zarr.open_group(r['output'],mode='r')
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
