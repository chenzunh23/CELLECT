"""Precompute once; keep eligibility aligned through packing/loading/flips."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from astropy.table import Table

from preprocessing.tests.test_segmentation_zarr import labels,task
from preprocessing.build_image_level_zarr import write_classified_patch
from preprocessing.utils.inputs import TileSpec
from preprocessing.utils.parent_zarr import merge_parent_stores
from astro_train_data import CutoutRecord,collate_cutouts
from astro_train_zarr_data import PatchZarrReader,ZarrCutoutDataset


class MomentEligibilityStorageTests(unittest.TestCase):
    def test_jwst_high_snr_flags_and_policy_survive_pack_load_and_collate(self):
        from preprocessing.utils.mask_moments import JWST_POLICY
        lab=labels();lab.dense=np.zeros((64,64),np.uint8)
        lab.table=Table(dict(classification_snr=[np.nan,9.,10.,np.nan]))
        lab.table.meta['mask_moment_weak_snr_threshold']=7.
        lab.source_ids=np.arange(101,105,dtype=np.int64)
        lab.label_classes=np.array([1,2,2,2]);lab.geom_x=np.array([8.,24.,40.,56.]);lab.geom_y=np.full(4,24.)
        lab.geom_major=np.full(4,4.);lab.geom_minor=np.full(4,2.);lab.geom_theta=np.full(4,.3)
        with tempfile.TemporaryDirectory() as td,patch('preprocessing.build_image_level_zarr._scale_image_chw',
                side_effect=lambda image,task:image[None]):
            root=Path(td);t=replace(task(root/'parts'),band='F444W',tile_size=64,stride=64)
            parts=[]
            for name in ['a','b']:
                result=write_classified_patch(replace(t,patch=name),np.ones((64,64)),lab,
                    tile_specs=[TileSpec(name,0,0,64)],
                    provenance=dict(dataset='abell',snr_thresholds=dict(weak_below=10.)))
                parts.append((Path(result['output']),(0,0)))
            path=root/'parent.zarr';merge_parent_stores(parts,path,attrs={},chunk_tiles=20)
            reader=PatchZarrReader(path)
            self.assertEqual(reader.attrs['mask_moment_policy'],JWST_POLICY)
            self.assertEqual(reader.attrs['mask_moment_weak_snr_threshold'],10.)
            self.assertEqual(reader.attrs['mask_moment_snr_source'],'classification_snr')
            np.testing.assert_array_equal(reader.read_full_small('shape_source_moment_eligible'),
                                           [1,0,1,0,1,0,1,0])
            rec=CutoutRecord(name='test',image_paths=(f'zarr://{path}#1',),meas_path='',x0=0,y0=0)
            item=ZarrCutoutDataset([rec],augment=False)[0]
            self.assertEqual(item['proposal_metadata']['mask_moment_policy'],JWST_POLICY)
            b=collate_cutouts([item])
            self.assertEqual(b['proposal_metadata'][0]['mask_moment_policy'],JWST_POLICY)
            torch.testing.assert_close(b['band_shape_source_moment_eligible'][0][0],
                                       torch.tensor([True,False,True,False]))

    def test_write_pack_load_flip_and_collate(self):
        lab=labels();lab.dense=np.zeros((64,64),np.uint8)
        lab.table=Table(dict(classification_reason=['ordinary_snr_clean','ordinary_snr_weak_shape',
                           'isolated_bright_galaxy_weak_shape','high_blendedness_requires_gaia']))
        lab.source_ids=np.arange(43158185032286339,43158185032286343,dtype=np.int64)
        lab.label_classes=np.array([1,2,2,2]);lab.geom_x=np.array([8.,24.,40.,56.]);lab.geom_y=np.full(4,24.)
        lab.geom_major=np.full(4,4.);lab.geom_minor=np.full(4,2.);lab.geom_theta=np.full(4,.3)
        with tempfile.TemporaryDirectory() as td,patch('preprocessing.build_image_level_zarr._scale_image_chw',
                side_effect=lambda image,task:image[None]):
            root=Path(td);t=replace(task(root/'parts'),band='HSC-I',tile_size=64,stride=64)
            parts=[]
            for name in ['a','b']:
                result=write_classified_patch(replace(t,patch=name),np.ones((64,64)),lab,
                                               tile_specs=[TileSpec(name,0,0,64)])
                parts.append((Path(result['output']),(0,0)))
            path=root/'parent.zarr';merge_parent_stores(parts,path,attrs={},chunk_tiles=20)
            reader=PatchZarrReader(path)
            np.testing.assert_array_equal(reader.read_full_small('shape_source_moment_eligible'),
                                           [1,0,1,0,1,0,1,0])
            records=[CutoutRecord(name=str(i),image_paths=(f'zarr://{path}#{i}',),meas_path='',x0=0,y0=0) for i in range(2)]
            ds=ZarrCutoutDataset(records,augment=True)
            with patch('astro_train_zarr_data.random.random',return_value=0.):item=ds[1]
            np.testing.assert_array_equal(item['band_shape_source_moment_eligible'][0],[True,False,True,False])
            np.testing.assert_array_equal(item['shape_source_ids'],lab.source_ids)
            np.testing.assert_allclose(item['shape_source_values'][:,2],-.3)
            b=collate_cutouts([item,item]);torch.testing.assert_close(b['band_shape_source_moment_eligible'][1][0],torch.tensor([True,False,True,False]))


if __name__=='__main__':unittest.main()
