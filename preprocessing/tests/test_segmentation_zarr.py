"""Positive-only segmentation survives patch tiling and Zarr serialization."""
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import zarr
from astropy.table import Table

from preprocessing.build_image_level_zarr import write_classified_patch
from preprocessing.utils.image_level import PatchLabels, StoreTask, _tile_targets
from preprocessing.zarr_writing import ImageLevelTrainingBatch, write_training_image_level_zarr


def labels():
    empty = np.empty(0)
    return PatchLabels(Table(), np.zeros((32, 32), np.uint8), empty.astype(int),
        empty, empty, empty, empty, empty, empty.astype(np.int64),
        empty, empty, empty.astype(np.int64))


def task(root):
    values = {f.name: 1 for f in fields(StoreTask)}
    values.update(output_root=Path(root), band='f444w', patch='19', tract='default',
        dataset_source='coadd', group='', tile_size=16, stride=16, max_tiles=0,
        overwrite=True, chunk_tiles=1, image_scaling_scope='patch',
        image_scaling_mode='log-lupton', bright_mask_mode='log-lupton',
        coadd_lsst_background_root=None, variant_lsst_background_root=None,
        bright_object_mask_root=None, confidence_mode='manhattan', confidence_config_path=None,
        confidence_fwhm_min=1.6, confidence_fwhm_max=8., confidence_fwhm_pixels=None)
    return StoreTask(**values)


class SegmentationZarrTests(unittest.TestCase):
    def test_tiling_preserves_global_ids_and_origin(self):
        lab = labels()
        lab.segmentation_ids = np.zeros((32,32), np.int32)
        lab.segmentation_ids[14:18,14:18] = 684029
        lab.segmentation_weight = (lab.segmentation_ids > 0).astype(np.float32)*.25
        spec = SimpleNamespace(x0=116, y0=216, size=16)
        result = _tile_targets(lab, spec, (100,200))
        np.testing.assert_array_equal(result['segmentation_ids'], lab.segmentation_ids[16:,16:])
        self.assertEqual(result['segmentation_ids'][0,0], 684029)
        self.assertEqual(result['segmentation_weight'][0,0], .25)

    def test_cosmos_attach_tile_write_read(self):
        lab = labels()
        lab.source_ids = np.array([684029])
        lab.label_classes = np.array([1])
        lab.geom_x = lab.geom_y = np.array([15.])
        lab.geom_major = lab.geom_minor = np.array([3.])
        lab.geom_theta = np.array([0.])
        seg = np.zeros((32,32), np.int32); seg[14:18,14:18] = 684029
        with tempfile.TemporaryDirectory() as tmp, \
             patch('preprocessing.utils.segmentation.read_cosmos_segment_cutout',
                   return_value=(seg, np.zeros_like(seg,bool), np.ones_like(seg,bool))), \
             patch('preprocessing.build_image_level_zarr._scale_image_chw',
                   side_effect=lambda image, task: image[None]):
            result = write_classified_patch(task(tmp), np.ones((32,32)), lab,
                cosmos_catalog='catalog.fits', image_wcs=object())
            group = zarr.open_group(result['output'], mode='r')
            ids = group['band_segmentation_ids'][:]
            weights = group['band_segmentation_weight'][:]
            self.assertEqual(ids.shape, (4,1,16,16))
            self.assertEqual(ids.dtype, np.dtype('int32'))
            self.assertEqual(weights.dtype, np.dtype('float32'))
            self.assertEqual(np.count_nonzero(ids), 16)
            self.assertEqual(set(np.unique(ids)), {0,684029})
            self.assertTrue(np.all(weights[ids>0] == .25))
            self.assertTrue(np.all(weights[ids==0] == 0))
            self.assertEqual(group.attrs['segmentation_supervision'], 'positive_only')
            self.assertTrue(group.attrs['segmentation_policy']['allow_nested'])
            self.assertEqual(group.attrs['segmentation_policy']['version'], 'raw_nested_fill_gaussian_overlap_v3')

    def test_default_regularization_written_before_tiling(self):
        lab = labels()
        lab.source_ids = np.array([100,200])
        lab.label_classes = np.array([1,1])
        lab.geom_x = lab.geom_y = np.array([14.,25.])
        lab.geom_major = lab.geom_minor = np.array([3.,1.])
        lab.geom_theta = np.zeros(2)
        seg = np.zeros((32,32), np.int32)
        seg[9:20,9:20] = 100; seg[12,12] = 0  # 120 raw pixels: smooth and fill.
        seg[24:27,24:27] = 200; seg[25,25] = 0  # 8 raw pixels: fill only.
        with tempfile.TemporaryDirectory() as tmp, \
             patch('preprocessing.utils.segmentation.read_cosmos_segment_cutout',
                   return_value=(seg,np.zeros_like(seg,bool),np.ones_like(seg,bool))), \
             patch('preprocessing.build_image_level_zarr._scale_image_chw',
                   side_effect=lambda image, task: image[None]):
            result = write_classified_patch(task(tmp),np.ones((32,32)),lab,
                cosmos_catalog='catalog.fits',image_wcs=object())
            group=zarr.open_group(result['output'],mode='r')
            self.assertEqual(lab.segmentation_ids[12,12],100)  # Hole filled.
            self.assertEqual(lab.segmentation_ids[9,9],0)  # Gaussian corner removed.
            self.assertEqual(lab.segmentation_ids[24,24],200)  # Small source not blurred.
            self.assertEqual(lab.segmentation_ids[25,25],200)
            stored=group['band_segmentation_ids'][:,0]
            assembled=np.block([[stored[0],stored[1]],[stored[2],stored[3]]])
            np.testing.assert_array_equal(assembled,lab.segmentation_ids)
            policy=group.attrs['segmentation_policy']
            self.assertEqual(policy['clearance_pixels'],0)
            self.assertEqual(policy['gaussian_sigma_pixels'],1)
            self.assertEqual(policy['gaussian_min_raw_area'],100)

    def test_explicit_valid_mask_used_before_segmentation_selection(self):
        lab=labels();lab.source_ids=np.array([100]);lab.label_classes=np.array([1])
        lab.geom_x=lab.geom_y=np.array([15.]);lab.geom_major=lab.geom_minor=np.array([3.])
        lab.geom_theta=np.array([0.])
        seg=np.zeros((32,32),np.int32);seg[14:18,14:18]=100
        valid=np.ones_like(seg,bool);valid[15,15]=False
        with tempfile.TemporaryDirectory() as tmp, \
             patch('preprocessing.utils.segmentation.read_cosmos_segment_cutout',
                   return_value=(seg,np.zeros_like(seg,bool),np.ones_like(seg,bool))), \
             patch('preprocessing.build_image_level_zarr._scale_image_chw',
                   side_effect=lambda image, task: image[None]):
            write_classified_patch(task(tmp),np.ones((32,32)),lab,valid_mask=valid,
                cosmos_catalog='catalog.fits',image_wcs=object())
            self.assertFalse(lab.segmentation_ids.any())

    def test_overlapping_parent_round_trip_and_validity(self):
        from preprocessing.utils.segmentation import isolated_segmentation_targets
        from preprocessing.utils.segmentation_storage import iter_training_segmentation_masks
        from preprocessing.labels import SourceLabels
        lab=labels();seg=np.zeros((32,32),np.int32)
        seg[8:24,8:24]=100;seg[14:18,14:18]=200
        out=isolated_segmentation_targets(seg,[100,200],
            SourceLabels(np.array([1,1]),np.array(['test','test'])),clearance=2)
        lab.segmentation_ids=out.instance_ids;lab.segmentation_weight=out.positive_weight
        lab.segmentation_overlap_masks=out.overlap_masks
        valid=np.ones((32,32),bool);valid[14,14]=False
        with tempfile.TemporaryDirectory() as tmp, patch(
                'preprocessing.build_image_level_zarr._scale_image_chw',
                side_effect=lambda image, task: image[None]):
            result=write_classified_patch(task(tmp),np.ones((32,32)),lab,valid_mask=valid)
            group=zarr.open_group(result['output'],mode='r')
            meta=group['segmentation_overlap_meta'][:]
            self.assertEqual(meta.shape,(4,8))
            self.assertTrue(np.all(meta[:,2]==100))  # No per-instance storage for child.
            self.assertEqual(group['segmentation_overlap_data'].size,32)  # Four 8x8 bboxes.
            for tile in range(4):
                masks={r['source_id']:r for r in iter_training_segmentation_masks(group,tile)}
                self.assertEqual(set(masks),{100,200})
                self.assertTrue(np.all(masks[100]['mask'][masks[200]['mask']]))
            masks={r['source_id']:r for r in iter_training_segmentation_masks(group,0)}
            self.assertFalse(masks[100]['mask'][14,14])
            self.assertFalse(masks[200]['mask'][14,14])
            self.assertEqual(np.count_nonzero(masks[100]['mask']),62)
            self.assertEqual(np.count_nonzero(masks[200]['mask']),3)

    def test_no_segmentation_keeps_legacy_arrays(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch('preprocessing.build_image_level_zarr._scale_image_chw',
                   side_effect=lambda image, task: image[None]):
            result = write_classified_patch(task(tmp), np.ones((32,32)), labels())
            group = zarr.open_group(result['output'], mode='r')
            self.assertNotIn('band_segmentation_ids', group)
            self.assertNotIn('band_segmentation_weight', group)
            self.assertIn('band_confidence', group)
            self.assertEqual(group.attrs['segmentation_supervision'], 'none')

    def test_invalid_segmentation_rejected_before_overwrite(self):
        # Validation occurs before the writer creates or replaces a store.
        values = {f.name: None for f in fields(ImageLevelTrainingBatch)}
        values.update(images=np.zeros((1,1,4,4)), band_segmentation_ids=np.zeros((1,1,4,4),np.int32))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'store';root.mkdir();marker=root/'keep';marker.touch()
            for ids, weight in [(values['band_segmentation_ids'], None),
                                (np.zeros((1,1,4,4),np.int32), np.ones((1,1,4,4))),
                                (np.full((1,1,4,4),2**32,np.int64), np.zeros((1,1,4,4))),
                                (np.zeros((1,1,4,4),np.int32), np.full((1,1,4,4),np.nan)),
                                (np.zeros((4,4),np.int32), np.zeros((4,4)))]:
                values.update(band_segmentation_ids=ids,band_segmentation_weight=weight)
                with self.assertRaises(ValueError):
                    write_training_image_level_zarr(root, ImageLevelTrainingBatch(**values), overwrite=True)
                self.assertTrue(marker.exists())


if __name__ == '__main__':
    unittest.main()
