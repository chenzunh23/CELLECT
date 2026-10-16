"""Area weighting survives independent views, packing, augmentation and losses."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from preprocessing.tests.test_segmentation_zarr import labels, task
from preprocessing.build_image_level_zarr import write_classified_patch
from preprocessing.labels import DenseLabel
from preprocessing.utils.image_level import _tile_targets
from preprocessing.utils.inputs import TileSpec
from preprocessing.utils.parent_zarr import merge_parent_stores
from preprocessing.utils.source_weights import kron_area_weights
from astro_train_data import CutoutRecord, collate_cutouts
from astro_train_ops import LossWeights, dense_losses_any, hard_ce_confidence_loss, source_center_shape_loss
from astro_train_zarr_data import PatchZarrReader, ZarrCutoutDataset


def weighted_labels():
    lab = labels()
    lab.dense = np.full((96, 96), int(DenseLabel.BACKGROUND), np.uint8)
    lab.source_ids = np.arange(10, 16, dtype=np.int64)
    # Small clean, medium weak, large strict, ignored, dropped, large clean.
    lab.label_classes = np.array([1, 2, 4, 3, 0, 1])
    lab.geom_x = np.array([8., 24., 40., 56., 72., 88.])
    lab.geom_y = np.full(6, 24.)
    lab.geom_major = lab.geom_minor = np.array([10., 35., 60., 60., 60., 60.])
    lab.geom_theta = np.zeros(6)
    lab.strict_x, lab.strict_y, lab.strict_ids = np.array([24.]), np.array([72.]), np.array([100])
    return lab


class SourceAreaWeightTests(unittest.TestCase):
    def test_boundaries_invalid_and_unknown(self):
        # a=1/pi makes pi*a*b exactly b for the boundary checks.
        b = np.array([0., 2499., 2500., 2501., 10000., 10001., np.nan, np.inf, -20.])
        np.testing.assert_array_equal(kron_area_weights(np.full(len(b), 1/np.pi), b),
                                      [1, 1, 1, 2, 2, 4, 1, 1, 1])

    def test_positive_only_all_confidence_modes(self):
        lab = weighted_labels()
        spec = SimpleNamespace(x0=0, y0=0, size=96)
        for config in [None, dict(mode='psf-ee', level_radii_pixels={'4': .5, '3': 1., '2': 2., '1': 3.}),
                       dict(mode='psf-matched', fwhm_pixels_used=2., fwhm_clip_pixels=[1.6, 8.])]:
            target = _tile_targets(lab, spec, (0, 0), confidence=config)
            np.testing.assert_array_equal(target['conf_weight'][24, lab.geom_x.astype(int)], [1,2,4,1,1,4])
            self.assertEqual(target['confidence'][24, 56], 0)  # ignore
            self.assertEqual(target['confidence'][24, 72], 0)  # drop
            self.assertEqual(target['conf_weight'][72,24], 1)  # external
            self.assertEqual(target['conf_weight'][0,0], 1)  # sky
            np.testing.assert_array_equal(target['shape_ids'], [10,11,15])
            np.testing.assert_array_equal(target['shape_weights'], [1,2,4])
            # Weight changes neither EE levels nor dense labels nor GT axes.
            with patch('preprocessing.utils.source_weights.kron_area_weights', return_value=np.ones(6)):
                baseline = _tile_targets(lab, spec, (0,0), confidence=config)
            np.testing.assert_array_equal(target['confidence'], baseline['confidence'])
            np.testing.assert_array_equal(target['pu'], baseline['pu'])
            np.testing.assert_array_equal(target['shape_values'], baseline['shape_values'])
            negative = target['confidence'] == 0
            np.testing.assert_array_equal(target['conf_weight'][negative], baseline['conf_weight'][negative])

    def test_overlap_max_is_order_independent(self):
        from preprocessing.utils.confidence import paint_ee_confidence
        config = dict(level_radii_pixels={'4': .6, '3': 1., '2': 2., '1': 3.})
        outputs=[]
        for centers, weights in [([[5.,5.],[6.,5.]],[1.,4.]), ([[6.,5.],[5.,5.]],[4.,1.])]:
            c=np.zeros((12,12), np.uint8);w=np.ones((12,12), np.float32)
            paint_ee_confidence(c,w,centers,value_weight=weights,**config);outputs.append((c,w))
        for a,b in zip(*outputs):np.testing.assert_array_equal(a,b)
        self.assertEqual(outputs[0][1][5,5],4)
        self.assertEqual(outputs[0][1][0,0],1)

    def test_write_pack_load_flip_and_per_band_losses(self):
        lab=weighted_labels()
        # Invalid source center must remove its sparse weight too.
        valid=np.ones((96,96),bool);valid[24,24]=False
        with tempfile.TemporaryDirectory() as td, patch('preprocessing.build_image_level_zarr._scale_image_chw',
                                                      side_effect=lambda image,task:image[None]):
            root=Path(td);t=replace(task(root/'parts'),tile_size=96,stride=96,chunk_tiles=20)
            a=write_classified_patch(replace(t,patch='a'), np.ones((96,96)), lab,
                tile_specs=[TileSpec('grid',0,0,96)],valid_mask=valid)
            b=write_classified_patch(replace(t,patch='b'), np.ones((96,96)), lab,
                tile_specs=[TileSpec('centered',0,0,96,kind='large_source')], valid_mask=valid)
            out=root/'parent.zarr'
            merge_parent_stores([(a['output'],(0,0)),(b['output'],(0,0))],out,attrs={'patch':'parent'},chunk_tiles=20)
            reader=PatchZarrReader(out)
            np.testing.assert_array_equal(reader.read_full_small('shape_source_weights'), [1,4,1,4])
            np.testing.assert_array_equal(reader.read_full_small('shape_source_offsets'), [[0,2],[2,4]])
            self.assertEqual(reader.attrs['source_area_weight_policy']['thresholds_exclusive'], [2500,10000])
            records=[CutoutRecord(name=str(i),image_paths=(f'zarr://{out}#{i}',),meas_path='',x0=0,y0=0) for i in range(2)]
            dataset=ZarrCutoutDataset(records,augment=True)
            with patch('astro_train_zarr_data.random.random',return_value=0):item=dataset[1]
            np.testing.assert_array_equal(item['shape_source_weights'],[1,4])
            np.testing.assert_allclose(item['shape_source_centers'][:,0],[87,7])
            self.assertEqual(item['band_confidence_weight'][0,24,55],4)  # flipped strict center
            self.assertEqual(item['band_confidence_weight'][0,24,71],0)  # invalid center
            batch=collate_cutouts([item,item])
            shape=torch.ones((2,1,3,96,96),requires_grad=True)
            conf=torch.zeros((2,1,5,96,96),requires_grad=True)
            losses=dense_losses_any(dict(confidence=conf,shape=shape),batch,
                weights=LossWeights(segmentation_outer_weight=0,confidence_loss_mode='ce_hard',
                                    center_position=0,shape_geometry_loss='log_spd'),
                device=torch.device('cpu'),center_radius_px=2)
            direct=source_center_shape_loss(shape[:,0],batch['shape_source_centers'],batch['shape_source_values'],
                batch['shape_source_classes'],source_weights_list=batch['shape_source_weights'],
                center_size=3,center_only_factor=.2,angle_weight=4,geometry_loss='log_spd')
            torch.testing.assert_close(losses['shape'],direct)
            ce=hard_ce_confidence_loss(conf[:,0],batch['band_confidence'][:,0],class_weights=(1,4,8,16,32),
                                       weight=batch['band_confidence_weight'][:,0])
            torch.testing.assert_close(losses['confidence'],ce)
            losses['total'].backward()
            self.assertTrue(torch.isfinite(shape.grad).all());self.assertTrue(torch.isfinite(conf.grad).all())
            # A legacy store with no new metadata retains historical unit weights.
            shutil.rmtree(out/'shape_source_weights')
            from astro_train_zarr_data import _reader
            _reader.cache_clear()
            old=ZarrCutoutDataset(records)[0]
            np.testing.assert_array_equal(old['shape_source_weights'],[1,1])

    def test_normalization_reliability_and_gradient(self):
        centers=[torch.tensor([[2.,2.],[6.,2.],[10.,2.]])]
        values=[torch.tensor([[3.,2.,0.],[5.,3.,.2],[6.,4.,.1]])]
        pred=torch.ones((1,3,5,13),requires_grad=True)
        kwargs=dict(center_size=1,center_only_factor=.2,angle_weight=4,geometry_loss='log_spd')
        each=torch.stack([source_center_shape_loss(pred,[centers[0][i:i+1]],[values[0][i:i+1]],
                         [torch.ones(1)],**kwargs) for i in range(3)])
        actual=source_center_shape_loss(pred,centers,values,[torch.tensor([1,2,1])],
                                       source_weights_list=[torch.tensor([1.,2.,4.])],**kwargs)
        expected=(each*torch.tensor([1.,.4,4.])).sum()/5.4
        torch.testing.assert_close(actual,expected)
        actual.backward();self.assertTrue(torch.isfinite(pred.grad).all())
        legacy=source_center_shape_loss(pred,centers,values,[torch.tensor([1,2,1])],**kwargs)
        torch.testing.assert_close(legacy,(each*torch.tensor([1.,.2,1.])).sum()/2.2)


if __name__=='__main__':unittest.main()
