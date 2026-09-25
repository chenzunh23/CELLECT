"""NaN rejection precedes catalog cuts; separately inserted Gaia survives."""
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import zarr

from preprocessing.labels import SourceClass as C, SourceLabels
from preprocessing.meas_processing import classify_catalog_basics
from preprocessing.ordinary_common import classify_ordinary_sources
from preprocessing.ordinary_a2744 import A2744OrdinaryConfig
from preprocessing.tests.test_ordinary_datasets import sources
from preprocessing.tests.test_segmentation_zarr import task, labels
from preprocessing.utils.no_data import center_no_data
from preprocessing.build_image_level_zarr import write_classified_patch


class NaNCenterTests(unittest.TestCase):
    def test_first_filter_overrides_area_and_source_cuts(self):
        data = sources([20, 40, 60, 80], a=[10000, 2, 2, 2], mag=[20, 24, 24, 24])
        image = np.zeros((100, 100)); image[50, [20, 40]] = np.nan
        for dataset in ('a2744', 'cosmos'):
            result = classify_catalog_basics(data.geometry, data.mag, dataset=dataset,
                pixel_scale_arcsec=.03, image=image, source_mask=[True, False, True, True])
            np.testing.assert_array_equal(result.nan_center_ignore, [True, True, False, False])
            np.testing.assert_array_equal(result.after_a, [False, False, True, True])
            np.testing.assert_array_equal(result.labels.source_class[:2], [C.ORDINARY_IGNORE]*2)
            self.assertFalse(result.a_large[0])
            self.assertEqual(result.labels.reason[0], 'center_no_data')

    def test_prefill_validity_and_no_resurrection(self):
        data = sources([20, 40, 60])
        valid = np.ones((100, 100), bool); valid[50, 20] = False
        for dataset in ('a2744', 'cosmos'):
            initial = SourceLabels.empty(3)
            initial.assign([True, False, False], C.DROPPED, 'old_drop')
            kwargs = dict(config=A2744OrdinaryConfig(1, enable_aperture_snr=False)) if dataset=='a2744' else {}
            out = classify_ordinary_sources(data, [False, True, True], initial,
                dataset=dataset, image=np.zeros(valid.shape), valid_mask=valid, **kwargs)
            self.assertEqual(out.labels.source_class[0], C.ORDINARY_IGNORE)
            for stage in out.stages.values():
                self.assertEqual(stage[0], C.ORDINARY_IGNORE)
            np.testing.assert_array_equal(out.labels.source_class[1:], [C.CLEAN]*2)

    def test_nan_neighbor_cannot_demote_survivor(self):
        data = sources([40, 40.6], mag=[20, 24])
        image = np.zeros((100, 100)); image[50, 40] = np.nan
        out = classify_ordinary_sources(data, [True, True], SourceLabels.empty(2),
            dataset='a2744', image=image,
            config=A2744OrdinaryConfig(1, enable_aperture_snr=False))
        np.testing.assert_array_equal(out.labels.source_class, [C.ORDINARY_IGNORE, C.CLEAN])

    def test_pixel_rounding_origin_and_no_edge_clipping(self):
        geom = SimpleNamespace(x=np.array([10.49,10.5,9.4,13.5,np.nan]), y=np.full(5,20.))
        valid = np.ones((4,4), bool); valid[0,1] = False
        np.testing.assert_array_equal(center_no_data(geom, valid_mask=valid, origin=(10,20)),
                                      [False, True, True, True, True])

    def test_inserted_gaia_retained_but_invalid_pixel_has_no_loss(self):
        lab = labels()
        lab.strict_x = np.array([4., 6.]); lab.strict_y = np.array([4., 6.])
        lab.strict_ids = np.array([-123, -456]); lab.strict_is_gaia = np.array([True, False])
        image = np.ones((32,32), np.float32); valid = np.ones((32,32),bool)
        valid[4,4] = valid[6,6] = False
        with tempfile.TemporaryDirectory() as tmp, patch('preprocessing.build_image_level_zarr._scale_image_chw', side_effect=lambda a,t:a[None]):
            out = write_classified_patch(task(tmp), image, lab, valid_mask=valid, max_invalid_fraction=.1)
            store = zarr.open_group(out['output'], mode='r')
            self.assertIn(-123, store['strict_center_only_ids'][:])
            self.assertNotIn(-456, store['strict_center_only_ids'][:])
            v = store['band_valid_mask'][:].astype(bool)
            self.assertTrue(np.all(store['band_conf_weight'][:][~v] == 0))


if __name__ == '__main__':
    unittest.main()
