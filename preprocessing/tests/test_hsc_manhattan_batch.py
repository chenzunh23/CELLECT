"""HSC batch generation must honor Manhattan mode and explicit legacy SNR cuts."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import zarr

from preprocessing.build_image_level_zarr import parse_args, write_classified_patch
from preprocessing.training_batch import task_for
from preprocessing.snr import SnrThresholds, thresholds_from_args, snr_policy_from_args
from preprocessing.utils.confidence import resolve_confidence
from preprocessing.utils.image_level import _paint_confidence
from preprocessing.tests.test_segmentation_zarr import task as fixture_task, labels


def args(root, mode='manhattan'):
    return parse_args(['--training-batch', '--training-kinds', 'hsc_half', 'hsc_noisy',
                       '--output-root', str(root), '--confidence-mode', mode,
                       '--snr-ignore-threshold', '3', '--snr-weak-threshold', '5',
                       '--hsc-snr-ignore-threshold', '3', '--hsc-snr-weak-threshold', '5',
                       '--narrow-snr-ignore-threshold', '3', '--narrow-snr-weak-threshold', '5'])


class HscManhattanBatchTests(unittest.TestCase):
    def test_task_dispatch_and_shared_snr_threshold_type(self):
        a = args('/tmp/hsc-manhattan-test')
        self.assertEqual(thresholds_from_args(a, 'hsc'), SnrThresholds(3, 5))
        self.assertEqual(snr_policy_from_args(a, 'hsc'),
                         dict(ignore_below=3., weak_below=5., narrow_ignore_below=3., narrow_weak_below=5.))
        for band in ['HSC-I', 'NB0816']:
            for source in ['half_coadd', 'noisy']:
                t = task_for(a, band=band, dataset_source=source)
                self.assertEqual(t.confidence_mode, 'manhattan')
                self.assertEqual(resolve_confidence(t)['mode'], 'manhattan')
                self.assertEqual((t.snr_ignore_threshold,t.snr_weak_threshold), (3.,5.))
                self.assertEqual((t.narrow_snr_ignore_threshold,t.narrow_snr_weak_threshold), (3.,5.))
        # Explicit modes now apply to JWST too; normal JWST defaults stay EE.
        self.assertEqual(task_for(a, band='F444W').confidence_mode, 'manhattan')
        self.assertEqual(task_for(args('/tmp/test', 'psf-ee'),band='F444W').confidence_mode, 'psf-ee')
        self.assertEqual(task_for(args('/tmp/test', 'psf-ee'),band='HSC-I').confidence_mode, 'psf-ee')

    def test_written_hsc_targets_use_original_painter_and_record_mode(self):
        lab = labels()
        lab.geom_x = np.array([8.2, 24.1])
        lab.geom_y = np.array([8.3, 24.2])
        lab.label_classes = np.array([1, 1])
        lab.geom_major = np.array([2., 2.])
        lab.geom_minor = np.array([2., 2.])
        lab.geom_theta = np.zeros(2)
        lab.source_ids = np.array([101, 102], dtype=np.int64)
        raw = np.ones((32, 32), np.float32)
        with tempfile.TemporaryDirectory() as tmp, patch(
                'preprocessing.build_image_level_zarr._scale_image_chw',
                side_effect=lambda image,t: np.repeat(image[None],3,axis=0)):
            t = replace(fixture_task(tmp), band='HSC-I', confidence_mode=task_for(args(tmp),band='HSC-I').confidence_mode)
            result = write_classified_patch(t, raw, lab)
            group = zarr.open_group(result['output'], mode='r')
            self.assertEqual(group.attrs['confidence_config']['mode'], 'manhattan')
            expected = np.zeros((16,16), np.uint8)
            _paint_confidence(expected, np.zeros_like(expected, dtype=np.float32), np.array([[8.2,8.3]],np.float32))
            np.testing.assert_array_equal(group['band_confidence'][0,0], expected)


if __name__ == '__main__':
    unittest.main()
