"""Unit/grid compatibility and source-aware finite repair."""

import unittest

import numpy as np

from preprocessing.image_processing import (
    BrightRegionConfig, ImagePreparationConfig, ImageProcessingConfig,
    build_bright_components, prepare_image, scale_image_for_training,
)
from data_filtering.sam_input_scaling import scale_training_image
from preprocessing.utils.image import hsc_surface_brightness_factor


class ImagePreparationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.raw = rng.normal(0, 0.01, (128, 128)).astype(np.float32)
        self.raw[55:73, 55:73] += 10

    def test_surface_brightness_units(self):
        expected = 1e6 * 0.168**2 / 206265**2 / 3631 * 10**10.8
        factor, _ = hsc_surface_brightness_factor({"BUNIT": "MJy/sr"})
        self.assertAlmostEqual(factor, expected)
        milli, _ = hsc_surface_brightness_factor(input_unit="mJy/sr")
        self.assertAlmostEqual(factor / milli / 1e9, 1)
        for scale in (0.03, 0.063, 0.168):
            jy = 1e6 * scale**2 / 206265**2
            for unit, value in (("Jy", jy), ("Jy/pixel", jy), ("nJy/pixel", jy * 1e9)):
                f, _ = hsc_surface_brightness_factor(input_unit=unit, pixel_scale_arcsec=scale)
                self.assertAlmostEqual(value * f, expected)
            f, _ = hsc_surface_brightness_factor(input_unit="zp-flux", input_zeropoint=25,
                                                 pixel_scale_arcsec=scale)
            flux = jy / 3631 * 10**10
            self.assertAlmostEqual(flux * f, expected)

    def test_invalid_units_fail_explicitly(self):
        for kwargs in ({}, {"input_unit": "electron/s"}, {"input_unit": "Jy"},
                       {"input_unit": "zp-flux", "pixel_scale_arcsec": 0.03},
                       {"input_unit": "Jy", "pixel_scale_arcsec": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                hsc_surface_brightness_factor(**kwargs)

    def test_fill_preserves_observed_pixels_and_grid(self):
        raw = self.raw.copy()
        raw[63:65, 63:65] = np.nan
        raw[10:12, 10:12] = np.inf
        prepared = prepare_image(raw, config=ImagePreparationConfig(input_unit="native"))
        self.assertEqual(prepared.image.shape, raw.shape)
        self.assertTrue(np.isfinite(prepared.image).all())
        np.testing.assert_array_equal(prepared.image[prepared.finite_mask], raw[np.isfinite(raw)])
        self.assertEqual(prepared.metadata["nan_pixels_nearest_fill"], 4)
        self.assertEqual(prepared.metadata["nan_pixels_background_fill"], 4)
        self.assertGreater(prepared.image[63, 63], 9)
        with self.assertRaises(ValueError):
            prepare_image(prepared)

    def test_filled_core_is_bright(self):
        raw = self.raw.copy()
        raw[63:65, 63:65] = np.nan
        prepared = prepare_image(raw, config=ImagePreparationConfig(input_unit="native"))
        bright, labels = build_bright_components(prepared, config=BrightRegionConfig(
            threshold=5, clip_threshold=5, statistics_clip_sigma=5))
        self.assertTrue(bright[63, 63])
        self.assertEqual(labels[63, 63], labels[60, 60])
        self.assertFalse(prepared.finite_mask[63, 63])

    def test_all_invalid_has_no_bright_components(self):
        prepared = prepare_image(np.full((16, 16), np.nan),
                                 config=ImagePreparationConfig(input_unit="native"))
        self.assertTrue(np.isfinite(prepared.image).all())
        for mode in ("log-lupton", "anscombe", "zscore"):
            bright, labels = build_bright_components(prepared, config=BrightRegionConfig(mode=mode))
            self.assertFalse(bright.any())
            self.assertFalse(labels.any())

    def test_hsc_legacy_scaling_and_convenience_api(self):
        for mode in ("zscore-log-lupton-rgb", "anscombe-rgb"):
            config = ImageProcessingConfig(scaling_mode=mode)
            actual = scale_image_for_training(self.raw, config=config)
            expected = scale_training_image(self.raw, mode=mode, anscombe_scale=1)
            np.testing.assert_array_equal(actual, expected)
            preparation = ImagePreparationConfig(input_unit="hsc-zp27")
            explicit = scale_image_for_training(prepare_image(self.raw, config=preparation), config=config)
            implicit = scale_image_for_training(self.raw, config=config, preparation=preparation)
            np.testing.assert_array_equal(actual, explicit)
            np.testing.assert_array_equal(actual, implicit)

    def test_reference_configuration_clips(self):
        prepared = prepare_image(self.raw, header={"BUNIT": "MJy/sr"})
        for mode in ("zscore-log-lupton-rgb", "anscombe-rgb"):
            result = scale_image_for_training(prepared, config=ImageProcessingConfig(
                scaling_mode=mode, clip_threshold=5, statistics_clip_sigma=5,
                log_a=1000, anscombe_scale=1000))
            self.assertEqual(result.shape, (3, 128, 128))
            self.assertTrue(np.isfinite(result).all())
            self.assertLessEqual(result.max(), 5)
            self.assertGreaterEqual(result.min(), -5)


if __name__ == "__main__":
    unittest.main()
