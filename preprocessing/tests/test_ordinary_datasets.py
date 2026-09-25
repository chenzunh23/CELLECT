"""Focused ordinary-rule boundaries and SNR candidate regression tests."""

import unittest
from unittest.mock import patch
import numpy as np
from astropy.table import Table
from preprocessing.labels import SourceClass as C, SourceLabels
from preprocessing.ordinary_common import OrdinaryInput, classify_ordinary_sources
from preprocessing.ordinary_a2744 import A2744OrdinaryConfig
from preprocessing.meas_processing import classify_catalog_basics
from preprocessing.utils.geometry import EllipseGeometry
from preprocessing.snr import apply_remeasured_aperture_snr, remeasure_aperture_snr


def sources(x, a=None, flags=None, mag=None, snr=None, model=None, star=None):
    x = np.asarray(x, float)
    n = len(x)
    a = np.full(n, 2.) if a is None else np.asarray(a, float)
    geom = EllipseGeometry(x, np.full(n, 50.), a, a.copy(), np.zeros(n), np.pi * a*a)
    return OrdinaryInput(geom, np.full(n, 24.) if mag is None else np.asarray(mag),
                         snr=np.full(n, 10.) if snr is None else np.asarray(snr),
                         flags=np.zeros(n, int) if flags is None else np.asarray(flags),
                         model_mag=np.full(n, 24.) if model is None else np.asarray(model),
                         star_mask=np.zeros(n, bool) if star is None else np.asarray(star))


def classify(data, dataset, **kwargs):
    return classify_ordinary_sources(data, np.ones(len(data.mag), bool), SourceLabels.empty(len(data.mag)),
                                     dataset=dataset, **kwargs)


class OrdinaryDatasetTests(unittest.TestCase):
    def test_area_containment_matches_hsc(self):
        from preprocessing.utils.containment import area_fraction, boundary_fraction
        from preprocessing.ordinary_hsc import _contained_fraction
        data=sources([2872.476674,2868.176356],a=[23.268784,14.57692],flags=[3,3])
        g=data.geometry;g.y[:]=[1224.471671,1230.175738];g.minor[:]=[11.00302,10.761042]
        g.theta[:]=np.deg2rad([-53.378918,-59.975822]);g.area[:]=np.pi*g.major*g.minor
        def row(i):
            return dict(x=g.x[i],y=g.y[i],major=g.major[i],minor=g.minor[i],theta_deg=np.degrees(g.theta[i]))
        self.assertLess(boundary_fraction(g,0,1),.8)
        self.assertGreater(area_fraction(g,0,1),.97)
        self.assertEqual(area_fraction(g,0,1),_contained_fraction(row(1),row(0)))
        out=classify(data,'a2744',config=A2744OrdinaryConfig(1))
        np.testing.assert_array_equal(out.labels.source_class,[4,2])

    def test_a2744_explicit_snr_skip(self):
        data=sources([30,70],snr=[1,2])
        config=A2744OrdinaryConfig(1,enable_catalog_snr=False,enable_aperture_snr=False)
        with patch('preprocessing.ordinary_a2744.remeasure_aperture_snr') as measure:
            out=classify(data,'a2744',config=config,image=np.ones((100,100)))
            measure.assert_not_called()
        np.testing.assert_array_equal(out.labels.source_class,[1,1])
        self.assertTrue(out.diagnostics['aperture_snr_skipped'])

    def test_a2744_aperture_all_retained(self):
        data = sources([30, 70, 110, 150], mag=[24, 25.5, 26, 26], snr=[10, 10, 10, 2])
        def measure(image, geom, **kwargs):
            np.testing.assert_array_equal(np.isfinite(geom.x), [True, True, True, False])
            self.assertEqual(kwargs['radius'], 10)
            return dict(snr=np.array([2., 4., 2., np.nan]), trusted=np.array([True, True, True, False]))
        with patch('preprocessing.ordinary_a2744.remeasure_aperture_snr', side_effect=measure):
            result = classify(data, 'a2744', config=A2744OrdinaryConfig(1), image=np.ones((200, 200)))
        np.testing.assert_array_equal(result.labels.source_class, [3, 4, 3, 3])

    def test_fractional_aperture_completeness(self):
        from utils.source_snr import measure_local_aperture_snr, disk_kernel
        image = np.ones((128, 128), dtype=np.float32)
        x = np.array([64.948718211206, 64.0, 64.948718211206, 4.5])
        y = np.array([64.40723194147, 64.1, 64.40723194147, 64.5])
        expected = int(disk_kernel(16).sum())
        background = (np.zeros_like(image), np.ones_like(image), np.full(image.shape, 100))
        noise = (np.ones_like(image), np.full(image.shape, 100), np.zeros_like(image), expected)
        with patch('utils.source_snr.block_background', return_value=background), \
             patch('utils.source_snr.aperture_noise_by_block', return_value=noise):
            result = measure_local_aperture_snr(image, x, y, sky_mask=np.ones_like(image, bool))
            self.assertGreater(result['aperture_pixels'][0], expected)
            self.assertLess(result['aperture_pixels'][1], expected)
            np.testing.assert_array_equal(result['trusted'], [True, True, True, False])
            image[64, 64] = np.nan
            result = measure_local_aperture_snr(image, x, y, sky_mask=np.ones_like(image, bool))
            self.assertFalse(result['trusted'][0])

    def test_hsc_compatibility_and_legacy_fill_branch(self):
        from preprocessing.ordinary import label_ordinary_sources
        from preprocessing.ordinary_hsc import label_ordinary_sources as hsc
        self.assertIs(label_ordinary_sources, hsc)
        data = sources([0, 100, 200, 300, 400], a=[2, 2, 2, 2, 20])
        table = Table({'base_CircularApertureFlux_6_0_instFlux': np.ones(5),
                       'ext_photometryKron_KronFlux_instFlux': np.ones(5),
                       'pu_refit_aperture_pixel_count': [10, 10, 10, 10, 1]})
        with patch('preprocessing.ordinary_hsc.compute_kron_ellipse', return_value=data.geometry):
            out = hsc(table, np.ones(5, bool), SourceLabels.empty(5),
                      is_narrow_band=False, snr=np.array([2, 4, 6, np.nan, -1]))
        np.testing.assert_array_equal(out.labels.source_class, [3, 2, 1, 3, 4])

    def test_cosmos_categories_not_bits_and_photometry(self):
        data = sources(np.arange(9)*30, flags=[1, 4, 5, 6, 2, 3, 0, 0, 0],
                       model=[24, 24, 24, 24, 24, 24, 25, 26, 26.01])
        out = classify(data, "cosmos")
        np.testing.assert_array_equal(out.labels.source_class, [3, 3, 3, 3, 1, 1, 4, 4, 3])
        self.assertEqual(list(out.stages), ["center_validity", "warn", "star_mask", "photometry", "containment", "bright_fill", "snr"])

    def test_cosmos_bright_fill_boundaries_and_snr(self):
        data = sources(np.arange(8)*30, mag=[24, 25.5, 24, 24, 24, 24, 24, 24],
                       model=[24, 25.5, 24, 24, 24, 24, 24, 24],
                       flags=[0, 0, 0, 0, 0, 1, 0, 0], snr=[10, 10, 10, 10, 10, 10, 2, 10])
        data.segmentation_fill_ratio = np.array([.099, .01, .1, np.nan, -1, .01, .01, 0])
        out = classify(data, "cosmos")
        np.testing.assert_array_equal(out.stages['bright_fill'], [4, 1, 1, 1, 1, 3, 4, 4])
        np.testing.assert_array_equal(out.labels.source_class, [4, 1, 1, 1, 1, 3, 3, 4])
        self.assertEqual(out.labels.reason[0], 'cosmos_bright_low_fill')

    def test_cosmos_fill_preserves_dropped_and_validates_shape(self):
        data = sources([0, 30])
        data.segmentation_fill_ratio = np.zeros(2)
        labels = SourceLabels.empty(2)
        labels.assign(np.array([True, False]), C.DROPPED, 'area_drop')
        out = classify_ordinary_sources(data, np.ones(2, bool), labels, dataset='cosmos')
        np.testing.assert_array_equal(out.labels.source_class, [0, 4])
        with self.assertRaises(ValueError):
            OrdinaryInput(data.geometry, data.mag, segmentation_fill_ratio=np.zeros(3))

    def test_segmentation_unsigned_and_missing_coverage(self):
        from preprocessing.utils.segmentation import measure
        raw = np.full((20, 20), 7 - 2**31, dtype=np.int32)
        h = {'BZERO': 2**31}
        self.assertEqual(measure(raw, h, 10, 10, 3, 2, 0, 7)['own_fill'], 1)
        self.assertEqual(measure(raw, h, 10, 10, 3, 2, 0, 8)['own_fill'], 0)
        self.assertTrue(np.isnan(measure(raw, h, 0, 0, 3, 2, 0, 7)['own_fill']))

    def test_star_not_promoted_by_photometry_or_snr(self):
        data = sources([0, 30, 60], star=[True]*3, snr=[10, 2, 5])
        np.testing.assert_array_equal(classify(data, "cosmos").labels.source_class, [4, 3, 4])

    def test_low_snr_neighbor_cannot_demote_cosmos(self):
        data = sources([50, 53], a=[10, 2], snr=[10, 1])
        np.testing.assert_array_equal(classify(data, "cosmos").labels.source_class, [1, 3])

    def test_psf_minor_and_close_keep_fainter(self):
        data = sources([40, 40.5, 80], mag=[20, 24, 24])
        data.geometry.minor[2] = 0.5
        out = classify(data, "a2744", config=A2744OrdinaryConfig(1))
        np.testing.assert_array_equal(out.labels.source_class, [3, 1, 3])

    def test_nested_priority_and_same_priority(self):
        for flags, expected in (([0, 3], [2, 2]), ([3, 0], [3, 1]),
                                ([3, 3], [4, 2]), ([0, 16], [2, 4])):
            data = sources([50, 53], a=[10, 2], flags=flags)
            with self.subTest(flags=flags):
                out = classify(data, "a2744", config=A2744OrdinaryConfig(1))
                np.testing.assert_array_equal(out.labels.source_class, expected)

    def test_iou_keeps_brighter(self):
        data = sources([50, 51], a=[10, 10], mag=[20, 24])
        np.testing.assert_array_equal(classify(data, "a2744", config=A2744OrdinaryConfig(.5)).labels.source_class, [1, 3])

    def test_ignored_parent_stops_demoting_neighbors(self):
        data = sources([50, 54, 40], a=[20, 2, 2], flags=[2, 0, 3])
        out = classify(data, "a2744", config=A2744OrdinaryConfig(1))
        np.testing.assert_array_equal(out.labels.source_class, [3, 1, 1])

    def test_snr_applies_to_strict_candidates(self):
        data = sources(np.arange(5)*30, flags=[16]*5, snr=[-1, 2.99, 3, 5, np.nan])
        out = classify(data, "a2744", config=A2744OrdinaryConfig(1))
        np.testing.assert_array_equal(out.labels.source_class, [3, 3, 4, 4, 3])
        apply_remeasured_aperture_snr(out, data.mag, dict(snr=[100, 100, -2, 4, 100], trusted=[False]*5))
        np.testing.assert_array_equal(out.labels.source_class, [3, 3, 3, 4, 3])

    def test_star_footprint_protects_bright_sources(self):
        data = sources([30, 60], mag=[21.9, 22])
        mask = np.ones((100, 100), bool)
        out = classify(data, "a2744", config=A2744OrdinaryConfig(1), star_footprint=mask)
        np.testing.assert_array_equal(out.labels.source_class, [1, 3])

    def test_a_scaled_faint_drop_and_no_resurrection(self):
        data = sources([50, 80], a=[100, 2], mag=[29, 24])
        stage = classify_catalog_basics(data.geometry, data.mag, dataset="cosmos", pixel_scale_arcsec=.03)
        self.assertTrue(stage.a_faint_large[0])
        out = classify_ordinary_sources(data, np.ones(2, bool), stage.labels, dataset="cosmos")
        self.assertEqual(out.labels.source_class[0], C.DROPPED)

    def test_noise_measurement_and_empty_sky(self):
        rng = np.random.default_rng(4)
        image = rng.normal(0, 1, (128, 128)).astype(np.float32)
        data = sources([64])
        data.geometry.y[:] = 64
        image[60:69, 60:69] += 5
        sky = np.ones(image.shape, bool)
        sky[50:78, 50:78] = False
        out = remeasure_aperture_snr(image, data.geometry, sky_mask=sky, radius=4, background_box=64)
        self.assertGreater(out["snr"][0], 3)
        self.assertEqual(out["background_method"], "lsst_sky")
        empty = remeasure_aperture_snr(image, data.geometry, sky_mask=np.zeros_like(sky), radius=4)
        self.assertFalse(empty["trusted"][0])
        self.assertTrue(np.isnan(empty["snr"][0]))


if __name__ == "__main__":
    unittest.main()
