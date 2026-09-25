"""JWST post-ordinary bright policies and HSC-compatible output contract."""

import unittest
import numpy as np
from astropy.table import Table
from preprocessing.bright_label import BrightLabelResult
from preprocessing.bright_label_jwst import JWSTBrightConfig, label_bright_sources
from preprocessing.labels import SourceClass as C, SourceLabels
from preprocessing.ordinary_common import OrdinaryInput
from preprocessing.utils.geometry import EllipseGeometry


def data(x, y, flags=None, mag=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    a = np.full(n, 2.)
    geom = EllipseGeometry(x, y, a, a, np.zeros(n), np.pi*a*a)
    return OrdinaryInput(geom, np.full(n, 20.) if mag is None else np.asarray(mag),
                         flags=np.zeros(n, int) if flags is None else np.asarray(flags))


def labels(values):
    out = SourceLabels.empty(len(values))
    out.source_class[:] = values
    out.reason[:] = 'ordinary_done'
    return out


def gaia(x, y, sid=100, evidence=True):
    return dict(x=x, y=y, source_id=sid, phot_g_mean_mag=18., astrometric_evidence=evidence)


class JWSTBrightTests(unittest.TestCase):
    def test_cosmos_accept_insert_skip_and_preserve_ignore(self):
        d = data([10, 50], [10, 50])
        initial = labels([C.ORDINARY_IGNORE, C.CLEAN])
        rows = [gaia(10.1, 10), gaia(51, 50, 101, False), gaia(80, 80, 102, False)]
        out = label_bright_sources(d, initial, image_shape=(100, 100), gaia_rows=rows,
             config=JWSTBrightConfig('cosmos', .03, psf_fwhm_arcsec=.145))
        self.assertIsInstance(out, BrightLabelResult)
        np.testing.assert_array_equal(out.labels.source_class, [3, 1])
        np.testing.assert_array_equal(out.strict_center_x, [80])
        self.assertEqual(out.strict_center_source_id.dtype, np.int64)
        self.assertFalse(out.restricted_fallback_component_ids.size)
        self.assertTrue(out.cluster_rows[0]['accepted_kron'])
        self.assertFalse(out.cluster_rows[1]['inserted'])

    def test_cosmos_multiple_matches_inserts_no_geometry(self):
        out = label_bright_sources(data([10, 11], [10, 10]), labels([1, 1]), image_shape=(100, 100),
            gaia_rows=[gaia(10, 10), gaia(10, 10)], bright_region=np.ones((100, 100), bool),
            config=JWSTBrightConfig('cosmos', .03, .145))
        self.assertEqual(len(out.strict_center_x), 1)
        self.assertEqual(out.strict_center_component_id[0], 0)

    def test_isolated_flags_cap_without_reviving(self):
        mask = np.zeros((120, 120), bool)
        mask[0:40, 0:40] = True
        out = label_bright_sources(data([20, 60, 80, 100], [20, 60, 80, 100], flags=[0, 2, 3, 0]),
            labels([4, 1, 1, 3]), image_shape=mask.shape, bright_region=mask,
            config=JWSTBrightConfig('a2744', .03))
        np.testing.assert_array_equal(out.labels.source_class, [4, 2, 3, 3])
        self.assertFalse(out.strict_center_x.size)
        self.assertEqual(out.restricted_fallback_component_ids.tolist(), [1])

    def test_empty_large_gaia_first_and_small_ignore(self):
        mask = np.zeros((100, 140), bool)
        mask[5:30, 5:45] = True  # exactly 1000
        mask[40:65, 5:45] = True
        mask[5:15, 70:80] = True
        out = label_bright_sources(data([], []), labels([]), image_shape=mask.shape, bright_region=mask,
            gaia_rows=[gaia(20, 20)], config=JWSTBrightConfig('a2744', .03))
        self.assertEqual(len(out.strict_center_x), 2)
        self.assertEqual(out.restricted_fallback_component_ids.tolist(), [1, 3])
        self.assertEqual(out.ordinary_ignore_component_ids.tolist(), [2])
        self.assertIn('a2744_empty_component_gaia', out.strict_center_reason)
        self.assertIn('a2744_empty_component_geometry', out.strict_center_reason)

    def test_gaia_must_be_inside_component(self):
        mask = np.zeros((80, 80), bool); mask[10:50, 10:50] = True
        out = label_bright_sources(data([20, 40], [20, 40]), labels([1, 1]), image_shape=mask.shape,
             bright_region=mask, gaia_rows=[gaia(9, 20)], config=JWSTBrightConfig('a2744', .03))
        np.testing.assert_array_equal(out.labels.source_class, [3, 3])
        self.assertEqual(out.strict_center_reason.tolist(), ['a2744_empty_component_geometry'])

    def test_small_component_gaia_point_does_not_restore_bright_mask(self):
        mask = np.zeros((80, 80), bool); mask[10:30, 10:30] = True
        out = label_bright_sources(data([15, 25], [15, 25]), labels([1, 1]), image_shape=mask.shape,
             bright_region=mask, gaia_rows=[gaia(20, 20)], config=JWSTBrightConfig('a2744', .03))
        self.assertEqual(len(out.strict_center_x), 1)
        self.assertFalse(out.restricted_fallback_component_ids.size)
        self.assertEqual(out.ordinary_ignore_component_ids.tolist(), [1])

    def test_pm_projection_and_evidence(self):
        from astropy.wcs import WCS
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [51, 51]; wcs.wcs.crval = [150, 2]
        wcs.wcs.cdelt = [-.03/3600, .03/3600]; wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        header = wcs.to_header(); header['DATE-OBS'] = '2026-01-01'
        tab = Table(dict(source_id=[123456789012345678], ra=[150.], dec=[2.],
                         pmra=[30.], pmdec=[0.], phot_g_mean_mag=[18.], ref_epoch=[2016.]))
        out = label_bright_sources(data([], []), labels([]), image_shape=(100, 100), gaia_table=tab,
             image_header=header, config=JWSTBrightConfig('cosmos', .03, .145))
        self.assertLess(out.strict_center_x[0], 45)
        self.assertEqual(int(out.strict_center_source_id[0]), -123456789012345679)

    def test_requires_ordinary_complete(self):
        with self.assertRaises(ValueError):
            label_bright_sources(data([10], [10]), SourceLabels.empty(1), image_shape=(100, 100),
                                 config=JWSTBrightConfig('cosmos', .03, .145))


if __name__ == '__main__':
    unittest.main()
