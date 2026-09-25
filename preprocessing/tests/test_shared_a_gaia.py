"""Regression checks for shared area policy and Gaia astrometry."""

import numpy as np
import unittest
from unittest.mock import patch
from astropy.table import Table, MaskedColumn
from astropy.time import Time
from astropy.wcs import WCS

from preprocessing.utils.area_filter import area_filter_masks
from preprocessing.utils.gaia import matching_radius_pixels, observation_time, propagated_gaia_radec
from preprocessing.bright_label import (
    BrightLabelConfig, classify_component_bright, project_gaia_rows,
)
from preprocessing.meas_processing import classify_meas_basics
from preprocessing.labels import SourceClass


def test_area_scales_and_boundaries():
    area = np.array([10000., 10001., 901., 900., 901.])
    mag = np.array([22., 22., 29., 29., 28.])
    expected = area_filter_masks(area, mag)
    assert expected[0].tolist() == [False, True, False, False, False]
    assert expected[1].tolist() == [False, False, True, False, False]
    for scale in (0.03, 0.063):
        actual = area_filter_masks(area * (0.168 / scale)**2, mag, pixel_scale_arcsec=scale)
        for got, want in zip(actual, expected):
            np.testing.assert_array_equal(got, want)
    with unittest.TestCase().assertRaises(ValueError):
        area_filter_masks(area, mag, pixel_scale_arcsec=0)


def test_a_faint_large_is_dropped():
    from types import SimpleNamespace
    import preprocessing.meas_processing as mod
    geom = SimpleNamespace(area=np.array([901.]), x=np.array([10.]), y=np.array([10.]),
                           valid=lambda: np.array([True]), axis_ratio=lambda: np.array([1.]))
    table = Table({'ext_photometryKron_KronFlux_instFlux': [10**(-0.8)], 'deblend_nChild': [0]})
    with patch.object(mod, 'compute_kron_ellipse', lambda *args: geom):
        result = classify_meas_basics(table)
    assert result.labels.mask(SourceClass.DROPPED)[0]
    assert not result.after_a[0]


def test_motion_and_missing_values():
    table = Table({'ra': [10., 10., 10.], 'dec': [60., 60., 60.],
                   'pmra': [100., 100., 100.], 'pmdec': [0., 0., 0.],
                   'ref_epoch': [2015.5, 2016., np.nan]})
    table['pmra'] = MaskedColumn(table['pmra'], mask=[False, True, False])
    ra, dec, moved = propagated_gaia_radec(table, Time(2025.5, format='jyear'))
    assert moved.tolist() == [True, False, False]
    np.testing.assert_allclose((ra[0]-10)*3600, 2., atol=1e-5)
    np.testing.assert_array_equal(ra[1:], [10., 10.])
    assert not propagated_gaia_radec(table, None)[2].any()
    assert observation_time({}) is None
    assert observation_time({'MJD-AVG': 60000., 'MJD-OBS': 59000.}).mjd == 60000.


def test_projection_origin_and_ids():
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [50., 50.]
    wcs.wcs.crval = [10., 20.]
    wcs.wcs.cdelt = [-0.168/3600, 0.168/3600]
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    sid = 1234567890123456789
    table = Table({'ra': [10.], 'dec': [20.], 'source_id': [sid]})
    for origin in (0, 1):
        row = project_gaia_rows(table, image_shape=(100, 100), image_header=wcs.to_header(), pixel_origin=origin)[0]
        assert row['source_id'] == sid
        np.testing.assert_allclose(row['x'], 49+origin)
        np.testing.assert_allclose(row['y'], 49+origin)


def test_outside_component_gaia_not_inserted():
    components = np.zeros((50, 50), dtype=np.int32)
    components[5:40, 5:40] = 1
    gaia = [{'x': 41., 'y': 20., 'source_id': 123, 'phot_g_mean_mag': 17.}]
    sources, _, _, _ = classify_component_bright(
        sources=[], gaia_rows=gaia, component_labels=components,
        quality_mask=None, config=BrightLabelConfig())
    assert not any(s.get('class') == 'gaia_star' for s in sources)
    gaia[0]['x'] = 20.
    sources, _, _, _ = classify_component_bright(
        sources=[], gaia_rows=gaia, component_labels=components,
        quality_mask=None, config=BrightLabelConfig())
    assert any(s.get('class') == 'gaia_star' for s in sources)
    for blocked in (components[:30, :30], np.pad(np.ones((10, 10), dtype=np.int32), 15)):
        sources, _, _, ignored = classify_component_bright(
            sources=[], gaia_rows=gaia, component_labels=blocked,
            quality_mask=None, config=BrightLabelConfig())
        assert not sources
        assert 1 in ignored


def test_matching_radius_is_angular():
    for hsc_pixels in (6., 10.):
        hsc = matching_radius_pixels(0.168, hsc_pixels=hsc_pixels)
        jwst = matching_radius_pixels(0.03, hsc_pixels=hsc_pixels)
        np.testing.assert_allclose(hsc, hsc_pixels)
        np.testing.assert_allclose(hsc * 0.168, jwst * 0.03)
    assert matching_radius_pixels(0.03, arcsec=0.6) == 20.


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(fn) for name, fn in globals().items()
                              if name.startswith('test_') and callable(fn))


if __name__ == '__main__':
    unittest.main()
