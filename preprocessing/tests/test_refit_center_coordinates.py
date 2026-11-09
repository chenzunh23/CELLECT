"""FITS refit centers convert once, at the HSC training boundary only."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from astropy.table import Table
from astropy.wcs import WCS
import zarr

from preprocessing.refit import (
    HSC_REFIT_CENTER_POLICY, RefitConfig, attach_refit_geometry,
    attach_refit_radius_from_table, compute_kron_ellipse,
    image_positions_from_reference,
)
from preprocessing.build_image_level_zarr import write_classified_patch
from preprocessing.tests.test_segmentation_zarr import labels, task
from preprocessing.utils.inputs import TileSpec


def catalog(n=4):
    return Table(dict(
        id=np.arange(43158185032286339, 43158185032286339+n, dtype=np.int64),
        base_SdssCentroid_x=np.arange(n, dtype=float)+108.25,
        base_SdssCentroid_y=np.arange(n, dtype=float)+208.5,
        base_SdssShape_xx=np.full(n, 4.), base_SdssShape_yy=np.ones(n),
        base_SdssShape_xy=np.zeros(n), pu_refit_kron_radius=np.full(n, 4.),
        pu_refit_kron_radius_matched=np.ones(n, bool),
    ))


def refit_rows(table):
    return Table(dict(source_id=table['id'][::-1],
                      proxy_nan0_flux_aperture_radius=np.full(len(table), 4.),
                      proxy_nan0_good=np.ones(len(table), bool),
                      x_image=np.arange(len(table), dtype=float)[::-1]+9.25,
                      y_image=np.arange(len(table), dtype=float)[::-1]+9.5))


class RefitCenterCoordinateTests(unittest.TestCase):
    def test_only_complete_refit_pairs_are_converted_once(self):
        t=catalog()
        t['pu_refit_x_image']=[1.,4.75,np.nan,8.]
        t['pu_refit_y_image']=[1.,8.25,7.,np.nan]
        before=t.copy(copy_data=True)
        g=compute_kron_ellipse(t)
        np.testing.assert_allclose(g.x,[0.,3.75,110.25,111.25])
        np.testing.assert_allclose(g.y,[0.,7.25,210.5,211.5])
        again=compute_kron_ellipse(t)
        for name in ['x','y','major','minor','theta','area']:
            np.testing.assert_array_equal(getattr(g,name),getattr(again,name))
        for name in t.colnames:
            np.testing.assert_array_equal(t[name],before[name])

    def test_missing_refit_coordinates_leave_zero_based_fallback_unchanged(self):
        t=catalog()
        t['pu_refit_x_image']=[1.,2.,3.,4.]  # no matching y column
        g=compute_kron_ellipse(t)
        np.testing.assert_array_equal(g.x,t['base_SdssCentroid_x'])
        np.testing.assert_array_equal(g.y,t['base_SdssCentroid_y'])

    def test_csv_and_in_memory_paths_preserve_fits_columns_and_geometry(self):
        t=catalog();rows=refit_rows(t)
        memory=attach_refit_radius_from_table(t,rows)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'refit.csv';rows.write(path,format='ascii.csv')
            attached=attach_refit_geometry(t,path)
        for table in [memory,attached]:
            g=compute_kron_ellipse(table)
            np.testing.assert_allclose(g.x,np.arange(4)+8.25)
            np.testing.assert_allclose(g.y,np.arange(4)+8.5)
            np.testing.assert_allclose(table['pu_refit_x_image'],np.arange(4)+9.25)
            np.testing.assert_array_equal(table['id'],t['id'])
        np.testing.assert_allclose(attached['v3_x'],np.arange(4)+8.25)
        np.testing.assert_allclose(attached['v3_y'],np.arange(4)+8.5)
        # Position conversion must not change radii/axes/orientation.
        old=compute_kron_ellipse(t)
        for name in ['major','minor','theta','area']:
            np.testing.assert_array_equal(getattr(old,name),getattr(g,name))

    def test_direct_export_stays_fits1_with_global_lsst_fallback(self):
        w=WCS(naxis=2)
        w.wcs.crpix=[30.,40.];w.wcs.cdelt=[-.001,.001]
        w.wcs.crval=[150.,2.];w.wcs.ctype=['RA---TAN','DEC--TAN']
        h=w.to_header();h['LTV1']=-100.;h['LTV2']=-200.
        local=np.array([[8.25,8.5],[9.25,9.5]])
        ra,dec=w.all_pix2world(local[:,0],local[:,1],0)
        ra[1]=np.nan  # exercises a real LSST global-zero-based fallback row
        kwargs=dict(header=h,ra_rad=np.deg2rad(ra),dec_rad=np.deg2rad(dec),
                    fallback_x=local[:,0]+100.,fallback_y=local[:,1]+200.)
        x,y=image_positions_from_reference(**kwargs)
        np.testing.assert_allclose(np.c_[x,y],local+1,atol=1e-8)
        with patch('preprocessing.refit.WCS',side_effect=ValueError('bad WCS')):
            x,y=image_positions_from_reference(**kwargs)
        np.testing.assert_allclose(np.c_[x,y],local+1)

    def test_saved_hsc_targets_share_corrected_centers_without_moving_gaia(self):
        t=catalog(1);t=attach_refit_radius_from_table(t,refit_rows(t))
        g=compute_kron_ellipse(t)
        lab=labels();lab.geom_x=g.x;lab.geom_y=g.y;lab.geom_major=g.major
        lab.geom_minor=g.minor;lab.geom_theta=g.theta;lab.source_ids=np.asarray(t['id'])
        lab.label_classes=np.ones(1,np.uint8)
        lab.strict_x=np.array([12.25]);lab.strict_y=np.array([12.5]);lab.strict_ids=np.array([-999],np.int64)
        lab.strict_is_gaia=np.array([True])
        raw=np.ones((32,32),np.float32)
        with tempfile.TemporaryDirectory() as tmp,patch(
            'preprocessing.build_image_level_zarr._scale_image_chw',side_effect=lambda image,t:image[None]):
            result=write_classified_patch(replace(task(tmp),band='HSC-I'),raw,lab,
                origin=(100,200),tile_specs=[TileSpec('one',100,200,16)])
            store=zarr.open_group(result['output'],mode='r')
            np.testing.assert_allclose(store['source_centers'][:],[[8.25,8.5]])
            np.testing.assert_array_equal(store['source_centers'][:],store['shape_source_centers'][:])
            np.testing.assert_array_equal(store['source_ids'][:],store['shape_source_ids'][:])
            np.testing.assert_allclose(store['strict_center_only_centers'][:],[[12.25,12.5]])
            self.assertEqual(int(store['band_confidence'][0,0,8,8]),4)
            self.assertEqual(float(store['band_shape'][0,0,0,8,8]),float(np.float32(g.major[0])))
            self.assertEqual(store.attrs['hsc_refit_center_policy'],HSC_REFIT_CENTER_POLICY)

    def test_jwst_writer_does_not_apply_hsc_shift_or_policy(self):
        lab=labels();lab.geom_x=np.array([8.25]);lab.geom_y=np.array([8.5])
        lab.geom_major=lab.geom_minor=np.array([2.]);lab.geom_theta=np.array([0.])
        lab.source_ids=np.array([101],np.int64);lab.label_classes=np.ones(1,np.uint8)
        with tempfile.TemporaryDirectory() as tmp,patch(
            'preprocessing.build_image_level_zarr._scale_image_chw',side_effect=lambda image,t:image[None]):
            result=write_classified_patch(task(tmp),np.ones((32,32),np.float32),lab,
                                         tile_specs=[TileSpec('one',0,0,16)])
            store=zarr.open_group(result['output'],mode='r')
            np.testing.assert_allclose(store['source_centers'][:],[[8.25,8.5]])
            self.assertNotIn('hsc_refit_center_policy',store.attrs)


if __name__=='__main__':unittest.main()
