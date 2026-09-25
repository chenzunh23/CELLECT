"""PSF confidence boundaries, metadata and actual Zarr integration."""
from dataclasses import replace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import zarr
from astropy.wcs import WCS
from preprocessing.utils.confidence import paint_psf_confidence, resolve_confidence
from preprocessing.utils.image_level import _paint_confidence, _tile_targets
from preprocessing.utils.inputs import TileSpec
from preprocessing.tests.test_segmentation_zarr import task, labels
from preprocessing.build_image_level_zarr import write_classified_patch, parse_args
from preprocessing.abell_zarr import store_task


def wcs(scale=.03):
    w=WCS(naxis=2);w.wcs.ctype=['RA---TAN','DEC--TAN'];w.wcs.crval=[3,-30]
    w.wcs.crpix=[16,16];w.wcs.cdelt=[-scale/3600,scale/3600]
    return w


def paint(f, centers=((5,5),), minimum=1.6, maximum=8):
    a=np.zeros((11,11),np.uint8);weights=np.zeros_like(a,float)
    paint_psf_confidence(a,weights,np.array(centers),fwhm_pixels=f,minimum=minimum,maximum=maximum)
    return a,weights


class ConfidenceTests(unittest.TestCase):
    def test_known_four_level_ring_and_weights(self):
        a,weight=paint(2)
        expected=np.array([[0,1,1,1,0],[1,2,3,2,1],[1,3,4,3,1],
                           [1,2,3,2,1],[0,1,1,1,0]],np.uint8)
        np.testing.assert_array_equal(a[3:8,3:8],expected)
        self.assertEqual(np.count_nonzero(a),21)
        np.testing.assert_array_equal(weight,a>0)

    def test_clip_limits_and_nearest_pixel_tie(self):
        np.testing.assert_array_equal(paint(.8)[0],paint(1.6)[0])
        np.testing.assert_array_equal(paint(12)[0],paint(8)[0])
        a,_=paint(2,centers=[(5.5,5.5)])
        self.assertEqual(a[6,6],4);self.assertEqual((a==4).sum(),1)
        self.assertEqual(a[5,5],3)
        for f in [0,-1,np.nan,np.inf]:
            with self.assertRaises(ValueError):paint(f)
        with self.assertRaises(ValueError):paint(2,minimum=9,maximum=8)

    def test_overlap_max_and_empty(self):
        a,_=paint(2,centers=[(3.2,4.1)]);b,_=paint(2,centers=[(5.2,4.1)])
        both,_=paint(2,centers=[(3.2,4.1),(5.2,4.1)])
        np.testing.assert_array_equal(both,np.maximum(a,b))
        self.assertFalse(paint(2,centers=[])[0].any())

    def test_output_wcs_and_override(self):
        t=replace(task('.'),band='F115W',confidence_mode='psf-matched')
        config=resolve_confidence(t,image_wcs=wcs())
        self.assertAlmostEqual(config['fwhm_pixels_raw'],4/3)
        self.assertEqual(config['fwhm_pixels_used'],1.6)
        config=resolve_confidence(replace(t,band='F150W'),image_wcs=wcs())
        self.assertAlmostEqual(config['fwhm_pixels_used'],5/3)
        config=resolve_confidence(replace(t,band='F444W'),image_wcs=wcs(.06))
        self.assertAlmostEqual(config['fwhm_pixels_used'],.145/.06)
        self.assertEqual(resolve_confidence(replace(t,band='HSC-I',confidence_mode='auto'))['mode'],'manhattan')
        with self.assertRaises(ValueError):resolve_confidence(t)
        with self.assertRaises(ValueError):resolve_confidence(replace(t,band='HSC-I',confidence_mode='psf-matched'))
        config=resolve_confidence(replace(t,band='HSC-I',confidence_mode='psf-matched',confidence_fwhm_pixels=3.5))
        self.assertEqual(config['fwhm_pixels_used'],3.5)

    def test_integer_degeneracy_but_subpixel_difference(self):
        np.testing.assert_array_equal(paint(1.6)[0],paint(5/3)[0])
        differences=0
        for x,y in np.random.default_rng(4).uniform(0,1,(100,2)):
            a,_=paint(1.6,[(5+x,5+y)]);b,_=paint(5/3,[(5+x,5+y)])
            differences+=not np.array_equal(a,b)
        self.assertGreater(differences,30)

    def test_manhattan_unmodified_and_cli_abell_defaults(self):
        lab=labels();lab.strict_x=np.array([8.2]);lab.strict_y=np.array([8.3]);lab.strict_ids=np.array([-9])
        out=_tile_targets(lab,TileSpec('test',0,0,16),(0,0),confidence={'mode':'manhattan'})
        expected=np.zeros((16,16),np.uint8)
        _paint_confidence(expected,np.zeros_like(expected,float),np.array([[8.2,8.3]],np.float32))
        np.testing.assert_array_equal(out['confidence'],expected)
        args=parse_args(['--datasets','abell','--output-root','/tmp/confidence-test'])
        task_abell=store_task(args,dict(band='F150W',patch='x+00_y+00'))
        self.assertEqual(task_abell.confidence_mode,'auto')
        self.assertEqual(task_abell.confidence_fwhm_min,1.6)
        self.assertEqual(task_abell.confidence_fwhm_max,8.)

    def test_written_zarr_config_confidence_and_invalid_loss(self):
        lab=labels();lab.strict_x=np.array([7.3]);lab.strict_y=np.array([7.2]);lab.strict_ids=np.array([-123])
        lab.strict_is_gaia=np.array([True])
        raw=np.ones((32,32),np.float32);valid=np.ones(raw.shape,bool);valid[7,7]=False
        with tempfile.TemporaryDirectory() as tmp,patch('preprocessing.build_image_level_zarr._scale_image_chw',side_effect=lambda a,t:a[None]):
            t=replace(task(tmp),band='F115W',confidence_mode='auto')
            result=write_classified_patch(t,raw,lab,image_wcs=wcs(),valid_mask=valid,max_invalid_fraction=.1)
            group=zarr.open_group(result['output'],mode='r')
            config=group.attrs['confidence_config']
            self.assertEqual(config['mode'],'psf-ee')
            self.assertIsNone(config['fwhm_clip_pixels'])
            self.assertEqual(config['ee_fractions']['4'],.1)
            self.assertIn('asset_sha256',config)
            self.assertIn(-123,group['strict_center_only_ids'][:])
            self.assertEqual(group['band_conf_weight'][0,0,7,7],0)
            a=group['band_confidence'][0,0]
            self.assertEqual(a.max(),2)  # missing forced center zeroed, valid EE neighbors survive
            self.assertGreater(np.count_nonzero(a),0)


class EEConfidenceTests(unittest.TestCase):
    def config(self, band='F115W', scale=.03, **kwargs):
        return resolve_confidence(replace(task('.'), band=band, confidence_mode='auto', **kwargs), image_wcs=wcs(scale))

    def paint(self, band='F115W', centers=((10,10),), shape=(23,23)):
        from preprocessing.utils.confidence import paint_ee_confidence
        cfg=self.config(band)
        a=np.zeros(shape,np.uint8);weight=np.zeros(shape,float)
        paint_ee_confidence(a,weight,centers,level_radii_pixels=cfg['level_radii_pixels'])
        return a,weight

    def test_asset_radii_normalization_and_grid(self):
        cfg=self.config()
        self.assertEqual(cfg['mode'],'psf-ee')
        self.assertEqual(cfg['ee_fractions'],{'4':.1,'3':.35,'2':.6,'1':.7})
        self.assertEqual(cfg['normalization']['denominator'],1.)
        self.assertFalse(cfg['normalization']['renormalize_by_stamp_sum'])
        self.assertIsNone(cfg['fwhm_clip_pixels'])
        self.assertAlmostEqual(cfg['level_radii_pixels']['4'],.307,places=3)
        self.assertEqual(len(cfg['asset_sha256']),64)
        scaled=self.config(scale=.06)
        for key,r in cfg['level_radii_pixels'].items():
            self.assertAlmostEqual(scaled['level_radii_pixels'][key],r/2)
        # EE assets also support filters absent from the old nominal FWHM table.
        self.assertEqual(self.config('F164N')['mode'],'psf-ee')
        with self.assertRaises(ValueError):self.config('F999W')
        with self.assertRaises(ValueError):self.config(confidence_fwhm_pixels=3.)
        warped=wcs();warped.wcs.cdelt=[-.03/3600,.06/3600]
        with self.assertRaises(ValueError):resolve_confidence(replace(task('.'),band='F115W',confidence_mode='psf-ee'),image_wcs=warped)

    def test_reviewed_six_band_counts_at_integer_center(self):
        expected={'F115W':[1,0,4,16],'F150W':[1,0,4,16],'F200W':[1,4,4,28],
                  'F277W':[1,8,12,40],'F356W':[1,8,20,68],'F444W':[5,16,24,92]}
        for band,counts in expected.items():
            a,weight=self.paint(band)
            self.assertEqual([int((a==k).sum()) for k in [4,3,2,1]],counts)
            np.testing.assert_array_equal(weight,a>0)

    def test_fallback_tie_and_natural_multiple_level4(self):
        for center in [(9.5,9.5),(9.6,9.6),(10.,9.5)]:
            a,_=self.paint(centers=[center]);self.assertEqual((a==4).sum(),1)
            self.assertEqual(a[10,10],4)
        a,_=self.paint('F444W',centers=[(9.5,9.5)])
        self.assertEqual((a==4).sum(),4)
        self.assertTrue(np.all(a[9:11,9:11]==4))

    def test_overlap_weights_empty_and_image_edge(self):
        from preprocessing.utils.confidence import paint_ee_confidence
        a,_=self.paint(centers=[(9.6,9.6)]);b,_=self.paint(centers=[(11.6,9.6)])
        both,_=self.paint(centers=[(9.6,9.6),(11.6,9.6)])
        np.testing.assert_array_equal(both,np.maximum(a,b))
        empty,_=self.paint(centers=[(np.nan,5),(-1,1),(100,1)])
        self.assertFalse(empty.any())
        a,_=self.paint(centers=[(22.9,22.9)])
        self.assertEqual(a[22,22],4)
        weights=np.full(a.shape,.2)
        paint_ee_confidence(a,weights,[(0,0)],level_radii_pixels=self.config()['level_radii_pixels'],value_weight=.8)
        self.assertEqual(weights[0,0],.8);self.assertEqual(weights[15,15],.2)

    def test_custom_asset_validation_and_cli(self):
        import json
        from pathlib import Path
        from preprocessing.utils.confidence import EE_CONFIG
        cfg=json.loads(EE_CONFIG.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'config.json';p.write_text(json.dumps(cfg))
            args=parse_args(['--datasets','abell','--output-root',tmp,'--confidence-mode','psf-ee','--confidence-config-path',str(p)])
            t=store_task(args,dict(band='F150W',patch='x+00_y+00'))
            self.assertEqual(t.confidence_config_path,str(p))
            resolved=resolve_confidence(t,image_wcs=wcs())
            self.assertEqual(resolved['asset_path'],str(p))
            cfg['boundaries'][0]['ee_fraction']=.2;p.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):resolve_confidence(t,image_wcs=wcs())
