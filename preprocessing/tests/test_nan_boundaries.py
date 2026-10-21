"""Regression coverage for original-FITS holes versus open no-data boundaries."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from preprocessing.image_processing import ImagePreparationConfig, prepare_image
from preprocessing.utils.image import fill_nonfinite_hybrid
from preprocessing.utils.nan_boundaries import _packed_edges, edge_connected_nonfinite
from preprocessing.utils.parent_scaling import fit_parent_rgb, apply_parent_rgb


def header():
    w=WCS(naxis=2)
    w.wcs.ctype=['RA---TAN','DEC--TAN'];w.wcs.crval=[150,2]
    w.wcs.crpix=[32,32];w.wcs.cdelt=[-.03/3600,.03/3600]
    h=w.to_header();h['BUNIT']='MJy/sr'
    return h


class NaNBoundaryTests(unittest.TestCase):
    def test_original_fits_sampled_noise_is_shared_by_small_source_dominated_crop(self):
        from preprocessing.utils.nan_boundaries import source_nan_background
        from astropy.stats import sigma_clipped_stats
        a=np.random.default_rng(31).normal(1,.02,(1010,780)).astype('f4')
        a[:40]=np.nan;a[490:510,490:510]+=10;a[498:502,498:502]=np.nan
        h=header()
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'sampled.fits';fits.PrimaryHDU(a,h).writeto(source)
            stats=source_nan_background(source)
            sample=a[::2,::2];values=sample[np.isfinite(sample)]
            _,median,sigma=sigma_clipped_stats(values,sigma=3.,maxiters=None)
            np.testing.assert_allclose(stats,(median,sigma),rtol=1e-6)
            ch=h.copy();ch['CRPIX1']-=488;ch['CRPIX2']-=488
            p=prepare_image(a[488:514,488:514],header=ch,source_path=source)
            self.assertTrue(p.input_valid_mask[10:14,10:14].all())
            self.assertEqual(p.metadata['nan_threshold_reference'],'original_fits_sampled_finite')

    def test_authoritative_original_mask_does_not_spread_through_padding(self):
        a=np.random.default_rng(20).normal(1,.01,(64,64)).astype('f4')
        a[8:16,2:10]+=10
        a[10:14,:5]=np.nan
        a[:,:2]=np.nan  # Artificial join: only the first columns are protected.
        blocked=np.zeros_like(a,bool);blocked[:,:2]=True
        filled,_=fill_nonfinite_hybrid(a,edge_connected_mask=blocked)
        self.assertTrue(np.isnan(filled[:,:2]).all())
        self.assertTrue(np.isfinite(filled[10:14,2:5]).all())

    def test_padding_does_not_change_any_rgb_statistics(self):
        from data_filtering.sam_input_scaling import scale_training_image, build_bright_mask
        a=np.random.default_rng(5).normal(0,.03,(64,64)).astype('f4')
        a[20:24,20:24]+=5
        padded=np.pad(a,((0,64),(0,192)),constant_values=np.nan)
        p,pp=fit_parent_rgb(a),fit_parent_rgb(padded)
        for name in ('log_stats','lupton_stats'):
            np.testing.assert_allclose(p[name],pp[name],rtol=1e-6)
        np.testing.assert_allclose(apply_parent_rgb(a,p),apply_parent_rgb(padded,pp)[:,:64,:64],atol=1e-6)
        for mode in ('zscore-log-lupton-rgb','anscombe-rgb'):
            opts=dict(mode=mode,clip_threshold=5,anscombe_clip=True)
            np.testing.assert_allclose(scale_training_image(a,**opts),
                scale_training_image(padded,**opts)[:,:64,:64],atol=1e-5)
        opts=dict(threshold=5,clip_threshold=5,dilation=0)
        np.testing.assert_array_equal(build_bright_mask(a,**opts),build_bright_mask(padded,**opts)[:64,:64])

    def test_hard_quality_failure_is_not_repaired_as_science(self):
        a=np.random.default_rng(21).normal(1,.01,(64,64)).astype('f4')
        a[5:12,5:12]+=10;a[8,8]=np.nan
        bad=np.zeros_like(a,bool);bad[16,16]=True
        p=prepare_image(a,config=ImagePreparationConfig(input_unit='native'),protected_invalid_mask=bad)
        self.assertTrue(p.input_valid_mask[8,8])
        self.assertFalse(p.finite_mask[8,8])
        self.assertFalse(p.input_valid_mask[16,16])
        self.assertFalse(p.finite_mask[16,16])

    def test_tiled_connectivity_matches_full_flood(self):
        # Test non-multiple edge tile sizes, winding joins, and enclosed holes.
        rng=np.random.default_rng(7)
        for shape in ((23,29),(32,32)):
            for density in (.1,.65):
                a=rng.normal(size=shape).astype('f4')
                a[rng.random(shape)<density]=np.nan
                a[3:7,0:19]=np.nan
                with tempfile.TemporaryDirectory() as tmp:
                    packed=_packed_edges(a,tmp,tile_size=8)
                actual=np.unpackbits(packed,axis=1)[:,:shape[1]].astype(bool)
                np.testing.assert_array_equal(actual,edge_connected_nonfinite(a))

    def test_open_component_beside_bright_source_stays_nan(self):
        a=np.random.default_rng(1).normal(0,.01,(128,128)).astype('f4')
        a[40:80,:30]=np.nan
        a[40:80,30:40]=10
        a[90:94,90:94]=np.nan  # Closed background hole must stay invalid too.
        repaired,stats=fill_nonfinite_hybrid(a)
        self.assertTrue(np.isnan(repaired[40:80,:30]).all())
        self.assertTrue(np.isnan(repaired[90:94,90:94]).all())
        self.assertEqual(stats['nan_pixels_unfilled'],1216)
        self.assertEqual(stats['nan_pixels_background_unfilled'],16)
        self.assertEqual(stats['nan_pixels_nearest_fill'],0)
        np.testing.assert_array_equal(repaired[np.isfinite(a)],a[np.isfinite(a)])

    def test_crop_edge_is_not_original_fits_edge(self):
        a=np.random.default_rng(2).normal(0,.01,(64,64)).astype('f4')
        a[20:42,18:42]+=10
        a[28:32,22:28]=np.nan  # Closed original hole, cut by x=24 crop.
        a[40:44,:30]=np.nan    # Original edge-connected component.
        h=header()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{'CELLECT_NAN_CACHE':tmp}):
            source=Path(tmp)/'source.fits';fits.PrimaryHDU(a,h).writeto(source)
            cut=a[24:48,24:48];ch=h.copy();ch['CRPIX1']-=24;ch['CRPIX2']-=24
            prepared=prepare_image(cut,header=ch,source_path=source)
        self.assertTrue(np.isfinite(prepared.image[4:8,:4]).all())
        self.assertGreater(prepared.image[4,0],1)
        self.assertTrue(np.isnan(prepared.image[16:20,:6]).all())
        self.assertFalse(prepared.finite_mask[4,0])

    def test_parent_provenance_uses_original_not_parent_border(self):
        a=np.random.default_rng(3).normal(0,.01,(64,64)).astype('f4')
        a[18:26,20:30]+=10
        a[20:24,22:28]=np.nan
        h=header()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{'CELLECT_NAN_CACHE':tmp}):
            source=Path(tmp)/'original.fits';fits.PrimaryHDU(a,h).writeto(source)
            ch=h.copy();ch['ORIGFILE']=str(source);ch['CRPIX1']-=24;ch['CRPIX2']-=16
            cut=a[16:40,24:48]
            prepared=prepare_image(cut,header=ch)
            self.assertTrue(np.isfinite(prepared.image).all())
            self.assertEqual(prepared.metadata['nan_pixels_unfilled'],0)

    def test_rgb_invalid_pixels_are_zero_without_fake_bright_fill(self):
        a=np.random.default_rng(4).normal(0,.01,(128,128)).astype('f4')
        a[30:70,:20]=np.nan;a[30:70,20:30]=10
        prep=prepare_image(a,config=ImagePreparationConfig(input_unit='native'))
        rgb=apply_parent_rgb(prep.image,fit_parent_rgb(prep.image))
        self.assertTrue(np.isfinite(rgb).all())
        self.assertTrue((rgb[:,~np.isfinite(a)]==0).all())
        self.assertTrue(np.isnan(prep.image[30:70,:20]).all())


if __name__=='__main__':unittest.main()
