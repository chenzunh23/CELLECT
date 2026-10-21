"""Adaptive smoothing in science pixels, native AUTO fit and cache behavior."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from preprocessing.utils.jwst_background_local import (
    LocalSkyConfig, local_sextractor_background, file_sha256, implementation_sha256)


class PsfBackgroundTests(unittest.TestCase):
    def test_blue_floor_and_red_psf_resolve_to_one_native_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);stamp=root/'psf.fits'
            fits.PrimaryHDU(np.ones((9,9),np.float32)).writeto(stamp)
            info=stamp.stat()
            asset=root/'psf.json'
            asset.write_text(json.dumps({b:dict(path=str(stamp),source_bytes=info.st_size,
                source_mtime_ns=info.st_mtime_ns,lsst_adaptive_sigma_arcsec=sigma)
                for b,sigma in [('F115W',.015),('F444W',.060)]}))
            cfg=LocalSkyConfig(str(asset),file_sha256(asset),implementation_sha256())
            w=WCS(naxis=2);w.wcs.crpix=[64,64];w.wcs.crval=[150,2]
            w.wcs.cdelt=[-.03/3600,.03/3600];w.wcs.ctype=['RA---TAN','DEC--TAN']
            h=w.to_header();h['BUNIT']='MJy/sr'
            yy,xx=np.mgrid[:128,:128]
            raw=np.random.default_rng(7).normal(0,.1,(128,128)).astype('f4')
            raw+=10*np.exp(-((xx-64)**2+(yy-64)**2)/8)
            for band,sigma in [('F115W',1.),('F444W',2.)]:
                h['FILTER']=band;out=root/f'{band}.npz'
                sky=local_sextractor_background(raw,h,out,config=cfg)
                self.assertFalse(sky[64,64]);self.assertTrue(sky.any())
                meta=json.loads(out.with_suffix('.json').read_text())
                self.assertEqual(len(meta['passes']),1)
                self.assertAlmostEqual(meta['passes'][0]['smoothing_sigma_pixels'],sigma)
                self.assertEqual(meta['passes'][0]['minarea_native_pixels'],1)
                with patch('preprocessing.utils.jwst_background_local.fit_native_background',side_effect=AssertionError('cache miss')):
                    np.testing.assert_array_equal(sky,local_sextractor_background(raw,h,out,config=cfg))


if __name__=='__main__':unittest.main()
