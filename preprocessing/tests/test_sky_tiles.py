import unittest
import numpy as np
from astropy.wcs import WCS
from preprocessing.utils.sky_tiles import SkyTile, plan_sky_tiles, tile_header, reproject_tile


def header(n=32):
    w=WCS(naxis=2);w.wcs.ctype=['RA---TAN','DEC--TAN']
    w.wcs.crval=[3.57,-30.38];w.wcs.crpix=[16,16];w.wcs.cdelt=[-.03/3600,.03/3600]
    h=w.to_header();h['NAXIS1']=n;h['NAXIS2']=n;return h


class SkyTilesTest(unittest.TestCase):
    def test_anchor_overlap_and_wcs(self):
        h=header(80);ts=plan_sky_tiles(h,h,anchor=(13,17),size=16,overlap=4)
        t=next(t for t in ts if (t.ix,t.iy)==(0,0));u=next(t for t in ts if (t.ix,t.iy)==(1,0))
        self.assertEqual((t.x0,t.y0),(13,17));self.assertEqual(t.x0+16-u.x0,4)
        p=WCS(tile_header(h,t)).all_pix2world([[0,0],[15,15]],0)
        q=WCS(h).all_pix2world([[13,17],[28,32]],0)
        np.testing.assert_allclose(p,q,atol=1e-10,rtol=0)

    def test_surface_brightness_and_holes(self):
        h=header();data=np.full((32,32),7.,np.float32);data[10:20,10:20]=np.nan
        out=reproject_tile(data,WCS(h),tile_header(h,SkyTile(0,0,0,0,32)),block_rows=7)
        np.testing.assert_allclose(out[np.isfinite(out)],7,atol=1e-5)
        self.assertTrue(np.isnan(out[12:18,12:18]).all())
        self.assertTrue(np.isfinite(out[:5,:5]).all())

    def test_subpixel_mapping_and_empty(self):
        h=header();target=h.copy();target['CRPIX1']+=.25
        data=np.tile(np.arange(32,dtype=np.float32),(32,1))
        out=reproject_tile(data,WCS(h),target)
        np.testing.assert_allclose(out[5:25,5:25],data[5:25,5:25]-.25,atol=1e-5)
        self.assertTrue(np.isnan(reproject_tile(np.full_like(data,np.nan),WCS(h),target)).all())

if __name__=='__main__':unittest.main()
