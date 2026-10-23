"""Angular apertures, local-mask equivalence and normal-view SNR concurrency."""
import math
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from eval.datasets.parallel_load import ParallelLoad
from eval.hsctiles.browser_core import BrowserState
from eval.tests.test_browser_parallel_export import make_state
from utils import source_snr as snr


def legacy_annulus(image, row, rows, config):
    height,width=image.shape;x,y,*_=snr._ellipse_params(row)
    pad=math.ceil(config.annulus_outer_radius)+2
    x0=max(0,math.floor(x)-pad);x1=min(width,math.floor(x)+pad+1)
    y0=max(0,math.floor(y)-pad);y1=min(height,math.floor(y)+pad+1)
    if x1<=x0 or y1<=y0:return np.nan,np.nan,0,0
    yy,xx=np.mgrid[y0:y1,x0:x1];rr=(xx-x)**2+(yy-y)**2
    ap=rr<=config.aperture_radius**2
    ann=(rr>=config.annulus_inner_radius**2)&(rr<=config.annulus_outer_radius**2)
    other=snr._source_mask([r for r in rows if r is not row],image.shape,scale=1)[y0:y1,x0:x1]
    local=image[y0:y1,x0:x1];ap &= np.isfinite(local);ann &= np.isfinite(local)&~other
    na,nann=int(ap.sum()),int(ann.sum())
    if not na:return np.nan,np.nan,nann,na
    bg=float(np.median(local[ann]))*na if nann>=config.min_annulus_pixels else np.nan
    flux=float(local[ap].sum(dtype=np.float64))-bg
    return flux,bg,nann,na


class AngularSnrTests(unittest.TestCase):
    def test_angular_diameters_and_small_ztf_annulus(self):
        for scale,diameter in ((.168,2.),(1.013,2.),(.03,.3),(.06,.3)):
            c=snr.angular_aperture_config(snr.SourceSnrConfig(),pixel_scale_arcsec=scale,diameter_arcsec=diameter)
            self.assertAlmostEqual(c.aperture_radius*2*scale,diameter)
            self.assertAlmostEqual(c.annulus_inner_radius,2*c.aperture_radius)
            self.assertAlmostEqual(c.annulus_outer_radius,3*c.aperture_radius)
        self.assertEqual(c.aperture_radius,2.5)
        z=snr.angular_aperture_config(snr.SourceSnrConfig(),pixel_scale_arcsec=1.013,diameter_arcsec=2)
        self.assertEqual(z.min_annulus_pixels,4)
        image=np.random.default_rng(3).normal(size=(64,64)).astype('f4')
        row=dict(x=32.,y=32.,major=1.,minor=1.,theta=0.)
        _,_,nann,nap=snr._annulus_flux(image,row,[row],config=z)
        self.assertGreaterEqual(nann,z.min_annulus_pixels)
        self.assertGreater(nap,0)
        for bad in (0,-1,np.nan,np.inf):
            with self.assertRaises(ValueError):snr.angular_aperture_config(z,pixel_scale_arcsec=bad,diameter_arcsec=2)

    def test_browser_ztf_default_6arcsec_and_explicit_override(self):
        s=object.__new__(BrowserState);s.dataset_id='ztf';s.snr_config=snr.SourceSnrConfig()
        s.access=SimpleNamespace(pixel_scale_arcsec=1.013)
        c=s._snr_config_for_ref(SimpleNamespace())
        self.assertEqual(c.aperture_diameter_arcsec,6.)
        self.assertAlmostEqual(c.aperture_radius,3/1.013)
        self.assertEqual(c.min_annulus_pixels,36)
        s.snr_aperture_diameter_arcsec=2.
        self.assertEqual(s._snr_config_for_ref(SimpleNamespace()).aperture_diameter_arcsec,2.)
        s.snr_aperture_diameter_arcsec=None;s.dataset_id='hsc_image'
        self.assertEqual(s._snr_config_for_ref(SimpleNamespace()).aperture_diameter_arcsec,2.)

    def test_actual_jwst_reference_wcs_overrides_native_scale(self):
        s=object.__new__(BrowserState);s.dataset_id='jwst_nircam';s.snr_config=snr.SourceSnrConfig();s._snr_scale_cache={}
        s._bands_for_patch=lambda _:['F115W','F277W'];calls=[]
        def wcs(band,patch):
            calls.append(band);return SimpleNamespace(pixel_scale_matrix=np.diag([-.03/3600,.03/3600]))
        s.access=SimpleNamespace(pixel_scale_arcsec=.06,_present_bands=lambda *a:('f115w','f277w'),_reference_band=lambda *a:'f115w',_wcs_for=wcs)
        c=s._snr_config_for_ref(SimpleNamespace(patch='p',band='F277W'))
        self.assertEqual(c.aperture_radius,5.)
        self.assertEqual(c.min_annulus_pixels,100)
        self.assertEqual(calls,['f115w'])
        s._snr_config_for_ref(SimpleNamespace(patch='p',band='F115W'));self.assertEqual(len(calls),1)

    def test_local_mask_preserves_original_annulus_photometry(self):
        rng=np.random.default_rng(21);image=rng.normal(size=(96,112)).astype('f4');image[20:25,30:40]=np.nan
        rows=[dict(x=float(rng.uniform(-20,130)),y=float(rng.uniform(-15,110)),major=float(rng.uniform(1,16)),minor=float(rng.uniform(1,8)),theta=float(rng.uniform(-3,3))) for _ in range(60)]
        rows += [dict(x=32.,y=32.,major=6.,minor=3.,theta=0.)]
        for radius in (5.,.9871668311944719,2.5,5.952380952380952):
            c=snr.SourceSnrConfig(aperture_radius=radius)
            for row in rows:
                np.testing.assert_allclose(snr._annulus_flux(image,row,rows,config=c),legacy_annulus(image,row,rows,c),rtol=0,atol=1e-12,equal_nan=True)

    def test_normal_browsing_page_runs_multiple_snr_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=make_state(tmp,6);s.snr_workers=3;s._page_tile_slots=lambda _:[(str(i),0) for i in range(6)];s._bands_for_patch=lambda _:['F115W']
            s._refs_by_tile_slot=lambda:{(str(i),0):{'F115W':r} for i,r in enumerate(s.refs)}
            s.detect_rows_by_token={r.token:[{}] for r in s.refs};s._snr_load=ParallelLoad(3)
            active=peak=0;lock=threading.Lock()
            def measure(token,method):
                nonlocal active,peak
                with lock:active+=1;peak=max(peak,active)
                time.sleep(.04)
                with lock:active-=1
                return [dict(snr=8.,snr_bad=0)]
            s._snr_rows_for_token=measure
            result=s.prepare_snr_page(0,'ap2')
            self.assertEqual(result['n_sources'],6);self.assertGreater(peak,1);self.assertLessEqual(peak,3)
            self.assertEqual(list(result['by_token']),[r.token for r in s.refs])

    def test_equal_requests_share_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=make_state(tmp,1);s._snr_load=ParallelLoad(2);ref=s.refs[0];rows=[dict(x=16.,y=16.,major=2.,minor=1.,theta=0.)]
            s.detect_rows_by_token[ref.token]=rows;count=0;lock=threading.Lock()
            def measured(*args,**kwargs):
                nonlocal count
                with lock:count+=1
                time.sleep(.07);return [dict(snr=10.)],None
            with patch('eval.hsctiles.browser_core.measure_source_snrs',side_effect=measured):
                with ThreadPoolExecutor(4) as pool:
                    results=list(pool.map(lambda _:s._measure_snr_rows_for_ref(ref,s._raw_image_for_ref(ref),rows,methods=('ap2',)),range(4)))
            self.assertEqual(count,1);self.assertEqual(len(results),4)


if __name__=='__main__':unittest.main()
