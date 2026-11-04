"""UNCOVER native flags, missing geometry, non-ignore nesting and SNR priority."""
import unittest
import numpy as np
from astropy.table import Table
from astropy.wcs import WCS
from preprocessing.labels import SourceClass as C
from preprocessing.ordinary_common import OrdinaryInput
from preprocessing.ordinary_a2744 import A2744OrdinaryConfig
from preprocessing.utils.geometry import EllipseGeometry
from preprocessing.utils.uncover_labels import (adapt_catalog,classify_ordinary,UncoverSnrConfig,
    band_photometry,project_product,coverage_on_grid,training_coverage,model_for_band)
from preprocessing.snr import SnrThresholds


def make_input(n):
    x=np.arange(n,dtype=float)*40+50
    g=EllipseGeometry(x,np.full(n,100.),np.full(n,5.),np.full(n,4.),np.zeros(n),np.full(n,20*np.pi))
    d=OrdinaryInput(g,np.full(n,25.),snr=np.full(n,10.),flags=np.zeros(n,int))
    t=Table({k:np.zeros(n,int) for k in ['flag_nophot','flag_artifact','flag_nearbcg','flag_lowsnr']})
    return d,t,np.ones(n,bool)


class UncoverTests(unittest.TestCase):
    def test_bash_passes_actual_uncover_workers_and_background_arguments(self):
        import os,sys,json,tempfile,subprocess
        from pathlib import Path
        repo=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);stub=root/'python_stub';capture=root/'argv.json'
            stub.write_text('#!'+sys.executable+'\n'+
                'import json,os,sys\n'+
                'sys.path.insert(0,'+repr(str(repo))+')\n'+
                'if sys.argv[1:4]==["-u","-m","preprocessing.build_image_level_zarr"]:\n'+
                ' from preprocessing.build_image_level_zarr import parse_args\n'+
                ' from pathlib import Path\n'+
                ' a=parse_args(sys.argv[4:])\n'+
                ' Path(os.environ["CAPTURE_ARGV"]).write_text(json.dumps(dict(workers=a.workers,use_uncover=a.use_uncover,background_root=a.background_root,stage=a.abell_stage,no_kron=a.uncover_no_kron_snr,risk_no_kron=a.uncover_risk_no_kron_snr,snr_ignore=a.snr_ignore_threshold,snr_weak=a.snr_weak_threshold)))\n')
            stub.chmod(0o755)
            env={**os.environ,'JWST_PYTHON':str(stub),'OUTPUT_ROOT':str(root/'out'),
                 'CAPTURE_ARGV':str(capture),'ZARR_WORKERS':'64','ABELL_STAGE':'zarr','USE_UNCOVER':'1',
                 'BACKGROUND_ROOT':'/test/background with spaces','UNCOVER_NO_KRON_SNR':'3','UNCOVER_RISK_NO_KRON_SNR':'5',
                 'SNR_IGNORE_THRESHOLD':'6','SNR_WEAK_THRESHOLD':'9'}
            p=subprocess.run(['bash',str(repo/'preprocessing/run_abell_preprocessing.sh')],env=env,
                             text=True,capture_output=True,timeout=30)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)
            actual=json.loads(capture.read_text())
            self.assertEqual(actual,dict(workers=64,use_uncover=True,background_root=env['BACKGROUND_ROOT'],stage='zarr',no_kron=3.,risk_no_kron=5.,snr_ignore=6.,snr_weak=9.))
            self.assertIn('[abell-run] COMMAND ',p.stdout)

    def test_scheduler_heartbeat_and_task_failure_logs(self):
        from concurrent.futures import Future
        from contextlib import redirect_stdout
        from io import StringIO
        from types import SimpleNamespace
        from unittest.mock import patch
        from preprocessing.abell_cutouts import completed_tasks,cut_band
        from preprocessing.abell_zarr import build_parent
        future=Future();future.set_result([])
        out=StringIO()
        with redirect_stdout(out), patch('preprocessing.abell_cutouts.wait',side_effect=[(set(),{future}),({future},set())]):
            self.assertEqual(list(completed_tasks({future:'task'},'test',heartbeat_seconds=.01)),[future])
        self.assertIn('WAIT completed=0/1',out.getvalue())
        self.assertIn('PROGRESS completed=1/1',out.getvalue())
        out=StringIO()
        with redirect_stdout(out),patch('preprocessing.abell_cutouts._cut_band',side_effect=ValueError('test failure')):
            with self.assertRaisesRegex(ValueError,'test failure'):
                cut_band(SimpleNamespace(band='F444W'),None,None,None,4096,128)
        self.assertIn('START band=F444W',out.getvalue())
        self.assertIn('FAILED band=F444W',out.getvalue())
        out=StringIO()
        with redirect_stdout(out),patch('preprocessing.abell_zarr._build_parent',return_value={'status':'complete','grid_samples':2}):
            build_parent(None,{'band':'F444W','patch':'x+00_y+00'})
        self.assertIn('START parent=F444W/x+00_y+00',out.getvalue())
        self.assertIn('END parent=F444W/x+00_y+00 status=complete grid_samples=2',out.getvalue())

    def config(self,**kw):return A2744OrdinaryConfig(2.,**kw)
    def measurement(self,snr):return dict(snr=np.array(snr,float),trusted=np.ones(len(snr),bool))
    def test_each_band_uses_its_own_photometry(self):
        t=Table(dict(f_f115w=[10.,10.,10.],e_f115w=[1.,0.,np.nan],
                     f_f444w=[1e4,1e4,1e4],e_f444w=[1.,1.,1.]))
        blue,_=band_photometry(t,'F115W');red,_=band_photometry(t,'F444W')
        self.assertAlmostEqual(blue[0],26.4);self.assertAlmostEqual(red[0],18.9)
        self.assertTrue(np.isnan(blue[1:]).all())
        self.assertTrue(np.isnan(band_photometry(t,'F200W')[0]).all())

    def test_union_gate_does_not_double_count_shared_invalid_pixels(self):
        from preprocessing.utils.inputs import TileSpec
        from preprocessing.utils.large_sources import valid_tile_specs
        observed=np.ones((10,10),bool);observed[0]=False
        coverage=observed.copy();spec=[TileSpec('one',0,0,10)]
        union=training_coverage(observed,coverage)
        self.assertEqual(len(valid_tile_specs(spec,union,(0,0),.1)),1)
        coverage[1,0]=False
        self.assertEqual(len(valid_tile_specs(spec,training_coverage(observed,coverage),(0,0),.1)),0)
        self.assertEqual(len(valid_tile_specs(spec,observed,(0,0),.1)),1)

    def test_wcs_projection_units_labels_and_no_band_fallback(self):
        import tempfile
        from pathlib import Path
        from astropy.io import fits
        w=WCS(naxis=2);w.wcs.crpix=[4,4];w.wcs.crval=[3.58,-30.4]
        w.wcs.cdelt=[-.04/3600,.04/3600];w.wcs.ctype=['RA---TAN','DEC--TAN'];h=w.to_header()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);det=np.ones((8,8),np.float32);det[3,3]=np.nan
            fits.writeto(root/'det.fits',det,h)
            mask=coverage_on_grid(root/'det.fits',h,(8,8),cache_root=root/'cache')
            self.assertFalse(mask[3,3]);self.assertEqual(int(mask.sum()),63)
            np.testing.assert_array_equal(mask,coverage_on_grid(root/'det.fits',h,(8,8),cache_root=root/'cache'))
            seg=np.zeros((8,8),np.int32);seg[2:5,2:5]=700
            fits.writeto(root/'seg.fits',seg,h)
            hp=h.copy();hp['CRPIX1']+=.3
            labels,_=project_product(root/'seg.fits',hp,(8,8),mode='labels')
            self.assertEqual(set(np.unique(labels)),{0,700})
            mh=h.copy();mh['BUNIT']='10.0*nanoJansky'
            model=root/'uncover_f115w_bcgs_models.fits.gz';fits.writeto(model,np.ones((8,8),np.float64),mh)
            projected,_=project_product(model,h,(8,8),mode='surface')
            expected=1e-8/((.04/3600*np.pi/180)**2)/1e6
            self.assertAlmostEqual(float(projected[3,3]),expected,places=6)
            self.assertEqual(model_for_band(root,'F115W'),model)
            with self.assertRaises(FileNotFoundError):model_for_band(root,'F444W')
            partial=root/'uncover_f444w_bcgs_models.fits'
            fits.writeto(partial,np.ones((8,8)),mh)
            with partial.open('r+b') as stream:stream.truncate(partial.stat().st_size-100)
            with self.assertRaises(ValueError):model_for_band(root,'F444W')

    def test_band_model_only_defines_risk_and_never_changes_snr_reference(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from pathlib import Path
        import tempfile
        from preprocessing.abell_zarr import uncover_labels_for_window
        from preprocessing.utils.image import hsc_surface_brightness_factor
        from preprocessing.image_processing import prepare_image
        w=WCS(naxis=2);w.wcs.crpix=[8,8];w.wcs.crval=[3.58,-30.4]
        w.wcs.cdelt=[-.03/3600,.03/3600];w.wcs.ctype=['RA---TAN','DEC--TAN']
        h=w.to_header();h['BUNIT']='MJy/sr';image=np.full((16,16),3.,np.float32)
        cov=np.ones(image.shape,bool);seg=np.zeros(image.shape,np.int32)
        mh=h.copy();mh['FILTER']='F115W'
        factor=hsc_surface_brightness_factor({'BUNIT':'MJy/sr'})[0]
        with tempfile.TemporaryDirectory() as td:
            args=SimpleNamespace(output_root=td,uncover_detection='det',uncover_segmentation='seg',
                uncover_model_root='models',uncover_risk_sigma=3.,uncover_no_kron_snr=3.,
                uncover_risk_no_kron_snr=5.,uncover_reject_untrusted=True,uncover_risk_bright_mag=22.)
            with patch('preprocessing.utils.uncover_labels.coverage_on_grid',return_value=cov), \
                 patch('preprocessing.utils.uncover_labels.project_product',side_effect=[(seg,h),(np.ones(image.shape),mh)]), \
                 patch('preprocessing.utils.uncover_labels.model_for_band',return_value=Path('f115w_model.fits')) as model, \
                 patch('preprocessing.abell_zarr.ensure_background') as background, \
                 patch('utils.source_snr.measure_local_aperture_snr') as measure, \
                 patch('preprocessing.utils.uncover_labels.classify_reference',return_value=(object(),{})) as classify:
                _,summary,_=uncover_labels_for_window(args,image,h,Table(),Table(),cov,'F115W',cov,prepare_image(image,header=h),h,Path(td))
                model.assert_called_once_with('models','F115W')
                np.testing.assert_array_equal(classify.call_args.args[0],image)
                self.assertIs(classify.call_args.args[4],cov)
                np.testing.assert_allclose(classify.call_args.kwargs['snr_risk_model'],factor)
                self.assertNotIn('aperture_measurement',classify.call_args.kwargs)
                self.assertEqual(summary['model_subtraction'],'none')
                background.assert_not_called();measure.assert_not_called()

    def test_raw_full_measurement_path_keeps_bright_star_exclusions(self):
        from unittest.mock import patch
        from preprocessing.utils.uncover_labels import classify_reference
        from preprocessing.utils.image import hsc_surface_brightness_factor
        from preprocessing.image_processing import prepare_image
        class ProbeComplete(Exception):pass
        w=WCS(naxis=2);w.wcs.crpix=[8,8];w.wcs.crval=[3.58,-30.4]
        w.wcs.cdelt=[-.03/3600,.03/3600];w.wcs.ctype=['RA---TAN','DEC--TAN']
        h=w.to_header();h['BUNIT']='MJy/sr'
        image=np.random.default_rng(3).normal(size=(16,16)).astype(np.float32)
        ra,dec=w.pixel_to_world_values([8.],[8.])
        tab=Table(dict(id=[1],x=[8.],y=[8.],ra=ra,dec=dec,theta_J2000=[0.],
            kron_radius=[2.],a_image=[3.],b_image=[2.],flag_artifact=[0],flag_nearbcg=[0],
            **{'fauto_f277w+f356w+f444w':[10.],'faper_f277w+f356w+f444w':[8.],
               'eaper_f277w+f356w+f444w':[1.]}))
        sky=np.ones(image.shape,bool);bright=np.zeros(image.shape,bool);bright[2,2]=True
        stars=np.zeros(image.shape,bool);stars[3,3]=True
        for amplitude in [0.,1e10]:
            with patch('preprocessing.image_processing.build_bright_components',return_value=(bright,np.zeros(image.shape,int))), \
                 patch('preprocessing.bright_label.project_gaia_rows',return_value=Table()), \
                 patch('preprocessing.utils.abell_labels.star_footprint',return_value=stars), \
                 patch('preprocessing.utils.uncover_labels.classify_ordinary',side_effect=ProbeComplete) as classify:
                with self.assertRaises(ProbeComplete):
                    classify_reference(image,h,tab,Table(),sky,'F115W',native_header=h,
                        bright_image=prepare_image(image,header=h),bright_header=h,
                        snr_risk_model=np.full(image.shape,amplitude))
                kw=classify.call_args.kwargs
                np.testing.assert_allclose(kw['image'],image*hsc_surface_brightness_factor(h)[0])
                np.testing.assert_array_equal(kw['excluded_mask'],bright|stars)
                self.assertIs(kw['sky_mask'],sky)
                self.assertIsNone(kw['aperture_measurement'])
                self.assertEqual(bool(kw['snr_contaminated_mask'].all()),amplitude>0)

    def test_native_flags_and_missing_kron(self):
        d,t,has=make_input(6);has[:3]=False
        for a in (d.geometry.major,d.geometry.minor,d.geometry.theta,d.geometry.area):a[:3]=np.nan
        t['flag_nophot'][0]=1;t['flag_artifact'][1]=1;t['flag_nearbcg'][3]=1;t['flag_lowsnr'][4]=1
        d.snr[4]=1.
        r=classify_ordinary(d,t,has,config=self.config(),aperture_measurement=self.measurement([10]*6))
        np.testing.assert_array_equal(r.labels.source_class,[C.DROPPED,C.ORDINARY_IGNORE,C.STRICT_CENTER_ONLY,C.ORDINARY_IGNORE,C.CLEAN,C.CLEAN])
        self.assertTrue(np.isnan(d.geometry.major[:3]).all())
    def test_remeasured_snr_5_7_overrides_native_low_flag(self):
        d,t,has=make_input(5);t['flag_lowsnr'][:3]=1;d.snr[:3]=1.
        r=classify_ordinary(d,t,has,config=self.config(),aperture_measurement=self.measurement([4.9,5.,8.,6.,7.]))
        np.testing.assert_array_equal(r.labels.source_class,[C.ORDINARY_IGNORE,C.WEAK_SHAPE,C.CLEAN,C.WEAK_SHAPE,C.CLEAN])
        self.assertFalse(r.diagnostics['catalog_snr_applied'])

    def test_custom_uncover_cutoffs_preserve_missing_kron_and_native_bad_flags(self):
        d,t,has=make_input(6);has[3:5]=False;t['flag_artifact'][5]=1
        t['flag_lowsnr'][:]=1
        r=classify_ordinary(d,t,has,config=self.config(),
            aperture_measurement=self.measurement([5.99,6.,9.,5.99,6.,100.]),
            snr_config=UncoverSnrConfig(thresholds=SnrThresholds(6,9)))
        np.testing.assert_array_equal(r.labels.source_class,
            [C.ORDINARY_IGNORE,C.WEAK_SHAPE,C.CLEAN,C.ORDINARY_IGNORE,C.STRICT_CENTER_ONLY,C.ORDINARY_IGNORE])
        np.testing.assert_array_equal(r.diagnostics['no_kron_snr_threshold'],np.full(6,6.))

    def test_no_kron_3_and_5_thresholds_and_untrusted_region(self):
        d,t,has=make_input(6);has[:]=False
        for a in (d.geometry.major,d.geometry.minor,d.geometry.theta,d.geometry.area):a[:]=np.nan
        m=self.measurement([2.99,3.,4.99,5.,10.,4.])
        m['trusted'][4]=False
        risk=np.zeros((160,300),bool);risk[:,100:240]=True
        r=classify_ordinary(d,t,has,config=self.config(),aperture_measurement=m,
            snr_contaminated_mask=risk,snr_config=UncoverSnrConfig(thresholds=SnrThresholds(3,5)))
        np.testing.assert_array_equal(r.labels.source_class,[C.ORDINARY_IGNORE,C.STRICT_CENTER_ONLY,
            C.ORDINARY_IGNORE,C.STRICT_CENTER_ONLY,C.ORDINARY_IGNORE,C.STRICT_CENTER_ONLY])
        r=classify_ordinary(d,t,has,config=self.config(),aperture_measurement=m,
            snr_config=UncoverSnrConfig(no_kron_min_snr=5.))
        self.assertTrue(np.all(r.labels.source_class[[0,1,2,5]]==C.ORDINARY_IGNORE))
        self.assertEqual(r.labels.source_class[3],C.STRICT_CENTER_ONLY)

    def test_measure_all_centers_including_ignored_and_dropped(self):
        from unittest.mock import patch
        d,t,has=make_input(4);has[0]=False;t['flag_nophot'][0]=1;t['flag_artifact'][1]=1
        d.snr[:]=1.;d.mag[:]=[25,25,18,28]
        m=self.measurement([10.]*4)
        with patch('utils.source_snr.measure_local_aperture_snr',return_value=m) as measure:
            r=classify_ordinary(d,t,has,config=self.config(),image=np.ones((200,256)))
        np.testing.assert_array_equal(measure.call_args.args[1],d.geometry.x)
        np.testing.assert_array_equal(measure.call_args.args[2],d.geometry.y)
        np.testing.assert_array_equal(r.labels.source_class,[C.DROPPED,C.ORDINARY_IGNORE,C.CLEAN,C.CLEAN])

    def test_measured_low_snr_applies_even_when_untrusted(self):
        d,t,has=make_input(3);m=self.measurement([2.5,6.,np.nan]);m['trusted'][:]=False
        r=classify_ordinary(d,t,has,config=self.config(),aperture_measurement=m)
        np.testing.assert_array_equal(r.labels.source_class,[C.ORDINARY_IGNORE,C.WEAK_SHAPE,C.ORDINARY_IGNORE])

    def test_bright_inband_exemption_only_bypasses_risk_untrusted_gate(self):
        d,t,has=make_input(7);has[2]=False;t['flag_artifact'][3]=1
        m=self.measurement([100.,100.,100.,100.,2.,6.,100.]);m['trusted'][:]=False
        risk=np.ones((180,400),bool)
        mag=np.array([19.5,22.,19.,19.,19.,19.,np.nan])
        r=classify_ordinary(d,t,has,config=self.config(),aperture_measurement=m,
            snr_contaminated_mask=risk,band_magnitude=mag)
        np.testing.assert_array_equal(r.labels.source_class,[C.CLEAN,C.ORDINARY_IGNORE,
            C.ORDINARY_IGNORE,C.ORDINARY_IGNORE,C.ORDINARY_IGNORE,C.WEAK_SHAPE,C.ORDINARY_IGNORE])
        np.testing.assert_array_equal(r.diagnostics['snr_risk_bright_exempt'],[True,False,False,False,False,True,False])
        off=classify_ordinary(d,t,has,config=self.config(),aperture_measurement=m,
            snr_contaminated_mask=risk,band_magnitude=mag,
            snr_config=UncoverSnrConfig(risk_bright_mag_max=None))
        self.assertTrue(np.all(off.labels.source_class==C.ORDINARY_IGNORE))
    def test_ignore_sources_do_not_demote_containing_parent(self):
        d,t,has=make_input(2);g=d.geometry;g.x[:]=[100,101];g.major[:]=[20,3];g.minor[:]=[20,3];g.area[:]=np.pi*g.major*g.minor
        t['flag_artifact'][1]=1
        r=classify_ordinary(d,t,has,config=self.config(),aperture_measurement=self.measurement([10,10]))
        self.assertEqual(r.labels.source_class[0],C.CLEAN)
        self.assertEqual(r.labels.source_class[1],C.ORDINARY_IGNORE)
    def test_original_psf_and_close_brighter_rules_remain(self):
        d,t,has=make_input(3);g=d.geometry;g.major[0]=1.;g.area[0]=np.pi*4
        g.x[1:]=[120.,121.];d.mag[1:]=[20.,25.]
        r=classify_ordinary(d,t,has,config=self.config(),aperture_measurement=self.measurement([10]*3))
        self.assertEqual(r.labels.source_class[0],C.ORDINARY_IGNORE)
        self.assertEqual(r.labels.source_class[1],C.ORDINARY_IGNORE)
        self.assertEqual(r.labels.source_class[2],C.CLEAN)
    def test_native_image_theta_not_sky_pa(self):
        w=WCS(naxis=2);w.wcs.crpix=[100,100];w.wcs.crval=[3.58,-30.4];w.wcs.cdelt=[-.04/3600,.04/3600];w.wcs.ctype=['RA---TAN','DEC--TAN']
        h=w.to_header();x,y=np.array([100.,130.]),np.array([100.,130.]);ra,dec=w.pixel_to_world_values(x,y)
        t=Table(dict(id=[1,2],x=x,y=y,ra=ra,dec=dec,theta_J2000=[.65,-.2],kron_radius=[2.,np.nan],a_image=[5.,4.],b_image=[2.,2.],
            **{'fauto_f277w+f356w+f444w':[10.,2.],'faper_f277w+f356w+f444w':[8.,1.],'eaper_f277w+f356w+f444w':[1.,1.]},flag_artifact=[0,0],flag_nearbcg=[0,0]))
        tab,d,has=adapt_catalog(t,h,(256,256),h)
        self.assertLess(abs((d.geometry.theta[0]-.65+np.pi/2)%np.pi-np.pi/2),1e-5)
        np.testing.assert_allclose([d.geometry.major[0],d.geometry.minor[0]],[10.,4.],rtol=1e-5)
        self.assertFalse(has[1]);self.assertTrue(np.isnan(d.geometry.major[1]))
        self.assertAlmostEqual(27-2.5*np.log10(tab['FLUX_AUTO'][0]),d.mag[0])

if __name__=='__main__':unittest.main()
