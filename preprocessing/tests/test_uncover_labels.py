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
    band_photometry,project_product,coverage_on_grid,training_coverage,model_for_band,kron_segmentation_quality)
from preprocessing.snr import SnrThresholds


def make_input(n):
    x=np.arange(n,dtype=float)*40+50
    g=EllipseGeometry(x,np.full(n,100.),np.full(n,5.),np.full(n,4.),np.zeros(n),np.full(n,20*np.pi))
    d=OrdinaryInput(g,np.full(n,25.),snr=np.full(n,10.),flags=np.zeros(n,int))
    t=Table({k:np.zeros(n,int) for k in ['flag_nophot','flag_artifact','flag_nearbcg','flag_lowsnr']})
    return d,t,np.ones(n,bool)


class UncoverTests(unittest.TestCase):
    def test_halo_no_kron_veto_preserves_gaia_geometry_and_existing_rejections(self):
        from types import SimpleNamespace
        from preprocessing.utils.uncover_labels import ignore_native_halo_without_kron
        table=Table(dict(kron_radius=[np.nan,2.,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
            snr_contaminated_region=[True,True,False,True,True,True,True,True],
            gaia_psf_model=[False,False,False,True,False,False,False,False],
            classification_reason=['original']*8))
        labels=SimpleNamespace(table=table,source_ids=np.array([1,2,3,4,-5,6,7,8]),
            label_classes=np.array([C.STRICT_CENTER_ONLY,C.CLEAN,C.STRICT_CENTER_ONLY,C.WEAK_SHAPE,
                                    C.STRICT_CENTER_ONLY,C.DROPPED,C.ORDINARY_IGNORE,C.STRICT_IGNORE]))
        summary=ignore_native_halo_without_kron(labels)
        self.assertEqual(summary['ignored'],1)
        np.testing.assert_array_equal(labels.label_classes,[C.ORDINARY_IGNORE,C.CLEAN,C.STRICT_CENTER_ONLY,
            C.WEAK_SHAPE,C.STRICT_CENTER_ONLY,C.DROPPED,C.ORDINARY_IGNORE,C.STRICT_IGNORE])
        self.assertEqual(table['class_before_halo_no_kron'][0],C.STRICT_CENTER_ONLY)
        self.assertEqual(table['classification_reason'][0],'uncover_halo_no_kron_ignore_before_blue')

    def test_halo_veto_before_blue_allows_independent_center_insertion(self):
        from types import SimpleNamespace
        from preprocessing.utils.uncover_labels import ignore_native_halo_without_kron,add_blue_compact_centers
        def labels():
            return SimpleNamespace(table=Table(dict(kron_radius=[np.nan],snr_contaminated_region=[True],
                classification_reason=['uncover_no_kron'])),source_ids=np.array([123]),
                label_classes=np.array([C.STRICT_CENTER_ONLY]),geom_x=np.array([50.]),geom_y=np.array([50.]),
                dense=np.zeros((100,100),np.uint8),strict_x=np.array([]),strict_y=np.array([]),
                strict_ids=np.array([],np.int64),strict_is_gaia=np.array([],bool))
        candidates=Table(dict(debug_id=[8],blue_candidate=[True],x=[51.],y=[50.],
            F115W_snr=[6.],F150W_snr=[6.],F200W_snr=[6.]))
        old=labels();old_result=add_blue_compact_centers(old,candidates,'F115W',scope='test')
        ignore_native_halo_without_kron(old)
        self.assertEqual(old_result['recentered'],1);self.assertEqual(len(old.strict_ids),0)
        new=labels();ignore_native_halo_without_kron(new)
        new_result=add_blue_compact_centers(new,candidates,'F115W',scope='test')
        self.assertEqual(new.label_classes[0],C.ORDINARY_IGNORE)
        self.assertEqual(new_result['added'],1);self.assertEqual(new_result['recentered'],0)
        self.assertEqual(new.strict_x[0],51.)
        # A red-band candidate still needs the original fixed-PSF threshold.
        red=labels();ignore_native_halo_without_kron(red)
        failed=add_blue_compact_centers(red,candidates,'F444W',core_snr=np.array([3.9]),scope='test')
        self.assertEqual(failed['added'],0)

    def test_blue_recovery_reuses_only_exact_predecessor_successes(self):
        from copy import deepcopy
        from preprocessing.utils.abell_blue_detection import compatible_signature,COMPATIBLE_CODE_HASHES
        old=dict(code=next(iter(COMPATIBLE_CODE_HASHES)),config={'snr':4},assets=['same'],job={'name':'x+00_y-01'})
        new={**old,'code':'new_recovery_only'}
        self.assertTrue(compatible_signature(old,new))
        for key,value in [('assets',['different']),('config',{'snr':5}),('job',{'name':'another'})]:
            changed=deepcopy(new);changed[key]=value
            self.assertFalse(compatible_signature(old,changed))
        self.assertFalse(compatible_signature({**old,'code':'unknown'},new))

    def test_blue_tiled_detection_recurses_and_preserves_boundary_centers(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from astropy.io import fits
        from preprocessing.utils.abell_blue_detection import tiled_sextractor,_sextractor,SExtractorDeblendOverflow
        yy,xx=np.mgrid[:256,:256];image=np.zeros((256,256),np.float32)
        points=np.array([[127.7,80.3],[190.2,128.4],[60.1,193.6]])
        for x,y in points:image+=50*np.exp(-((xx-x)**2+(yy-y)**2)/8)
        def split_large(source,*args,**kwargs):
            if max(fits.getdata(source).shape)>140:raise SExtractorDeblendOverflow('forced big-parent case')
            return _sextractor(source,*args,**kwargs)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);fits.writeto(p/'input.fits',image)
            with patch('preprocessing.utils.abell_blue_detection._sextractor',side_effect=split_large):
                t=tiled_sextractor(p/'input.fits',p/'tiles',5,core_size=128,halo=16,min_core=64)
            self.assertEqual(len(t),3)
            xy=np.c_[t['XWIN_IMAGE'],t['YWIN_IMAGE']]-1
            for point in points:self.assertLess(float(np.linalg.norm(xy-point,axis=1).min()),.3)

    def test_blue_sextractor_memory_options_and_silent_overflow_rejection(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from preprocessing.utils.abell_blue_detection import _sextractor
        def overflow(cmd,**kwargs):
            Path(cmd[cmd.index('-CATALOG_NAME')+1]).write_text('')
            kwargs['stderr'].write('WARNING: Pixel stack overflow, some objects may be merged\n')
        with tempfile.TemporaryDirectory() as td:
            with patch('preprocessing.utils.abell_blue_detection.subprocess.check_output',return_value=''):
                with patch('preprocessing.utils.abell_blue_detection.subprocess.run',side_effect=overflow) as call:
                    with self.assertRaisesRegex(RuntimeError,'stack failure'):
                        _sextractor(Path(td)/'input.fits',Path(td)/'sex',2,
                            memory=dict(MEMORY_PIXSTACK=64000000,MEMORY_OBJSTACK=60000,MEMORY_BUFSIZE=2048))
                    cmd=call.call_args.args[0]
                    for option,value in [('MEMORY_PIXSTACK','64000000'),('MEMORY_OBJSTACK','60000'),('MEMORY_BUFSIZE','2048')]:
                        self.assertEqual(cmd[cmd.index('-'+option)+1],value)

    def test_blue_real_sextractor_defaults_detect_a_compact_peak(self):
        import tempfile
        from pathlib import Path
        from astropy.io import fits
        from preprocessing.utils.abell_blue_detection import _sextractor
        yy,xx=np.mgrid[:128,:128]
        image=np.random.default_rng(11).normal(0,1,(128,128))+50*np.exp(-((xx-64)**2+(yy-64)**2)/8)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);fits.writeto(p/'input.fits',image.astype(np.float32))
            t=_sextractor(p/'input.fits',p/'sex',5)
            distance=np.hypot(np.asarray(t['X_IMAGE'])-65,np.asarray(t['Y_IMAGE'])-65)
            self.assertLess(float(distance.min()),1.)

    def test_four_blue_confirmation_and_unconfirmed_band_psf_gate(self):
        from types import SimpleNamespace
        from preprocessing.utils.uncover_labels import add_blue_compact_centers,direct_blue_detection
        t=Table(dict(debug_id=[1,2],blue_candidate=[True,True],x=[30.,70.],y=[40.,40.],
            F090W_snr=[6.,6.],F115W_snr=[2.,6.],F150W_snr=[6.,6.],F200W_snr=[6.,6.],
            F090W_detected=[True,True],F115W_detected=[False,True],F150W_detected=[True,True],F200W_detected=[True,True]))
        t.meta.update(DETBANDS='F090W,F115W,F150W,F200W',MINBANDS=3)
        def label():return SimpleNamespace(dense=np.zeros((100,100),np.uint8),label_classes=np.array([],np.int16),
            geom_x=np.array([]),geom_y=np.array([]),source_ids=np.array([],np.int64),
            strict_x=np.array([]),strict_y=np.array([]),strict_ids=np.array([],np.int64),strict_is_gaia=np.array([],bool))
        self.assertEqual(direct_blue_detection(t,'F115W').tolist(),[False,True])
        self.assertFalse(direct_blue_detection(t,'F070W').any())
        result=add_blue_compact_centers(label(),t,'F115W',core_snr=[3.99,np.nan])
        self.assertEqual(result['added'],1);self.assertEqual(result['status'][0],'band_psf_snr_below_threshold')
        self.assertEqual(add_blue_compact_centers(label(),t,'F070W',core_snr=[4.,3.99])['added'],1)

    def test_blue_raw_quadratic_fit_and_analytic_jacobian(self):
        from preprocessing.utils.abell_blue_detection import gaussian_model_jacobian,fit_blue_core
        yy,xx=np.mgrid[-8:9,-8:9];xx=xx.ravel().astype(float);yy=yy.ravel().astype(float)
        bg=np.c_[np.ones(len(xx)),xx/8,yy/8,(xx/8)**2,xx*yy/64,(yy/8)**2]
        p=np.array([30.,.3,-.4,1.2,.4,.1,.2,.03,.02,-.01])
        model,jac=gaussian_model_jacobian(p,xx,yy,bg)
        for j in range(len(p)):
            step=np.zeros(len(p));step[j]=1e-5
            numeric=(gaussian_model_jacobian(p+step,xx,yy,bg)[0]-gaussian_model_jacobian(p-step,xx,yy,bg)[0])/2e-5
            np.testing.assert_allclose(jac[:,j],numeric,rtol=2e-5,atol=2e-7)
        yy,xx=np.mgrid[:64,:64];dx=(xx-32).ravel();dy=(yy-32).ravel()
        bg=np.c_[np.ones(len(dx)),dx/8,dy/8,(dx/8)**2,dx*dy/64,(dy/8)**2]
        image=gaussian_model_jacobian(p,dx,dy,bg)[0].reshape(64,64)
        result=fit_blue_core(image,np.full(image.shape,.02),32.,32.)
        self.assertTrue(result['success']);self.assertAlmostEqual(result['x'],32.3,places=4)
        self.assertAlmostEqual(result['y'],31.6,places=4);self.assertGreater(result['snr'],4)

    def test_parallel_blue_fits_match_serial_results(self):
        from preprocessing.utils.abell_blue_detection import gaussian_model_jacobian,refine_seeds,BANDS
        yy,xx=np.mgrid[:64,:64];dx=(xx-32).ravel();dy=(yy-32).ravel()
        bg=np.c_[np.ones(len(dx)),dx/8,dy/8,(dx/8)**2,dx*dy/64,(dy/8)**2]
        p=np.array([30.,.3,-.4,1.2,.4,.1,.2,.03,.02,-.01])
        image=gaussian_model_jacobian(p,dx,dy,bg)[0].reshape(64,64)
        raw={b:image for b in BANDS};rms={b:np.full(image.shape,.02) for b in BANDS}
        seeds=[dict(x_init=32.,y_init=32.,isoarea=10,a_image=1.2,b_image=1.2)]
        w=WCS(naxis=2);w.wcs.crpix=[1,1];w.wcs.crval=[3.58,-30.4]
        w.wcs.cdelt=[-.03/3600,.03/3600];w.wcs.ctype=['RA---TAN','DEC--TAN']
        one=refine_seeds(seeds,raw,rms,np.zeros_like(image),w.to_header(),'test',fit_workers=1)
        two=refine_seeds(seeds,raw,rms,np.zeros_like(image),w.to_header(),'test',fit_workers=2)
        for key in ('x','y','F090W_snr','F200W_snr'):np.testing.assert_allclose(one[key],two[key])
        self.assertTrue(two['blue_candidate'][0])

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
                ' Path(os.environ["CAPTURE_ARGV"]).write_text(json.dumps(dict(workers=a.workers,use_uncover=a.use_uncover,background_root=a.background_root,stage=a.abell_stage,no_kron=a.uncover_no_kron_snr,risk_no_kron=a.uncover_risk_no_kron_snr,snr_ignore=a.snr_ignore_threshold,snr_weak=a.snr_weak_threshold,kron_other=a.uncover_kron_other_fraction,blue=a.uncover_blue_catalog,dedup=a.uncover_compact_dedup_pixels,native_psf=a.uncover_native_psf_review,psf_snr=a.uncover_native_psf_snr)))\n')
            stub.chmod(0o755)
            env={**os.environ,'JWST_PYTHON':str(stub),'OUTPUT_ROOT':str(root/'out'),
                 'CAPTURE_ARGV':str(capture),'ZARR_WORKERS':'64','ABELL_STAGE':'zarr','USE_UNCOVER':'1',
                 'BACKGROUND_ROOT':'/test/background with spaces','UNCOVER_NO_KRON_SNR':'3','UNCOVER_RISK_NO_KRON_SNR':'5',
                 'SNR_IGNORE_THRESHOLD':'6','SNR_WEAK_THRESHOLD':'9','UNCOVER_KRON_OTHER_FRACTION':'0.24'}
            p=subprocess.run(['bash',str(repo/'preprocessing/run_abell_preprocessing.sh')],env=env,
                             text=True,capture_output=True,timeout=30)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)
            actual=json.loads(capture.read_text())
            self.assertEqual(actual,dict(workers=64,use_uncover=True,background_root=env['BACKGROUND_ROOT'],stage='zarr',no_kron=3.,risk_no_kron=5.,snr_ignore=6.,snr_weak=9.,kron_other=.24,
                blue=str(repo/'preprocessing/assets/abell_verified_blue_centers_v1.fits'),dedup=2.,native_psf=True,psf_snr=4.))
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
                uncover_risk_no_kron_snr=5.,uncover_reject_untrusted=True,uncover_risk_bright_mag=22.,
                uncover_native_psf_review=False)
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

    def test_bright_inband_exception_caps_at_weak_and_preserves_other_vetoes(self):
        d,t,has=make_input(7);has[2]=False;t['flag_artifact'][3]=1
        m=self.measurement([100.,100.,100.,100.,2.,6.,100.]);m['trusted'][:]=False
        risk=np.ones((180,400),bool)
        mag=np.array([19.5,22.,19.,19.,19.,19.,np.nan])
        r=classify_ordinary(d,t,has,config=self.config(),aperture_measurement=m,
            snr_contaminated_mask=risk,band_magnitude=mag)
        np.testing.assert_array_equal(r.labels.source_class,[C.WEAK_SHAPE,C.ORDINARY_IGNORE,
            C.ORDINARY_IGNORE,C.ORDINARY_IGNORE,C.ORDINARY_IGNORE,C.WEAK_SHAPE,C.ORDINARY_IGNORE])
        np.testing.assert_array_equal(r.diagnostics['snr_risk_bright_exempt'],[True,False,False,False,False,True,False])
        off=classify_ordinary(d,t,has,config=self.config(),aperture_measurement=m,
            snr_contaminated_mask=risk,band_magnitude=mag,
            snr_config=UncoverSnrConfig(risk_bright_mag_max=None))
        self.assertTrue(np.all(off.labels.source_class==C.ORDINARY_IGNORE))

    def test_background_reliability_changes_risk_classes_but_not_measured_snr(self):
        d,t,has=make_input(7)
        m=self.measurement([100.,100.,100.,100.,4.,100.,100.])
        m['background_trusted']=np.array([False,False,False,False,False,True,False])
        before=m['snr'].copy();has[3]=False;t['flag_artifact'][6]=1
        # Outside risk, the new metadata does not add another hard cut.
        risk=np.ones((160,400),bool);risk[:,125:150]=False
        r=classify_ordinary(d,t,has,config=self.config(),aperture_measurement=m,
            snr_contaminated_mask=risk,band_magnitude=[25.,21.,25.,21.,21.,25.,21.])
        np.testing.assert_array_equal(r.labels.source_class,[C.ORDINARY_IGNORE,C.WEAK_SHAPE,
            C.CLEAN,C.ORDINARY_IGNORE,C.ORDINARY_IGNORE,C.CLEAN,C.ORDINARY_IGNORE])
        np.testing.assert_array_equal(r.diagnostics['aperture_snr'],before)
        self.assertTrue(r.diagnostics['aperture_snr_trusted'].all())
        self.assertFalse(r.diagnostics['aperture_snr_effective_trusted'][0])

    def test_global_fallback_metadata_preserves_numeric_measurement(self):
        from utils.source_snr import block_background,measure_local_aperture_snr
        image=np.random.default_rng(27).normal(0,.01,(256,256)).astype(np.float32)
        image[:128,:128]+=.15
        sky=np.ones(image.shape,bool);sky[:128,:128]=False;sky[10:30,10:30]=True
        legacy=block_background(image,sky,box=128)
        audited=block_background(image,sky,box=128,return_reliability=True)
        for expected,actual in zip(legacy,audited[:3]):np.testing.assert_array_equal(expected,actual)
        self.assertFalse(audited[3][64,64]);self.assertTrue(audited[3][192,192])
        measured=measure_local_aperture_snr(image,[64.,192.],[64.,192.],radius=2,sky_mask=sky)
        np.testing.assert_array_equal(measured['trusted'],[True,True])
        np.testing.assert_array_equal(measured['background_trusted'],[False,True])
        yy,xx=np.mgrid[:256,:256];ap=(xx-64)**2+(yy-64)**2<=4
        flux=np.sum((image-legacy[0])[ap],dtype=np.float64)
        self.assertEqual(measured['flux'][0],np.float32(flux))
        self.assertAlmostEqual(measured['snr'][0],flux/measured['aperture_sigma'][64,64],places=4)

    def test_kron_denominator_is_entire_area_not_occupied_segmentation(self):
        d,_,_=make_input(4);g=d.geometry;g.x[:]=15;g.y[:]=15
        g.major[:]=5;g.minor[:]=5;g.area[:]=25*np.pi
        seg=np.zeros((32,32),np.int32)
        yy,xx=np.mgrid[:32,:32];pix=np.flatnonzero((xx-15)**2+(yy-15)**2<=25)
        seg.flat[pix[:20]]=2;seg.flat[pix[20:25]]=3;seg.flat[pix[25:30]]=99
        seg.flat[pix[30]]=1
        q=kron_segmentation_quality(g,[1,2,3,4],[C.CLEAN,C.CLEAN,C.ORDINARY_IGNORE,C.WEAK_SHAPE],seg)
        self.assertEqual(q['kron_pixel_count'][0],81)
        self.assertEqual(q['kron_other_seg_pixel_count'][0],20)
        self.assertAlmostEqual(q['kron_other_seg_fraction'][0],20/81)
        self.assertFalse(q['kron_other_seg_bad'][0]) # segmented-only fraction would be 20/21
        seg.flat[pix[20:25]]=2
        q=kron_segmentation_quality(g,[1,2,3,4],[C.CLEAN,C.CLEAN,C.ORDINARY_IGNORE,C.WEAK_SHAPE],seg)
        self.assertTrue(q['kron_other_seg_bad'][0]) # 25/81 > .30
        boundary=kron_segmentation_quality(g,[1,2,3,4],[C.CLEAN,C.CLEAN,C.ORDINARY_IGNORE,C.WEAK_SHAPE],seg,other_fraction_max=25/81)
        self.assertFalse(boundary['kron_other_seg_bad'][0]) # strict greater-than
        order=[3,1,0,2]
        gg=EllipseGeometry(*(getattr(g,k)[order] for k in ('x','y','major','minor','theta','area')))
        shuffled=kron_segmentation_quality(gg,np.array([1,2,3,4])[order],np.array([1,1,3,2])[order],seg)
        np.testing.assert_array_equal(shuffled['kron_other_seg_bad'][np.argsort(order)],q['kron_other_seg_bad'])

    def test_kron_off_parent_pixels_remain_in_denominator(self):
        d,_,_=make_input(2);g=d.geometry;g.x[:]=0;g.y[:]=10
        g.major[:]=5;g.minor[:]=5;g.area[:]=25*np.pi
        q=kron_segmentation_quality(g,[1,2],[C.CLEAN,C.CLEAN],np.full((24,24),2,np.int32))
        self.assertEqual(q['kron_pixel_count'][0],81)
        self.assertEqual(q['kron_visible_pixel_count'][0],46)
        self.assertEqual(q['kron_other_seg_pixel_count'][0],46)
        self.assertAlmostEqual(q['kron_other_seg_fraction'][0],46/81)
        self.assertEqual(q['kron_other_seg_fraction'][1],0.) # own pixels never contaminate

    def test_blue_compact_centers_preserve_shapes_and_require_red_core_snr(self):
        from types import SimpleNamespace
        from preprocessing.utils.uncover_labels import add_blue_compact_centers
        def label():
            return SimpleNamespace(dense=np.zeros((300,300),np.uint8),
                label_classes=np.array([C.CLEAN,C.WEAK_SHAPE,C.STRICT_CENTER_ONLY]),
                geom_x=np.array([50.,90.,130.]),geom_y=np.full(3,100.),
                source_ids=np.array([1,2,3],np.int64),
                strict_x=np.array([220.]),strict_y=np.array([100.]),strict_ids=np.array([-123],np.int64),
                strict_is_gaia=np.array([True]))
        candidates=Table(dict(x=[50.,90.,130.,175.,195.,220.,250.],y=np.full(7,100.),
            debug_id=np.arange(7),blue_candidate=[True,True,True,True,True,True,False]))
        labels=label();before=labels.label_classes.copy()
        r=add_blue_compact_centers(labels,candidates,'F444W',core_snr=[100,100,100,3.99,4,100,100],scope='test')
        np.testing.assert_array_equal(labels.label_classes,before)
        self.assertEqual(r['added'],1);self.assertEqual(r['status'][3],'red_psf_snr_below_threshold')
        self.assertEqual(r['status'][4],'added_strict');self.assertEqual(r['status'][6],'not_eligible')
        np.testing.assert_array_equal(labels.strict_x,[220.,195.])
        np.testing.assert_array_equal(labels.strict_is_gaia,[True,False])
        labels=label();blue=add_blue_compact_centers(labels,candidates,'F115W',scope='test')
        self.assertEqual(blue['added'],2)
        candidates['flag_artifact']=np.array([0,0,0,1,0,0,0])
        self.assertEqual(add_blue_compact_centers(label(),candidates,'F115W')['added'],2)

    def test_fixed_psf_core_recovers_signed_flux_with_neighbor_and_curved_sky(self):
        from preprocessing.utils.jwst_star_psf import pixel_integrated_psf,measure_fixed_psf_core
        yy,xx=np.mgrid[:64,:64];x,y=30.2,31.4
        psf=pixel_integrated_psf('F115W',xx-x,yy-y)
        neighbor=pixel_integrated_psf('F115W',xx-34.,yy-33.)
        bg=.1+.0001*xx+.0002*yy+.00001*xx*yy
        for flux in (.7,-.3):
            image=flux*psf+.4*neighbor+bg
            result=measure_fixed_psf_core(image,np.full(image.shape,.01),x,y,'F115W',neighbor_xy=[[34.,33.]])
            self.assertTrue(result['success']);self.assertEqual(result['neighbor_count'],1)
            self.assertAlmostEqual(result['flux'],flux,places=7)
            self.assertEqual(np.sign(result['snr']),np.sign(flux))
            self.assertEqual(result['residual_error_factor'],1.)
        self.assertEqual(measure_fixed_psf_core(image,np.ones(image.shape),2,2,'F115W')['status'],'edge')

    def test_dedup_two_pixels_replaces_old_center_and_preserves_id_and_axes(self):
        from preprocessing.utils.uncover_labels import add_blue_compact_centers
        from types import SimpleNamespace
        labels=SimpleNamespace(dense=np.zeros((100,100),np.uint8),
            table=Table(dict(classification_reason=['old'])),source_ids=np.array([123],np.int64),
            label_classes=np.array([C.CLEAN]),geom_x=np.array([50.]),geom_y=np.array([50.]),
            geom_major=np.array([6.]),geom_minor=np.array([4.]),geom_theta=np.array([.2]),
            strict_x=np.array([]),strict_y=np.array([]),strict_ids=np.array([],np.int64),strict_is_gaia=np.array([],bool))
        candidates=Table(dict(debug_id=[8,9],blue_candidate=[True,True],x=[51.8,52.1],y=[50.,50.],
            F115W_snr=[8.,6.],F150W_snr=[8.,6.],F200W_snr=[8.,6.]))
        result=add_blue_compact_centers(labels,candidates,'F115W',dedup_pixels=2)
        self.assertEqual(result['recentered'],1);self.assertEqual(result['added'],0)
        self.assertEqual(result['status'][1],'duplicate_blue')
        self.assertEqual(result['source_id'][0],123)
        self.assertAlmostEqual(labels.geom_x[0],51.8)
        self.assertEqual(labels.table['center_before_blue_x'][0],50.)
        self.assertEqual(labels.table['center_blue_debug_id'][0],8)
        np.testing.assert_array_equal(labels.geom_major,[6.]);np.testing.assert_array_equal(labels.geom_minor,[4.])
        self.assertEqual(labels.label_classes[0],C.CLEAN)

    def test_native_psf_veto_inclusive_four_is_risk_and_compact_only(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from preprocessing.utils.uncover_labels import refine_uncover_compact_sources
        from preprocessing.region_filling import fill_dense_regions
        from preprocessing.labels import SourceLabels,DenseLabel as D
        from preprocessing.utils.geometry import EllipseGeometry
        n=6;x=np.arange(n)*40.+30.;y=np.full(n,100.);axes=np.full(n,4.)
        classes=np.array([C.CLEAN,C.WEAK_SHAPE,C.STRICT_CENTER_ONLY,C.CLEAN,C.CLEAN,C.WEAK_SHAPE])
        table=Table(dict(a_image=[1.,1.,1.,3.,1.,1.],b_image=[1.,1.,1.,3.,1.,1.],
            snr_contaminated_region=[True,True,True,True,False,True],gaia_psf_model=[False]*5+[True],
            classification_reason=['original']*n,catalog_band_snr=np.full(n,10.),catalog_band_mag=np.full(n,20.)))
        geom=EllipseGeometry(x,y,axes,axes,np.zeros(n),np.pi*axes**2)
        dense=fill_dense_regions(table,SourceLabels(classes,np.full(n,'',object)),(300,300),geometry=geom,background_mask=np.ones((300,300),bool))
        labels=SimpleNamespace(table=table,dense=dense,label_classes=classes,source_ids=np.arange(n),
            geom_x=x,geom_y=y,geom_major=axes,geom_minor=axes,geom_theta=np.zeros(n),
            strict_x=np.array([]),strict_y=np.array([]),strict_ids=np.array([],np.int64),strict_is_gaia=np.array([],bool))
        w=WCS(naxis=2);w.wcs.crpix=[1,1];w.wcs.crval=[3.58,-30.4];w.wcs.cdelt=[-.03/3600,.03/3600];w.wcs.ctype=['RA---TAN','DEC--TAN']
        def fit(image,rms,px,py,band,**kw):
            snr={30.:4.,70.:4.01,110.:np.nan}[px]
            return dict(snr=snr,chi2=.5,success=np.isfinite(snr),status='ok' if np.isfinite(snr) else 'invalid',neighbor_count=0)
        with patch('preprocessing.utils.jwst_star_psf.measure_fixed_psf_core',side_effect=fit):
            summary=refine_uncover_compact_sources(labels,np.ones((300,300)),w.to_header(),'F444W',rms=np.ones((300,300)),noise_factor=1.)
        np.testing.assert_array_equal(labels.label_classes,[C.ORDINARY_IGNORE,C.WEAK_SHAPE,C.ORDINARY_IGNORE,C.CLEAN,C.CLEAN,C.WEAK_SHAPE])
        self.assertEqual(summary['native_psf_review']['ignored'],2)
        self.assertEqual(labels.dense[100,30],D.ORDINARY_IGNORE)
        self.assertEqual(labels.dense[100,70],D.WEAK_SHAPE)
        self.assertFalse(labels.table['native_psf_tested'][3]) # resolved
        self.assertFalse(labels.table['native_psf_tested'][4]) # outside risk
        self.assertFalse(labels.table['native_psf_tested'][5]) # Gaia theoretical model

    def test_recenter_precedes_psf_review_and_cannot_revive_rejected_native(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from preprocessing.utils.uncover_labels import refine_uncover_compact_sources
        w=WCS(naxis=2);w.wcs.crpix=[1,1];w.wcs.crval=[3.58,-30.4];w.wcs.cdelt=[-.03/3600,.03/3600];w.wcs.ctype=['RA---TAN','DEC--TAN']
        for value,expected in [(5.,C.CLEAN),(4.,C.ORDINARY_IGNORE)]:
            labels=SimpleNamespace(table=Table(dict(a_image=[1.],b_image=[1.],snr_contaminated_region=[True],
                classification_reason=['original'],catalog_band_snr=[10.])),dense=np.full((100,100),4,np.uint8),
                label_classes=np.array([C.CLEAN]),source_ids=np.array([123]),geom_x=np.array([50.]),geom_y=np.array([50.]),
                geom_major=np.array([4.]),geom_minor=np.array([4.]),geom_theta=np.array([0.]),
                strict_x=np.array([]),strict_y=np.array([]),strict_ids=np.array([],np.int64),strict_is_gaia=np.array([],bool))
            c=Table(dict(debug_id=[8],blue_candidate=[True],x=[51.8],y=[50.]))
            def fit(im,rms,x,y,band,**kw):
                self.assertAlmostEqual(x,51.8)
                return dict(snr=value,chi2=.5,success=True,status='ok',neighbor_count=0)
            with patch('preprocessing.utils.jwst_star_psf.measure_fixed_psf_core',side_effect=fit):
                refine_uncover_compact_sources(labels,np.ones((100,100)),w.to_header(),'F115W',
                    candidates=c,rms=np.ones((100,100)),noise_factor=1.)
            self.assertEqual(labels.label_classes[0],expected)
            self.assertEqual(len(labels.strict_x),0)
            if value==4:self.assertEqual(labels.compact_center_audit['status'][0],'recentered_native_psf_ignored')

    def test_compact_window_interface_uses_world_coordinates_and_band_provenance(self):
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        import tempfile
        from preprocessing.abell_zarr import uncover_labels_for_window
        from preprocessing.image_processing import prepare_image
        w=WCS(naxis=2);w.wcs.crpix=[8,8];w.wcs.crval=[3.58,-30.4]
        w.wcs.cdelt=[-.03/3600,.03/3600];w.wcs.ctype=['RA---TAN','DEC--TAN']
        h=w.to_header();h['BUNIT']='MJy/sr';h['FILTER']='F444W'
        image=np.ones((16,16),np.float32);sky=np.ones(image.shape,bool)
        ra,dec=w.pixel_to_world_values([8.],[8.])
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);parent='x+00_y+00';audit=root/'audit'/parent;audit.mkdir(parents=True)
            folder=root/'compact'/parent/'F444W';folder.mkdir(parents=True)
            t=Table(dict(ra=ra,dec=dec,x=[10000.],y=[10000.],debug_id=[7],
                blue_candidate=[True],fixed_psf_core_snr=[4.]));t.meta['PSFBAND']='F444W'
            path=folder/'blue_center_selection.fits';t.write(path)
            args=SimpleNamespace(output_root=td,uncover_detection='det',uncover_segmentation='seg',
                uncover_model_root='models',uncover_risk_sigma=3.,uncover_no_kron_snr=3.,
                uncover_risk_no_kron_snr=5.,uncover_reject_untrusted=True,uncover_risk_bright_mag=22.,
                uncover_compact_root=str(root/'compact'),uncover_native_psf_review=False)
            labels=SimpleNamespace(dense=np.zeros(image.shape,np.uint8),label_classes=np.array([],np.int16),
                geom_x=np.array([]),geom_y=np.array([]),strict_x=np.array([]),strict_y=np.array([]),
                strict_ids=np.array([],np.int64),strict_is_gaia=np.array([],bool),table=Table(),
                source_ids=np.array([],np.int64),geom_major=np.array([]),geom_minor=np.array([]),geom_theta=np.array([]))
            with patch('preprocessing.utils.uncover_labels.coverage_on_grid',return_value=sky), \
                 patch('preprocessing.utils.uncover_labels.project_product',side_effect=lambda *a,**k:(np.zeros(image.shape),h)), \
                 patch('preprocessing.utils.uncover_labels.model_for_band',return_value=Path('f444w_model.fits')), \
                 patch('preprocessing.utils.uncover_labels.classify_reference',return_value=(labels,{})):
                out,summary,_=uncover_labels_for_window(args,image,h,Table(),Table(),sky,'F444W',
                    sky,prepare_image(image,header=h),h,audit)
                self.assertEqual(summary['compact_centers']['added'],1)
                np.testing.assert_allclose([out.strict_x[0],out.strict_y[0]],[8,8],atol=1e-5)
                t.meta['PSFBAND']='F115W';t.write(path,overwrite=True)
                with self.assertRaisesRegex(ValueError,'band does not match'):
                    uncover_labels_for_window(args,image,h,Table(),Table(),sky,'F444W',
                        sky,prepare_image(image,header=h),h,audit)

    def test_verified_blue_catalog_projects_into_independent_windows_and_checks_three_bands(self):
        import tempfile
        from pathlib import Path
        from preprocessing.utils.uncover_labels import blue_candidates_on_window
        w=WCS(naxis=2);w.wcs.crpix=[1,1];w.wcs.crval=[3.58,-30.4]
        w.wcs.cdelt=[-.03/3600,.03/3600];w.wcs.ctype=['RA---TAN','DEC--TAN']
        ra,dec=w.pixel_to_world_values([25.,400.],[35.,400.])
        t=Table(dict(debug_id=[8,9],ra=ra,dec=dec,blue_candidate=[True,True],
            F115W_snr=[4.,8.],F150W_snr=[5.,8.],F200W_snr=[6.,8.]))
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'blue.fits';t.write(p)
            h=w.to_header();h['CRPIX1']-=10;h['CRPIX2']-=20
            selected=blue_candidates_on_window(p,h,(64,64))
            self.assertEqual(list(selected['debug_id']),[8])
            np.testing.assert_allclose([selected['x'][0],selected['y'][0]],[15,15],atol=1e-5)
            t['F150W_snr'][0]=3.99;t.write(p,overwrite=True)
            with self.assertRaisesRegex(ValueError,'all three bands'):
                blue_candidates_on_window(p,h,(64,64))

    def test_bundled_blue_asset_provenance_and_mixed_batch_cache_signature(self):
        import json,hashlib,tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from preprocessing.utils.uncover_labels import DEFAULT_BLUE_CATALOG,POLICY_VERSION
        from preprocessing.abell_zarr import uncover_signature
        asset=Table.read(DEFAULT_BLUE_CATALOG);manifest=json.loads(DEFAULT_BLUE_CATALOG.with_suffix('.json').read_text())
        self.assertEqual(len(asset),manifest['merged_rows'])
        self.assertEqual(hashlib.sha256(DEFAULT_BLUE_CATALOG.read_bytes()).hexdigest(),manifest['output_sha256'])
        self.assertEqual(len(np.unique(asset['debug_id'])),len(asset))
        for band in ('F115W','F150W','F200W'):self.assertTrue(np.all(asset[band+'_snr']>=4))
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for name in ('cat','det','seg','model'):(root/name).write_text('asset')
            args=SimpleNamespace(use_uncover=True,uncover_catalog=str(root/'cat'),uncover_detection=str(root/'det'),
                uncover_segmentation=str(root/'seg'),uncover_model_root=td,uncover_blue_catalog=str(DEFAULT_BLUE_CATALOG),
                uncover_no_kron_snr=3.,uncover_risk_no_kron_snr=5.,uncover_risk_bright_mag=22.,
                uncover_risk_sigma=3.,uncover_reject_untrusted=True)
            with patch('preprocessing.utils.uncover_labels.model_for_band',return_value=root/'model'):
                first=uncover_signature(args,'F150W');args.uncover_compact_dedup_pixels=1.
                changed=uncover_signature(args,'F150W')
            self.assertEqual(first['policy'],POLICY_VERSION)
            self.assertTrue(any(item['path']==str(DEFAULT_BLUE_CATALOG) for item in first['assets']))
            self.assertNotEqual(first,changed)

    def test_formal_postprocess_measures_current_red_band_and_writes_strict_targets(self):
        import tempfile,copy
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from preprocessing.abell_zarr import uncover_labels_for_window
        from preprocessing.utils.image_level import PatchLabels,_tile_targets
        from preprocessing.utils.inputs import TileSpec
        from preprocessing.image_processing import prepare_image
        w=WCS(naxis=2);w.wcs.crpix=[1,1];w.wcs.crval=[3.58,-30.4]
        w.wcs.cdelt=[-.03/3600,.03/3600];w.wcs.ctype=['RA---TAN','DEC--TAN']
        h=w.to_header();h['BUNIT']='MJy/sr';h['FILTER']='F356W'
        image=np.ones((64,64),np.float32);sky=np.ones(image.shape,bool)
        ra,dec=w.pixel_to_world_values([20.,40.],[30.,30.])
        table=Table(dict(debug_id=[8,9],blue_candidate=[True,True],ra=ra,dec=dec,
            F115W_snr=[5.,5.],F150W_snr=[5.,5.],F200W_snr=[5.,5.]))
        empty=PatchLabels(table=Table(),dense=np.full(image.shape,4,np.uint8),
            label_classes=np.array([],np.int16),source_ids=np.array([],np.int64),
            geom_x=np.array([]),geom_y=np.array([]),geom_major=np.array([]),geom_minor=np.array([]),geom_theta=np.array([]),
            strict_x=np.array([]),strict_y=np.array([]),strict_ids=np.array([],np.int64),strict_is_gaia=np.array([],bool))
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p=root/'blue.fits';table.write(p);audit=root/'x+01_y+00';audit.mkdir()
            args=SimpleNamespace(output_root=td,uncover_detection='det',uncover_segmentation='seg',
                uncover_model_root='models',uncover_risk_sigma=3.,uncover_no_kron_snr=3.,
                uncover_risk_no_kron_snr=5.,uncover_reject_untrusted=True,uncover_risk_bright_mag=22.,
                uncover_blue_catalog=str(p),uncover_native_psf_review=False)
            def fit(im,rms,x,y,band,**kw):
                self.assertEqual(band,'F356W');np.testing.assert_array_equal(im,image)
                self.assertEqual(kw['noise_factor'],2.)
                return dict(snr=4. if x<30 else 3.99,success=True)
            with patch('preprocessing.utils.uncover_labels.coverage_on_grid',return_value=sky), \
                 patch('preprocessing.utils.uncover_labels.project_product',side_effect=lambda *a,**k:(np.zeros(image.shape),h)), \
                 patch('preprocessing.utils.uncover_labels.model_for_band',return_value=Path('f356w_model.fits')), \
                 patch('preprocessing.utils.uncover_labels.classify_reference',side_effect=lambda *a,**k:(copy.deepcopy(empty),{})), \
                 patch('preprocessing.utils.jwst_star_psf.empirical_core_rms',return_value=np.ones(image.shape)), \
                 patch('preprocessing.utils.jwst_star_psf.calibrate_core_noise',return_value=(2.,dict(status='ok'))), \
                 patch('preprocessing.utils.jwst_star_psf.measure_fixed_psf_core',side_effect=fit) as measure:
                labels,summary,_=uncover_labels_for_window(args,image,h,Table(),Table(),sky,'F356W',
                    sky,prepare_image(image,header=h),h,audit)
                self.assertEqual(measure.call_count,2);self.assertEqual(summary['compact_centers']['added'],1)
                target=_tile_targets(labels,TileSpec('one',0,0,64),(0,0))
                np.testing.assert_allclose(target['strict_centers'],[[20,30]],atol=1e-5)
                self.assertEqual(len(target['source_centers']),0);self.assertFalse(target['shape_weight'].any())
                saved=Table.read(audit/'reference_blue_center_selection.fits')
                self.assertEqual(saved.meta['PSFBAND'],'F356W');self.assertEqual(saved['supplement_status'][1],'red_psf_snr_below_threshold')
                self.assertTrue((audit/'reference_extra_strict_centers.fits').exists())
                # The same sky candidates reach another BLUE band without a red-band cut.
                h['FILTER']='F150W'
                labels,summary,_=uncover_labels_for_window(args,image,h,Table(),Table(),sky,'F150W',
                    sky,prepare_image(image,header=h),h,audit,prefix='source_123')
                self.assertEqual(summary['compact_centers']['added'],2)
                self.assertEqual(measure.call_count,2)
                self.assertTrue((audit/'source_123_blue_center_selection.fits').exists())

    def test_final_kron_demotion_preserves_center_and_ignores_uncertain_shape(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from preprocessing.utils.uncover_labels import classify_reference
        from preprocessing.labels import DenseLabel as D
        w=WCS(naxis=2);w.wcs.crpix=[32,32];w.wcs.crval=[3.58,-30.4]
        w.wcs.cdelt=[-.03/3600,.03/3600];w.wcs.ctype=['RA---TAN','DEC--TAN']
        h=w.to_header();h['BUNIT']='MJy/sr'
        image=np.random.default_rng(2).normal(0,.01,(64,64)).astype(np.float32)
        ra,dec=w.pixel_to_world_values([24.,48.],[24.,48.])
        cat=Table(dict(id=[1,2],x=[24.,48.],y=[24.,48.],ra=ra,dec=dec,
            theta_J2000=[0.,0.],kron_radius=[1.,1.],a_image=[5.,5.],b_image=[5.,5.],
            flag_artifact=[0,0],flag_nearbcg=[0,0],flag_nophot=[0,0],flag_lowsnr=[0,0],
            **{'fauto_f277w+f356w+f444w':[40.,40.],'faper_f277w+f356w+f444w':[40.,40.],
               'eaper_f277w+f356w+f444w':[1.,1.]}))
        seg=np.zeros(image.shape,np.int32);yy,xx=np.indices(image.shape)
        seg[(xx-24)**2+(yy-24)**2<=25]=2
        sky=np.ones(image.shape,bool);empty=np.zeros(image.shape,bool)
        def bright(data,labels,**kw):
            return SimpleNamespace(labels=labels,strict_center_x=np.array([]),strict_center_y=np.array([]),
                strict_center_source_id=np.array([],np.int64),strict_center_reason=np.array([],object),
                ordinary_ignore_component_ids=np.array([]),restricted_fallback_component_ids=np.array([]))
        def protect(labels,**kw):labels.star_protection_policy={};return None
        with patch('preprocessing.image_processing.build_bright_components',return_value=(empty,np.zeros(image.shape,int))), \
             patch('preprocessing.bright_label.project_gaia_rows',return_value=Table()), \
             patch('preprocessing.utils.abell_labels.star_footprint',return_value=empty), \
             patch('preprocessing.bright_label_jwst.label_bright_sources',side_effect=bright), \
             patch('preprocessing.utils.jwst_star_protection.apply_star_protection',side_effect=protect):
            labels,summary=classify_reference(image,h,cat,Table(),sky,'F115W',native_header=h,
                native_segmentation=seg,aperture_measurement=self.measurement([100.,100.]))
        np.testing.assert_array_equal(labels.label_classes,[C.STRICT_CENTER_ONLY,C.CLEAN])
        np.testing.assert_allclose([labels.geom_x[0],labels.geom_y[0]],[24.,24.],atol=1e-6)
        self.assertEqual(labels.dense[24,24],D.ORDINARY_IGNORE)
        self.assertEqual(labels.dense[48,48],D.CLEAN)
        self.assertEqual(labels.table['classification_reason'][0],'uncover_kron_neighbor_segmentation')
        self.assertEqual(summary['kron_segmentation']['downgraded'],1)
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
