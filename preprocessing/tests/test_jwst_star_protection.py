"""Gaia model replacement, wing priority, faint-star eligibility and Zarr contract."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
import unittest
import numpy as np
from astropy.table import Table
from astropy.wcs import WCS
import zarr
from preprocessing.labels import DenseLabel as D, SourceClass as C, SourceLabels, LabelWeights
from preprocessing.region_filling import fill_dense_regions
from preprocessing.utils.geometry import EllipseGeometry
from preprocessing.utils.image_level import PatchLabels, _tile_targets
from preprocessing.utils.jwst_star_protection import (apply_star_protection, bright_star_footprint,
    ellipse_intersects, untrusted_near_stars)
from preprocessing.utils.jwst_star_psf import fit_angle, sample_template, star_asset, assess_point_source
from preprocessing.utils.jwst_star_masks import StarMask, build_star_masks, detection_magnitudes, neighbor_veto
from preprocessing.build_image_level_zarr import write_classified_patch
from preprocessing.tests.test_segmentation_zarr import task


def header(size=256):
    w=WCS(naxis=2);w.wcs.crpix=[size/2+1,size/2+1];w.wcs.crval=[3.57,-30.4]
    w.wcs.cdelt=[-.03/3600,.03/3600];w.wcs.ctype=['RA---TAN','DEC--TAN']
    h=w.to_header();h['BUNIT']='MJy/sr';return h


def labels(x=(64.,104.,108.,200.), classes=(C.CLEAN,C.CLEAN,C.WEAK_SHAPE,C.CLEAN),flags=(0,0,2,0),size=256):
    n=len(x);tab=Table(dict(NUMBER=np.arange(1,n+1,dtype=np.int32),FLAGS=flags,
                           FLUX_AUTO=np.full(n,10**((27-21)/2.5))))
    return PatchLabels(tab,np.zeros((size,size),np.uint8),np.array(classes,np.int16),np.array(x),
        np.full(n,64.),np.full(n,5.),np.full(n,4.),np.zeros(n),np.arange(1,n+1,dtype=np.int64),
        np.array([64.]),np.array([64.]),np.array([-2320205826528294145],np.int64),strict_is_gaia=np.array([True]))


def gaia(x=(64.,),g=(23.,),plx=(5.,),pm=(0.,)):
    n=len(x)
    return Table(dict(source_id=np.arange(n,dtype=np.int64)+2320205826528294144,
        x=x,y=np.full(n,64.),phot_g_mean_mag=g,phot_rp_mean_mag=np.full(n,19.),
        parallax=plx,parallax_error=np.ones(n),pmra=pm,pmdec=np.zeros(n),
        pmra_error=np.ones(n),pmdec_error=np.ones(n)))


class StarProtectionTests(unittest.TestCase):
    @staticmethod
    def masks(image,header,stars,band,pixel_scale):
        # Deterministic segmentation fixture; the real SExtractor connection
        # and mask geometry are tested independently below.
        result={}
        yy,xx=np.indices(image.shape)
        for row in stars:
            region=np.hypot(xx-row['x'],yy-row['y'])<=50
            result[int(row['source_id'])]=StarMask((0,0),region,dict(status='fixture',mask_pixels=int(region.sum())))
        return result

    def apply(self,lab,stars,**kwargs):
        kwargs.setdefault('trusted',np.ones(len(lab.source_ids),bool))
        with patch('preprocessing.utils.jwst_star_protection.fit_angle',return_value=dict(angle_rad=0.,status='test')), \
             patch('preprocessing.utils.jwst_star_protection.build_star_masks',side_effect=self.masks):
            return apply_star_protection(lab,image=np.ones(lab.dense.shape),header=header(),gaia=stars,
                band='F444W',pixel_scale=.03,**kwargs)

    def test_faint_star_replacement_and_flag_intersection(self):
        lab=labels();wings=self.apply(lab,gaia())
        np.testing.assert_array_equal(lab.label_classes,[C.DROPPED,C.CLEAN,C.ORDINARY_IGNORE,C.CLEAN,C.WEAK_SHAPE])
        self.assertEqual(lab.source_ids[-1],2320205826528294144)
        self.assertEqual(lab.table['NUMBER'].dtype,np.dtype('int64'))
        self.assertTrue(lab.table['FLUX_AUTO'].mask[-1])
        self.assertEqual(len(lab.strict_ids),0)
        self.assertTrue(wings[64,100]);self.assertFalse(wings[64,200])
        expected=star_asset()[0]['filters']['F444W']['kron_a_arcsec']/.03
        self.assertAlmostEqual(lab.geom_major[-1],expected)
        self.assertIn('no magnitude limit',lab.star_protection_policy['gaia_model_selection'])

    def test_existing_motion_rule_negative_parallax_and_galaxy(self):
        # Negative parallax alone fails; strong proper motion still qualifies.
        stars=gaia(x=(64.,120.,200.),g=(25.,23.,21.),plx=(-1.,-1.,.1),pm=(4.,0.,0.))
        lab=labels();self.apply(lab,stars)
        self.assertEqual(lab.star_protection_policy['model_stars'],1)
        self.assertEqual(lab.source_ids[-1],stars['source_id'][0])
        self.assertEqual(lab.label_classes[3],C.CLEAN)  # Galaxy-like Gaia row stays ordinary.

    def test_missing_g_not_a_magnitude_veto(self):
        lab=labels();self.apply(lab,gaia(g=(np.nan,)))
        self.assertEqual(lab.star_protection_policy['model_stars'],1)

    def test_dropped_neighbor_not_resurrected(self):
        lab=labels(classes=(C.CLEAN,C.ORDINARY_IGNORE,C.DROPPED,C.CLEAN))
        self.apply(lab,gaia());self.assertEqual(lab.label_classes[1],C.ORDINARY_IGNORE)
        self.assertEqual(lab.label_classes[2],C.DROPPED)

    def test_continuous_intersection_not_center_membership(self):
        self.assertTrue(ellipse_intersects((12.,0.,3.,1.,0.),(0.,0.,10.,5.,0.)))
        self.assertTrue(ellipse_intersects((13.,0.,3.,1.,0.),(0.,0.,10.,5.,0.)))
        self.assertFalse(ellipse_intersects((13.1,0.,3.,1.,0.),(0.,0.,10.,5.,0.)))

    def test_rp_threshold_inclusive_independent_of_g(self):
        rng=np.random.default_rng(9);image=rng.normal(0,1,(100,100));image[45:56,45:56]+=30
        rows=[dict(x=50.,y=50.,phot_rp_mean_mag=18.,phot_g_mean_mag=25.)]
        selected=bright_star_footprint(image,np.zeros(image.shape,bool),rows)
        self.assertTrue(selected[50,50])
        rows[0].update(phot_rp_mean_mag=18.01,phot_g_mean_mag=12.)
        self.assertFalse(bright_star_footprint(image,np.zeros(image.shape,bool),rows).any())

    def test_snr_margin_and_trust_precedence(self):
        lab=labels(x=(64.,94.,100.,200.));fp=np.zeros(lab.dense.shape,bool);fp[64,64]=True
        got=untrusted_near_stars(np.ones(fp.shape),lab,fp,None,trusted=[True,False,False,False],magnitudes=np.full(4,25.))
        np.testing.assert_array_equal(got,[False,True,False,False])
        lab.geom_x[1]=94
        with patch('utils.source_snr.measure_local_aperture_snr',return_value={'trusted':np.array([False,False])}):
            got=untrusted_near_stars(np.ones(fp.shape),lab,fp,None,magnitudes=np.full(4,25.))
        np.testing.assert_array_equal(got,[True,True,False,False])

    def test_dense_priority_and_positive_core(self):
        lab=labels(x=(64.,100.,106.,200.),classes=(C.WEAK_SHAPE,C.CLEAN,C.ORDINARY_IGNORE,C.CLEAN))
        geom=EllipseGeometry(lab.geom_x,lab.geom_y,lab.geom_major,lab.geom_minor,lab.geom_theta,
                             np.pi*lab.geom_major*lab.geom_minor)
        wings=np.zeros(lab.dense.shape,bool);wings[30:100,30:120]=True
        invalid=np.zeros_like(wings);invalid[30,30]=True
        lab.dense=fill_dense_regions(lab.table,SourceLabels(lab.label_classes,np.full(4,'',object)),
            wings.shape,geometry=geom,background_mask=np.ones_like(wings),quality_ignore_mask=invalid,
            restricted_fallback_mask=wings&~invalid,star_wings_mask=wings)
        self.assertEqual(lab.dense[64,64],D.WEAK_SHAPE)
        self.assertEqual(lab.dense[64,100],D.CLEAN)
        self.assertEqual(lab.dense[64,110],D.STAR_WINGS)
        self.assertEqual(lab.dense[30,30],D.STRICT_IGNORE)
        result=_tile_targets(lab,SimpleNamespace(x0=0,y0=0,size=256),(0,0))
        self.assertAlmostEqual(float(result['conf_weight'][64,110]),LabelWeights().star_wings)
        self.assertEqual(result['confidence'][64,110],0)
        self.assertGreater(result['confidence'][64,64],0)
        self.assertEqual(result['conf_weight'][64,64],1)

    def test_rotation_fit_and_fixed_size(self):
        yy,xx=np.indices((128,128));angle=np.deg2rad(37)
        image=sample_template('F444W',xx-64,yy-64,.03,angle)*1e6+2.
        result=fit_angle(image,(64,64),'F444W',.03)
        fitted=sample_template('F444W',xx-64,yy-64,.03,result['angle_rad'])
        self.assertGreater(np.corrcoef(image.ravel(),fitted.ravel())[0,1],.99)

    def test_psf_fallback_refines_center_without_resizing(self):
        yy,xx=np.indices((128,128));model=sample_template('F444W',xx-70,yy-64,.03,.7)
        image=model*100/model.max()+np.random.default_rng(6).normal(0,1,model.shape)
        lab=labels(x=(70.,),classes=(C.CLEAN,),flags=(0,),size=128)
        apply_star_protection(lab,image=image,header=header(128),gaia=gaia(plx=(-.1,)),band='F444W',pixel_scale=.03)
        self.assertEqual(lab.star_protection_policy['psf_selected'],1)
        self.assertEqual(lab.label_classes[0],C.DROPPED)
        self.assertEqual(lab.label_classes[-1],C.WEAK_SHAPE)
        self.assertLess(abs(lab.geom_x[-1]-70),.2)
        self.assertAlmostEqual(lab.geom_major[-1],star_asset()[0]['filters']['F444W']['kron_a_arcsec']/.03)
        self.assertEqual(lab.star_protection_sources[0]['selection'],'psf')

    def test_psf_fallback_rejects_extended_gaia_preserving_label(self):
        from scipy.ndimage import gaussian_filter
        yy,xx=np.indices((128,128));model=sample_template('F444W',xx-64,yy-64,.03,.7)
        model=gaussian_filter(model,3.)
        image=model*100/model.max()+np.random.default_rng(6).normal(0,1,model.shape)
        lab=labels(x=(64.,),classes=(C.CLEAN,),flags=(0,),size=128)
        stars=gaia(plx=(-.1,),pm=(4.,));stars['pmra_error']=100.
        # A large nominal proper motion with an enormous error is not
        # trusted astrometry: it must pass the image fallback as well.
        apply_star_protection(lab,image=image,header=header(128),gaia=stars,band='F444W',pixel_scale=.03)
        self.assertEqual(lab.star_protection_policy['model_stars'],0)
        self.assertEqual(lab.label_classes.tolist(),[C.CLEAN])
        self.assertEqual(lab.star_protection_policy['psf_fallback_audit'][0]['status'],'extended')

    def test_psf_fallback_missing_core_does_not_invent_evidence(self):
        image=np.random.default_rng(8).normal(size=(128,128));image[48:80,48:80]=np.nan
        result=assess_point_source(image,(64,64),'F444W',.03)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['status'],'insufficient_finite_core')

    def test_psf_fallback_cap_rejects_compact_extended_core(self):
        from scipy.ndimage import gaussian_filter
        yy,xx=np.indices((128,128))
        model=gaussian_filter(sample_template('F444W',xx-64,yy-64,.03,.7),1.2083333333)
        image=model*2000/model.max()+np.random.default_rng(31).normal(0,1,model.shape)
        previous=assess_point_source(image,(64,64),'F444W',.03,
            config_override={'unresolved_sigma_cap_pixels':float('inf')})
        current=assess_point_source(image,(64,64),'F444W',.03)
        self.assertTrue(previous['accepted'])
        self.assertFalse(current['accepted'])
        self.assertEqual(current['status'],'extended')
        self.assertEqual(current['unresolved_sigma_limit_pixels'],1.)

    def test_cosmos_final_pipeline_inserts_faint_star(self):
        from preprocessing.utils.cosmos_labels import classify_window
        hdr=header();world=WCS(hdr).celestial.pixel_to_world([64.,108.],[64.,64.])
        cat=Table(dict(id=[10,11],ra=world.ra.deg,dec=world.dec.deg,kron2_a=[.15,.15],
            kron2_b=[.12,.12],theta_world=[0.,0.],warn_flag=[0,2],flag_star=[0,0],
            mag_auto_f444w=[25.7,25.7],mag_model_f444w=[25.7,25.7],snr_f444w=[20.,20.]))
        image=np.random.default_rng(4).normal(0,.001,(256,256)).astype('f4')
        with patch('preprocessing.utils.cosmos_labels.cosmos_fill_ratios',return_value=np.ones(2)), \
             patch('preprocessing.utils.jwst_star_protection.fit_angle',return_value=dict(angle_rad=0.,status='test')), \
             patch('preprocessing.utils.jwst_star_protection.build_star_masks',side_effect=self.masks), \
             patch('utils.source_snr.measure_local_aperture_snr',return_value={'trusted':np.ones(2,bool)}):
            lab=classify_window(image,hdr,cat,'unused.fits',gaia(),np.ones(image.shape,bool),'F444W')
        np.testing.assert_array_equal(lab.label_classes,[C.DROPPED,C.ORDINARY_IGNORE,C.WEAK_SHAPE])
        self.assertTrue((lab.dense==D.STAR_WINGS).any())
        self.assertEqual(lab.star_protection_policy['model_stars'],1)

    def test_abell_final_pipeline_inserts_faint_star(self):
        from preprocessing.utils.abell_labels import classify_reference
        hdr=header();world=WCS(hdr).celestial.pixel_to_world([64.,108.],[64.,64.])
        cat=Table(dict(NUMBER=np.array([10,11],np.int32),ALPHA_J2000=world.ra.deg,DELTA_J2000=world.dec.deg,
            KRON_RADIUS=[2.5,2.5],A_IMAGE=[2.,2.],B_IMAGE=[1.6,1.6],THETA_IMAGE=[0.,0.],
            FLAGS=[0,2],FLUX_AUTO=[15.,15.],FLUXERR_AUTO=[1.,1.],SNR_WIN=[20.,20.]))
        image=np.random.default_rng(4).normal(0,.001,(256,256)).astype('f4')
        with patch('preprocessing.utils.jwst_star_protection.fit_angle',return_value=dict(angle_rad=0.,status='test')), \
             patch('preprocessing.utils.jwst_star_protection.build_star_masks',side_effect=self.masks), \
             patch('utils.source_snr.measure_local_aperture_snr',return_value={'trusted':np.ones(2,bool)}):
            lab,summary=classify_reference(image,hdr,cat,gaia(),np.ones(image.shape,bool),'F444W',
                bright_image=image,bright_header=hdr,training_background=np.ones(image.shape,bool))
        self.assertEqual(lab.label_classes[0],C.DROPPED)
        self.assertEqual(lab.label_classes[1],C.ORDINARY_IGNORE)
        self.assertEqual(lab.label_classes[-1],C.WEAK_SHAPE)
        self.assertTrue((lab.dense==D.STAR_WINGS).any())
        self.assertEqual(summary['star_protection']['model_stars'],1)

    def test_zarr_roundtrip_repaired_model_center(self):
        lab=labels(x=(64.,),flags=(0,),classes=(C.CLEAN,),size=128);wings=self.apply(lab,gaia())
        geom=EllipseGeometry(lab.geom_x,lab.geom_y,lab.geom_major,lab.geom_minor,lab.geom_theta,
                             np.pi*lab.geom_major*lab.geom_minor)
        lab.dense=fill_dense_regions(lab.table,SourceLabels(lab.label_classes,np.full(2,'',object)),
            lab.dense.shape,geometry=geom,star_wings_mask=wings)
        raw=np.ones(lab.dense.shape,np.float32);valid=np.ones(raw.shape,bool);valid[64,64]=False
        context=np.ones_like(valid);context[:,0]=False;valid[:,0]=False
        with tempfile.TemporaryDirectory() as tmp, patch('preprocessing.build_image_level_zarr._scale_image_chw',side_effect=lambda a,t:a[None]):
            out=write_classified_patch(replace(task(tmp),tile_size=128,stride=128,band='F444W'),raw,lab,
                valid_mask=valid,input_valid_mask=context,max_invalid_fraction=.1)
            z=zarr.open_group(out['output'],mode='r')
            self.assertIn(2320205826528294144,z['shape_source_ids'][:])
            self.assertIn(2320205826528294144,z['source_ids'][:])
            self.assertEqual(z['shape_source_classes'][-1],C.WEAK_SHAPE)
            self.assertFalse(z['band_valid_mask'][0,0,64,64])
            self.assertEqual(z['band_confidence'][0,0,64,64],4)
            self.assertAlmostEqual(float(z['band_conf_weight'][0,0,64,100]),LabelWeights().star_wings)
            self.assertEqual(z['band_conf_weight'][0,0,64,0],0)
            self.assertEqual(z.attrs['jwst_star_protection']['model_stars'],1)
            from astro_train_zarr_data import _target_defaults_from_pu
            import torch
            target=_target_defaults_from_pu(confidence=torch.as_tensor(z['band_confidence'][0,0]),
                shape=torch.as_tensor(z['band_shape'][0,0]),shape_weight=torch.as_tensor(z['band_shape_weight'][0,0]),
                conf_weight=torch.as_tensor(z['band_conf_weight'][0,0]),pu=torch.as_tensor(z['band_pu_class_mask'][0,0]))
            self.assertAlmostEqual(target['confidence_weight'][64,100].item(),LabelWeights().star_wings)
            self.assertEqual(target['background_mask'][64,100].item(),0)  # Not mask-BCE background.
            from astro_train_ops import hard_ce_confidence_loss
            logits=torch.zeros((1,5,1,3),requires_grad=True)
            hard_ce_confidence_loss(logits,torch.zeros((1,1,3),dtype=torch.long),
                class_weights=[1,4,8,16,32],weight=torch.tensor([[[1.,.1,0.]]])).backward()
            self.assertAlmostEqual(logits.grad[0,0,0,1].item()/logits.grad[0,0,0,0].item(),.1,places=6)
            self.assertEqual(logits.grad[0,0,0,2].item(),0)

    def test_star_mask_flags_beyond_ee95_and_no_added_negatives(self):
        lab=labels(x=(64.,175.,180.,190.),flags=(0,2,0,4),
                   classes=(C.CLEAN,C.CLEAN,C.WEAK_SHAPE,C.DROPPED))
        # A long catalog Kron touches the mask although its center is outside.
        lab.geom_major[1]=20
        mask=np.zeros(lab.dense.shape,bool);mask[62:67,155:165]=True
        star=gaia(g=(26.,));star['phot_rp_mean_mag']=21.
        m=StarMask((0,0),mask,dict(status='fixture',mask_pixels=int(mask.sum())))
        with patch('preprocessing.utils.jwst_star_protection.build_star_masks',return_value={int(star['source_id'][0]):m}), \
             patch('preprocessing.utils.jwst_star_protection.fit_angle',return_value=dict(angle_rad=0.,status='test')):
            wings=apply_star_protection(lab,image=np.ones(mask.shape),header=header(),gaia=star,band='F444W',pixel_scale=.03,trusted=np.ones(4,bool))
        self.assertEqual(lab.label_classes.tolist(),[C.DROPPED,C.ORDINARY_IGNORE,C.WEAK_SHAPE,C.DROPPED,C.WEAK_SHAPE])
        self.assertFalse(wings[64,160])  # Selection mask is NOT STAR_WINGS.
        self.assertTrue(wings[64,100])  # Existing EE95 negatives are retained.
        self.assertEqual(lab.star_protection_sources[0]['intersecting_catalog_ids'],[2])

    def test_all_rp_use_same_magnitude_and_flag_rule(self):
        for rp in [17.,18.,18.01,22.,np.nan]:
            lab=labels(x=(64.,94.,104.,108.,112.,114.,200.),
                classes=(C.CLEAN,C.CLEAN,C.STRICT_CENTER_ONLY,C.WEAK_SHAPE,C.STRICT_IGNORE,C.DROPPED,C.CLEAN),
                flags=(0,0,0,2,0,2,0))
            lab.table['FLUX_AUTO']=10**((27-np.array([21.,23.99,24.,21.,25.,25.,25.]))/2.5)
            stars=gaia();stars['phot_rp_mean_mag']=rp
            self.apply(lab,stars)
            self.assertEqual(lab.label_classes.tolist(),[C.DROPPED,C.CLEAN,C.ORDINARY_IGNORE,
                C.ORDINARY_IGNORE,C.STRICT_IGNORE,C.DROPPED,C.CLEAN,C.WEAK_SHAPE])
            self.assertTrue(lab.table['star_protection_magnitude_veto'][2])

    def test_catalog_magnitude_not_gaia_magnitude_or_invalid_flux(self):
        lab=labels(x=(64.,94.,104.,108.,110.,112.),flags=(0,0,0,0,0,2))
        lab.table['FLUX_AUTO']=np.ma.array([100.,0.,-1.,np.nan,10.,1.],mask=[0,0,0,0,1,0])
        mag=detection_magnitudes(lab,'F444W')
        self.assertEqual(mag[0],22.)
        self.assertTrue(np.isnan(mag[1:5]).all())
        cosmos=SimpleNamespace(table=Table(dict(id=[1,2],mag_auto_f115w=[21.,24.])))
        np.testing.assert_array_equal(detection_magnitudes(cosmos,'F115W'),[21.,24.])

    def test_faint_kron_contact_without_center_is_only_vetoed_if_flagged(self):
        lab=labels(x=(60.,60.,50.),flags=(0,2,0),classes=(C.CLEAN,)*3)
        lab.geom_major[:]=11.;mask=StarMask((50,64),np.ones((1,1),bool),{})
        f,m,_=neighbor_veto(lab,[{'source_id':123}],{123:mask},[0,2,0],[25.,21.,24.])
        np.testing.assert_array_equal(f,[False,True,False])
        np.testing.assert_array_equal(m,[False,False,True])

    def test_mask_pixel_area_continuous_contact(self):
        mask=StarMask((50,50),np.ones((1,1),bool),{})
        self.assertTrue(mask.intersects((52.5,50.,2.,.1,0.)))
        self.assertFalse(mask.intersects((52.5001,50.,2.,.1,0.)))
        self.assertTrue(mask.intersects((50.49,50.49,.001,.001,0.)))

    def test_faint_zero_flag_still_requires_trusted_snr_in_32px_margin(self):
        lab=labels(x=(64.,120.,140.,200.),flags=(0,0,0,0))
        lab.table['FLUX_AUTO']=10**((27-np.array([21.,23.99,24.,25.]))/2.5)
        star=gaia();star['phot_rp_mean_mag']=21.
        region=np.zeros(lab.dense.shape,bool);region[64,120]=True
        m=StarMask((0,0),region,{'status':'fixture'})
        with patch('preprocessing.utils.jwst_star_protection.build_star_masks',return_value={int(star['source_id'][0]):m}), \
             patch('preprocessing.utils.jwst_star_protection.fit_angle',return_value=dict(angle_rad=0.,status='test')):
            apply_star_protection(lab,image=np.ones(region.shape),header=header(),gaia=star,band='F444W',pixel_scale=.03,
                trusted=[True,True,False,False])
        self.assertEqual(lab.label_classes.tolist(),[C.DROPPED,C.CLEAN,C.ORDINARY_IGNORE,C.CLEAN,C.WEAK_SHAPE])
        self.assertTrue(lab.table['star_protection_untrusted'][2])

    def test_manual_nonstellar_overrides_astrometry_and_bright_insertion(self):
        from preprocessing.bright_label_jwst import _gaia_rows,JWSTBrightConfig
        stars=gaia(plx=(10.,));stars['source_id'][0]=2320211530244866176
        stars['phot_rp_mean_mag']=17.
        lab=labels();before=lab.label_classes.copy()
        with patch('preprocessing.utils.jwst_star_protection.assess_point_source') as match:
            wings=self.apply(lab,stars)
        match.assert_not_called()
        self.assertIsNone(wings)
        np.testing.assert_array_equal(lab.label_classes,before)
        self.assertEqual(lab.star_protection_sources,[])
        self.assertEqual(_gaia_rows(stars,None,header(),(256,256),
            JWSTBrightConfig('a2744',.03,gaia_mag_max=None)),[])
        image=np.ones((256,256));image[60:69,60:69]=100
        rows=[dict(source_id=2320211530244866176,x=64.,y=64.,phot_rp_mean_mag=17.)]
        self.assertFalse(bright_star_footprint(image,np.zeros(image.shape,bool),rows).any())

    def test_real_sextractor_faint_star_mask_selects_wings_not_other_islands(self):
        import shutil
        if not (shutil.which('source-extractor') or shutil.which('sex')):self.skipTest('SExtractor unavailable')
        yy,xx=np.indices((256,256));rng=np.random.default_rng(41)
        dx,dy=xx-64,yy-128
        image=rng.normal(0,1,xx.shape)+150*np.exp(-(dx*dx+dy*dy)/18)
        image+=9*np.exp(-dy*dy/5)*np.exp(-dx*dx/11000)
        image+=80*np.exp(-((xx-225)**2+(yy-210)**2)/12)
        row=dict(source_id=123,x=64.,y=128.,phot_rp_mean_mag=21.,phot_g_mean_mag=23.)
        masks=build_star_masks(image,header(),[row],'F444W',.03)
        self.assertEqual(masks[123].metadata['status'],'seeded_component')
        self.assertTrue(masks[123].intersects((120.,128.,2.,2.,0.)))
        self.assertFalse(masks[123].intersects((225.,210.,2.,2.,0.)))
        self.assertFalse(masks[123].metadata['negative_supervision'])

if __name__=='__main__':unittest.main()
