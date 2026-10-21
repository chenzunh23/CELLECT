"""Bounded FITS / common-WCS / mixed-Zarr selection regression checks."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from eval.datasets.jwst_fields import fits_window, AbellAccess, CosmosAccess
from eval.datasets.jwst_zarr import resolve_field_zarr, zarr_header
from preprocessing.dataset_inputs import ImageInput


def header(n=32):
    w = WCS(naxis=2)
    w.wcs.crpix = [n/2, n/2]
    w.wcs.crval = [150, 2]
    w.wcs.cdelt = [-.03/3600, .03/3600]
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    h = w.to_header()
    h['BUNIT'] = 'MJy/sr'
    return h


def array(store, name, data):
    data = np.asarray(data)
    p = store/name
    p.mkdir()
    (p/'.zarray').write_text(json.dumps(dict(zarr_format=2, shape=data.shape, chunks=data.shape,
        dtype=data.dtype.str, compressor=None, filters=None, order='C', fill_value=0)))
    (p/'.'.join(['0']*data.ndim)).write_bytes(data.tobytes())


def store(root, proposal):
    p=root/f'proposal_{proposal}'/'P0019_x00000_y00000.zarr';p.mkdir(parents=True)
    attrs=dict(dataset='cosmos',proposal=proposal,patch='P0019_x00000_y00000',bands=['F115W','F444W'],
        dataset_source='coadd',sky_wcs_header=header().tostring(sep='\n'))
    (p/'.zattrs').write_text(json.dumps(attrs))
    # No image chunks needed for metadata-only selection.
    (p/'images').mkdir();(p/'images'/'.zarray').write_text(json.dumps(dict(shape=[2,2,3,8,8],chunks=[2,2,3,8,8],dtype='<f4')))
    array(p,'tile_x0',np.array([0,8],dtype='<i4'));array(p,'tile_y0',np.array([0,8],dtype='<i4'))
    return p


class JwstInputTests(unittest.TestCase):
    def test_invalid_pixels_are_excluded_before_detection_nms(self):
        import torch
        from eval.eval_utils import detection_rows
        confidence=torch.full((1,5,8,8),-10.)
        confidence[:,0]=0
        confidence[0,4,2,2]=12.;confidence[0,4,2,3]=10.
        outputs=dict(confidence=confidence,shape=torch.ones(1,3,8,8))
        valid=np.ones((8,8),bool);valid[2,2]=False
        rows=detection_rows(outputs,threshold=2.,nms_radius=1,
            confidence_score='ordinal_expectation',center_refinement='integer',
            center_refinement_radius=1,width=8,height=8,valid_mask=valid)
        self.assertEqual([(r['x'],r['y']) for r in rows],[(3.,2.)])
        self.assertEqual(float(outputs['confidence'][0,4,2,2]),12.)

    def test_same_source_hole_policy_in_writer_bright_and_browser(self):
        from dataclasses import replace
        import zarr
        from preprocessing.image_processing import prepare_image, build_bright_components, BrightRegionConfig
        from preprocessing.utils.parent_scaling import fit_parent_rgb, apply_parent_rgb
        from preprocessing.build_image_level_zarr import write_classified_patch
        from preprocessing.tests.test_segmentation_zarr import labels, task
        from preprocessing.labels import DenseLabel
        raw=np.random.default_rng(30).normal(0,.01,(128,128)).astype('f4')
        raw[44:58,44:58]+=10
        raw[49:52,49:52]=np.nan                    # Enclosed bright-source core.
        raw[90:94,90:94]=np.nan                    # Enclosed sky hole.
        raw[20:40,:6]=np.nan;raw[20:40,6:10]+=10   # Open boundary beside a star.
        source=self.root/'same.fits';fits.PrimaryHDU(raw,header(128)).writeto(source)
        quality=self.root/'same_quality.npz';np.savez(quality,bad=~np.isfinite(raw))
        (self.root/'manifest.json').write_text(json.dumps({'items':[
            dict(image_fits=str(source),quality_mask_npz=str(quality),sample_name='same',proposal=1727,pointing=19,band='F444W')]}))
        prep=prepare_image(raw,header=header(128),source_path=source)
        params=fit_parent_rgb(prep.image);rgb=apply_parent_rgb(prep.image,params)
        bright,_=build_bright_components(prep,config=BrightRegionConfig(threshold=5,clip_threshold=5))
        self.assertTrue(bright[50,50]);self.assertFalse(bright[90:94,90:94].any())
        self.assertFalse(bright[20:40,:6].any())
        lab=labels();lab.dense=np.full((128,128),int(DenseLabel.BACKGROUND),np.uint8)
        lab.strict_x=lab.strict_y=np.array([50.]);lab.strict_ids=np.array([-1])
        lab.strict_is_gaia=np.array([True])
        out=write_classified_patch(replace(task(self.root/'zarr'),tile_size=128,stride=128),
            prep.image,lab,valid_mask=prep.finite_mask,input_valid_mask=prep.input_valid_mask,
            scaled_image_chw=rgb,max_invalid_fraction=.1)
        store=zarr.open_group(out['output'],mode='r')
        a=CosmosAccess(self.root,tile_size=128);a.scaling_parameters=lambda *args:params
        image,web,_,ok=a.scaled_window('F444W','p1727_P0019',0,0,128,128)
        np.testing.assert_allclose(image,prep.image,equal_nan=True)
        np.testing.assert_allclose(web,store['images'][0,0])
        for ys,xs in ((slice(90,94),slice(90,94)),(slice(20,40),slice(0,6))):
            self.assertTrue(np.isnan(image[ys,xs]).all());self.assertFalse(ok[ys,xs].any())
            self.assertTrue((web[:,ys,xs]==0).all())
            self.assertTrue((store['band_conf_weight'][0,0,ys,xs]==0).all())
        self.assertTrue(ok[50,50]);self.assertGreater(store['band_conf_weight'][0,0,50,50],0)
        self.assertFalse(store['band_valid_mask'][0,0,50,50])  # No synthetic mask truth.
        self.assertEqual(prep.metadata['nan_pixels_background_fill'],0)
        self.assertEqual(prep.metadata['nan_pixels_nearest_fill'],9)

    def test_cli_fits_repairs_cut_hole_before_scaling_without_filling_padding(self):
        from unittest.mock import patch
        from eval.eval_utils import build_scaled_tensor_from_fits
        p=self.root/'cli.fits'
        raw=np.random.default_rng(9).normal(0,.01,(64,64)).astype('f4')
        raw[20:38,18:38]+=10
        raw[28:32,22:28]=np.nan
        raw[40:44,:30]=np.nan
        fits.PrimaryHDU(raw,header(64)).writeto(p)
        with patch.dict('os.environ',{'CELLECT_NAN_CACHE':str(self.root/'cache')}):
            tensor,images,scaled,_=build_scaled_tensor_from_fits([p],hdu=0,x0=24,y0=24,
                width=24,height=24,scaling_mode='log_lupton',clip_threshold=5,log_a=1000,
                log_high_percentile=99.5,lupton_stretch=.5,lupton_q=20,
                anscombe_clip=False,anscombe_scale=1000)
        self.assertEqual(tuple(tensor.shape),(1,1,3,24,24))
        self.assertTrue((images[0][4:8,:4]>1).all())
        self.assertTrue((scaled[0][:,16:20,:6]==0).all())

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()

    def test_scaled_integer_fits_padding_and_quality(self):
        p=self.root/'raw.fits'
        sci=np.arange(32*32,dtype=np.uint16).reshape(32,32)
        bad=np.zeros((32,32),np.uint8);bad[2,2]=1
        fits.HDUList([fits.PrimaryHDU(),fits.ImageHDU(sci,header(),name='SCI'),fits.ImageHDU(bad,name='TRAIN_BAD')]).writeto(p)
        src=ImageInput('cosmos','test','F444W','0019',p,p)
        raw,valid,h=fits_window(src,-2,-1,8,8)
        self.assertTrue(np.isnan(raw[:1]).all());self.assertEqual(raw[3,4],sci[2,2]);self.assertFalse(valid[3,4])
        a=WCS(h).celestial.pixel_to_world_values(4,3)
        b=WCS(header()).celestial.pixel_to_world_values(2,2)
        np.testing.assert_allclose(a,b,atol=1e-10)

    def test_cosmos_namespace_and_uppercase_frame(self):
        p=self.root/'science.fits';fits.PrimaryHDU(np.ones((32,32),np.float32),header()).writeto(p)
        q=self.root/'quality.npz';np.savez(q,bad=np.zeros((32,32),bool))
        rows=[dict(image_fits=str(p),quality_mask_npz=str(q),sample_name='test',proposal=i,pointing=19,band='F444W') for i in (1727,5893)]
        (self.root/'manifest.json').write_text(json.dumps(dict(items=rows)))
        a=CosmosAccess(self.root,tile_size=16)
        self.assertEqual(a.available_patches(),['p1727_P0019','p5893_P0019'])
        self.assertEqual(a.available_bands(),['F444W'])
        tiles=a.choose_tiles('p5893_P0019',['F444W'],n_tiles=1,all_tiles=False,seed=7)
        ref=a.make_ref(token='test',patch='p5893_P0019',band='F444W',tile_id=tiles[0],frame_slot=0,frame_rank=0,frames_per_tile=1,visit=None,strict_visit=False)
        self.assertEqual(ref.band,'F444W');self.assertEqual(ref.dataset,'jwst_cosmos')
        self.assertEqual(a.read_frame(ref).shape,(16,16))

    def test_abell_fallback_reads_half_pixels_on_common_wcs(self):
        half=self.root/'half.fits';full=self.root/'full.fits'
        data=np.arange(32*32,dtype=np.float32).reshape(32,32)
        fits.PrimaryHDU(data,header()).writeto(half)
        fits.PrimaryHDU(np.full((32,32),9999,np.float32),header()).writeto(full)
        (self.root/'manifests').mkdir()
        row=dict(band='F444W',patch='x+00_y+00',status='kept',training_fits=str(self.root/'missing.fits'),
            original_training=str(half),ix=0,iy=0,x0=0,y0=0,size=32)
        (self.root/'manifests'/'F444W.json').write_text(json.dumps([row]))
        (self.root/'grid_plan.json').write_text(json.dumps(dict(reference_fits=str(full))))
        a=AbellAccess(self.root)
        crop,valid,h=a.read_window('F444W','x+00_y+00',4,5,8,8)
        np.testing.assert_allclose(crop,data[5:13,4:12],atol=.001)
        self.assertTrue(valid.all())
        self.assertEqual(a._shape('F444W','x+00_y+00'),(32,32))

    def test_zarr_coordinate_proposal_band_and_bounds(self):
        p=store(self.root,1727);store(self.root,5893)
        r,i,b,attrs=resolve_field_zarr(root=self.root,dataset='cosmos',proposal=5893,patch='P0019',band='f444w',xy=(10,10))
        self.assertEqual((attrs['proposal'],i,b),(5893,1,1))
        sky=WCS(zarr_header(attrs)).celestial.pixel_to_world_values(10,10)
        _,j,_,_=resolve_field_zarr(zarr_store=r.root,radec=sky,band='F444W')
        self.assertEqual(j,1)
        _,j,_,_=resolve_field_zarr(zarr_store=p,sample_index=1)
        self.assertEqual(j,1)
        for kwargs in [dict(sample_index=-1),dict(sample_index=2),dict(band='F150W'),dict(xy=(50,50))]:
            with self.assertRaises((LookupError,ValueError)):
                resolve_field_zarr(zarr_store=p,**kwargs)

    def test_scaled_window_zeros_quality_pixels_and_open_nan_boundary(self):
        from unittest.mock import patch
        from preprocessing.image_processing import prepare_image
        from preprocessing.utils.parent_scaling import fit_parent_rgb
        p=self.root/'science.fits'
        raw=np.random.default_rng(2).normal(0,.01,(64,64)).astype('f4')
        raw[20:35,:10]=np.nan;raw[20:35,10:15]+=10
        raw[40:44,22:28]=np.nan  # Closed hole cut by the requested crop.
        raw[38:46,20:30]+=10   # Source-like ring; sky holes now remain invalid.
        h=header(64);fits.PrimaryHDU(raw,h).writeto(p)
        q=self.root/'quality.npz';bad=np.zeros((64,64),bool);bad[25,12]=True
        np.savez(q,bad=bad)
        (self.root/'manifest.json').write_text(json.dumps({'items':[
            dict(image_fits=str(p),quality_mask_npz=str(q),sample_name='test',proposal=1727,pointing=19,band='F444W')]}))
        with patch.dict('os.environ',{'CELLECT_NAN_CACHE':str(self.root/'cache')}):
            a=CosmosAccess(self.root)
            params=fit_parent_rgb(prepare_image(raw,header=h,source_path=p).image)
            a.scaling_parameters=lambda *args:params
            image,rgb,_,valid=a.scaled_window('F444W','p1727_P0019',0,16,32,32)
            self.assertTrue(np.isnan(image[4:19,:10]).all())
            self.assertTrue((rgb[:,~valid]==0).all())
            image,rgb,_,valid=a.scaled_window('F444W','p1727_P0019',24,36,24,20)
            self.assertTrue(np.isfinite(image[4:8,:4]).all())
            self.assertTrue(valid[4:8,:4].all())
            self.assertTrue(np.any(rgb[:,4:8,:4]!=0))
            self.assertFalse(a.read_window('F444W','p1727_P0019',24,36,24,20)[1][4:8,:4].any())


if __name__=='__main__': unittest.main()
