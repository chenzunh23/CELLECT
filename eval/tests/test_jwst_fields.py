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


if __name__=='__main__': unittest.main()
