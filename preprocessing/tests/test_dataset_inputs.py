import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import zarr

from preprocessing.dataset_inputs import (
    ImageInput, discover_cosmos, discover_abell, load_image,
    reference_to_training, load_paths, bind_task,
)
from preprocessing.image_processing import read_fits_image
from preprocessing.build_image_level_zarr import parse_args, write_classified_patch
from preprocessing.tests.test_segmentation_zarr import task, labels


def image(path, value=1., *, primary=False, crpix=8., band='F115W', pupil='CLEAR', extra=()):
    h=fits.Header()
    h['CTYPE1']='RA---TAN';h['CTYPE2']='DEC--TAN'
    h['CRVAL1']=3.573;h['CRVAL2']=-30.376
    h['CRPIX1']=crpix;h['CRPIX2']=crpix
    h['CD1_1']=-.03/3600;h['CD2_2']=.03/3600
    h['CD1_2']=0.;h['CD2_1']=0.;h['BUNIT']='MJy/sr'
    h['FILTER']=band;h['PUPIL']=pupil
    data=np.full((32,32),value,np.float32)
    if primary:
        h['EXTNAME']='SCI';hd=[fits.PrimaryHDU(data,header=h)]
    else:hd=[fits.PrimaryHDU(),fits.ImageHDU(data,header=h,name='SCI')]
    fits.HDUList(hd+list(extra)).writeto(path)


class DatasetInputTests(unittest.TestCase):
    def test_cosmos_proposals_and_strict_quality_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);image(r/'a.fits');np.savez(r/'q.npz',bad=np.zeros((32,32),bool))
            rows=[dict(proposal=p,pointing=11,band='F115W',sample_name=f'cosmos_p0011_f115w_p{p}',
                       image_fits='a.fits',quality_mask_npz='q.npz') for p in [1727,5893]]
            (r/'manifest.json').write_text(json.dumps({'items':rows}))
            cfg=dict(manifest=str(r/'manifest.json'),catalog_root=str(r))
            sources=discover_cosmos(cfg)
            self.assertEqual(len(sources),2)
            self.assertNotEqual(sources[0].sample_name,sources[1].sample_name)
            self.assertEqual(sources[0].split_group,sources[1].split_group)
            self.assertEqual(discover_cosmos(cfg,proposals=[5893])[0].proposal,5893)
            (r/'q.npz').unlink()
            with self.assertRaises(FileNotFoundError):discover_cosmos(cfg)

    def test_abell_pairing_primary_hdu_pupil_and_wcs(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);half=r/'half';full=r/'full';half.mkdir();full.mkdir()
            name='field_f150w2_f162m_i2d_mbkg.fits'
            image(half/name,1,primary=True,crpix=10,band='F150W2',pupil='F162M')
            image(full/name,2,primary=True,crpix=8,band='F150W2',pupil='F162M')
            cfg=dict(image_root=str(half),reference_root=str(full),catalog_root=str(r))
            source=discover_abell(cfg)[0]
            self.assertEqual(source.band,'F162M')
            self.assertTrue(np.all(load_image(source,shape=(8,8)).image==1))
            self.assertTrue(np.all(load_image(source,role='reference',shape=(8,8)).image==2))
            self.assertTrue(np.all(read_fits_image(half/name)[0]==1))
            x,y=reference_to_training(source,[5.],[6.])
            np.testing.assert_allclose([x[0],y[0]],[7.,8.],atol=1e-7)
            cut=load_image(source,origin=(3,4),shape=(8,8))
            np.testing.assert_allclose(WCS(cut.header).all_pix2world([[0,0]],0),
                WCS(cut.full_header).all_pix2world([[3,4]],0),atol=1e-10)
            (full/name).unlink()
            with self.assertRaises(FileNotFoundError):discover_abell(cfg)

    def test_validity_mask_not_applied_to_reference_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);w=np.ones((32,32),np.float32);w[6,7]=0
            bad=np.zeros((32,32),np.uint8);bad[7,8]=1
            image(r/'half.fits',extra=[fits.ImageHDU(w,name='WHT'),fits.ImageHDU(bad,name='TRAIN_BAD')])
            image(r/'full.fits',2)
            q=np.zeros((32,32),bool);q[8,9]=True;np.savez(r/'q.npz',bad=q)
            s=ImageInput('abell','half','F115W','field',r/'half.fits',r/'full.fits',quality_mask_npz=r/'q.npz')
            loaded=load_image(s,origin=(5,5),shape=(6,6))
            self.assertEqual(int(loaded.bad.sum()),3)
            self.assertFalse(load_image(s,role='reference',origin=(5,5),shape=(6,6)).bad.any())
            with self.assertRaises(ValueError):load_image(s,origin=(30,30),shape=(8,8))
            np.savez(r/'q.npz',bad=np.zeros((3,3),bool))
            with self.assertRaises(ValueError):load_image(s,shape=(6,6))

    def test_scaled_unsigned_primary_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'unsigned.fits';a=np.arange(64,dtype=np.uint16).reshape(8,8)
            fits.PrimaryHDU(a).writeto(p)
            s=ImageInput('abell','unsigned','F115W','field',p,p)
            np.testing.assert_array_equal(load_image(s,origin=(1,2),shape=(3,4)).image,a[2:5,1:5])
            np.testing.assert_array_equal(read_fits_image(p)[0],a)

    def test_paths_override_cli_and_legacy_hsc(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'paths.json';p.write_text(json.dumps({'cosmos':{'manifest':'custom.json'},'hsc':{'data_root':'subaru'}}))
            cfg=load_paths(p)
            self.assertEqual(cfg['cosmos']['manifest'],str(p.parent/'custom.json'))
            args=parse_args(['--paths-config',str(p),'--data-root','/custom/hsc','--output-root',tmp])
            self.assertEqual(args.data_root,'/custom/hsc');self.assertEqual(args.datasets,['hsc'])
            self.assertIn('mosaic_v2_half',args.paths['abell']['image_root'])
            h=ImageInput('hsc','hsc','HSC-I','4,5',p,p)
            t=task(tmp);self.assertIs(bind_task(t,h),t)

    def test_proposals_write_to_distinct_zarr_stores(self):
        with tempfile.TemporaryDirectory() as tmp, patch('preprocessing.build_image_level_zarr._scale_image_chw',side_effect=lambda image,task:image[None]):
            results=[]
            for prop in [1727,5893]:
                src=ImageInput('cosmos',f'cosmos_p0011_f115w_p{prop}','F115W','0011',Path('/science.fits'),Path('/science.fits'),proposal=prop)
                result=write_classified_patch(task(tmp),np.ones((32,32)),labels(),input_source=src)
                store=zarr.open_group(result['output'],mode='r')
                self.assertEqual(store.attrs['proposal'],prop)
                self.assertEqual(store.attrs['image_fits'],'/science.fits')
                results.append(result['output'])
            self.assertNotEqual(*results)
            self.assertTrue(all(Path(p).is_dir() for p in results))


if __name__=='__main__':unittest.main()
