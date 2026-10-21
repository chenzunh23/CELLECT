"""Training-image bright selection and exact HSC half-background provenance."""
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from astropy.table import Table
from preprocessing.image_processing import build_bright_components
from preprocessing.utils.abell_labels import classify_reference
from preprocessing.training_batch import precomputed_hsc_background
from eval.tests.test_jwst_fields import header


class HalfBrightBackgroundTests(unittest.TestCase):
    def inputs(self):
        names=['ALPHA_J2000','DELTA_J2000','KRON_RADIUS','A_IMAGE','B_IMAGE',
               'THETA_IMAGE','FLUX_AUTO','FLUXERR_AUTO','SNR_WIN','FLAGS','NUMBER']
        catalog=Table({k:np.empty(0,dtype='i8' if k in ('FLAGS','NUMBER') else 'f8') for k in names})
        gaia=Table({k:np.empty(0) for k in ('ra','dec','pmra','pmdec','phot_g_mean_mag','ref_epoch')})
        full=np.random.default_rng(13).normal(0,.01,(128,128)).astype('f4')
        half=full.copy();full[55:73,55:73]+=10;half[20:38,20:38]+=10
        return full,half,catalog,gaia

    def test_abell_half_controls_bright_while_ordinary_image_remains_full(self):
        from preprocessing.utils.abell_labels import classify_ordinary_sources
        full,half,catalog,gaia=self.inputs();captured=[]
        def bright(image,**kw):
            result=build_bright_components(image,**kw);captured.append(result[0]);return result
        with patch('preprocessing.utils.abell_labels.build_bright_components',side_effect=bright), \
             patch('preprocessing.utils.abell_labels.classify_ordinary_sources',wraps=classify_ordinary_sources) as ordinary:
            _,summary=classify_reference(full,header(128),catalog,gaia,np.ones(full.shape,bool),
                'F444W',training_background=np.ones(full.shape,bool),bright_image=half,bright_header=header(128))
        self.assertTrue(captured[0][25,25]);self.assertFalse(captured[0][64,64])
        np.testing.assert_array_equal(ordinary.call_args.kwargs['image'],full*np.float32(11.527661950741242))
        self.assertEqual(summary['bright_source'],'training_half_coadd')

    def test_abell_rejects_misaligned_training_bright(self):
        full,half,catalog,gaia=self.inputs();wrong=header(128);wrong['CRPIX1']+=1
        with self.assertRaisesRegex(ValueError,'align'):
            classify_reference(full,header(128),catalog,gaia,np.ones(full.shape,bool),'F444W',
                bright_image=half,bright_header=wrong)

    def test_precomputed_hsc_requires_matching_half_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'half.fits';source.write_bytes(b'science')
            row=dict(kind='hsc_half',band='HSC-I',patch='4,5',source=str(source))
            folder=root/'hsc/half_coadd/HSC-I/4,5/half';folder.mkdir(parents=True)
            stamp=source.stat()
            doc=dict(status='done',signature=dict(source=dict(path=str(source),bytes=stamp.st_size,mtime_ns=stamp.st_mtime_ns)),
                     job=dict(source=str(source),input_kind='half_coadd'))
            (folder/'complete.json').write_text(json.dumps(doc))
            args=SimpleNamespace(background_root=str(root))
            self.assertEqual(precomputed_hsc_background(args,row)[0],folder/'background_mask.npz')
            doc['job']['input_kind']='calexp';(folder/'complete.json').write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError,'variant'):precomputed_hsc_background(args,row)


if __name__=='__main__':unittest.main()
