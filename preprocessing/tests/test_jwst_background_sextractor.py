"""SExtractor reference sky: compact and extended sources, NaNs, cache changes."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import json
import numpy as np
from preprocessing.utils.jwst_background_sextractor import (
    AggressiveSkyConfig, aggressive_sextractor_background, METHOD)


class AggressiveSkyTests(unittest.TestCase):
    def test_multiscale_mask_and_cache_invalidation(self):
        rng=np.random.default_rng(25)
        n=512;y,x=np.mgrid[:n,:n]
        image=rng.normal(0,1,(n,n)).astype('float32')
        image+=20*np.exp(-((x-160)**2+(y-230)**2)/(2*12**2))
        image+=4*np.exp(-((x-385)**2+(y-360)**2)/(2*55**2))
        image[:12,:]=np.nan
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'sky.npz'
            sky=aggressive_sextractor_background(image,None,output,source='synthetic')
            self.assertEqual(sky.shape,image.shape)
            self.assertFalse(sky[230,160]);self.assertFalse(sky[360,385])
            self.assertTrue(sky[90,90]);self.assertFalse(sky[:12].any())
            first=json.loads(output.with_suffix('.json').read_text())
            self.assertEqual(first['signature']['method'],METHOD)
            self.assertEqual(len(first['passes']),3)
            with patch('preprocessing.utils.jwst_background_sextractor._detect_pass',
                       side_effect=AssertionError('cache missed')):
                np.testing.assert_array_equal(
                    aggressive_sextractor_background(image,None,output,source='synthetic'),sky)
            altered=image.copy();altered[90,90]+=1
            self.assertNotEqual(first['signature']['image_sha256'],
                 __import__('hashlib').sha256(np.ascontiguousarray(altered).view(np.uint8)).hexdigest())
            with patch('preprocessing.utils.jwst_background_sextractor._detect_pass',
                       side_effect=RuntimeError('image changed; recomputation requested')):
                with self.assertRaisesRegex(RuntimeError,'image changed'):
                    aggressive_sextractor_background(altered,None,output,source='synthetic')

    def test_empty_image_and_invalid_config(self):
        with self.assertRaises(ValueError):AggressiveSkyConfig(factors=(1,2))
        with tempfile.TemporaryDirectory() as tmp:
            result=aggressive_sextractor_background(np.full((16,16),np.nan),None,Path(tmp)/'empty.npz')
            self.assertFalse(result.any())


if __name__=='__main__':unittest.main()
