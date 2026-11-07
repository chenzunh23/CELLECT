"""JWST batch mode must reach both writers and invalidate EE job receipts."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import zarr
from astropy.io import fits

from preprocessing.build_image_level_zarr import parse_args, write_classified_patch
from preprocessing.training_batch import task_for, abell_job, run_job
from preprocessing.abell_zarr import store_task
from preprocessing.tests.test_segmentation_zarr import task as fixture_task, labels
from preprocessing.utils.image_level import _paint_confidence


def args(root, mode='manhattan'):
    return parse_args(['--training-batch','--training-kinds','cosmos','abell',
        '--output-root',str(root),'--confidence-mode',mode])


class JwstManhattanBatchTests(unittest.TestCase):
    def test_cosmos_dispatch_to_written_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=args(tmp)
            t=task_for(a,band='F115W',dataset_source='half_coadd')
            self.assertEqual(t.confidence_mode,'manhattan')
            self.assertEqual(task_for(args(tmp,'psf-ee'),band='F115W').confidence_mode,'psf-ee')
            lab=labels();lab.strict_x=np.array([8.2]);lab.strict_y=np.array([8.3]);lab.strict_ids=np.array([-9])
            t=replace(fixture_task(tmp),band='F115W',confidence_mode=t.confidence_mode)
            with patch('preprocessing.build_image_level_zarr._scale_image_chw',
                       side_effect=lambda image,task:np.repeat(image[None],3,axis=0)):
                result=write_classified_patch(t,np.ones((32,32),np.float32),lab)
            group=zarr.open_group(result['output'],mode='r')
            self.assertEqual(group.attrs['confidence_config']['mode'],'manhattan')
            expected=np.zeros((16,16),np.uint8)
            _paint_confidence(expected,np.zeros_like(expected,dtype=np.float32),np.array([[8.2,8.3]],np.float32))
            np.testing.assert_array_equal(group['band_confidence'][0,0],expected)

    def test_abell_batch_copy_preserves_requested_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            for mode in ['manhattan','psf-ee']:
                with self.subTest(mode=mode):
                    a=args(tmp,mode)
                    ref=Path(tmp)/'abell/references/F115W/x+00_y+00_full.fits'
                    ref.parent.mkdir(parents=True,exist_ok=True);ref.touch()
                    source=SimpleNamespace(reference_fits=Path(tmp)/'full.fits',image_fits=Path(tmp)/'half.fits')
                    row={'job':dict(tile={},band='F115W',name='x+00_y+00',grid_reference='grid.fits',output=tmp)}
                    with patch('preprocessing.dataset_inputs.discover_abell',return_value=[source]), \
                         patch('preprocessing.training_batch.header_info',return_value=(fits.Header(),(32,32),0)), \
                         patch('preprocessing.utils.sky_tiles.tile_header',return_value=fits.Header()), \
                         patch('preprocessing.utils.sky_tiles.SkyTile',return_value=object()), \
                         patch('preprocessing.abell_zarr.build_parent',side_effect=lambda aa,r:store_task(aa,r)):
                        task=abell_job(a,row)
                    self.assertEqual(task.confidence_mode,mode)
                    self.assertEqual(a.confidence_mode,mode)

    def test_switching_jwst_mode_rebuilds_cached_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=args(root)
            paths={name:root/name for name in ['science.fits','quality.npz','background.json','gaia.fits','COSMOSWeb_mastercatalog_v1.1.fits']}
            for p in paths.values():p.write_bytes(b'test fixture')
            a.paths['cosmos']['gaia_fits']=str(paths['gaia.fits'])
            row=dict(kind='cosmos',name='one_parent',source=dict(image_fits=str(paths['science.fits']),
                quality_mask_npz=str(paths['quality.npz']),catalog_root=str(root)),background=str(paths['background.json']))
            output=root/'one.zarr';output.mkdir();Path(str(output)+'_manifest.json').write_text('{}')
            with patch('preprocessing.training_batch.cosmos_parent',return_value={'output':str(output)}) as writer, \
                 patch('preprocessing.utils.jwst_star_psf.star_policy',return_value={'test':True}):
                run_job(a,row)
                self.assertEqual(run_job(a,row)['status'],'cached')
                self.assertEqual(writer.call_count,1)
                a.confidence_mode='psf-ee'
                run_job(a,row)
                self.assertEqual(writer.call_count,2)
                self.assertEqual(run_job(a,row)['status'],'cached')


if __name__=='__main__':unittest.main()
