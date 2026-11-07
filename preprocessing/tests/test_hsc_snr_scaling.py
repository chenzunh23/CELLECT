"""Half/noisy depth must change SNR classes without using full-coadd depth."""
import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from astropy.io import fits
from astropy.table import Table
import zarr

from preprocessing.snr import (SnrConfig, ap2_snr, compute_snr_for_sample,
    hsc_snr_job_metadata, predicted_snr_from_weight_ratio)
from preprocessing.build_image_level_zarr import parse_args, write_classified_patch
from preprocessing.tests.test_segmentation_zarr import task, labels


class HalfRelativeSnrTests(unittest.TestCase):
    def fixture(self, root, *, export_half=True):
        band, patch_name = 'HSC-I', '4,5'
        half_dir = root/'half_coadd/9813'/band/patch_name
        noisy_dir = root/'noisy/9813'/band/patch_name
        half_dir.mkdir(parents=True); noisy_dir.mkdir(parents=True)
        half = half_dir/f'warp_half-{band}-9813-{patch_name}-0.fits'
        noisy = noisy_dir/f'warp_noisy-{band}-9813-{patch_name}-0.fits'
        for image, n, kind in [(half, 10, 'half'), (noisy, 2, 'noisy')]:
            fits.PrimaryHDU(np.ones((24,24), np.float32)).writeto(image)
            fits.PrimaryHDU(np.full((24,24), n, np.float32)).writeto(
                image.parent/f'effective_count_{kind}-{band}-9813-{patch_name}-0.fits')
        weight_csv = root/'warp_weights/9813'/band/patch_name/'weights.csv'
        weight_csv.parent.mkdir(parents=True)
        rows = [dict(product_collection=kind, group_num=0, product_image_mask_fits=str(image),
                     weight=1, visit=i) for kind, image, n in
                [('half_coadd',half,10),('noisy',noisy,2)] for i in range(n)
                if export_half or kind != 'half_coadd']
        with weight_csv.open('w') as handle:
            writer=csv.DictWriter(handle, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        table = Table(dict(base_CircularApertureFlux_6_0_instFlux=[8.,12.,20.],
                           base_CircularApertureFlux_6_0_instFluxErr=[1.,1.,1.],
                           base_SdssCentroid_x=[10.,11.,12.],base_SdssCentroid_y=[10.,11.,12.]))
        config=SnrConfig(coadd_weight_root=root/'warp_weights',denoised_fits_root=root)
        kwargs=dict(band=band,patch=patch_name,group=0,is_narrow_band=False,
                    config=config,coadd_image_fits=root/'full_should_never_be_opened.fits')
        return table, half, noisy, kwargs

    def test_half_identity_no_metadata_and_noisy_reclassification(self):
        with tempfile.TemporaryDirectory() as tmp:
            table, half, noisy, kwargs=self.fixture(Path(tmp))
            baseline=compute_snr_for_sample(table,dataset_source='half_coadd',image_fits=half,**kwargs)
            np.testing.assert_array_equal(baseline.snr,ap2_snr(table))
            self.assertEqual(baseline.method,'half_coadd_ap2')
            result=compute_snr_for_sample(table,dataset_source='noisy',image_fits=noisy,**kwargs)
            np.testing.assert_allclose(result.snr,ap2_snr(table)*np.sqrt(.2))
            np.testing.assert_array_equal(result.snr_class,['ignore','weak_shape','clean'])
            self.assertEqual(result.method,'weight_half_relative')
            self.assertEqual(result.details['coadd_weight_count'],10)

    def test_local_coverage_is_relative_to_half_and_capped(self):
        snr,t=predicted_snr_from_weight_ratio(np.full(4,10.),coadd_weight_sum=10,
            selected_weight_sum=2,selected_weight_count=2,
            local_effective_count=np.array([2.,1.,1.,2.]),
            reference_effective_count=np.array([10.,10.,5.,0.]),reference_weight_count=10)
        np.testing.assert_allclose(t,[.2,.1,.2,np.nan],equal_nan=True)
        np.testing.assert_allclose(snr[:3],10*np.sqrt([.2,.1,.2]))
        _,t=predicted_snr_from_weight_ratio(np.ones(1),coadd_weight_sum=1,
            selected_weight_sum=2,selected_weight_count=2,local_effective_count=np.array([2.]),
            reference_effective_count=np.array([1.]),reference_weight_count=1)
        self.assertEqual(t[0],1.)

    def test_missing_half_csv_uses_exact_product_manifest_or_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);table,half,noisy,kwargs=self.fixture(root,export_half=False)
            with self.assertRaisesRegex(RuntimeError,'No half-coadd weights'):
                compute_snr_for_sample(table,dataset_source='noisy',image_fits=noisy,**kwargs)
            run=root/'_runs/test';summary=run/'analysis_summaries/4,5/half/HSC-I/source_summary.json'
            summary.parent.mkdir(parents=True);manifest=run/'inputs.json'
            manifest.write_text(json.dumps(dict(band='HSC-I',patch='4,5',
                selected_inputs=[dict(weight=1.,visit=i) for i in range(10)])))
            summary.write_text(json.dumps(dict(outputs=dict(product_image_mask_fits=str(half)),
                n_inputs=10,input_manifest=str(manifest))))
            result=compute_snr_for_sample(table,dataset_source='noisy',image_fits=noisy,**kwargs)
            np.testing.assert_allclose(result.t_eff,.2)
            self.assertIn(str(manifest),result.details['metadata_paths'])
            args=parse_args(['--output-root',str(root/'out'),'--hsc-products-root',tmp,
                             '--coadd-weight-root',str(root/'warp_weights')])
            metadata=hsc_snr_job_metadata(args,dict(kind='hsc_noisy',band='HSC-I',patch='4,5',group=0,source=str(noisy)))
            self.assertEqual(metadata['missing_policy'],'error')
            self.assertIn(str(manifest),metadata['metadata_paths'])

    def test_written_zarr_records_actual_snr_method(self):
        lab=labels();lab.table.meta['hsc_snr']=dict(method='weight_half_relative',reference='half_coadd',
                                                 factor_quantiles=[.4,.5,.6])
        with tempfile.TemporaryDirectory() as tmp, patch(
                'preprocessing.build_image_level_zarr._scale_image_chw',
                side_effect=lambda image,t:np.repeat(image[None],3,axis=0)):
            result=write_classified_patch(replace(task(tmp),band='HSC-I'),np.ones((32,32),np.float32),lab)
            group=zarr.open_group(result['output'],mode='r')
            self.assertEqual(group.attrs['hsc_snr'],lab.table.meta['hsc_snr'])


if __name__=='__main__':unittest.main()
