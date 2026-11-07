"""HSC FITS grids must not depend on training Zarr coverage or noisy variants."""
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from astropy.io import fits

from eval.datasets.hsc_image import HscImageAccess
from eval.hsctiles.browser_core import make_access


class HscImageGridTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.image = np.arange(600 * 1040, dtype=np.float32).reshape(600, 1040)
        self.half = self.root / 'half/9813/HSC-I/4,5/warp_half-HSC-I-9813-4,5-0.fits'
        self.full = self.root / 'full/9813/HSC-I/4,5/calexp-HSC-I-9813-4,5.fits'
        self.noisy = self.root / 'noisy/9813/HSC-I/4,5/warp_8-HSC-I-9813-4,5-0.fits'
        for path, offset in [(self.half, 0), (self.full, 100), (self.noisy, 200)]:
            path.parent.mkdir(parents=True, exist_ok=True)
            hdr = fits.Header({'LTV1': -15900., 'LTV2': -19900.})
            fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(self.image + offset, header=hdr)]).writeto(path)

    def tearDown(self):
        self.tmp.cleanup()

    def access(self, full=False, **kwargs):
        return HscImageAccess(self.root / 'no_zarr', sources=['coadd'],
                              coadd_fits_root=self.root / ('full' if full else 'half'),
                              noisy_fits_root=self.root / 'noisy', cutout_plan=self.root/'plan.json', **kwargs)

    def ref(self, access, tile='x001_y001', slot=0):
        return access.make_ref(token='test', patch='4,5', band='HSC-I', tile_id=tile,
                               frame_slot=slot, frame_rank=0, frames_per_tile=1,
                               visit=None, strict_visit=False)

    def test_header_only_fixed_grid_without_zarr_and_without_noisy(self):
        a = self.access()
        with patch('eval.datasets.hsc_image.read_fits_image', side_effect=AssertionError('eager image read')):
            self.assertEqual(a.available_bands(), ['HSC-I'])
            self.assertEqual(a.available_patches(), ['4,5'])
            ids = a.choose_tiles('4,5', ['HSC-I'], n_tiles=None, all_tiles=True, seed=7)
            self.assertEqual(len(ids), 4)
            self.assertEqual(set(a._scan()), {('coadd', 'HSC-I', '4,5', '')})
            self.assertEqual(a.tile_slot_count('4,5', ids[0], ['HSC-I'], frames_per_tile=1, visit=None), 1)
            r = self.ref(a)
            self.assertEqual((r.x0, r.y0, r.width, r.height), (15900+528, 19900+88, 512, 512))
            self.assertEqual(a.input_fits_path(r), self.half)

    def test_half_full_actual_pixels_ltv_and_edge_padding(self):
        for full, offset in [(False, 0), (True, 100)]:
            a = self.access(full=full)
            with patch('preprocessing.image_processing.prepare_image', side_effect=lambda x, **kw: SimpleNamespace(image=x)):
                image = a.read_detection_frame(self.ref(a))
                np.testing.assert_array_equal(image, self.image[88:, 528:] + offset)
                self.assertEqual(image.shape, (512, 512))
                edge = a.read_frame(self.ref(a, 'x000_y000'))
                np.testing.assert_array_equal(edge, self.image[:512,:512] + offset)

    def test_latest_half_coadd_zarr_alias_and_filter(self):
        from astro_train_zarr_data import PatchZarrReader
        for source in ['half_coadd', 'noisy']:
            (self.root / 'zarr/hsc/image_level' / source / 'HSC-I/4,5.zarr').mkdir(parents=True)
        with patch.object(PatchZarrReader, '__init__', side_effect=RuntimeError('no pixels needed')):
            for root in [self.root / 'zarr', self.root / 'zarr/hsc', self.root / 'zarr/hsc/image_level']:
                a = HscImageAccess(root, sources=['coadd'], grid_mode='zarr')
                self.assertEqual(set(a._scan()), {('coadd', 'HSC-I', '4,5', '')})

    def test_access_factory_forwards_grid_and_sources(self):
        args = SimpleNamespace(hsc_image_root=self.root/'unused', hsc_sources=['coadd'],
                               hsc_grid_mode='fits', hsc_coadd_fits_root=self.root/'half',
                               hsc_cutout_plan=self.root/'factory_plan.json')
        a = make_access('hsc_image', args, '9813')
        self.assertEqual(a.grid_mode, 'fits')
        self.assertEqual(a.sources, ('coadd',))
        self.assertEqual(a.available_bands(), ['HSC-I'])

    def test_plan_contains_no_pixels_and_reuses_unchanged_headers(self):
        from astro_train_zarr_data import PatchZarrReader
        a = self.access()
        with patch.object(PatchZarrReader, '__init__', side_effect=AssertionError('Zarr opened')):
            a.prepare_cutout_plan()
            plan=json.loads(a.cutout_plan.read_text())
            self.assertEqual(plan['files'][0]['origin_xy'],[15900,19900])
            self.assertEqual(plan['files'][0]['shape_yx'],[600,1040])
            self.assertEqual(len(plan['grids']),1)
            with patch('astropy.io.fits.open',side_effect=AssertionError('unchanged header reread')):
                b=self.access()
                self.assertEqual(b.tile_by_id('HSC-I','4,5'),a.tile_by_id('HSC-I','4,5'))
        # A changed input must invalidate its cached header even at the same path.
        fits.setval(self.half,'LTV1',ext=1,value=-12345.)
        c=self.access();self.assertEqual(c.tile_by_id('HSC-I','4,5')['x000_y000'].x0,12345)

    def test_unsigned_small_fits_padding_and_science_hdu_selection(self):
        data=np.arange(300*400,dtype=np.uint16).reshape(300,400)
        fits.HDUList([fits.PrimaryHDU(np.ones((300,400),np.float32)),
            fits.ImageHDU(data,name='IMAGE')]).writeto(self.half,overwrite=True)
        a=self.access()
        with patch('preprocessing.image_processing.prepare_image',side_effect=lambda x,**kw:SimpleNamespace(image=x)):
            im=a.read_detection_frame(self.ref(a,'x000_y000'))
        np.testing.assert_array_equal(im[:300,:400],data)
        self.assertTrue(np.isnan(im[300:]).all())
        self.assertTrue(np.isnan(im[:,400:]).all())

    def test_native_plan_matches_4200_preprocessing_grid(self):
        from preprocessing.utils.inputs import make_tile_specs
        # Header-only discovery supports a sparse FITS without inspecting SCI.
        fits.HDUList([fits.PrimaryHDU(),fits.ImageHDU(np.zeros((4200,4200),np.float32),
            header=fits.Header({'LTV1':-15900.,'LTV2':-19900.}))]).writeto(self.half,overwrite=True)
        a=self.access();tiles=list(a.tile_by_id('HSC-I','4,5').values())
        expected=make_tile_specs(parent_origin=(15900,19900),image_shape=(4200,4200),
            tile_size=512,stride=368,compare_origin=None)
        self.assertEqual([(t.x0,t.y0) for t in tiles],[(t.x0,t.y0) for t in expected])
        self.assertEqual(len(tiles),121)
        t=a.tile_by_id('HSC-I','4,5')['x010_y007']
        self.assertEqual((t.x0,t.y0),(19588,22476))

    def test_explicit_legacy_zarr_mode_reports_incompatible_schema(self):
        (self.root/'legacy/9813/coadd/4,5.zarr').mkdir(parents=True)
        a=HscImageAccess(self.root/'legacy',grid_mode='zarr')
        with self.assertRaisesRegex(ValueError,'Legacy multi-band'):
            a.available_bands()

    def test_browser_scales_live_fits_without_reading_zarr_or_loading_model(self):
        from eval.hsctiles.browser_core import BrowserState
        from eval.datasets.array_cache import ArrayCache
        from eval.datasets.parallel_load import ParallelLoad
        from astro_train_zarr_data import PatchZarrReader
        state=object.__new__(BrowserState)
        state.access=self.access(full=True)
        state.dataset_id='hsc_image';state.jwst_input_scaling='training_parent'
        state._frame_cache=ArrayCache();state._image_load=ParallelLoad(1);state._scale_load=ParallelLoad(1)
        state.clip_threshold=5.;state.log_a=300.;state.log_high_percentile=99.5
        state.lupton_stretch=.5;state.lupton_q=20.;state.anscombe_clip=False;state.anscombe_scale=1000.
        with patch.object(PatchZarrReader,'__init__',side_effect=AssertionError('Zarr opened')):
            ref=self.ref(state.access)
            for mode in ['log_lupton','anscombe','zscore_no_upper']:
                state.scaling_mode=mode
                rgb=state._scaled_input_for_ref(ref)
                self.assertEqual(rgb.shape,(3,512,512))
                self.assertTrue(np.isfinite(rgb).all())


if __name__ == '__main__':
    unittest.main()
