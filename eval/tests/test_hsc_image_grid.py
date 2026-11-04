"""HSC FITS grids must not depend on training Zarr coverage or noisy variants."""
import tempfile
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
        return HscImageAccess(self.root / 'no_zarr', sources=['coadd'], grid_mode='fits',
                              coadd_fits_root=self.root / ('full' if full else 'half'),
                              noisy_fits_root=self.root / 'noisy', **kwargs)

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
            self.assertEqual(len(ids), 6)
            self.assertEqual(set(a._scan()), {('coadd', 'HSC-I', '4,5', '')})
            self.assertEqual(a.tile_slot_count('4,5', ids[0], ['HSC-I'], frames_per_tile=1, visit=None), 1)
            r = self.ref(a)
            self.assertEqual((r.x0, r.y0, r.width, r.height), (512, 512, 512, 512))
            self.assertEqual(a.input_fits_path(r), self.half)

    def test_half_full_actual_pixels_ltv_and_edge_padding(self):
        for full, offset in [(False, 0), (True, 100)]:
            a = self.access(full=full)
            with patch('preprocessing.image_processing.prepare_image', side_effect=lambda x, **kw: SimpleNamespace(image=x)):
                image = a.read_detection_frame(self.ref(a))
                np.testing.assert_array_equal(image[:88, :], self.image[512:, 512:1024] + offset)
                self.assertTrue(np.isnan(image[88:, :]).all())
                self.assertEqual(image.shape, (512, 512))
                edge = a.read_frame(self.ref(a, 'x002_y000'))
                np.testing.assert_array_equal(edge[:, :16], self.image[:512, 1024:] + offset)
                self.assertTrue(np.isnan(edge[:, 16:]).all())

    def test_latest_half_coadd_zarr_alias_and_filter(self):
        from astro_train_zarr_data import PatchZarrReader
        for source in ['half_coadd', 'noisy']:
            (self.root / 'zarr/hsc/image_level' / source / 'HSC-I/4,5.zarr').mkdir(parents=True)
        with patch.object(PatchZarrReader, '__init__', side_effect=RuntimeError('no pixels needed')):
            for root in [self.root / 'zarr', self.root / 'zarr/hsc', self.root / 'zarr/hsc/image_level']:
                a = HscImageAccess(root, sources=['coadd'])
                self.assertEqual(set(a._scan()), {('coadd', 'HSC-I', '4,5', '')})

    def test_access_factory_forwards_grid_and_sources(self):
        args = SimpleNamespace(hsc_image_root=self.root/'unused', hsc_sources=['coadd'],
                               hsc_grid_mode='fits', hsc_coadd_fits_root=self.root/'half')
        a = make_access('hsc_image', args, '9813')
        self.assertEqual(a.grid_mode, 'fits')
        self.assertEqual(a.sources, ('coadd',))
        self.assertEqual(a.available_bands(), ['HSC-I'])


if __name__ == '__main__':
    unittest.main()
