"""Header-only search and bounded, reusable COSMOS parent reads."""
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.io import fits
from eval.datasets import jwst_fields
from eval.datasets.jwst_fields import CosmosAccess
from eval.hsctiles.browser_core import BrowserState
from eval.tests.test_jwst_fields import header


class LazyCosmosTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        data = np.arange(70*70, dtype=np.float32).reshape(70, 70)
        data[:16, :16] = np.nan
        bad = np.zeros_like(data, dtype=np.uint8)
        bad[20, 20] = 1
        path = self.root / 'science.fits'
        fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data, header(70), name='SCI'),
                      fits.ImageHDU(bad, name='TRAIN_BAD')]).writeto(path)
        quality = self.root/'quality.npz'
        np.savez(quality, bad=bad.astype(bool))
        (self.root/'manifest.json').write_text(json.dumps({'items': [dict(
            image_fits=str(path), quality_mask_npz=str(quality), sample_name='test', proposal=1727, pointing=19, band='F444W')]}))
        self.a = CosmosAccess(self.root, tile_size=16, lazy_browser=True)
        self.a.parent_size = 32  # Exercise the same tiling with small fixtures.
        self.p = 'p1727_P0019'
        self.a.available_patches()

    def tearDown(self):
        self.tmp.cleanup()

    def test_full_index_search_and_detection_page_are_metadata_only(self):
        with patch.object(jwst_fields, 'fits_window', side_effect=AssertionError('eager science read')):
            ids = self.a.choose_tiles(self.p, ['F444W'], n_tiles=1, all_tiles=False, seed=7)
            self.assertEqual(len(ids), 25)  # Includes empty and partially padded tiles.
            pages = self.a.browse_pages(self.p, ['F444W'], 3)
            self.assertEqual([i for page in pages for i in page], ids)
            for page in pages:
                parents = {(self.a.tile_by_id('F444W', self.p)[i].x0//32,
                            self.a.tile_by_id('F444W', self.p)[i].y0//32) for i in page}
                self.assertEqual(len(parents), 1)
            refs = [self.a.make_ref(token=i, patch=self.p, band='F444W', tile_id=i,
                     frame_slot=0, frame_rank=0, frames_per_tile=1, visit=None, strict_visit=False) for i in ids]
            s = BrowserState.__new__(BrowserState)
            s.patch = self.p
            s.tiles_per_page = 3
            s._tile_pages_by_patch = {self.p: pages}
            s.selected_tiles_by_patch = {self.p: ids}
            s.refs_by_patch = {self.p: refs}
            s.frame_slots_by_patch = {self.p: dict.fromkeys(ids, 1)}
            for ref in refs:
                result = s.find_tile('pixel_xy', ref.x0, ref.y0)
                page = result['page']
                self.assertIn(ref.tile_id, s._page_tile_ids(page))
                self.assertIn((ref.tile_id, 0), s._page_tile_slots(page))
            self.assertEqual(s.find_tile('tile_xy', 4, 4)['tile_id'], 'x004_y004')

    def test_parent_reuse_seams_padding_quality_and_cache_bound(self):
        with patch.object(jwst_fields, 'fits_window', wraps=jwst_fields.fits_window) as read:
            self.a._read_window('F444W', self.p, 0, 0, 16, 16)
            self.a._read_window('F444W', self.p, 16, 16, 16, 16)
            self.assertEqual(read.call_count, 1)
            actual = self.a._read_window('F444W', self.p, 20, 20, 20, 20)
            self.assertEqual(read.call_count, 4)
            edge = self.a._read_window('F444W', self.p, 64, 64, 16, 16)
            self.assertTrue(np.isnan(edge[0][6:, :]).all())
            self.assertFalse(actual[1][0, 0])
            self.assertTrue(all(c.args[-2:] == (32, 32) for c in read.call_args_list))
        expected = jwst_fields.fits_window(self.a._sources[(self.p, 'f444w')], 20, 20, 20, 20)
        np.testing.assert_allclose(actual[0], expected[0], equal_nan=True)
        np.testing.assert_array_equal(actual[1], expected[1])
        self.assertEqual(actual[2]['CRPIX1'], expected[2]['CRPIX1'])
        self.a._parent_pixels.max_bytes = 32*32*5
        self.a._read_window('F444W', self.p, 64, 0, 16, 16)
        self.assertLessEqual(self.a._parent_pixels.bytes, self.a._parent_pixels.max_bytes)

    def test_concurrent_requests_share_parent_read(self):
        with patch.object(jwst_fields, 'fits_window', wraps=jwst_fields.fits_window) as read:
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda _: self.a._read_window('F444W', self.p, 16, 16, 16, 16), range(8)))
            self.assertEqual(read.call_count, 1)


if __name__ == '__main__':
    unittest.main()
