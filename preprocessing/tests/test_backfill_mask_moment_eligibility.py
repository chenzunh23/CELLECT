"""Metadata-only backfill rejects mismatched coordinates and preserves chunks."""
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest

import numpy as np
from astropy.table import Table

from astro_train_zarr_data import PatchZarrReader
from direct_zarr_preprocessing.zarr_writer import ZarrGroupWriter
from preprocessing.backfill_mask_moment_eligibility import (
    ARRAY, aligned_flags, commit_flags, process_one, snapshot,
)
from preprocessing.utils.mask_moments import POLICY


def fixture(root):
    store = root / 'test.zarr'
    writer = ZarrGroupWriter(store, overwrite=False, attrs=dict(
        hsc_refit_center_policy='fits1_to_numpy0_v1', custom='preserve'))
    ids = np.array([43158185032286339, 43158185032286340, 43158185032286341], np.int64)
    xy = np.array([[8., 9.], [18., 9.], [28., 9.]], np.float32)
    arrays = dict(shape_source_ids=np.tile(ids, 2),
                  shape_source_classes=np.tile(np.array([1, 2, 2], np.uint8), 2),
                  shape_source_centers=np.concatenate([xy, xy - [4, 0]]).astype(np.float32),
                  shape_source_values=np.tile(np.array([[4., 2., .3]], np.float32), (6, 1)),
                  shape_source_offsets=np.array([[0, 3], [3, 6]], np.int64),
                  tile_x0=np.array([100, 104]), tile_y0=np.array([200, 200]),
                  images=np.ones((2, 1, 3, 32, 32), np.float32))
    for name, data in arrays.items():
        writer.array(name, shape=data.shape, chunks=data.shape, dtype=data.dtype).write_full(data)
    labels = SimpleNamespace(source_ids=ids, label_classes=np.array([1, 2, 2]),
                             geom_x=xy[:, 0], geom_y=xy[:, 1],
                             geom_major=np.full(3, 4.), geom_minor=np.full(3, 2.),
                             geom_theta=np.full(3, .3),
                             table=Table(dict(classification_reason=[
                                 'ordinary_snr_clean', 'ordinary_snr_weak_shape',
                                 'isolated_bright_galaxy_weak_shape'])))
    return store, labels


class BackfillTests(unittest.TestCase):
    def test_large_int_ids_repeated_tiles_and_metadata_only_commit(self):
        with tempfile.TemporaryDirectory() as td:
            store, labels = fixture(Path(td))
            before = {str(p.relative_to(store)): p.read_bytes() for p in store.rglob('*') if p.is_file()}
            flags = aligned_flags(PatchZarrReader(store), labels, (100, 200))
            np.testing.assert_array_equal(flags, [1, 0, 1, 1, 0, 1])
            commit_flags(store, flags, snapshot(store))
            for name, data in before.items():
                if name != '.zattrs':
                    self.assertEqual((store / name).read_bytes(), data)
            self.assertEqual((store / '.zattrs.before_mask_moment_backfill').read_bytes(), before['.zattrs'])
            reader = PatchZarrReader(store)
            self.assertEqual(reader.attrs['custom'], 'preserve')
            self.assertEqual(reader.attrs['mask_moment_policy'], POLICY)
            np.testing.assert_array_equal(reader.read_full_small(ARRAY), flags)
            self.assertEqual(process_one((str(store), '/unused', True))['status'], 'already_done')

    def test_coordinate_class_shape_and_id_mismatch_rejected(self):
        for field in ['geom_x', 'label_classes', 'geom_major', 'geom_theta', 'source_ids']:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as td:
                store, labels = fixture(Path(td))
                setattr(labels, field, getattr(labels, field) + 1)
                with self.assertRaises(ValueError):
                    aligned_flags(PatchZarrReader(store), labels, (100, 200))
                self.assertFalse((store / ARRAY).exists())

    def test_concurrent_change_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            store, labels = fixture(Path(td))
            flags = aligned_flags(PatchZarrReader(store), labels, (100, 200))
            expected = snapshot(store)
            (store / '.zattrs').write_text(json.dumps(dict(changed=True)))
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                commit_flags(store, flags, expected)
            self.assertFalse((store / ARRAY).exists())

    def test_interrupted_array_publication_can_resume(self):
        with tempfile.TemporaryDirectory() as td:
            store, labels = fixture(Path(td))
            flags = aligned_flags(PatchZarrReader(store), labels, (100, 200))
            from direct_zarr_preprocessing.zarr_writer import ZarrArrayWriter
            ZarrArrayWriter(store, ARRAY, shape=flags.shape, chunks=flags.shape, dtype=np.uint8).write_full(flags)
            commit_flags(store, flags, snapshot(store))
            self.assertEqual(PatchZarrReader(store).attrs['mask_moment_policy'], POLICY)

    def test_old_coordinate_policy_rejected_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as td:
            store, _ = fixture(Path(td))
            (store / '.zattrs').write_text('{}')
            result = process_one((str(store), '/unused', True))
            self.assertEqual(result['status'], 'error')
            self.assertIn('regenerate', result['error'])
            self.assertFalse((store / ARRAY).exists())


if __name__ == '__main__':
    unittest.main()
