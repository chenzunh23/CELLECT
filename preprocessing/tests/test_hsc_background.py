import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from contextlib import ExitStack
import numpy as np
from preprocessing.utils.hsc_background import resolve_lsst_background, read_lsst_background


class HscBackgroundTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def product(self, kind='hsc_half', group='coadd'):
        source = self.root / 'science.fits'
        source.touch()
        variant = 'half_coadd' if kind == 'hsc_half' else 'noisy'
        row = dict(kind=kind, patch='4,5', band='HSC-G', group='0', source=str(source), name='test')
        folder = self.root / variant / '9813' / '4,5' / group / 'HSC-G'
        folder.mkdir(parents=True, exist_ok=True)
        mask = np.array([[True, False], [False, True]])
        np.savez(folder / 'background_mask.npz', background_mask=mask)
        (folder / 'summary.json').write_text(json.dumps(dict(status='ok', input=str(source), shape_yx=[2, 2], origin_xy=[100, 200])))
        return row, folder, mask

    def test_matching_variant_and_group(self):
        for kind, group in [('hsc_half','coadd'), ('hsc_half','0'), ('hsc_noisy','group_00')]:
            with self.subTest(kind=kind, group=group):
                row, folder, expected = self.product(kind, group)
                path, receipt = resolve_lsst_background(self.root, row, 9813)
                self.assertEqual(path, folder / 'background_mask.npz')
                np.testing.assert_array_equal(read_lsst_background(path, (2,2), (100,200), receipt), expected)

    def test_different_science_image_is_rejected(self):
        row, _, _ = self.product()
        row['source'] = str(self.root / 'another_group.fits')
        with self.assertRaisesRegex(ValueError, 'belongs to'):
            resolve_lsst_background(self.root, row, 9813)

    def test_half_alias_requires_provenance(self):
        row, folder, _ = self.product()
        (folder / 'summary.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Cannot verify'):
            resolve_lsst_background(self.root, row, 9813)

    def test_missing_variant_does_not_fall_back_to_full(self):
        row, _, _ = self.product()
        (self.root / 'half_coadd').rename(self.root / 'coadd')
        with self.assertRaises(FileNotFoundError):
            resolve_lsst_background(self.root, row, 9813)

    def test_mismatched_grid_is_rejected(self):
        row, _, _ = self.product()
        path, receipt = resolve_lsst_background(self.root, row, 9813)
        for shape, origin, match in [((3,2),(100,200),'shape'), ((2,2),(0,0),'origin')]:
            with self.subTest(match=match), self.assertRaisesRegex(ValueError, match):
                read_lsst_background(path, shape, origin, receipt)

    def test_invalid_npz_does_not_become_all_ignore(self):
        path = self.root / 'bad.npz'
        np.savez(path, unrelated=np.ones((2,2)))
        with self.assertRaisesRegex(ValueError, 'No background'):
            read_lsst_background(path, (2,2), (0,0))

    def test_batch_lsst_dispatch_uses_mask_and_provenance(self):
        from preprocessing import training_batch as batch
        row, _, expected = self.product()
        args = SimpleNamespace(output_root=str(self.root/'out'), data_root=str(self.root),
            hsc_products_root=str(self.root), hsc_background_method='lsst',
            variant_lsst_background_root=str(self.root), tract=9813)
        image = np.ones((2,2), np.float32)
        with ExitStack() as stack:
            stack.enter_context(patch.object(batch, 'task_for', return_value=SimpleNamespace()))
            stack.enter_context(patch.object(batch, 'replace', side_effect=lambda task, **kw: SimpleNamespace(**vars(task), **kw)))
            stack.enter_context(patch('preprocessing.utils.image_level._coadd_image_path', return_value=Path(row['source'])))
            stack.enter_context(patch('preprocessing.utils.image_level._read_image_header_origin', return_value=(image, {}, (100,200))))
            stack.enter_context(patch('preprocessing.utils.hsc_psf_ee.measure_psf'))
            classify = stack.enter_context(patch('preprocessing.build_image_level_zarr._classify_patch', return_value=object()))
            writer = stack.enter_context(patch('preprocessing.build_image_level_zarr.write_classified_patch', return_value={'output':'dummy'}))
            sex = stack.enter_context(patch('preprocessing.utils.jwst_background_sextractor.aggressive_sextractor_background', side_effect=AssertionError('SExtractor must not run')))
            stack.enter_context(patch('preprocessing.utils.hsc_quality.hsc_training_tile_quality', return_value=(np.ones((2,2),bool), [], {})))
            batch.hsc_job(args, row)
            sex.assert_not_called()
            np.testing.assert_array_equal(classify.call_args.kwargs['background_override'], expected)
            self.assertEqual(writer.call_args.kwargs['provenance']['hsc_background_method'], 'lsst')
            self.assertEqual(writer.call_args.kwargs['provenance']['background_method'], 'LSST on matching training input')


if __name__ == '__main__':
    unittest.main()
