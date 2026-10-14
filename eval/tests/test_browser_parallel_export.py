"""Export concurrency, format preservation and SNR numerical equivalence."""
import csv
import io
import json
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

from eval.datasets.array_cache import ArrayCache
from eval.datasets.base import FrameRef
from eval.datasets.parallel_load import ParallelLoad
from eval.hsctiles.browser_core import BrowserState
from eval.hsctiles import browser_core
from eval.hsctiles.browser_masks import decode_masks
from utils import source_snr


def make_state(root, count=6):
    state = object.__new__(BrowserState)
    state.patch = 'p'
    refs = [FrameRef(token=f't{i}', root='/fake/source', tract='default', patch='p', band='F115W',
                     pack_path='fake.fits', tile_index=i, tile_id=f'x{i:03d}_y000',
                     x0=0, y0=0, x1=32, y1=32, frame_slot=0, frame_rank=0,
                     frame_index=0, tile_length=1, visit=None, weight=None, scale=None)
            for i in range(count)]
    state.refs_by_patch = {'p': refs}
    state.bands_by_patch = {'p': ['F115W']}
    state.bands = ['F115W']
    state.ref_by_token = {ref.token: ref for ref in refs}
    state.selected = {ref.token for ref in refs}
    state.export_dir = Path(root)
    state.export_workers = 4
    state.detect_batch_size = 80
    state._export_lock = threading.Lock()
    state._export_progress_lock = threading.Lock()
    state._export_progress = {'stage': 'idle', 'completed': 0, 'total': 0}
    state._mask_lock = threading.RLock()
    state._mask_cache = ArrayCache(0)  # Force eviction; sink must pin the masks.
    state._frame_cache = ArrayCache(1024**2)
    state._image_load = ParallelLoad(4)
    image = np.random.default_rng(41).normal(size=(32, 32)).astype(np.float32)
    state.access = Mock(spec=['read_frame'])
    state.access.read_frame.return_value = image
    state.detect_rows_by_token = {}
    state.snr_rows_by_token_method = {}
    state.snr_config = source_snr.SourceSnrConfig(background_box=7)
    state.make_masks = True
    state.shape_overlay_centers = False
    state.scaling_mode = 'log_lupton'
    state.clip_threshold = 5.0
    state.checkpoint = Path('/fake/checkpoint.pt')
    state._inference_parameters = lambda: {'scaling_mode': 'log_lupton'}
    state._scaled_input_for_ref = lambda ref: np.stack([image, image, image])
    state._masks_for_token = Mock(side_effect=AssertionError('repeated GPU inference'))
    masks = decode_masks(Mock(spec=['forward_sam_masks']), {}, [], device=torch.device('cpu'), width=32, height=32)

    def detect(slots, refs_by_key, *, mask_sink=None):
        for slot in slots:
            for ref in refs_by_key[slot].values():
                state.detect_rows_by_token[ref.token] = []
                mask_sink[ref.token] = masks

    state._detect_tile_slots = detect
    return state


class ParallelExportTests(unittest.TestCase):
    def test_parallel_files_match_serial_and_order_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            serial = make_state(Path(tmp)/'serial')
            parallel = make_state(Path(tmp)/'parallel')
            serial.export_selected(workers=1)
            original = parallel._export_ref
            lock = threading.Lock()
            active = peak = 0

            def export(*args):
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                try:
                    time.sleep(0.02 * (6-int(args[0].token[1:])))
                    return original(*args)
                finally:
                    with lock:
                        active -= 1

            parallel._export_ref = export
            result = parallel.export_selected(workers=4)
            self.assertGreater(peak, 1)
            self.assertLessEqual(peak, 4)
            self.assertEqual(result['n_exported'], 6)
            for path in serial.export_dir.rglob('*'):
                if path.suffix in {'.png', '.npz', '.csv'}:
                    self.assertEqual(path.read_bytes(), (parallel.export_dir/path.relative_to(serial.export_dir)).read_bytes())
            rows = [json.loads(line) for line in (parallel.export_dir/'export_manifest.jsonl').read_text().splitlines()]
            self.assertEqual([row['token'] for row in rows], [ref.token for ref in parallel.refs])
            for row in rows:
                self.assertTrue(Path(row['mask_png_path']).exists())
            parallel._masks_for_token.assert_not_called()
            self.assertEqual(parallel.export_status()['stage'], 'done')

    def test_no_png_does_not_decode_or_measure_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = make_state(tmp, 2)
            state.export_selected(write_png=False, workers=2)
            self.assertEqual(len(list(Path(tmp).rglob('*.png'))), 0)
            state._masks_for_token.assert_not_called()
            self.assertFalse(state.snr_rows_by_token_method)

    def test_duplicate_export_rejected_and_failure_does_not_replace_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = make_state(tmp)
            state._export_lock.acquire()
            try:
                with self.assertRaisesRegex(RuntimeError, 'already running'):
                    state.export_selected()
            finally:
                state._export_lock.release()
            manifest = Path(tmp)/'export_manifest.jsonl'
            manifest.write_text('previous export\n')
            state._export_ref = Mock(side_effect=OSError('disk failure'))
            with self.assertRaisesRegex(OSError, 'disk failure'):
                state.export_selected()
            self.assertEqual(manifest.read_text(), 'previous export\n')
            self.assertEqual(state.export_status()['stage'], 'error')
            self.assertFalse(state._export_lock.locked())

    def test_selection_is_snapshotted_before_navigation(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = make_state(tmp)
            detect = state._detect_tile_slots

            def navigate(*args, **kwargs):
                state.patch = 'another_patch'
                state.selected.clear()
                detect(*args, **kwargs)

            state._detect_tile_slots = navigate
            state.export_selected(write_png=False)
            with (Path(tmp)/'selection.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 6)

    def test_shared_snr_matches_independent_measurements(self):
        rng = np.random.default_rng(123)
        image = rng.normal(size=(64,64)).astype(np.float32)
        image[:3,:5] = np.nan
        image[28:33,28:33] += 7
        rows = [dict(x=30., y=30., major=3., minor=2., theta=.3),
                dict(x=1., y=5., major=4., minor=3., theta=0.)]
        config = source_snr.SourceSnrConfig(background_box=7)
        expected = {method: source_snr.measure_source_snrs(image, rows, method=method, config=config)[0]
                    for method in ('ap2', 'kron')}
        with patch.object(source_snr, 'estimate_background', wraps=source_snr.estimate_background) as background:
            actual = source_snr.measure_source_snrs_both(image, rows, config=config)
            self.assertEqual(background.call_count, 1)
        for method in expected:
            for old, new in zip(expected[method], actual[method]):
                self.assertEqual(old.keys(), new.keys())
                for key in old:
                    if isinstance(old[key], (int, float)):
                        np.testing.assert_allclose(new[key], old[key], rtol=0, atol=0, equal_nan=True)
                    else:
                        self.assertEqual(new[key], old[key])
        with patch.object(source_snr, 'estimate_background', side_effect=AssertionError('empty detections')):
            self.assertEqual(source_snr.measure_source_snrs_both(image, [], config=config), {'ap2': [], 'kron': []})

    def test_detection_pins_masks_even_when_lru_cannot_store_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = make_state(tmp, 3)
            state.device_name = 'cpu'
            state.amp = 'none'
            state.confidence_threshold = 2.
            state.nms_radius = 2
            state.confidence_score = 'ordinal_expectation'
            state.center_refinement = 'none'
            state.center_refinement_radius = 1
            state.detect_png_by_token = {}
            state.input_shape_png_by_token = {}
            state.warnings = []
            state._load_model = lambda bands: (None, {})
            state._prepare_detection_inputs = lambda refs: [
                (state._raw_image_for_ref(ref), state._scaled_input_for_ref(ref)) for ref in refs]
            # Use the real batching/locking/store path, with only neural calls mocked.
            state._detect_tile_slots = BrowserState._detect_tile_slots.__get__(state)
            masks = {'packed': np.ones(64, np.uint8)}

            def store(*args):
                state._mask_cache.put(args[-1].token, masks)
                return masks

            state._store_masks = store
            sink = {}
            with patch.object(browser_core, 'infer_cellect', return_value={}), \
                 patch.object(browser_core, 'select_band_outputs', return_value={}), \
                 patch.object(browser_core, 'detection_rows', return_value=[]):
                state._detect_tile_slots(list(state._refs_by_tile_slot()), mask_sink=sink)
            self.assertEqual(set(sink), set(state.ref_by_token))
            self.assertEqual(state._mask_cache.bytes, 0)
            self.assertFalse(state.warnings)

    def test_snr_logs_stages_cache_and_does_not_change_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = make_state(tmp, 1)
            state.session_dir = Path(tmp)
            ref = state.refs[0]
            image = state._raw_image_for_ref(ref)
            rows = [dict(x=16., y=16., major=3., minor=2., theta=0.)]
            state.detect_rows_by_token[ref.token] = rows
            expected = source_snr.measure_source_snrs_both(image, rows, config=state.snr_config)
            output = io.StringIO()
            with redirect_stdout(output):
                measured, costs = state._measure_snr_rows_for_ref(ref, image, rows)
            log = output.getvalue()
            self.assertEqual((Path(tmp)/'snr.log').read_text(), log)
            for stage in ('background', 'ap2', 'kron', 'total'):
                self.assertIn(f'stage={stage} status=start', log)
                self.assertIn(f'stage={stage} status=done', log)
            self.assertIn('sources=1', log)
            self.assertIn('context=export', log)
            self.assertIn('cache=miss', log)
            for key in ('snr_background', 'snr_ap2', 'snr_kron'):
                self.assertGreaterEqual(costs[key], 0)
            for method in expected:
                for key, value in expected[method][0].items():
                    if isinstance(value, (int, float)):
                        np.testing.assert_allclose(measured[method][0][key], value, rtol=0, atol=0, equal_nan=True)
                    else:
                        self.assertEqual(measured[method][0][key], value)
            output = io.StringIO()
            with redirect_stdout(output), patch.object(browser_core, 'measure_source_snrs_both',
                                                       side_effect=AssertionError('cache miss')):
                state._measure_snr_rows_for_ref(ref, image, rows)
                state._snr_rows_for_token(ref.token, 'ap2')
            self.assertIn('status=cached', output.getvalue())
            self.assertIn('cache=hit', output.getvalue())
            self.assertIn('context=view', output.getvalue())
            self.assertNotIn('status=start', output.getvalue())

    def test_snr_partial_cache_reports_only_missing_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = make_state(tmp, 1)
            ref = state.refs[0]
            rows = [dict(x=16., y=16., major=3., minor=2., theta=0.)]
            state.detect_rows_by_token[ref.token] = rows
            state.snr_rows_by_token_method[(ref.token, 'ap2')] = []
            output = io.StringIO()
            with redirect_stdout(output):
                result, costs = state._measure_snr_rows_for_ref(ref, state._raw_image_for_ref(ref), rows)
            self.assertIn('methods=ap2 sources=0 stage=total status=cached', output.getvalue())
            self.assertIn('methods=kron sources=1 stage=background status=start', output.getvalue())
            self.assertNotIn('stage=ap2 status=start', output.getvalue())
            self.assertEqual(set(costs), {'snr_background', 'snr_kron'})
            self.assertEqual(result['ap2'], [])

    def test_snr_failure_and_empty_sources_are_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = make_state(tmp, 1)
            ref = state.refs[0]
            rows = [dict(x=16., y=16., major=3., minor=2., theta=0.)]
            state.detect_rows_by_token[ref.token] = rows
            output = io.StringIO()
            with redirect_stdout(output), patch.object(source_snr, 'background_residual',
                                                       side_effect=RuntimeError('background failure')):
                with self.assertRaisesRegex(RuntimeError, 'background failure'):
                    state._measure_snr_rows_for_ref(ref, state._raw_image_for_ref(ref), rows)
            self.assertIn('stage=background status=error', output.getvalue())
            self.assertIn('stage=total status=error', output.getvalue())
            output = io.StringIO()
            state.detect_rows_by_token[ref.token] = []
            with redirect_stdout(output):
                state._snr_rows_for_token(ref.token, 'kron')
            self.assertIn('status=skipped', output.getvalue())
            self.assertIn('reason=no_detections', output.getvalue())


if __name__ == '__main__':
    unittest.main()
