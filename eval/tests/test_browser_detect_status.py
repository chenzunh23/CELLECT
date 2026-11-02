"""Regression checks for visible detection failures and threaded WCS warnings."""
import io
import sys
import tempfile
import threading
import types
import unittest
import warnings
from pathlib import Path
from unittest.mock import Mock
from astropy import log
from astropy.utils.exceptions import AstropyWarning
from eval.hsctiles.browser_core import BrowserState
from eval.hsctiles.serve_hsctile_pack_browser import Server, _configure_threaded_warnings


class DetectStatusTests(unittest.TestCase):
    def test_session_cannot_be_replaced_during_detection(self):
        server = object.__new__(Server)
        server._session_lock = threading.Lock()
        server._start_browser = Mock()
        state = types.SimpleNamespace(session_name='current')
        def detect(page):
            with self.assertRaisesRegex(RuntimeError, 'already running'):
                server.start_browser({})
            self.assertEqual(page, 2)
            return {'n_images': 10, 'n_detections': 900}
        state.detect_page = detect
        server.browser_state = state
        result = server.detect_page({'page': 2, 'session_name': 'current'})
        self.assertEqual(result['session_name'], 'current')
        server._start_browser.assert_not_called()
        self.assertFalse(server._session_lock.locked())

    def test_stale_session_is_rejected_before_inference(self):
        server = object.__new__(Server)
        server._session_lock = threading.Lock()
        server.browser_state = types.SimpleNamespace(session_name='new', detect_page=Mock())
        with self.assertRaisesRegex(RuntimeError, 'session changed'):
            server.detect_page({'page': 0, 'session_name': 'old'})
        server.browser_state.detect_page.assert_not_called()
        self.assertFalse(server._session_lock.locked())

    def test_simultaneous_start_is_rejected_and_failure_releases_lock(self):
        server = object.__new__(Server)
        server._session_lock = threading.Lock()
        def start(payload):
            with self.assertRaisesRegex(RuntimeError, 'already running'):
                server.start_browser(payload)
            raise ValueError('bad configuration')
        server._start_browser = start
        with self.assertRaisesRegex(ValueError, 'bad configuration'):
            server.start_browser({})
        self.assertFalse(server._session_lock.locked())

    def test_warning_module_scan_can_mutate_but_standard_handler_is_safe(self):
        class ImportDuringOriginLookup(types.ModuleType):
            def __getattribute__(self, name):
                if name == '__file__':
                    sys.modules['_browser_warning_imported'] = types.ModuleType('_browser_warning_imported')
                return super().__getattribute__(name)
        sys.modules['_browser_warning_origin'] = ImportDuringOriginLookup('_browser_warning_origin')
        try:
            # Reproduce the installed Astropy logger's live-dictionary failure.
            with self.assertRaisesRegex(RuntimeError, 'dictionary changed size'):
                log._showwarning(AstropyWarning('test'), AstropyWarning, '/nonexistent/origin.py', 1)
            _configure_threaded_warnings()
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                warnings.warn('still visible', AstropyWarning)
            self.assertEqual(len(captured), 1)
            self.assertEqual(str(captured[0].message), 'still visible')
        finally:
            sys.modules.pop('_browser_warning_origin', None)
            sys.modules.pop('_browser_warning_imported', None)

    def test_failed_load_reports_zero_successful_images(self):
        state = object.__new__(BrowserState)
        state.patch = 'test'
        state._bands_for_patch = lambda patch: ['F115W']
        state.device_name = 'cpu'
        state._load_model = Mock(side_effect=ValueError('bad checkpoint'))
        state.detect_rows_by_token = {}
        state.warnings = []
        ref = types.SimpleNamespace(token='p1', patch='test')
        refs = {('x000_y000', 0): {'F115W': ref}}
        with tempfile.TemporaryDirectory() as tmp:
            state.session_dir = Path(tmp)
            result = state._detect_tile_slots_locked([('x000_y000', 0)], refs)
            self.assertEqual(result['n_images'], 0)
            self.assertEqual(result['n_failed_images'], 1)
            self.assertIn('bad checkpoint', result['errors'][0])
            self.assertEqual(state.detect_status()['stage'], 'failed')
            self.assertIn('stage=failed', (Path(tmp)/'detect.log').read_text())

    def test_status_poll_does_not_acquire_inference_lock(self):
        state = object.__new__(BrowserState)
        state._mask_lock = threading.Lock()
        state._set_detect_progress('inference', completed=0, total=10)
        with state._mask_lock:
            self.assertEqual(state.detect_status()['stage'], 'inference')


if __name__ == '__main__':
    unittest.main()
