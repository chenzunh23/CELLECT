"""Concurrency regressions: independent requests, dedup, errors, FITS safety."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event, Lock
from types import SimpleNamespace
import tempfile
import time
import unittest
import numpy as np
from astropy.io import fits
from eval.datasets.array_cache import ArrayCache
from eval.datasets.parallel_load import ParallelLoad
from eval.datasets.jwst import JwstNircamAccess
from eval.hsctiles.browser_core import BrowserState


class ParallelReadTests(unittest.TestCase):
    def test_parallelism_is_bounded(self):
        loader = ParallelLoad(4)
        lock = Lock()
        active = peak = 0
        rendezvous = Barrier(4, timeout=5)

        def load():
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            rendezvous.wait()
            with lock:
                active -= 1
            return 1

        with ThreadPoolExecutor(8) as pool:
            values = list(pool.map(lambda i: loader.run(i, load), range(8)))
        self.assertEqual(values, [1]*8)
        self.assertEqual(peak, 4)

    def test_duplicate_requests_read_once_and_failures_retry(self):
        loader = ParallelLoad(4)
        cache = ArrayCache(1024)
        calls = []
        start = Barrier(8, timeout=5)

        def load():
            calls.append(1)
            time.sleep(.05)
            return np.arange(4)

        def request(i):
            start.wait()
            return loader.cached(cache, "same", load)

        with ThreadPoolExecutor(8) as pool:
            arrays = list(pool.map(request, range(8)))
        self.assertEqual(len(calls), 1)
        self.assertTrue(all(a is arrays[0] for a in arrays))
        with self.assertRaisesRegex(ValueError, 'failed'):
            loader.run('error', lambda: (_ for _ in ()).throw(ValueError('failed')))
        self.assertEqual(loader.run('error', lambda: 2), 2)

    def test_slow_scaling_does_not_block_raw_image(self):
        state = object.__new__(BrowserState)
        state._frame_cache = ArrayCache(1024)
        state._image_load = ParallelLoad(1)
        state._scale_load = ParallelLoad(1)
        state.access = SimpleNamespace(read_frame=lambda ref: np.ones((2,2)))
        entered, release = Event(), Event()

        def slow():
            entered.set()
            if not release.wait(5):
                raise TimeoutError()

        with ThreadPoolExecutor(2) as pool:
            scaling = pool.submit(state._scale_load.run, 'parent', slow)
            self.assertTrue(entered.wait(5))
            try:
                raw = pool.submit(state._raw_image_for_ref, SimpleNamespace(token='raw'))
                self.assertEqual(raw.result(timeout=2).shape, (2,2))
            finally:
                release.set()
            scaling.result()

    def test_concurrent_first_fits_reads_share_safe_handle(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'test.fits'
            data = np.arange(128*128, dtype=np.float32).reshape(128,128)
            fits.PrimaryHDU(data).writeto(path)
            access = JwstNircamAccess(Path(tmp))
            def read(i):
                return access._read_section(path, slice(i,i+8), slice(10,30))
            with ThreadPoolExecutor(4) as pool:
                arrays = list(pool.map(read, range(16)))
            for i, arr in enumerate(arrays):
                np.testing.assert_array_equal(arr, data[i:i+8,10:30])
            self.assertEqual(len(access._hdul_cache), 1)
            access.close()


if __name__ == '__main__':
    unittest.main()
