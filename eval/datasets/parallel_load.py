"""Bounded I/O concurrency and per-key single-flight, without resident pools.

HTTP threads perform the work. The registry lock never covers I/O. Factories
must not recursively enter the same loader (resolve dependencies beforehand).
"""
from concurrent.futures import Future
from threading import BoundedSemaphore, Lock
from time import perf_counter


class ParallelLoad:
    def __init__(self, workers=4, name="image-load"):
        self.workers = max(1, int(workers))
        self.name = name
        self._slots = BoundedSemaphore(self.workers)
        self._lock = Lock()
        self._pending = {}

    def run(self, key, factory):
        with self._lock:
            future = self._pending.get(key)
            owner = future is None
            if owner:
                future = self._pending[key] = Future()
        if not owner:
            return future.result()
        start = perf_counter()
        try:
            with self._slots:
                entered = perf_counter()
                result = factory()
                elapsed = perf_counter() - entered
            future.set_result(result)
            if elapsed + entered - start >= 1:
                print(f"[{self.name}] key={key!r} wait={entered-start:.2f}s work={elapsed:.2f}s", flush=True)
            return result
        except BaseException as exc:
            future.set_exception(exc)
            raise
        finally:
            with self._lock:
                self._pending.pop(key, None)

    def cached(self, cache, key, factory):
        value = cache.get(key)
        if value is not None:
            return value

        def load():
            value = cache.get(key)  # Another caller may have just finished.
            return value if value is not None else cache.put(key, factory())

        return self.run(key, load)
