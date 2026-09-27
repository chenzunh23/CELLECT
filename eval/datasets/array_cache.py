"""Small thread-safe byte-bounded LRU for browser arrays (no full mosaics)."""
from collections import OrderedDict
from threading import RLock
import numpy as np


def nbytes(value):
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, (tuple, list)):
        return sum(nbytes(v) for v in value)
    if isinstance(value, dict):
        return sum(nbytes(v) for v in value.values())
    return 0


class ArrayCache:
    def __init__(self, max_bytes=128 * 1024**2):
        self.max_bytes = max_bytes
        self.bytes = 0
        self.values = OrderedDict()
        self.lock = RLock()

    def get(self, key):
        with self.lock:
            item = self.values.get(key)
            if item is None:
                return None
            self.values.move_to_end(key)
            return item[0]

    def put(self, key, value):
        size = nbytes(value)
        with self.lock:
            old = self.values.pop(key, None)
            if old is not None:
                self.bytes -= old[1]
            if size > self.max_bytes:
                return value
            while self.values and self.bytes + size > self.max_bytes:
                _, (_, freed) = self.values.popitem(last=False)
                self.bytes -= freed
            self.values[key] = (value, size)
            self.bytes += size
        return value
