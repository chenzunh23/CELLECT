"""Local stack snapshots for stalled diagnostic training, without ptrace."""
from __future__ import annotations

import faulthandler
import functools
import json
import os
import signal
import threading
import time
from pathlib import Path


_worker_stack_file = None


def register_worker_stack_dump(worker_id, *, output_dir, rank):
    """SIGUSR2 dumps this worker's Python stacks instead of terminating it."""
    global _worker_stack_file
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    _worker_stack_file = (path / f'rank{rank}_worker{worker_id}_pid{os.getpid()}.log').open('a')
    faulthandler.register(signal.SIGUSR2, file=_worker_stack_file, all_threads=True)


class TrainingHangWatchdog:
    def __init__(self, output_dir, *, rank, timeout=120):
        self.output_dir = Path(output_dir)
        self.rank = rank
        self.timeout = float(timeout)
        self.state = {}
        self.last_progress = time.monotonic()
        self.stop = threading.Event()
        self.thread = None
        self.handle = None

    def mark(self, phase, **metadata):
        # Called first after the first batch arrives: DataLoader workers have
        # already forked, so this monitoring thread is never inherited by them.
        self.state = dict(self.state, **metadata, phase=phase)
        self.last_progress = time.monotonic()
        if self.thread is None and self.timeout > 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.handle = (self.output_dir / f'rank{self.rank}_pid{os.getpid()}.log').open('a')
            self.thread = threading.Thread(target=self._monitor, daemon=True, name='training-hang-watchdog')
            self.thread.start()

    def _monitor(self):
        last_dump = 0.
        while not self.stop.wait(min(1., self.timeout / 4)):
            now = time.monotonic()
            if now - self.last_progress < self.timeout or now - last_dump < self.timeout:
                continue
            self.handle.write(json.dumps(dict(self.state, rank=self.rank, pid=os.getpid(),
                stalled_seconds=now-self.last_progress, timestamp=time.time())) + '\n')
            self.handle.flush()
            # Direct file output avoids W&B/tqdm console locks and needs no
            # external debugger permission. The blocked main thread is included.
            faulthandler.dump_traceback(file=self.handle, all_threads=True)
            last_dump = now

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        if self.handle is not None:
            self.handle.close()


def watch_training_hangs(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        debugger = kwargs.get('gradient_debugger')
        if debugger is None or debugger.hang_timeout_seconds <= 0:
            return function(*args, **kwargs)
        rank = int(os.environ.get('RANK', '0'))
        watchdog = TrainingHangWatchdog(debugger.output_dir / 'hang_stacks',
            rank=rank, timeout=debugger.hang_timeout_seconds)
        debugger.watchdog = watchdog
        try:
            return function(*args, **kwargs)
        finally:
            watchdog.close()
            debugger.watchdog = None
    return wrapped
