"""A stalled Python thread can report its stack without attaching a debugger."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class HangDebugTests(unittest.TestCase):
    def test_stall_dump_records_phase_and_blocked_thread(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = '''
import sys,time
from utils.hang_debug import TrainingHangWatchdog
w=TrainingHangWatchdog(sys.argv[1],rank=1,timeout=.05)
w.mark('waiting_for_next_batch',batch=32)
time.sleep(.18)
w.close()
'''
            subprocess.run([sys.executable, '-c', code, tmp], check=True, timeout=10)
            content=next(Path(tmp).glob('rank1_pid*.log')).read_text()
            self.assertIn('waiting_for_next_batch',content)
            self.assertIn('"batch": 32',content)
            self.assertIn('File "<string>"',content)
            self.assertIn('Thread ',content)

    def test_worker_signal_dumps_without_terminating(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = '''
import os,signal,sys
from utils.hang_debug import register_worker_stack_dump
register_worker_stack_dump(2,output_dir=sys.argv[1],rank=0)
os.kill(os.getpid(),signal.SIGUSR2)
print('worker still alive')
'''
            result=subprocess.run([sys.executable,'-c',code,tmp],check=True,
                                  capture_output=True,text=True,timeout=10)
            self.assertIn('worker still alive',result.stdout)
            content=next(Path(tmp).glob('rank0_worker2_pid*.log')).read_text()
            self.assertIn('File "<string>"',content)


if __name__=='__main__':unittest.main()
