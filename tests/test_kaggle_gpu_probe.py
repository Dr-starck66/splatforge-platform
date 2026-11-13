import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kaggle_gpu_probe import probe

class NoCuda:
    class cuda:
        @staticmethod
        def is_available():return False

class Tests(unittest.TestCase):
    def test_cpu_refused(self):
        with self.assertRaisesRegex(RuntimeError,'CUDA_UNAVAILABLE'):
            probe(NoCuda())
    def test_no_implicit_success_without_cuda(self):
        self.assertFalse(NoCuda.cuda.is_available())

if __name__=='__main__':unittest.main()
