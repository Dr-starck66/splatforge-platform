"""Fail-closed regression tests; do not require a GPU or pretend to train a model."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import pipeline


class Real3DGSProofGate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.inputs = self.root / "input"
        self.inputs.mkdir()
        for i in range(5):
            (self.inputs / f"photo_{i}.jpg").write_bytes(b"test-only-not-a-real-photo")
        self.outputs = self.root / "output"
        # Emulate COLMAP file layout only; these are NOT real SfM results.
        sparse = self.outputs / "negative-test" / "colmap" / "sparse" / "0"
        sparse.mkdir(parents=True)
        for name in ("cameras", "images", "points3D"):
            (sparse / (name + ".bin")).write_bytes(b"test fixture")

    def invoke(self):
        return pipeline.process_dataset(
            input_folder=str(self.inputs),
            output_folder=str(self.outputs),
            session_id="negative-test",
            enable_3dgs=True,
        )

    @patch.object(pipeline, "run_command", return_value="")
    def test_missing_colmap_model_never_returns_success(self, mocked):
        sparse = self.outputs / "negative-test" / "colmap" / "sparse" / "0"
        for f in sparse.glob("*"):
            f.unlink()
        with self.assertRaisesRegex(RuntimeError, "COLMAP_MODEL_MISSING"):
            self.invoke()

    @patch.object(pipeline, "run_command", return_value="")
    def test_missing_gpu_trainer_never_produces_fake_splat(self, mocked):
        with patch.dict(os.environ, {"SPLATFORGE_TRAIN_SCRIPT": ""}):
            with self.assertRaisesRegex(RuntimeError, "REAL_3DGS_UNAVAILABLE"):
                self.invoke()
        self.assertFalse(list(self.outputs.rglob("*.splat")))

    @patch.object(pipeline, "run_command", return_value="")
    def test_no_training_output_cannot_be_reported_as_success(self, mocked):
        script = self.root / "train.py"
        script.write_text("# test fixture only")
        with patch.dict(os.environ, {"SPLATFORGE_TRAIN_SCRIPT": str(script)}):
            with self.assertRaisesRegex(RuntimeError, "REAL_3DGS_OUTPUT_MISSING"):
                self.invoke()
        self.assertFalse(list(self.outputs.rglob("*.splat")))


if __name__ == "__main__":
    unittest.main()
