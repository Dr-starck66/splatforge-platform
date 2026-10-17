"""Negative production gates. Synthetic COLMAP fixtures are never treated as real reconstruction evidence."""
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

    def invoke(self):
        return pipeline.process_dataset(
            input_folder=str(self.inputs),
            output_folder=str(self.outputs),
            session_id="negative-test",
            enable_3dgs=True,
        )

    @staticmethod
    def emulate_colmap(cmd, cwd=None):
        if len(cmd) > 1 and cmd[1] == "mapper":
            sparse = Path(cmd[cmd.index("--output_path") + 1]) / "0"
            sparse.mkdir(parents=True, exist_ok=True)
            (sparse / "cameras.bin").write_bytes(b"fixture-not-a-real-camera-file")
            (sparse / "points3D.bin").write_bytes(b"fixture-not-a-real-point-file")
        return ""

    @patch.object(pipeline, "run_command", return_value="")
    def test_empty_colmap_reconstruction_must_fail(self, mocked):
        with self.assertRaisesRegex(RuntimeError, "COLMAP_EMPTY_RECONSTRUCTION"):
            self.invoke()
        self.assertFalse(list(self.outputs.rglob("*.splat")))

    @patch.object(pipeline, "run_command")
    def test_bad_external_trainer_never_produces_fake_splat(self, mocked):
        mocked.side_effect = self.emulate_colmap
        with patch.dict(os.environ, {"SPLATFORGE_TRAIN_SCRIPT": str(self.root / "missing-trainer.py")}):
            with self.assertRaisesRegex(RuntimeError, "REAL_3DGS_UNAVAILABLE"):
                self.invoke()
        self.assertFalse(list(self.outputs.rglob("*.splat")))
        self.assertTrue(list(self.outputs.rglob("images")))

    @patch.object(pipeline, "run_command")
    def test_no_training_output_cannot_be_reported_as_success(self, mocked):
        mocked.side_effect = self.emulate_colmap
        script = self.root / "train.py"
        script.write_text("# test fixture only")
        with patch.dict(os.environ, {"SPLATFORGE_TRAIN_SCRIPT": str(script)}):
            with self.assertRaisesRegex(RuntimeError, "REAL_3DGS_OUTPUT_MISSING"):
                self.invoke()
        self.assertFalse(list(self.outputs.rglob("*.splat")))


if __name__ == "__main__":
    unittest.main()
