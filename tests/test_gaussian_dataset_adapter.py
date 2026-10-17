"""CPU-only filesystem checks: never report an actual COLMAP/3DGS GPU run."""
import tempfile
import unittest
from pathlib import Path

from app.gs_adapter import find_colmap_model, prepare_graphdeco_dataset


class GraphdecoStagingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.frames = self.base / "input_frames"
        self.frames.mkdir()
        (self.frames / "frame0001.jpg").write_bytes(b"test image placeholder")
        self.model = self.base / "colmap" / "sparse" / "0"
        self.model.mkdir(parents=True)

    def create_model(self):
        for name in ("cameras", "images", "points3D"):
            (self.model / f"{name}.bin").write_bytes(b"test fixture only")

    def test_fails_without_reconstructed_model(self):
        with self.assertRaisesRegex(RuntimeError, "COLMAP_MODEL_MISSING"):
            prepare_graphdeco_dataset(self.frames, self.model.parent, self.base / "scene")

    def test_stage_references_input_frames_and_colmap_model(self):
        self.create_model()
        scene = prepare_graphdeco_dataset(self.frames, self.model.parent, self.base / "scene")
        self.assertEqual((scene / "images" / "frame0001.jpg").read_bytes(), b"test image placeholder")
        self.assertEqual((scene / "sparse" / "0" / "points3D.bin").read_bytes(), b"test fixture only")
        self.assertEqual(find_colmap_model(self.model.parent), self.model)
        self.assertEqual(prepare_graphdeco_dataset(self.frames, self.model.parent, scene), scene)

    def test_reject_conflicting_dataset(self):
        self.create_model()
        scene = self.base / "scene"
        prepare_graphdeco_dataset(self.frames, self.model.parent, scene)
        different = self.base / "different"
        different.mkdir()
        (different / "img.jpg").write_bytes(b"another")
        with self.assertRaisesRegex(RuntimeError, "3DGS_STAGING_CONFLICT"):
            prepare_graphdeco_dataset(different, self.model.parent, scene)

    def test_reject_zero_length_model_file(self):
        self.create_model()
        (self.model / "images.bin").write_bytes(b"")
        with self.assertRaisesRegex(RuntimeError, "COLMAP_MODEL_MISSING"):
            find_colmap_model(self.model.parent)


if __name__ == "__main__":
    unittest.main()
