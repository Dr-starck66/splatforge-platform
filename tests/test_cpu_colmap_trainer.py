"""Actual CPU training on a deterministic COLMAP-text fixture; no fake CUDA success."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from tools.train_colmap_cpu import train


def make_fixture(root):
    rng = np.random.default_rng(42)
    d = root / "sparse" / "0"
    d.mkdir(parents=True)
    (root / "images").mkdir()
    (d / "cameras.txt").write_text("1 PINHOLE 32 32 26 26 15.5 15.5\n")
    positions = rng.uniform(-.30, .30, size=(12, 3)).astype("float32")
    positions[:,2] = rng.uniform(1.7,2.,size=12)
    colors = rng.integers(40,250,size=(12,3))
    poses=[]
    xx,yy=np.meshgrid(np.arange(32),np.arange(32))
    for j,dx in enumerate(np.linspace(-.25,.25,5)):
        local=positions+np.array([-dx,0.,0.])
        px=local[:,0]/local[:,2]*26+15.5
        py=local[:,1]/local[:,2]*26+15.5
        sigma=26*.16/local[:,2]
        dist=(xx[None]-px[:,None,None])**2+(yy[None]-py[:,None,None])**2
        w=np.exp(-dist/(2*sigma[:,None,None]**2))*.85
        rgb=(w[:,:,:,None]*(colors/255)[:,None,None,:]).sum(0)/(1+w.sum(0)[:,:,None])
        name=f"frame_{j:02}.png"
        Image.fromarray(np.uint8(np.clip(rgb*255,0,255))).save(root/"images"/name)
        poses.append(f"{j+1} 1 0 0 0 {-dx:.4f} 0 0 1 {name}\n\n")
    (d/"images.txt").write_text("".join(poses))
    (d/"points3D.txt").write_text("".join(
        f"{j+1} {pt[0]} {pt[1]} {pt[2]} {c[0]} {c[1]} {c[2]} 0.1\n"
        for j,(pt,c) in enumerate(zip(positions,colors))))
    return root


class RealCpuColmapTrainer(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.source=make_fixture(self.root/"dataset")

    def test_binary_artifact_and_actual_gradient_descent(self):
        out=self.root/"output"
        proof=train(self.source,out,steps=35)
        self.assertEqual(proof["engine"],"pytorch_cpu_gaussian_colmap")
        self.assertEqual(proof["image_count"],5)
        self.assertLess(proof["final_mse"],proof["initial_mse"]*.98)
        self.assertEqual((out/"scene.splat").stat().st_size,proof["gaussian_count"]*32)
        self.assertTrue((out/"render_view_0.png").is_file())

    def test_absent_camera_fails_closed(self):
        (self.source/"sparse"/"0"/"cameras.txt").unlink()
        with self.assertRaisesRegex(RuntimeError,"COLMAP_TEXT_REQUIRED"):
            train(self.source,self.root/"output",steps=10)
        self.assertFalse((self.root/"output"/"scene.splat").exists())

    def test_absent_3d_points_fails_closed(self):
        (self.source/"sparse"/"0"/"points3D.txt").write_text("")
        with self.assertRaisesRegex(RuntimeError,"COLMAP_INSUFFICIENT_SPARSE_POINTS"):
            train(self.source,self.root/"output",steps=10)
        self.assertFalse((self.root/"output"/"scene.splat").exists())

    def test_model_byte_for_byte_repeatable(self):
        a=train(self.source,self.root/"a",steps=35)
        b=train(self.source,self.root/"b",steps=35)
        self.assertEqual(a["artifact_sha256"],b["artifact_sha256"])
        self.assertEqual(a["final_mse"],b["final_mse"])


if __name__ == "__main__":
    unittest.main()
