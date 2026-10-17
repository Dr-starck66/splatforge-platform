"""SplatForge upload -> server background task -> durable session -> query/download tests.
These isolate COLMAP/3DGS from the HTTP test; native COLMAP must be separately verified.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from app import main


class RealHttpPipelineDispatch(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.old_up=main.UPLOAD_DIR
        self.old_out=main.OUTPUT_DIR
        main.UPLOAD_DIR=self.root/"uploads"
        main.OUTPUT_DIR=self.root/"outputs"
        main.UPLOAD_DIR.mkdir()
        main.OUTPUT_DIR.mkdir()
        main.sessions.clear()
        self.addCleanup(self.cleanup)

    def cleanup(self):
        main.UPLOAD_DIR=self.old_up
        main.OUTPUT_DIR=self.old_out
        main.sessions.clear()

    def test_upload_dispatch_persists_and_survives_store_reload(self):
        def executor(**kwargs):
            sid=kwargs["session_id"]
            dest=main.OUTPUT_DIR/sid/"3dgs"
            dest.mkdir(parents=True,exist_ok=True)
            (dest/"scene.splat").write_bytes(b"\x00"*32)
            return {
                "status":"success",
                "session_id":sid,
                "outputs":{"splat":"3dgs/scene.splat","training_mode":"test-isolated-stub"}
            }
        with patch("app.pipeline.process_dataset",side_effect=executor) as mocked:
            with TestClient(main.app) as client:
                files=[("files",(f"frame_{i}.jpg",b"dummy-image","image/jpeg")) for i in range(5)]
                resp=client.post("/api/upload",files=files)
                self.assertEqual(resp.status_code,200,resp.text)
                sid=resp.json()["sid"]
                self.assertTrue(mocked.called)
                result=client.get("/api/session/"+sid).json()
                self.assertEqual(result["status"],"done")
                self.assertTrue(result["has_splat"])
                raw=client.get(f"/api/file/{sid}/3dgs/scene.splat")
                self.assertEqual(raw.status_code,200)
                self.assertEqual(len(raw.content),32)
                self.assertTrue((main.OUTPUT_DIR/sid/"session.json").is_file())
                main.sessions.clear()
                main._load_sessions()
                self.assertEqual(client.get("/api/session/"+sid).json()["status"],"done")

    def test_upload_rejects_four_images_before_starting_worker(self):
        with TestClient(main.app) as client:
            files=[("files",(f"frame_{i}.jpg",b"dummy-image","image/jpeg")) for i in range(4)]
            resp=client.post("/api/upload",files=files)
            self.assertEqual(resp.status_code,400,resp.text)
            self.assertFalse(main.sessions)


if __name__=="__main__":
    unittest.main()
