"""
SPLAT·FORGE — Production Pipeline Module
Optimized for VPS with NVIDIA GPU
Handles Video extraction, COLMAP SfM, and 3D Gaussian Splatting
"""
import os
import time
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("splatforge.pipeline")

def run_command(cmd: list[str], cwd: Optional[Path] = None):
    """Helper to run shell commands and log errors."""
    log.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"Command failed: {result.stderr}")
        raise RuntimeError(f"Command failed with exit code {result.returncode}")
    return result.stdout

def extract_frames_from_video(video_path: Path, output_dir: Path, fps: int = 2) -> list[Path]:
    """Extracts frames from a video file using ffmpeg."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = output_dir / "frame_%04d.jpg"
    
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        str(output_pattern)
    ]
    run_command(cmd)
    return list(output_dir.glob("*.jpg"))

def process_dataset(
    input_folder: str,
    output_folder: str,
    session_id: str,
    enable_3dgs: bool = True,
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> dict:
    """
    Production pipeline for 3D reconstruction.
    Requires: COLMAP and NVIDIA GPU with CUDA.
    """
    input_path = Path(input_folder)
    session_output_path = Path(output_folder) / session_id
    session_output_path.mkdir(parents=True, exist_ok=True)
    
    # Paths
    colmap_path = session_output_path / "colmap"
    gs_path = session_output_path / "3dgs"
    # Canonical COLMAP/3DGS dataset layout expected by real Gaussian trainers.
    frames_path = session_output_path / "images"
    database_path = colmap_path / "database.db"
    sparse_path = session_output_path / "sparse"
    
    for p in [colmap_path, gs_path, frames_path, sparse_path]:
        p.mkdir(exist_ok=True)
    
    def update_progress(pct: int, step: str):
        if progress_callback:
            progress_callback(pct, step)
        log.info(f"[{session_id}] {pct}% - {step}")

    # Run COLMAP SIFT on CPU whenever CUDA is unavailable. Respect explicit
    # SPLATFORGE_COLMAP_USE_GPU=0/1 only as an operator override.
    gpu_override = os.environ.get("SPLATFORGE_COLMAP_USE_GPU")
    if gpu_override in ("0", "1"):
        use_gpu = gpu_override
    else:
        try:
            import torch
            use_gpu = "1" if torch.cuda.is_available() else "0"
        except ImportError:
            use_gpu = "0"

    try:
        # --- STEP 1: Input Handling ---
        update_progress(5, "Processing input data")
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
        video_files = [f for f in input_path.iterdir() if f.suffix.lower() in video_extensions]
        
        if video_files:
            update_progress(10, "Extracting frames from video")
            images = extract_frames_from_video(video_files[0], frames_path)
        else:
            images = [f for f in input_path.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
            for img in images:
                shutil.copy(img, frames_path / img.name)
            images = list(frames_path.glob("*"))

        if len(images) < 5:
            raise ValueError(f"Insufficient images for quality reconstruction. Found {len(images)}.")

        # --- STEP 2: COLMAP SfM (Structure from Motion) ---
        # 2.1 Feature Extraction
        update_progress(20, "COLMAP: Extracting features")
        run_command([
            "colmap", "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(frames_path),
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.use_gpu", use_gpu
        ])
        
        # 2.2 Feature Matching
        update_progress(40, "COLMAP: Matching features")
        run_command([
            "colmap", "exhaustive_matcher",
            "--database_path", str(database_path),
            "--SiftMatching.use_gpu", use_gpu
        ])

        # 2.3 Sparse Reconstruction
        update_progress(60, "COLMAP: Sparse reconstruction")
        run_command([
            "colmap", "mapper",
            "--database_path", str(database_path),
            "--image_path", str(frames_path),
            "--output_path", str(sparse_path)
        ])

        # A COLMAP process can exit zero despite registering no cameras. The
        # genuine trainer must not be run on a nonexistent reconstruction.
        models = list(sparse_path.glob("*/cameras.bin")) + list(sparse_path.glob("*/cameras.txt"))
        if not models:
            raise RuntimeError("COLMAP_EMPTY_RECONSTRUCTION: no registered camera model")

        # Require COLMAP point observations, not only a camera marker.
        model_dir = next((p.parent for p in sorted(models) if (p.parent / "points3D.bin").is_file() or (p.parent / "points3D.txt").is_file()), None)
        if model_dir is None:
            raise RuntimeError("COLMAP_EMPTY_POINTS: reconstruction has no 3D point file")

        # Convert binary reconstruction to documented text format for CPU path.
        # Real external GPU trainers can continue to read binary COLMAP format.
        update_progress(70, "COLMAP: Validating and converting geometry")
        run_command([
            "colmap", "model_converter",
            "--input_path", str(model_dir),
            "--output_path", str(model_dir),
            "--output_type", "TXT"
        ])

        # --- STEP 3: 3D Gaussian Splatting ---
        # Never manufacture a fake .splat, even when COLMAP succeeds.
        model_relpath = None
        if enable_3dgs:
            update_progress(75, "3DGS: Checking the real trainer")
            trainer = os.environ.get("SPLATFORGE_TRAIN_SCRIPT", "").strip()
            mode = "configured_external_trainer" if trainer else "cpu_colmap_gaussian"
            if not trainer:
                trainer = str(Path(__file__).resolve().parent.parent / "tools" / "train_colmap_cpu.py")
            if not Path(trainer).is_file():
                raise RuntimeError("REAL_3DGS_UNAVAILABLE: training script not installed")
            update_progress(80, "3DGS: Training (" + mode + ")")
            # The CPU trainer requires real COLMAP photos, poses and sparse 3D points.
            run_command([
                sys.executable, str(Path(trainer).resolve()),
                "-s", str(session_output_path),
                "-m", str(gs_path),
            ])
            if mode == "cpu_colmap_gaussian":
                proof_path = gs_path / "proof.json"
                if not proof_path.is_file():
                    raise RuntimeError("CPU_PROOF_MISSING: no measured training evidence")
                import json
                proof = json.loads(proof_path.read_text(encoding="utf-8"))
                if proof.get("engine") != "pytorch_cpu_gaussian_colmap" or not proof.get("final_mse", float("inf")) < proof.get("initial_mse", 0) * 0.98:
                    raise RuntimeError("CPU_PROOF_INVALID: training must improve measured loss")
            candidates = sorted(
                [*gs_path.rglob("*.splat"), *gs_path.rglob("*.ply")],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            genuine = []
            for candidate in candidates:
                if not candidate.is_file() or candidate.stat().st_size < 32:
                    continue
                if candidate.suffix == ".splat" and candidate.stat().st_size % 32 != 0:
                    continue
                with candidate.open("rb") as model_file:
                    header = model_file.read(64)
                if b"REAL_GPU_DATA_PLACEHOLDER" in header:
                    continue
                if candidate.suffix == ".ply" and not header.startswith(b"ply\n"):
                    continue
                genuine.append(candidate)
            if not genuine:
                raise RuntimeError(
                    "REAL_3DGS_OUTPUT_MISSING: training ended but no valid model artifact "
                    "was found; refusing a false success"
                )
            model_relpath = str(genuine[0].relative_to(session_output_path))

        update_progress(100, "Pipeline complete")
        
        return {
            "session_id": session_id,
            "status": "success",
            "input_type": "video" if video_files else "images",
            "image_count": len(images),
            "outputs": {
                "sparse": str(sparse_path),
                "splat": model_relpath if model_relpath and model_relpath.endswith(".splat") else None,
                "model": model_relpath,
                "training_mode": mode if enable_3dgs else "colmap_only",
            }
        }

    except Exception as e:
        log.error(f"Pipeline failed: {str(e)}")
        update_progress(0, f"Error: {str(e)}")
        raise
