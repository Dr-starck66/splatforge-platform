"""Fail-closed COLMAP -> Graphdeco 3D Gaussian Splatting data adapter.

No rendering is performed here.  The Graphdeco trainer expects images/ and
sparse/0/ alongside one another, not colmap/sparse/0/ and input_frames/.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path


MODEL_NAMES = ("cameras", "images", "points3D")


def find_colmap_model(sparse_root: Path) -> Path:
    """Select a nonempty COLMAP model; fail rather than report fake progress."""
    sparse_root = Path(sparse_root)
    candidates = [sparse_root]
    if sparse_root.is_dir():
        candidates.extend(sorted((p for p in sparse_root.iterdir() if p.is_dir()), key=lambda p: p.name))
    for candidate in candidates:
        for extension in (".bin", ".txt"):
            files = [candidate / (name + extension) for name in MODEL_NAMES]
            if all(file.is_file() and file.stat().st_size > 0 for file in files):
                return candidate
    raise RuntimeError(
        "COLMAP_MODEL_MISSING: mapper did not produce a nonempty "
        "cameras/images/points3D model; refusing to launch 3DGS"
    )


def _link_or_copy(source: Path, target: Path) -> None:
    source = source.resolve(strict=True)
    if target.is_symlink() or target.exists():
        if target.is_dir() and target.resolve() == source:
            return
        raise RuntimeError(f"3DGS_STAGING_CONFLICT: {target} already exists")
    try:
        target.symlink_to(source, target_is_directory=True)
    except (OSError, NotImplementedError):
        shutil.copytree(source, target)


def prepare_graphdeco_dataset(frames: Path, sparse_root: Path, session_root: Path) -> Path:
    """Stage the *existing* COLMAP output in the canonical Graphdeco layout."""
    frames = Path(frames)
    if not frames.is_dir() or not any(p.is_file() for p in frames.iterdir()):
        raise RuntimeError("3DGS_IMAGES_MISSING: cannot stage empty frames")
    model = find_colmap_model(sparse_root)
    root = Path(session_root)
    root.mkdir(parents=True, exist_ok=True)
    sparse_stage = root / "sparse"
    sparse_stage.mkdir(exist_ok=True)
    _link_or_copy(frames, root / "images")
    _link_or_copy(model, sparse_stage / "0")
    return root
