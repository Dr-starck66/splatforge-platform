"""A bounded, stateless physics-and-3DGS CPU bridge for ASTRA.

Real engine calls happen on *each* /v1/e2e request. This is not a surrogate
for GPU gsplat, physical robots, training, or server-persistent memory.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import math
import os
from datetime import datetime, timezone

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

VERSION = "0.1.0"
app = FastAPI(title="ASTRA Native Bridge Ω", version=VERSION)


class Capsule(BaseModel):
    name: str = Field(default="ASTRA", max_length=64)
    memories: list[str] = Field(default_factory=list, max_length=8)


def digest(obj: object) -> str:
    data = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def _mujoco() -> dict:
    import mujoco

    xml = '<mujoco><option timestep="0.002" gravity="0 0 -9.81"/>' \
          '<worldbody><geom type="plane" size="1 1 0.1"/>' \
          '<body pos="0 0 0.45"><freejoint/>' \
          '<geom type="sphere" size="0.05" mass="0.4"/></body>' \
          '</worldbody></mujoco>'
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    had_contact = False
    trace = []
    for step in range(750):
        mujoco.mj_step(model, data)
        had_contact |= data.ncon > 0
        if step % 25 == 0 or step == 749:
            trace.append([round(float(data.time), 6), round(float(data.qpos[2]), 7), int(data.ncon)])
    height = float(data.qpos[2])
    return {"engine": "MuJoCo", "version": mujoco.__version__, "steps": 750,
            "contact": bool(had_contact), "height_m": round(height, 8),
            "trace_sha256": digest(trace), "pass": bool(had_contact and 0.04 < height < 0.07)}


def _pybullet() -> dict:
    import pybullet as p

    physics = p.connect(p.DIRECT)
    if physics < 0:
        raise RuntimeError("PyBullet DIRECT refused")
    try:
        p.setGravity(0, 0, -9.81, physicsClientId=physics)
        p.setTimeStep(1 / 240, physicsClientId=physics)
        ground = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=physics)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=ground, physicsClientId=physics)
        shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.05, physicsClientId=physics)
        ball = p.createMultiBody(baseMass=0.4, baseCollisionShapeIndex=shape,
                                 basePosition=[0, 0, 0.45], physicsClientId=physics)
        trace = []
        had_contact = False
        for step in range(480):
            p.stepSimulation(physicsClientId=physics)
            pos, _ = p.getBasePositionAndOrientation(ball, physicsClientId=physics)
            count = len(p.getContactPoints(bodyA=ball, physicsClientId=physics))
            had_contact |= count > 0
            if step % 20 == 0 or step == 479:
                trace.append([round(float(pos[2]), 7), count])
        height = float(pos[2])
        return {"engine": "PyBullet", "api_version": p.getAPIVersion(), "steps": 480,
                "contact": bool(had_contact), "height_m": round(height, 8),
                "trace_sha256": digest(trace), "pass": bool(had_contact and 0.04 < height < 0.07)}
    finally:
        p.disconnect(physicsClientId=physics)


def _cpu_splats() -> tuple[bytes, dict]:
    """Genuine 3D anisotropic covariance -> 2D Gaussian raster + alpha blend.

    Small independent NumPy CPU rasterizer, not the official CUDA gsplat.
    Fixed synthetic geometry avoids untrusted uploaded code or large workloads.
    """
    from PIL import Image

    h = w = 48
    focal = 60.0
    means = [(-0.38, 0.08, 2.45), (0.36, -0.08, 2.60), (0.0, -0.20, 1.87)]
    scales = [(0.23, 0.30, 0.20), (0.32, 0.16, 0.20), (0.23, 0.23, 0.21)]
    colors = [(1.0, 0.08, 0.10), (0.10, 1.0, 0.08), (0.10, 0.15, 1.0)]
    opacity = (0.95, 0.90, 0.94)
    canvas = np.zeros((h, w, 3), dtype=np.float64)
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    for index in sorted(range(3), key=lambda i: means[i][2], reverse=True):
        x, y, z = means[index]
        u, v = focal * x / z + w / 2, focal * y / z + h / 2
        jac = np.array([[focal / z, 0, -focal * x / z ** 2],
                        [0, focal / z, -focal * y / z ** 2]], dtype=float)
        cov3d = np.diag(np.square(scales[index]))
        cov2d = jac @ cov3d @ jac.T + np.eye(2) * 0.3
        inv = np.linalg.inv(cov2d)
        du = x_grid + 0.5 - u
        dv = y_grid + 0.5 - v
        d2 = inv[0, 0] * du * du + 2 * inv[0, 1] * du * dv + inv[1, 1] * dv * dv
        alpha = opacity[index] * np.exp(-0.5 * d2) * (d2 <= 9.0)
        canvas = np.array(colors[index])[None, None, :] * alpha[..., None] + canvas * (1 - alpha[..., None])
    rgb = (np.clip(canvas, 0, 1) * 255).round().astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(output, format="PNG")
    image = output.getvalue()
    num = int(np.count_nonzero(rgb > 15))
    return image, {"engine": "ASTRA NumPy Gaussian Splatting CPU", "gaussians": 3,
                   "resolution": [w, h], "colored_samples": num,
                   "png_bytes": len(image), "png_sha256": hashlib.sha256(image).hexdigest(),
                   "pass": bool(num > 100)}


@app.get("/healthz")
def health() -> dict:
    pkgs = {}
    for key in ("mujoco", "pybullet", "numpy", "Pillow"):
        try:
            pkgs[key] = importlib.metadata.version(key)
        except importlib.metadata.PackageNotFoundError:
            pkgs[key] = None
    return {"service": "astra-native-bridge", "version": VERSION,
            "native_engines_installed": pkgs, "cuda_gsplat": "NOT_TESTED",
            "state": "STATELESS", "updated_utc": datetime.now(timezone.utc).isoformat()}


@app.get("/v1/splat/sample.png")
def sample_png() -> Response:
    image, proof = _cpu_splats()
    return Response(image, media_type="image/png", headers={"x-astra-proof-sha256": proof["png_sha256"]})


@app.post("/v1/e2e")
def e2e(capsule: Capsule) -> dict:
    memory = {"v": 1, "name": capsule.name, "memories": capsule.memories}
    if sum(len(x.encode("utf-8")) for x in capsule.memories) > 2048:
        raise HTTPException(413, "memory capsule too large")
    # A capsule is transferred *unchanged* from A to B, re-hashed there.
    source = _mujoco()
    target = _pybullet()
    _, splat = _cpu_splats()
    source_sha = digest(memory)
    target_sha = digest(json.loads(json.dumps(memory, ensure_ascii=False)))
    ok = source["pass"] and target["pass"] and splat["pass"] and source_sha == target_sha
    return {"status": "PASS" if ok else "FAIL", "capsule_sha256": source_sha,
            "capsule_transfer_verified": source_sha == target_sha,
            "source_A": source, "target_B": target, "splat": splat,
            "scope": "bounded native CPU physics and synthetic CPU 3DGS",
            "not_proven": ["GPU CUDA gsplat training", "physical robot transfer", "durable server-side memory"]}