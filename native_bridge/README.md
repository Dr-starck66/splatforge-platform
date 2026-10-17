# ASTRA Native Bridge Ω

Testable CPU API for a *synthetic* A -> B SOULBRIDGE memory capsule, MuJoCo A physics, PyBullet B physics, and a CPU 3D Gaussian rasterization artifact. **Not** a substitute for an actual robot, trained 3D scene, gsplat CUDA or persistent-memory data store.

## Free Render deployment (when authorized)

Use `Dr-starck66/splatforge-platform` source branch `astra/native-compute-bridge-20261017`, Runtime Python, Plan Free, Python 3.11. Build command: `pip install -r native_bridge/requirements.txt`. Start command: `uvicorn native_bridge.app:app --host 0.0.0.0 --port $PORT`. Render Free sleeps when idle and loses local files, so this API is stateless and does not claim 24/7 GPU.

## Endpoints

- `GET /healthz`: installed versions and honest CUDA status
- `POST /v1/e2e` with `{"name":"ASTRA","memories":["Bonjour"]}`: physically step MuJoCo and PyBullet, hash capsule, render CPU 3D Gaussian output hash
- `GET /v1/splat/sample.png`: synthetic image, PNG SHA-256 in `x-astra-proof-sha256`

Fixed-size scenes only. No remote shell, commands, model uploads, or arbitrary external URLs. Real integration into existing public SOULBRIDGE and SplatForge frontends requires a separate application update and public smoke test. Default requests perform a small bounded computation and are not a paid GPU job.