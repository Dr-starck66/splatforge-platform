"""
SPLAT·FORGE PLATFORM — FastAPI Backend
Full pipeline: upload → COLMAP → 3DGS → viewer
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

# ── Logging ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("splatforge")

# ── Paths ──────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
UPLOAD_DIR  = BASE_DIR / "data" / "uploads"
OUTPUT_DIR  = BASE_DIR / "data" / "outputs"
STATIC_DIR  = Path(__file__).parent / "static"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Supported input extensions
SUPPORTED_EXT = {".jpg", ".jpeg", ".png"}  # matches COLMAP and pipeline input formats

# ── In-memory session store (swap for Redis in prod) ──────
# sessions[sid] = {
#   status: queued|running|done|error
#   progress: 0-100
#   step: str
#   result: dict | None
#   error: str | None
#   created_at: float
#   file_count: int
#   name: str
# }
sessions: dict[str, dict] = {}

def _save_session(sid: str):
    """Persist session to disk atomically; same output volume holds its results."""
    folder = OUTPUT_DIR / sid
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f".session.{uuid.uuid4().hex}.tmp"
    try:
        tmp.write_text(json.dumps(sessions[sid], ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, folder / "session.json")
    finally:
        tmp.unlink(missing_ok=True)

def _load_sessions():
    for statefile in OUTPUT_DIR.glob("*/session.json"):
        try:
            sid = statefile.parent.name
            if not sid.isalnum():
                continue
            data = json.loads(statefile.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or "status" not in data:
                continue
            sessions[sid] = data
        except (ValueError, OSError) as exc:
            log.warning("Skipping unreadable session %s: %s", statefile, exc)

_load_sessions()

# SSE event queues per session
sse_queues: dict[str, list[asyncio.Queue]] = {}
main_loop: asyncio.AbstractEventLoop | None = None

# ── App ────────────────────────────────────────────────────
app = FastAPI(
    title="SPLAT·FORGE Platform",
    version="2.0.0",
    docs_url="/api/docs",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ══════════════════════════════════════════════════════════
# SESSION HELPERS
# ══════════════════════════════════════════════════════════
def session_update(sid: str, **kwargs):
    """Update session state and broadcast SSE event."""
    if sid in sessions:
        sessions[sid].update(kwargs)
        _save_session(sid)
        _broadcast(sid, {k: v for k, v in sessions[sid].items()
                         if k != "result"} | {"sid": sid})


def _broadcast(sid: str, payload: dict):
    """Push SSE event to all listeners of this session."""
    msg = json.dumps(payload)
    def push_on_loop():
        for q in tuple(sse_queues.get(sid, [])):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass
    # asyncio.Queue cannot be safely mutated from the background worker thread.
    if main_loop is not None and main_loop.is_running():
        main_loop.call_soon_threadsafe(push_on_loop)


def progress_cb(sid: str):
    """Return a progress callback bound to a session."""
    def cb(pct: int, step: str):
        session_update(sid, progress=pct, step=step)
        log.info(f"[{sid}] {pct}% — {step}")
    return cb


# ══════════════════════════════════════════════════════════
# PIPELINE RUNNER
# (imports done here to avoid circular / slow startup)
# ══════════════════════════════════════════════════════════
def _run_pipeline(sid: str, session_path: Path):
    try:
        session_update(sid, status="running", progress=2, step="initializing")

        # Lazy import so startup is instant
        from app.pipeline import process_dataset  # noqa: PLC0415

        result = process_dataset(
            input_folder=str(session_path),
            output_folder=str(OUTPUT_DIR),
            session_id=sid,
            enable_3dgs=True,
            progress_callback=progress_cb(sid),
        )

        # Resolve relative paths to just filenames for the API
        session_update(
            sid,
            status="done",
            progress=100,
            step="complete",
            result=result,
        )
        log.info(f"[{sid}] Pipeline complete.")

    except Exception as exc:
        log.exception(f"[{sid}] Pipeline error: {exc}")
        session_update(sid, status="error", progress=0,
                       step="failed", error=str(exc))


# ══════════════════════════════════════════════════════════
# ROUTES — Pages
# ══════════════════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    index = STATIC_DIR / "index.html"
    return HTMLResponse(content=index.read_text(encoding="utf-8"))


@app.get("/viewer/{sid}", response_class=HTMLResponse)
async def viewer_page(sid: str):
    if sid not in sessions:
        raise HTTPException(404, "Session not found")
    viewer = STATIC_DIR / "viewer.html"
    # Inject session ID so the viewer knows what to load
    html = viewer.read_text(encoding="utf-8").replace(
        "{{SESSION_ID}}", sid
    )
    return HTMLResponse(content=html)


# ══════════════════════════════════════════════════════════
# ROUTES — API
# ══════════════════════════════════════════════════════════
@app.post("/api/upload")
async def upload(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
):
    """Upload images and start the pipeline."""
    # Validate
    valid = [f for f in files if Path(f.filename).suffix.lower() in SUPPORTED_EXT]
    if len(valid) < 5:
        raise HTTPException(
            400,
            f"Need ≥ 5 supported images "
            f"({', '.join(sorted(SUPPORTED_EXT))}). Got {len(valid)}.",
        )

    sid = uuid.uuid4().hex[:10]
    session_path = UPLOAD_DIR / sid
    session_path.mkdir(parents=True, exist_ok=True)

    # Save files
    saved = []
    used_names: set[str] = set()
    for f in valid:
        safe_name = Path(f.filename).name
        if safe_name in used_names:
            raise HTTPException(400, "Duplicate image filenames are not allowed")
        used_names.add(safe_name)
        dest = session_path / safe_name
        with open(dest, "wb") as buf:
            shutil.copyfileobj(f.file, buf)
        saved.append(safe_name)
        await f.close()

    # Register session
    sessions[sid] = {
        "status": "queued",
        "progress": 0,
        "step": "queued",
        "result": None,
        "error": None,
        "created_at": time.time(),
        "file_count": len(saved),
        "name": saved[0].rsplit(".", 1)[0][:32] if saved else sid,
        "files": saved,
    }

    # Starlette dispatches synchronous BackgroundTasks in a thread pool.
    # Do not invoke get_event_loop from a worker thread (there is no loop).
    _save_session(sid)
    background_tasks.add_task(_run_pipeline, sid, session_path)

    log.info(f"[{sid}] Session created — {len(saved)} images")
    return {"sid": sid, "files": len(saved), "status": "queued"}


@app.get("/api/sessions")
async def list_sessions():
    """Return all sessions (newest first)."""
    out = []
    for sid, s in sorted(sessions.items(), key=lambda x: -x[1]["created_at"]):
        out.append({
            "sid": sid,
            "status": s["status"],
            "progress": s["progress"],
            "step": s["step"],
            "file_count": s["file_count"],
            "name": s["name"],
            "created_at": s["created_at"],
            "has_splat": _has_file(sid, "3dgs/scene.splat"),
            "has_ply": _has_file(sid, "3dgs/gaussians_final.ply"),
            "error": s.get("error"),
        })
    return out


@app.get("/api/session/{sid}")
async def get_session(sid: str):
    if sid not in sessions:
        raise HTTPException(404, "Session not found")
    s = sessions[sid]
    return {
        "sid": sid,
        **{k: v for k, v in s.items() if k not in ("result",)},
        "has_splat": _has_file(sid, "3dgs/scene.splat"),
        "has_ply": _has_file(sid, "3dgs/gaussians_final.ply"),
        "has_stitched": _has_file(sid, "stitched.png"),
        "has_depth": _has_file(sid, "depth_viz.png"),
        "has_seg": _has_file(sid, "segmentation.png"),
        "has_coco": _has_file(sid, "annotations_coco.json"),
    }


@app.delete("/api/session/{sid}")
async def delete_session(sid: str):
    if sid not in sessions:
        raise HTTPException(404, "Session not found")
    if sessions[sid]["status"] in {"queued", "running"}:
        raise HTTPException(409, "Cannot delete an active job")
    sessions.pop(sid, None)
    sse_queues.pop(sid, None)
    # Clean up files
    for d in [UPLOAD_DIR / sid, OUTPUT_DIR / sid]:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
    return {"deleted": sid}


# ── SSE progress stream ────────────────────────────────────
@app.get("/api/events/{sid}")
async def sse_events(sid: str, request: Request):
    """Server-Sent Events stream for session progress."""
    if sid not in sessions:
        raise HTTPException(404, "Session not found")

    q: asyncio.Queue[str] = asyncio.Queue(maxsize=64)
    sse_queues.setdefault(sid, []).append(q)

    async def generator() -> AsyncGenerator[str, None]:
        try:
            # Send current state immediately
            yield _sse(json.dumps({**sessions[sid], "sid": sid}))
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield _sse(msg)
                    # Stop streaming once terminal state reached
                    s = sessions.get(sid, {})
                    if s.get("status") in ("done", "error"):
                        yield _sse(json.dumps({**s, "sid": sid}))
                        break
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"  # keep-alive comment
        finally:
            sse_queues.get(sid, []).remove(q) if q in sse_queues.get(sid, []) else None

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _sse(data: str) -> str:
    return f"data: {data}\n\n"


# ── File download ──────────────────────────────────────────
@app.get("/api/file/{sid}/{filepath:path}")
async def serve_file(sid: str, filepath: str):
    """Serve any output file for a session."""
    if sid not in sessions:
        raise HTTPException(404, "Session not found")
    path = OUTPUT_DIR / sid / filepath
    if not path.exists() or not path.is_file():
        raise HTTPException(404, f"File not found: {filepath}")
    # Security: ensure path stays within output dir
    try:
        path.resolve().relative_to((OUTPUT_DIR / sid).resolve())
    except ValueError:
        raise HTTPException(403, "Path traversal denied")
    return FileResponse(str(path), filename=path.name)


# ── Health ─────────────────────────────────────────────────
@app.on_event("startup")
async def recover_interrupted_sessions():
    """Resume interrupted persisted uploads after a server restart."""
    global main_loop
    main_loop = asyncio.get_running_loop()
    for sid, item in list(sessions.items()):
        if item.get("status") in {"queued", "running"}:
            source = UPLOAD_DIR / sid
            if not source.exists():
                session_update(sid, status="error", step="failed", error="INPUT_LOST_AFTER_RESTART")
                continue
            session_update(sid, status="queued", step="recovered_after_restart")
            asyncio.create_task(asyncio.to_thread(_run_pipeline, sid, source))


@app.get("/api/health")
async def health():
    import torch  # noqa: PLC0415
    return {
        "status": "ok",
        "sessions": len(sessions),
        "gpu": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


# ── Helpers ────────────────────────────────────────────────
def _has_file(sid: str, rel: str) -> bool:
    if sid not in sessions:
        return False
    base = (OUTPUT_DIR / sid).resolve()
    target = (base / rel).resolve()
    if not target.is_relative_to(base):
        return False
    return target.is_file()


# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,  # 1 worker — GPU is not fork-safe
        log_level="info",
    )
