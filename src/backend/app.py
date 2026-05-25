#!/usr/bin/env python3
"""Stage 17 video-upload backend API.

Synchronous MVP backend for short uploaded videos. It runs the Stage 17 pipeline
and serves session artifacts through constrained per-session paths.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.backend.local_archive import (  # noqa: E402
    ArchiveDisabledError,
    ArchiveValidationError,
    archive_db_path,
    get_archive,
    is_archive_enabled,
)
from src.runtime.realtime_temporal_state import REALTIME_RULE_WARNING, RealtimeTemporalState  # noqa: E402

RUNS_ROOT = PROJECT_ROOT / "outputs" / "system_video_upload_runs"
AUDIT_DIR = PROJECT_ROOT / "artifacts" / "audits" / "stage17_video_upload_mvp_2026-05-09"
STATIC_DIR = PROJECT_ROOT / "src" / "backend" / "static"
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".m4v"}
MAX_UPLOAD_BYTES = 750 * 1024 * 1024
DEFAULT_CORS_ORIGINS = (
    "http://127.0.0.1:3000",
    "http://localhost:3000",
    "http://127.0.0.1:3001",
    "http://localhost:3001",
)
ALLOWED_ORIGINS_ENV = "VISIONGUARD_ALLOWED_ORIGINS"
ARCHIVE_WRITE_TOKEN_ENV = "VISIONGUARD_ARCHIVE_WRITE_TOKEN"
WARNING = (
    "This output is a rule-based drowsiness warning-candidate analysis, "
    "not final system-level drowsiness accuracy."
)
REALTIME_WARNING = REALTIME_RULE_WARNING
MAX_REALTIME_FRAME_BYTES = 8 * 1024 * 1024
REALTIME_SESSIONS: dict[str, dict[str, Any]] = {}


try:
    from fastapi import FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
    from fastapi.staticfiles import StaticFiles
except ImportError as exc:  # pragma: no cover - exercised by preflight in minimal envs.
    FastAPI = None
    File = None
    Form = None
    Header = None
    HTTPException = None
    Query = None
    Request = None
    UploadFile = None
    CORSMiddleware = None
    FileResponse = None
    JSONResponse = None
    RedirectResponse = None
    Response = None
    StaticFiles = None
    FASTAPI_IMPORT_ERROR: ImportError | None = exc
else:
    FASTAPI_IMPORT_ERROR = None


def sanitize_filename(name: str) -> str:
    base = Path(name).name
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base).strip("._")
    return base or "uploaded_video.mp4"


def validate_extension(filename: str) -> None:
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported video type. Allowed: {sorted(ALLOWED_EXTENSIONS)}")


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid numeric field: {value}") from exc


def parse_optional_int(value: str | None) -> int | None:
    parsed = parse_optional_float(value)
    return None if parsed is None else int(parsed)


def parse_allowed_origins(value: str | None) -> list[str]:
    origins = []
    for origin in (value or "").split(","):
        normalized = origin.strip().rstrip("/")
        if normalized:
            origins.append(normalized)
    return origins


def configured_cors_origins() -> list[str]:
    """Return explicit local defaults plus comma-separated deployment frontend origins."""

    origins = []
    configured_origins = parse_allowed_origins(os.environ.get(ALLOWED_ORIGINS_ENV))
    for origin in [*DEFAULT_CORS_ORIGINS, *configured_origins]:
        if origin not in origins:
            origins.append(origin)
    return origins


def require_archive_write_token(token: str | None) -> None:
    configured_token = os.environ.get(ARCHIVE_WRITE_TOKEN_ENV)
    if configured_token and token != configured_token:
        raise HTTPException(status_code=401, detail="Archive write token is missing or invalid.")


async def read_json_object(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Request body must be JSON.") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
    return payload


def session_dir(session_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", session_id):
        raise ValueError("Invalid session id")
    return RUNS_ROOT / session_id


def safe_session_file(session_id: str, relative_path: str) -> Path:
    root = session_dir(session_id).resolve()
    candidate = (root / relative_path).resolve()
    if root not in candidate.parents and candidate != root:
        raise ValueError("Requested path escapes session directory")
    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def keyframe_urls(session_id: str, summary: dict[str, Any]) -> list[dict[str, Any]]:
    root = session_dir(session_id).resolve()
    rows = []
    for item in summary.get("keyframes", []):
        path = Path(str(item.get("keyframe_path", ""))).resolve()
        if root in path.parents or path == root:
            rel = path.relative_to(root).as_posix()
            row = dict(item)
            row["url"] = f"/api/runs/{session_id}/files/{rel}"
            rows.append(row)
    return rows


def run_pipeline(session_id: str, input_video: Path) -> tuple[dict[str, Any], str, float]:
    output_dir = session_dir(session_id)
    log_path = AUDIT_DIR / f"{session_id}_backend_pipeline.log"
    command = [
        sys.executable,
        "src/runtime/system_video_upload_pipeline.py",
        "--input-video",
        str(input_video),
        "--session-id",
        session_id,
        "--output-dir",
        str(output_dir),
        "--sample-every-n-frames",
        "5",
        "--max-frames",
        "300",
        "--save-debug",
        "--save-keyframes",
        "--force",
    ]
    started = time.time()
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log.write(result.stdout)
        log.write(f"\n[exit_code] {result.returncode}\n")
    duration = time.time() - started
    if result.returncode != 0:
        raise RuntimeError(f"Pipeline failed. See {log_path}. Output: {result.stdout[-1000:]}")
    return load_json(output_dir / "summary.json"), str(log_path), duration


def build_response(session_id: str, summary: dict[str, Any]) -> dict[str, Any]:
    warning_counts = {
        "normal_frames": summary.get("normal_frames", 0),
        "eye_warning_candidate_frames": summary.get("eye_warning_candidate_frames", 0),
        "mouth_warning_candidate_frames": summary.get("mouth_warning_candidate_frames", 0),
        "high_confidence_drowsiness_candidate_frames": summary.get(
            "high_confidence_drowsiness_candidate_frames", 0
        ),
        "signal_unreliable_frames": summary.get("signal_unreliable_frames", 0),
        "weak_eye_warning_evidence_frames": summary.get("weak_eye_warning_evidence_frames", 0),
        "moderate_eye_closure_candidate_frames": summary.get(
            "moderate_eye_closure_candidate_frames", 0
        ),
        "strong_eye_closure_candidate_frames": summary.get(
            "strong_eye_closure_candidate_frames", 0
        ),
        "suppressed_high_confidence_weak_eye_evidence_frames": summary.get(
            "suppressed_high_confidence_weak_eye_evidence_frames", 0
        ),
    }
    return {
        "session_id": session_id,
        "status": summary.get("pipeline_status", "completed"),
        "summary": summary,
        "warning_counts": warning_counts,
        "timeline_url": f"/api/runs/{session_id}/timeline",
        "fusion_figure_url": f"/api/runs/{session_id}/files/figures/fusion_timeline.png",
        "keyframes": keyframe_urls(session_id, summary),
        "report_url": f"/api/runs/{session_id}/files/SYSTEM_VIDEO_UPLOAD_ANALYSIS_REPORT.md",
        "warning": WARNING,
    }


def create_app():
    if FASTAPI_IMPORT_ERROR is not None:
        raise RuntimeError(
            "FastAPI backend dependencies are missing. Install: "
            "python -m pip install fastapi uvicorn python-multipart"
        ) from FASTAPI_IMPORT_ERROR

    app = FastAPI(
        title="Stage 17 Video Upload Warning-Candidate API",
        version="0.1.0",
        description=WARNING,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=configured_cors_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH"],
        allow_headers=["*"],
    )
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def root():
        return RedirectResponse(url="/static/upload_test.html")

    @app.post("/api/analyze-video")
    async def analyze_video(file: UploadFile = File(...)):
        filename = sanitize_filename(file.filename or "uploaded_video.mp4")
        try:
            validate_extension(filename)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        session_id = f"upload_{uuid.uuid4().hex[:12]}"
        out_dir = session_dir(session_id)
        input_dir = out_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        input_path = input_dir / filename

        size = 0
        try:
            with input_path.open("wb") as f:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(status_code=413, detail="Uploaded video is too large")
                    f.write(chunk)
        finally:
            await file.close()

        try:
            summary, log_path, duration = run_pipeline(session_id, input_path)
        except Exception as exc:
            failure = {
                "session_id": session_id,
                "status": "failed",
                "error": str(exc),
                "warning": WARNING,
            }
            (out_dir / "summary.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
            raise HTTPException(status_code=500, detail=failure) from exc

        response = build_response(session_id, summary)
        response["runtime_duration_sec"] = duration
        response["audit_log"] = log_path
        return JSONResponse(response)

    @app.get("/api/realtime/health")
    async def realtime_health():
        from src.runtime.realtime_frame_inference import get_realtime_service

        return JSONResponse(get_realtime_service().health())

    @app.get("/api/archive/health")
    async def archive_health():
        if not is_archive_enabled():
            return JSONResponse(
                {
                    "ok": True,
                    "enabled": False,
                    "db_path": str(archive_db_path()),
                    "db_exists": archive_db_path().exists(),
                    "db_writable": False,
                    "record_count": 0,
                    "latest_record_timestamp": None,
                }
            )
        try:
            return JSONResponse(get_archive().health())
        except Exception as exc:
            return JSONResponse(
                {
                    "ok": False,
                    "enabled": True,
                    "db_path": str(archive_db_path()),
                    "db_exists": archive_db_path().exists(),
                    "db_writable": False,
                    "record_count": 0,
                    "latest_record_timestamp": None,
                    "error": str(exc),
                },
                status_code=503,
            )

    @app.get("/api/archive/records")
    async def archive_records(
        range_value: str = Query(default="48h", alias="range"),
        source: str | None = Query(default=None),
        record_type: str | None = Query(default=None),
        limit: int = Query(default=200, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
    ):
        if not is_archive_enabled():
            return JSONResponse(
                {
                    "ok": True,
                    "enabled": False,
                    "range": range_value,
                    "source": source,
                    "record_type": record_type,
                    "limit": limit,
                    "offset": offset,
                    "total": 0,
                    "records": [],
                }
            )
        try:
            return JSONResponse(
                get_archive().list_records(
                    range_value=range_value,
                    source=source,
                    record_type=record_type,
                    limit=limit,
                    offset=offset,
                )
            )
        except ArchiveValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Archive records unavailable: {exc}") from exc

    @app.post("/api/archive/live-event")
    async def archive_live_event(
        request: Request,
        x_visionguard_archive_token: str | None = Header(
            default=None,
            alias="X-VisionGuard-Archive-Token",
        ),
    ):
        require_archive_write_token(x_visionguard_archive_token)
        payload = await read_json_object(request)
        try:
            record = get_archive().upsert_record(
                payload,
                record_type="live_event",
                source="live_monitor",
            )
            return JSONResponse({"ok": True, "record": record})
        except ArchiveDisabledError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ArchiveValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Archive save failed: {exc}") from exc

    @app.post("/api/archive/live-session")
    async def archive_live_session(
        request: Request,
        x_visionguard_archive_token: str | None = Header(
            default=None,
            alias="X-VisionGuard-Archive-Token",
        ),
    ):
        require_archive_write_token(x_visionguard_archive_token)
        payload = await read_json_object(request)
        try:
            record = get_archive().upsert_record(
                payload,
                record_type="session_summary",
                source="live_monitor",
                event_type="drive_session",
            )
            return JSONResponse({"ok": True, "record": record})
        except ArchiveDisabledError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ArchiveValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Archive session save failed: {exc}") from exc

    @app.post("/api/archive/video-run")
    async def archive_video_run(
        request: Request,
        x_visionguard_archive_token: str | None = Header(
            default=None,
            alias="X-VisionGuard-Archive-Token",
        ),
    ):
        require_archive_write_token(x_visionguard_archive_token)
        payload = await read_json_object(request)
        try:
            record = get_archive().upsert_record(
                payload,
                record_type="video_run",
                source="video_upload",
                event_type="upload_analysis",
            )
            return JSONResponse({"ok": True, "record": record})
        except ArchiveDisabledError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ArchiveValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Archive save failed: {exc}") from exc

    @app.patch("/api/archive/records/{record_id}/review")
    async def archive_record_review(
        record_id: str,
        request: Request,
        x_visionguard_archive_token: str | None = Header(
            default=None,
            alias="X-VisionGuard-Archive-Token",
        ),
    ):
        require_archive_write_token(x_visionguard_archive_token)
        payload = await read_json_object(request)
        reviewed = payload.get("reviewed")
        review_note = payload.get("review_note")
        try:
            record = get_archive().update_review(
                record_id,
                reviewed=bool(reviewed) if reviewed is not None else None,
                review_note=str(review_note) if review_note is not None else None,
            )
            return JSONResponse({"ok": True, "record": record})
        except ArchiveDisabledError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown archive record id: {record_id}") from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Archive review update failed: {exc}") from exc

    @app.get("/api/archive/export")
    async def archive_export():
        if not is_archive_enabled():
            return JSONResponse(
                {
                    "ok": False,
                    "enabled": False,
                    "archive_version": "stage22-local-sqlite-v1",
                    "exported_at": now_iso(),
                    "records": [],
                    "error": "Local archive is disabled.",
                },
                status_code=503,
            )
        try:
            return JSONResponse(get_archive().export_records())
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Archive export failed: {exc}") from exc

    @app.post("/api/realtime/session/start")
    async def realtime_session_start():
        session_id = f"realtime_{uuid.uuid4().hex[:12]}"
        started_at = now_iso()
        REALTIME_SESSIONS[session_id] = {
            "session_id": session_id,
            "started_at": started_at,
            "last_frame_at": None,
            "stopped_at": None,
            "status": "active",
            "temporal_state": RealtimeTemporalState(),
        }
        return JSONResponse(
            {
                "ok": True,
                "session_id": session_id,
                "started_at": started_at,
                "note": (
                    "Lightweight in-memory realtime session. Frame-level evidence is fused into "
                    "a session-local realtime warning-candidate state; no alarm output, "
                    "system-level conclusion, or history ingestion is computed."
                ),
                "warning": REALTIME_WARNING,
            }
        )

    @app.post("/api/realtime/session/stop")
    async def realtime_session_stop(session_id: str = Form(...)):
        session = REALTIME_SESSIONS.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Unknown realtime session_id: {session_id}")

        if session.get("status") == "stopped":
            stopped_at = str(session.get("stopped_at") or now_iso())
            note = (
                "Realtime session was already stopped. Model singleton remains loaded for future sessions."
            )
        else:
            stopped_at = now_iso()
            session["status"] = "stopped"
            session["stopped_at"] = stopped_at
            temporal_state = session.get("temporal_state")
            if isinstance(temporal_state, RealtimeTemporalState):
                temporal_state.freeze()
            note = (
                "Realtime session stopped in memory. Session-local temporal state is frozen; "
                "model singleton remains loaded and checkpoints are not unloaded."
            )

        return JSONResponse(
            {
                "ok": True,
                "session_id": session_id,
                "stopped_at": stopped_at,
                "note": note,
                "warning": REALTIME_WARNING,
            }
        )

    @app.post("/api/realtime/frame")
    async def realtime_frame(
        session_id: str = Form(...),
        frame: UploadFile = File(...),
        client_timestamp_ms: str | None = Form(default=None),
        frame_width: str | None = Form(default=None),
        frame_height: str | None = Form(default=None),
        sampling_fps: str | None = Form(default=None),
    ):
        session = REALTIME_SESSIONS.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Unknown realtime session_id: {session_id}")
        if session.get("status") == "stopped":
            raise HTTPException(
                status_code=409,
                detail={
                    "ok": False,
                    "session_id": session_id,
                    "error": "Realtime session has been stopped.",
                    "warning": REALTIME_WARNING,
                },
            )
        if frame.content_type not in {"image/jpeg", "image/jpg", "application/octet-stream"}:
            raise HTTPException(status_code=400, detail="Realtime frame must be an image/jpeg upload.")

        frame_bytes = await frame.read()
        await frame.close()
        if not frame_bytes:
            raise HTTPException(status_code=400, detail="Realtime frame upload is empty.")
        if len(frame_bytes) > MAX_REALTIME_FRAME_BYTES:
            raise HTTPException(status_code=413, detail="Realtime frame upload is too large.")

        from src.runtime.realtime_frame_inference import RealtimeFrameInferenceError, get_realtime_service

        try:
            result = get_realtime_service().analyze_frame(
                session_id=session_id,
                frame_bytes=frame_bytes,
                client_timestamp_ms=parse_optional_float(client_timestamp_ms),
                frame_width=parse_optional_int(frame_width),
                frame_height=parse_optional_int(frame_height),
                sampling_fps=parse_optional_float(sampling_fps),
            )
        except RealtimeFrameInferenceError as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "ok": False,
                    "session_id": session_id,
                    "error": str(exc),
                    "warning": REALTIME_WARNING,
                },
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "ok": False,
                    "session_id": session_id,
                    "error": f"Realtime frame inference failed: {exc}",
                    "warning": REALTIME_WARNING,
                },
            ) from exc

        temporal_state = session.get("temporal_state")
        if not isinstance(temporal_state, RealtimeTemporalState):
            temporal_state = RealtimeTemporalState()
            session["temporal_state"] = temporal_state
        result["temporal"] = temporal_state.update_from_frame(result)

        session["last_frame_at"] = now_iso()
        return JSONResponse(result, status_code=200 if result.get("ok") else 400)

    @app.get("/api/runs/{session_id}/summary")
    async def get_summary(session_id: str):
        try:
            summary = load_json(session_dir(session_id) / "summary.json")
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(summary)

    @app.get("/api/runs/{session_id}/timeline")
    async def get_timeline(session_id: str):
        try:
            path = safe_session_file(session_id, "timeline.csv")
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return Response(path.read_text(encoding="utf-8"), media_type="text/csv")

    @app.get("/api/runs/{session_id}/keyframes")
    async def get_keyframes(session_id: str):
        try:
            summary = load_json(session_dir(session_id) / "summary.json")
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse({"session_id": session_id, "keyframes": keyframe_urls(session_id, summary)})

    @app.get("/api/runs/{session_id}/files/{relative_path:path}")
    async def get_file(session_id: str, relative_path: str):
        try:
            path = safe_session_file(session_id, relative_path)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(path)

    return app


app = create_app() if FASTAPI_IMPORT_ERROR is None else None


def run_preflight() -> int:
    checks = {
        "fastapi_available": FASTAPI_IMPORT_ERROR is None,
        "runs_root": str(RUNS_ROOT),
        "audit_dir": str(AUDIT_DIR),
        "static_dir": str(STATIC_DIR),
        "pipeline_script_exists": str(PROJECT_ROOT / "src/runtime/system_video_upload_pipeline.py"),
        "cors_origins": configured_cors_origins(),
        "archive_enabled": is_archive_enabled(),
        "archive_db_path": str(archive_db_path()),
        "warning": WARNING,
    }
    if FASTAPI_IMPORT_ERROR is not None:
        checks["missing_dependency_error"] = str(FASTAPI_IMPORT_ERROR)
        checks["install_suggestion"] = "python -m pip install fastapi uvicorn python-multipart"
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    (AUDIT_DIR / "backend_preflight.json").write_text(
        json.dumps(checks, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(checks, indent=2))
    return 0 if FASTAPI_IMPORT_ERROR is None else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 17 backend service")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.preflight:
        return run_preflight()
    if FASTAPI_IMPORT_ERROR is not None:
        print(
            "FastAPI backend dependencies are missing. Install: "
            "python -m pip install fastapi uvicorn python-multipart",
            file=sys.stderr,
        )
        return 2
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is missing. Install: python -m pip install uvicorn", file=sys.stderr)
        return 2
    uvicorn.run(app, host=args.host, port=args.port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
