#!/usr/bin/env python3
"""Stage 17 video-upload backend API.

Synchronous MVP backend for short uploaded videos. It runs the Stage 17 pipeline
and serves session artifacts through constrained per-session paths.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = PROJECT_ROOT / "outputs" / "system_video_upload_runs"
AUDIT_DIR = PROJECT_ROOT / "artifacts" / "audits" / "stage17_video_upload_mvp_2026-05-09"
STATIC_DIR = PROJECT_ROOT / "src" / "backend" / "static"
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".m4v"}
MAX_UPLOAD_BYTES = 750 * 1024 * 1024
WARNING = (
    "This output is a rule-based drowsiness warning-candidate analysis, "
    "not final system-level drowsiness accuracy."
)


try:
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
    from fastapi.staticfiles import StaticFiles
except ImportError as exc:  # pragma: no cover - exercised by preflight in minimal envs.
    FastAPI = None
    File = None
    HTTPException = None
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
        allow_origins=[
            "http://127.0.0.1:3000",
            "http://localhost:3000",
            "http://127.0.0.1:3001",
            "http://localhost:3001",
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
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
