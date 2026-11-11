"""Goal 3 additions: behavior and failure boundaries, using inference doubles."""

from types import SimpleNamespace

import pytest

from src.backend import app as backend
from src.runtime import realtime_frame_inference as inference


@pytest.mark.parametrize("session_id", [".", "..", "../other", "/tmp/other"])
def test_session_directory_rejects_escape(client, session_id):
    with pytest.raises(ValueError):
        backend.session_dir(session_id)


def test_session_files_reject_symlinks_and_traversal(client, tmp_path):
    root = backend.session_dir("session_1")
    root.mkdir(parents=True)
    outside = tmp_path / "private.txt"
    outside.write_text("not public")
    (root / "link.txt").symlink_to(outside)
    for path in ("../../private.txt", "link.txt"):
        with pytest.raises(ValueError):
            backend.safe_session_file("session_1", path)
    (root / "safe.txt").write_text("public")
    assert client.get("/api/runs/session_1/files/safe.txt").text == "public"


@pytest.mark.parametrize(
    "name,content,status",
    [("bad.exe", b"x", 400), ("empty.mp4", b"", 400), ("large.mp4", b"12345", 413)],
)
def test_invalid_upload_never_runs_pipeline_or_leaves_input(
    client, monkeypatch, name, content, status
):
    monkeypatch.setattr(backend, "MAX_UPLOAD_BYTES", 4)

    def unexpected(*args):
        pytest.fail("Invalid upload reached inference")

    monkeypatch.setattr(backend, "run_pipeline", unexpected)
    response = client.post("/api/analyze-video", files={"file": (name, content, "video/mp4")})
    assert response.status_code == status
    assert not list(backend.RUNS_ROOT.glob("*/input/*"))


def test_pipeline_failure_returns_diagnostic(client, monkeypatch):
    def fail(*args):
        raise RuntimeError("Model checkpoint unavailable (test double)")

    monkeypatch.setattr(backend, "run_pipeline", fail)
    response = client.post(
        "/api/analyze-video", files={"file": ("sample.mp4", b"synthetic", "video/mp4")}
    )
    assert response.status_code == 500
    assert response.json()["detail"]["status"] == "failed"
    assert "checkpoint" in response.json()["detail"]["error"]
    assert len(list(backend.RUNS_ROOT.glob("*/summary.json"))) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("frame_width", "1.5"),
        ("frame_height", "0"),
        ("frame_width", "-2"),
        ("sampling_fps", "NaN"),
        ("sampling_fps", "inf"),
        ("sampling_fps", "-1"),
        ("client_timestamp_ms", "oops"),
        ("client_timestamp_ms", "-3"),
    ],
)
def test_bad_frame_parameters_are_client_errors(client, monkeypatch, field, value):
    def unexpected(**kwargs):
        pytest.fail("Invalid metadata reached inference")

    monkeypatch.setattr(
        inference, "get_realtime_service", lambda: SimpleNamespace(analyze_frame=unexpected)
    )
    session_id = client.post("/api/realtime/session/start").json()["session_id"]
    response = client.post(
        "/api/realtime/frame",
        data={"session_id": session_id, field: value},
        files={"frame": ("frame.jpg", b"synthetic", "image/jpeg")},
    )
    assert response.status_code == 400


def test_missing_model_frame_returns_service_unavailable(client, monkeypatch):
    def fail(**kwargs):
        raise inference.RealtimeFrameInferenceError("Missing checkpoint (test double)")

    monkeypatch.setattr(
        inference, "get_realtime_service", lambda: SimpleNamespace(analyze_frame=fail)
    )
    session_id = client.post("/api/realtime/session/start").json()["session_id"]
    response = client.post(
        "/api/realtime/frame",
        data={"session_id": session_id},
        files={"frame": ("f.jpg", b"synthetic", "image/jpeg")},
    )
    assert response.status_code == 503
    assert "Missing checkpoint" in response.json()["detail"]["error"]


def test_realtime_sessions_are_isolated_and_stop_is_idempotent(client):
    first = client.post("/api/realtime/session/start").json()["session_id"]
    second = client.post("/api/realtime/session/start").json()["session_id"]
    a = backend.REALTIME_SESSIONS[first]["temporal_state"]
    b = backend.REALTIME_SESSIONS[second]["temporal_state"]
    a.update_from_frame({}, now=1)
    assert len(a.frames) == 1 and len(b.frames) == 0
    response = client.post("/api/realtime/session/stop", data={"session_id": first})
    assert response.status_code == 200 and len(a.frames) == 0
    assert (
        client.post("/api/realtime/session/stop", data={"session_id": first}).json()["stopped_at"]
        == response.json()["stopped_at"]
    )
    assert backend.REALTIME_SESSIONS[second]["status"] == "active"
    assert (
        client.post(
            "/api/realtime/frame",
            data={"session_id": first},
            files={"frame": ("f.jpg", b"x", "image/jpeg")},
        ).status_code
        == 409
    )


@pytest.mark.parametrize(
    "content,kind,status",
    [(b"", "image/jpeg", 400), (b"12345", "image/jpeg", 413), (b"x", "text/plain", 400)],
)
def test_realtime_frame_input_limits(client, monkeypatch, content, kind, status):
    monkeypatch.setattr(backend, "MAX_REALTIME_FRAME_BYTES", 4)
    session_id = client.post("/api/realtime/session/start").json()["session_id"]
    assert (
        client.post(
            "/api/realtime/frame",
            data={"session_id": session_id},
            files={"frame": ("f.jpg", content, kind)},
        ).status_code
        == status
    )


def test_archive_write_query_review_export_survives_new_connection(client):
    payload = {"id": "synthetic-session", "title": "Synthetic test", "evidence": {"count": 3}}
    saved = client.post("/api/archive/live-session", json=payload)
    assert saved.status_code == 200
    created = saved.json()["record"]["created_at"]
    updated = client.post("/api/archive/live-session", json={**payload, "title": "Updated"})
    assert updated.json()["record"]["created_at"] == created
    queried = client.get("/api/archive/records?range=all&source=live_monitor").json()
    assert queried["total"] == 1 and queried["records"][0]["title"] == "Updated"
    assert (
        client.get("/api/archive/records", params={"range": "all", "source": "' OR 1=1 --"}).json()[
            "total"
        ]
        == 0
    )
    reviewed = client.patch(
        "/api/archive/records/synthetic-session/review",
        json={"reviewed": True, "review_note": "Synthetic review test, not human peer feedback"},
    )
    assert reviewed.json()["record"]["reviewed"] is True
    exported = client.get("/api/archive/export").json()
    assert exported["record_count"] == 1 and exported["records"][0]["evidence"] == {"count": 3}


@pytest.mark.parametrize(
    "evidence", [{"raw_frame": "abc"}, {"nested": ["data:image/jpeg;base64,abc"]}]
)
def test_archive_rejects_raw_media(client, evidence):
    assert client.post("/api/archive/live-event", json={"evidence": evidence}).status_code == 400
    assert client.get("/api/archive/export").json()["record_count"] == 0


def test_archive_token_and_disabled_modes(client, monkeypatch):
    monkeypatch.setenv("VISIONGUARD_ARCHIVE_WRITE_TOKEN", "synthetic-test-token")
    assert client.post("/api/archive/live-event", json={}).status_code == 401
    assert (
        client.post(
            "/api/archive/live-event",
            json={},
            headers={"X-VisionGuard-Archive-Token": "synthetic-test-token"},
        ).status_code
        == 200
    )
    monkeypatch.setenv("VISIONGUARD_ARCHIVE_ENABLED", "0")
    assert client.get("/api/archive/health").json()["enabled"] is False
    assert client.get("/api/archive/export").status_code == 503
