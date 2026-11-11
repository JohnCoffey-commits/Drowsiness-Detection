"""Health contracts and deterministic temporal behavior; no models loaded."""

from types import SimpleNamespace

import pytest

from src.runtime import realtime_frame_inference as inference
from src.runtime.realtime_temporal_state import RealtimeTemporalState


def test_liveness_does_not_load_models_and_readiness_reports_failure(client, monkeypatch):
    def missing():
        raise RuntimeError("Missing assets (test double)")

    monkeypatch.setattr(
        inference, "get_realtime_service", lambda: SimpleNamespace(ensure_loaded=missing)
    )
    assert client.get("/health/live").status_code == 200
    response = client.get("/health/ready")
    assert response.status_code == 503 and response.json()["ready"] is False


def test_readiness_requires_loaded_models(client, monkeypatch):
    service = SimpleNamespace(
        ensure_loaded=lambda: None, models_loaded=True, device_name=lambda: "cpu"
    )
    monkeypatch.setattr(inference, "get_realtime_service", lambda: service)
    assert client.get("/health/ready").json() == {"ready": True, "device": "cpu"}


def test_explicit_cpu_does_not_select_host_mps(monkeypatch):
    torch = SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        device=lambda name: name,
    )
    monkeypatch.setenv("VISIONGUARD_DEVICE", "cpu")
    assert inference.select_realtime_device(torch) == "cpu"
    monkeypatch.setenv("VISIONGUARD_DEVICE", "auto")
    assert inference.select_realtime_device(torch) == "mps"
    monkeypatch.setenv("VISIONGUARD_DEVICE", "typo")
    with pytest.raises(inference.RealtimeFrameInferenceError):
        inference.select_realtime_device(torch)


def test_missing_signal_is_not_a_drowsiness_conclusion():
    state = RealtimeTemporalState()
    result = state.update_from_frame({}, now=1)
    assert result["fusion_state"] == "signal_unreliable"
    assert result["signal_unreliable"] is True
    assert result["valid_eye_frame_count"] == 0


def test_stop_clears_mouth_and_eye_context():
    state = RealtimeTemporalState()
    frame = {
        "eye": {"available": True, "mean_p_eye_closed": 0.9},
        "mouth": {"available": True, "p_yawn": 0.9},
        "face": {"detected": True, "tracking_status": "ok"},
        "signal_quality": {"status": "ok"},
    }
    for i in range(8):
        state.update_from_frame(frame, now=10 + i * 0.3)
    assert state.mouth_active and state.eye_warning_active
    state.freeze()
    assert not state.frames and not state.mouth_active and not state.eye_warning_active
    assert state.last_yawn_monotonic is None
