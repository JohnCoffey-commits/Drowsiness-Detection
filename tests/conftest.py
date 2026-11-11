"""Isolated API fixtures: no private models, videos, or real archive writes."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.backend import app as backend  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONGUARD_ARCHIVE_DB_PATH", str(tmp_path / "archive.sqlite"))
    monkeypatch.setenv("VISIONGUARD_ARCHIVE_ENABLED", "1")
    monkeypatch.delenv("VISIONGUARD_ARCHIVE_WRITE_TOKEN", raising=False)
    monkeypatch.setattr(backend, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(backend, "AUDIT_DIR", tmp_path / "audits")
    backend.REALTIME_SESSIONS.clear()
    with TestClient(backend.create_app()) as test_client:
        yield test_client
    backend.REALTIME_SESSIONS.clear()
