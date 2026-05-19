"""Local SQLite archive for compact VisionGuard shared-record summaries."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE_DB_PATH = PROJECT_ROOT / "data" / "visionguard_archive.sqlite"
ARCHIVE_ENABLED_ENV = "VISIONGUARD_ARCHIVE_ENABLED"
ARCHIVE_DB_PATH_ENV = "VISIONGUARD_ARCHIVE_DB_PATH"
ARCHIVE_VERSION = "stage22-local-sqlite-v1"
MAX_JSON_CHARS = 60_000
MAX_TEXT_CHARS = 4_000
FORBIDDEN_JSON_KEY_PARTS = (
    "base64",
    "blob",
    "raw_frame",
    "raw_image",
    "raw_video",
    "frame_bytes",
    "image_bytes",
    "video_bytes",
    "file_bytes",
    "payload",
)


class ArchiveDisabledError(RuntimeError):
    """Raised when archive writes are requested while the archive is disabled."""


class ArchiveValidationError(ValueError):
    """Raised when an archive payload is not safe to store."""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_archive_enabled() -> bool:
    value = os.environ.get(ARCHIVE_ENABLED_ENV)
    if value is None:
        return True
    return value.strip().lower() not in {"0", "false", "no", "off", "disabled"}


def archive_db_path() -> Path:
    configured = os.environ.get(ARCHIVE_DB_PATH_ENV)
    if not configured:
        return DEFAULT_ARCHIVE_DB_PATH
    path = Path(configured).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _truncate_text(value: Any, limit: int = MAX_TEXT_CHARS) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "..."


def _validate_safe_json(value: Any, path: str = "root") -> None:
    if value is None or isinstance(value, (bool, int, float)):
        return
    if isinstance(value, str):
        stripped = value.lstrip().lower()
        if stripped.startswith(("data:image", "data:video", "blob:")):
            raise ArchiveValidationError(f"Archive field {path} looks like raw media data.")
        if len(value) > MAX_JSON_CHARS:
            raise ArchiveValidationError(f"Archive field {path} is too large for summary storage.")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_safe_json(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            if any(part in key_text for part in FORBIDDEN_JSON_KEY_PARTS):
                raise ArchiveValidationError(f"Archive field {path}.{key} is not allowed.")
            _validate_safe_json(item, f"{path}.{key}")
        return
    raise ArchiveValidationError(f"Archive field {path} has unsupported type {type(value).__name__}.")


def _json_dumps(value: Any) -> str:
    payload = value if value is not None else {}
    _validate_safe_json(payload)
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if len(text) > MAX_JSON_CHARS:
        raise ArchiveValidationError("Archive JSON payload is too large for summary storage.")
    return text


def _json_loads(value: str | None) -> Any:
    if not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


def _coerce_reviewed(value: Any) -> int:
    if isinstance(value, str):
        return 1 if value.strip().lower() in {"1", "true", "yes", "reviewed"} else 0
    return 1 if bool(value) else 0


def _normalize_record(
    payload: dict[str, Any],
    *,
    record_type: str,
    source: str,
    event_type: str | None = None,
) -> dict[str, Any]:
    created_at = _truncate_text(payload.get("created_at") or payload.get("timestamp") or now_iso())
    updated_at = _truncate_text(payload.get("updated_at") or now_iso())
    evidence = payload.get("evidence")
    if evidence is None:
        evidence = payload.get("evidence_json")
    metadata = payload.get("metadata")
    if metadata is None:
        metadata = payload.get("metadata_json")

    return {
        "id": _truncate_text(payload.get("id") or f"archive_{uuid.uuid4().hex}", 160),
        "record_type": record_type,
        "source": source,
        "client_id": _truncate_text(payload.get("client_id"), 160),
        "account_id": _truncate_text(payload.get("account_id"), 160),
        "session_id": _truncate_text(payload.get("session_id"), 160),
        "event_type": _truncate_text(payload.get("event_type") or event_type, 160),
        "severity": _truncate_text(payload.get("severity"), 80),
        "title": _truncate_text(payload.get("title"), 240),
        "summary": _truncate_text(payload.get("summary"), MAX_TEXT_CHARS),
        "started_at": _truncate_text(payload.get("started_at") or payload.get("timestamp"), 80),
        "ended_at": _truncate_text(payload.get("ended_at"), 80),
        "created_at": created_at,
        "updated_at": updated_at,
        "reviewed": _coerce_reviewed(payload.get("reviewed", 0)),
        "review_note": _truncate_text(payload.get("review_note"), MAX_TEXT_CHARS),
        "evidence_json": _json_dumps(evidence),
        "metadata_json": _json_dumps(metadata),
    }


class LocalArchive:
    def __init__(self, db_path: Path | None = None) -> None:
        if not is_archive_enabled():
            raise ArchiveDisabledError("Local archive is disabled.")
        self.db_path = db_path or archive_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS archive_records (
                    id TEXT PRIMARY KEY,
                    record_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    client_id TEXT,
                    account_id TEXT,
                    session_id TEXT,
                    event_type TEXT,
                    severity TEXT,
                    title TEXT,
                    summary TEXT,
                    started_at TEXT,
                    ended_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT,
                    reviewed INTEGER DEFAULT 0,
                    review_note TEXT,
                    evidence_json TEXT,
                    metadata_json TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_archive_created_at ON archive_records(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_archive_source ON archive_records(source)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_archive_record_type ON archive_records(record_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_archive_session_id ON archive_records(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_archive_client_id ON archive_records(client_id)")

    def health(self) -> dict[str, Any]:
        db_exists = self.db_path.exists()
        writable_target = self.db_path if db_exists else self.db_path.parent
        db_writable = os.access(writable_target, os.W_OK)
        with self._connect() as conn:
            count_row = conn.execute("SELECT COUNT(*) AS count FROM archive_records").fetchone()
            latest_row = conn.execute(
                "SELECT created_at FROM archive_records ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        return {
            "ok": True,
            "enabled": True,
            "db_path": str(self.db_path),
            "db_exists": self.db_path.exists(),
            "db_writable": db_writable,
            "record_count": int(count_row["count"] if count_row else 0),
            "latest_record_timestamp": latest_row["created_at"] if latest_row else None,
            "archive_version": ARCHIVE_VERSION,
        }

    def upsert_record(
        self,
        payload: dict[str, Any],
        *,
        record_type: str,
        source: str,
        event_type: str | None = None,
    ) -> dict[str, Any]:
        record = _normalize_record(payload, record_type=record_type, source=source, event_type=event_type)
        columns = list(record.keys())
        placeholders = ", ".join("?" for _ in columns)
        update_columns = [column for column in columns if column not in {"id", "created_at"}]
        update_sql = ", ".join(f"{column}=excluded.{column}" for column in update_columns)
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO archive_records ({", ".join(columns)})
                VALUES ({placeholders})
                ON CONFLICT(id) DO UPDATE SET {update_sql}
                """,
                [record[column] for column in columns],
            )
        return self.get_record(str(record["id"]))

    def get_record(self, record_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM archive_records WHERE id = ?", (record_id,)).fetchone()
        if row is None:
            raise KeyError(record_id)
        return self._row_to_record(row)

    def list_records(
        self,
        *,
        range_value: str = "48h",
        source: str | None = None,
        record_type: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> dict[str, Any]:
        range_value = range_value.strip().lower()
        where = []
        params: list[Any] = []
        if range_value != "all":
            if range_value == "48h":
                cutoff = datetime.now(timezone.utc) - timedelta(hours=48)
            elif range_value == "7d":
                cutoff = datetime.now(timezone.utc) - timedelta(days=7)
            elif range_value == "30d":
                cutoff = datetime.now(timezone.utc) - timedelta(days=30)
            else:
                raise ArchiveValidationError("range must be one of: 48h, 7d, 30d, all")
            where.append("created_at >= ?")
            params.append(cutoff.isoformat())
        if source:
            where.append("source = ?")
            params.append(source)
        if record_type:
            where.append("record_type = ?")
            params.append(record_type)

        limit = max(1, min(int(limit), 1000))
        offset = max(0, int(offset))
        where_sql = " WHERE " + " AND ".join(where) if where else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM archive_records
                {where_sql}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                [*params, limit, offset],
            ).fetchall()
            count_row = conn.execute(
                f"SELECT COUNT(*) AS count FROM archive_records{where_sql}",
                params,
            ).fetchone()
        return {
            "ok": True,
            "enabled": True,
            "range": range_value,
            "source": source,
            "record_type": record_type,
            "limit": limit,
            "offset": offset,
            "total": int(count_row["count"] if count_row else 0),
            "records": [self._row_to_record(row) for row in rows],
        }

    def update_review(
        self,
        record_id: str,
        *,
        reviewed: bool | None = None,
        review_note: str | None = None,
    ) -> dict[str, Any]:
        fields = {"updated_at": now_iso()}
        if reviewed is not None:
            fields["reviewed"] = 1 if reviewed else 0
        if review_note is not None:
            fields["review_note"] = _truncate_text(review_note, MAX_TEXT_CHARS)
        assignments = ", ".join(f"{key} = ?" for key in fields)
        with self._connect() as conn:
            result = conn.execute(
                f"UPDATE archive_records SET {assignments} WHERE id = ?",
                [*fields.values(), record_id],
            )
        if result.rowcount == 0:
            raise KeyError(record_id)
        return self.get_record(record_id)

    def export_records(self) -> dict[str, Any]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM archive_records ORDER BY created_at DESC"
            ).fetchall()
        records = [self._row_to_record(row) for row in rows]
        return {
            "ok": True,
            "archive_version": ARCHIVE_VERSION,
            "exported_at": now_iso(),
            "db_path": str(self.db_path),
            "record_count": len(records),
            "records": records,
        }

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> dict[str, Any]:
        record = dict(row)
        record["reviewed"] = bool(record.get("reviewed"))
        record["evidence"] = _json_loads(record.pop("evidence_json", None))
        record["metadata"] = _json_loads(record.pop("metadata_json", None))
        return record


def get_archive() -> LocalArchive:
    return LocalArchive(archive_db_path())
