# Stage 22 - Local Backend Archive

Last updated: 2026-05-19

## Purpose

Stage 22 centralizes shared testing records on the developer's Mac without adding user registration, Supabase, or a cloud database. It keeps the current remote access architecture and stores compact analysis summaries in a local SQLite database owned by the FastAPI backend.

## Architecture

```text
Remote browser
  -> Vercel SystemUI
  -> Cloudflare Quick Tunnel
  -> local FastAPI backend
  -> local SQLite archive
```

## What Is Stored

- Stable Live Monitor event summaries.
- Uploaded-video analysis summaries.
- Timestamps, session IDs, client IDs, and local account IDs.
- Severity and event type metadata.
- Compact evidence metadata such as warning counts and probability summaries.
- Review state and optional review notes.

## What Is Not Stored

- Raw webcam frames.
- Raw images.
- Raw videos.
- Base64/blob payloads.
- Uploaded raw video files as archive records.

The existing Stage 17 upload pipeline still writes its normal local run artifacts under `outputs/system_video_upload_runs/`; the archive stores only compact summaries.

## Database Path

Default path:

```text
data/visionguard_archive.sqlite
```

Override with:

```bash
VISIONGUARD_ARCHIVE_DB_PATH=/absolute/or/project-relative/path/to/archive.sqlite
```

The backend creates the parent directory automatically. SQLite sidecar WAL files are local runtime artifacts and should not be committed.

## Environment Variables

```bash
VISIONGUARD_ARCHIVE_ENABLED=1
VISIONGUARD_ARCHIVE_DB_PATH=data/visionguard_archive.sqlite
VISIONGUARD_ARCHIVE_WRITE_TOKEN=<optional-light-write-token>
VISIONGUARD_ALLOWED_ORIGINS=https://<vercel-app-url>,http://localhost:3000,http://127.0.0.1:3000
```

`VISIONGUARD_ARCHIVE_ENABLED` defaults to enabled when unset. If `VISIONGUARD_ARCHIVE_WRITE_TOKEN` is set, archive write endpoints require the `X-VisionGuard-Archive-Token` header. That token is only a light misuse guard, not production authentication.

## Startup Flow

1. Start the local FastAPI backend.
2. Start Cloudflare Quick Tunnel.
3. If the tunnel URL changed, update Vercel `NEXT_PUBLIC_API_BASE_URL` and redeploy.
4. Run `scripts/deployment_preflight.sh`.
5. Check `http://127.0.0.1:8000/api/archive/health`.
6. If using a tunnel, check `https://<cloudflare-url>/api/archive/health`.

## Backup And Export

- Use the History page export action or `GET /api/archive/export` to download JSON.
- For a direct SQLite backup, stop the backend first and back up `data/visionguard_archive.sqlite` plus any SQLite sidecar files.
- Keep SQLite databases, SQLite sidecars, backups, and `visionguard-archive-export-*.json` files out of Git.

## Test Record Cleanup

Only delete archive records that are clearly marked as validation or preflight records, such as records with `metadata.validation_record=true`, `metadata.preflight_write_test=true`, or obvious validation titles. Do not delete ambiguous shared testing records.

## Limitations

- The Mac must remain on.
- The local backend must run.
- The Cloudflare Quick Tunnel must run.
- Quick Tunnel URLs may change.
- There is no user registration.
- This is not a cloud database.
- SQLite is intended for low-volume shared testing, not high-concurrency production traffic.
