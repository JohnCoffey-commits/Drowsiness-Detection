#!/usr/bin/env bash
set -euo pipefail

FAILURES=0

normalize_base_url() {
  local value="${1:-}"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  while [[ "${value}" == */ ]]; do
    value="${value%/}"
  done
  printf '%s' "${value}"
}

record_failure() {
  local message="$1"
  echo "[fail] ${message}" >&2
  FAILURES=$((FAILURES + 1))
}

check_health() {
  local label="$1"
  local base_url
  base_url="$(normalize_base_url "$2")"
  local url="${base_url}/api/realtime/health"

  echo "[check] ${label} health: ${url}"
  if curl -fsS --connect-timeout 3 --max-time 10 "${url}" >/dev/null; then
    echo "[ok] ${label} health reachable"
  else
    record_failure "${label} health check failed. Check that the FastAPI backend is running and the backend URL is correct."
  fi
}

print_archive_health_summary() {
  local body="$1"
  if command -v python3 >/dev/null 2>&1; then
    python3 - "$body" <<'PY'
import json
import sys

try:
    payload = json.loads(sys.argv[1])
except Exception:
    print("[info] archive health response could not be parsed")
    raise SystemExit(0)

print(
    "[info] archive enabled={enabled} db_exists={exists} db_writable={writable} "
    "record_count={count} latest={latest}".format(
        enabled=payload.get("enabled"),
        exists=payload.get("db_exists"),
        writable=payload.get("db_writable"),
        count=payload.get("record_count"),
        latest=payload.get("latest_record_timestamp"),
    )
)
PY
  else
    echo "[info] archive health response: ${body}"
  fi
}

check_archive_health() {
  local label="$1"
  local base_url
  base_url="$(normalize_base_url "$2")"
  local url="${base_url}/api/archive/health"
  local body

  echo "[check] ${label} archive health: ${url}"
  if body="$(curl -fsS --connect-timeout 3 --max-time 10 "${url}")"; then
    echo "[ok] ${label} archive health reachable"
    print_archive_health_summary "${body}"
  else
    record_failure "${label} archive health check failed. Check that the backend is running with Stage 22 archive endpoints."
  fi
}

archive_write_test() {
  local base_url
  base_url="$(normalize_base_url "$1")"
  local url="${base_url}/api/archive/live-event"
  local now
  local body_file
  local curl_args

  now="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  body_file="$(mktemp)"
  cat >"${body_file}" <<JSON
{
  "id": "preflight-live-event-${now}",
  "client_id": "deployment-preflight",
  "account_id": "local-preflight",
  "session_id": "deployment-preflight",
  "event_type": "signal_quality",
  "severity": "unreliable",
  "title": "Deployment preflight archive write test",
  "summary": "Clearly marked non-media archive write test.",
  "started_at": "${now}",
  "created_at": "${now}",
  "evidence": {
    "candidate_severity_score": 0
  },
  "metadata": {
    "preflight_write_test": true
  }
}
JSON

  echo "[check] archive write test: ${url}"
  curl_args=(
    -fsS
    --connect-timeout 3
    --max-time 10
    -X POST
    "${url}"
    -H "Content-Type: application/json"
    --data-binary "@${body_file}"
  )
  if [[ -n "${VISIONGUARD_ARCHIVE_WRITE_TOKEN:-}" ]]; then
    curl_args+=(-H "X-VisionGuard-Archive-Token: ${VISIONGUARD_ARCHIVE_WRITE_TOKEN}")
  fi

  if curl "${curl_args[@]}" >/dev/null; then
    echo "[ok] archive write test saved a clearly marked summary record"
  else
    record_failure "Archive write test failed. Check VISIONGUARD_ARCHIVE_ENABLED, database path permissions, and optional write token."
  fi
  rm -f "${body_file}"
}

check_cors_preflight() {
  local base_url
  base_url="$(normalize_base_url "$1")"
  local origin="$2"
  local url="${base_url}/api/realtime/session/start"
  local headers_file
  local error_file
  local status

  headers_file="$(mktemp)"
  error_file="$(mktemp)"

  echo "[check] CORS preflight: ${origin} -> ${url}"
  status="$(
    curl -sS \
      --connect-timeout 3 \
      --max-time 10 \
      -o /dev/null \
      -D "${headers_file}" \
      -w "%{http_code}" \
      -X OPTIONS "${url}" \
      -H "Origin: ${origin}" \
      -H "Access-Control-Request-Method: POST" \
      -H "Access-Control-Request-Headers: content-type" \
      2>"${error_file}" || true
  )"

  if [[ "${status}" =~ ^2[0-9][0-9]$ ]] && grep -qi '^access-control-allow-origin:' "${headers_file}"; then
    echo "[ok] CORS preflight allowed for ${origin}"
  else
    record_failure "CORS preflight failed with HTTP ${status:-000}. Add ${origin} to VISIONGUARD_ALLOWED_ORIGINS and restart the backend."
    if [[ -s "${error_file}" ]]; then
      sed 's/^/[curl] /' "${error_file}" >&2
    fi
  fi

  rm -f "${headers_file}" "${error_file}"
}

echo "VisionGuard deployment preflight"
echo "================================="

if ! command -v curl >/dev/null 2>&1; then
  record_failure "curl is required for deployment preflight checks."
else
  LOCAL_BACKEND_URL="http://127.0.0.1:8000"
  check_health "local backend" "${LOCAL_BACKEND_URL}"
  check_archive_health "local backend" "${LOCAL_BACKEND_URL}"

  if [[ -n "${NEXT_PUBLIC_API_BASE_URL:-}" ]]; then
    check_health "NEXT_PUBLIC_API_BASE_URL backend" "${NEXT_PUBLIC_API_BASE_URL}"
  else
    echo "[skip] NEXT_PUBLIC_API_BASE_URL is not set"
  fi

  if [[ -n "${VISIONGUARD_REMOTE_API_BASE_URL:-}" ]]; then
    check_health "VISIONGUARD_REMOTE_API_BASE_URL backend" "${VISIONGUARD_REMOTE_API_BASE_URL}"
    check_archive_health "VISIONGUARD_REMOTE_API_BASE_URL backend" "${VISIONGUARD_REMOTE_API_BASE_URL}"
  else
    echo "[skip] VISIONGUARD_REMOTE_API_BASE_URL is not set"
  fi

  if [[ -n "${VISIONGUARD_ALLOWED_ORIGINS:-}" ]]; then
    echo "[info] VISIONGUARD_ALLOWED_ORIGINS=${VISIONGUARD_ALLOWED_ORIGINS}"
  else
    echo "[info] VISIONGUARD_ALLOWED_ORIGINS is not set; backend local defaults should still allow local frontend origins"
  fi

  if [[ -n "${VISIONGUARD_FRONTEND_ORIGIN:-}" ]]; then
    CORS_BACKEND_URL="${VISIONGUARD_REMOTE_API_BASE_URL:-${NEXT_PUBLIC_API_BASE_URL:-${LOCAL_BACKEND_URL}}}"
    check_cors_preflight "${CORS_BACKEND_URL}" "${VISIONGUARD_FRONTEND_ORIGIN}"
  else
    echo "[skip] VISIONGUARD_FRONTEND_ORIGIN is not set; skipping CORS preflight"
  fi

  if [[ "${VISIONGUARD_ARCHIVE_PREFLIGHT_WRITE_TEST:-0}" == "1" ]]; then
    archive_write_test "${LOCAL_BACKEND_URL}"
  else
    echo "[skip] VISIONGUARD_ARCHIVE_PREFLIGHT_WRITE_TEST is not 1; skipping archive write test"
  fi
fi

echo
if [[ "${FAILURES}" -gt 0 ]]; then
  echo "Deployment preflight failed with ${FAILURES} issue(s)." >&2
  echo "Fix checklist:" >&2
  echo "- Start the local FastAPI backend on http://127.0.0.1:8000." >&2
  echo "- Start Cloudflare Tunnel with: cloudflared tunnel --url http://localhost:8000." >&2
  echo "- Set NEXT_PUBLIC_API_BASE_URL or VISIONGUARD_REMOTE_API_BASE_URL to the current HTTPS tunnel URL." >&2
  echo "- Set VISIONGUARD_ALLOWED_ORIGINS to include the exact Vercel frontend origin, then restart the backend." >&2
  echo "- For archive checks, verify VISIONGUARD_ARCHIVE_ENABLED and VISIONGUARD_ARCHIVE_DB_PATH permissions." >&2
  exit 1
fi

echo "Deployment preflight passed."
