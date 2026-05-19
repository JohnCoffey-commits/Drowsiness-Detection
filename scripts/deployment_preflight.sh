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

  if [[ -n "${NEXT_PUBLIC_API_BASE_URL:-}" ]]; then
    check_health "NEXT_PUBLIC_API_BASE_URL backend" "${NEXT_PUBLIC_API_BASE_URL}"
  else
    echo "[skip] NEXT_PUBLIC_API_BASE_URL is not set"
  fi

  if [[ -n "${VISIONGUARD_REMOTE_API_BASE_URL:-}" ]]; then
    check_health "VISIONGUARD_REMOTE_API_BASE_URL backend" "${VISIONGUARD_REMOTE_API_BASE_URL}"
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
fi

echo
if [[ "${FAILURES}" -gt 0 ]]; then
  echo "Deployment preflight failed with ${FAILURES} issue(s)." >&2
  echo "Fix checklist:" >&2
  echo "- Start the local FastAPI backend on http://127.0.0.1:8000." >&2
  echo "- Start Cloudflare Tunnel with: cloudflared tunnel --url http://localhost:8000." >&2
  echo "- Set NEXT_PUBLIC_API_BASE_URL or VISIONGUARD_REMOTE_API_BASE_URL to the current HTTPS tunnel URL." >&2
  echo "- Set VISIONGUARD_ALLOWED_ORIGINS to include the exact Vercel frontend origin, then restart the backend." >&2
  exit 1
fi

echo "Deployment preflight passed."
