#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/activate_deployment_env.sh"
cd "${ROOT_DIR}"

BACKEND_HOST="127.0.0.1"
BACKEND_PORT="8000"
FRONTEND_HOST="127.0.0.1"
FRONTEND_PORT="3000"
BACKEND_URL="http://${BACKEND_HOST}:${BACKEND_PORT}"
FRONTEND_URL="http://${FRONTEND_HOST}:${FRONTEND_PORT}/video-upload"
BACKEND_PID=""
FRONTEND_PID=""
CLEANED_UP=0

cleanup() {
  if [[ "${CLEANED_UP}" -eq 1 ]]; then
    return
  fi
  CLEANED_UP=1
  echo
  echo "[stage17-ui] Stopping services..."
  if [[ -n "${FRONTEND_PID}" ]] && kill -0 "${FRONTEND_PID}" 2>/dev/null; then
    pkill -TERM -P "${FRONTEND_PID}" 2>/dev/null || true
    kill "${FRONTEND_PID}" 2>/dev/null || true
  fi
  if [[ -n "${BACKEND_PID}" ]] && kill -0 "${BACKEND_PID}" 2>/dev/null; then
    pkill -TERM -P "${BACKEND_PID}" 2>/dev/null || true
    kill "${BACKEND_PID}" 2>/dev/null || true
  fi
  wait "${FRONTEND_PID}" 2>/dev/null || true
  wait "${BACKEND_PID}" 2>/dev/null || true
  echo "[stage17-ui] Stopped."
}

trap cleanup EXIT
trap 'exit 130' INT TERM

wait_for_url() {
  local name="$1"
  local url="$2"
  local pid="$3"
  local max_attempts="${4:-60}"

  for ((attempt = 1; attempt <= max_attempts; attempt++)); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "[stage17-ui] ${name} exited before it became ready." >&2
      return 1
    fi
    if curl -fsS "${url}" >/dev/null 2>&1; then
      echo "[stage17-ui] ${name} ready: ${url}"
      return 0
    fi
    sleep 1
  done

  echo "[stage17-ui] Timed out waiting for ${name}: ${url}" >&2
  return 1
}

if [[ ! -x "${ROOT_DIR}/.venv-stage10/bin/python" ]]; then
  echo "[stage17-ui] Missing backend Python: ${ROOT_DIR}/.venv-stage10/bin/python" >&2
  exit 1
fi

if [[ "${VIRTUAL_ENV:-}" != "${ROOT_DIR}/.venv-stage10" ]]; then
  echo "[stage17-ui] Expected VIRTUAL_ENV=${ROOT_DIR}/.venv-stage10, got ${VIRTUAL_ENV:-<unset>}." >&2
  exit 1
fi

if [[ ! -f "${ROOT_DIR}/src/backend/app.py" ]]; then
  echo "[stage17-ui] Missing backend app: ${ROOT_DIR}/src/backend/app.py" >&2
  exit 1
fi

if [[ ! -f "${ROOT_DIR}/SystemUI/package.json" ]]; then
  echo "[stage17-ui] Missing frontend package.json: ${ROOT_DIR}/SystemUI/package.json" >&2
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "[stage17-ui] npm is required to start the Next.js frontend." >&2
  exit 1
fi

echo "[stage17-ui] Starting FastAPI backend on ${BACKEND_URL}"
python "${ROOT_DIR}/src/backend/app.py" \
  --host "${BACKEND_HOST}" \
  --port "${BACKEND_PORT}" &
BACKEND_PID=$!

echo "[stage17-ui] Starting Next.js frontend on http://${FRONTEND_HOST}:${FRONTEND_PORT}"
(
  cd "${ROOT_DIR}/SystemUI"
  npm run dev -- --hostname "${FRONTEND_HOST}" --port "${FRONTEND_PORT}"
) &
FRONTEND_PID=$!

wait_for_url "Backend" "${BACKEND_URL}/static/upload_test.html" "${BACKEND_PID}" 45
wait_for_url "Frontend" "${FRONTEND_URL}" "${FRONTEND_PID}" 60

cat <<EOF

[stage17-ui] Stage 17.5 Video Upload Evidence Review UI is running.
[stage17-ui] Frontend URL: ${FRONTEND_URL}
[stage17-ui] Backend URL:  ${BACKEND_URL}
[stage17-ui] VIRTUAL_ENV:  ${VIRTUAL_ENV}
[stage17-ui] API base URL: ${NEXT_PUBLIC_API_BASE_URL}
[stage17-ui] Press Ctrl+C to stop both services.

EOF

while true; do
  if ! kill -0 "${BACKEND_PID}" 2>/dev/null; then
    echo "[stage17-ui] Backend process exited."
    exit 1
  fi
  if ! kill -0 "${FRONTEND_PID}" 2>/dev/null; then
    echo "[stage17-ui] Frontend process exited."
    exit 1
  fi
  sleep 2
done
