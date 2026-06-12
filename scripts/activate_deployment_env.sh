#!/usr/bin/env bash
set -Eeuo pipefail

if [[ -n "${BASH_VERSION:-}" ]]; then
  SCRIPT_SOURCE="${BASH_SOURCE[0]}"
  SCRIPT_SOURCED=0
  if [[ "${SCRIPT_SOURCE}" != "${0}" ]]; then
    SCRIPT_SOURCED=1
  fi
elif [[ -n "${ZSH_VERSION:-}" ]]; then
  SCRIPT_SOURCE="${(%):-%x}"
  SCRIPT_SOURCED=0
  if [[ "${ZSH_EVAL_CONTEXT:-}" == *:file* ]]; then
    SCRIPT_SOURCED=1
  fi
else
  echo "Unsupported shell. Source this script from bash or zsh." >&2
  return 1 2>/dev/null || exit 1
fi

if [[ "${SCRIPT_SOURCED}" -ne 1 ]]; then
  echo "Source this script instead: source scripts/activate_deployment_env.sh" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${SCRIPT_SOURCE}")/.." && pwd)"
PYTHON_ENV_DIR="${ROOT_DIR}/.venv-stage10"
PYTHON_ACTIVATE="${PYTHON_ENV_DIR}/bin/activate"
PYTHON_BIN="${PYTHON_ENV_DIR}/bin/python"

if [[ ! -f "${PYTHON_ACTIVATE}" ]] || [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[visionguard-env] Missing Stage 10 Python environment: ${PYTHON_BIN}" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "${PYTHON_ACTIVATE}"

export VISIONGUARD_PROJECT_ROOT="${ROOT_DIR}"
export VISIONGUARD_ALLOWED_ORIGINS="${VISIONGUARD_ALLOWED_ORIGINS:-https://visionguard-systemui.vercel.app,http://localhost:3000,http://127.0.0.1:3000}"
export VISIONGUARD_ARCHIVE_ENABLED="${VISIONGUARD_ARCHIVE_ENABLED:-1}"
export VISIONGUARD_ARCHIVE_DB_PATH="${VISIONGUARD_ARCHIVE_DB_PATH:-data/visionguard_archive.sqlite}"
export NEXT_PUBLIC_API_BASE_URL="${NEXT_PUBLIC_API_BASE_URL:-http://127.0.0.1:8000}"
export VISIONGUARD_FRONTEND_ORIGIN="${VISIONGUARD_FRONTEND_ORIGIN:-https://visionguard-systemui.vercel.app}"

if [[ -z "${VISIONGUARD_REMOTE_API_BASE_URL:-}" && "${NEXT_PUBLIC_API_BASE_URL}" =~ ^https:// ]]; then
  export VISIONGUARD_REMOTE_API_BASE_URL="${NEXT_PUBLIC_API_BASE_URL}"
fi

if [[ -d "${ROOT_DIR}/SystemUI/node_modules/.bin" ]]; then
  export PATH="${ROOT_DIR}/SystemUI/node_modules/.bin:${PATH}"
fi
