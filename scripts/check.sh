#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
scope="${1:-all}"
if [[ "$scope" == backend || "$scope" == all ]]; then
  mkdir -p reports
  python -m pip check
  python -m pytest -q --junitxml=reports/pytest.xml --cov=src.backend --cov=src.runtime.realtime_temporal_state --cov-report=xml:reports/coverage.xml --cov-report=term-missing
  python -m ruff check src/backend/app.py src/backend/local_archive.py src/runtime/realtime_frame_inference.py tests scripts/verify_assets.py
  python -m black --check src/backend/app.py src/backend/local_archive.py src/runtime/realtime_frame_inference.py tests/conftest.py tests/test_delivery_boundaries.py tests/test_runtime_health.py scripts/verify_assets.py
  python -m mypy src/backend/local_archive.py scripts/verify_assets.py
fi
if [[ "$scope" == frontend || "$scope" == all ]]; then
  npm --prefix SystemUI run lint
  npm --prefix SystemUI run typecheck
  npm --prefix SystemUI run build
fi
if [[ "$scope" == audit || "$scope" == all ]]; then
  mkdir -p reports
  python -m pip_audit -r requirements/dev.txt --format json --output reports/python-dev-audit.json
  python -m pip_audit -r requirements/runtime.txt --format json --output reports/python-runtime-audit.json
  npm --prefix SystemUI audit --audit-level=high
fi
