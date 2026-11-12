# syntax=docker/dockerfile:1
FROM python:3.11.18-slim-bookworm@sha256:a83ba18aef0f865b1a4a611c2cf460fe446f2795673bbb9fe43be570439e3b39
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 MPLCONFIGDIR=/tmp/matplotlib \
    VISIONGUARD_DEVICE=cpu VISIONGUARD_ARCHIVE_DB_PATH=/app/data/archive.sqlite \
    VISIONGUARD_RUNS_ROOT=/app/data/runs VISIONGUARD_AUDIT_DIR=/app/data/audits
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libegl1 libgles2 libglib2.0-0 libportaudio2 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app && useradd --uid 10001 --gid app --create-home app
COPY requirements/ requirements/
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install pip==26.3 setuptools==84.2.1 \
    && python -m pip install -r requirements/runtime.txt -r requirements/torch-constraints.txt \
    && python -m pip install -r requirements/torch-cpu.txt -c requirements/torch-constraints.txt \
    && python -m pip check
COPY src/ src/
COPY scripts/verify_assets.py scripts/verify_assets.py
RUN mkdir -p data outputs/mrl_eye/checkpoints checkpoints artifacts/models \
    && chown -R app:app data
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=6 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=2)"
CMD ["python", "-m", "uvicorn", "src.backend.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
