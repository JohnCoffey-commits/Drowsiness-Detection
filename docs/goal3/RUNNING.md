# Run VisionGuard from a fresh checkout

This is a local research prototype. The Compose ports bind only to localhost. No camera, cloud deployment or GPU is required. Use a video you own or have permission to process. Do not upload private inputs or model checkpoints to GitHub.

## Prerequisites and checkout

Install Git and Docker Desktop with Compose. Native checks use Python 3.11.18 and Node 24.22.2. The verified development platform is macOS arm64; the container target is Linux arm64 CPU. CI checks target Ubuntu 24.04 x86_64 with model doubles. A working x86_64 model container must not be assumed from CI.

```bash
git clone --branch learning/reproducible-delivery https://github.com/JohnCoffey-commits/Drowsiness-Detection.git VisionGuard
cd VisionGuard
git rev-parse HEAD
# For exact reproduction, git checkout the SHA recorded in the evidence pack.
```

## Prepare the three private model files

Follow `MODEL_ASSETS.md`, placing the exact files at their documented paths. Obtain authorized access from the owner; the public repository does not grant redistribution rights. Existing verified files may be copied as data assets; never copy a virtual environment or node_modules from another checkout.

```bash
python3.11 scripts/verify_assets.py
```

All three `valid` fields must be true. This verifies identity, not model accuracy. Missing bind sources cause Compose to fail rather than silently create directories.

## One command to build and start

```bash
docker compose up --build -d --wait
```

Open http://127.0.0.1:3000/video-upload. Backend: http://127.0.0.1:8000. An optional `.env` can be copied from `.env.example` when paths/ports differ; defaults need no hidden configuration.

The browser cannot resolve the Compose service name `backend`. `NEXT_PUBLIC_API_BASE_URL` is a browser URL baked into the frontend build. If the API port changes, update both `API_PORT` and that URL, update `FRONTEND_ORIGIN` when the frontend port changes, then rebuild the frontend. Runtime environment changes alone do not rewrite an existing Next.js browser bundle.

```bash
curl --fail http://127.0.0.1:8000/health/live
curl --fail http://127.0.0.1:8000/health/ready
curl --fail -F 'file=@/absolute/path/authorized-video.mp4' http://127.0.0.1:8000/api/analyze-video
```

Liveness checks the web process. Readiness loads the actual three model assets and reports the device, or returns 503. A successful video response plus its summary/timeline provides separate pipeline evidence. First readiness or inference takes longer while dependencies and models load. The single worker is intentional because realtime sessions live in process memory.

## Persistence and stopping

SQLite, upload inputs, analysis outputs and audit logs live in the named `archive-data` volume at `/app/data`. Models are separate readonly file mounts. Container processes run as non-root users. `docker compose stop`, `start`, `restart`, and `down` preserve this named volume. Do not use `down -v` unless intentionally deleting that project's retained data.

```bash
docker compose logs --tail=100 backend
docker compose stop
docker compose start --wait
```

In-memory realtime sessions reset on backend restart. SQLite archive rows and completed upload outputs persist. Uploaded faces and outputs remain private; establish retention and authorized access before any broader deployment.

## Local automated checks

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install pip==26.3 setuptools==84.2.1
python -m pip install -r requirements/dev.txt
npm --prefix SystemUI ci
bash scripts/check.sh all
```

These tests require no model, video, torch or camera. To run native macOS inference, additionally install `python -m pip install -r requirements.txt` and set `VISIONGUARD_DEVICE=cpu`. Use `python -m uvicorn src.backend.app:app --host 127.0.0.1 --port 8000`; use `npm --prefix SystemUI run dev` for the UI. Historical helper scripts that refer to `.venv-stage10` are not the Goal 3 entry point.

Build layers cache locked dependencies before source files. A cached build is recorded separately from a first build. The container is rebuilt when relevant source or lockfiles change. Network access to official registries is still required for an uncached build.
