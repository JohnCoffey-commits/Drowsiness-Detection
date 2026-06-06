# Stage 17.3 Local Launch Guide

## One-command Launcher

From the repository root:

```bash
./scripts/start_stage17_ui.sh
```

Or with Make:

```bash
make stage17-ui
```

## URLs

- Frontend: `http://127.0.0.1:3000/video-upload`
- Backend: `http://127.0.0.1:8000`
- Backend API used by the UI: `POST http://127.0.0.1:8000/api/analyze-video`

## What the Launcher Starts

- FastAPI backend:

```bash
.venv-stage10/bin/python src/backend/app.py --host 127.0.0.1 --port 8000
```

- Next.js frontend from `SystemUI/`:

```bash
npm run dev -- --hostname 127.0.0.1 --port 3000
```

## How to Stop

Press `Ctrl+C` in the terminal running the launcher. The script traps the stop signal and terminates both backend and frontend processes.

## Common Troubleshooting

### Backend Python is missing

If you see:

```text
Missing backend Python: .venv-stage10/bin/python
```

Use the repository virtual environment expected by Stage 17.3, or recreate it before running the launcher.

### Frontend dependencies are missing

If `npm run dev` fails because packages are missing:

```bash
cd SystemUI
npm install
cd ..
./scripts/start_stage17_ui.sh
```

### Port already in use

The launcher uses:

- Backend port `8000`
- Frontend port `3000`

If either port is already occupied, stop the existing process and rerun the launcher.

### Backend dependencies are missing in system Python

Do not start the backend with plain `python` unless that environment has FastAPI dependencies installed. The launcher intentionally uses:

```bash
.venv-stage10/bin/python
```

### Upload analysis fails

Confirm that the backend is reachable at `http://127.0.0.1:8000`, then retry the upload from `http://127.0.0.1:3000/video-upload`.

This launcher only starts the Stage 17.3 local UI and backend. It does not change model logic, Stage 17.1 fusion logic, training code, or model checkpoints.
