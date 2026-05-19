# VisionGuard Deployment Runbook

Last updated: 2026-05-19

## Purpose

Deploy the `SystemUI/` frontend for external access, connect it to the local FastAPI inference backend through Cloudflare Tunnel, and validate the modular VisionGuard Drowsiness Detection and Monitoring System remotely.

The system uses modular CNN-based eye and mouth evidence extraction with temporal rule-based fusion for drowsiness detection and monitoring. This runbook covers deployment configuration and validation only; it does not change model inference, temporal logic, thresholds, or local checkpoint requirements.

## Architecture

```text
Remote browser
  -> Vercel-hosted Next.js SystemUI frontend
  -> HTTPS Cloudflare Tunnel backend URL
  -> local FastAPI backend
  -> local model checkpoints / Python environment
```

## Required Local Services

- FastAPI backend running locally on `http://127.0.0.1:8000`.
- Cloudflare Tunnel running and forwarding HTTPS traffic to `http://localhost:8000`.
- Local model checkpoints and Python environment available on the developer machine.
- Optional local Next.js frontend for local validation before using Vercel.

## Environment Variables

Frontend / Vercel:

```bash
NEXT_PUBLIC_API_BASE_URL=https://<cloudflare-tunnel-url>
```

Backend:

```bash
VISIONGUARD_ALLOWED_ORIGINS=https://<vercel-app-url>,http://localhost:3000,http://127.0.0.1:3000
```

Restart the backend after changing `VISIONGUARD_ALLOWED_ORIGINS`. Redeploy the Vercel frontend after changing `NEXT_PUBLIC_API_BASE_URL`.

## Local Backend Start Command

The inspected project launcher is `scripts/start_stage17_ui.sh`, exposed through:

```bash
make stage17-ui
```
That launcher starts both the local FastAPI backend and local Next.js frontend. The inspected backend command inside the launcher is:

```bash
.venv-stage10/bin/python src/backend/app.py --host 127.0.0.1 --port 8000
```

For remote frontend validation, the local backend must be reachable at `http://127.0.0.1:8000`. If `.venv-stage10/bin/python` is missing, restore the project Python environment before starting the backend.

## Cloudflare Tunnel

Start a Quick Tunnel to the local backend:

```bash
cloudflared tunnel --url http://localhost:8000
```

Copy the generated HTTPS URL and use it as `NEXT_PUBLIC_API_BASE_URL` in Vercel. Temporary Quick Tunnel URLs may change. If the tunnel URL changes, update the Vercel environment variable and redeploy the frontend. A named Cloudflare Tunnel can be configured later for a stable backend URL.

## Vercel Frontend Deployment

- Deploy `SystemUI/` as the Vercel project root.
- Set `NEXT_PUBLIC_API_BASE_URL=https://<cloudflare-tunnel-url>`.
- Redeploy after changing `NEXT_PUBLIC_API_BASE_URL`.
- Do not deploy Python checkpoints or the FastAPI backend to Vercel in this architecture.
- The local FastAPI backend remains responsible for model inference and API responses.

## Backend CORS Setup

Set `VISIONGUARD_ALLOWED_ORIGINS` before starting the backend. Include the Vercel app URL and local development origins:

```bash
VISIONGUARD_ALLOWED_ORIGINS=https://<vercel-app-url>,http://localhost:3000,http://127.0.0.1:3000
```

The backend strips trailing slashes, ignores empty entries, deduplicates origins, and keeps local defaults. Restart the backend after changing this variable.

## Preflight Script

Run the deployment preflight script from the repository root:

```bash
scripts/deployment_preflight.sh
```

Optional environment variables:

```bash
NEXT_PUBLIC_API_BASE_URL=https://<cloudflare-tunnel-url>
VISIONGUARD_REMOTE_API_BASE_URL=https://<cloudflare-tunnel-url>
VISIONGUARD_FRONTEND_ORIGIN=https://<vercel-app-url>
VISIONGUARD_ALLOWED_ORIGINS=https://<vercel-app-url>,http://localhost:3000,http://127.0.0.1:3000
```

The script checks local backend health, optional remote backend health, prints configured CORS origins, and can run a CORS preflight OPTIONS request when `VISIONGUARD_FRONTEND_ORIGIN` is set.

If using `make`, the project also provides:

```bash
make deployment-preflight
```

## Validation Checklist

- Local backend health opens: `http://127.0.0.1:8000/api/realtime/health`.
- Tunnel backend health opens: `https://<cloudflare-url>/api/realtime/health`.
- Vercel frontend loads.
- Login works.
- `/` loads.
- Live Monitor health check works.
- Start Camera works.
- Backend evidence returns.
- `/video-upload` small test video upload works.
- `/history-48h` loads.
- `/insights` loads.
- Backend unavailable error is understandable when the backend or tunnel is stopped.
- No raw webcam frame/image/video persistence is introduced.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| CORS error in browser console | Confirm `VISIONGUARD_ALLOWED_ORIGINS` includes the exact Vercel origin without a trailing slash, then restart the backend. |
| Tunnel URL expired or changed | Start `cloudflared tunnel --url http://localhost:8000`, copy the new HTTPS URL, update `NEXT_PUBLIC_API_BASE_URL`, and redeploy Vercel. |
| Backend not running | Start the backend with `make stage17-ui` or `.venv-stage10/bin/python src/backend/app.py --host 127.0.0.1 --port 8000`. |
| Vercel env changed but frontend still calls old URL | Redeploy the Vercel frontend after changing `NEXT_PUBLIC_API_BASE_URL`. |
| Local checkpoints missing | Restore the local model checkpoint files and Python environment expected by the backend pipeline. |
| Camera permission blocked | Allow browser camera permission and use HTTPS for the deployed frontend. |
| Upload request timeout | Use a smaller test video, verify tunnel/backend connectivity, and check local backend terminal output. |

## Rollback

- Revert the Vercel `NEXT_PUBLIC_API_BASE_URL` value to the previous backend URL.
- Redeploy the previous Vercel frontend build if needed.
- Stop the Cloudflare Tunnel process.
- Fall back to the local run command:

```bash
make stage17-ui
```
