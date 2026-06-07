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
  -> local SQLite archive
  -> local model checkpoints / Python environment
```

## Required Local Services

- FastAPI backend running locally on `http://127.0.0.1:8000`.
- Cloudflare Tunnel running and forwarding HTTPS traffic to `http://localhost:8000`.
- Local model checkpoints and Python environment available on the developer machine.
- Optional local SQLite archive at `data/visionguard_archive.sqlite` for shared summary records.
- Optional local Next.js frontend for local validation before using Vercel.

## Environment Variables

Frontend / Vercel:

```bash
NEXT_PUBLIC_API_BASE_URL=https://<cloudflare-tunnel-url>
```

Backend:

```bash
VISIONGUARD_ALLOWED_ORIGINS=https://<vercel-app-url>,http://localhost:3000,http://127.0.0.1:3000
VISIONGUARD_ARCHIVE_ENABLED=1
VISIONGUARD_ARCHIVE_DB_PATH=data/visionguard_archive.sqlite
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

## Daily Startup After Mac Restart

Use this flow when the Mac has restarted or the Cloudflare Quick Tunnel URL has changed. The SQLite archive is local state on this Mac; do not delete `data/visionguard_archive.sqlite`. Archive data remains saved across restarts as long as the backend uses the same database file.

1. Enter the project root:

```bash
cd /Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection
```

2. Start the FastAPI backend with CORS allowed origins:

```bash
VISIONGUARD_ALLOWED_ORIGINS="https://visionguard-systemui.vercel.app,http://localhost:3000,http://127.0.0.1:3000" \
.venv-stage10/bin/python src/backend/app.py --host 127.0.0.1 --port 8000
```

3. Check local backend health:

```bash
curl http://127.0.0.1:8000/api/realtime/health
curl http://127.0.0.1:8000/api/archive/health
```

Confirm `/api/archive/health` reports the expected `record_count`. If `record_count` is missing or unexpectedly reset, stop and check `VISIONGUARD_ARCHIVE_DB_PATH` before continuing.

4. Start Cloudflare Quick Tunnel in a separate terminal:

```bash
npx -y cloudflared tunnel --url http://localhost:8000
```

5. Copy the new `trycloudflare.com` HTTPS URL from the tunnel output.

6. Check remote tunnel health before changing Vercel:

```bash
curl https://<new-tunnel-url>/api/realtime/health
curl https://<new-tunnel-url>/api/archive/health
```

Both commands must pass before the tunnel is considered usable. If tunnel creation times out, or if a URL is printed but either health check fails, do not update Vercel. Retry later, switch network, or check VPN/proxy/firewall settings.

7. Update the Vercel production environment variable only after remote health passes:

```text
NEXT_PUBLIC_API_BASE_URL=https://<new-tunnel-url>
```

8. Redeploy the Vercel production frontend.

9. Run deployment preflight:

```bash
export VISIONGUARD_REMOTE_API_BASE_URL="https://<new-tunnel-url>"
export VISIONGUARD_FRONTEND_ORIGIN="https://visionguard-systemui.vercel.app"
bash scripts/deployment_preflight.sh
```

10. Open the production frontend:

```text
https://visionguard-systemui.vercel.app
```

11. Test:

- `/`
- Start Camera
- `/api/realtime/frame` returns 200 during active monitoring
- `/video-upload` small test upload
- `/history-48h` reads `backend_archive`
- `/insights` reads `backend_archive`
- `/api/archive/export` works

12. Keep both terminal processes running:

- FastAPI backend
- Cloudflare Quick Tunnel

If the Quick Tunnel URL changes, update Vercel `NEXT_PUBLIC_API_BASE_URL` and redeploy only after the new tunnel health checks pass. The CORS allowed origin is the Vercel frontend URL, not the Cloudflare backend URL. Backend/archive data lives on the Mac at `data/visionguard_archive.sqlite`; back up that file or use `/api/archive/export`. Do not commit SQLite databases, SQLite sidecars, backups, or archive export JSON files.

If Quick Tunnel fails during startup:

```text
failed to request quick Tunnel: Post "https://api.trycloudflare.com/tunnel": context deadline exceeded
```

or if `cloudflared` reports edge TLS or port `7844` timeouts, leave Vercel unchanged and retry:

```bash
npx -y cloudflared tunnel --url http://localhost:8000
```

If retrying on the same network still fails, switch to a different network such as a phone hotspot, temporarily toggle VPN/proxy tools manually, or wait and retry. See `docs/archive/deployment/TUNNEL_DIAGNOSTIC_REPORT.md` when present for the latest network diagnostic notes.

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

## Local Backend Archive

Stage 22 adds a local SQLite archive for compact Live Monitor and uploaded-video summary records from shared clients. It stores metadata only and does not store raw webcam frames, raw images, raw videos, base64 payloads, or blob payloads.

Health check:

```bash
curl http://127.0.0.1:8000/api/archive/health
```

Export endpoint:

```bash
curl http://127.0.0.1:8000/api/archive/export
```

See `docs/LOCAL_BACKEND_ARCHIVE.md` for schema, environment variables, backup guidance, and limitations.

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

The script also checks local archive health and remote archive health when `VISIONGUARD_REMOTE_API_BASE_URL` is set. It does not write archive records by default. To write one clearly marked non-media test record, set:

```bash
VISIONGUARD_ARCHIVE_PREFLIGHT_WRITE_TEST=1
```

If using `make`, the project also provides:

```bash
make deployment-preflight
```

## Validation Checklist

- Local backend health opens: `http://127.0.0.1:8000/api/realtime/health`.
- Local archive health opens: `http://127.0.0.1:8000/api/archive/health`.
- Tunnel backend health opens: `https://<cloudflare-url>/api/realtime/health`.
- Tunnel archive health opens: `https://<cloudflare-url>/api/archive/health`.
- Vercel frontend loads.
- Login works.
- `/` loads.
- Live Monitor health check works.
- Start Camera works.
- Backend evidence returns.
- `/video-upload` small test video upload works.
- `/history-48h` loads.
- `/history-48h` shows backend_archive records when the archive is reachable, with local_only fallback.
- `/insights` loads.
- `/insights` can derive aggregates from backend_archive records when available.
- Archive export downloads a compact JSON file.
- Backend unavailable error is understandable when the backend or tunnel is stopped.
- No raw webcam frame/image/video persistence is introduced.

## Troubleshooting

| Problem | Likely cause | Action |
| --- | --- | --- |
| Cloudflare tunnel creation timeout | Cloudflare Quick Tunnel API/network/TLS issue | Do not update Vercel; retry `npx -y cloudflared tunnel --url http://localhost:8000`, switch network, or wait. |
| Quick Tunnel prints a URL but health checks timeout | Tunnel URL was created but the local network cannot complete Cloudflare edge/TLS connectivity | Do not update Vercel; keep testing the URL, retry tunnel creation, or switch network. |
| Vercel page loads but backend calls fail | Vercel still points to an old tunnel URL | Verify new tunnel realtime/archive health, update `NEXT_PUBLIC_API_BASE_URL`, then redeploy. |
| History/Insights records missing | Frontend cannot reach `/api/archive/records` or backend is using a different SQLite path | Check `/api/archive/health`, `record_count`, browser network requests, and `VISIONGUARD_ARCHIVE_DB_PATH`. |
| `record_count` reset or missing | Wrong DB path or missing `data/visionguard_archive.sqlite` | Check `VISIONGUARD_ARCHIVE_DB_PATH` and restore a backup or exported archive if needed. |
| CORS error in browser console | Backend CORS does not include the exact Vercel frontend origin | Confirm `VISIONGUARD_ALLOWED_ORIGINS` includes `https://visionguard-systemui.vercel.app` without a trailing slash, then restart the backend. |
| Tunnel URL expired or changed | Quick Tunnel URLs are temporary | Start `cloudflared tunnel --url http://localhost:8000`, copy the new HTTPS URL, verify remote health, update `NEXT_PUBLIC_API_BASE_URL`, and redeploy Vercel. |
| Backend not running | Local FastAPI process stopped | Start the backend with `make stage17-ui` or `.venv-stage10/bin/python src/backend/app.py --host 127.0.0.1 --port 8000`. |
| Vercel env changed but frontend still calls old URL | Frontend was not redeployed after env change | Redeploy the Vercel frontend after changing `NEXT_PUBLIC_API_BASE_URL`. |
| Local checkpoints missing | Backend cannot find the local model checkpoint files | Restore the local model checkpoint files and Python environment expected by the backend pipeline. |
| Camera permission blocked | Browser camera permission denied | Allow browser camera permission and use HTTPS for the deployed frontend. |
| Upload request timeout | Large upload or unstable tunnel/backend connectivity | Use a smaller test video, verify tunnel/backend connectivity, and check local backend terminal output. |
| Archive health fails | Archive disabled, wrong DB path, or database directory permissions issue | Check `VISIONGUARD_ARCHIVE_ENABLED`, `VISIONGUARD_ARCHIVE_DB_PATH`, and database directory permissions, then restart the backend. |
| Archive writes fail | Backend unavailable or optional write-token mismatch | Confirm the backend is running. If `VISIONGUARD_ARCHIVE_WRITE_TOKEN` is set, remember that browser writes need the matching header and this is not production authentication. |

## Rollback

- Revert the Vercel `NEXT_PUBLIC_API_BASE_URL` value to the previous backend URL.
- Redeploy the previous Vercel frontend build if needed.
- Stop the Cloudflare Tunnel process.
- Back up or remove the local SQLite archive if you need to reset shared summary records.
- Fall back to the local run command:

```bash
make stage17-ui
```
