# VisionGuard Daily Startup Checklist

Use this after a Mac restart or whenever the Cloudflare Quick Tunnel URL changes.

## 1. Start Backend

```bash
cd /Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection
VISIONGUARD_ALLOWED_ORIGINS="https://visionguard-systemui.vercel.app,http://localhost:3000,http://127.0.0.1:3000" \
.venv-stage10/bin/python src/backend/app.py --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
curl http://127.0.0.1:8000/api/realtime/health
curl http://127.0.0.1:8000/api/archive/health
```

## 2. Start Tunnel

```bash
npx -y cloudflared tunnel --url http://localhost:8000
```

Copy the new `https://<new-tunnel-url>` value.

```bash
curl https://<new-tunnel-url>/api/realtime/health
curl https://<new-tunnel-url>/api/archive/health
```

## 3. Update Vercel

Set production env:

```text
NEXT_PUBLIC_API_BASE_URL=https://<new-tunnel-url>
```

Redeploy production frontend:

```text
https://visionguard-systemui.vercel.app
```

## 4. Run Preflight

```bash
export VISIONGUARD_REMOTE_API_BASE_URL="https://<new-tunnel-url>"
export VISIONGUARD_FRONTEND_ORIGIN="https://visionguard-systemui.vercel.app"
bash scripts/deployment_preflight.sh
```

## 5. Smoke Test

- `/`
- Start Camera
- `/video-upload` with a small test upload
- `/history-48h` shows `backend_archive` when records exist
- `/insights` reads archive data when records exist
- `/api/archive/export`

Keep both the FastAPI backend terminal and Cloudflare Tunnel terminal running.
