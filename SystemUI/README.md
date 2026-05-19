# VisionGuard SystemUI

Next.js frontend for the VisionGuard Drowsiness Detection and Monitoring System.

Routes:

- `/` Live Monitor webcam UI.
- `/video-upload` uploaded-video evidence review page.
- `/history-48h` frontend-only history review page.
- `/insights` frontend-only analytics page.

The UI keeps the project boundary text for warning-candidate outputs and must not be described as final system-level drowsiness accuracy.

## Local Development

Install dependencies:

```bash
npm install
```

Run the frontend:

```bash
npm run dev
```

Run the FastAPI backend from the repository root:

```bash
python src/backend/app.py --host 127.0.0.1 --port 8000
```

The frontend defaults to `http://127.0.0.1:8000` for backend API calls. Override it with:

```bash
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

## Vercel Frontend External Access

For external access, deploy only this `SystemUI` directory to Vercel and keep the Python FastAPI model backend on the developer's local machine behind an HTTPS Cloudflare Tunnel.

Recommended Vercel settings:

- Root Directory: `SystemUI`
- Install Command: `npm install`
- Build Command: `npm run build`
- Environment Variable: `NEXT_PUBLIC_API_BASE_URL=https://<cloudflare-tunnel-url>`

The backend must allow the Vercel frontend origin through `VISIONGUARD_ALLOWED_ORIGINS`, for example:

```bash
VISIONGUARD_ALLOWED_ORIGINS=https://<vercel-app-url>,http://localhost:3000,http://127.0.0.1:3000
```

Browser webcam access requires HTTPS in deployed environments.

See `../docs/DEPLOYMENT_RUNBOOK.md` for the full Vercel, Cloudflare Tunnel, CORS, and preflight validation flow.
