# VisionGuard SystemUI

Next.js frontend for the VisionGuard warning-candidate prototype.

Routes:

- `/` Live Monitor webcam warning-candidate UI.
- `/video-upload` uploaded-video warning-candidate review page.
- `/history-48h` frontend-only history review page.

The UI is a warning-candidate prototype. It must not be described as final system-level drowsiness accuracy.

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

## Simple Vercel Frontend Deploy

For a simple hosted demo, deploy only this `SystemUI` directory to Vercel and keep the Python FastAPI model backend on a separate server.

Recommended Vercel settings:

- Root Directory: `SystemUI`
- Install Command: `npm install`
- Build Command: `npm run build`
- Environment Variable: `NEXT_PUBLIC_API_BASE_URL=https://your-backend-domain.example`

The backend must allow the Vercel frontend origin through `VISIONGUARD_CORS_ORIGINS`, for example:

```bash
VISIONGUARD_CORS_ORIGINS=https://your-vercel-app.vercel.app
```

Browser webcam access requires HTTPS in deployed environments.
