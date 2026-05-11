# Stage 17 SystemUI and Backend Audit

## 1. Frontend Framework

`SystemUI/` is a Next.js application.

Evidence:

- `SystemUI/package.json`
- `next.config.ts`
- `src/app/`
- React 19 / Next 16 dependencies
- Existing dashboard routes under `SystemUI/src/app/*`

The current UI is a dashboard-style app named VisionGuard, with mock dashboard data in `SystemUI/src/lib/mockData.ts`.

## 2. Existing Backend

No existing Python backend or upload API was found in the repository.

No existing Next.js API route for video upload was found.

## 3. Existing Video Upload Component

No existing video-upload component or route was found.

The existing UI contains pages for dashboard, live monitor, history, session review, insights, model details, alerts, and settings.

## 4. Safest Integration Approach

The safest approach is:

1. Add a standalone Python FastAPI backend under `src/backend/app.py`.
2. Add a standalone backend test page under `src/backend/static/upload_test.html`.
3. Add a new isolated SystemUI route under `SystemUI/src/app/video-upload/page.tsx`.
4. Add a sidebar link to the new route.
5. Keep existing dashboard/mock pages untouched.
6. Keep all Stage 8-16 model logic intact.

This avoids destructive SystemUI changes and keeps the upload MVP separable from the existing dashboard.

## 5. Files Changed

- `SystemUI/src/components/dashboard/Sidebar.tsx`
  - Added a `Video Upload` navigation item.

## 6. Files Created

- `src/runtime/system_video_upload_pipeline.py`
- `src/runtime/keyframe_extractor.py`
- `src/backend/app.py`
- `src/backend/static/upload_test.html`
- `SystemUI/src/app/video-upload/page.tsx`
- `docs/STAGE17_VIDEO_UPLOAD_RESULT_SCHEMA.md`
- `docs/STAGE17_VIDEO_UPLOAD_DETECTION_LOG.md`
- `reports/stage17_video_upload_detection_mvp_report.md`
- `artifacts/audits/stage17_video_upload_mvp_2026-05-09/stage17_systemui_backend_audit.md`

## 7. Blockers and Caveats

- The backend depends on FastAPI, uvicorn, and python-multipart. Backend preflight initially reported that FastAPI was missing, so the minimal backend dependencies were installed into `.venv-stage10`.
- The MVP processes short demo videos synchronously. Long videos may take too long for a browser request and should be moved to a background job later.
- SystemUI and the FastAPI backend run as separate services for the MVP.
- The backend enables local-development CORS for `localhost` / `127.0.0.1` SystemUI origins.
- Results are warning-candidate analysis only. They are not final drowsy/not-drowsy truth and not final system-level drowsiness accuracy.
