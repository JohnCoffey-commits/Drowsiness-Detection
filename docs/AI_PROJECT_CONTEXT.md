# VisionGuard AI Project Context

Last updated: 2026-05-29

Purpose: this document is a compact handoff brief for a new AI assistant window. Read this first when continuing work on VisionGuard, then inspect the referenced files before making changes.

## 1. Project In One Paragraph

VisionGuard is a modular driver drowsiness monitoring project. It is not a single end-to-end `drowsy / not-drowsy` classifier. The system extracts two specialist visual evidence signals, `p_eye_closed` from an MRL Eye open/closed model and `p_yawn` from a YawDD/YawDD+ mouth/yawn model, then uses rule-based temporal logic and signal-quality checks to produce alert or warning-candidate states. The current product surface includes a Next.js frontend (`SystemUI/`) with Live Monitor, Video Upload Analysis, History, Insights, local MVP login/theme/settings/notifications, and a Python FastAPI backend for upload analysis, realtime frame inference, and local SQLite archive records.

## 2. Current Product Surface

| Area | Route / location | Current role |
| --- | --- | --- |
| Live Monitor | `/` | Realtime webcam monitoring. Starts backend realtime sessions, samples frames, shows Drowsiness Risk, warning overlays, sound alerts, critical warning acknowledgement, and optional Minimal Live Monitor Mode. |
| Video Upload Analysis | `/video-upload` | Backend-connected upload analysis workspace. Uploads a video, runs the Python pipeline, displays analysis summary, alert intervals, backend-generated evidence figures, keyframes, and collapsed technical evidence. |
| History | `/history-48h` | User-facing History page. Sidebar/title label is `History`, but route remains `/history-48h` for compatibility. Default time window remains `Last 48 hours`. Data shown here is Live Monitor history only. |
| Insights | `/insights` | Compact analytics summary over recent Live Monitor alerts. Includes Key Insight Summary, Drive Highlights, Alert Mix, Time of Day, Camera Signal, Attention Areas, and HTML report download. |
| Settings | Top-right profile menu | Minimal local settings modal. Currently includes `Minimal Live Monitor Mode`, persisted in `visionguard.settings.v1`. |
| Local archive | FastAPI + SQLite | Stores compact summary records in `data/visionguard_archive.sqlite`; no raw media. |
| Remote access | Vercel + Cloudflare Quick Tunnel | Hosted frontend calls the developer Mac's local backend through a tunnel. This is external-access testing, not a cloud-native backend. |

## 3. Current Non-Goals And Safety Boundaries

Do not accidentally introduce any of these unless the user explicitly asks:

- No model retraining or inference changes during UI/doc work.
- No changes to `p_eye_closed`, `p_yawn`, temporal fusion, thresholds, debounce/cooldown, sound logic, upload pipeline, keyframe extraction, or archive schema.
- No deletion, reset, migration, or overwrite of localStorage history records.
- No deletion or modification of backend SQLite archive data.
- No raw webcam frame, uploaded video, blob, base64, or large binary storage in history/archive/report exports.
- No production authentication, registration, password change, Supabase/cloud database, browser notification API, GPS/maps, driver safety score, medical diagnosis score, or final pass/fail safety judgment.
- Do not claim final system-level drowsiness accuracy. Specialist model metrics are not complete runtime-system accuracy.

## 4. High-Level Architecture

```text
Browser / SystemUI
  -> Live Monitor webcam sampling or Video Upload
  -> FastAPI backend
  -> MediaPipe face/landmark ROI extraction
  -> eye specialist model -> p_eye_closed
  -> mouth/yawn specialist model -> p_yawn
  -> signal quality checks
  -> rule-based temporal state / fusion
  -> UI alerts, upload summaries, keyframes, evidence figures
  -> localStorage summaries and/or local SQLite archive
```

Remote testing architecture:

```text
Remote browser
  -> https://visionguard-systemui.vercel.app
  -> NEXT_PUBLIC_API_BASE_URL=https://<trycloudflare-url>
  -> local FastAPI backend on developer Mac
  -> local checkpoints + local SQLite archive
```

## 5. Runtime Models And Their Roles

| Module | Final runtime model | Dataset | Runtime output | Notes |
| --- | --- | --- | --- | --- |
| Eye open/closed specialist | MobileNetV2 | MRL Eye | `p_eye_closed` | Selected as primary eye model after Stage 9/9B. Checkpoint: `outputs/mrl_eye/checkpoints/best_mobilenet_v2_mrl_eye.pt`. |
| Mouth/yawn specialist | ResNet18 | YawDD/YawDD+ Dash mouth crops | `p_yawn` | Selected mouth/yawn runtime model. Checkpoint exists at `checkpoints/resnet18_best.pt` and recovered source under `artifacts/recovered_stage7_mouth_yawn/`. |
| EfficientNet-B0 / other backbones | Not final runtime defaults | Used for comparison | N/A | Useful for model comparison figures/tables only, not current runtime inference. |
| Fusion layer | Rule-based logic, not ML | Runtime timelines | alert / warning-candidate states | No trained fusion model. |

Key metric sources:

- Mouth/yawn final metrics: `artifacts/recovered_stage7_mouth_yawn/metrics_summary.json`
- Mouth/yawn refreshed report assets: `report_assets/mouth_yawn_evaluation_refresh/`
- Eye final metrics/model selection: `outputs/mrl_eye/results/mrl_eye_initial_results.csv`, `outputs/mrl_eye/results/mrl_eye_stage9b_model_selection.json`
- Learning overview: `docs/tech_learning/PROJECT_LEARNING_GUIDE_ZH.md`
- Data preprocessing details: `docs/tech_learning/DATA_PREPROCESSING_TECHNICAL_GUIDE_ZH.md`

Important training-setting nuance: the completed Stage 7 Colab run used `DEFAULT_EPOCHS = 8` and `DEFAULT_PATIENCE = 2` in `colab_file/stage7_yawdd_training_r.ipynb`; some reusable local script/docs defaults mention `12 / 3`. Do not mix actual completed-run settings with later/default script settings without explaining the distinction.

## 6. Data And Artifact Map

| Path | Meaning |
| --- | --- |
| `dataset/` | Local raw/reconstructed datasets. Large local data; normally ignored by Git. |
| `dataset/YawDD_raw/` | Original YawDD Dash videos. |
| `dataset/YawDD+/` | YawDD+ annotation files. |
| `dataset/YawDD_plus_reconstructed/` | Reconstructed labelled frames and generated mouth crops. |
| `dataset/mrlEyes_2018_01/` | MRL Eye image dataset. |
| `artifacts/mappings/` | CSV manifests for YawDD/MRL/NTHUDDD2 preprocessing. |
| `artifacts/splits/` | Subject-level train/val/test split CSVs. |
| `artifacts/recovered_stage7_mouth_yawn/` | Recovered final Stage 7 mouth/yawn checkpoint and metrics. |
| `outputs/mrl_eye/` | MRL Eye Stage 9/9B checkpoints, metrics, sweeps, and reports. |
| `outputs/system_video_upload_runs/` | Per-upload backend analysis artifacts, reports, timelines, figures, keyframes. |
| `data/visionguard_archive.sqlite` | Local SQLite archive for compact shared summary records. Do not delete casually. |
| `report_assets/all_figures/` | Final report figures and generated assets. |
| `docs/final/` | Generated final report Markdown/PDF material. |

Core labels:

- YawDD/YawDD+ mouth/yawn: `0 = no_yawn`, `1 = yawn`
- MRL Eye: `0 = closed`, `1 = open`
- NTHUDDD2 was explored but is not the current final runtime direction.

## 7. Backend Overview

Primary backend file: `src/backend/app.py`

Important endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /` | Redirects to static upload test page. |
| `POST /api/analyze-video` | Upload video analysis. Runs `src/runtime/system_video_upload_pipeline.py`. |
| `GET /api/realtime/health` | Health for realtime model service. |
| `POST /api/realtime/session/start` | Starts in-memory realtime session. |
| `POST /api/realtime/frame` | Receives JPEG frame, runs realtime inference, updates temporal state. |
| `POST /api/realtime/session/stop` | Stops/freeze realtime session state. |
| `GET /api/runs/{session_id}/summary` | Upload run summary JSON. |
| `GET /api/runs/{session_id}/timeline` | Upload run timeline CSV. |
| `GET /api/runs/{session_id}/keyframes` | Upload run keyframe metadata. |
| `GET /api/runs/{session_id}/files/{relative_path}` | Safe artifact serving for figures/reports/keyframes. |
| `GET /api/archive/health` | Local SQLite archive health. |
| `GET /api/archive/records` | List compact archive records by range/source/type. |
| `POST /api/archive/live-event` | Save compact Live Monitor alert event. |
| `POST /api/archive/live-session` | Save compact Live Monitor drive/session summary. |
| `POST /api/archive/video-run` | Save compact video upload run summary. |
| `PATCH /api/archive/records/{record_id}/review` | Update review metadata. |
| `GET /api/archive/export` | Export archive JSON. |

Archive implementation: `src/backend/local_archive.py`

Archive safety rules:

- Allows compact text/metadata/evidence summaries only.
- Rejects suspicious keys or values related to `base64`, `blob`, `raw_frame`, `raw_image`, `raw_video`, bytes, or large payloads.
- Default DB path: `data/visionguard_archive.sqlite`
- Env override: `VISIONGUARD_ARCHIVE_DB_PATH`
- Optional lightweight write token: `VISIONGUARD_ARCHIVE_WRITE_TOKEN`

## 8. Runtime Pipeline Files

| File | Role |
| --- | --- |
| `src/runtime/realtime_frame_inference.py` | Singleton realtime inference service for Live Monitor frames. Loads eye/mouth models and MediaPipe, returns per-frame evidence. |
| `src/runtime/realtime_temporal_state.py` | Session-local realtime temporal state and fusion for Live Monitor. |
| `src/runtime/system_video_upload_pipeline.py` | Full video upload pipeline: frame sampling, eye branch, mouth branch, rule-based fusion, intervals, figures, keyframes, report. |
| `src/runtime/keyframe_extractor.py` | Extracts evidence keyframes for upload analysis. |
| `src/runtime/stage10_eye_roi_consistency.py` through `stage15_real_mouth_eye_fusion_validation.py` | Research/runtime validation scripts and evidence generation. |

Avoid changing these during frontend/doc tasks unless the user explicitly asks for backend/model/runtime behavior changes.

## 9. Frontend Overview

Frontend root: `SystemUI/`

Framework:

- Next.js App Router
- React
- TypeScript
- Tailwind CSS
- lucide-react icons
- Recharts for charts

Routes:

| File | Route |
| --- | --- |
| `SystemUI/src/app/page.tsx` | `/` |
| `SystemUI/src/app/video-upload/page.tsx` | `/video-upload` |
| `SystemUI/src/app/history-48h/page.tsx` | `/history-48h` |
| `SystemUI/src/app/insights/page.tsx` | `/insights` |

App shell:

- `SystemUI/src/components/dashboard/AppShell.tsx`
- `SystemUI/src/components/dashboard/Sidebar.tsx`
- `SystemUI/src/components/dashboard/TopBar.tsx`
- `SystemUI/src/components/dashboard/UserProfileMenu.tsx`
- Providers: auth, theme, settings, notifications.

Current sidebar labels:

- `Live Monitor`
- `Video Upload Analysis`
- `History`
- `Insights`

Note: `History` still uses `/history-48h`. Do not rename the route unless explicitly requested.

## 10. Frontend Storage Keys

| Key | File | Meaning |
| --- | --- | --- |
| `visionguard.auth.v1` | `SystemUI/src/lib/authStore.tsx` | Local MVP auth/session state. |
| `visionguard.settings.v1` | `SystemUI/src/lib/settingsStore.tsx` | Minimal Live Monitor Mode setting. |
| `visionguard.notifications.v1` | `SystemUI/src/lib/notificationStore.tsx` | Top-right local Notification Center state. |
| `visionguard.liveMonitorDashboard.v1` | `SystemUI/src/lib/liveMonitorDashboardStore.ts` | Live Monitor events/risk points. |
| `visionguard.history48h.v1` | `SystemUI/src/lib/history48hStorage.ts` | History page local event/session records. Keep this internal key despite page label `History`. |
| `visionguard.videoUpload.backendUrl` | `SystemUI/src/components/video-upload/VideoUploadAnalysis.tsx` | Video Upload backend URL preference. |

Do not clear or migrate these keys unless the user explicitly asks and the data-loss implications are clear.

## 11. History And Insights Data Behavior

This is a frequent source of confusion.

- History page label: `History`
- History route: `/history-48h`
- Default time window: `Last 48 hours`
- Current History and Insights data source: Live Monitor records only.
- Video Upload Analysis may save `video_run` records to backend archive, but History/Insights currently filter to `source = live_monitor`, so uploaded video analyses do not enter their alert statistics.
- History merges backend archive records with frontend-local records and de-duplicates by ingestion/session keys.
- History Recent Drives are drive/session selectors. Live Monitor sessions should be recorded when the camera monitoring session starts, even if there are zero alerts.
- Recent Drives shows latest 3 by default and has a `Show all X drives` toggle when more exist.
- Insights uses the selected 48h Live Monitor scope and produces a user-facing report via `insightsExportUtils.ts`.

Relevant files:

- `SystemUI/src/components/history-48h/History48hPage.tsx`
- `SystemUI/src/components/history-48h/RecentSessionsSummary.tsx`
- `SystemUI/src/lib/history48hStorage.ts`
- `SystemUI/src/lib/liveMonitorHistoryIngestion.ts`
- `SystemUI/src/lib/history48hExportUtils.ts`
- `SystemUI/src/components/insights/InsightsPage.tsx`
- `SystemUI/src/lib/insightsUtils.ts`
- `SystemUI/src/lib/insightsExportUtils.ts`

## 12. Video Upload Behavior

Video Upload Analysis is evidence-oriented and backend-connected.

Current behavior:

- User selects video file.
- Frontend calls `POST /api/analyze-video`.
- Backend writes run artifacts under `outputs/system_video_upload_runs/{session_id}/`.
- UI shows analysis summary, alert intervals, backend-generated evidence figure images, keyframes, and collapsed technical evidence.
- `Download report` generates a user-readable HTML report on the frontend.
- Backend evidence figures are real backend artifacts; do not replace them with frontend-only charts.

Important files:

- `SystemUI/src/components/video-upload/VideoUploadAnalysis.tsx`
- `SystemUI/src/components/video-upload/EvidenceFigures.tsx`
- `SystemUI/src/components/video-upload/IntervalReviewTable.tsx`
- `SystemUI/src/components/video-upload/KeyframeEvidenceGallery.tsx`
- `SystemUI/src/components/video-upload/TechnicalEvidencePanel.tsx`
- `SystemUI/src/lib/videoUploadUtils.ts`
- `SystemUI/src/lib/videoUploadTypes.ts`

## 13. Local MVP Account, Theme, Notifications, Settings

This project has a local MVP account layer, not production auth.

- Auth store: `SystemUI/src/lib/authStore.tsx`
- One local configured user: `John_Coffey`
- Do not add registration, password changes, production auth, billing, email verification, or cloud sync unless explicitly requested.
- Theme is controlled separately by `ThemeToggle`; do not add day/night theme settings inside Settings.
- Notification Center state is local-only and should not affect Live Monitor overlays, sound alerts, risk gauge, or realtime warning logic.
- Minimal Live Monitor Mode only changes display layout; it must not disable webcam sampling, backend realtime calls, sound alerts, overlays, or critical warning acknowledgement.

## 14. Development And Validation Commands

Backend/local frontend launcher:

```bash
cd /Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection
make stage17-ui
```

Direct backend:

```bash
VISIONGUARD_ALLOWED_ORIGINS="https://visionguard-systemui.vercel.app,http://localhost:3000,http://127.0.0.1:3000" \
.venv-stage10/bin/python src/backend/app.py --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd SystemUI
npm run dev
npm run lint
npm run build
```

Python tests:

```bash
pytest
```

Deployment preflight:

```bash
bash scripts/deployment_preflight.sh
```

Before committing or editing broadly:

```bash
git status --short
```

The working tree may already contain unrelated generated artifacts or user changes. Do not revert files you did not intentionally change.

## 15. Deployment Notes

Current external-access setup:

- Frontend is deployed from `SystemUI/` to Vercel.
- Production URL: `https://visionguard-systemui.vercel.app`
- Backend remains local on the developer Mac.
- Cloudflare Quick Tunnel forwards public HTTPS to local `http://localhost:8000`.
- Vercel env var `NEXT_PUBLIC_API_BASE_URL` must point to the current tunnel URL.
- Backend CORS env var `VISIONGUARD_ALLOWED_ORIGINS` must include the Vercel frontend origin.
- Quick Tunnel URLs change; update Vercel and redeploy when they change.

Reference:

- `docs/DEPLOYMENT_RUNBOOK.md`
- `docs/DAILY_STARTUP_CHECKLIST.md`
- `docs/TUNNEL_DIAGNOSTIC_REPORT.md`

## 16. Best Starting Points For Future AI Work

For project understanding:

1. `docs/AI_PROJECT_CONTEXT.md`
2. `docs/PROJECT_CURRENT_STATUS.md`
3. `docs/PROJECT_STRUCTURE.md`
4. `docs/tech_learning/PROJECT_LEARNING_GUIDE_ZH.md`
5. `docs/tech_learning/DATA_PREPROCESSING_TECHNICAL_GUIDE_ZH.md`

For frontend changes:

1. `SystemUI/src/components/dashboard/AppShell.tsx`
2. `SystemUI/src/components/dashboard/Sidebar.tsx`
3. Relevant route component under `SystemUI/src/components/`
4. Relevant helper under `SystemUI/src/lib/`

For backend/API changes:

1. `src/backend/app.py`
2. `src/backend/local_archive.py`
3. Relevant runtime file under `src/runtime/`
4. `docs/DEPLOYMENT_RUNBOOK.md` if remote access is involved

For model/data/report work:

1. `docs/tech_learning/`
2. `docs/PROJECT_CURRENT_STATUS.md`
3. `reports/`
4. `artifacts/recovered_stage7_mouth_yawn/`
5. `outputs/mrl_eye/`
6. `report_assets/all_figures/`

## 17. Common Mistakes To Avoid

- Treating `History` page name as a reason to rename `/history-48h`; keep route unless explicitly requested.
- Letting Video Upload records enter History/Insights statistics; these pages currently summarize Live Monitor only.
- Clearing localStorage or SQLite archive while trying to fix UI counts.
- Confusing specialist model accuracy with full system-level drowsiness accuracy.
- Confusing EfficientNet-B0 comparison results with runtime model selection.
- Replacing backend-generated upload evidence figures with frontend charts.
- Changing realtime thresholds, debounce/cooldown, or sound behavior during UI cleanup.
- Committing local SQLite DBs, raw datasets, checkpoints, uploaded videos, or generated large media unless explicitly curated.
- Trusting stale `artifacts/results/initial_results.csv` for final Stage 7 metrics. Use recovered Stage 7 artifacts and completed Colab output instead.

## 18. Short Prompt For A New AI Window

```text
You are working on VisionGuard. First read docs/AI_PROJECT_CONTEXT.md, then inspect the relevant files before editing. Keep changes targeted. Do not delete or migrate history/archive data. Do not change model inference, temporal logic, thresholds, sound alerts, upload pipeline semantics, keyframe extraction, backend APIs, or archive schema unless explicitly asked. History is the visible page label, but the route remains /history-48h. History and Insights currently summarize Live Monitor records only.
```
