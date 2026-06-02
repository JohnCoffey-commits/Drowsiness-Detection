# Testing, Validation, and Troubleshooting Guide

## 1. Purpose of This Document

This guide explains how to validate that VisionGuard components work and how to troubleshoot issues. It focuses on operational validation and debugging, not model accuracy evaluation.

Core principle: a page opening, a video uploading, or an alert appearing does not prove final drowsiness accuracy.

## 2. What Testing Means in This Project

VisionGuard testing has several layers:

| Test layer | What it validates | What it does not prove |
|---|---|---|
| Data/preprocessing checks | Manifests, splits, crops, and sample counts exist and look reasonable | Does not prove runtime correctness |
| Model artifact checks | Checkpoints and metrics files exist | Does not prove full-system accuracy |
| Backend API checks | FastAPI endpoints are reachable | Does not prove model decisions are correct |
| Realtime runtime checks | Live Monitor frame flow works | Does not prove warnings are ground truth |
| Video Upload checks | Upload pipeline produces summaries/timelines/figures/keyframes | Does not prove real fatigue labels |
| Frontend UI checks | Pages, buttons, exports, and state are usable | Does not prove algorithm accuracy |
| Archive/History/Insights checks | Records are saved and summarized | Does not prove evaluation metrics |
| Report evidence checks | Figures/tables are traceable to sources | Does not prove experiments that were not run |

Source: `docs/PROJECT_CURRENT_STATUS.md`, `docs/AI_PROJECT_CONTEXT.md`

## 3. Read-Only Safety Before Testing

Before troubleshooting, run:

```bash
git status --short
```

Safety principles:

- do not casually delete localStorage or the SQLite archive;
- do not delete `data/visionguard_archive.sqlite` to clear UI counts;
- do not overwrite `outputs/`, `artifacts/`, or checkpoints;
- do not adjust thresholds while troubleshooting;
- do not treat retraining as normal debugging;
- confirm URL, CORS, backend process, and checkpoint paths before assuming a code issue.

## 4. Data and Artifact Checks

Common read-only checks:

| Target | Path |
|---|---|
| MRL Eye trainable manifest | `artifacts/mappings/mrl_eye_trainable_with_split.csv` |
| YawDD/YAWDD+ mouth crops manifest | `artifacts/mappings/yawdd_dash_all_mouth_crops_trainable.csv` |
| recovered Stage 7 mouth/yawn artifacts | `artifacts/recovered_stage7_mouth_yawn/` |
| mouth/yawn final refresh metrics | `report_assets/mouth_yawn_evaluation_refresh/` |
| MRL Eye results | `outputs/mrl_eye/results/` |
| MRL Eye figures | `outputs/mrl_eye/figures/` |
| eye checkpoint | `outputs/mrl_eye/checkpoints/best_mobilenet_v2_mrl_eye.pt` |
| mouth/yawn checkpoint | `checkpoints/resnet18_best.pt` |

Source: `docs/PROJECT_STRUCTURE.md`, `docs/tech_learning/MODEL_EVALUATION_AND_SELECTION_GUIDE_ZH.md`

If a file is missing, do not immediately retrain. First confirm the correct project root, whether external artifacts were synced, and whether the path is correct.

## 5. Backend Health Checks

Confirmed backend endpoints:

| Endpoint | Purpose |
|---|---|
| `/` | backend root/status response |
| `/api/realtime/health` | realtime inference service health |
| `/api/archive/health` | local archive health |
| `/api/archive/records` | archive records list |
| `/api/analyze-video` | video upload analysis |

Source: `src/backend/app.py`

Safe check examples:

```bash
curl -fsS http://127.0.0.1:8000/api/realtime/health
curl -fsS http://127.0.0.1:8000/api/archive/health
```

These commands validate service reachability, not model accuracy.

## 6. Realtime Live Monitor Validation

Live Monitor validation should check:

1. browser camera permission;
2. `/api/realtime/session/start` success;
3. repeated `/api/realtime/frame` responses;
4. response includes frame evidence and temporal state;
5. UI risk display updates;
6. warning overlay appears according to state;
7. sound/critical acknowledgement follows UI logic;
8. `/api/realtime/session/stop` succeeds after stopping the camera;
9. stable events enter notification/history ingestion;
10. History page shows Live Monitor records.

Source: `SystemUI/src/components/dashboard/LiveVideoCard.tsx`, `SystemUI/src/components/dashboard/LiveMonitorPage.tsx`, `src/backend/app.py`

Boundary: seeing a warning overlay only means a runtime rule produced a warning-candidate. It does not mean the frame or video segment is labelled ground-truth fatigue.

## 7. Video Upload Validation

Video Upload validation should check:

1. file selection succeeds;
2. `/api/analyze-video` returns a response;
3. `outputs/system_video_upload_runs/{session_id}/` is created;
4. `summary.json` is readable;
5. `timeline.csv` / `fusion_timeline.csv` exists;
6. Alert Intervals display;
7. Evidence Timeline figures load;
8. keyframe metadata and thumbnails exist;
9. `Download report` generates HTML;
10. Technical Details links to backend artifacts.

Source: `SystemUI/src/components/video-upload/VideoUploadAnalysis.tsx`, `SystemUI/src/components/video-upload/EvidenceFigures.tsx`, `src/runtime/system_video_upload_pipeline.py`, `src/runtime/keyframe_extractor.py`

Boundary: successful upload analysis is pipeline validation, not labelled drowsiness accuracy validation.

## 8. History and Insights Validation

History/Insights validation should check:

- Live Monitor events are written to `visionguard.history48h.v1`;
- backend archive contains source=`live_monitor` records;
- History default time window is 48h;
- Recent Drives show sessions/drives;
- local/archive merge deduplicates records;
- Insights is generated from Live Monitor scope;
- Video Upload results should not be assumed to enter History/Insights Live Monitor statistics.

Source: `SystemUI/src/components/history-48h/History48hPage.tsx`, `SystemUI/src/components/insights/InsightsPage.tsx`, `SystemUI/src/lib/history48hStorage.ts`, `SystemUI/src/lib/backendArchiveApi.ts`

## 9. Frontend Build and Lint Validation

Confirmed commands:

```bash
cd SystemUI
npm run lint
npm run build
```

`npm run lint` checks ESLint rules. `npm run build` checks the Next.js production build. They can catch TypeScript/React/build issues, but they do not prove backend availability or model accuracy.

Source: `SystemUI/package.json`

## 10. Deployment Validation

Remote testing validation should check:

- local backend `/api/realtime/health` is reachable;
- local backend `/api/archive/health` is reachable;
- Cloudflare tunnel URL health endpoints are reachable;
- Vercel `NEXT_PUBLIC_API_BASE_URL` points to the current tunnel;
- backend CORS includes the exact Vercel origin;
- frontend was redeployed after Vercel env changes;
- `scripts/deployment_preflight.sh` passes.

Source: `docs/DEPLOYMENT_RUNBOOK.md`, `scripts/deployment_preflight.sh`

## 11. Troubleshooting Matrix

| Symptom | Likely Cause | Where to Check | Safe Fix | What Not To Do |
|---|---|---|---|---|
| CORS error | Allowed origins do not include frontend origin | browser console, `VISIONGUARD_ALLOWED_ORIGINS` | Update env and restart backend | Do not change model code |
| Backend unreachable | Backend process stopped or URL wrong | terminal, `/api/realtime/health` | Start backend or correct URL | Do not clear history |
| Upload fails | File too large, backend stopped, pipeline error | backend logs, run folder | Check logs and artifacts | Do not retrain models |
| No face detected | Camera angle/light/face visibility issue | Live Monitor UI, backend response | Adjust camera and lighting | Do not treat no-face as safe |
| No keyframes | No warning intervals or artifact missing | run folder, keyframes endpoint | Check summary/intervals | Do not fabricate keyframes |
| Evidence figures missing | Figure artifact path missing or backend URL wrong | run `figures/`, network tab | Check artifact URL | Do not replace with frontend-only charts |
| History empty | No Live Monitor stable event or archive unavailable | localStorage, archive health | Generate Live Monitor event first | Do not delete DB |
| Insights empty | No Live Monitor records | History page, archive records | Confirm data scope first | Do not mix in upload records |
| Archive write rejected | Unsafe payload or token mismatch | backend response, archive code | Check payload/token | Do not store raw frame/base64 |
| Build fails | TS/ESLint/Next issue | build output | Fix frontend code | Do not deploy while ignoring errors |
| Checkpoint missing | File not synced | checkpoint paths | Restore checkpoint | Do not casually retrain |

## 12. Validation Boundary

Keep these distinctions clear:

- health check passed ≠ model accuracy;
- upload analysis passed ≠ ground-truth drowsiness detection;
- frontend warning displayed ≠ manually labelled alert;
- History/Insights charts ≠ model evaluation metrics;
- specialist model metrics ≠ full-system accuracy;
- deployment success ≠ safety certification.

## 13. Beginner Checklist

- Did I check `git status --short` first?
- Did I confirm the backend URL is current?
- Did I confirm exact CORS origin matching?
- Did I confirm checkpoints exist?
- Can I distinguish localStorage from the SQLite archive?
- Can I explain that upload figures are runtime evidence?
- Did I avoid writing demo success as accuracy?

## 14. Common Mistakes

- Deleting archive/localStorage to fix UI counts.
- Adjusting thresholds to make demos look better.
- Retraining models when the backend URL is wrong.
- Treating a working demo as an accuracy evaluation.
- Reporting issues without source paths.
- Mixing upload artifacts with History analytics.
- Updating Vercel before tunnel health passes.
