# Project Structure Guide

Last reviewed: 2026-05-26

## 1. Purpose of This Document

This document explains how the repository is organized for the modular driver drowsiness project. It is intended as a living guide for teammates who need to find datasets, preprocessing outputs, training scripts, reports, model outputs, and current analysis artifacts quickly.

The repository currently supports a modular driver monitoring system rather than one monolithic drowsiness classifier:

- YawDD/YawDD+ Dash mouth/yawn specialist -> `p_yawn`
- MRL Eye open/closed specialist -> `p_eye_closed`
- Runtime video analysis and rule-based fusion -> warning-candidate states
- FastAPI + Next.js Stage 17 video-upload analysis workstation
- FastAPI + Next.js Stage 19 Live Monitor realtime webcam warning-candidate prototype with product-style visual warning overlays, default-on sound alerts after camera start, a realtime-bound warning-candidate risk gauge, dashboard widgets bound to stable Live Monitor events/score samples, and a product-style session-window state-derived display severity waveform chart
- Stage 20A local MVP account/app-shell foundation, Stage 20B frontend-local Live Monitor history ingestion for `/history-48h`, and Stage 21A split between History review and Insights analytics
- Stage 22 local backend SQLite archive for compact shared Live Monitor and uploaded-video summary records
- May 19 UI polish for Live Monitor desktop column balance, stronger selected sidebar contrast, collapsed sidebar control alignment, and a cleaner `/history-48h` header
- May 26 Settings popover and Minimal Live Monitor Mode, which hides raw webcam preview and secondary dashboard panels while keeping webcam sampling, backend realtime calls, warning overlays, sound alerts, critical-eye acknowledgement, and the Drowsiness Risk gauge active
- External-access deployment validation for Vercel-hosted `SystemUI/` calling a Cloudflare Tunnel HTTPS URL that forwards to the developer's local FastAPI backend

## 2. High-Level Project Architecture

| Module | Dataset source | Specialist task | Output concept | Current status |
| --- | --- | --- | --- | --- |
| Mouth/yawn module | Original YawDD Dash videos plus YawDD+ annotation files | Binary mouth/yawn classification | `p_yawn` | Stage 7 completed |
| Eye open/closed module | MRL Eye | Binary eye-state classification | `p_eye_closed` | Stage 9 and Stage 9B completed |
| Runtime temporal analysis | Controlled A/B/C/D videos and upload videos | Eye/mouth ROI extraction, timeline generation, rule-based fusion | warning-candidate timelines | Stage 10-15 completed as controlled-validation prototype |
| Video Upload Analysis MVP | Uploaded local videos through FastAPI + SystemUI | Professional warning-candidate review page | summary, intervals, figures, keyframes, technical files | Stage 17.5 evidence-review UI polished |
| Live Monitor realtime prototype | Browser webcam sampled frames through SystemUI + FastAPI | Single-frame evidence, session-local temporal warning-candidate state, product-style visual warning overlays, face visibility cue, default-on sound alerts after camera start, realtime-bound warning-candidate risk gauge, event/score-bound dashboard widgets, state-derived session-window waveform chart rendering, frontend-local 48h history ingestion from stable events, and optional Minimal Mode display | `p_eye_closed`, `p_yawn`, ROI quality, realtime candidate state, product warning overlays, stable frontend warning-candidate events, frontend warning-candidate severity score samples, local history records | Stage 19.7C local prototype plus Stage 20B frontend-local history ingestion; May 26 Minimal Mode keeps monitoring active while making the Drowsiness Risk gauge the main visible UI |
| History and Insights frontend | Browser-local history records under `visionguard.history48h.v1` plus optional local backend archive records | Event-level review workflow and aggregate current-user/shared-record analytics | Local and backend-archive warning-candidate summary records and derived summaries, not backend truth labels | Stage 21A separates `/history-48h` review workstation from `/insights`; Stage 22 adds backend_archive fallback-aware display and export |
| Local backend archive | FastAPI + SQLite under `data/visionguard_archive.sqlite` by default | Central compact summary storage for shared remote clients | Stable Live Monitor event summaries, uploaded-video summary records, review state | Stage 22 local archive; no raw webcam frames/images/videos or cloud database |
| NTHUDDD2 branch | Official NTHU considered; Kaggle extracted-frame version explored | Drowsy/not-drowsy frame classification | Not part of final module direction | Not main direction |

Stage 17 currently produces rule-based drowsiness warning-candidate analysis for uploaded videos. Stage 19.7C adds a local Live Monitor realtime webcam warning-candidate feasibility prototype polish pass with a clean product-style webcam frame, product yawn/eye/critical-eye warning overlays, face-not-visible signal-quality overlay, automatic sampling from Start Camera, default-on sound alerts after the camera user gesture, a Drowsiness Risk card bound to frontend warning-candidate severity state, dashboard widgets bound to stable frontend warning-candidate events/score samples instead of mock data, and a session-window state-derived display severity waveform chart that uses real timestamp-based X-axis positioning. The May 26 Settings popover adds a persisted Minimal Live Monitor Mode under `visionguard.settings.v1`; in that mode the raw webcam preview, recent events, charts, and extra dashboard panels are hidden, while the existing Drowsiness Risk gauge becomes the main visible UI and realtime monitoring/alerts continue. Stage 20A adds a local MVP login gate, user profile menu, manual theme toggle, and notification center. Stage 20B normalizes stable Live Monitor frontend warning-candidate events into `/history-48h` frontend-local history. Stage 21A keeps `/history-48h` focused on event filtering, details, sessions, and manual review workflow, while `/insights` provides read-only aggregate analytics from the same current-user local history records. Stage 22 adds local SQLite summary archive storage. Current reported accuracies are specialist-module results, not final system-level driver drowsiness accuracy. The project does not currently include browser notification, production authentication, cloud database storage, raw image/video storage, a hosted production backend, or final system-level performance claims.

## 3. Repository Layout

Important top-level locations:

```text
Drowsiness_Detection/
  artifacts/
  checkpoints/
  colab_file/
  dataset/
  data/
  docs/
  outputs/
  report_assets/
  reports/
  scripts/
  src/
  SystemUI/
  tests/
  upload_test/
  .gitignore
  Makefile
  README_*.md
  requirements.txt
```

| Path | Purpose |
| --- | --- |
| `dataset/` | Raw or locally reconstructed datasets. This is large local data and is ignored by Git. |
| `data/` | Local runtime data such as the Stage 22 SQLite archive database. SQLite files are ignored by Git. |
| `artifacts/` | Preprocessing outputs, manifests, split files, visual checks, and intermediate results. |
| `reports/` | Human-readable reports for dataset inspection, preprocessing, split validation, training summaries, and model selection. |
| `src/` | Python source code for dataset preparation, preprocessing, training, and runtime checks. |
| `src/backend/` | FastAPI backend for Stage 17 upload analysis, safe artifact serving, Stage 19 realtime frame evidence endpoints, and Stage 22 local archive endpoints. |
| `SystemUI/` | Independent Next.js App Router frontend for Live Monitor, Stage 17 video-upload analysis, History, Insights, and local app-shell workflows. |
| `scripts/` | Local helper scripts, including the Stage 17 one-command launcher and deployment preflight. |
| `upload_test/` | Local short videos for upload UI/backend validation. |
| `colab_file/` | Google Colab notebooks used for GPU training and Colab workflows. |
| `outputs/` | Synced final training outputs and runtime evidence outputs, including MRL Eye Stage 9/9B and Stage 10-17 runtime/upload evidence. |
| `checkpoints/` | Legacy or local model checkpoint location. Large checkpoint files should not be committed to normal Git. |
| `docs/` | GitHub-friendly project structure and current-status documentation. |
| `docs/final/` | Generated final-report Markdown/PDF material and rendered page checks. |
| `report_assets/` | Figure assets used for report generation and final-report composition. |
| `tests/` | Lightweight regression tests for selected Python logic, currently including Stage 17.5 eye-evidence calibration behavior. |

## 4. Dataset Folders

`dataset/` is for local raw or reconstructed data. It should generally not be committed.

Observed local dataset folders:

| Path | Purpose |
| --- | --- |
| `dataset/YawDD_raw/` | Original YawDD Dash video data. Used as the video source for reconstructed Dash frames. |
| `dataset/YawDD+/` | YawDD+ annotation files. These provide frame indices and class labels. |
| `dataset/YawDD_plus_reconstructed/` | Reconstructed Dash full frames and generated mouth crops derived from YawDD raw videos plus YawDD+ annotations. |
| `dataset/mrlEyes_2018_01/` | MRL Eye subject folders and annotation/stat files. Used for the eye open/closed module. |
| `dataset/NTHUDDD2/` | Kaggle extracted-frame NTHUDDD2 exploration data, not the current main system direction. |

Dataset notes:

- YawDD/YawDD+ uses `0 = no_yawn`, `1 = yawn`.
- MRL Eye uses `0 = closed`, `1 = open`.
- NTHUDDD2 Kaggle uses `notdrowsy = 0`, `drowsy = 1`, but this branch is not the current main module direction.

## 5. Artifacts and Manifests

`artifacts/` stores reproducible intermediate files from inspection, reconstruction, preprocessing, and splitting.

Important subfolders:

| Path | Purpose |
| --- | --- |
| `artifacts/mappings/` | CSV manifests produced during preprocessing and dataset preparation. |
| `artifacts/splits/` | Subject-level train/validation/test split files. |
| `artifacts/visual_checks/` | Contact sheets and visual sanity-check images. |
| `artifacts/results/` | Earlier baseline result outputs. Note that the local YawDD `initial_results.csv` currently appears stale and should not be treated as the final Stage 7 source. |
| `artifacts/preprocessed/` | Regenerable preprocessing outputs. Ignored by Git. |
| `artifacts/cache/` | Local cache files. Ignored by Git. |
| `artifacts/models/` | Local model artifacts. Ignored by Git. |

Key manifest examples:

| File | Meaning |
| --- | --- |
| `artifacts/mappings/yawdd_dash_all_labeled_frames.csv` | Reconstructed YawDD+ Dash labeled-frame manifest. |
| `artifacts/mappings/yawdd_dash_all_mouth_crops.csv` | All attempted YawDD mouth-crop rows, including failures. |
| `artifacts/mappings/yawdd_dash_all_mouth_crops_trainable.csv` | Trainable YawDD mouth-crop rows with split labels. |
| `artifacts/splits/yawdd_dash_subject_split.csv` | Leakage-safe subject-level YawDD split. |
| `artifacts/mappings/mrl_eye_all_images.csv` | Full MRL Eye image manifest. |
| `artifacts/mappings/mrl_eye_trainable.csv` | Trainable MRL Eye rows. |
| `artifacts/mappings/mrl_eye_trainable_with_split.csv` | MRL Eye trainable manifest with subject-level split labels. |
| `artifacts/splits/mrl_eye_subject_split.csv` | Subject-level MRL Eye split. |
| `artifacts/mappings/nthuddd2_kaggle_all_images*.csv` | Kaggle NTHUDDD2 exploration manifests. |

## 6. Reports

`reports/` contains human-readable Markdown and CSV summaries. These are useful for GitHub because they explain the decisions and checks behind the data pipeline.

Important reports:

| File | Purpose |
| --- | --- |
| `reports/yawdd_raw_dash_report.md` | Inspection of original YawDD Dash videos. |
| `reports/yawdd_plus_annotation_format_report.md` | Interpretation of YawDD+ annotation format and class IDs. |
| `reports/yawdd_dash_reconstruction_report.md` | Dash frame reconstruction summary. |
| `reports/yawdd_dash_visual_sanity_check.md` | Visual confirmation that YawDD+ class `1` corresponds to yawning and class `0` to non-yawning. |
| `reports/yawdd_dash_mouth_crop_report.md` | MediaPipe mouth-crop preprocessing summary and readiness result. |
| `reports/yawdd_dash_split_report.md` | YawDD subject-level split report and leakage checks. |
| `reports/mrl_eye_dataset_report.md` | MRL Eye Stage 8 inspection report. |
| `reports/mrl_eye_split_report.md` | MRL Eye subject-level split report. |
| `reports/mrl_eye_stage9_training_plan.md` | Stage 9 MRL Eye training design. |
| `reports/mrl_eye_stage9b_error_analysis.md` | Final Stage 9B MRL Eye model-selection report. |
| `reports/stage10_runtime_eye_roi_acceptance_report.md` | Stage 10 controlled-video runtime eye ROI acceptance report. |
| `reports/stage13_mouth_eye_fusion_design_report.md` | Stage 13 rule-based mouth-eye fusion design/prototype report. |
| `reports/stage14_mouth_yawn_runtime_validation_report.md` | Stage 14 runtime mouth/yawn validation report using the recovered Stage 7 checkpoint. |
| `reports/stage15_real_mouth_eye_fusion_validation_report.md` | Stage 15 real synchronized rule-based mouth-eye fusion validation report. |
| `docs/archive/stage16/reports/stage16_final_integration_summary_report.md` | Stage 16 final integration package summary and conservative claim boundary. |
| `reports/stage17_video_upload_detection_mvp_report.md` | Stage 17 video-upload backend/pipeline MVP report. |
| `reports/stage17_2_manual_review_interpretation_report.md` | Stage 17.2 manual interpretation report for safe warning-candidate wording. |
| `reports/stage17_4_video_upload_mvp_stabilization_report.md` | Stage 17.4 stabilization report covering launcher, acceptance, demo, and current limitations. |
| `reports/stage17_5_eye_evidence_calibration_report.md` | Stage 17.5 eye-evidence calibration report for weak/moderate/strong evidence wording and strength-gate behavior. |
| `reports/nthuddd2_kaggle_dataset_report.md` | Kaggle NTHUDDD2 exploration report and limitations. |
| `docs/archive/reports/nthu_dataset_report.md` | Earlier NTHUDDD2 inspection notes. |

## 7. Source Code

`src/` is organized by function.

### `src/data/`

Dataset inspection, manifest-building, splitting, validation, and spot-check scripts.

Important examples:

| File | Purpose |
| --- | --- |
| `src/data/inspect_mrl_eye.py` | Inspect raw MRL Eye files and produce dataset report. |
| `src/data/build_mrl_eye_manifest.py` | Build MRL Eye all-image and trainable manifests. |
| `src/data/split_mrl_eye_subjects.py` | Create leakage-safe subject-level MRL Eye split. |
| `src/data/spotcheck_mrl_eye.py` | Generate visual MRL Eye contact sheets. |
| `src/data/build_yawdd_dash_mapping.py` | Build YawDD/YawDD+ Dash frame mapping. |
| `src/data/extract_yawdd_dash_labeled_frames.py` | Reconstruct labeled Dash frames from original videos. |
| `src/data/build_yawdd_split.py` | Build YawDD subject-level split. |
| `src/data/build_nthuddd2_kaggle_manifest.py` | Build Kaggle NTHUDDD2 exploration manifest. |
| `src/data/split_nthuddd2_kaggle_subject.py` | Build Kaggle NTHUDDD2 subject split and LOSO folds. |

### `src/preprocessing/`

Mouth ROI crop generation for the YawDD mouth/yawn module.

| File | Purpose |
| --- | --- |
| `src/preprocessing/generate_yawdd_mouth_crops.py` | Generate mouth crops using MediaPipe Face Mesh landmarks. |
| `src/preprocessing/precompute_yawdd_mouth_crops.py` | Earlier mouth-crop preprocessing entrypoint. |

### `src/training/`

Training scripts for CNN baselines.

| File | Purpose |
| --- | --- |
| `src/training/train_classifier.py` | YawDD mouth/yawn classifier training helper. |
| `src/training/run_initial_baselines.py` | YawDD three-model baseline runner. |
| `src/training/train_mrl_eye_baselines.py` | MRL Eye Stage 9 baseline training pipeline. |

### `src/runtime/`

Runtime scripts for safe post-training checks.

| File | Purpose |
| --- | --- |
| `src/runtime/stage10_eye_roi_consistency.py` | Stage 10 runtime eye ROI consistency test using MediaPipe FaceLandmarker and the selected MRL Eye MobileNetV2 checkpoint. |
| `src/runtime/stage11_eye_temporal_analysis.py` | Stage 11 eye-only temporal smoothing / PERCLOS-like analysis. |
| `src/runtime/stage12_eye_alert_rule_analysis.py` | Stage 12 eye-only alert rule comparison and recommendation. |
| `src/runtime/stage13_mouth_eye_fusion_design.py` | Stage 13 rule-based mouth-eye fusion design/prototype. |
| `src/runtime/stage14_mouth_yawn_runtime.py` | Stage 14 runtime mouth ROI consistency and mouth/yawn inference using the recovered Stage 7 ResNet18 checkpoint. |
| `src/runtime/stage15_real_mouth_eye_fusion_validation.py` | Stage 15 real synchronized rule-based mouth-eye fusion validation using Stage 12 eye timelines and Stage 14 model-generated `p_yawn` timelines. |
| `src/runtime/system_video_upload_pipeline.py` | Stage 17 single-video upload analysis pipeline that runs eye branch, mouth branch, F5 fusion, and keyframe extraction. |
| `src/runtime/realtime_frame_inference.py` | Stage 19 realtime single-frame webcam evidence inference service for Live Monitor. |
| `src/runtime/realtime_temporal_state.py` | Stage 19 session-local realtime temporal warning-candidate state for Live Monitor. |
| `src/runtime/keyframe_extractor.py` | Stage 17 helper for saving warning-candidate keyframe screenshots. |

### `src/backend/`

FastAPI backend for local upload analysis, safe artifact serving, and realtime frame evidence endpoints.

| File | Purpose |
| --- | --- |
| `src/backend/app.py` | Stage 17 backend entrypoint plus Stage 19 realtime endpoints. Provides `POST /api/analyze-video`, `/api/realtime/health`, `/api/realtime/session/start`, `/api/realtime/frame`, `/api/realtime/session/stop`, and safe session file URLs under `/api/runs/{session_id}/...`. |
| `src/backend/local_archive.py` | Stage 22 SQLite archive layer for compact shared Live Monitor and uploaded-video summary records. |
| `src/backend/static/upload_test.html` | Minimal standalone backend-hosted upload test page. The primary frontend is now SystemUI `/video-upload`. |

## 8. SystemUI Frontend

`SystemUI/` is an independent Next.js App Router frontend using TypeScript, Tailwind CSS, shadcn/base-ui style components, `lucide-react`, and `recharts`.

Important frontend files:

| Path | Purpose |
| --- | --- |
| `SystemUI/src/app/page.tsx` | Lightweight route entry for `/`; the persistent Live Monitor UI is owned by `AppShell` so camera/session state does not reset on normal in-app navigation. |
| `SystemUI/src/components/dashboard/LiveMonitorPage.tsx` | Persistent Live Monitor route content; stores the latest frontend warning-candidate risk state emitted by `LiveVideoCard`, records stable dashboard events/score samples, appends stable warning-candidate events to frontend-local 48h history and the Stage 22 local backend archive, skips camera-off fake stored chart points, passes real current-drive/today/current-session data into the dashboard widgets, owns the smooth right-column Recent Events overlay state, keeps the May 19 desktop layout split balanced when the sidebar is expanded, and switches to the May 26 Minimal Mode layout when enabled. |
| `SystemUI/src/components/dashboard/LiveVideoCard.tsx` | Stage 19.7A Live Monitor product UI: clean webcam frame, single Start/Stop Camera button, automatic 2 FPS sampling, product-style yawn/eye warning overlays, concise critical-eye modal, face-not-visible signal-quality overlay, default-on sound after camera start, risk-state callback emission, and stable warning-candidate dashboard event emission. In Minimal Mode, it keeps the video element/canvas, sampling, realtime session, warning overlays, sound behavior, and critical-eye modal active while hiding the raw preview. |
| `SystemUI/src/components/dashboard/DrowsinessRiskCard.tsx` | Stage 19.6B presentational risk gauge bound to the Live Monitor frontend warning-candidate severity state with smooth score and needle animation; supports a prominent Minimal Mode variant. |
| `SystemUI/src/components/dashboard/StatusMetricCard.tsx` | Stage 19.7A top Live Monitor cards showing current-drive EYE/YAWN stable warning-candidate event counts instead of mock counts. |
| `SystemUI/src/components/dashboard/RecentEventsList.tsx` | Stage 19.7A today-only Recent Events card using real stable Live Monitor events, a fixed `Today` badge, and compact/expanded rendering for the smooth right-column overlay that does not navigate to `/history-48h`. |
| `SystemUI/src/components/dashboard/DrowsinessLevelChart.tsx` | Stage 19.7C product-style session-window state-derived display severity waveform chart using current-session throttled state samples, numeric timestamp X-axis binding, display-only time labels, gray Idle baseline, Low/Medium/High segmented line colors, soft area fill, dynamic session-start-to-1-hour windowing with no artificial minimum duration, and bucket compaction instead of mock chart data. |
| `SystemUI/src/app/video-upload/page.tsx` | Route entry for `/video-upload`. |
| `SystemUI/src/components/dashboard/Sidebar.tsx` | Sidebar menu; Live Monitor, Video Upload Analysis, 48h History, and Insights navigation with Stage 20A day/night styling, high-contrast white active labels/icons, and a collapsed top control row where the logo and expand button remain aligned and uncut. |
| `SystemUI/src/components/dashboard/AppShell.tsx` | Shared dashboard app shell/layout wrapper with the Stage 20A local login gate and app-level providers. |
| `SystemUI/src/components/dashboard/ThemeToggle.tsx` | Stage 20A manual Day/Night theme toggle. |
| `SystemUI/src/components/dashboard/NotificationCenter.tsx` | Stage 20A clickable local notification center for warning-candidate, system, and review notifications. |
| `SystemUI/src/components/dashboard/UserProfileMenu.tsx` | Stage 20A current local user profile menu/logout action plus the May 26 right-top Settings popover for Minimal Live Monitor Mode. |
| `SystemUI/src/components/auth/LoginScreen.tsx` | Stage 20A local MVP login screen for the assigned local account. |
| `SystemUI/src/components/video-upload/VideoUploadAnalysis.tsx` | Main Stage 17 upload analysis workstation component; posts compact completed-analysis summaries to the Stage 22 local backend archive when available. |
| `SystemUI/src/components/video-upload/AnalysisSummaryCards.tsx` | Summary metric cards for duration, sampled frames, warning-candidate frame counts, yawn events, and suppressed escalation. |
| `SystemUI/src/components/video-upload/IntervalReviewTable.tsx` | Warning-candidate interval review table. |
| `SystemUI/src/components/video-upload/KeyframeEvidenceGallery.tsx` | Keyframe evidence gallery with timestamp, frame index, fusion state, probabilities, and reason metadata. |
| `SystemUI/src/components/video-upload/TechnicalEvidencePanel.tsx` | Collapsible technical evidence/download links. |
| `SystemUI/src/components/video-upload/InterpretationNotice.tsx` | Permanent safe interpretation warning and Stage 17 explanation text. |
| `SystemUI/src/lib/liveMonitorAlertUtils.ts` | Stage 19.5 frontend-only visual alert mapping, debounce, normal-clear, cooldown, and session-local alert event helper. |
| `SystemUI/src/lib/liveMonitorSoundUtils.ts` | Stage 19.6A Web Audio helper for Live Monitor warning sounds after the camera user gesture, including gentle one-shot normal warning cues and repeated urgent critical-eye cues until confirmation. |
| `SystemUI/src/lib/liveMonitorRiskUtils.ts` | Stage 19.6B frontend utility mapping camera/sampling, alert kind, temporal state, and face/signal quality into a product-facing warning-candidate severity score. |
| `SystemUI/src/lib/liveMonitorDashboardTypes.ts` | Stage 19.7C lightweight local dashboard event/risk point types for stable warning-candidate events and state-derived display severity samples. |
| `SystemUI/src/lib/liveMonitorDashboardStore.ts` | Stage 19.7C browser localStorage helper using `visionguard.liveMonitorDashboard.v1`; derives bounded display severity scores, validates and compacts current-session risk points, stores lightweight dashboard events and risk points only, and never stores raw images/video/frame payloads. |
| `SystemUI/src/lib/liveMonitorHistoryIngestion.ts` | Stage 20B mapper/helper that converts stable Live Monitor dashboard events into `/history-48h` frontend-local history records. |
| `SystemUI/src/lib/backendArchiveApi.ts` | Stage 22 frontend archive client for health, records, writes, review updates, export, and history-compatible mapping. |
| `SystemUI/src/lib/backendArchiveTypes.ts` | Stage 22 TypeScript types for archive health, records, writes, and export payloads. |
| `SystemUI/src/lib/archiveClientId.ts` | Stage 22 stable browser-local client identifier for distinguishing shared clients without user registration. |
| `SystemUI/src/lib/authStore.tsx` | Stage 20A fixed local MVP account/session state using `visionguard.auth.v1`; not production authentication. |
| `SystemUI/src/lib/themeStore.tsx` | Stage 20A manual theme state using `visionguard.theme.v1`. |
| `SystemUI/src/lib/notificationStore.tsx` | Stage 20A local notification state using `visionguard.notifications.v1`. |
| `SystemUI/src/lib/settingsStore.tsx` | May 26 local settings store using `visionguard.settings.v1`, currently for `liveMonitor.minimalMode`. |
| `SystemUI/src/app/history-48h/page.tsx` | Stage 18/20B/21A route entry for frontend-local 48h History. |
| `SystemUI/src/components/history-48h/` | Stage 21A event-level history/review workstation components: focused header, filters, compact summary, prioritized review queue, event timeline/details, recent sessions, and interpretation note. |
| `SystemUI/src/app/insights/page.tsx` | Stage 21A route entry for user-scoped local Insights. |
| `SystemUI/src/components/insights/` | Stage 21A read-only analytics components for summary patterns, warning-candidate trend, composition, time-of-day pattern, session comparison, signal quality insights, recommendations, and empty state. |
| `SystemUI/src/lib/insightsTypes.ts` | Stage 21A TypeScript types for local analytics summaries, trends, composition, sessions, signal-quality summaries, and recommendations. |
| `SystemUI/src/lib/insightsUtils.ts` | Stage 21A pure helper functions deriving user-scoped local Insights from existing 48h history records without a new storage dataset. |

## Deployment And Archive Docs

| File | Purpose |
| --- | --- |
| `docs/DEPLOYMENT_RUNBOOK.md` | Practical Vercel + Cloudflare Tunnel + local FastAPI deployment validation runbook. Current production frontend alias: `https://visionguard-systemui.vercel.app`. |
| `docs/DAILY_STARTUP_CHECKLIST.md` | Concise command checklist for restarting the local backend, Quick Tunnel, Vercel env, and deployment preflight after a Mac restart. |
| `docs/LOCAL_BACKEND_ARCHIVE.md` | Stage 22 local SQLite archive purpose, storage boundaries, environment variables, startup flow, export/backup, and limitations. |
| `scripts/deployment_preflight.sh` | Deployment preflight script for realtime health, archive health, optional remote tunnel health, optional CORS preflight, and optional archive write test. |
| `SystemUI/src/lib/history48hStorage.ts` | Stage 18/20B storage helpers for `visionguard.history48h.v1`, including pruning, deduping, user merge behavior, and Live Monitor append support. |
| `SystemUI/src/lib/videoUploadTypes.ts` | TypeScript types for backend response, summary, intervals, figures, and keyframes. |
| `SystemUI/src/lib/videoUploadUtils.ts` | URL, formatting, interval, figure, keyframe, and copy-summary helpers. |
| `SystemUI/src/lib/apiConfig.ts` | Shared frontend API base URL helper. Reads `NEXT_PUBLIC_API_BASE_URL`, falls back to `http://127.0.0.1:8000`, normalizes trailing slashes, and builds backend API URLs for upload and realtime client calls. |

Current Stage 17 route:

```text
http://127.0.0.1:3000/video-upload
```

Permanent wording boundary for the page:

```text
This output is a rule-based drowsiness warning-candidate analysis, not final system-level drowsiness accuracy.
```

Current Stage 19 Live Monitor route:

```text
http://127.0.0.1:3000/
```

Permanent wording boundary for Live Monitor:

```text
This output is a realtime rule-based warning-candidate analysis, not final system-level drowsiness accuracy.
```

Current Stage 19.7C Live Monitor includes webcam capture, mirrored preview for frontend display only, automatic 2 FPS frame sampling after Start Camera, single-frame backend evidence, session-local temporal warning-candidate state, a frontend visual alert debounce layer, product-style yawn/eye warning overlays, a concise critical-eye modal, a face-not-visible signal-quality overlay, a right-side Drowsiness Risk card bound to frontend warning-candidate severity state, smooth score/needle animation, current-drive EYES CLOSED/YAWN warning-candidate event counts, today-only Recent Events with a smooth right-column overlay expansion, current-session state-derived display severity waveform chart data, real timestamp-based chart X-axis positioning, dynamic session-start-to-1-hour chart windowing with no artificial minimum duration, render-only gray Idle baseline only outside active monitoring segments, 5-second risk-point bucket compaction, session-local alert events kept out of the default video frame, no default in-card diagnostic panel, and default-on sound alerts initialized by the camera user gesture. Normal warning-candidate events play gentle one-shot sound cues; the critical-eye modal repeats an urgent multi-beep cue until confirmation. The May 26 Minimal Mode keeps this realtime behavior active but hides the raw webcam preview, Recent Events, chart, and extra dashboard panels, making the existing Drowsiness Risk gauge the main visible UI. The Live Monitor component remains mounted during normal in-app navigation so the camera/session state and current-drive counts do not reset when visiting `/video-upload` or `/history-48h`. Dashboard widgets avoid raw frame counting and use stable frontend events/score samples only. Stage 20B additionally writes stable warning-candidate events to `/history-48h` frontend-local history, and Stage 22 can archive compact stable-event summaries to the local SQLite backend. May 19 layout polish reserves a comfortable right information column on desktop when the sidebar is expanded, while keeping the video/chart column visually dominant and preserving stacked behavior on narrower screens. It still does not include production authentication, cloud database storage, raw image/video storage, a hosted production backend, final system-level drowsiness accuracy, or final system-level performance claims.

Stage 20A — Local Account, Theme, and Notification Foundation adds a local MVP login gate using one assigned local username/password account, current-user profile menu, user-scoped frontend Live Monitor dashboard records and notifications where implemented, manual Day/Night theme toggle, light day sidebar, coherent night app-shell styling, white active sidebar label/icon treatment, an aligned collapsed sidebar logo/expand-control row, and a clickable notification center for warning-candidate, system, and review notifications. The top-right profile area now also hosts the Settings popover. This does not add production authentication, backend user-owned persisted history, cloud database-backed history, model changes, final system-level claims, or a hosted production backend; true multi-user isolation requires backend persistence and server-side ownership checks in a future stage.

Stage 20B — Live Monitor Local History Ingestion normalizes stable Live Monitor frontend warning-candidate events into `/history-48h` frontend-local records under `visionguard.history48h.v1`, scoped to the current local MVP user where Stage 20A auth is available. It uses stable debounced/cooldown events only, not raw frames, raw probabilities, pending debounce states, display-only reminders, or chart display points. It stores no raw image/video/blob payloads. Stage 22 separately adds local SQLite summary archive storage; Stage 20B itself does not add production authentication, backend user-owned persisted history, browser notification API, model changes, final system-level claims, or a hosted production backend.

Stage 21A — History / Insights Separation and User-Scoped Local Insights Dashboard makes `/history-48h` the event-level frontend-local warning-candidate history and review workstation, with filtering, details, sessions, and manual review workflow. `/insights` is now the user-scoped aggregate analytics page derived from the same local records, including trend, composition, time-of-day pattern, session comparison, signal-quality insights, and review recommendations. The two pages intentionally avoid duplicating the same charts and event lists. May 19 History polish removes the extra top explanatory notice block from `/history-48h` while keeping safe interpretation wording in the event/review surfaces. This remains warning-candidate review and analytics only, using backend archive records when available and frontend-local fallback otherwise; it does not add production authentication, model changes, raw image/video storage, cloud database storage, a hosted production backend, or final system-level claims.

Stage 19.7C — Product-Style Drowsiness Level Waveform Chart:

- The Drowsiness Level chart now uses a state-derived display severity trend.
- The waveform is derived from stable realtime warning-candidate states.
- Small deterministic bounded visual variation is used for readability only and does not affect backend logic, warning overlays, sound alerts, event counts, or `/history-48h`.
- Low, Medium, and High colors match the Drowsiness Risk gauge.
- The chart now renders only the current browser-run Live Monitor session, starts at the session start with no artificial minimum duration, grows toward 1 hour, then rolls forward as a standard 1-hour view.
- Camera-off and pre-camera gaps use a render-only gray `Idle` baseline at the Low-height display level; those idle scaffold points are not persisted as alert events or backend/history records.
- The chart avoids raw frame-level plotting, formatted-time X-axis collisions, duplicate timestamp clusters, and mock data.
- The project has a local SQLite summary archive, but still has no cloud database, production authentication, raw image/video storage, browser notification, or final system-level drowsiness accuracy claim.

## 9. Colab Notebooks

`colab_file/` stores notebooks for GPU-based training and audit runs.

| Notebook | Purpose |
| --- | --- |
| `colab_file/stage7_yawdd_training.ipynb` | Stage 7 YawDD training notebook template. |
| `colab_file/stage7_yawdd_training_r.ipynb` | Completed Stage 7 YawDD run notebook with output metrics. |
| `colab_file/stage8_nthuddd2_kaggle_training.ipynb` | Kaggle NTHUDDD2 exploration notebook. Not current main direction. |
| `colab_file/stage9_mrl_eye_training.ipynb` | Stage 9 MRL Eye Colab training notebook. |
| `colab_file/stage9_mrl_eye_training_r.ipynb` | Completed Stage 9 MRL Eye run notebook. |

The completed Stage 7 YawDD result values are present in `colab_file/stage7_yawdd_training_r.ipynb`. The local `artifacts/results/initial_results.csv` currently appears stale and contains `not_run`, so teammates should not use that CSV as the final Stage 7 result source unless it is refreshed.

## 10. Outputs

`outputs/` is for synced final outputs from cloud/Colab runs.

The main current output tree is:

```text
outputs/mrl_eye/
  README.md
  artifact_inventory.md
  results/
  reports/
  figures/
  error_analysis/
  checkpoints/
```

Important MRL Eye output groups:

| Path | Purpose |
| --- | --- |
| `outputs/mrl_eye/results/` | Metrics CSV/JSON, histories, threshold sweeps, and Stage 9B model-selection machine-readable summaries. |
| `outputs/mrl_eye/reports/` | Colab-generated Stage 9 experiment summary. |
| `outputs/mrl_eye/figures/` | Training curves, confusion matrices, and closed-class precision-recall curves. |
| `outputs/mrl_eye/error_analysis/` | False-open and false-closed contact sheets. |
| `outputs/mrl_eye/checkpoints/` | Best model checkpoint files for ResNet18, MobileNetV2, and EfficientNet-B0. |

`outputs/mrl_eye/artifact_inventory.md` confirms the expected local MRL Eye output set is complete.

Stage 10 runtime output:

| Path | Purpose |
| --- | --- |
| `outputs/stage10_eye_roi_consistency_IMG_4901_controlled_terminal/` | Successful controlled-video runtime eye ROI consistency evidence package for `IMG_4901.mp4`, including predictions, failures CSV, contact sheets, debug frames, crops, temporal summary, figures, and acceptance JSON. |

Stage 10 supporting docs:

| File | Purpose |
| --- | --- |
| `docs/stages/stage10/STAGE10_RUNTIME_EYE_ROI_DESIGN.md` | Stage 10 design notes and runtime ROI constraints. |
| `docs/stages/stage10/STAGE10_IMPLEMENTATION_LOG.md` | Stage 10 implementation log. |
| `docs/stages/stage10/STAGE10_ENVIRONMENT_SETUP.md` | Dedicated `.venv-stage10` environment setup and validation evidence. |
| `docs/stages/stage10/STAGE10_CONTROLLED_VIDEO_TEST_LOG.md` | Controlled-video test log covering Codex/sandbox failure and successful manual Terminal run. |

Stage 13 fusion design outputs:

| Path | Purpose |
| --- | --- |
| `docs/stages/stage13/STAGE13_MOUTH_EYE_FUSION_DESIGN.md` | Stage 13 fusion states, schema, and recommended tiered rule design. |
| `docs/archive/audits/stage13_mouth_eye_fusion_design_2026-05-09/stage13_mouth_runtime_audit.md` | Audit confirming that real synchronized mouth/yawn timelines for A/B/C/D are not currently available. |
| `outputs/stage13_mouth_eye_fusion_design/` | Stage 13 design/prototype outputs, including synthetic mouth timelines, fusion timelines, rule comparison CSV, summary JSON, report, and figures. |
| `reports/stage13_mouth_eye_fusion_design_report.md` | Human-readable Stage 13 fusion design/prototype report. |

Stage 14 mouth/yawn runtime audit:

| Path | Purpose |
| --- | --- |
| `docs/archive/audits/stage14_mouth_yawn_runtime_2026-05-09/stage14_mouth_model_audit.md` | Audit of mouth/yawn checkpoint, architecture, transform, and label mapping availability. |
| `docs/archive/audits/stage14_mouth_yawn_runtime_2026-05-09/STAGE14_BLOCKED_MISSING_MOUTH_MODEL_INFO.md` | Historical blocking report from before the Stage 7 checkpoint was recovered locally. |
| `docs/archive/audits/stage14_mouth_yawn_runtime_2026-05-09/STAGE14_DRIVE_CHECKPOINT_RECOVERY_REPORT.md` | Google Drive recovery report identifying the completed Stage 7 mouth/yawn checkpoint candidate. |
| `docs/archive/audits/stage14_mouth_yawn_runtime_2026-05-09/STAGE14_CHECKPOINT_LOCAL_COPY.md` | Local copy record for the recovered checkpoint. |
| `docs/archive/audits/stage14_mouth_yawn_runtime_2026-05-09/STAGE14_RECOVERED_CHECKPOINT_VERIFICATION.md` | Local checkpoint payload and ResNet18 compatibility verification. |
| `outputs/stage14_mouth_yawn_runtime_A_normal_open_baseline/` | Stage 14 runtime mouth/yawn output for A baseline video. |
| `outputs/stage14_mouth_yawn_runtime_B_realistic_drowsy_simulation/` | Stage 14 runtime mouth/yawn output for B realistic drowsy simulation video. |
| `outputs/stage14_mouth_yawn_runtime_C_mild_head_motion/` | Stage 14 runtime mouth/yawn output for C mixed fatigue/head-motion/occlusion video. |
| `outputs/stage14_mouth_yawn_runtime_D_controlled_long_open_closed/` | Stage 14 runtime mouth/yawn output for D controlled long open/closed reference video. |
| `reports/stage14_mouth_yawn_runtime_validation_report.md` | Human-readable Stage 14 multi-video runtime mouth/yawn validation report. |
| `docs/stages/stage14/STAGE14_MOUTH_YAWN_RUNTIME_LOG.md` | Stage 14 implementation and run log. |

Stage 15 real synchronized fusion validation:

| Path | Purpose |
| --- | --- |
| `docs/archive/audits/stage15_real_mouth_eye_fusion_2026-05-09/stage15_input_audit.md` | Input audit confirming Stage 12 eye timelines and Stage 14 model-generated mouth timelines were available and aligned. |
| `outputs/stage15_real_mouth_eye_fusion/` | Stage 15 real synchronized rule-based fusion outputs, including combined real mouth timeline, fusion timelines, rule comparison CSV, summary JSON, reports, and figures. |
| `reports/stage15_real_mouth_eye_fusion_validation_report.md` | Human-readable Stage 15 real mouth-eye fusion validation report. |
| `docs/stages/stage15/STAGE15_REAL_MOUTH_EYE_FUSION_LOG.md` | Stage 15 run log and evidence summary. |

Stage 16 final integration package:

| Path | Purpose |
| --- | --- |
| `docs/archive/stage16/reports/stage16_final_integration_summary_report.md` | Final high-level integration summary, architecture, evidence inventory, claim boundaries, and demo plan. |
| `docs/archive/stage16/STAGE16_FINAL_EVIDENCE_PACKAGE.md` | Structured checklist of final evidence files. |
| `docs/archive/stage16/STAGE16_DEMO_AND_PRESENTATION_OUTLINE.md` | Conservative demo and presentation outline. |
| `docs/archive/stage16/PROJECT_FINAL_STATUS_STAGE16.md` | Concise final Stage 16 status snapshot. |
| `docs/archive/stage16/audits/STAGE15_FIGURE_TITLE_FIX.md` | Audit note for Stage 15 figure-title correction. |
| `docs/archive/stage16/audits/final_repo_artifact_audit.md` | Non-destructive final repository artifact audit. |

Stage 17 video-upload MVP:

| Path | Purpose |
| --- | --- |
| `src/backend/app.py` | FastAPI backend for video upload analysis and safe session artifact serving. |
| `src/backend/static/upload_test.html` | Standalone backend-hosted upload test page. |
| `SystemUI/src/app/video-upload/page.tsx` | SystemUI video-upload analysis page. |
| `SystemUI/src/components/video-upload/` | Modular Stage 17 UI components for upload, summary cards, interval table, keyframes, technical evidence, and interpretation notice. |
| `SystemUI/src/lib/apiConfig.ts` | Shared frontend backend API base URL helper using `NEXT_PUBLIC_API_BASE_URL` with local fallback. |
| `SystemUI/src/lib/videoUploadTypes.ts` | TypeScript response and evidence types. |
| `SystemUI/src/lib/videoUploadUtils.ts` | Safe URL construction, formatting, interval merging, figure/keyframe grouping, and copy-summary helpers. |
| `outputs/system_video_upload_runs/` | Per-session Stage 17 upload-analysis outputs. |
| `reports/stage17_video_upload_detection_mvp_report.md` | Stage 17 implementation and validation report. |
| `reports/stage17_2_manual_review_interpretation_report.md` | Stage 17.2 interpretation-layer report for conservative manual review wording. |
| `reports/stage17_4_video_upload_mvp_stabilization_report.md` | Stage 17.4 current stabilization report. |
| `docs/stages/stage17/STAGE17_VIDEO_UPLOAD_RESULT_SCHEMA.md` | Result JSON, timeline, API, and keyframe schema. |
| `docs/stages/stage17/STAGE17_VIDEO_UPLOAD_DETECTION_LOG.md` | Stage 17 command log and validation summary. |
| `docs/stages/stage17/STAGE17_2_RESULT_INTERPRETATION_SCHEMA_ADDENDUM.md` | Result interpretation schema addendum for Stage 17.2. |
| `docs/stages/stage17/STAGE17_2_MANUAL_REVIEW_INTERPRETATION_NOTES.md` | Manual review interpretation notes for safe result discussion. |
| `docs/stages/stage17/STAGE17_3_VIDEO_UPLOAD_UI_PAGE_REPORT.md` | Stage 17.3 UI page report and safe wording notes. |
| `docs/stages/stage17/STAGE17_3_LOCAL_LAUNCH_GUIDE.md` | One-command local launch guide for backend and frontend. |
| `docs/stages/stage17/STAGE17_4_VIDEO_UPLOAD_UI_ACCEPTANCE_CHECKLIST.md` | Manual acceptance checklist for the Stage 17.3/17.4 video-upload MVP. |
| `docs/stages/stage17/STAGE17_4_DEMO_SCRIPT.md` | Demo script for presenting the Stage 17.4 warning-candidate MVP. |
| `docs/DEPLOYMENT_RUNBOOK.md` | External-access deployment runbook for Vercel-hosted `SystemUI/`, Cloudflare Tunnel, local FastAPI backend, environment variables, preflight validation, troubleshooting, and rollback. Current production frontend alias: `https://visionguard-systemui.vercel.app`. |
| `scripts/deployment_preflight.sh` | Safe repeatable deployment preflight for backend health and optional CORS OPTIONS validation. |
| `scripts/start_stage17_ui.sh` | Starts FastAPI backend and Next.js frontend, and stops both on Ctrl+C. |
| `Makefile` | Includes `make stage17-ui` and `make deployment-preflight` targets. |
| `tests/test_stage17_5_eye_evidence_calibration.py` | Regression tests for Stage 17.5 eye-evidence calibration and strength-gate behavior. |
| `docs/archive/audits/stage17_video_upload_mvp_2026-05-09/stage17_systemui_backend_audit.md` | SystemUI/backend audit for Stage 17. |

Stage 18, Stage 19, and Stage 20 frontend/realtime additions:

| Path | Purpose |
| --- | --- |
| `SystemUI/src/app/history-48h/page.tsx` | Stage 18/20B frontend-only 48h History page. |
| `SystemUI/src/components/history-48h/` | Stage 18/20B demo/local and Live Monitor local history charts, timeline, source badges, sessions, and review queue components. |
| `SystemUI/src/lib/history48hTypes.ts` | Stage 18/20B history event/session types with frontend-local source and optional user scoping fields. |
| `SystemUI/src/lib/history48hStorage.ts` | Stage 18/20B browser localStorage load/save/reset/clear/append helpers with user merge behavior, pruning, deduping, and compatibility normalization. |
| `SystemUI/src/lib/liveMonitorHistoryIngestion.ts` | Stage 20B mapper/helper for converting stable Live Monitor dashboard events into frontend-local history records. |
| `SystemUI/src/lib/authStore.tsx` | Stage 20A fixed local MVP account/session state; not production authentication. |
| `SystemUI/src/lib/themeStore.tsx` | Stage 20A manual Day/Night theme state. |
| `SystemUI/src/lib/notificationStore.tsx` | Stage 20A local notification state for warning-candidate, system, and review notifications. |
| `src/runtime/realtime_frame_inference.py` | Stage 19 single-frame webcam evidence service. |
| `src/runtime/realtime_temporal_state.py` | Stage 19 realtime warning-candidate temporal state with yawn context/reminder and conservative eye active/reminder semantics. |
| `SystemUI/src/lib/liveMonitorAlertUtils.ts` | Stage 19.5 frontend visual alert debounce/cooldown state helper for Live Monitor. |
| `SystemUI/src/lib/liveMonitorSoundUtils.ts` | Stage 19.6A Web Audio sound pattern and playback helper for Live Monitor warning sounds after the camera user gesture, with gentle one-shot normal warning cues and repeated urgent critical-eye cues. |
| `SystemUI/src/lib/liveMonitorRiskUtils.ts` | Stage 19.6B frontend risk mapping for idle, normal, yawn warning, eye warning, critical eye warning, and signal-quality states. |
| `SystemUI/src/lib/liveMonitorDashboardTypes.ts` | Stage 19.7C lightweight local dashboard event/risk point types. |
| `SystemUI/src/lib/liveMonitorDashboardStore.ts` | Stage 19.7C localStorage helper for stable Live Monitor warning-candidate events and timestamp-clean, bucketed state-derived display severity points. |
| `SystemUI/src/lib/settingsStore.tsx` | May 26 persisted settings helper for Minimal Live Monitor Mode under `visionguard.settings.v1`. |
| `SystemUI/src/components/dashboard/LiveVideoCard.tsx` | Stage 19.7A Live Monitor UI for webcam capture, automatic sampling, clean video frame, product yawn/eye warning overlays, concise critical-eye modal, face visibility cue, single Start/Stop Camera button, risk-state callback emission, and stable dashboard event emission. |
| `SystemUI/src/components/dashboard/DrowsinessRiskCard.tsx` | Stage 19.6B dynamic Drowsiness Risk gauge with animated frontend warning-candidate severity score and needle. |
| `SystemUI/src/components/dashboard/StatusMetricCard.tsx` | Stage 19.7A current-drive EYE/YAWN stable warning-candidate event count cards. |
| `SystemUI/src/components/dashboard/RecentEventsList.tsx` | Stage 19.7A real today-only event list with `Today` badge, no mock rows, and compact/expanded rendering for the Live Monitor right-column overlay. |
| `SystemUI/src/components/dashboard/DrowsinessLevelChart.tsx` | Stage 19.7C real current-session state-derived display severity waveform with numeric timestamp X-axis positioning, gray Idle baseline, dynamic session-start-to-1-hour windowing, Low/Medium/High segmented colors, and no formatted-time coordinate collisions. |

Expected C upload validation markers recorded for Stage 17.4:

| Marker | Expected value |
| --- | ---: |
| High-confidence warning-candidate frames | 9 |
| Suppressed brief-eye escalation frames | 8 |
| Keyframes | 4 |
| Figures | 3 |
| Interval table | present |

## 11. Model Checkpoints

Checkpoint locations:

| Path | Meaning |
| --- | --- |
| `outputs/mrl_eye/checkpoints/best_resnet18_mrl_eye.pt` | Best MRL Eye ResNet18 checkpoint. |
| `outputs/mrl_eye/checkpoints/best_mobilenet_v2_mrl_eye.pt` | Best MRL Eye MobileNetV2 checkpoint. This is the selected primary eye model. |
| `outputs/mrl_eye/checkpoints/best_efficientnet_b0_mrl_eye.pt` | Best MRL Eye EfficientNet-B0 checkpoint. |
| `checkpoints/resnet18_best.pt` | Recovered Stage 7 ResNet18 mouth/yawn checkpoint used by Stage 14+ mouth runtime. |
| `checkpoints/` | Legacy/local checkpoint folder for earlier training scripts and recovered local checkpoint copies. |

`outputs/mrl_eye/checkpoints/` and `checkpoints/resnet18_best.pt` are needed locally for model loading, but checkpoint binaries should generally not be committed to normal Git. Use Git LFS if checkpoint versioning is required.

## 12. Current Module Status

| Area | Status |
| --- | --- |
| YawDD/YawDD+ Dash mouth/yawn module | Completed and stable as the mouth/yawn specialist. |
| MRL Eye open/closed module | Stage 8 preparation, Stage 9 training, and Stage 9B model selection completed. |
| Selected MRL Eye model | MobileNetV2, default argmax / `p_eye_closed >= 0.50`. |
| Stage 10 runtime eye ROI consistency | Accepted for controlled video `IMG_4901.mp4`; not final drowsiness accuracy. |
| Stage 11 eye temporal analysis | Completed controlled-validation timeline analysis. |
| Stage 12 eye alert rule analysis | Completed eye-only rule comparison; not final system output. |
| Stage 13 mouth-eye fusion design | Completed rule-based fusion design/prototype. |
| Stage 14 mouth/yawn runtime | Completed runtime validation using recovered Stage 7 mouth/yawn checkpoint. |
| Stage 15 real synchronized fusion | Completed rule-based validation using synchronized eye and mouth timelines. |
| Stage 16 final integration package | Completed historical evidence package; superseded by Stage 17.5 upload UI, Stage 19 Live Monitor, Stage 21A History/Insights, Stage 22 local archive, and the May 26 Settings/Minimal Mode work. |
| Stage 17.1 sustained-eye gate | Completed; high-confidence warning candidate requires recent mouth/yawn evidence plus sustained eye-warning evidence. |
| Stage 17.2 interpretation wording | Completed; eye-warning evidence is not automatically described as verified sustained full eye closure. |
| Stage 17.3 Video Upload Analysis UI | Completed in SystemUI route `/video-upload`. |
| Stage 17.4 launcher and acceptance/demo docs | Completed; `make stage17-ui` starts local backend and frontend. |
| Stage 17.5 `/video-upload` evidence review cleanup | Completed; compact evidence review UI with safe optional-field handling. |
| Stage 18 `/history-48h` frontend history page | Completed frontend-only; uses demo/local browser history data. |
| Stage 19 Live Monitor realtime prototype | Completed through Stage 19.7C with local webcam preview, automatic 2 FPS sampling from Start Camera, realtime frame evidence, rule-based temporal warning-candidate state, conservative yawn/eye semantics, product-style yawn/eye/critical-eye warning overlays, face-not-visible signal-quality overlay, default-on sound after camera start, dynamic Drowsiness Risk gauge binding, current-drive EYE/YAWN stable event counts, today-only Recent Events, real timestamp-based current-session state-derived display severity waveform chart, frame-level overplotting protection, no default in-card diagnostic panel, and May 26 Minimal Mode that hides the raw preview/secondary panels while keeping monitoring and alerts active. |
| Stage 20A local account/app shell | Completed local MVP login gate for one assigned local account, user-aware profile menu, manual Day/Night theme toggle, redesigned day/night shell, local notification center, high-contrast active sidebar state, and aligned collapsed sidebar controls. |
| Stage 20B Live Monitor local history ingestion | Completed frontend-local ingestion from stable Live Monitor warning-candidate events into `/history-48h`, scoped to the current local MVP user where available. |
| Stage 21A History / Insights split | Completed event-level `/history-48h` review workstation and user-scoped `/insights` aggregate analytics page; latest polish keeps the History header concise while retaining warning-candidate boundaries in lower-level notes. |
| Stage 22 local backend archive | Completed local SQLite summary archive for stable Live Monitor events/sessions and uploaded-video analysis records, with History/Insights fallback behavior and export support. |
| External-access deployment | Validated Vercel-hosted frontend plus Cloudflare Tunnel to local FastAPI backend; depends on local backend/tunnel availability and is not a hosted production backend. |
| Safety-prioritized MRL Eye reference | ResNet18 with validation-selected threshold around `0.30`. |
| NTHUDDD2 branch | Explored but no longer the main system direction. |
| Current claim boundary | Warning-candidate prototypes with local SQLite summary archive and validated external-access frontend deployment only; no browser notification, production authentication, cloud database, raw image/video storage, hosted production backend, or final system-level performance claim. |

## 13. What Should and Should Not Be Committed to GitHub

Generally useful to commit:

- Documentation in `docs/`
- Human-readable reports in `reports/`
- Lightweight CSV/JSON metrics and manifests in `artifacts/mappings/`, `artifacts/splits/`, and `outputs/mrl_eye/results/`
- Figures in `outputs/mrl_eye/figures/`
- Error-analysis contact sheets in `outputs/mrl_eye/error_analysis/`
- Colab notebooks in `colab_file/`
- Source code in `src/`
- Frontend source code in `SystemUI/src/`
- Local launcher scripts in `scripts/`
- Lightweight Stage 17 UI/launcher documentation

Generally do not commit to normal Git:

- Raw datasets under `dataset/`
- Dataset zip files (`*.zip`)
- Model checkpoints (`*.pt`, `*.pth`, `*.ckpt`)
- `outputs/**/checkpoints/`
- `SystemUI/node_modules/`
- `SystemUI/.next/`
- Python virtual environments such as `.venv-stage10/`
- Browser/test caches such as `.playwright-cli/`
- Local caches and preprocessed bulk data
- Generated upload-session evidence under `outputs/system_video_upload_runs/`, unless a specific lightweight report or selected demo artifact is intentionally being versioned.

The current `.gitignore` protects datasets, zip files, model checkpoints, `outputs/**/checkpoints/`, cache folders, and Python/Jupyter cache artifacts.

## 14. How to Update This Document

Update this document whenever:

- A new major folder is added.
- A dataset is adopted or retired.
- A preprocessing stage creates a new canonical artifact.
- A training stage creates a new output directory.
- A checkpoint location or selected model changes.
- The FastAPI upload API, SystemUI route structure, or launcher command changes.
- A Stage 17 acceptance checklist, demo script, or stabilization report is superseded.
- Git tracking policy changes for outputs or large files.

When adding a new result, prefer citing the exact local CSV/JSON/report path used as the source of truth.
