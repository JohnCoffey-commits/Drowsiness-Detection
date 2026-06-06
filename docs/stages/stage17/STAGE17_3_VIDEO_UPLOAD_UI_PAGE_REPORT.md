# Stage 17.3 Video Upload UI Page Report

## Page Route

- Frontend route: `/video-upload`
- Page title/menu label: `Video Upload Analysis`
- Backend endpoint used: `POST {backendUrl}/api/analyze-video`
- Default backend URL: `http://127.0.0.1:8000`

## Menu Placement

The left sidebar now places `Video Upload Analysis` directly under `Dashboard` with an `Active MVP` badge.

Menu order:

1. Dashboard
2. Video Upload Analysis
3. Live Monitor
4. 48h History
5. Session Review
6. Insights
7. Model Details
8. Alerts
9. Settings

## Files Changed

Modified:

- `SystemUI/src/app/video-upload/page.tsx`
- `SystemUI/src/components/dashboard/Sidebar.tsx`
- `SystemUI/src/components/dashboard/TopBar.tsx`

Created:

- `SystemUI/src/components/video-upload/VideoUploadAnalysis.tsx`
- `SystemUI/src/components/video-upload/AnalysisSummaryCards.tsx`
- `SystemUI/src/components/video-upload/IntervalReviewTable.tsx`
- `SystemUI/src/components/video-upload/InterpretationNotice.tsx`
- `SystemUI/src/components/video-upload/KeyframeEvidenceGallery.tsx`
- `SystemUI/src/components/video-upload/TechnicalEvidencePanel.tsx`
- `SystemUI/src/lib/videoUploadTypes.ts`
- `SystemUI/src/lib/videoUploadUtils.ts`
- `docs/stages/stage17/STAGE17_3_VIDEO_UPLOAD_UI_PAGE_REPORT.md`

## UI Sections Implemented

- Polished upload area with file picker, drag/drop-style target, selected file name, size, type, reset, and choose-another-video action.
- Collapsible advanced settings with backend URL.
- Multi-step processing indicator for video-upload analysis.
- Result header with session id, pipeline status, duration, sampled frames, and Stage 17.1/17.2 labels.
- Safety and interpretation banner with the required permanent warning text.
- Summary metric cards, including suppressed brief-eye escalation when returned.
- `Warning-candidate intervals` table merging high-confidence, eye-warning, mouth-warning, and signal-unreliable intervals.
- Stage 17.1 / Stage 17.2 interpretation guidance card.
- Figures section for `fusion_timeline.png`, `p_eye_closed_over_time.png`, and `p_yawn_over_time.png`.
- Keyframe evidence gallery grouped by warning-candidate category with timestamp, frame index, fusion state, `p_eye_closed`, `p_yawn`, recent-yawn, sustained-eye-warning, and reason metadata.
- Collapsible technical evidence links for report, summary, timelines, and keyframe metadata.
- Copyable safe-worded analysis summary.
- Empty state describing the evidence review workflow before upload.

## Safe Wording Notes

The page uses warning-candidate wording only. It includes the permanent warning:

> This output is a rule-based drowsiness warning-candidate analysis, not final system-level drowsiness accuracy.

The UI avoids final driver-state claims and describes outputs as:

- High-confidence warning candidate
- Eye-warning candidate
- Mouth-warning candidate
- Signal unreliable
- Sustained eye-warning evidence
- Reduced eye openness
- Brief blink-like activity
- Brief eye-closure evidence
- Rule-based fusion

## Backend/File Safety

- Upload uses `multipart/form-data`.
- File links are built only from returned backend API paths or a validated `session_id`.
- The UI does not expose local absolute filesystem paths from backend metadata.
- Technical links use `/api/runs/{session_id}/summary`, `/api/runs/{session_id}/timeline`, and constrained `/api/runs/{session_id}/files/...` paths.

## Limitations

- The page depends on the FastAPI backend being started separately.
- The current backend accepts the video file payload only, so max-keyframes and session-name controls were not added.
- Manual interpretation fields are supported if later returned in interval/keyframe metadata, but the UI does not create manual annotations.
- This page is not a webcam page and does not implement live monitoring.
- No model logic, Stage 17.1 fusion logic, training code, or model checkpoints were changed.

## Testing Performed

- `npm run lint` from `SystemUI/`: passed. A non-blocking `pyenv` rehash warning appeared.
- `npm run build` from `SystemUI/`: passed.
- `python src/backend/app.py --preflight`: default Python reported missing FastAPI dependencies.
- `.venv-stage10/bin/python src/backend/app.py --preflight`: passed using the repository virtual environment.
- Started backend with `.venv-stage10/bin/python src/backend/app.py --host 127.0.0.1 --port 8000`.
- Started frontend with `npm run dev -- --hostname 127.0.0.1 --port 3000`.
- Opened `/video-upload` and confirmed the sidebar item appears directly under Dashboard.
- Uploaded `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/upload_test/C_upload_test.mp4` through the UI.
- Backend returned `POST /api/analyze-video` status `200`.
- Verified rendered result for session `upload_680971e85f3e`:
  - `Analysis completed`
  - high-confidence warning candidates found
  - duration `21.9s`
  - sampled frames `106`
  - high-confidence warning candidate frames `9`
  - suppressed brief-eye escalation frames `8`
  - warning-candidate interval table with `5` intervals
  - fusion, eye-signal, and mouth/yawn figures loaded
  - keyframe gallery loaded `4` high-confidence warning-candidate keyframes with metadata
  - technical evidence links appeared for report, summary, timelines, and keyframe metadata
- Playwright console check: `0` errors, `0` warnings.
- Screenshots were not required; verification was performed with browser snapshots and backend request logs.
