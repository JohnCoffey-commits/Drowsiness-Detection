# Frontend Rebuild Backend Context

Date: 2026-05-11

This document preserves the backend, API, runtime-decision, and frontend rebuild context after removing the current `SystemUI/` frontend. It is intended as the source checklist for rebuilding a new frontend without touching backend logic, runtime fusion logic, checkpoints, datasets, or output artifacts.

## 1. Current Frontend Removal Boundary

Removed frontend boundary:

- `SystemUI/`

Preserved project areas:

- `src/backend/`
- `src/runtime/`
- `src/training/`
- `src/preprocessing/`
- `checkpoints/`
- `outputs/`
- `artifacts/`
- `dataset/`
- `upload_test/`
- `docs/`
- `reports/`
- `scripts/`
- `Makefile`

Important note: `scripts/start_stage17_ui.sh` and `make stage17-ui` are intentionally preserved, but they expect `SystemUI/package.json`. After deleting `SystemUI/`, those launch commands will not work until the new frontend is recreated under `SystemUI/` or the launcher is updated later.

## 2. Claim Boundary and Safe Wording

Permanent warning text:

```text
This output is a rule-based drowsiness warning-candidate analysis, not final system-level drowsiness accuracy.
```

Safe wording for the future frontend:

- rule-based drowsiness warning-candidate analysis
- eye-warning candidate
- mouth-warning candidate
- high-confidence warning candidate
- signal unreliable
- weak eye-warning evidence
- reduced eye openness
- moderate eye-closure candidate
- strong eye-closure candidate
- blink-like activity
- manual review recommended
- rule-based fusion
- rule-based calibration

Unsafe wording to avoid:

- driver is drowsy
- final drowsiness detected
- final system-level accuracy
- final accuracy
- deployment-ready
- certified alert
- real-world validated
- webcam system
- sustained eye closure detected, unless manually confirmed

The frontend must present outputs as warning candidates for uploaded-video review, not as final drowsiness truth.

## 3. Backend Service

Backend entry point:

- `src/backend/app.py`

Recommended local command:

```bash
.venv-stage10/bin/python src/backend/app.py --host 127.0.0.1 --port 8000
```

Backend URL:

```text
http://127.0.0.1:8000
```

Preflight command:

```bash
.venv-stage10/bin/python src/backend/app.py --preflight
```

Backend CORS currently allows:

- `http://127.0.0.1:3000`
- `http://localhost:3000`
- `http://127.0.0.1:3001`
- `http://localhost:3001`

## 4. Backend API Endpoints

### `POST /api/analyze-video`

Purpose: run the uploaded-video warning-candidate pipeline.

Request:

- `multipart/form-data`
- file field name: `file`
- allowed extensions: `.mp4`, `.mov`, `.avi`, `.m4v`
- max upload size: `750 MB`

Response fields:

| Field | Meaning |
| --- | --- |
| `session_id` | Upload/run id, generated as `upload_<uuid12>`. |
| `status` | Pipeline status, usually `completed`; failure returns HTTP 500 detail. |
| `summary` | Parsed `summary.json` from the run folder. |
| `warning_counts` | Selected frame/evidence counts copied from summary. |
| `timeline_url` | `/api/runs/{session_id}/timeline`. |
| `fusion_figure_url` | `/api/runs/{session_id}/files/figures/fusion_timeline.png`. |
| `keyframes` | Keyframe metadata rows with safe `url` fields added. |
| `report_url` | `/api/runs/{session_id}/files/SYSTEM_VIDEO_UPLOAD_ANALYSIS_REPORT.md`. |
| `warning` | Permanent warning text. |
| `runtime_duration_sec` | Backend-side pipeline runtime. |
| `audit_log` | Local backend pipeline log path. Technical-only; do not expose as a user-facing local file link. |

### Read endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /api/runs/{session_id}/summary` | Returns `summary.json`. |
| `GET /api/runs/{session_id}/timeline` | Returns `timeline.csv` as CSV text. |
| `GET /api/runs/{session_id}/keyframes` | Returns keyframe metadata with safe URLs. |
| `GET /api/runs/{session_id}/files/{relative_path}` | Serves a file inside the session output folder. |

Path safety:

- `session_id` must match `[A-Za-z0-9_.-]+`.
- requested files are resolved under `outputs/system_video_upload_runs/<session_id>/`.
- paths escaping the session directory are rejected.
- keyframe URLs are only generated if the stored keyframe path resolves inside the session directory.

## 5. Runtime Pipeline

Runtime entry point:

- `src/runtime/system_video_upload_pipeline.py`

The backend calls it with:

```bash
python src/runtime/system_video_upload_pipeline.py \
  --input-video <uploaded_file> \
  --session-id <session_id> \
  --output-dir outputs/system_video_upload_runs/<session_id> \
  --sample-every-n-frames 5 \
  --max-frames 300 \
  --save-debug \
  --save-keyframes \
  --force
```

End-to-end flow:

1. uploaded video is saved under the session output folder.
2. sampled frames are processed.
3. eye ROI pipeline runs.
4. eye model writes `p_eye_closed`.
5. eye temporal rule produces `eye_warning_candidate` and `signal_unreliable`.
6. mouth/yawn runtime writes `p_yawn`.
7. yawn events become `recent_yawn_event`.
8. F5 rule-based fusion creates `fusion_state`.
9. Stage 17.1 sustained-eye gate suppresses brief eye-warning escalation.
10. Stage 17.5 eye evidence calibration adds weak/moderate/strong interpretation fields.
11. intervals, summary, figures, keyframes, and report are written.
12. backend returns summary plus safe file URLs.

## 6. Core Model and Logic Facts

Eye model:

- checkpoint: `outputs/mrl_eye/checkpoints/best_mobilenet_v2_mrl_eye.pt`
- model family: MobileNetV2
- MRL Eye mapping: `0 = closed`, `1 = open`
- formula: `p_eye_closed = softmax(logits)[0]`

Mouth/yawn model:

- checkpoint: `checkpoints/resnet18_best.pt`
- model family: ResNet18
- YawDD/YawDD+ mapping: `0 = no_yawn`, `1 = yawn`
- formula: `p_yawn = softmax(logits)[1]`

Base eye-warning rule:

- binary eye closed threshold: `p_eye_closed >= 0.50`
- rolling PERCLOS-like condition: `rolling_perclos_mean_binary >= 0.60`
- excludes signal-unreliable rows
- requires at least `2` consecutive sampled rows

Signal-unreliable rule:

- Stage 10 failure row, or
- `recent_no_face_ratio > 0.20` over `5` sampled rows
- no-face/signal-unreliable is quality evidence, not drowsiness evidence.

Yawn and recent-yawn logic:

- `yawn_event = p_yawn >= 0.50`
- `recent_yawn_event` remains true for `8.0` seconds after a yawn event.

F5 fusion logic:

| Priority | Condition | Output state |
| --- | --- | --- |
| 1 | eye signal unreliable and no recent yawn | `signal_unreliable` |
| 2 | eye signal unreliable and recent yawn | `mouth_warning_candidate` |
| 3 | eye warning and recent yawn | initial high-confidence candidate, then gated |
| 4 | eye warning only | `eye_warning_candidate` |
| 5 | recent yawn only | `mouth_warning_candidate` |
| 6 | otherwise | `normal` |

Stage 17.1 sustained-eye gate:

- high-confidence warning candidates remain only if `sustained_eye_warning == true`.
- `sustained_eye_warning` is true when:
  - current eye-warning interval duration is at least `1.0` second, or
  - current eye-warning interval has at least `5` sampled frames.
- if recent-yawn and eye-warning overlap but the eye-warning interval is too brief, the final state becomes `mouth_warning_candidate` and `high_confidence_suppressed_by_brief_eye_warning` is marked.

Stage 17.5 eye evidence calibration:

| Strength | Rule |
| --- | --- |
| `weak` | `p_eye_closed >= 0.50`, or a temporal eye-warning candidate row with lower current `p_eye_closed`. |
| `moderate` | `p_eye_closed >= 0.70`. |
| `strong` | `p_eye_closed >= 0.85`. |
| `signal_unreliable` | eye signal quality is unreliable. |

Stage 17.5 strength-aware high-confidence gate requires Stage 17.1 sustained-eye plus at least one of:

- interval mean `p_eye_closed >= 0.70`
- interval max `p_eye_closed >= 0.85`
- at least `1` strong eye-closure candidate frame
- at least `2` moderate-or-strong eye evidence frames

If recent-yawn plus sustained eye-warning evidence overlaps but calibrated eye evidence remains weak, the final state remains `mouth_warning_candidate` and `high_confidence_suppressed_by_weak_eye_evidence` is marked.

## 7. Output Folder and Files

Run folder pattern:

```text
outputs/system_video_upload_runs/<session_id>/
```

Important output files:

| File | Frontend use |
| --- | --- |
| `summary.json` | main result data. |
| `timeline.csv` | technical timeline download. |
| `fusion_timeline.csv` | technical fusion timeline download. |
| `fusion_summary.json` | technical summary/download if needed. |
| `figures/fusion_timeline.png` | fusion timeline figure. |
| `figures/p_eye_closed_over_time.png` | eye probability figure. |
| `figures/p_yawn_over_time.png` | mouth/yawn probability figure. |
| `keyframes/high_confidence/` | high-confidence warning candidate keyframes. |
| `keyframes/eye_warning/` | eye-warning candidate keyframes. |
| `keyframes/mouth_warning/` | mouth-warning candidate keyframes. |
| `keyframes/signal_unreliable/` | signal-unreliable keyframes. |
| `keyframes/keyframes_metadata.csv` | technical keyframe metadata. |
| `keyframes/keyframes_metadata.json` | technical keyframe metadata. |
| `keyframes/keyframes_summary.json` | technical keyframe summary. |
| `SYSTEM_VIDEO_UPLOAD_ANALYSIS_REPORT.md` | run report. |

Frontend should use backend file URLs, not absolute local filesystem paths.

Examples:

```text
/api/runs/{session_id}/files/summary.json
/api/runs/{session_id}/files/fusion_timeline.csv
/api/runs/{session_id}/files/figures/p_eye_closed_over_time.png
/api/runs/{session_id}/files/figures/p_yawn_over_time.png
/api/runs/{session_id}/files/keyframes/keyframes_metadata.json
```

## 8. Summary Fields to Display

Important `summary` fields:

- `session_id`
- `pipeline_status`
- `duration_sec`
- `total_frames_sampled`
- `normal_frames`
- `eye_warning_candidate_frames`
- `mouth_warning_candidate_frames`
- `high_confidence_drowsiness_candidate_frames`
- `signal_unreliable_frames`
- `yawn_event_count`
- `recent_yawn_event_count`
- `suppressed_high_confidence_brief_eye_warning_frames`
- `suppressed_high_confidence_weak_eye_evidence_frames`
- `weak_eye_warning_evidence_frames`
- `moderate_eye_closure_candidate_frames`
- `strong_eye_closure_candidate_frames`
- `eye_evidence_strength_counts`
- `mean_p_eye_closed`
- `max_p_eye_closed`
- `mean_p_yawn`
- `max_p_yawn`
- `high_confidence_intervals`
- `eye_warning_intervals`
- `mouth_warning_intervals`
- `signal_unreliable_intervals`
- `keyframes`
- `warning`
- `limitations`

## 9. Interval Object Fields

Intervals may include:

- `start_frame_index`
- `end_frame_index`
- `start_timestamp_sec`
- `end_timestamp_sec`
- `duration_sampled_frames`
- `max_p_eye_closed`
- `max_p_yawn`
- `eye_evidence_strength`
- `eye_evidence_label`
- `eye_evidence_interpretation`
- `eye_strength_gate_passed`
- `eye_strength_gate_reason`
- `eye_strength_interval_mean_p_eye_closed`
- `eye_strength_interval_max_p_eye_closed`
- `eye_strength_interval_strong_frame_count`
- `eye_strength_interval_moderate_or_strong_frame_count`
- `high_confidence_suppressed_by_weak_eye_evidence`

Future frontend should merge interval arrays from:

- `high_confidence_intervals`
- `eye_warning_intervals`
- `mouth_warning_intervals`
- `signal_unreliable_intervals`

Friendly labels:

| Raw state | UI label |
| --- | --- |
| `high_confidence_drowsiness_candidate` | High-confidence warning candidate |
| `eye_warning_candidate` | Eye-warning candidate |
| `mouth_warning_candidate` | Mouth-warning candidate |
| `signal_unreliable` | Signal unreliable |
| `normal` | Normal |

## 10. Keyframe Metadata Fields

Keyframe rows can include:

- `url`
- `keyframe_path`
- `video_path`
- `session_id`
- `frame_index`
- `timestamp_sec`
- `fusion_state`
- `p_eye_closed`
- `p_yawn`
- `recent_yawn_event`
- `sustained_eye_warning`
- `eye_evidence_strength`
- `eye_evidence_label`
- `eye_evidence_interpretation`
- `eye_strength_gate_passed`
- `eye_strength_gate_reason`
- `high_confidence_suppressed_by_weak_eye_evidence`
- `warning_type`
- `reason`
- `segment_id`
- `is_primary`

The frontend should display both image and metadata. Do not show keyframes as images only.

## 11. Recommended Future Frontend Scope

Recommended frontend directory:

```text
SystemUI/
```

Recommended framework:

- Next.js App Router
- TypeScript
- Tailwind CSS or equivalent maintainable CSS
- lucide-react for icons
- recharts or lightweight SVG charts if needed

Recommended routes:

| Route | Purpose |
| --- | --- |
| `/` | Dashboard or landing dashboard. |
| `/video-upload` | Video Upload Analysis page. |

Recommended frontend URL:

```text
http://127.0.0.1:3000/video-upload
```

The upload page should call:

```text
POST {backendUrl}/api/analyze-video
```

with `multipart/form-data` and the file field named `file`.

Default backend URL input:

```text
http://127.0.0.1:8000
```

## 12. Required Video Upload UI Sections

Future `/video-upload` should use progressive disclosure:

1. upload card
2. processing/loading pipeline
3. result header
4. permanent safety/interpretation banner
5. summary metric cards
6. warning-candidate interval review table
7. Stage 17.1 / Stage 17.5 interpretation explanation
8. figures
9. keyframe evidence gallery
10. technical evidence and downloads

Suggested loading steps:

1. Saving uploaded video
2. Extracting eye and mouth ROIs
3. Running eye-warning model
4. Running mouth/yawn model
5. Applying rule-based fusion
6. Applying eye-evidence calibration
7. Extracting warning-candidate keyframes
8. Preparing analysis report

## 13. Figure URLs for New Frontend

Returned directly:

```text
fusion_figure_url
```

Construct only from `session_id`:

```text
{backendUrl}/api/runs/{session_id}/files/figures/p_eye_closed_over_time.png
{backendUrl}/api/runs/{session_id}/files/figures/p_yawn_over_time.png
```

Never expose arbitrary absolute local paths in the UI.

## 14. Technical Links for New Frontend

If a `session_id` exists, safe links can be constructed as:

```text
{backendUrl}/api/runs/{session_id}/summary
{backendUrl}/api/runs/{session_id}/timeline
{backendUrl}/api/runs/{session_id}/keyframes
{backendUrl}/api/runs/{session_id}/files/SYSTEM_VIDEO_UPLOAD_ANALYSIS_REPORT.md
{backendUrl}/api/runs/{session_id}/files/fusion_timeline.csv
{backendUrl}/api/runs/{session_id}/files/fusion_summary.json
{backendUrl}/api/runs/{session_id}/files/keyframes/keyframes_metadata.csv
{backendUrl}/api/runs/{session_id}/files/keyframes/keyframes_metadata.json
```

## 15. Test Video and Manual Validation

Primary local test video:

```text
upload_test/C_upload_test.mp4
```

Expected Stage 17.4/17.5 validation markers may vary by runtime state, but previous acceptance docs mention:

- high-confidence frames
- suppressed brief-eye escalation frames
- keyframes
- three figures
- interval table present

The frontend should show backend-returned data rather than hardcoding expected numbers.

## 16. Rebuild Checklist

Before rebuilding:

- Do not retrain.
- Do not modify checkpoints.
- Do not change `src/runtime/` decision logic unless explicitly requested.
- Do not change `src/backend/app.py` unless the API itself must be adjusted.
- Keep the permanent warning text visible near the result header.
- Keep technical files behind progressive disclosure.
- Use backend URLs only for files.
- Keep empty/failed backend states visible and user-friendly.
- Keep the wording as warning-candidate only.

After rebuilding:

- run frontend lint/typecheck/build if scripts exist.
- start backend on `127.0.0.1:8000`.
- start frontend on `127.0.0.1:3000`.
- upload `upload_test/C_upload_test.mp4`.
- verify summary, intervals, figures, keyframes, and technical links.
- verify unsafe wording is absent.
- verify backend/runtime/model files were not changed.

