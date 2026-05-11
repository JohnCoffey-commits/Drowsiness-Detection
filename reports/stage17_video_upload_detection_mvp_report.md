# Stage 17 Video Upload Detection MVP Report

## 1. Purpose

Stage 17 implements a video-upload inference/demo MVP. A user can upload a video, run the existing eye-mouth rule-based pipeline, view warning-candidate results, and inspect keyframe screenshots.

This stage does not train models, does not modify checkpoints, does not introduce a fusion classifier, and does not claim final system-level drowsiness accuracy.

## 2. Backend/API Implementation

Backend file:

- `src/backend/app.py`

Implemented endpoints:

| Endpoint | Purpose |
| --- | --- |
| `POST /api/analyze-video` | Upload a video and run the Stage 17 pipeline synchronously. |
| `GET /api/runs/{session_id}/summary` | Return `summary.json`. |
| `GET /api/runs/{session_id}/timeline` | Return timeline CSV. |
| `GET /api/runs/{session_id}/keyframes` | Return keyframe metadata and URLs. |
| `GET /api/runs/{session_id}/files/...` | Serve files under the session output directory only. |

The MVP validates file extensions and stores uploaded videos under:

```text
outputs/system_video_upload_runs/<session_id>/input/
```

Processing is synchronous for short demo videos. Long-video async/background processing is future work.

## 3. Pipeline Implementation

Pipeline file:

- `src/runtime/system_video_upload_pipeline.py`

For each uploaded video, the pipeline creates:

```text
outputs/system_video_upload_runs/<session_id>/
```

It then runs:

1. Stage 10 eye ROI runtime inference.
2. Stage 11 eye temporal analysis.
3. Stage 17 single-video Stage 12-style eye alert adapter.
4. Stage 14 mouth/yawn runtime inference.
5. Stage 17 F5 real eye-mouth fusion.
6. Keyframe extraction.

No A/B/C/D precomputed outputs are used for uploaded videos.

Stage 17.1 update:

- High-confidence warning candidates now require sustained eye-warning evidence.
- A frame can be upgraded to `high_confidence_drowsiness_candidate` only when recent-yawn evidence overlaps an eye-warning interval that lasts at least 1.0 second or at least 5 sampled frames.
- Brief normal-blink-like eye events overlapping recent-yawn evidence are suppressed from high-confidence escalation and remain `mouth_warning_candidate` when recent-yawn evidence is active.

## 4. Keyframe Extraction Method

Keyframe helper:

- `src/runtime/keyframe_extractor.py`

Primary keyframes are extracted from `high_confidence_drowsiness_candidate` intervals. If no high-confidence intervals exist, the extractor uses `eye_warning_candidate` and `mouth_warning_candidate` intervals. Signal-quality examples are saved separately from `signal_unreliable` intervals.

For each warning segment, the extractor saves a small subset of frames:

- first frame
- midpoint frame
- max-score frame
- last frame for longer segments

It does not save every frame. Default maximum: 20 keyframes.

## 5. SystemUI Integration Status

SystemUI is a Next.js app.

Added:

- `SystemUI/src/app/video-upload/page.tsx`
- Sidebar link in `SystemUI/src/components/dashboard/Sidebar.tsx`

Also added a backend-hosted standalone test page:

- `src/backend/static/upload_test.html`

The SystemUI page talks to a separate FastAPI backend, defaulting to:

```text
http://127.0.0.1:8000
```

## 6. Test Result on B

Test video:

```text
/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/temp/test/B_realistic_drowsy_simulation.mp4
```

Direct CLI session:

```text
stage17_test_B_realistic_drowsy_simulation
```

Direct CLI output:

```text
outputs/system_video_upload_runs/stage17_test_B_realistic_drowsy_simulation/
```

Summary:

| Metric | Value |
| --- | ---: |
| Total sampled frames | 103 |
| Normal frames | 49 |
| Eye-warning candidate frames | 18 |
| Mouth-warning candidate frames | 30 |
| High-confidence warning candidate frames | 6 |
| Signal-unreliable frames | 0 |
| Yawn-event count | 14 |
| Recent-yawn-event count | 36 |
| Mean `p_yawn` | 0.149097 |
| Max `p_yawn` | 0.997966 |
| Mean `p_eye_closed` | 0.225228 |
| Max `p_eye_closed` | 0.729110 |
| Keyframes extracted | 3 |

High-confidence warning-candidate interval:

```text
16.882456s-17.924583s
```

Backend upload test also passed. Backend-created session:

```text
upload_c8194a181da6
```

Backend output:

```text
outputs/system_video_upload_runs/upload_c8194a181da6/
```

The backend JSON response returned the expected summary, figure URL, report URL, and keyframe URLs. A keyframe URL was fetched successfully.

## 7. Key Output Paths

Direct CLI output:

- `outputs/system_video_upload_runs/stage17_test_B_realistic_drowsy_simulation/summary.json`
- `outputs/system_video_upload_runs/stage17_test_B_realistic_drowsy_simulation/timeline.csv`
- `outputs/system_video_upload_runs/stage17_test_B_realistic_drowsy_simulation/fusion_timeline.csv`
- `outputs/system_video_upload_runs/stage17_test_B_realistic_drowsy_simulation/figures/fusion_timeline.png`
- `outputs/system_video_upload_runs/stage17_test_B_realistic_drowsy_simulation/keyframes/keyframes_metadata.csv`
- `outputs/system_video_upload_runs/stage17_test_B_realistic_drowsy_simulation/SYSTEM_VIDEO_UPLOAD_ANALYSIS_REPORT.md`

Backend audit/test artifacts:

- `artifacts/audits/stage17_video_upload_mvp_2026-05-09/backend_preflight.json`
- `artifacts/audits/stage17_video_upload_mvp_2026-05-09/backend_upload_B_response.json`
- `artifacts/audits/stage17_video_upload_mvp_2026-05-09/backend_first_keyframe_check.jpg`
- `artifacts/audits/stage17_video_upload_mvp_2026-05-09/backend_summary_check.json`

## 8. Limitations

- Video-upload inference/demo MVP only.
- Synchronous request handling is intended for short demo videos.
- Not final system-level drowsiness accuracy.
- Not deployment readiness.
- Not a trained fusion classifier.
- Runtime quality still depends on MediaPipe ROI success, lighting, camera angle, occlusion, and subject variability.

## 9. Next Step

The next technical step is a real-time webcam detection system or an async/background-job version of this upload MVP.

Before any deployment claim, collect more synchronized videos, add temporal labels, evaluate across more subjects and environments, and harden runtime behavior.
