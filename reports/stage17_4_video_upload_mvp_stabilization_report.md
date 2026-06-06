# Stage 17.4 Video Upload MVP Stabilization Report

## Purpose

Stage 17.4 stabilizes the local video-upload warning-candidate MVP for manual acceptance and demonstration. It consolidates the Stage 17.1 sustained-eye gate, Stage 17.2 interpretation boundary, Stage 17.3 Video Upload Analysis UI, and local launcher workflow into a clear acceptance package.

This report documents the local operating boundary. It does not introduce model changes, Stage 17.1 fusion changes, webcam support, retraining, or deployment-readiness claims.

Stage 17.5 is implemented after this stabilization package as a conservative rule-based eye evidence calibration layer. It does not retrain models, replace specialist models, modify checkpoints, or change `p_eye_closed = softmax(logits)[0]` / `p_yawn = softmax(logits)[1]`.

## Completed Components

### Stage 17.1 Sustained-eye Gate

Stage 17.1 adds a sustained-eye gate to the rule-based fusion pathway. High-confidence warning candidates require recent mouth/yawn evidence plus sustained eye-warning evidence. Brief blink-like activity overlapping recent-yawn evidence is conservatively suppressed from high-confidence escalation.

### Stage 17.2 Interpretation Documentation

Stage 17.2 documents the interpretation layer for manual review and safe wording. Eye-warning evidence is not automatically treated as sustained full eye closure. It may reflect reduced eye openness, blink-like activity, brief closure, fatigue-like appearance, or ROI-sensitive cases.

### Stage 17.3 Video Upload Analysis UI

Stage 17.3 implements the `/video-upload` analysis workstation in the independent Next.js frontend. The page supports upload, loading pipeline status, summary metrics, warning-candidate interval review, figures, keyframe evidence, technical file links, and the permanent warning text.

### Stage 17.4 One-command Launcher

Stage 17.4 adds a local launcher for the backend and frontend:

- `scripts/start_stage17_ui.sh`
- `make stage17-ui`
- `docs/stages/stage17/STAGE17_3_LOCAL_LAUNCH_GUIDE.md`

The launcher starts both services and stops them together when interrupted.

### Stage 17.5 Eye Evidence Calibration

Stage 17.5 adds weak, moderate, and strong eye evidence interpretation fields to runtime outputs, summaries, intervals, keyframes, and the UI. It keeps weak eye-warning evidence visible for manual review while adding a strength-aware gate before high-confidence warning candidates are retained.

## Local Launch Command

From the repository root:

```bash
make stage17-ui
```

## URLs

- Backend: `http://127.0.0.1:8000`
- Frontend: `http://127.0.0.1:3000/video-upload`

## Test Status

- Backend verified: yes.
- Frontend verified: yes.
- Launcher stop behavior verified: yes.

Verification performed during Stage 17.3/17.4 stabilization confirmed:

- Backend readiness endpoint responded locally.
- Frontend `/video-upload` page responded locally.
- Launcher signal handling stopped both services.

No expensive model test was required for creating this Stage 17.4 documentation package.

## Expected C Upload Validation Markers

For:

```text
/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/upload_test/C_upload_test.mp4
```

Expected local UI validation markers:

| Marker | Expected value |
|---|---:|
| High-confidence warning candidate frames | 9 |
| Suppressed brief-eye escalation frames | 8 |
| Keyframes | 4 |
| Figures | 3 |
| Interval table | Present |

The expected values are warning-candidate analysis markers from the local Stage 17 workflow. They are not final system-level drowsiness accuracy.

## Safe Wording Notes

Use:

- Rule-based drowsiness warning-candidate analysis
- High-confidence warning candidate
- Eye-warning candidate
- Mouth-warning candidate
- Signal unreliable
- Sustained eye-warning evidence
- Reduced eye openness
- Brief blink-like activity
- Brief closure
- Fatigue-like appearance
- ROI-sensitive cases
- Rule-based fusion

Required warning:

> This output is a rule-based drowsiness warning-candidate analysis, not final system-level drowsiness accuracy.

Avoid final-state or deployment claims. The Stage 17.4 MVP is for local uploaded-video analysis and manual review support.

## Remaining Limitations

- Local backend/frontend only.
- No webcam support.
- No real-time detection.
- No final system-level drowsiness accuracy.
- No deployment readiness.
- No trained fusion classifier.
- Rule-based fusion only.
- Manual review is still required for interpretation-sensitive intervals.

## Recommended Next Stage

Stage 18: real-time webcam warning-candidate feasibility prototype, only after Stage 17.4 acceptance is complete.

Stage 18 should preserve the same claim boundary: warning-candidate output only, no final system-level drowsiness accuracy claim, and no deployment-readiness claim unless separately validated.
