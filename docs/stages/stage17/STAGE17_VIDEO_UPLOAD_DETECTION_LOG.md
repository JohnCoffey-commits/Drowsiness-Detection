# Stage 17 Video Upload Detection Log

## Purpose

Stage 17 implemented a video-upload inference/demo MVP. It lets a user upload a video, run the eye-mouth rule-based warning-candidate pipeline, view summary results, and inspect keyframe screenshots.

This is not final system-level drowsiness accuracy and not deployment readiness.

## Files Created

- `src/runtime/system_video_upload_pipeline.py`
- `src/runtime/keyframe_extractor.py`
- `src/backend/app.py`
- `src/backend/static/upload_test.html`
- `SystemUI/src/app/video-upload/page.tsx`
- `docs/stages/stage17/STAGE17_VIDEO_UPLOAD_RESULT_SCHEMA.md`
- `docs/stages/stage17/STAGE17_VIDEO_UPLOAD_DETECTION_LOG.md`
- `reports/stage17_video_upload_detection_mvp_report.md`
- `docs/archive/audits/stage17_video_upload_mvp_2026-05-09/stage17_systemui_backend_audit.md`

## Files Updated

- `SystemUI/src/components/dashboard/Sidebar.tsx`
- `docs/PROJECT_CURRENT_STATUS.md`
- `docs/PROJECT_STRUCTURE.md`

## Environment Note

Backend validation initially reported missing FastAPI dependencies. The following minimal backend packages were installed into `.venv-stage10`:

```bash
python -m pip install fastapi uvicorn python-multipart
```

No model training was run. No checkpoints were modified.

## Commands Run

Compile validation:

```bash
source .venv-stage10/bin/activate
python -m py_compile src/runtime/system_video_upload_pipeline.py
python -m py_compile src/runtime/keyframe_extractor.py
python -m py_compile src/backend/app.py
```

Backend preflight:

```bash
python src/backend/app.py --preflight
```

Direct CLI pipeline test:

```bash
python src/runtime/system_video_upload_pipeline.py \
  --input-video "/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/temp/test/B_realistic_drowsy_simulation.mp4" \
  --session-id "stage17_test_B_realistic_drowsy_simulation" \
  --output-dir "outputs/system_video_upload_runs/stage17_test_B_realistic_drowsy_simulation" \
  --sample-every-n-frames 5 \
  --max-frames 300 \
  --save-debug \
  --save-keyframes \
  --force
```

Backend upload test:

```bash
python src/backend/app.py --host 127.0.0.1 --port 8000
curl -X POST \
  -F "file=@/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/temp/test/B_realistic_drowsy_simulation.mp4" \
  http://127.0.0.1:8000/api/analyze-video
```

SystemUI lint:

```bash
cd SystemUI
npm run lint
```

## Validation Result

| Check | Result |
| --- | --- |
| Runtime pipeline compile | Passed |
| Keyframe extractor compile | Passed |
| Backend compile | Passed |
| Backend preflight | Passed after installing FastAPI backend dependencies |
| Direct CLI B test | Passed |
| Backend B upload test | Passed |
| Backend keyframe URL GET | Passed |
| SystemUI lint | Passed |

## B Test Summary

Direct CLI session:

```text
outputs/system_video_upload_runs/stage17_test_B_realistic_drowsy_simulation/
```

Metrics:

- Total sampled frames: 103
- Normal frames: 49
- Eye-warning candidate frames: 18
- Mouth-warning candidate frames: 30
- High-confidence warning candidate frames: 6
- Signal-unreliable frames: 0
- Yawn-event count: 14
- Recent-yawn-event count: 36
- Keyframes extracted: 3

High-confidence warning-candidate interval:

```text
16.882456s-17.924583s
```

Backend session:

```text
outputs/system_video_upload_runs/upload_c8194a181da6/
```

## Known Limitations

- MVP uses synchronous backend request handling.
- Long videos should be moved to background processing.
- The SystemUI upload page expects the FastAPI backend to run separately.
- Outputs are warning-candidate states, not final drowsy/not-drowsy truth.
- No final system-level drowsiness accuracy is claimed.

## How to Run

Start backend:

```bash
source .venv-stage10/bin/activate
python src/backend/app.py --host 127.0.0.1 --port 8000
```

Open backend test page:

```text
http://127.0.0.1:8000/static/upload_test.html
```

Start SystemUI:

```bash
cd SystemUI
npm run dev
```

Open:

```text
http://127.0.0.1:3000/video-upload
```

Use backend URL:

```text
http://127.0.0.1:8000
```
