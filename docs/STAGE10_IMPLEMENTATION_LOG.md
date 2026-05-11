# Stage 10 Implementation Log

## 1. What Was Done

This Codex run implemented the next safe Python-only step before fusion/demo:

Stage 10 Runtime Eye ROI Consistency Test.

The implementation adds a runtime script that can perform preflight validation, process image inputs, process sampled video frames, crop left/right eye ROIs using MediaPipe FaceLandmarker landmarks, run the selected MRL Eye MobileNetV2 specialist checkpoint, and write inspectable audit artifacts.

No model training was performed.

## 2. Files Created

- `src/runtime/__init__.py`
- `src/runtime/stage10_eye_roi_consistency.py`
- `docs/STAGE10_RUNTIME_EYE_ROI_DESIGN.md`
- `docs/STAGE10_IMPLEMENTATION_LOG.md`

The preflight command also writes runtime audit outputs under:

- `outputs/stage10_eye_roi_consistency/runtime_eye_roi_predictions.csv`
- `outputs/stage10_eye_roi_consistency/failures.csv`
- `outputs/stage10_eye_roi_consistency/summary.json`
- `outputs/stage10_eye_roi_consistency/STAGE10_RUNTIME_EYE_ROI_REPORT.md`

The output subdirectories are:

- `outputs/stage10_eye_roi_consistency/crops/`
- `outputs/stage10_eye_roi_consistency/debug_frames/`
- `outputs/stage10_eye_roi_consistency/contact_sheets/`

## 3. Existing Files Moved

No additional audit cleanup moves were needed in this run. The pre-Stage-10 audit files were already under:

`artifacts/audits/stage10_pre_audit_2026-05-09/`

The expected `README.md`, `repo_snapshot/`, and `codex_audit/` structure was present.

## 4. Validations Run

Requested validations:

```bash
python -m py_compile src/runtime/stage10_eye_roi_consistency.py
python src/runtime/stage10_eye_roi_consistency.py --preflight
```

## 5. Validation Result

`python -m py_compile src/runtime/stage10_eye_roi_consistency.py`

- Result: PASSED.

`python src/runtime/stage10_eye_roi_consistency.py --preflight`

- Result: FAILED in the active `python` environment because external runtime dependencies are missing.
- Missing dependencies reported by the script: `mediapipe`, `opencv-python`, and `torch/torchvision`.
- Suggested install command reported by the script:

```bash
pip install mediapipe opencv-python pillow numpy torch torchvision
```

The failed preflight wrote a failure summary to:

`outputs/stage10_eye_roi_consistency/summary.json`

If preflight passes, it proves that dependencies are importable in the active Python environment, required assets exist, the MobileNetV2 model can be constructed, the checkpoint payload can be loaded from `payload["state_dict"]`, Stage 9 evaluation transforms can be built, and checkpoint metadata can be checked.

## 6. What This Implementation Currently Proves

The implementation proves the Stage 10 runtime test harness has been created without modifying Stage 8 outputs, Stage 9 outputs, or checkpoints.

In the active `python` environment, setup and checkpoint compatibility are not fully proven yet because preflight is blocked before checkpoint loading by missing external dependencies.

## 7. What It Does Not Prove Yet

It does not prove runtime ROI quality on real driver video.

It does not prove stable `p_eye_closed` values across time.

It does not prove final driver drowsiness accuracy.

It does not produce a fatigue score.

It does not connect Python runtime output to `SystemUI`.

It does not use or revive NTHUDDD2.

## 8. Input Needed Next

The next controlled test needs a short pre-recorded face video or a small image folder that is safe to process outside `dataset/` and `data/`.

Recommended next command shape:

```bash
python src/runtime/stage10_eye_roi_consistency.py \
  --input-video path/to/controlled_face_video.mp4 \
  --sample-every-n-frames 5 \
  --max-frames 120 \
  --save-crops \
  --save-debug-frames
```

## 9. Exact Document And Output Paths

- `docs/STAGE10_RUNTIME_EYE_ROI_DESIGN.md`
- `docs/STAGE10_IMPLEMENTATION_LOG.md`
- `src/runtime/stage10_eye_roi_consistency.py`
- `outputs/stage10_eye_roi_consistency/summary.json`
- `outputs/stage10_eye_roi_consistency/STAGE10_RUNTIME_EYE_ROI_REPORT.md`
- `outputs/stage10_eye_roi_consistency/runtime_eye_roi_predictions.csv`
- `outputs/stage10_eye_roi_consistency/failures.csv`

## 10. Warning

This Stage 10 implementation is runtime ROI consistency testing only. It is not final system-level drowsiness accuracy.
