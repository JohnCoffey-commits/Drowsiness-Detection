# Stage 10 Controlled Video Test Log

## Purpose

This document records the Stage 10 runtime eye ROI consistency test for the controlled video `IMG_4901.mp4`.

Stage 10 checks whether full-frame video can be processed with MediaPipe FaceLandmarker, converted into left/right eye ROIs, passed through the selected MRL Eye MobileNetV2 checkpoint, and logged as per-eye probabilities.

This remains runtime eye ROI consistency testing only. It is not final system-level drowsiness accuracy, not a fatigue score, not fusion performance, and not deployment readiness.

## Input Video

```text
/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/temp/test/IMG_4901.mp4
```

Existence check from the Codex run:

```text
-rw-r--r--@ 1 zhengpeixian  staff    90M May  9 17:15 /Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/temp/test/IMG_4901.mp4
```

The video existed. It was not copied into the repository.

## Environment

Dedicated environment:

```text
.venv-stage10
```

Python observed in the Codex run:

```text
/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/.venv-stage10/bin/python
Python 3.12.11
```

The active Python path pointed inside `.venv-stage10`.

Successful manual Terminal controlled run device:

```text
mps
```

## Preflight Result

Preflight result:

```text
PASSED
```

The preflight validated dependency imports, the FaceLandmarker asset, checkpoint loading, MobileNetV2 construction, Stage 9 evaluation transform construction, and the label mapping `0=closed`, `1=open`, where `p_eye_closed = softmax(logits)[0]`.

Codex preflight output:

```text
artifacts/audits/stage10_controlled_video_IMG_4901_2026-05-09/preflight_terminal.log
```

## Original Codex/Sandbox Run

The original Codex/sandbox smoke test failed before the controlled run.

Command:

```bash
python src/runtime/stage10_eye_roi_consistency.py \
  --input-video "/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/temp/test/IMG_4901.mp4" \
  --sample-every-n-frames 5 \
  --max-frames 20 \
  --save-crops \
  --save-debug-frames \
  --output-dir outputs/stage10_eye_roi_consistency_IMG_4901_smoke
```

Result:

```text
FAILED
exit code: 134
```

Failure reason:

```text
MediaPipe FaceLandmarker crashed during graph initialization in the Codex/sandbox environment.
```

Key terminal error:

```text
graph_service.h:139 Check failed: service_ Service is unavailable.
DrishtiMetalHelper initWithCalculatorContext
mediapipe::api2::TensorsToDetectionsCalculator::Open()
```

Codex smoke terminal log:

```text
artifacts/audits/stage10_controlled_video_IMG_4901_2026-05-09/smoke_terminal.log
```

Because the Codex/sandbox smoke test failed, the Codex/sandbox controlled run was skipped. No Codex/sandbox runtime predictions, crops, contact sheets, or debug frames were generated for that failed attempt.

## Manual macOS Terminal Smoke Test

The same Stage 10 smoke command was later rerun manually from a normal macOS Terminal using `.venv-stage10`.

Output directory:

```text
outputs/stage10_eye_roi_consistency_IMG_4901_smoke_terminal/
```

Result:

```text
SUCCEEDED
```

Smoke metrics from `summary.json`:

```text
sampled_frame_count: 20
attempted_frame_image_count: 20
successful_eye_crop_count: 40
failure_count: 0
no_face_count: 0
invalid_crop_count: 0
inference_failed_count: 0
device: mps
model_name: mobilenet_v2
predicted_closed: 10
predicted_open: 30
p_eye_closed mean: 0.240149533
p_eye_closed min: 0.00728992
p_eye_closed max: 0.82408834
```

Manual smoke artifacts:

```text
outputs/stage10_eye_roi_consistency_IMG_4901_smoke_terminal/summary.json
outputs/stage10_eye_roi_consistency_IMG_4901_smoke_terminal/runtime_eye_roi_predictions.csv
outputs/stage10_eye_roi_consistency_IMG_4901_smoke_terminal/failures.csv
outputs/stage10_eye_roi_consistency_IMG_4901_smoke_terminal/STAGE10_RUNTIME_EYE_ROI_REPORT.md
outputs/stage10_eye_roi_consistency_IMG_4901_smoke_terminal/contact_sheets/
outputs/stage10_eye_roi_consistency_IMG_4901_smoke_terminal/debug_frames/
outputs/stage10_eye_roi_consistency_IMG_4901_smoke_terminal/crops/
```

Manual smoke artifact counts:

```text
contact_sheets: 5
debug_frames: 20
crops: 40
```

## Manual macOS Terminal Controlled Test

The full controlled test was later rerun manually from a normal macOS Terminal using `.venv-stage10`.

Output directory:

```text
outputs/stage10_eye_roi_consistency_IMG_4901_controlled_terminal/
```

Result:

```text
SUCCEEDED
```

Controlled artifacts read:

```text
outputs/stage10_eye_roi_consistency_IMG_4901_controlled_terminal/summary.json
outputs/stage10_eye_roi_consistency_IMG_4901_controlled_terminal/STAGE10_RUNTIME_EYE_ROI_REPORT.md
outputs/stage10_eye_roi_consistency_IMG_4901_controlled_terminal/runtime_eye_roi_predictions.csv
outputs/stage10_eye_roi_consistency_IMG_4901_controlled_terminal/failures.csv
```

Controlled run summary:

```text
mode: runtime_processing
sampled_frame_count: 119
attempted_frame_image_count: 119
successful_eye_crop_count: 238
failure_count: 0
no_face_count: 0
invalid_crop_count: 0
inference_failed_count: 0
device: mps
model_name: mobilenet_v2
checkpoint: outputs/mrl_eye/checkpoints/best_mobilenet_v2_mrl_eye.pt
closed_threshold: 0.5
decision_rule: argmax / p_eye_closed >= 0.50 default; runtime threshold uses --closed-threshold
```

Checkpoint metadata:

```text
model_name: mobilenet_v2
label_mapping: 0=closed, 1=open
image_size: 224
outputs: p_eye_closed, p_eye_open
checkpoint_metadata_warnings: []
```

Runtime probability summary:

```text
p_eye_closed mean: 0.46761147462184877
p_eye_closed min: 0.00512431
p_eye_closed max: 0.89752436
mean p_eye_closed left eye: 0.525946164369748
mean p_eye_closed right eye: 0.4092767848739496
predicted closed eyes: 138
predicted open eyes: 100
```

CSV row checks:

```text
runtime_eye_roi_predictions.csv: 238 prediction rows plus header
failures.csv: header only
```

Controlled visual artifact counts:

```text
contact_sheets: 5
debug_frames: 119
crops: 238
```

Generated contact sheets:

```text
outputs/stage10_eye_roi_consistency_IMG_4901_controlled_terminal/contact_sheets/high_p_eye_closed.jpg
outputs/stage10_eye_roi_consistency_IMG_4901_controlled_terminal/contact_sheets/left_eye_samples.jpg
outputs/stage10_eye_roi_consistency_IMG_4901_controlled_terminal/contact_sheets/low_p_eye_closed.jpg
outputs/stage10_eye_roi_consistency_IMG_4901_controlled_terminal/contact_sheets/mixed_runtime_eye_samples.jpg
outputs/stage10_eye_roi_consistency_IMG_4901_controlled_terminal/contact_sheets/right_eye_samples.jpg
```

## Quantitative Interpretation

The successful manual Terminal controlled run processed 119 sampled frames and produced 238 successful eye crops, which corresponds to left and right eye crops for each sampled frame.

There were no failure rows, no no-face rows, no invalid-crop rows, and no inference-failed rows in the successful controlled run.

The MRL Eye MobileNetV2 checkpoint produced a non-constant `p_eye_closed` range from `0.00512431` to `0.89752436`, with mean `0.46761147462184877`. This indicates that the runtime pipeline generated per-eye probabilities from the controlled video. It does not, by itself, validate semantic correctness of the crops or establish drowsiness accuracy.

No visual claims are made here because the contact sheets, crops, and debug frames still require human inspection.

## Next Required Inspection

Before fusion/demo, a human must inspect the contact sheets and debug frames to verify that the eye ROIs are spatially correct and stable.

Inspect:

```text
outputs/stage10_eye_roi_consistency_IMG_4901_controlled_terminal/contact_sheets/
outputs/stage10_eye_roi_consistency_IMG_4901_controlled_terminal/debug_frames/
outputs/stage10_eye_roi_consistency_IMG_4901_controlled_terminal/crops/
```

If the crops look good and `p_eye_closed` changes reasonably, proceed to temporal smoothing/fusion planning. If crops are poor, adjust the eye ROI crop policy before fusion.

## What This Proves

The successful manual Terminal run proves that the Stage 10 runtime pipeline can process the controlled video in a normal local macOS Terminal, extract eye ROIs, load the selected MRL Eye MobileNetV2 checkpoint, and produce per-eye `p_eye_closed` / `p_eye_open` predictions for sampled frames.

## What This Does Not Prove

This does not prove final drowsiness detection accuracy.

It does not prove fusion performance.

It does not prove deployment readiness.

It does not prove robustness across lighting, camera angle, glasses, all subjects, all camera orientations, or all runtime environments.

It does not change the project constraint that Stage 10 is runtime eye ROI consistency testing only.
