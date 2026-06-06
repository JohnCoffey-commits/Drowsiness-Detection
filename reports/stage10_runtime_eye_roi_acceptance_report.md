# Stage 10 Runtime Eye ROI Acceptance Report

## 1. Purpose

Stage 10 validates runtime eye ROI consistency for the MRL Eye specialist. It checks whether full-frame video can be processed into MediaPipe FaceLandmarker eye crops, whether those crops can be passed into the selected MRL Eye MobileNetV2 checkpoint, and whether per-eye `p_eye_closed` values are produced consistently.

This report is not final drowsiness accuracy. It is not fusion. It is not a fatigue score.

## 2. Evidence Sources

Input files read:

- `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/outputs/stage10_eye_roi_consistency_IMG_4901_controlled_terminal/summary.json`
- `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/outputs/stage10_eye_roi_consistency_IMG_4901_controlled_terminal/runtime_eye_roi_predictions.csv`
- `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/outputs/stage10_eye_roi_consistency_IMG_4901_controlled_terminal/failures.csv`
- `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/outputs/stage10_eye_roi_consistency_IMG_4901_controlled_terminal/STAGE10_RUNTIME_EYE_ROI_REPORT.md`
- `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/docs/stages/stage10/STAGE10_CONTROLLED_VIDEO_TEST_LOG.md`

Controlled video:

- `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/temp/test/IMG_4901.mp4`

Controlled video output directory:

- `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/outputs/stage10_eye_roi_consistency_IMG_4901_controlled_terminal`

Visual inspection directories:

- `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/outputs/stage10_eye_roi_consistency_IMG_4901_controlled_terminal/contact_sheets`
- `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/outputs/stage10_eye_roi_consistency_IMG_4901_controlled_terminal/debug_frames`
- `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/outputs/stage10_eye_roi_consistency_IMG_4901_controlled_terminal/crops`

Generated acceptance artifacts:

- `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/outputs/stage10_eye_roi_consistency_IMG_4901_controlled_terminal/stage10_acceptance_summary.json`
- `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/outputs/stage10_eye_roi_consistency_IMG_4901_controlled_terminal/stage10_temporal_eye_summary.csv`
- `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/outputs/stage10_eye_roi_consistency_IMG_4901_controlled_terminal/figures/p_eye_closed_over_time.png`
- `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/outputs/stage10_eye_roi_consistency_IMG_4901_controlled_terminal/figures/p_eye_closed_by_eye_side.png`
- `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/outputs/stage10_eye_roi_consistency_IMG_4901_controlled_terminal/figures/p_eye_closed_histogram.png`

## 3. Environment and Model

| Item | Value |
| --- | --- |
| Environment | `.venv-stage10` |
| Python | `3.12.11` |
| Device used in successful controlled run | `mps` |
| Model | `MobileNetV2` / `mobilenet_v2` |
| Checkpoint | `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/outputs/mrl_eye/checkpoints/best_mobilenet_v2_mrl_eye.pt` |
| Label mapping | `0 = closed`, `1 = open` |
| Closed-eye probability | `p_eye_closed = softmax(logits)[0]` |
| Evaluation preprocessing | Stage 9 MRL Eye evaluation transform was used |
| Decision rule | `argmax / p_eye_closed >= 0.50 default; runtime threshold uses --closed-threshold` |

## 4. Runtime Execution Result

| Metric | Value |
| --- | ---: |
| Sampled frames | 119 |
| Attempted frames/images | 119 |
| Successful eye crops | 238 |
| Expected eye crops for sampled frames | 238 |
| Failure rows | 0 |
| No-face rows | 0 |
| Invalid-crop rows | 0 |
| Inference-failed rows | 0 |
| Prediction CSV rows | 238 |
| `failures.csv` status | header only; no failure rows |

## 5. Probability Behavior

Overall probability summary:

| Probability | Mean | Std | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| `p_eye_closed` | 0.467611474622 | 0.322994273165 | 0.00512431 | 0.89752436 |
| `p_eye_open` | 0.532388527563 | 0.322994274554 | 0.10247567 | 0.99487573 |

`p_eye_closed` by eye side:

| Eye side | Mean | Std | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| left | 0.525946164370 | 0.315825707462 | 0.01853061 | 0.89752436 |
| right | 0.409276784874 | 0.320819711487 | 0.00512431 | 0.87707412 |

Predicted closed/open count by eye side:

| Eye side | Predicted closed | Predicted open |
| --- | ---: | ---: |
| left | 70 | 49 |
| right | 68 | 51 |

First 10 prediction rows:

| source_id | frame_index | timestamp_sec | eye_side | p_eye_closed | p_eye_open | pred_label | crop_width | crop_height |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| video_000001_frame_00000000 | 0 | 0.000000 | left | 0.109565 | 0.890435 | open | 459 | 135 |
| video_000001_frame_00000000 | 0 | 0.000000 | right | 0.026915 | 0.973085 | open | 451 | 146 |
| video_000002_frame_00000005 | 5 | 0.166709 | left | 0.553069 | 0.446931 | closed | 466 | 134 |
| video_000002_frame_00000005 | 5 | 0.166709 | right | 0.014243 | 0.985757 | open | 457 | 144 |
| video_000003_frame_00000010 | 10 | 0.333418 | left | 0.067602 | 0.932398 | open | 450 | 136 |
| video_000003_frame_00000010 | 10 | 0.333418 | right | 0.039086 | 0.960914 | open | 438 | 141 |
| video_000004_frame_00000015 | 15 | 0.500126 | left | 0.319198 | 0.680802 | open | 457 | 131 |
| video_000004_frame_00000015 | 15 | 0.500126 | right | 0.029617 | 0.970383 | open | 450 | 142 |
| video_000005_frame_00000020 | 20 | 0.666835 | left | 0.091213 | 0.908787 | open | 456 | 134 |
| video_000005_frame_00000020 | 20 | 0.666835 | right | 0.030082 | 0.969918 | open | 447 | 141 |

Top 10 highest `p_eye_closed` rows:

| source_id | frame_index | timestamp_sec | eye_side | p_eye_closed | p_eye_open | pred_label | crop_width | crop_height |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| video_000085_frame_00000420 | 420 | 14.003535 | left | 0.897524 | 0.102476 | closed | 491 | 81 |
| video_000021_frame_00000100 | 100 | 3.334175 | left | 0.890847 | 0.109153 | closed | 493 | 77 |
| video_000082_frame_00000405 | 405 | 13.503409 | right | 0.877074 | 0.122926 | closed | 479 | 59 |
| video_000041_frame_00000200 | 200 | 6.668350 | left | 0.875004 | 0.124996 | closed | 496 | 88 |
| video_000043_frame_00000210 | 210 | 7.001768 | left | 0.860549 | 0.139451 | closed | 496 | 83 |
| video_000086_frame_00000425 | 425 | 14.170244 | left | 0.854827 | 0.145173 | closed | 487 | 79 |
| video_000022_frame_00000105 | 105 | 3.500884 | left | 0.854228 | 0.145772 | closed | 497 | 78 |
| video_000118_frame_00000585 | 585 | 19.504924 | left | 0.849270 | 0.150730 | closed | 483 | 77 |
| video_000105_frame_00000520 | 520 | 17.337710 | left | 0.843033 | 0.156967 | closed | 491 | 84 |
| video_000035_frame_00000170 | 170 | 5.668098 | left | 0.835959 | 0.164041 | closed | 491 | 87 |

Top 10 lowest `p_eye_closed` rows:

| source_id | frame_index | timestamp_sec | eye_side | p_eye_closed | p_eye_open | pred_label | crop_width | crop_height |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| video_000077_frame_00000380 | 380 | 12.669865 | right | 0.005124 | 0.994876 | open | 454 | 155 |
| video_000075_frame_00000370 | 370 | 12.336448 | right | 0.006118 | 0.993882 | open | 450 | 158 |
| video_000009_frame_00000040 | 40 | 1.333670 | right | 0.007290 | 0.992710 | open | 453 | 148 |
| video_000071_frame_00000350 | 350 | 11.669613 | right | 0.007576 | 0.992424 | open | 449 | 149 |
| video_000059_frame_00000290 | 290 | 9.669108 | right | 0.008474 | 0.991526 | open | 454 | 150 |
| video_000008_frame_00000035 | 35 | 1.166961 | right | 0.008536 | 0.991464 | open | 450 | 150 |
| video_000062_frame_00000305 | 305 | 10.169234 | right | 0.008974 | 0.991026 | open | 453 | 153 |
| video_000076_frame_00000375 | 375 | 12.503157 | right | 0.009394 | 0.990606 | open | 455 | 153 |
| video_000048_frame_00000235 | 235 | 7.835311 | right | 0.009461 | 0.990539 | open | 460 | 163 |
| video_000073_frame_00000360 | 360 | 12.003030 | right | 0.009555 | 0.990445 | open | 453 | 151 |

## 6. Temporal Behavior

A per-frame temporal summary was computed by grouping both eyes by `frame_index`, averaging left/right `p_eye_closed`, and adding a rolling mean over 5 sampled frames.

Temporal CSV:

- `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/outputs/stage10_eye_roi_consistency_IMG_4901_controlled_terminal/stage10_temporal_eye_summary.csv`

Generated figures:

- `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/outputs/stage10_eye_roi_consistency_IMG_4901_controlled_terminal/figures/p_eye_closed_over_time.png`
- `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/outputs/stage10_eye_roi_consistency_IMG_4901_controlled_terminal/figures/p_eye_closed_by_eye_side.png`
- `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/outputs/stage10_eye_roi_consistency_IMG_4901_controlled_terminal/figures/p_eye_closed_histogram.png`

## 7. Human Visual Inspection Note

The user manually inspected the Stage 10 contact sheets and debug frames. The user reported that the eye ROIs looked basically accurate.

This supports Stage 10 controlled-video acceptance for this one video. It does not prove robustness across all videos, subjects, lighting conditions, glasses, camera angles, camera orientations, or deployment settings.

## 8. Acceptance Decision

Stage 10 controlled video test: **PASSED for IMG_4901.mp4**.

Reason:

- Preflight passed.
- Manual macOS Terminal smoke test succeeded.
- Manual macOS Terminal controlled run succeeded.
- 238/238 expected eye crops were generated.
- `failures.csv` had 0 failure rows.
- No no-face, invalid-crop, or inference-failed rows were produced.
- User visual inspection found the eye ROIs basically accurate.

Limit: this is one controlled video only.

## 9. Recommended Next Step

Proceed to Stage 11: temporal smoothing / PERCLOS-like eye signal analysis. After the runtime eye signal is stable, later work can plan fusion with `p_yawn`.

Do not connect SystemUI yet unless the runtime signal is stable. Do not claim final drowsiness accuracy yet.
