# Stage 10/11 Multi-Video Temporal Validation Report

## Purpose

This report summarizes multi-video validation for runtime eye ROI extraction and eye-only temporal behavior using Stage 10 and Stage 11.

This is not final system-level drowsiness accuracy. It is not mouth/yawn fusion and not final fatigue scoring.

## Videos Tested

| Video slug | Scenario | Video path |
| --- | --- | --- |
| `A_normal_open_baseline` | Normal-open baseline | `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/temp/test/A_normal_open_baseline.mp4` |
| `B_realistic_drowsy_simulation` | Realistic drowsy simulation | `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/temp/test/B_realistic_drowsy_simulation.mp4` |
| `C_mild_head_motion` | Mild head motion | `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/temp/test/C_mild_head_motion.mp4` |
| `D_controlled_long_open_closed` | Controlled long open/closed reference | `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/temp/test/D_controlled_long_open_closed.mp4` |


## Summary Table

| video_slug | sampled_frame_count | successful_eye_crop_count | failure_count | no_face_count | invalid_crop_count | inference_failed_count | mean_p_eye_closed | mean_closed_binary_frames | candidate_event_count | max_rolling_perclos_mean_binary | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_normal_open_baseline | 70 | 140 | 0 | 0 | 0 | 0 | 0.017715 | 1 | 1 | 0.200000 | PASS |
| B_realistic_drowsy_simulation | 103 | 206 | 0 | 0 | 0 | 0 | 0.225228 | 24 | 7 | 1.000000 | PASS |
| C_mild_head_motion | 95 | 178 | 6 | 6 | 0 | 0 | 0.188989 | 16 | 5 | 1.000000 | PASS |
| D_controlled_long_open_closed | 119 | 238 | 0 | 0 | 0 | 0 | 0.467611 | 67 | 2 | 1.000000 | PASS |

## Stage 10 Result by Video

- `A_normal_open_baseline`: status `PASS`, sampled frames 70, eye crops 140, failures 0, no_face 0, invalid_crop 0, inference_failed 0; contact sheets 5, debug frames 70.
- `B_realistic_drowsy_simulation`: status `PASS`, sampled frames 103, eye crops 206, failures 0, no_face 0, invalid_crop 0, inference_failed 0; contact sheets 5, debug frames 103.
- `C_mild_head_motion`: status `PASS`, sampled frames 95, eye crops 178, failures 6, no_face 6, invalid_crop 0, inference_failed 0; contact sheets 5, debug frames 89.
- `D_controlled_long_open_closed`: status `PASS`, sampled frames 119, eye crops 238, failures 0, no_face 0, invalid_crop 0, inference_failed 0; contact sheets 5, debug frames 119.

All Stage 10 outputs include `summary.json`, `runtime_eye_roi_predictions.csv`, `failures.csv`, contact sheets, crops, and debug frames in each per-video output directory.

## Stage 11 Result by Video

- `A_normal_open_baseline`: mean p_eye_closed 0.0177, mean_closed frames 1, either-eye frames 1, both-eyes frames 1, candidate events 1, max rolling mean-binary PERCLOS-like 0.200.
- `B_realistic_drowsy_simulation`: mean p_eye_closed 0.2252, mean_closed frames 24, either-eye frames 35, both-eyes frames 12, candidate events 7, max rolling mean-binary PERCLOS-like 1.000.
- `C_mild_head_motion`: mean p_eye_closed 0.1890, mean_closed frames 16, either-eye frames 24, both-eyes frames 11, candidate events 5, max rolling mean-binary PERCLOS-like 1.000.
- `D_controlled_long_open_closed`: mean p_eye_closed 0.4676, mean_closed frames 67, either-eye frames 71, both-eyes frames 67, candidate events 2, max rolling mean-binary PERCLOS-like 1.000.

All Stage 11 outputs include temporal summary CSV, event CSV, summary JSON, report Markdown, and three figures in each per-video output directory.

## Scenario-Level Interpretation

- `A_normal_open_baseline` (Normal-open baseline): Stage 10 completed with 0 failures, 0 no-face rows, 0 invalid crops. Stage 11 mean p_eye_closed=0.0177, mean-closed frames=1, candidate events=1, max rolling mean-binary PERCLOS-like=0.200. This is mostly consistent with a normal-open baseline: probability stayed very low, with one short candidate event that should be considered in alert-rule hysteresis/window design.
- `B_realistic_drowsy_simulation` (Realistic drowsy simulation): Stage 10 completed with 0 failures, 0 no-face rows, 0 invalid crops. Stage 11 mean p_eye_closed=0.2252, mean-closed frames=24, candidate events=7, max rolling mean-binary PERCLOS-like=1.000. This shows meaningful eye-closure temporal segments for the simulated drowsy scenario.
- `C_mild_head_motion` (Mild head motion): Stage 10 completed with 6 failures, 6 no-face rows, 0 invalid crops. Stage 11 mean p_eye_closed=0.1890, mean-closed frames=16, candidate events=5, max rolling mean-binary PERCLOS-like=1.000. ROI extraction was mostly successful under mild head motion, but the nonzero no-face rows should be reviewed before relying on live robustness.
- `D_controlled_long_open_closed` (Controlled long open/closed reference): Stage 10 completed with 0 failures, 0 no-face rows, 0 invalid crops. Stage 11 mean p_eye_closed=0.4676, mean-closed frames=67, candidate events=2, max rolling mean-binary PERCLOS-like=1.000. This shows clear temporal closure segments for the controlled open/closed reference.

## Limitations

- Small validation set.
- No ground truth annotation timeline.
- One subject or limited subjects.
- No mouth/yawn fusion yet.
- No live webcam validation.
- Not final drowsiness accuracy.
- Human review of contact sheets, debug frames, and Stage 11 figures is still required before relying on the signal.

## Recommendation

Stage 12 eye-only alert rule design can begin cautiously. The design should explicitly account for A's single short false candidate event and C's nonzero no-face rows before any live/demo rule is treated as stable.

If A is stable, B detects closure, and C has low failure rate, Stage 12 eye-only alert rule design can begin. If A has many long false closure events, threshold/window logic needs adjustment before Stage 12. If C has many no_face or invalid_crop failures, ROI robustness must be fixed before Stage 12.

## Artifact Paths

- Summary CSV: `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/outputs/stage11_multi_video_validation_summary.csv`
- Summary JSON: `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/outputs/stage11_multi_video_validation_summary.json`
- Audit logs: `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/artifacts/audits/stage10_11_multi_video_validation_2026-05-09`
