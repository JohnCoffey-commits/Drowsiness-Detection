# Stage 10/11 Multi-Video Validation Log

## Purpose

This log records the multi-video Stage 10 runtime eye ROI and Stage 11 eye-only temporal validation run on 2026-05-09.

This is not final system-level drowsiness accuracy. It is not mouth/yawn fusion and not final fatigue scoring.

## Environment

```text
Repository: /Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection
Python environment: .venv-stage10
Scripts:
  src/runtime/stage10_eye_roi_consistency.py
  src/runtime/stage11_eye_temporal_analysis.py
```

## Videos Discovered

```text
/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/temp/test/A_normal_open_baseline.mp4
/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/temp/test/B_realistic_drowsy_simulation.mp4
/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/temp/test/C_mild_head_motion.mp4
/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/temp/test/D_controlled_long_open_closed.mp4
```

A/B/C/D were present. The videos were not copied into the repository.

## Validation Commands

`py_compile` passed for Stage 10 and Stage 11. Stage 10 preflight passed.

Validation log:

```text
/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/artifacts/audits/stage10_11_multi_video_validation_2026-05-09/validation.log
```

## Per-Video Run Logs

```text
/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/artifacts/audits/stage10_11_multi_video_validation_2026-05-09/stage10_<SLUG>.log
/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/artifacts/audits/stage10_11_multi_video_validation_2026-05-09/stage11_<SLUG>.log
```

Run manifest:

```text
/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/artifacts/audits/stage10_11_multi_video_validation_2026-05-09/run_manifest.tsv
```

## Multi-Video Summary

| video_slug | sampled_frame_count | successful_eye_crop_count | failure_count | no_face_count | invalid_crop_count | inference_failed_count | mean_p_eye_closed | mean_closed_binary_frames | candidate_event_count | max_rolling_perclos_mean_binary | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_normal_open_baseline | 70 | 140 | 0 | 0 | 0 | 0 | 0.017715 | 1 | 1 | 0.200000 | PASS |
| B_realistic_drowsy_simulation | 103 | 206 | 0 | 0 | 0 | 0 | 0.225228 | 24 | 7 | 1.000000 | PASS |
| C_mild_head_motion | 95 | 178 | 6 | 6 | 0 | 0 | 0.188989 | 16 | 5 | 1.000000 | PASS |
| D_controlled_long_open_closed | 119 | 238 | 0 | 0 | 0 | 0 | 0.467611 | 67 | 2 | 1.000000 | PASS |

## Interpretation

- `A_normal_open_baseline` (Normal-open baseline): Stage 10 completed with 0 failures, 0 no-face rows, 0 invalid crops. Stage 11 mean p_eye_closed=0.0177, mean-closed frames=1, candidate events=1, max rolling mean-binary PERCLOS-like=0.200. This is mostly consistent with a normal-open baseline: probability stayed very low, with one short candidate event that should be considered in alert-rule hysteresis/window design.
- `B_realistic_drowsy_simulation` (Realistic drowsy simulation): Stage 10 completed with 0 failures, 0 no-face rows, 0 invalid crops. Stage 11 mean p_eye_closed=0.2252, mean-closed frames=24, candidate events=7, max rolling mean-binary PERCLOS-like=1.000. This shows meaningful eye-closure temporal segments for the simulated drowsy scenario.
- `C_mild_head_motion` (Mild head motion): Stage 10 completed with 6 failures, 6 no-face rows, 0 invalid crops. Stage 11 mean p_eye_closed=0.1890, mean-closed frames=16, candidate events=5, max rolling mean-binary PERCLOS-like=1.000. ROI extraction was mostly successful under mild head motion, but the nonzero no-face rows should be reviewed before relying on live robustness.
- `D_controlled_long_open_closed` (Controlled long open/closed reference): Stage 10 completed with 0 failures, 0 no-face rows, 0 invalid crops. Stage 11 mean p_eye_closed=0.4676, mean-closed frames=67, candidate events=2, max rolling mean-binary PERCLOS-like=1.000. This shows clear temporal closure segments for the controlled open/closed reference.

## Outputs

```text
/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/outputs/stage11_multi_video_validation_summary.csv
/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/outputs/stage11_multi_video_validation_summary.json
/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/reports/stage11_multi_video_temporal_validation_report.md
```

## Recommendation

Stage 12 eye-only alert rule design can begin cautiously. The design should explicitly account for A's single short false candidate event and C's nonzero no-face rows before any live/demo rule is treated as stable.

This remains runtime eye ROI and eye-only temporal behavior validation only. It is not final system-level drowsiness accuracy.
