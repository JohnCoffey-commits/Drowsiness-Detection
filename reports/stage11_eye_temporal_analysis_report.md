# Stage 11 Eye Temporal Analysis Report

## 1. Purpose

Stage 11 analyzes eye-only temporal signal behavior from the successful Stage 10 controlled-video runtime output. It converts per-eye frame probabilities into smoothed frame-level signals and PERCLOS-like rolling ratios.

This is not final drowsiness accuracy. It is not mouth-eye fusion yet. It is not final fatigue scoring.

## 2. Input Source

Stage 10 controlled output path:

```text
/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/outputs/system_video_upload_runs/upload_e1a9578f689a/eye_stage10
```

Input CSV:

```text
/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/outputs/system_video_upload_runs/upload_e1a9578f689a/eye_stage10/runtime_eye_roi_predictions.csv
```

## 3. Method

- Read per-eye `p_eye_closed` and `p_eye_open` predictions from Stage 10.
- Group left/right eye predictions by `frame_index`.
- Compute frame-level `mean_p_eye_closed`, `max_p_eye_closed`, and `min_p_eye_closed`.
- Use threshold `0.50` for binary closed-eye indicators.
- Compute rolling statistics over `5` sampled frames.
- Treat rolling binary ratios as PERCLOS-like eye-only signals, not as final drowsiness labels.
- Detect candidate eye-closure events as contiguous sampled-frame sequences where `mean_closed_binary == 1`.

## 4. Summary Metrics

| Metric | Value |
| --- | ---: |
| Number of frames | 107 |
| Threshold used | 0.50 |
| Rolling window | 5 sampled frames |
| Mean `mean_p_eye_closed` | 0.226194021869 |
| Min `mean_p_eye_closed` | 0.00023945 |
| Max `mean_p_eye_closed` | 0.83566874 |
| Total `mean_closed_binary` frames | 29 |
| Total `either_eye_closed_binary` frames | 35 |
| Total `both_eyes_closed_binary` frames | 19 |
| Candidate eye-closure events | 5 |

## 5. Candidate Event Table

Top candidate eye-closure events by duration, then max probability:

| event_id | start_frame_index | end_frame_index | start_timestamp_sec | end_timestamp_sec | duration_sampled_frames | max_mean_p_eye_closed | mean_mean_p_eye_closed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 135 | 185 | 4.501264 | 6.168399 | 11 | 0.835669 | 0.763417 |
| 1 | 25 | 60 | 0.833567 | 2.000562 | 8 | 0.714331 | 0.635124 |
| 3 | 235 | 265 | 7.835534 | 8.835815 | 7 | 0.761746 | 0.683651 |
| 5 | 495 | 500 | 16.504635 | 16.671348 | 2 | 0.736083 | 0.625975 |
| 4 | 465 | 465 | 15.504354 | 15.504354 | 1 | 0.538363 | 0.538363 |

Candidate events are not confirmed drowsiness events; they are temporal eye-closure candidates derived from the eye-only signal.

## 6. Figures

- `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/outputs/system_video_upload_runs/upload_e1a9578f689a/eye_stage11/figures/p_eye_closed_rolling_mean.png`
- `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/outputs/system_video_upload_runs/upload_e1a9578f689a/eye_stage11/figures/eye_closed_binary_timeline.png`
- `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/outputs/system_video_upload_runs/upload_e1a9578f689a/eye_stage11/figures/perclos_like_score_over_time.png`

## 7. Interpretation

The controlled-video signal appears usable for temporal smoothing on this one video: Stage 10 provided complete per-eye predictions, Stage 11 produced a continuous per-frame signal, and rolling PERCLOS-like ratios were generated without missing-frame failures.

This does not claim final drowsiness accuracy.

## 8. Limitations

- One controlled video only.
- No ground truth temporal annotation.
- No mouth/yawn fusion yet.
- No live webcam validation.
- No deployment robustness validation.
- No claim is made for all lighting, camera angles, glasses, subjects, or runtime environments.

## 9. Recommended Next Step

If the temporal signal looks stable under human inspection, proceed to Stage 12 fusion design with `p_yawn`. If the signal is noisy, adjust smoothing window and threshold policy before fusion.
