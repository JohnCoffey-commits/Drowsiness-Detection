# Stage 11 Eye Temporal Analysis Report

## 1. Purpose

Stage 11 analyzes eye-only temporal signal behavior from the successful Stage 10 controlled-video runtime output. It converts per-eye frame probabilities into smoothed frame-level signals and PERCLOS-like rolling ratios.

This is not final drowsiness accuracy. It is not mouth-eye fusion yet. It is not final fatigue scoring.

## 2. Input Source

Stage 10 controlled output path:

```text
/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/outputs/system_video_upload_runs/upload_19aa1aecdc46/eye_stage10
```

Input CSV:

```text
/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/outputs/system_video_upload_runs/upload_19aa1aecdc46/eye_stage10/runtime_eye_roi_predictions.csv
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
| Number of frames | 106 |
| Threshold used | 0.50 |
| Rolling window | 5 sampled frames |
| Mean `mean_p_eye_closed` | 0.215382087547 |
| Min `mean_p_eye_closed` | 0.00078236 |
| Max `mean_p_eye_closed` | 0.76270768 |
| Total `mean_closed_binary` frames | 22 |
| Total `either_eye_closed_binary` frames | 36 |
| Total `both_eyes_closed_binary` frames | 17 |
| Candidate eye-closure events | 8 |

## 5. Candidate Event Table

Top candidate eye-closure events by duration, then max probability:

| event_id | start_frame_index | end_frame_index | start_timestamp_sec | end_timestamp_sec | duration_sampled_frames | max_mean_p_eye_closed | mean_mean_p_eye_closed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 155 | 205 | 6.457558 | 8.540641 | 11 | 0.751870 | 0.621920 |
| 6 | 355 | 365 | 14.789891 | 15.206507 | 3 | 0.762708 | 0.630780 |
| 1 | 30 | 35 | 1.249850 | 1.458158 | 2 | 0.714846 | 0.649295 |
| 7 | 375 | 380 | 15.623124 | 15.831432 | 2 | 0.706902 | 0.677486 |
| 2 | 45 | 45 | 1.874775 | 1.874775 | 1 | 0.667937 | 0.667937 |
| 8 | 390 | 390 | 16.248049 | 16.248049 | 1 | 0.591607 | 0.591607 |
| 3 | 75 | 75 | 3.124625 | 3.124625 | 1 | 0.567800 | 0.567800 |
| 5 | 275 | 275 | 11.456957 | 11.456957 | 1 | 0.546059 | 0.546059 |

Candidate events are not confirmed drowsiness events; they are temporal eye-closure candidates derived from the eye-only signal.

## 6. Figures

- `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/outputs/system_video_upload_runs/upload_19aa1aecdc46/eye_stage11/figures/p_eye_closed_rolling_mean.png`
- `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/outputs/system_video_upload_runs/upload_19aa1aecdc46/eye_stage11/figures/eye_closed_binary_timeline.png`
- `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/outputs/system_video_upload_runs/upload_19aa1aecdc46/eye_stage11/figures/perclos_like_score_over_time.png`

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
