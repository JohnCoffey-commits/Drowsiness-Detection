# Stage 16 Final Integration Summary Report

## 1. Project Objective

This project builds a modular driver drowsiness detection and monitoring prototype using specialist deep learning modules and rule-based temporal fusion.

The system uses two visible driver behavior signals:

- Eye closure behavior, represented by `p_eye_closed`.
- Yawning behavior, represented by `p_yawn`.

The current output is a rule-based warning-candidate timeline. It is not final system-level driver drowsiness accuracy, not deployment readiness, and not a trained fusion classifier.

## 2. Final System Architecture

### Eye Branch

1. Full-face video input.
2. MediaPipe FaceLandmarker eye ROI extraction.
3. MRL Eye MobileNetV2 eye specialist.
4. Per-eye `p_eye_closed`, where `p_eye_closed = softmax(logits)[0]`.
5. Stage 11 eye-only temporal analysis.
6. Stage 12 eye alert rule: `quality_gated_perclos_mean_ge_0.60_consec`.

### Mouth Branch

1. Full-face video input.
2. MediaPipe FaceLandmarker mouth/lip ROI extraction.
3. YawDD/YawDD+ Dash ResNet18 mouth/yawn specialist.
4. Runtime `p_yawn`, where `p_yawn = softmax(logits)[1]`.
5. Stage 14 runtime yawn timeline with `yawn_event` and `recent_yawn_event`.

### Fusion Branch

1. Stage 12 eye alert timeline.
2. Stage 14 model-generated mouth/yawn timeline.
3. Stage 15 synchronized rule-based fusion.
4. Recommended fusion rule: `F5_tiered_quality_aware_fusion`.

Fusion states:

- `normal`
- `eye_warning_candidate`
- `mouth_warning_candidate`
- `high_confidence_drowsiness_candidate`
- `signal_unreliable`

## 3. Completed Stages

| Stage | Summary |
| --- | --- |
| Stage 7 | Trained YawDD/YawDD+ Dash mouth/yawn specialists; ResNet18 selected as the final mouth/yawn model. |
| Stage 8 | Prepared MRL Eye dataset and confirmed label mapping `0 = closed`, `1 = open`. |
| Stage 9 | Trained MRL Eye baseline models. |
| Stage 9B | Selected MobileNetV2 as the primary MRL Eye model. |
| Stage 10 | Validated runtime eye ROI consistency on controlled video using MediaPipe and MRL Eye checkpoint. |
| Stage 11 | Built eye-only temporal smoothing and PERCLOS-like signal analysis. |
| Stage 12 | Compared eye-only alert rules and selected `quality_gated_perclos_mean_ge_0.60_consec`. |
| Stage 13 | Designed rule-based mouth-eye fusion and generated a prototype using synthetic/example mouth timelines. |
| Stage 13B | Ran manual B-yawn annotation sanity check for fusion logic only. |
| Stage 14 | Implemented runtime mouth/yawn inference and generated model-based `p_yawn` timelines. |
| Stage 15 | Completed real synchronized rule-based mouth-eye fusion using Stage 12 eye timelines and Stage 14 model-generated mouth timelines. |
| Stage 16 | Consolidated final evidence package, demo plan, and final integration summary. |

## 4. Final Selected Components

| Component | Selected item |
| --- | --- |
| Eye model | MobileNetV2 MRL Eye specialist |
| Eye label mapping | `0 = closed`, `1 = open` |
| Eye probability | `p_eye_closed = softmax(logits)[0]` |
| Eye rule | `quality_gated_perclos_mean_ge_0.60_consec` |
| Mouth model | ResNet18 YawDD/YawDD+ Dash specialist |
| Mouth label mapping | `0 = no_yawn`, `1 = yawn` |
| Mouth probability | `p_yawn = softmax(logits)[1]` |
| Fusion rule | `F5_tiered_quality_aware_fusion` |

## 5. Stage 15 Final Validation Summary

| Video | Expected behavior | Stage 15 result |
| --- | --- | --- |
| `A_normal_open_baseline` | Mostly normal, no yawn, no high-confidence candidate. | 70 normal frames, 0 mouth-warning frames, 0 high-confidence frames. |
| `B_realistic_drowsy_simulation` | Drowsiness-like behavior with yawning. | Stage 14 `p_yawn` was high during 14.3s-16.8s; Stage 15 high-confidence candidate occurred from 16.882456s to 17.924583s when recent-yawn evidence overlapped eye warning. |
| `C_mild_head_motion` | Mixed fatigue-like eye closure, head motion, and partial occlusion; preserve signal quality markers. | 12 signal-unreliable frames, 0 mouth-warning frames, 0 high-confidence frames. |
| `D_controlled_long_open_closed` | Eye-warning behavior only, no mouth escalation. | 65 eye-warning frames, 0 mouth-warning frames, 0 high-confidence frames. |

Stage 15 used real Stage 12 eye timelines and real Stage 14 model-generated mouth/yawn timelines. It did not use synthetic mouth timelines and did not use manual mouth annotation timelines for fusion decisions.

## 6. Evidence Inventory

| Evidence | Path |
| --- | --- |
| Stage 12 report | `reports/stage12_eye_alert_rule_analysis_report.md` |
| Stage 14 report | `reports/stage14_mouth_yawn_runtime_validation_report.md` |
| Stage 15 report | `reports/stage15_real_mouth_eye_fusion_validation_report.md` |
| Stage 15 summary JSON | `outputs/stage15_real_mouth_eye_fusion/stage15_real_fusion_summary.json` |
| Stage 15 rule comparison CSV | `outputs/stage15_real_mouth_eye_fusion/stage15_real_fusion_rule_comparison.csv` |
| Stage 15 B timeline CSV | `outputs/stage15_real_mouth_eye_fusion/timelines/fusion_timeline_B_realistic_drowsy_simulation.csv` |
| Stage 15 figures | `outputs/stage15_real_mouth_eye_fusion/figures/` |
| Stage 14 B contact sheets | `outputs/stage14_mouth_yawn_runtime_B_realistic_drowsy_simulation/contact_sheets/` |
| Checkpoint verification report | `artifacts/audits/stage14_mouth_yawn_runtime_2026-05-09/STAGE14_RECOVERED_CHECKPOINT_VERIFICATION.md` |
| Current status doc | `docs/PROJECT_CURRENT_STATUS.md` |

## 7. What Can Be Claimed

The project can claim:

- Runtime eye ROI inference works on the controlled validation videos.
- Runtime mouth/yawn inference works on the controlled validation videos.
- Stage 15 completed real synchronized rule-based fusion using model-generated eye and mouth timelines.
- A/B/C/D scenario expectations were met in the small controlled-realistic validation set.
- The B yawning interval was detected by the runtime mouth/yawn model and contributed to a high-confidence candidate state when combined with eye-warning evidence.

## 8. What Cannot Be Claimed

The project must not claim:

- Final system-level drowsiness accuracy.
- Deployment readiness.
- Real-world road validation.
- A trained fusion classifier.
- Clinical validation.
- Broad robustness across camera setups, lighting, drivers, or occlusions.

Important limitations:

- Small validation set.
- One or few subjects.
- No final ground-truth drowsiness timeline.
- Rule-based fusion only.

## 9. Demo Plan

A conservative demo should:

1. Show the A/B/C/D validation videos.
2. Show the Stage 12 eye timeline for each video.
3. Show the Stage 14 mouth/yawn timeline for each video.
4. Show the Stage 15 fusion timeline for each video.
5. Explain fusion states:
   - `normal`
   - `eye_warning_candidate`
   - `mouth_warning_candidate`
   - `high_confidence_drowsiness_candidate`
   - `signal_unreliable`
6. Avoid saying "final drowsiness detected."
7. Say "drowsiness warning candidate" or "high-confidence warning candidate."

## 10. Future Work

Recommended next steps:

- Collect more synchronized mouth-eye videos.
- Create temporal ground-truth labels.
- Evaluate on more subjects and more conditions.
- Implement a live webcam demo if needed.
- Consider a learned fusion classifier only after synchronized annotated data exists.

This Stage 16 report packages the current project evidence. It does not convert the prototype into a final production system.
