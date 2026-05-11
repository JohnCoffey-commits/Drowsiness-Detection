# Project Final Status at Stage 16

## Current Stage

Stage 16 is complete.

The project has reached final integration summary, evidence packaging, and demo planning for a controlled-validation prototype.

## Final Validated Pipeline

The validated pipeline is modular:

1. Full-face video input.
2. Eye ROI extraction with MediaPipe FaceLandmarker.
3. MRL Eye MobileNetV2 inference to produce `p_eye_closed`.
4. Eye-only temporal analysis and Stage 12 alert rule.
5. Mouth ROI extraction with MediaPipe FaceLandmarker.
6. YawDD/YawDD+ Dash ResNet18 inference to produce `p_yawn`.
7. Stage 15 real synchronized rule-based mouth-eye fusion.

## Completed Components

| Component | Status |
| --- | --- |
| YawDD/YawDD+ Dash mouth/yawn specialist | Completed; ResNet18 selected. |
| MRL Eye open/closed specialist | Completed; MobileNetV2 selected. |
| Runtime eye ROI consistency | Completed on controlled validation videos. |
| Eye-only temporal analysis | Completed. |
| Eye-only alert rule | Completed; `quality_gated_perclos_mean_ge_0.60_consec`. |
| Runtime mouth/yawn inference | Completed on controlled validation videos. |
| Real synchronized mouth-eye fusion | Completed using Stage 12 eye timelines and Stage 14 model-generated mouth timelines. |
| Final evidence package and demo plan | Completed. |

## Key Results

Stage 15 F5 fusion results:

| Video | Key result |
| --- | --- |
| `A_normal_open_baseline` | Mostly normal; no mouth warning or high-confidence candidate. |
| `B_realistic_drowsy_simulation` | Stage 14 detected high `p_yawn` during 14.3s-16.8s; Stage 15 produced high-confidence candidates from 16.882456s to 17.924583s when recent-yawn evidence overlapped eye warning. |
| `C_mild_head_motion` | Signal-unreliable intervals were preserved; no mouth/yawn false positives were used. |
| `D_controlled_long_open_closed` | Eye-warning behavior remained eye-only without mouth escalation. |

## Known Limitations

- Not final system-level drowsiness accuracy.
- Not deployment-ready.
- Not a trained fusion classifier.
- Small controlled-realistic validation set.
- One or few subjects.
- No final ground-truth drowsiness timeline.
- No real-world road validation.

## Next Recommended Action

Prepare the final demo and integration presentation using the Stage 16 evidence package.

Future technical work should collect more synchronized mouth-eye videos with temporal labels, evaluate across more subjects and conditions, and only then consider live deployment or learned fusion.
