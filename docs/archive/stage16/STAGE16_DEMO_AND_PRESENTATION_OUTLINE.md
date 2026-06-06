# Stage 16 Demo and Presentation Outline

This outline supports a conservative final demo. It should describe warning-candidate behavior, not final driver drowsiness accuracy.

## 1. Five-Minute Demo Structure

1. State the goal: modular driver drowsiness warning-candidate prototype using eye closure and yawning signals.
2. Show the four controlled-realistic videos:
   - `A_normal_open_baseline`
   - `B_realistic_drowsy_simulation`
   - `C_mild_head_motion`
   - `D_controlled_long_open_closed`
3. Show Stage 12 eye timeline output.
4. Show Stage 14 mouth/yawn timeline output.
5. Show Stage 15 fusion timeline output.
6. Emphasize that this is synchronized rule-based fusion, not final drowsiness accuracy.

## 2. Ten-Minute Presentation Structure

1. Problem and objective.
2. Modular architecture: eye branch, mouth branch, fusion branch.
3. Dataset and model summary.
4. Runtime ROI validation: Stage 10 and Stage 14.
5. Eye-only temporal rule: Stage 12.
6. Real synchronized fusion: Stage 15.
7. Evidence and scenario results.
8. Limitations and next work.

## 3. Suggested Slide Titles

1. Driver Drowsiness Monitoring: Modular Deep Learning Prototype
2. Why Modular Eye and Mouth Specialists
3. Eye Branch: MRL Eye MobileNetV2 to `p_eye_closed`
4. Mouth Branch: YawDD ResNet18 to `p_yawn`
5. Temporal Eye Rule: PERCLOS-Like Warning Candidate
6. Runtime Mouth/Yawn Timeline
7. Stage 15 Rule-Based Fusion State Machine
8. A/B/C/D Validation Scenarios
9. B Realistic Drowsy Simulation Result
10. What We Can and Cannot Claim
11. Future Work

## 4. Figures to Show

Use these figures:

- `outputs/stage15_real_mouth_eye_fusion/figures/fusion_state_counts_by_video.png`
- `outputs/stage15_real_mouth_eye_fusion/figures/fusion_timeline_B_realistic_drowsy_simulation.png`
- `outputs/stage15_real_mouth_eye_fusion/figures/fusion_timeline_A_normal_open_baseline.png`
- `outputs/stage15_real_mouth_eye_fusion/figures/fusion_timeline_C_mild_head_motion.png`
- `outputs/stage15_real_mouth_eye_fusion/figures/fusion_timeline_D_controlled_long_open_closed.png`
- `outputs/stage14_mouth_yawn_runtime_B_realistic_drowsy_simulation/figures/p_yawn_over_time.png`

Use contact sheets only as supporting visual evidence:

- `outputs/stage14_mouth_yawn_runtime_B_realistic_drowsy_simulation/contact_sheets/`
- `outputs/stage10_eye_roi_consistency_B_realistic_drowsy_simulation/contact_sheets/`

## 5. Claims to Make

Safe claims:

- The project implements a modular eye-mouth drowsiness warning-candidate pipeline.
- Runtime eye ROI inference works on the controlled validation videos.
- Runtime mouth/yawn inference works on the controlled validation videos.
- Stage 15 completed real synchronized rule-based fusion using model-generated eye and mouth timelines.
- A/B/C/D scenario expectations were met in the small controlled-realistic validation set.
- B yawning around 14.3s-16.8s produced high `p_yawn`, and high-confidence candidate state occurred when recent-yawn evidence overlapped eye warning.

## 6. Claims to Avoid

Avoid:

- "Final drowsiness detected."
- "Final system-level drowsiness accuracy."
- "Deployment-ready."
- "Real-world road validated."
- "Clinically validated."
- "Trained fusion classifier."
- "Robust to all subjects, lighting, and camera angles."

## 7. Anticipated Questions and Strict Answers

### Why rule-based fusion instead of classifier?

Because there is no large synchronized mouth-eye dataset with temporal ground-truth drowsiness labels in the project yet. A trained fusion classifier would be premature and could overfit. Rule-based fusion is interpretable and matches the available evidence.

### Why not final accuracy?

The validation set is small, controlled-realistic, and does not include final ground-truth drowsiness timelines. The project validates runtime ROI, specialist inference, and rule-based fusion behavior, not final system-level drowsiness accuracy.

### What does `high_confidence_drowsiness_candidate` mean?

It means the rule saw eye-warning evidence and recent-yawn evidence together. It is a warning-candidate state, not a final diagnosis, clinical label, or real-world safety certification.

### How do you handle no-face or occlusion?

No-face and occlusion are treated as signal-quality issues. They are represented as `signal_unreliable` and are not counted as drowsiness.

### Why does B high-confidence occur after the visible yawn?

The mouth branch uses `recent_yawn_event`, so a yawn can influence the fusion state for a short recent-yawn window. The high-confidence candidate appears when that recent-yawn evidence overlaps with eye-warning evidence after the visible yawn interval.

### What is the next step to make it deployable?

Collect more synchronized mouth-eye videos, create temporal ground-truth labels, evaluate on more subjects and conditions, test live webcam behavior, and only then consider deployment or learned fusion.
