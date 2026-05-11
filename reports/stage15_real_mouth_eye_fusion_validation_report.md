# Stage 15 Real Mouth-Eye Fusion Validation Report

## 1. Purpose

Stage 15 validates real synchronized rule-based mouth-eye fusion using Stage 12 real eye timelines and Stage 14 model-generated `p_yawn` timelines. This run is not synthetic, not manual mouth annotation, not a trained fusion classifier, and not final system-level drowsiness accuracy.

## 2. Inputs

- Stage 12 eye timelines: `outputs/stage12_eye_alert_rule_analysis/stage12_video_alert_timeline_<slug>.csv`
- Stage 14 mouth/yawn timelines: `outputs/stage14_mouth_yawn_runtime_<slug>/runtime_mouth_yawn_predictions.csv`
- Combined Stage 14 real mouth timeline: `outputs/stage15_real_mouth_eye_fusion/combined_stage14_real_mouth_timeline.csv`
- Mouth timeline source: `stage14_runtime_mouth_yawn_model`
- Input audit: `artifacts/audits/stage15_real_mouth_eye_fusion_2026-05-09/stage15_input_audit.md`

## 3. Input Audit Result

- All required inputs available: True
- Real Stage 14 mouth timelines used: true
- Synthetic mouth timelines used: false
- Manual mouth annotation used: false
- Stage 14 C no-face rows were represented as mouth signal-quality rows and were not treated as yawn.

## 4. Fusion Rule

The validated rule is `F5_tiered_quality_aware_fusion`:

- If eye signal is unreliable and no recent yawn exists, output `signal_unreliable`.
- If eye signal is unreliable and a recent yawn exists, output `mouth_warning_candidate`.
- If eye warning candidate and recent yawn event co-occur, output `high_confidence_drowsiness_candidate`.
- If only eye warning is active, output `eye_warning_candidate`.
- If only recent yawn is active, output `mouth_warning_candidate`.
- Otherwise output `normal`.

`signal_unreliable`, `eye_warning_candidate`, `mouth_warning_candidate`, and `high_confidence_drowsiness_candidate` remain rule-based candidate states, not final driver drowsiness accuracy.

## 5. Per-Video Results

| video_slug | total_rows | normal_frames | eye_warning_candidate_frames | mouth_warning_candidate_frames | high_confidence_drowsiness_candidate_frames | signal_unreliable_frames | yawn_event_count | recent_yawn_event_count | scenario_expectation_match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_normal_open_baseline | 70 | 70 | 0 | 0 | 0 | 0 | 0 | 0 | True |
| B_realistic_drowsy_simulation | 103 | 49 | 18 | 30 | 6 | 0 | 14 | 36 | True |
| C_mild_head_motion | 95 | 76 | 7 | 0 | 0 | 12 | 0 | 0 | True |
| D_controlled_long_open_closed | 119 | 54 | 65 | 0 | 0 | 0 | 0 | 0 | True |

## 6. Rule Comparison

| video_slug | rule_name | normal_frames | eye_warning_candidate_frames | mouth_warning_candidate_frames | high_confidence_drowsiness_candidate_frames | signal_unreliable_frames | alert_count | longest_any_warning_run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_normal_open_baseline | F1_eye_only_baseline | 70 | 0 | 0 | 0 | 0 | 0 | 0 |
| A_normal_open_baseline | F2_mouth_only_baseline | 70 | 0 | 0 | 0 | 0 | 0 | 0 |
| A_normal_open_baseline | F3_or_fusion | 70 | 0 | 0 | 0 | 0 | 0 | 0 |
| A_normal_open_baseline | F4_and_near_window_fusion | 70 | 0 | 0 | 0 | 0 | 0 | 0 |
| A_normal_open_baseline | F5_tiered_quality_aware_fusion | 70 | 0 | 0 | 0 | 0 | 0 | 0 |
| B_realistic_drowsy_simulation | F1_eye_only_baseline | 79 | 24 | 0 | 0 | 0 | 3 | 13 |
| B_realistic_drowsy_simulation | F2_mouth_only_baseline | 67 | 0 | 36 | 0 | 0 | 1 | 36 |
| B_realistic_drowsy_simulation | F3_or_fusion | 49 | 18 | 30 | 6 | 0 | 3 | 36 |
| B_realistic_drowsy_simulation | F4_and_near_window_fusion | 97 | 0 | 0 | 6 | 0 | 1 | 6 |
| B_realistic_drowsy_simulation | F5_tiered_quality_aware_fusion | 49 | 18 | 30 | 6 | 0 | 3 | 36 |
| C_mild_head_motion | F1_eye_only_baseline | 76 | 7 | 0 | 0 | 12 | 3 | 3 |
| C_mild_head_motion | F2_mouth_only_baseline | 89 | 0 | 0 | 0 | 6 | 0 | 0 |
| C_mild_head_motion | F3_or_fusion | 76 | 7 | 0 | 0 | 12 | 3 | 3 |
| C_mild_head_motion | F4_and_near_window_fusion | 83 | 0 | 0 | 0 | 12 | 0 | 0 |
| C_mild_head_motion | F5_tiered_quality_aware_fusion | 76 | 7 | 0 | 0 | 12 | 3 | 3 |
| D_controlled_long_open_closed | F1_eye_only_baseline | 54 | 65 | 0 | 0 | 0 | 2 | 36 |
| D_controlled_long_open_closed | F2_mouth_only_baseline | 119 | 0 | 0 | 0 | 0 | 0 | 0 |
| D_controlled_long_open_closed | F3_or_fusion | 54 | 65 | 0 | 0 | 0 | 2 | 36 |
| D_controlled_long_open_closed | F4_and_near_window_fusion | 119 | 0 | 0 | 0 | 0 | 0 | 0 |
| D_controlled_long_open_closed | F5_tiered_quality_aware_fusion | 54 | 65 | 0 | 0 | 0 | 2 | 36 |

## 7. B-Specific Real Yawn Validation

The user manually observed yawning in `B_realistic_drowsy_simulation` around 14.3s-16.8s. Stage 15 did not use that manual annotation for fusion decisions; it used Stage 14 model-generated `p_yawn` only.

- Rows in 14.3s-16.8s: 12
- Yawn-event rows in 14.3s-16.8s: 12
- Mean/min/max `p_yawn` in 14.3s-16.8s: 0.9810907541666666, 0.95027775, 0.99796569
- Mouth-warning candidate interval: 13.964501s to 21.25939s
- High-confidence candidate interval: 16.882456s to 17.924583s

High-confidence candidates can occur after the visible yawn interval because Stage 14 `recent_yawn_event` remains active for the recent-yawn window and can later overlap with eye warning candidates.

## 8. Scenario-Level Interpretation

- `A_normal_open_baseline`: mostly normal, with no high-confidence or mouth-warning frames.
- `B_realistic_drowsy_simulation`: Stage 14 generated high `p_yawn` during the observed yawn interval, and F5 fusion produced mouth/high-confidence candidates when recent yawn overlapped eye state.
- `C_mild_head_motion`: no mouth/yawn false positives were used; signal-quality intervals remain quality markers rather than confirmed drowsiness.
- `D_controlled_long_open_closed`: eye closure produced eye-warning candidates without mouth/yawn escalation.

## 9. Visual Acceptance Note

Stage 14 mouth contact sheets and debug frames were visually accepted as sufficient for Stage 15. High `p_yawn` crops in B corresponded to yawning/open-mouth frames. Some lower-probability or transition frames existed, but they did not materially affect yawn-event detection in the manually observed B interval.

## 10. Limitations

- Small A/B/C/D validation set.
- One or few subjects.
- No final drowsiness ground-truth timeline.
- No trained fusion classifier.
- No real-world deployment validation.
- This is not final system-level drowsiness accuracy.

## 11. Next Step

If Stage 15 behavior is accepted, the project can move to final integration summary and demo planning. A learned fusion classifier should only be considered after collecting synchronized annotated mouth-eye data.
