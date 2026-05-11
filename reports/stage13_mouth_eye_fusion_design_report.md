# Stage 13 Mouth-Eye Fusion Design Report

## 1. Purpose

Stage 13 designs rule-based mouth-eye fusion for the modular drowsiness monitoring system. It is not a trained fusion classifier, not final system-level drowsiness accuracy, and not deployment readiness.

## 2. Mouth/Yawn Runtime Audit Summary

The Stage 13 audit found no existing Python runtime mouth/yawn inference pipeline, no verified local runtime-ready mouth/yawn checkpoint path, no `p_yawn` timelines for A/B/C/D, and no synchronized mouth-eye timelines for A/B/C/D.

A manual mouth/yawn annotation CSV was provided for this run. It marks the user-confirmed yawn interval in `B_realistic_drowsy_simulation.mp4` from 14.3s to 16.8s. This is not runtime mouth/yawn model inference.

## 3. Inputs

- Stage 12 eye alert timelines: `outputs/stage12_eye_alert_rule_analysis/stage12_video_alert_timeline_<slug>.csv`
- Mouth timeline source: manual video review annotation, not model-generated `p_yawn`
- Stage 13 output directory: `outputs/stage13_mouth_eye_fusion_manual_B_yawn_annotation`

## 4. Fusion States

- `normal`: no warning candidate.
- `eye_warning_candidate`: Stage 12 eye-only temporal warning is active.
- `mouth_warning_candidate`: recent mouth/yawn signal is active.
- `high_confidence_drowsiness_candidate`: eye warning and recent yawn co-occur.
- `signal_unreliable`: signal quality is insufficient; this must not be counted as drowsiness.

## 5. Fusion Rules Compared

- F1 eye-only baseline: uses the Stage 12 eye warning and preserves eye signal unreliability.
- F2 mouth-only baseline: uses recent yawn events only.
- F3 OR fusion: emits a warning when either eye or mouth warning is present.
- F4 AND/near-window fusion: emits high-confidence only when eye warning and recent yawn co-occur.
- F5 recommended tiered quality-aware fusion: preserves unreliable eye intervals, supports eye-only and mouth-only warnings, and upgrades only on eye-mouth co-occurrence.

## 6. Rule Comparison Table

| video_slug | rule_name | normal_frames | eye_warning_candidate_frames | mouth_warning_candidate_frames | high_confidence_drowsiness_candidate_frames | signal_unreliable_frames | alert_count | longest_any_warning_run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_normal_open_baseline | F1_eye_only_baseline | 70 | 0 | 0 | 0 | 0 | 0 | 0 |
| A_normal_open_baseline | F2_mouth_only_baseline | 70 | 0 | 0 | 0 | 0 | 0 | 0 |
| A_normal_open_baseline | F3_or_fusion | 70 | 0 | 0 | 0 | 0 | 0 | 0 |
| A_normal_open_baseline | F4_and_near_window_fusion | 70 | 0 | 0 | 0 | 0 | 0 | 0 |
| A_normal_open_baseline | F5_tiered_quality_aware_fusion | 70 | 0 | 0 | 0 | 0 | 0 | 0 |
| B_realistic_drowsy_simulation | F1_eye_only_baseline | 79 | 24 | 0 | 0 | 0 | 3 | 13 |
| B_realistic_drowsy_simulation | F2_mouth_only_baseline | 69 | 0 | 34 | 0 | 0 | 1 | 34 |
| B_realistic_drowsy_simulation | F3_or_fusion | 51 | 18 | 28 | 6 | 0 | 3 | 34 |
| B_realistic_drowsy_simulation | F4_and_near_window_fusion | 97 | 0 | 0 | 6 | 0 | 1 | 6 |
| B_realistic_drowsy_simulation | F5_tiered_quality_aware_fusion | 51 | 18 | 28 | 6 | 0 | 3 | 34 |
| C_mild_head_motion | F1_eye_only_baseline | 76 | 7 | 0 | 0 | 12 | 3 | 3 |
| C_mild_head_motion | F2_mouth_only_baseline | 95 | 0 | 0 | 0 | 0 | 0 | 0 |
| C_mild_head_motion | F3_or_fusion | 76 | 7 | 0 | 0 | 12 | 3 | 3 |
| C_mild_head_motion | F4_and_near_window_fusion | 83 | 0 | 0 | 0 | 12 | 0 | 0 |
| C_mild_head_motion | F5_tiered_quality_aware_fusion | 76 | 7 | 0 | 0 | 12 | 3 | 3 |
| D_controlled_long_open_closed | F1_eye_only_baseline | 54 | 65 | 0 | 0 | 0 | 2 | 36 |
| D_controlled_long_open_closed | F2_mouth_only_baseline | 119 | 0 | 0 | 0 | 0 | 0 | 0 |
| D_controlled_long_open_closed | F3_or_fusion | 54 | 65 | 0 | 0 | 0 | 2 | 36 |
| D_controlled_long_open_closed | F4_and_near_window_fusion | 119 | 0 | 0 | 0 | 0 | 0 | 0 |
| D_controlled_long_open_closed | F5_tiered_quality_aware_fusion | 54 | 65 | 0 | 0 | 0 | 2 | 36 |

## 7. Recommended Rule

Recommended rule: `F5_tiered_quality_aware_fusion`.

This rule is selected because it:

- Preserves `signal_unreliable` instead of treating tracking failure as drowsiness.
- Supports an eye-only warning candidate when the Stage 12 eye rule is active.
- Supports a mouth-only warning candidate when a recent yawn exists and the eye signal is not usable.
- Upgrades to `high_confidence_drowsiness_candidate` only when eye warning and recent yawn co-occur.

Recommended-rule summary:

| video_slug | normal_frames | eye_warning_candidate_frames | mouth_warning_candidate_frames | high_confidence_drowsiness_candidate_frames | signal_unreliable_frames | alert_count | longest_any_warning_run |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A_normal_open_baseline | 70 | 0 | 0 | 0 | 0 | 0 | 0 |
| B_realistic_drowsy_simulation | 51 | 18 | 28 | 6 | 0 | 3 | 34 |
| C_mild_head_motion | 76 | 7 | 0 | 0 | 12 | 3 | 3 |
| D_controlled_long_open_closed | 54 | 65 | 0 | 0 | 0 | 2 | 36 |

## 8. Scenario-Level Interpretation

- `A_normal_open_baseline`: should stay mostly normal in demo mode.
- `B_realistic_drowsy_simulation`: manual yawn annotation creates mouth-warning candidates during 14.3s-16.8s and high-confidence candidates when the eye warning overlaps the recent-yawn window.
- `C_mild_head_motion`: is a mixed fatigue-like eye closure, mild head motion, and partial occlusion scenario; signal-unreliable intervals should remain quality markers rather than drowsiness labels.
- `D_controlled_long_open_closed`: should remain eye-warning driven unless a yawn event is present.

## 9. Limitations

- This manual annotation run is a sanity check only; it is not runtime mouth/yawn detection.
- No automatic `p_yawn` runtime timelines for A/B/C/D were found in this audit.
- No ground-truth drowsiness timeline is used.
- No trained fusion classifier is used.
- The validation set is small.
- This is not final system-level drowsiness accuracy.

## 10. Next Steps

1. Implement real runtime mouth/yawn inference on the same videos.
2. Generate automatic synchronized `p_yawn` timelines.
3. Rerun Stage 13 with real mouth timelines.
4. Optionally collect synchronized labeled data for future fusion classifier extension.

## Machine-Readable Summary

- Stage: 13
- Status: DESIGN_WITH_MANUAL_MOUTH_ANNOTATION
- Recommended rule: `F5_tiered_quality_aware_fusion`
- Uses synthetic mouth timeline: False
- Uses runtime mouth/yawn inference: False
- Warning: This is not runtime mouth/yawn inference and not final system-level drowsiness accuracy.
