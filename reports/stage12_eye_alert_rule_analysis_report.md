# Stage 12 Eye-Only Alert Rule Analysis Report

## 1. Purpose

Stage 12 designs and compares eye-only temporal alert rules using the completed Stage 10/11 multi-video validation outputs.

It is not mouth/yawn fusion. It is not final system-level drowsiness accuracy. It is not deployment readiness.

## 2. Literature-Inspired Rationale

PERCLOS is commonly described in driver monitoring literature as the percentage of time, over a time window, that the eyes are substantially closed, often more than 80% closed.

This project does not directly measure eyelid aperture percentage. It uses the trained MRL Eye specialist probability `p_eye_closed = softmax(logits)[0]` as a proxy. Therefore Stage 12 uses a **PERCLOS-like** or **PERCLOS-inspired** metric, not standard PERCLOS.

Temporal persistence and rolling windows are needed because single-frame probability spikes can create false warnings. Signal-quality gating is needed because no-face or tracking failures are not drowsiness; they are unreliable signal intervals.

## 3. Inputs

- Multi-video summary: `outputs/stage11_multi_video_validation_summary.csv`
- Stage 10 prefix: `outputs/stage10_eye_roi_consistency_`
- Stage 11 prefix: `outputs/stage11_eye_temporal_analysis_`
- Videos: `A_normal_open_baseline`, `B_realistic_drowsy_simulation`, `C_mild_head_motion` (mixed fatigue-like eye closure, mild head motion, and partial occlusion), `D_controlled_long_open_closed`

## 4. Rules Compared

- Rule 1: Rolling mean probability, `rolling_mean_p_eye_closed >= threshold` for at least `2` sampled frames.
- Rule 2: Rolling PERCLOS-like mean-binary ratio, `rolling_perclos_mean_binary >= threshold` for at least `2` sampled frames.
- Rule 3: Rolling PERCLOS-like both-eyes ratio, `rolling_perclos_both_eyes >= threshold` for at least `2` sampled frames.
- Rule 4: Candidate closure event duration, event duration >= 3, 5, or 8 sampled frames.
- Rule 5: Quality-gated rolling PERCLOS-like mean-binary ratio. If recent no-face ratio over `5` sampled frames is greater than `0.20`, the frame/window is marked `signal_unreliable`; otherwise the rolling PERCLOS-like threshold is applied.

## 5. Rule Comparison Table

| rule_name | video_slug | alert_count | total_alert_frames | longest_alert_run | signal_unreliable_frames | scenario_expectation_match |
| --- | --- | --- | --- | --- | --- | --- |
| rolling_mean_prob_ge_0.50_consec | A_normal_open_baseline | 0 | 0 | 0 | 0 | True |
| rolling_mean_prob_ge_0.60_consec | A_normal_open_baseline | 0 | 0 | 0 | 0 | True |
| rolling_mean_prob_ge_0.70_consec | A_normal_open_baseline | 0 | 0 | 0 | 0 | True |
| rolling_perclos_mean_ge_0.40_consec | A_normal_open_baseline | 0 | 0 | 0 | 0 | True |
| rolling_perclos_mean_ge_0.50_consec | A_normal_open_baseline | 0 | 0 | 0 | 0 | True |
| rolling_perclos_mean_ge_0.60_consec | A_normal_open_baseline | 0 | 0 | 0 | 0 | True |
| rolling_perclos_mean_ge_0.70_consec | A_normal_open_baseline | 0 | 0 | 0 | 0 | True |
| rolling_perclos_both_ge_0.40_consec | A_normal_open_baseline | 0 | 0 | 0 | 0 | True |
| rolling_perclos_both_ge_0.50_consec | A_normal_open_baseline | 0 | 0 | 0 | 0 | True |
| rolling_perclos_both_ge_0.60_consec | A_normal_open_baseline | 0 | 0 | 0 | 0 | True |
| rolling_perclos_both_ge_0.70_consec | A_normal_open_baseline | 0 | 0 | 0 | 0 | True |
| candidate_event_duration_ge_3 | A_normal_open_baseline | 0 | 0 | 0 | 0 | True |
| candidate_event_duration_ge_5 | A_normal_open_baseline | 0 | 0 | 0 | 0 | True |
| candidate_event_duration_ge_8 | A_normal_open_baseline | 0 | 0 | 0 | 0 | True |
| quality_gated_perclos_mean_ge_0.50_consec | A_normal_open_baseline | 0 | 0 | 0 | 0 | True |
| quality_gated_perclos_mean_ge_0.60_consec | A_normal_open_baseline | 0 | 0 | 0 | 0 | True |
| quality_gated_perclos_mean_ge_0.70_consec | A_normal_open_baseline | 0 | 0 | 0 | 0 | True |
| rolling_mean_prob_ge_0.50_consec | B_realistic_drowsy_simulation | 3 | 17 | 10 | 0 | True |
| rolling_mean_prob_ge_0.60_consec | B_realistic_drowsy_simulation | 1 | 2 | 2 | 0 | True |
| rolling_mean_prob_ge_0.70_consec | B_realistic_drowsy_simulation | 0 | 0 | 0 | 0 | False |
| rolling_perclos_mean_ge_0.40_consec | B_realistic_drowsy_simulation | 4 | 35 | 15 | 0 | True |
| rolling_perclos_mean_ge_0.50_consec | B_realistic_drowsy_simulation | 3 | 24 | 13 | 0 | True |
| rolling_perclos_mean_ge_0.60_consec | B_realistic_drowsy_simulation | 3 | 24 | 13 | 0 | True |
| rolling_perclos_mean_ge_0.70_consec | B_realistic_drowsy_simulation | 3 | 17 | 10 | 0 | True |
| rolling_perclos_both_ge_0.40_consec | B_realistic_drowsy_simulation | 2 | 17 | 12 | 0 | True |
| rolling_perclos_both_ge_0.50_consec | B_realistic_drowsy_simulation | 1 | 8 | 8 | 0 | True |
| rolling_perclos_both_ge_0.60_consec | B_realistic_drowsy_simulation | 1 | 8 | 8 | 0 | True |
| rolling_perclos_both_ge_0.70_consec | B_realistic_drowsy_simulation | 1 | 3 | 3 | 0 | True |
| candidate_event_duration_ge_3 | B_realistic_drowsy_simulation | 5 | 20 | 5 | 0 | True |
| candidate_event_duration_ge_5 | B_realistic_drowsy_simulation | 1 | 5 | 5 | 0 | True |
| candidate_event_duration_ge_8 | B_realistic_drowsy_simulation | 0 | 0 | 0 | 0 | False |
| quality_gated_perclos_mean_ge_0.50_consec | B_realistic_drowsy_simulation | 3 | 24 | 13 | 0 | True |
| quality_gated_perclos_mean_ge_0.60_consec | B_realistic_drowsy_simulation | 3 | 24 | 13 | 0 | True |
| quality_gated_perclos_mean_ge_0.70_consec | B_realistic_drowsy_simulation | 3 | 17 | 10 | 0 | True |
| rolling_mean_prob_ge_0.50_consec | C_mild_head_motion | 2 | 5 | 3 | 12 | False |
| rolling_mean_prob_ge_0.60_consec | C_mild_head_motion | 0 | 0 | 0 | 12 | False |
| rolling_mean_prob_ge_0.70_consec | C_mild_head_motion | 0 | 0 | 0 | 12 | False |
| rolling_perclos_mean_ge_0.40_consec | C_mild_head_motion | 7 | 24 | 5 | 12 | False |
| rolling_perclos_mean_ge_0.50_consec | C_mild_head_motion | 5 | 13 | 3 | 12 | False |
| rolling_perclos_mean_ge_0.60_consec | C_mild_head_motion | 5 | 13 | 3 | 12 | False |
| rolling_perclos_mean_ge_0.70_consec | C_mild_head_motion | 3 | 6 | 2 | 12 | False |
| rolling_perclos_both_ge_0.40_consec | C_mild_head_motion | 5 | 17 | 5 | 12 | False |
| rolling_perclos_both_ge_0.50_consec | C_mild_head_motion | 2 | 4 | 2 | 12 | False |
| rolling_perclos_both_ge_0.60_consec | C_mild_head_motion | 2 | 4 | 2 | 12 | False |
| rolling_perclos_both_ge_0.70_consec | C_mild_head_motion | 0 | 0 | 0 | 12 | False |
| candidate_event_duration_ge_3 | C_mild_head_motion | 5 | 13 | 5 | 12 | False |
| candidate_event_duration_ge_5 | C_mild_head_motion | 2 | 6 | 5 | 12 | False |
| candidate_event_duration_ge_8 | C_mild_head_motion | 0 | 0 | 0 | 12 | False |
| quality_gated_perclos_mean_ge_0.50_consec | C_mild_head_motion | 3 | 7 | 3 | 12 | True |
| quality_gated_perclos_mean_ge_0.60_consec | C_mild_head_motion | 3 | 7 | 3 | 12 | True |
| quality_gated_perclos_mean_ge_0.70_consec | C_mild_head_motion | 1 | 2 | 2 | 12 | True |
| rolling_mean_prob_ge_0.50_consec | D_controlled_long_open_closed | 2 | 64 | 36 | 0 | True |
| rolling_mean_prob_ge_0.60_consec | D_controlled_long_open_closed | 2 | 61 | 35 | 0 | True |
| rolling_mean_prob_ge_0.70_consec | D_controlled_long_open_closed | 5 | 51 | 25 | 0 | True |
| rolling_perclos_mean_ge_0.40_consec | D_controlled_long_open_closed | 2 | 68 | 37 | 0 | True |
| rolling_perclos_mean_ge_0.50_consec | D_controlled_long_open_closed | 2 | 65 | 36 | 0 | True |
| rolling_perclos_mean_ge_0.60_consec | D_controlled_long_open_closed | 2 | 65 | 36 | 0 | True |
| rolling_perclos_mean_ge_0.70_consec | D_controlled_long_open_closed | 2 | 62 | 35 | 0 | True |
| rolling_perclos_both_ge_0.40_consec | D_controlled_long_open_closed | 2 | 68 | 37 | 0 | True |
| rolling_perclos_both_ge_0.50_consec | D_controlled_long_open_closed | 2 | 65 | 36 | 0 | True |
| rolling_perclos_both_ge_0.60_consec | D_controlled_long_open_closed | 2 | 65 | 36 | 0 | True |
| rolling_perclos_both_ge_0.70_consec | D_controlled_long_open_closed | 2 | 62 | 35 | 0 | True |
| candidate_event_duration_ge_3 | D_controlled_long_open_closed | 2 | 67 | 38 | 0 | True |
| candidate_event_duration_ge_5 | D_controlled_long_open_closed | 2 | 67 | 38 | 0 | True |
| candidate_event_duration_ge_8 | D_controlled_long_open_closed | 2 | 67 | 38 | 0 | True |
| quality_gated_perclos_mean_ge_0.50_consec | D_controlled_long_open_closed | 2 | 65 | 36 | 0 | True |
| quality_gated_perclos_mean_ge_0.60_consec | D_controlled_long_open_closed | 2 | 65 | 36 | 0 | True |
| quality_gated_perclos_mean_ge_0.70_consec | D_controlled_long_open_closed | 2 | 62 | 35 | 0 | True |

## 6. Recommended Rule

Recommended rule:

```text
quality_gated_perclos_mean_ge_0.60_consec
```

Parameters:

```json
{
  "rule_family": "quality_gated_rolling_perclos_mean",
  "rule_threshold": 0.6,
  "rule_min_duration": null,
  "uses_quality_gate": true
}
```

Recommended rule behavior:

| video_slug | alert_count | total_alert_frames | longest_alert_run | signal_unreliable_frames | false_warning_on_A_baseline | detected_B_drowsy_simulation | handled_C_quality_issue | detected_D_long_closure | scenario_expectation_match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_normal_open_baseline | 0 | 0 | 0 | 0 | False | False | False | False | True |
| B_realistic_drowsy_simulation | 3 | 24 | 13 | 0 | False | True | False | False | True |
| C_mild_head_motion | 3 | 7 | 3 | 12 | False | False | True | False | True |
| D_controlled_long_open_closed | 2 | 65 | 36 | 0 | False | False | False | True | True |

Selection rationale:

- A's short false event was suppressed: `True`.
- B produced expected warning candidates: `True`.
- C's no-face rows were handled as signal quality issues: `True`.
- D produced expected long warning candidates: `True`.
- The rule is simple to explain: quality-gated rolling PERCLOS-like mean-binary ratio with persistence.


## Human Review Note for C_mild_head_motion

`C_mild_head_motion` is a mixed fatigue/head-motion/occlusion scenario, not a pure normal-open robustness negative. The correct interpretation is not "C should have zero alerts"; the correct interpretation is that visible closure should produce warning candidates, while occlusion/no-face should be treated as signal quality issues.

User-provided temporal interpretation:

- Around 3.0-4.8s: simulated fatigue eye closure with head movement. Near 4.8s the head moves downward and the front-facing view cannot see the eyes clearly; mostly hair is visible.
- Around 6.0-7.1s: short eye closure, not fully closed but longer than a normal blink, plus hand/hair movement that partially hides the eyes.
- Around 9.8-10.8s: eyes gradually close, head moves downward, then a simulated sudden awakening/head-up behavior occurs.
- Around 10.8-12.4s: eyes are open, although the user still looks somewhat tired. This is ambiguous and should not necessarily be expected to trigger an eye-closure alert.
- Around 12.6-14.2s: another fatigue-like eye closure/head-down/sudden-awakening sequence similar to 9.8-10.8s.
- Final stage: left-right head movement.

Short recommended alert markers in `C_mild_head_motion` are plausible because the user intentionally simulated fatigue-like eye closure in several segments. `signal_unreliable` markers are also plausible because parts of the video contain head-down motion, hair/hand occlusion, or eyes not visible from the front.

The Stage 12 recommended quality-gated rule remains appropriate because it handles no-face/tracking failure as signal quality rather than drowsiness. This still does not prove final system-level drowsiness accuracy.

## 7. Limitations

- Small validation set.
- One/few subjects.
- No ground-truth temporal annotation.
- No mouth/yawn fusion yet.
- No live webcam validation.
- Not final drowsiness accuracy.
- PERCLOS-like proxy is based on CNN probability, not true eyelid aperture percentage.

## 8. Next Step

If the recommended eye-only rule behaves correctly after human review of the generated timelines and figures, proceed to Stage 13 mouth-eye fusion design. Otherwise adjust thresholds/windowing and rerun Stage 12.

## Artifact Paths

- Rule comparison CSV: `outputs/stage12_eye_alert_rule_analysis/stage12_rule_comparison.csv`
- Summary JSON: `outputs/stage12_eye_alert_rule_analysis/stage12_eye_alert_summary.json`
- Output report: `outputs/stage12_eye_alert_rule_analysis/STAGE12_EYE_ALERT_RULE_REPORT.md`
- Timeline CSVs: outputs/stage12_eye_alert_rule_analysis/stage12_video_alert_timeline_A_normal_open_baseline.csv, outputs/stage12_eye_alert_rule_analysis/stage12_video_alert_timeline_B_realistic_drowsy_simulation.csv, outputs/stage12_eye_alert_rule_analysis/stage12_video_alert_timeline_C_mild_head_motion.csv, outputs/stage12_eye_alert_rule_analysis/stage12_video_alert_timeline_D_controlled_long_open_closed.csv
- Figures: outputs/stage12_eye_alert_rule_analysis/figures/alert_rule_comparison_by_video.png, outputs/stage12_eye_alert_rule_analysis/figures/alert_timeline_A_normal_open_baseline.png, outputs/stage12_eye_alert_rule_analysis/figures/alert_timeline_B_realistic_drowsy_simulation.png, outputs/stage12_eye_alert_rule_analysis/figures/alert_timeline_C_mild_head_motion.png, outputs/stage12_eye_alert_rule_analysis/figures/alert_timeline_D_controlled_long_open_closed.png

This report is eye-only alert rule design. It is not final system-level drowsiness accuracy.
