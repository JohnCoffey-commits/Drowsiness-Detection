# Stage 17.5 Eye Evidence Calibration and Interpretation Refinement

## Purpose

Stage 17.5 adds a conservative rule-based interpretation layer for eye evidence strength in the local video-upload warning-candidate MVP.

It does not retrain any model, modify checkpoints, replace specialist models, implement webcam behavior, or change the underlying probability formulas:

- MRL Eye: `p_eye_closed = softmax(logits)[0]`
- Mouth/yawn: `p_yawn = softmax(logits)[1]`

Permanent boundary:

```text
This output is a rule-based drowsiness warning-candidate analysis, not final system-level drowsiness accuracy.
```

## Calibration Thresholds

The thresholds are provisional interpretation-calibration thresholds, not final accuracy thresholds.

| Field | Value | Meaning |
|---|---:|---|
| `EYE_EVIDENCE_WEAK_MIN` | `0.50` | Weak eye-warning evidence / reduced eye openness range. |
| `EYE_EVIDENCE_MODERATE_MIN` | `0.70` | Moderate eye-closure candidate range. |
| `EYE_EVIDENCE_STRONG_MIN` | `0.85` | Strong eye-closure candidate range. |

## Strength-Aware High-Confidence Gate

Stage 17.1 sustained-eye gating remains required.

Stage 17.5 adds a second gate before high-confidence warning candidates are retained. The eye-warning interval must satisfy at least one of:

- interval mean `p_eye_closed >= 0.70`
- interval max `p_eye_closed >= 0.85`
- at least `1` strong eye-closure candidate frame
- at least `2` moderate-or-strong eye evidence frames

If recent-yawn and sustained eye-warning evidence overlap but calibrated eye evidence remains weak, the output is conservatively kept as a `mouth_warning_candidate` and marked with `high_confidence_suppressed_by_weak_eye_evidence`.

## Output Fields Added

Timeline, intervals, keyframes, and summary outputs can now include:

- `eye_evidence_strength`
- `eye_evidence_label`
- `eye_evidence_interpretation`
- `eye_strength_gate_passed`
- `eye_strength_gate_reason`
- `eye_strength_interval_mean_p_eye_closed`
- `eye_strength_interval_max_p_eye_closed`
- `eye_strength_interval_strong_frame_count`
- `eye_strength_interval_moderate_or_strong_frame_count`
- `high_confidence_suppressed_by_weak_eye_evidence`
- `suppressed_high_confidence_weak_eye_evidence_frames`
- `weak_eye_warning_evidence_frames`
- `moderate_eye_closure_candidate_frames`
- `strong_eye_closure_candidate_frames`
- `eye_evidence_strength_counts`

## Interpretation Guidance

Use:

- weak eye-warning evidence
- reduced eye openness
- moderate eye-closure candidate
- strong eye-closure candidate
- blink-like activity
- fatigue-like appearance
- ROI-sensitive cases
- manual review recommended

Avoid presenting calibrated eye evidence as final system-level drowsiness truth.

## Known Motivation

The previous `eye_warning_candidate` state was intentionally broad. It could mix:

- true eye closure
- reduced eye openness
- smiling or squinting
- blink-like activity
- fatigue-like small eye opening
- uncertain ROI, angle, or lighting effects

Stage 17.5 keeps that broad signal available but adds safer strength labels so weak evidence is not overstated as high-confidence evidence.
