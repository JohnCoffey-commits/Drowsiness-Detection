# Stage 17.5 Eye Evidence Calibration Report

## Purpose

Stage 17.5 implements conservative eye evidence calibration for the local uploaded-video warning-candidate pipeline.

The goal is to reduce overstatement of broad `eye_warning_candidate` evidence while preserving weak reduced-eye-openness evidence for manual review.

## Implemented

- Added rule-based `classify_eye_evidence(...)` calibration in `src/runtime/system_video_upload_pipeline.py`.
- Added weak, moderate, and strong eye evidence labels.
- Added a Stage 17.5 strength-aware gate before high-confidence warning candidates are retained.
- Preserved the Stage 17.1 sustained-eye gate.
- Added calibration fields to timeline rows, intervals, summary JSON, keyframes, backend response counts, and SystemUI displays.
- Updated result schema, demo wording, acceptance checklist, and UI interpretation text.
- Added focused non-model tests for threshold classification and strength-gate behavior.

## Boundaries Preserved

- No retraining.
- No checkpoint changes.
- No webcam implementation.
- No replacement of specialist models.
- No change to `p_eye_closed = softmax(logits)[0]`.
- No change to `p_yawn = softmax(logits)[1]`.
- No final system-level drowsiness accuracy claim.

## Calibration Thresholds

| Setting | Value |
|---|---:|
| Weak evidence min `p_eye_closed` | `0.50` |
| Moderate evidence min `p_eye_closed` | `0.70` |
| Strong evidence min `p_eye_closed` | `0.85` |
| Strength gate min interval mean `p_eye_closed` | `0.70` |
| Strength gate min interval max `p_eye_closed` | `0.85` |
| Strength gate min strong frames | `1` |
| Strength gate min moderate-or-strong frames | `2` |

## Safe Interpretation

Stage 17.5 supports terms such as:

- weak eye-warning evidence
- reduced eye openness
- moderate eye-closure candidate
- strong eye-closure candidate
- blink-like activity
- manual review recommended
- rule-based calibration

Permanent warning:

```text
This output is a rule-based drowsiness warning-candidate analysis, not final system-level drowsiness accuracy.
```

## Remaining Limitations

- Calibration thresholds are provisional and rule-based.
- The system still depends on runtime ROI quality and sampled frame timing.
- Weak evidence can still be important for manual review and should not be discarded.
- This is still an uploaded-video MVP, not a webcam system.
