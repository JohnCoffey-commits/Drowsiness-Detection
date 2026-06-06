# Stage 17.2 Manual Review Interpretation Notes

## Purpose

Stage 17.2 records manual-review interpretation notes after the Stage 17.1 sustained-eye gate update. It does not change model inference, thresholds, preprocessing, checkpoint loading, or the warning-candidate logic.

The main refinement is interpretation language: eye-warning evidence should not be automatically equated with sustained full eye closure. Depending on the frame and ROI, it may represent full closure, partial closure, reduced eye openness, blink-like events, fatigue-like appearance, or ROI-quality-sensitive cases.

High-confidence warning candidates remain rule-based candidates, not final drowsiness truth.

## Stage 17.1 Gate Reminder

Stage 17.1 requires all of the following before escalating to `high_confidence_drowsiness_candidate`:

- `recent_yawn_event == true`
- `eye_warning_candidate == true`
- `sustained_eye_warning == true`

`sustained_eye_warning` means the current eye-warning interval has either:

- duration `>= 1.0` second, or
- sampled frame count `>= 5`

Brief normal-blink-like eye-warning intervals that overlap recent-yawn evidence are suppressed from high-confidence escalation.

## C Upload Test Manual Review Table

| Item | Video interval | system_state_after_stage17_1 | manual_review_label | manual_observation | interpretation | wording_constraint | error_prevented |
|---|---|---|---|---|---|---|---|
| A | `C_upload_test.mp4`, `15.012s-16.680s` | `high_confidence_drowsiness_candidate` | `weak_match` | Subject appears visibly fatigued, with reduced eye openness and blinking, but no clear sustained full eye closure. | Keep as high-confidence warning candidate, but explain as recent mouth/yawn evidence plus sustained fatigue-like eye-warning evidence. | Do not describe as sustained eye closure detected. | N/A |
| B | `C_upload_test.mp4`, `18.140s-18.765s` | `mouth_warning_candidate` | `acceptable_after_stage17_1` | Normal blinking, not sustained eye closure. | Correctly suppressed from high-confidence by sustained-eye gate. | Use blink-like or normal blink wording, not sustained closure wording. | Normal blink plus recent-yawn no longer escalates to high-confidence. |
| C | `C_upload_test.mp4`, `21.267s-21.893s` | `mouth_warning_candidate` | `brief_true_eye_closure_downgraded` | Eye closure is visible near the end, but it is brief. | Conservative downgrade is acceptable because the event does not satisfy sustained-eye gate. | Do not call this a false positive. | N/A |

## Interpretation Summary

- Stage 17.2 does not change the warning-candidate logic.
- It refines interpretation and reporting language.
- Eye-warning evidence should not be automatically equated with sustained eye closure.
- Eye-warning evidence may represent full closure, partial closure, reduced eye openness, blink-like events, fatigue-like appearance, or ROI-quality-sensitive cases.
- High-confidence warning candidates remain rule-based candidates, not final drowsiness truth.

## Safe Wording

Use:

- drowsiness warning candidate
- high-confidence warning candidate
- eye-warning candidate
- mouth-warning candidate
- fatigue-like eye-warning evidence
- reduced eye openness
- brief eye closure
- signal unreliable

Avoid:

- driver is drowsy
- final drowsiness detected
- sustained eye closure detected, unless manually verified
- final accuracy
- deployment-ready

