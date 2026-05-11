# Stage 17.2 Result Interpretation Schema Addendum

## Purpose

This addendum documents interpretation-layer labels for manual review of Stage 17 video-upload results. Stage 17.5 now adds rule-based eye evidence calibration fields to `summary.json`, `fusion_timeline.csv`, intervals, and keyframes. It does not change model inference, checkpoint loading, preprocessing, or the `p_eye_closed` / `p_yawn` class-index formulas.

## Optional Manual Review Fields

These fields may be used in future review spreadsheets, UI annotations, or manual QA reports:

| Field | Type | Meaning |
|---|---|---|
| `manual_review_label` | string | Human interpretation label such as `weak_match`, `acceptable_after_stage17_1`, or `brief_true_eye_closure_downgraded`. |
| `manual_observation` | string | Short visual observation from human review. |
| `interpretation_note` | string | Safe explanation of why the rule-based state should be accepted, downgraded, or treated cautiously. |
| `wording_constraint` | string | Wording that must be avoided or preferred for this interval. |
| `error_prevented` | string | Optional note describing an error avoided by Stage 17.1 gating, such as suppressing normal blink plus recent-yawn from high-confidence escalation. |

## Stage 17.5 Calibration Fields

These generated fields support safer eye-evidence interpretation:

| Field | Type | Meaning |
|---|---|---|
| `eye_evidence_strength` | string | `none`, `weak`, `moderate`, `strong`, or `signal_unreliable`. |
| `eye_evidence_label` | string | Safe display label such as `Weak eye-warning evidence`, `Moderate eye-closure candidate`, or `Strong eye-closure candidate`. |
| `eye_evidence_interpretation` | string | Safe interpretation text explaining that eye evidence may reflect reduced eye openness, blink-like activity, fatigue-like appearance, or ROI-sensitive cases. |
| `eye_strength_gate_passed` | boolean | Whether the current eye-warning interval passed Stage 17.5 strength-aware high-confidence gating. |
| `eye_strength_gate_reason` | string | Rule-based explanation for the strength gate result. |
| `high_confidence_suppressed_by_weak_eye_evidence` | boolean | True when recent-yawn plus sustained eye-warning evidence remained a mouth-warning candidate because Stage 17.5 calibrated eye evidence was weak. |

## Recommended Manual Labels

| Label | Meaning |
|---|---|
| `weak_match` | The rule-based warning state is plausible, but manual review does not fully confirm the strongest visual interpretation. |
| `acceptable_after_stage17_1` | Stage 17.1 gating produced an acceptable conservative state after suppressing high-confidence escalation. |
| `brief_true_eye_closure_downgraded` | A real but brief eye-closure event was conservatively downgraded because it did not satisfy the sustained-eye gate. |

## Interpretation Boundary

`eye_warning_candidate` and `high_confidence_drowsiness_candidate` should not be presented as final drowsiness truth. In particular, eye-warning evidence should not automatically be described as sustained full eye closure. Safer terms include:

- fatigue-like eye-warning evidence
- reduced eye openness
- weak eye-warning evidence
- moderate eye-closure candidate
- strong eye-closure candidate
- blink-like evidence
- brief eye-closure evidence
- signal unreliable

## Warning

This addendum supports manual review and UI wording. It does not establish final system-level drowsiness accuracy.
