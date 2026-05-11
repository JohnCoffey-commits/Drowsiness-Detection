# Stage 17.2 Manual Review Interpretation Report

## 1. Purpose

Stage 17.2 documents manual-review interpretation after the Stage 17.1 sustained-eye gate update. This stage is reporting and interpretation refinement only. It does not retrain models, change checkpoints, change preprocessing, change probability mappings, weaken the sustained-eye gate, or implement webcam detection.

This remains rule-based warning-candidate analysis, not final system-level drowsiness accuracy.

## 2. Manual Review Input

The user manually reviewed selected `C_upload_test.mp4` intervals after Stage 17.1:

- `15.012s-16.680s`: visually fatigue-like, with reduced eye openness and blinking, but no clear sustained full eye closure.
- `18.140s-18.765s`: normal blink; Stage 17.1 suppression from high-confidence is correct.
- `21.267s-21.893s`: a brief true eye-closure event near the end; conservative downgrade is acceptable because it is brief.

## 3. C Interval-by-Interval Interpretation

| Interval | Stage 17.1 state | Manual review label | Interpretation |
|---|---|---|---|
| `15.012s-16.680s` | `high_confidence_drowsiness_candidate` | `weak_match` | Keep as high-confidence warning candidate, but describe it as recent mouth/yawn evidence plus sustained fatigue-like eye-warning evidence. Do not describe it as sustained full eye closure. |
| `18.140s-18.765s` | `mouth_warning_candidate` | `acceptable_after_stage17_1` | Normal blink was correctly suppressed from high-confidence by the sustained-eye gate. Mouth-warning state is acceptable if recent-yawn evidence remains active. |
| `21.267s-21.893s` | `mouth_warning_candidate` | `brief_true_eye_closure_downgraded` | Eye closure is visible but brief. Conservative downgrade is acceptable because it does not satisfy the sustained-eye gate. Do not record this as a false positive. |

## 4. Why 15.012s-16.680s Remains High-Confidence but Only Weak Match

The interval remains a high-confidence warning candidate because the rule-based fusion found recent mouth/yawn evidence overlapping sustained fatigue-like eye-warning evidence. Manual review supports a fatigue-like interpretation: the subject appears visibly tired, with reduced eye openness and blinking.

However, manual review did not confirm a clear sustained full eye closure. Therefore this interval should be treated as a `weak_match` for the intended warning behavior, not a full manual confirmation of sustained eye closure.

Recommended wording:

> High-confidence warning candidate based on recent mouth/yawn evidence and sustained fatigue-like eye-warning evidence. Manual review noted visible fatigue and reduced eye openness, but no clear sustained full eye closure.

## 5. Why 18.140s-18.765s Remains Suppressed

Manual review identified this interval as normal blinking, not sustained eye closure. Stage 17.1 correctly prevented this short blink-like eye-warning interval from escalating to high-confidence while recent-yawn evidence was active.

Recommended wording:

> Mouth-warning candidate. Brief blink-like eye activity was suppressed from high-confidence escalation by the sustained-eye gate.

## 6. Why 21.267s-21.893s Is Not a False Positive

Manual review found visible eye closure near the end of this interval, but the event was brief. The Stage 17.1 sustained-eye gate downgraded it because it did not meet the duration or sampled-frame requirement for high-confidence escalation.

This is acceptable conservative behavior. It should not be recorded as a false positive.

Recommended wording:

> Mouth-warning candidate with brief eye-closure evidence. The eye closure was visible but too brief to satisfy the sustained-eye gate.

## 7. Recommended UI Wording

For `C_upload_test.mp4`, `15.012s-16.680s`:

> High-confidence warning candidate based on recent mouth/yawn evidence and sustained fatigue-like eye-warning evidence. Manual review noted visible fatigue and reduced eye openness, but no clear sustained full eye closure.

For `C_upload_test.mp4`, `18.140s-18.765s`:

> Mouth-warning candidate. Brief blink-like eye activity was suppressed from high-confidence escalation by the sustained-eye gate.

For `C_upload_test.mp4`, `21.267s-21.893s`:

> Mouth-warning candidate with brief eye-closure evidence. The eye closure was visible but too brief to satisfy the sustained-eye gate.

## 8. Limitations

- Manual review is interval-specific and does not create final ground-truth drowsiness labels.
- Eye-warning evidence is model-derived and ROI-dependent.
- Eye-warning evidence may represent reduced eye openness, blinking, partial closure, full closure, fatigue-like appearance, or ROI-sensitive behavior.
- The sustained-eye gate reduces brief blink-like high-confidence escalation but does not prove system-level accuracy.
- This remains a small upload-test validation set.
- This is not deployment readiness, real-world road validation, or a trained fusion classifier.

## 9. Safe Final Conclusion

Stage 17.2 confirms that the Stage 17.1 sustained-eye gate should be interpreted conservatively:

- `15.012s-16.680s` remains a weak-match high-confidence warning candidate based on recent mouth/yawn evidence and sustained fatigue-like eye-warning evidence.
- `18.140s-18.765s` was correctly suppressed from high-confidence as normal blink-like activity.
- `21.267s-21.893s` is a brief true eye-closure event downgraded conservatively, not a false positive.

This is rule-based warning-candidate analysis, not final system-level drowsiness accuracy.

