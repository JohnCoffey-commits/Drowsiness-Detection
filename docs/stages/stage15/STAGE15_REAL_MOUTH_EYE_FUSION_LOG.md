# Stage 15 Real Mouth-Eye Fusion Log

## Purpose

Stage 15 performs real synchronized rule-based mouth-eye fusion validation using Stage 12 eye timelines and Stage 14 model-generated `p_yawn` timelines.

This is not synthetic mouth fusion, not manual mouth annotation fusion, and not final system-level drowsiness accuracy.

## Inputs

- Stage 12 eye timelines from `outputs/stage12_eye_alert_rule_analysis/`.
- Stage 14 mouth/yawn timelines from `outputs/stage14_mouth_yawn_runtime_<slug>/runtime_mouth_yawn_predictions.csv`.
- Combined real mouth timeline: `outputs/stage15_real_mouth_eye_fusion/combined_stage14_real_mouth_timeline.csv`.
- Input audit: `docs/archive/audits/stage15_real_mouth_eye_fusion_2026-05-09/stage15_input_audit.md`.

## Run Result

- Status: REAL_SYNCHRONIZED_RULE_BASED_FUSION_VALIDATION_COMPLETED
- Real Stage 14 mouth timelines used: true
- Synthetic mouth timelines used: false
- Manual mouth annotation used: false
- Rule validated: `F5_tiered_quality_aware_fusion`

## B Yawn Interval

- Manual observed yawn interval: 14.3s-16.8s.
- Fusion decisions used Stage 14 model output, not manual labels.
- Stage 14 yawn-event rows in interval: 12/12.
- Mean/min/max `p_yawn` in interval: 0.9810907541666666, 0.95027775, 0.99796569.
- High-confidence candidate interval: 16.882456s to 17.924583s.

## Outputs

- Output directory: `outputs/stage15_real_mouth_eye_fusion`
- Report: `reports/stage15_real_mouth_eye_fusion_validation_report.md`

## Warning

Stage 15 validates rule-based fusion behavior on a small controlled-realistic set. It is not final system-level drowsiness accuracy and not deployment readiness.
