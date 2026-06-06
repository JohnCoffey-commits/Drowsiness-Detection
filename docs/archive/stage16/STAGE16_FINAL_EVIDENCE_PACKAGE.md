# Stage 16 Final Evidence Package

This checklist identifies the main evidence files for the final project summary. The project remains a controlled-validation prototype and does not claim final system-level drowsiness accuracy.

## A. Core Reports

| Priority | Path | Purpose |
| --- | --- | --- |
| Required | `reports/stage16_final_integration_summary_report.md` | Final high-level integration summary and claim boundaries. |
| Required | `reports/stage15_real_mouth_eye_fusion_validation_report.md` | Real synchronized rule-based mouth-eye fusion validation report. |
| Required | `reports/stage14_mouth_yawn_runtime_validation_report.md` | Runtime mouth/yawn inference and ROI validation report. |
| Required | `reports/stage12_eye_alert_rule_analysis_report.md` | Eye-only alert rule comparison and recommendation report. |
| Recommended | `reports/stage10_runtime_eye_roi_acceptance_report.md` | Controlled-video eye ROI acceptance evidence. |
| Recommended | `reports/stage11_multi_video_temporal_validation_report.md` | Multi-video eye temporal validation report. |
| Recommended | `reports/stage13_mouth_eye_fusion_design_report.md` | Fusion design/prototype report before real mouth timeline was available. |

## B. Runtime Scripts

| Priority | Path | Purpose |
| --- | --- | --- |
| Required | `src/runtime/stage10_eye_roi_consistency.py` | Runtime eye ROI consistency test. |
| Required | `src/runtime/stage11_eye_temporal_analysis.py` | Eye-only temporal analysis. |
| Required | `src/runtime/stage12_eye_alert_rule_analysis.py` | Eye-only alert rule comparison. |
| Required | `src/runtime/stage13_mouth_eye_fusion_design.py` | Rule-based mouth-eye fusion design and reusable fusion logic. |
| Required | `src/runtime/stage14_mouth_yawn_runtime.py` | Runtime mouth ROI and mouth/yawn inference. |
| Required | `src/runtime/stage15_real_mouth_eye_fusion_validation.py` | Real synchronized rule-based mouth-eye fusion validation wrapper. |

## C. Checkpoint and Model Evidence

| Priority | Path | Purpose |
| --- | --- | --- |
| Required | `artifacts/audits/stage14_mouth_yawn_runtime_2026-05-09/STAGE14_RECOVERED_CHECKPOINT_VERIFICATION.md` | Verifies the recovered ResNet18 mouth/yawn checkpoint loads with a two-class head. |
| Required | `artifacts/audits/stage14_mouth_yawn_runtime_2026-05-09/STAGE14_CHECKPOINT_LOCAL_COPY.md` | Records local copy of recovered mouth/yawn checkpoint. |
| Required | `checkpoints/resnet18_best.pt` | Local recovered Stage 7 ResNet18 mouth/yawn checkpoint. Checkpoint binaries should not be committed to normal Git. |
| Required | `outputs/mrl_eye/checkpoints/best_mobilenet_v2_mrl_eye.pt` | Selected MRL Eye MobileNetV2 checkpoint. Checkpoint binaries should not be committed to normal Git. |
| Recommended | `outputs/mrl_eye/results/mrl_eye_stage9b_model_selection.json` | Machine-readable eye model selection evidence. |

## D. Stage 12 Eye Evidence

| Priority | Path | Purpose |
| --- | --- | --- |
| Required | `outputs/stage12_eye_alert_rule_analysis/stage12_eye_alert_summary.json` | Stage 12 selected eye-rule summary. |
| Required | `outputs/stage12_eye_alert_rule_analysis/stage12_rule_comparison.csv` | Eye-rule comparison across validation videos. |
| Required | `outputs/stage12_eye_alert_rule_analysis/stage12_video_alert_timeline_B_realistic_drowsy_simulation.csv` | B scenario eye alert timeline used by Stage 15. |
| Recommended | `outputs/stage12_eye_alert_rule_analysis/figures/` | Eye-only alert rule plots. |

## E. Stage 14 Mouth Evidence

| Priority | Path | Purpose |
| --- | --- | --- |
| Required | `outputs/stage14_mouth_yawn_runtime_B_realistic_drowsy_simulation/runtime_mouth_yawn_predictions.csv` | Model-generated `p_yawn` timeline for B. |
| Required | `outputs/stage14_mouth_yawn_runtime_multi_video_summary.csv` | Stage 14 multi-video mouth/yawn summary. |
| Required | `outputs/stage14_mouth_yawn_runtime_B_realistic_drowsy_simulation/summary.json` | B-specific Stage 14 summary. |
| Recommended | `outputs/stage14_mouth_yawn_runtime_B_realistic_drowsy_simulation/contact_sheets/` | Visual mouth ROI and high/low `p_yawn` evidence for B. |
| Recommended | `outputs/stage14_mouth_yawn_runtime_B_realistic_drowsy_simulation/debug_frames/` | Mouth bbox overlays for B. |

## F. Stage 15 Fusion Evidence

| Priority | Path | Purpose |
| --- | --- | --- |
| Required | `outputs/stage15_real_mouth_eye_fusion/combined_stage14_real_mouth_timeline.csv` | Stage 14 model-generated mouth timelines combined for Stage 15. |
| Required | `outputs/stage15_real_mouth_eye_fusion/stage15_real_fusion_summary.json` | Machine-readable Stage 15 result. |
| Required | `outputs/stage15_real_mouth_eye_fusion/stage15_real_fusion_rule_comparison.csv` | Stage 15 fusion rule comparison. |
| Required | `outputs/stage15_real_mouth_eye_fusion/timelines/fusion_timeline_B_realistic_drowsy_simulation.csv` | B fusion timeline showing mouth and eye interaction. |
| Required | `artifacts/audits/stage15_real_mouth_eye_fusion_2026-05-09/stage15_input_audit.md` | Confirms real Stage 14 mouth timelines were used, not synthetic/manual mouth timelines. |
| Required | `artifacts/audits/stage16_final_integration_2026-05-09/STAGE15_FIGURE_TITLE_FIX.md` | Documents Stage 15 figure-title correction. |

## G. Figures to Include in Final Presentation or Report

| Priority | Path | Purpose |
| --- | --- | --- |
| Required | `outputs/stage15_real_mouth_eye_fusion/figures/fusion_timeline_B_realistic_drowsy_simulation.png` | Shows B real mouth-eye fusion behavior. |
| Required | `outputs/stage15_real_mouth_eye_fusion/figures/fusion_state_counts_by_video.png` | Summarizes Stage 15 fusion states across A/B/C/D. |
| Recommended | `outputs/stage15_real_mouth_eye_fusion/figures/fusion_timeline_A_normal_open_baseline.png` | Baseline mostly-normal evidence. |
| Recommended | `outputs/stage15_real_mouth_eye_fusion/figures/fusion_timeline_C_mild_head_motion.png` | Mixed fatigue/head-motion/occlusion evidence. |
| Recommended | `outputs/stage15_real_mouth_eye_fusion/figures/fusion_timeline_D_controlled_long_open_closed.png` | Eye-warning-only reference evidence. |
| Recommended | `outputs/stage14_mouth_yawn_runtime_B_realistic_drowsy_simulation/figures/p_yawn_over_time.png` | B mouth/yawn model output over time. |

## H. Known Limitations

- Not final system-level drowsiness accuracy.
- Not deployment-ready.
- Not a trained fusion classifier.
- Small validation set.
- One or few subjects.
- No final ground-truth drowsiness timeline.
- Controlled-realistic videos are useful evidence but not real-world road validation.
- `high_confidence_drowsiness_candidate` is a rule-based warning-candidate state, not a clinical or final safety label.

## I. Recommended Final Wording

Use:

- "The system produces rule-based drowsiness warning candidates."
- "Stage 15 validates synchronized fusion behavior on a small controlled-realistic validation set."
- "The B yawning interval was detected by the runtime mouth/yawn model and contributed to high-confidence candidate state when paired with eye-warning evidence."
- "The project is ready for final integration/demo planning, not deployment."

Avoid:

- "Final drowsiness accuracy."
- "Deployment-ready driver monitoring."
- "Clinically validated."
- "Trained fusion classifier."
- "Real-world road validation."
