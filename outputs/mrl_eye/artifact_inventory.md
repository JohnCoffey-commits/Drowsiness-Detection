# MRL Eye Local Artifact Inventory

Generated: 2026-04-27T20:34:37

Local output root: `outputs/mrl_eye/`

Completeness verdict: **COMPLETE**

Notes:

- This inventory checks the expected Stage 9 MRL Eye full-training and Stage 9B model-selection artifacts synced into the local repository.
- Checkpoints are present locally but are large binary artifacts and should generally stay out of normal Git history.
- Metrics, reports, figures, and error-analysis images are lightweight enough to commit if desired.

| Artifact group | Expected file | Status | Size | Notes |
| --- | --- | --- | --- | --- |
| Directory structure | outputs/mrl_eye/results | FOUND | directory |  |
| Directory structure | outputs/mrl_eye/reports | FOUND | directory |  |
| Directory structure | outputs/mrl_eye/figures | FOUND | directory |  |
| Directory structure | outputs/mrl_eye/error_analysis | FOUND | directory |  |
| Directory structure | outputs/mrl_eye/checkpoints | FOUND | directory |  |
| Main results | outputs/mrl_eye/results/mrl_eye_initial_results.csv | FOUND | 3.6 KB |  |
| Main results | outputs/mrl_eye/results/mrl_eye_metrics_summary.json | FOUND | 12.9 KB |  |
| Main results | outputs/mrl_eye/reports/mrl_eye_experiment_summary.md | FOUND | 2.1 KB |  |
| Threshold sweeps | outputs/mrl_eye/results/resnet18_val_threshold_sweep.csv | FOUND | 1.6 KB |  |
| Threshold sweeps | outputs/mrl_eye/results/resnet18_test_threshold_sweep.csv | FOUND | 1.6 KB |  |
| Threshold sweeps | outputs/mrl_eye/results/mobilenet_v2_val_threshold_sweep.csv | FOUND | 1.6 KB |  |
| Threshold sweeps | outputs/mrl_eye/results/mobilenet_v2_test_threshold_sweep.csv | FOUND | 1.6 KB |  |
| Threshold sweeps | outputs/mrl_eye/results/efficientnet_b0_val_threshold_sweep.csv | FOUND | 1.6 KB |  |
| Threshold sweeps | outputs/mrl_eye/results/efficientnet_b0_test_threshold_sweep.csv | FOUND | 1.6 KB |  |
| Checkpoints | outputs/mrl_eye/checkpoints/best_resnet18_mrl_eye.pt | FOUND | 42.72 MB | large checkpoint; keep local or use Git LFS |
| Checkpoints | outputs/mrl_eye/checkpoints/best_mobilenet_v2_mrl_eye.pt | FOUND | 8.73 MB | large checkpoint; keep local or use Git LFS |
| Checkpoints | outputs/mrl_eye/checkpoints/best_efficientnet_b0_mrl_eye.pt | FOUND | 15.59 MB | large checkpoint; keep local or use Git LFS |
| Figures | outputs/mrl_eye/figures/resnet18_confusion_matrix.png | FOUND | 34.4 KB |  |
| Figures | outputs/mrl_eye/figures/resnet18_training_curve.png | FOUND | 44.0 KB |  |
| Figures | outputs/mrl_eye/figures/resnet18_pr_curve_closed.png | FOUND | 35.9 KB |  |
| Figures | outputs/mrl_eye/figures/mobilenet_v2_confusion_matrix.png | FOUND | 34.6 KB |  |
| Figures | outputs/mrl_eye/figures/mobilenet_v2_training_curve.png | FOUND | 48.2 KB |  |
| Figures | outputs/mrl_eye/figures/mobilenet_v2_pr_curve_closed.png | FOUND | 36.9 KB |  |
| Figures | outputs/mrl_eye/figures/efficientnet_b0_confusion_matrix.png | FOUND | 34.3 KB |  |
| Figures | outputs/mrl_eye/figures/efficientnet_b0_training_curve.png | FOUND | 46.0 KB |  |
| Figures | outputs/mrl_eye/figures/efficientnet_b0_pr_curve_closed.png | FOUND | 36.9 KB |  |
| Error analysis contact sheets | outputs/mrl_eye/error_analysis/resnet18_false_open_contact_sheet.jpg | FOUND | 118.4 KB |  |
| Error analysis contact sheets | outputs/mrl_eye/error_analysis/resnet18_false_closed_contact_sheet.jpg | FOUND | 120.5 KB |  |
| Error analysis contact sheets | outputs/mrl_eye/error_analysis/mobilenet_v2_false_open_contact_sheet.jpg | FOUND | 119.4 KB |  |
| Error analysis contact sheets | outputs/mrl_eye/error_analysis/mobilenet_v2_false_closed_contact_sheet.jpg | FOUND | 119.6 KB |  |
| Error analysis contact sheets | outputs/mrl_eye/error_analysis/efficientnet_b0_false_open_contact_sheet.jpg | FOUND | 119.4 KB |  |
| Error analysis contact sheets | outputs/mrl_eye/error_analysis/efficientnet_b0_false_closed_contact_sheet.jpg | FOUND | 118.6 KB |  |
