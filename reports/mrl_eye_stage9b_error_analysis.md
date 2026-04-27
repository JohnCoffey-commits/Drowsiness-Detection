# MRL Eye Stage 9B Error Analysis and Model Selection

## 1. Purpose

This report finalizes model selection for the MRL Eye eye open/closed specialist module using the complete local `outputs/mrl_eye/` artifacts synced from Colab.

These are specialist eye-state classification results on MRL Eye. They are not final system-level driver drowsiness accuracy.

Definitions used here:

- `closed = 0`
- `open = 1`
- `false_open`: true closed, predicted open. This is safety-critical because a closed-eye frame is missed.
- `false_closed`: true open, predicted closed. This is a false alarm tendency.

## 2. Files Inspected and Output Completeness

Source root: `outputs/mrl_eye/`

Completeness verdict: **COMPLETE**

| Status | Path | Purpose | Size |
| --- | --- | --- | --- |
| OK | outputs/mrl_eye/results | Directory structure | directory |
| OK | outputs/mrl_eye/reports | Directory structure | directory |
| OK | outputs/mrl_eye/figures | Directory structure | directory |
| OK | outputs/mrl_eye/error_analysis | Directory structure | directory |
| OK | outputs/mrl_eye/checkpoints | Directory structure | directory |
| OK | outputs/mrl_eye/results/mrl_eye_initial_results.csv | Main results | 3.6 KB |
| OK | outputs/mrl_eye/results/mrl_eye_metrics_summary.json | Main results | 12.9 KB |
| OK | outputs/mrl_eye/reports/mrl_eye_experiment_summary.md | Main results | 2.1 KB |
| OK | outputs/mrl_eye/results/resnet18_val_threshold_sweep.csv | Threshold sweeps | 1.6 KB |
| OK | outputs/mrl_eye/results/resnet18_test_threshold_sweep.csv | Threshold sweeps | 1.6 KB |
| OK | outputs/mrl_eye/results/mobilenet_v2_val_threshold_sweep.csv | Threshold sweeps | 1.6 KB |
| OK | outputs/mrl_eye/results/mobilenet_v2_test_threshold_sweep.csv | Threshold sweeps | 1.6 KB |
| OK | outputs/mrl_eye/results/efficientnet_b0_val_threshold_sweep.csv | Threshold sweeps | 1.6 KB |
| OK | outputs/mrl_eye/results/efficientnet_b0_test_threshold_sweep.csv | Threshold sweeps | 1.6 KB |
| OK | outputs/mrl_eye/checkpoints/best_resnet18_mrl_eye.pt | Checkpoints | 42.72 MB |
| OK | outputs/mrl_eye/checkpoints/best_mobilenet_v2_mrl_eye.pt | Checkpoints | 8.73 MB |
| OK | outputs/mrl_eye/checkpoints/best_efficientnet_b0_mrl_eye.pt | Checkpoints | 15.59 MB |
| OK | outputs/mrl_eye/figures/resnet18_confusion_matrix.png | Figures | 34.4 KB |
| OK | outputs/mrl_eye/figures/resnet18_training_curve.png | Figures | 44.0 KB |
| OK | outputs/mrl_eye/figures/resnet18_pr_curve_closed.png | Figures | 35.9 KB |
| OK | outputs/mrl_eye/figures/mobilenet_v2_confusion_matrix.png | Figures | 34.6 KB |
| OK | outputs/mrl_eye/figures/mobilenet_v2_training_curve.png | Figures | 48.2 KB |
| OK | outputs/mrl_eye/figures/mobilenet_v2_pr_curve_closed.png | Figures | 36.9 KB |
| OK | outputs/mrl_eye/figures/efficientnet_b0_confusion_matrix.png | Figures | 34.3 KB |
| OK | outputs/mrl_eye/figures/efficientnet_b0_training_curve.png | Figures | 46.0 KB |
| OK | outputs/mrl_eye/figures/efficientnet_b0_pr_curve_closed.png | Figures | 36.9 KB |
| OK | outputs/mrl_eye/error_analysis/resnet18_false_open_contact_sheet.jpg | Error analysis contact sheets | 118.4 KB |
| OK | outputs/mrl_eye/error_analysis/resnet18_false_closed_contact_sheet.jpg | Error analysis contact sheets | 120.5 KB |
| OK | outputs/mrl_eye/error_analysis/mobilenet_v2_false_open_contact_sheet.jpg | Error analysis contact sheets | 119.4 KB |
| OK | outputs/mrl_eye/error_analysis/mobilenet_v2_false_closed_contact_sheet.jpg | Error analysis contact sheets | 119.6 KB |
| OK | outputs/mrl_eye/error_analysis/efficientnet_b0_false_open_contact_sheet.jpg | Error analysis contact sheets | 119.4 KB |
| OK | outputs/mrl_eye/error_analysis/efficientnet_b0_false_closed_contact_sheet.jpg | Error analysis contact sheets | 118.6 KB |

## 3. Stage 9 Result Summary

Main source file: `outputs/mrl_eye/results/mrl_eye_initial_results.csv`

All three models report `pretrained_loaded = True`.

| Model | Train Acc | Val Acc | Test Acc | Test Macro F1 | Closed Recall | False Open | False Closed | Val-Selected Threshold | Fixed-Thresh Macro F1 | Fixed-Thresh Closed Recall | Fixed False Open | Fixed False Closed | Pretrained |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet18 | 99.16% | 98.37% | 98.46% | 98.46% | 98.59% | 89 | 109 | 0.30 | 97.60% | 99.08% | 58 | 251 | True |
| MobileNetV2 | 99.33% | 97.91% | 98.63% | 98.63% | 98.52% | 93 | 84 | 0.30 | 98.48% | 98.79% | 76 | 120 | True |
| EfficientNet-B0 | 99.44% | 97.91% | 98.62% | 98.62% | 98.24% | 111 | 67 | 0.30 | 98.52% | 98.65% | 85 | 106 | True |

## 4. Model Comparison

MobileNetV2 has the best default test accuracy and default test macro F1 while also being the strongest real-time deployment candidate. Its default metrics are:

- test accuracy: 98.63%
- test macro F1: 98.63%
- closed-eye recall: 98.52%
- false-open count: 93
- false-closed count: 84

ResNet18 has slightly stronger default closed-eye recall than MobileNetV2 and gives the most conservative validation-threshold option. However, its default test accuracy/macro F1 are lower and it is less attractive for real-time use.

EfficientNet-B0 is very close to MobileNetV2 on default test accuracy/macro F1, but it has lower default closed-eye recall and more default false-open errors.

## 5. Threshold Analysis

Threshold candidates are selected from validation sweeps only. The test set is used only for final reporting at the validation-selected threshold.

| Model | Selected From Val | Test Macro F1 @0.50 | Closed Recall @0.50 | False Open @0.50 | False Closed @0.50 | Test Macro F1 @Selected | Closed Recall @Selected | False Open @Selected | False Closed @Selected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet18 | 0.30 | 98.46% | 98.59% | 89 | 109 | 97.60% | 99.08% | 58 | 251 |
| MobileNetV2 | 0.30 | 98.63% | 98.52% | 93 | 84 | 98.48% | 98.79% | 76 | 120 |
| EfficientNet-B0 | 0.30 | 98.62% | 98.24% | 111 | 67 | 98.52% | 98.65% | 85 | 106 |

The validation-selected threshold is `0.30` for all three models. Lowering the closed-eye threshold from `0.50` to `0.30` makes predictions more conservative: it reduces false-open errors and improves closed-eye recall, but it increases false-closed errors.

For MobileNetV2, `0.30` reduces false-open errors from 93 to 76 and improves closed-eye recall from 98.52% to 98.79%, but false-closed errors increase from 84 to 120 and macro F1 decreases slightly. That makes it a reasonable safety-prioritized option, not the default.

For ResNet18, `0.30` gives the strongest safety-prioritized reference: false-open errors drop from 89 to 58 and closed-eye recall rises from 98.59% to 99.08%. The cost is a large increase in false-closed errors from 109 to 251 and a lower macro F1.

## 6. Error Contact Sheet Review

The six local contact sheets were visually inspected:

- `outputs/mrl_eye/error_analysis/resnet18_false_open_contact_sheet.jpg`
- `outputs/mrl_eye/error_analysis/resnet18_false_closed_contact_sheet.jpg`
- `outputs/mrl_eye/error_analysis/mobilenet_v2_false_open_contact_sheet.jpg`
- `outputs/mrl_eye/error_analysis/mobilenet_v2_false_closed_contact_sheet.jpg`
- `outputs/mrl_eye/error_analysis/efficientnet_b0_false_open_contact_sheet.jpg`
- `outputs/mrl_eye/error_analysis/efficientnet_b0_false_closed_contact_sheet.jpg`

Visible patterns:

- Many false-open examples are borderline closed-eye crops with partial eyelid opening, low contrast, blur, or strong reflections.
- Reflections/glasses are common in several high-confidence errors, especially around subjects such as `s0022`.
- Many false-closed examples are narrow open eyes, squinting eyes, shadowed crops, or crops where reflections obscure eyelid detail.
- Several high-confidence errors cluster around a small number of test subjects, which suggests that runtime ROI consistency and camera-domain variation remain important next risks.

Manual inspection of the original individual images is still recommended before presentation if a detailed qualitative error taxonomy is needed.

## 7. Final Model Selection

Primary selected model: **MobileNetV2**

Recommended default threshold: **argmax / `p_eye_closed >= 0.50`**

Safety-prioritized reference: **ResNet18 with validation-selected threshold `p_eye_closed >= 0.30`**

Reasoning:

- MobileNetV2 has the best overall default test accuracy and macro F1.
- MobileNetV2 is lightweight and better aligned with real-time or near-real-time inference.
- The default `0.50` decision rule keeps a better balance between missed closed-eye frames and false alarms.
- ResNet18 at `0.30` is useful as a conservative safety reference because it reduces false-open errors and raises closed-eye recall, but it also sharply increases false-closed errors.

## 8. Recommended Use in Later Fusion

The eye module should output per-frame probabilities:

- `p_eye_closed`
- `p_eye_open`

Later fusion should combine `p_eye_closed` with the completed YawDD/YawDD+ Dash mouth/yawn module output `p_yawn`. The next fusion stage should use temporal smoothing or PERCLOS-like logic instead of treating one frame-level eye prediction as final drowsiness detection.

## 9. Readiness for Stage 10

Stage 10 status: **READY**

Required selected checkpoint: `outputs/mrl_eye/checkpoints/best_mobilenet_v2_mrl_eye.pt`

Checkpoint availability: **FOUND**

Stage 10 can begin because the selected MobileNetV2 checkpoint is present locally. Stage 10 should focus on runtime eye ROI consistency: whether live/video eye crops have the same framing, scale, grayscale/contrast behavior, and domain characteristics as the MRL Eye training crops.

## 10. Missing Files or Limitations

No expected Stage 9 local artifacts are missing.

Limitations:

- This Stage 9B pass did not train models and did not rerun full inference.
- Contact-sheet observations are qualitative and should not be treated as additional quantitative metrics.
- These results are MRL Eye eye-state specialist results only, not final driver drowsiness system accuracy.
