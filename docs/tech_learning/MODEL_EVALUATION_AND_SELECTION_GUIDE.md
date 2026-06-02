# Model Evaluation and Model Selection Guide

This guide explains how VisionGuard model evaluation results should be interpreted, and why the project selects MobileNetV2 for the eye specialist and ResNet18 for the mouth/yawn specialist.

Recommended prerequisites:

- `docs/tech_learning/MODEL_TRAINING_TECHNICAL_GUIDE.md`
- `docs/tech_learning/DATA_PREPROCESSING_TECHNICAL_GUIDE.md`
- `docs/AI_PROJECT_CONTEXT.md`
- `docs/PROJECT_CURRENT_STATUS.md`

---

## 1. Purpose of This Document

This guide focuses on two questions:

1. How should accuracy, precision, recall, F1, confusion matrices, ROC/AUC, and related metrics be read in this project?
2. Why should model selection not be based on accuracy alone?

Core boundary:

- This project evaluates specialist image-level classifiers.
- These metrics are not final full-system drowsiness accuracy.
- The runtime system combines `p_eye_closed`, `p_yawn`, signal quality, and temporal fusion rules before producing alert states.

---

## 2. Evaluation Scope in VisionGuard

VisionGuard has two main offline specialist evaluations:

| Specialist | Input | Output | What is evaluated |
|---|---|---|---|
| Eye open/closed | Eye ROI image | `p_eye_closed` | Whether a single eye ROI is closed/open |
| Mouth/yawn | Mouth ROI image | `p_yawn` | Whether a single mouth ROI is no_yawn/yawn |

These evaluations answer:

- Can the model classify ROI images correctly on a held-out test set?
- Are key classes handled with acceptable recall and precision?
- Is the model suitable for runtime inference?

They do not answer:

- Is the driver truly fatigued?
- What is the final drowsiness accuracy of the full system?
- Does every alert interval have a ground-truth drowsiness label?

---

## 3. What Is Being Evaluated?

Project evaluation has at least five layers:

1. **Image-level classification performance**: whether a single ROI image is classified correctly.
2. **Class-level performance**: whether important classes are missed or over-triggered.
3. **Validation/test performance**: validation is used for model selection, while test is used for final reporting.
4. **Runtime usefulness**: model size, speed, checkpoint availability, and output stability.
5. **Qualitative error patterns**: whether errors cluster around glasses, lighting, blur, talking, or crop misalignment.

Model selection is therefore an engineering decision, not just a ranking by one number.

---

## 4. Confusion Matrix

A confusion matrix compares true labels with predicted labels. The common binary-classification terms are:

| Term | Meaning |
|---|---|
| True Positive | True class is positive and prediction is positive |
| True Negative | True class is negative and prediction is negative |
| False Positive | True class is negative but prediction is positive |
| False Negative | True class is positive but prediction is negative |

### 4.1 Eye Confusion Matrix

MRL Eye label mapping:

- `0 = closed`
- `1 = open`

If `closed` is the class of interest:

- false open: true closed, predicted open. This may miss closed-eye evidence.
- false closed: true open, predicted closed. This may add false closed-eye evidence.

Source: `src/training/train_mrl_eye_baselines.py`

### 4.2 Mouth/Yawn Confusion Matrix

YawDD/YawDD+ mouth/yawn label mapping:

- `0 = no_yawn`
- `1 = yawn`

If `yawn` is the class of interest:

- false yawn: true no_yawn, predicted yawn. This may add misleading yawn evidence.
- missed yawn: true yawn, predicted no_yawn. This may miss yawn evidence.

Sources:

- `src/training/train_classifier.py`
- `report_assets/mouth_yawn_evaluation_refresh/metrics/resnet18_metrics_summary.json`

---

## 5. Core Metrics

### 5.1 Accuracy

Accuracy is the percentage of all samples that are correctly predicted.

Advantage: easy to understand.

Limitation: it can be misleading when classes are imbalanced. If no_yawn samples greatly outnumber yawn samples, a model biased toward no_yawn can still show high accuracy.

### 5.2 Precision

Precision answers: of all samples predicted as a class, how many are correct?

For yawn:

```text
yawn precision = true_yawn_predictions / all_yawn_predictions
```

Low yawn precision means many false yawns, which can create misleading yawn evidence.

### 5.3 Recall

Recall answers: of all real samples in a class, how many were found?

For closed eye:

```text
closed recall = correctly_predicted_closed / all_true_closed
```

Low closed-eye recall means the model misses closed-eye evidence.

### 5.4 F1-Score

F1 is the harmonic mean of precision and recall. It is useful when both false positives and false negatives matter.

```text
F1 = 2 * precision * recall / (precision + recall)
```

### 5.5 Macro F1 and Weighted F1

Macro F1 averages the F1 of each class without weighting by class size. It is useful for imbalanced datasets because minority-class performance is not hidden by majority-class size.

Weighted F1 weights each class by its support. It is useful for overall dataset performance, but it can hide minority-class weakness.

### 5.6 ROC/AUC and PR/AUC

ROC curves show true positive rate versus false positive rate across thresholds. AUC measures probability-ranking ability.

PR curves focus on precision and recall. For imbalanced tasks, PR curves are often more informative than ROC curves.

In this project:

- The mouth/yawn evaluation refresh includes PR curve, ROC curve, and threshold sweep.
- The MRL Eye output includes PR curves, confusion matrices, training curves, and threshold sweeps; no eye ROC curve artifact was found in the local `outputs/mrl_eye/figures/` folder or in the inspected Google Drive MRL Eye figures folders.
- A full MRL Eye ROC curve requires per-sample `p_eye_closed` scores. The current local results save threshold sweep CSVs, but they do not save a per-sample prediction CSV. Therefore, the existing files can directly support only a coarse 9-threshold ROC-style plot. To generate a standard ROC/AUC, run an inference-only prediction pass with the existing checkpoint, manifest, and the prediction logic in `src/training/train_mrl_eye_baselines.py`, save `y_true` and `p_eye_closed`, and then compute ROC/AUC. No retraining is required for that.

Sources:

- `report_assets/mouth_yawn_evaluation_refresh/figures/`
- `outputs/mrl_eye/figures/`
- `outputs/mrl_eye/results/*_test_threshold_sweep.csv`
- `src/training/train_mrl_eye_baselines.py`

---

## 6. Evaluation Artifacts and Source of Truth

| Path | Content | Use |
|---|---|---|
| `outputs/mrl_eye/results/mrl_eye_initial_results.csv` | MRL Eye three-model results | Eye candidate comparison |
| `outputs/mrl_eye/results/mrl_eye_stage9b_model_selection.json` | Final eye model selection | Source for MobileNetV2 runtime selection |
| `outputs/mrl_eye/results/resnet18_metrics.json` | Detailed eye ResNet18 metrics | Candidate evaluation |
| `outputs/mrl_eye/results/mobilenet_v2_metrics.json` | Detailed eye MobileNetV2 metrics | Final model evaluation |
| `outputs/mrl_eye/results/efficientnet_b0_metrics.json` | Detailed eye EfficientNet-B0 metrics | Candidate evaluation |
| `outputs/mrl_eye/results/*_test_threshold_sweep.csv` | Eye threshold sweep results | Can support coarse threshold ROC points; not a full probability-ranked ROC |
| `outputs/mrl_eye/figures/` | Confusion matrices, PR curves, training curves | Report figures |
| `outputs/mrl_eye/error_analysis/` | False open / false closed examples | Qualitative analysis |
| `artifacts/recovered_stage7_mouth_yawn/metrics_summary.json` | Recovered mouth/yawn three-model results | Stage 7 model comparison |
| `report_assets/mouth_yawn_evaluation_refresh/metrics/resnet18_metrics_summary.json` | Final refreshed ResNet18 mouth/yawn evaluation | Recommended report source |
| `report_assets/mouth_yawn_evaluation_refresh/figures/` | Confusion matrix, PR, ROC, threshold sweep | Report figures |
| `report_assets/mouth_yawn_evaluation_refresh/predictions/` | Test predictions and probabilities | Threshold and ranking analysis |
| Google Drive `Drowsiness_Detection_Colab/outputs/results/*_history.json` | Original Stage 7 mouth/yawn training history | Source for mouth/yawn training curves |
| Google Drive `Drowsiness_Detection_Colab/outputs/figures/*_training_curve.png` | Original Stage 7 mouth/yawn training curves | Report training-curve figure source |

If older files conflict with recovered final artifacts, prefer `outputs/mrl_eye/results/mrl_eye_stage9b_model_selection.json` and `report_assets/mouth_yawn_evaluation_refresh/metrics/resnet18_metrics_summary.json` for final reporting.

---

## 7. Eye Model Evaluation

### 7.1 Candidate Models

MRL Eye evaluated:

- ResNet18
- MobileNetV2
- EfficientNet-B0

Source: `outputs/mrl_eye/results/mrl_eye_initial_results.csv`

### 7.2 Key Results

| Model | Best epoch | Test accuracy | Test macro F1 | Closed recall | False open | False closed |
|---|---:|---:|---:|---:|---:|---:|
| ResNet18 | 4 | 98.46% | 98.46% | 98.59% | 89 | 109 |
| MobileNetV2 | 8 | 98.63% | 98.63% | 98.52% | 93 | 84 |
| EfficientNet-B0 | 8 | 98.62% | 98.62% | 98.24% | 111 | 67 |

Source: `outputs/mrl_eye/results/mrl_eye_initial_results.csv`

### 7.3 Final Selection

The final eye runtime model is MobileNetV2.

Key source:

```text
outputs/mrl_eye/results/mrl_eye_stage9b_model_selection.json
```

Confirmed:

- `primary_selected_model = mobilenet_v2`
- `recommended_default_threshold = 0.5`
- `recommended_default_rule = argmax / p_eye_closed >= 0.50`
- runtime checkpoint found: `outputs/mrl_eye/checkpoints/best_mobilenet_v2_mrl_eye.pt`

Reasons for selecting MobileNetV2:

- It has the strongest or tied-strongest test accuracy and macro F1 among the candidates.
- It is lightweight and suitable for realtime eye ROI inference.
- The runtime files point to its checkpoint.

Sources:

- `outputs/mrl_eye/results/mrl_eye_stage9b_model_selection.json`
- `src/runtime/stage10_eye_roi_consistency.py`
- `src/runtime/realtime_frame_inference.py`

### 7.4 Remaining Limitations

Eye evaluation is still image-level ROI classification. It cannot prove fatigue by itself; it only shows strong open/closed classification performance on the held-out MRL Eye split.

Important limitations:

- glasses and reflection can cause errors
- motion blur and low lighting can affect ROI quality
- squinting, looking down, and side pose may not be clean open/closed cases
- runtime camera data may differ from the MRL Eye distribution

---

## 8. Mouth/Yawn Model Evaluation

### 8.1 Candidate Models

Stage 7 mouth/yawn evaluated:

- ResNet18
- MobileNetV2
- EfficientNet-B0

Source: `artifacts/recovered_stage7_mouth_yawn/metrics_summary.json`

### 8.2 Three-Model Results

| Model | Best epoch | Test accuracy | Yawn precision | Yawn recall | Yawn F1 | False yawn | Missed yawn |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet18 | 4 | 99.37% | 96.47% | 97.89% | 97.18% | 44 | 26 |
| MobileNetV2 | 4 | 98.75% | 91.74% | 97.48% | 94.52% | 108 | 31 |
| EfficientNet-B0 | 3 | 99.20% | 94.82% | 98.13% | 96.44% | 66 | 23 |

Source: `artifacts/recovered_stage7_mouth_yawn/metrics_summary.json`

### 8.3 Final ResNet18 Evaluation Refresh

For final reporting, the refreshed evaluation source is preferred:

```text
report_assets/mouth_yawn_evaluation_refresh/metrics/resnet18_metrics_summary.json
```

Confirmed:

| Metric | Value |
|---|---:|
| Test samples | 11,154 |
| Label distribution | no_yawn 9,924; yawn 1,230 |
| Accuracy | 99.37% |
| Macro precision | 98.11% |
| Macro recall | 98.72% |
| Macro F1 | 98.41% |
| Weighted F1 | 99.37% |
| Yawn precision | 96.47% |
| Yawn recall | 97.89% |
| Yawn F1 | 97.18% |
| ROC AUC | 99.84% |
| PR AUC / Average Precision | 99.45% |
| Confusion matrix | `[[9880, 44], [26, 1204]]` |

### 8.4 Why ResNet18 Was Selected

ResNet18 is the mouth/yawn runtime model because:

- it has the highest test accuracy in the recovered Stage 7 metrics
- it has the highest yawn F1
- it has fewer false yawns than MobileNetV2 and EfficientNet-B0
- its checkpoint is the runtime mouth/yawn checkpoint

Precise wording matters:

- EfficientNet-B0 has higher yawn recall and fewer missed yawns.
- ResNet18 has higher overall test accuracy and yawn F1.
- Therefore, do not write “EfficientNet-B0 had the best overall result.” A safer statement is: EfficientNet-B0 had the strongest yawn recall, while ResNet18 was selected for the strongest overall test accuracy and yawn F1.

Sources:

- `artifacts/recovered_stage7_mouth_yawn/metrics_summary.json`
- `report_assets/mouth_yawn_evaluation_refresh/metrics/resnet18_metrics_summary.json`
- `src/runtime/stage14_mouth_yawn_runtime.py`

---

## 9. Model Selection Is Not Just Highest Accuracy

Final selection should consider:

- test accuracy
- macro F1
- class-specific recall
- class-specific precision
- false positive / false negative cost
- validation behavior
- checkpoint completeness
- inference speed
- model size
- runtime integration simplicity
- qualitative error patterns
- compatibility with the ROI crop pipeline

VisionGuard examples:

- MobileNetV2 is selected for the eye specialist not only because its metrics are strong, but also because it is lightweight and runtime files point to it.
- ResNet18 is selected for the mouth/yawn specialist because it has the strongest overall test accuracy and yawn F1 in the final comparison.
- EfficientNet-B0 is useful for comparison, but it is not the current runtime default.

---

## 10. Error Analysis

Aggregate metrics tell you how many errors occurred. Error analysis helps explain why they occurred.

### 10.1 Common Eye Error Sources

- squinting or half-closed eyes
- glasses and reflection
- low light or strong light
- motion blur
- eye ROI misalignment
- side pose or occlusion

Project files:

- `outputs/mrl_eye/error_analysis/`
- `outputs/mrl_eye/figures/`

### 10.2 Common Mouth/Yawn Error Sources

- talking with an open mouth
- smiling or laughing
- mouth opening that is not a yawn
- head pose changes
- mouth landmark failure
- fallback crop including too much non-mouth area

Project files:

- `report_assets/mouth_yawn_evaluation_refresh/error_gallery/`
- `report_assets/mouth_yawn_evaluation_refresh/predictions/`

### 10.3 Why Keyframes and Video Evidence Matter

Video Upload evidence figures and keyframes help show how model probabilities change over time and how temporal fusion turns continuous evidence into alert intervals. However, they are still runtime demonstrations, not ground-truth drowsiness accuracy unless the video has manually labelled ground truth.

Source: `src/runtime/system_video_upload_pipeline.py`

---

## 11. Runtime Evaluation Boundary

The following levels must be kept separate:

| Level | Meaning | Can it be used as system accuracy? |
|---|---|---|
| Offline image-level test metrics | ROI image classification test | No |
| Video Upload evidence figures | Uploaded-video model probabilities and fusion timeline | No, unless ground truth exists |
| Realtime Live Monitor behavior | Camera-driven realtime alerts | No |
| History / Insights analytics | Product summaries of Live Monitor records | No |
| Full system validation | Independent evaluation with ground-truth drowsiness labels | Potentially yes |

History and Insights summarize Live Monitor records. They are not accuracy reports.

---

## 12. How to Report Metrics Correctly

Recommended wording:

- “The selected eye-state specialist model achieved strong test performance for open/closed eye classification.”
- “The mouth/yawn specialist provides `p_yawn` evidence for the later temporal fusion layer.”
- “The runtime system produces warning-candidate intervals based on temporal visual evidence, rather than direct ground-truth drowsiness labels.”
- “Specialist model metrics should be interpreted as ROI-level classification results, not final system-level drowsiness accuracy.”

Avoid:

- “The system detects drowsiness with 98% accuracy.”
- “The model proves the driver is drowsy.”
- “Every yawn means drowsiness.”
- “Every closed-eye frame means fatigue.”
- “Warning-candidate intervals are ground truth.”

---

## 13. Beginner Checklist

You should be able to answer:

- What does each cell of a confusion matrix mean?
- What is the difference between precision and recall?
- Why is macro F1 useful for imbalanced data?
- Why does yawn precision matter?
- Why does closed-eye recall matter?
- Why was MobileNetV2 selected for eye?
- Why was ResNet18 selected for mouth/yawn?
- Why is EfficientNet-B0 not the runtime default?
- Why are model test scores not final drowsiness detection accuracy?

---

## 14. Common Mistakes

| Mistake | Correct approach |
|---|---|
| Reporting only accuracy | Report precision, recall, F1, and confusion matrix as well |
| Mixing label mappings | MRL Eye: `0=closed,1=open`; mouth/yawn: `0=no_yawn,1=yawn` |
| Treating validation results as test results | Separate validation selection from held-out test reporting |
| Selecting a model only by highest recall | Also consider precision, F1, false positives, and runtime suitability |
| Saying EfficientNet-B0 is the final runtime model | It is a comparison model, not the current default |
| Using stale metric files | Check the final artifact and source path first |
| Treating alert intervals as ground truth | Alert intervals are rule-based runtime outputs |
| Writing “the system is 99% accurate” | Report specialist ROI-level metrics instead |

---

## 15. Current Inconsistencies to Watch Carefully

1. **Stage 7 epochs / patience**

   The completed Colab run constants are:

   - `DEFAULT_EPOCHS = 8`
   - `DEFAULT_PATIENCE = 2`

   Source: `colab_file/stage7_yawdd_training_r.ipynb`

   The local reusable script defaults are:

   - `--epochs = 12`
   - `--patience = 3`

   Source: `src/training/train_classifier.py`

2. **Mouth/yawn EfficientNet-B0 conclusion**

   EfficientNet-B0 has the highest yawn recall, but ResNet18 has higher test accuracy and yawn F1. The final runtime model is ResNet18.

3. **Eye optimizer**

   The MRL Eye training script uses AdamW, not Adam.

   Source: `src/training/train_mrl_eye_baselines.py`

4. **Mouth/yawn training curve**

   The local `report_assets/mouth_yawn_evaluation_refresh/skipped/training_curve_status.md` file only says the evaluation-refresh folder itself did not contain enough source data to reconstruct the real Stage 7 training curve. The original Stage 7 Google Drive outputs contain `Drowsiness_Detection_Colab/outputs/results/resnet18_history.json`, `mobilenet_v2_history.json`, `efficientnet_b0_history.json`, and matching `outputs/figures/*_training_curve.png` files. If the report needs a mouth/yawn training curve, cite those original Drive outputs; do not infer it from refreshed metrics or fabricate one.

5. **MRL Eye ROC curve**

   No generated MRL Eye ROC curve image was found locally or in the inspected Google Drive folders. The existing `outputs/mrl_eye/results/*_test_threshold_sweep.csv` files can be used to draw a coarse 9-threshold ROC-style plot, but that is not the same as a full ROC/AUC. Full ROC/AUC requires per-sample `p_eye_closed` scores. The training script `src/training/train_mrl_eye_baselines.py` already contains prediction logic, but it does not save per-sample MRL Eye predictions as CSV. To report MRL Eye ROC/AUC formally, run an inference-only prediction pass with the saved checkpoint and save the scores before computing the curve.
