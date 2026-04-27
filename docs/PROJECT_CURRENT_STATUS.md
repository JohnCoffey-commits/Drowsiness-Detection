# Project Current Status and Experimental Summary

## 1. Project Goal

The project goal is to build a driver drowsiness detection and monitoring system using deep learning. The current design is modular: each model specializes in one visible driver behavior and later fusion should combine their probability outputs over time.

Current specialist modules:

- YawDD/YawDD+ Dash mouth/yawn module -> `p_yawn`
- MRL Eye open/closed module -> `p_eye_closed`

The results in this document are specialist-module results. They should not be reported as final system-level driver drowsiness accuracy.

## 2. Current System Design

| Module | Input type | Specialist task | Labels | Output concept | Current state |
| --- | --- | --- | --- | --- | --- |
| YawDD/YawDD+ Dash mouth/yawn | Mouth crops from reconstructed Dash video frames | No-yawn vs yawn | `0 = no_yawn`, `1 = yawn` | `p_yawn` | Completed |
| MRL Eye | Eye crop images | Closed vs open | `0 = closed`, `1 = open` | `p_eye_closed` | Completed through Stage 9B |

The final fatigue score or warning state is still future fusion work. Current metrics measure frame/image-level specialist classification, not the complete drowsiness monitoring system.

## 3. Dataset Strategy

The final direction uses complementary specialist datasets:

| Dataset | Used for | Reason |
| --- | --- | --- |
| YawDD Dash + YawDD+ annotations | Mouth/yawn specialist | Provides driver Dash videos and frame-level yawn labels. |
| MRL Eye | Eye open/closed specialist | Provides eye-state labels and subject folders suitable for subject-level splitting. |
| NTHUDDD2 official | Considered but not used | Official access required additional institutional/laboratory approval. |
| NTHUDDD2 Kaggle extracted frames | Explored only | Random image-level splitting can be misleading; subject-level evaluation was limited by only four parsed subjects and weak cross-subject generalization risk. |

Subject-level splitting is used for the current specialist datasets to reduce identity/frame leakage.

## 4. YawDD/YawDD+ Dash Mouth/Yawn Module

### 4.1 Dataset Source and Reconstruction

Source materials:

- Original YawDD Dash videos under `dataset/YawDD_raw/`
- YawDD+ annotation files under `dataset/YawDD+/`

Frames were reconstructed from the original YawDD Dash videos using YawDD+ annotation frame indices. Reconstruction outputs are under `dataset/YawDD_plus_reconstructed/`.

Relevant source reports:

- `reports/yawdd_raw_dash_report.md`
- `reports/yawdd_plus_annotation_format_report.md`
- `reports/yawdd_dash_reconstruction_report.md`
- `reports/yawdd_dash_visual_sanity_check.md`

Reconstruction summary from `reports/yawdd_dash_reconstruction_report.md`:

| Class | Count |
| --- | ---: |
| `no_yawn` | 57,347 |
| `yawn` | 7,031 |

### 4.2 Annotation Interpretation

YawDD+ annotation files provided frame labels and frame indices. Visual sanity checks confirmed:

- Class `0` corresponds to non-yawning frames.
- Class `1` corresponds to yawning frames.

Class mapping:

| Class ID | Label |
| ---: | --- |
| 0 | `no_yawn` |
| 1 | `yawn` |

The original YawDD+ bounding boxes were not used as final training mouth crops because visual checks showed that they were not reliable mouth ROIs. They often covered larger face regions and did not consistently isolate the mouth.

### 4.3 Mouth ROI Generation

Mouth crops were generated using MediaPipe Face Mesh lip landmarks, with a lower-face fallback crop when landmarks failed.

Source report: `reports/yawdd_dash_mouth_crop_report.md`

Processing summary:

| Metric | Value |
| --- | ---: |
| Total frames processed | 64,378 |
| MediaPipe Face Mesh crops | 64,093 |
| Fallback lower-face crops | 109 |
| Failed crops | 176 |
| Saved trainable crops | 64,202 |
| Success rate | 99.73% |

Saved crop class distribution:

| Class | Count |
| --- | ---: |
| `no_yawn` | 57,171 |
| `yawn` | 7,031 |

The Stage 5 mouth-crop report verdict was `READY`.

### 4.4 Subject-Level Split

Source report: `reports/yawdd_dash_split_report.md`

The YawDD mouth/yawn split is subject-level, not random image-level. This prevents the same subject from appearing across train, validation, and test splits.

| Split | Subjects | Images | `no_yawn` | `yawn` | Yawn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 20 | 44,156 | 39,345 | 4,811 | 10.90% |
| val | 4 | 8,892 | 7,902 | 990 | 11.13% |
| test | 5 | 11,154 | 9,924 | 1,230 | 11.03% |

Leakage checks passed:

- No subject appears in more than one split.
- Every split contains both classes.
- All referenced mouth-crop files exist.

### 4.5 Stage 7 Training Setup

Stage 7 trained three CNN baselines:

- ResNet18
- MobileNetV2
- EfficientNet-B0

Training settings documented in `README_stage7_training.md` and `colab_file/stage7_yawdd_training_r.ipynb`:

| Setting | Value |
| --- | --- |
| Framework | PyTorch / torchvision |
| Input | Mouth crops from `artifacts/splits/yawdd_dash_subject_split.csv` |
| Labels | `no_yawn`, `yawn` |
| Image size | 224 x 224 |
| Optimizer | Adam |
| Learning rate | `1e-4` |
| Loss | Weighted cross entropy |
| Batch size | 32, with practical fallback to 16 |
| Epochs | 12 |
| Early stopping patience | 3 |
| Scheduler | ReduceLROnPlateau |
| Augmentation | Mild rotation, brightness/contrast jitter, slight affine scaling |
| Transfer learning | Freeze backbone first, then fine-tune full model |

### 4.6 Stage 7 Results

Source of truth for completed local results: `colab_file/stage7_yawdd_training_r.ipynb`.

Important note: `artifacts/results/initial_results.csv` currently appears stale and reports `not_run`; do not use it as the final Stage 7 result source unless it is refreshed from the completed Colab output.

| CNN Architecture | Train Accuracy | Validation Accuracy | Test Accuracy | Yawn Precision | Yawn Recall | Yawn F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CNN-1: ResNet18 | 98.92% | 98.85% | 99.37% | 96.47% | 97.89% | 97.18% |
| CNN-2: MobileNetV2 | 98.97% | 98.48% | 98.75% | 91.74% | 97.48% | 94.52% |
| CNN-3: EfficientNet-B0 | 98.76% | 99.08% | 99.20% | 94.82% | 98.13% | 96.44% |

ResNet18 achieved the strongest Stage 7 test accuracy in the completed run. EfficientNet-B0 had the strongest validation accuracy.

### 4.7 Current YawDD Module Status

The YawDD/YawDD+ Dash mouth/yawn module is completed and should be treated as the stable mouth/yawn specialist. It should not be retrained or modified unless a later project decision explicitly changes the module.

## 5. NTHUDDD2 Branch Status

NTHUDDD2 is no longer the main system direction.

Summary:

- The official NTHU dataset was considered but could not be obtained within the project timeframe because access required institutional/laboratory approval.
- The Kaggle extracted-frame NTHUDDD2 version was explored under `dataset/NTHUDDD2/`.
- The Kaggle version contains 66,521 JPG frames and only four parsed subjects: `001`, `002`, `005`, and `006`.
- Random image-level splitting was considered misleading because visually similar or adjacent frames can appear across splits.
- Subject-level evaluation is more appropriate but is limited by the small number of parsed subjects and weak cross-subject generalization risk.
- MRL Eye replaced NTHUDDD2 as the eye open/closed specialist because it complements the YawDD mouth/yawn module more directly.

Source report: `reports/nthuddd2_kaggle_dataset_report.md`.

## 6. MRL Eye Open/Closed Module

### 6.1 Dataset Source

MRL Eye is used as an eye-state specialist dataset.

Local dataset root:

```text
dataset/mrlEyes_2018_01/
```

Expected structure includes `annotation.txt`, `stats_2018_01.ods`, and subject folders such as `s0001/` through `s0037/`.

Label mapping:

| Label | Meaning |
| ---: | --- |
| 0 | `closed` |
| 1 | `open` |

### 6.2 Stage 8 Dataset Preparation

Source report: `reports/mrl_eye_dataset_report.md`

Stage 8 confirmed:

| Metric | Value |
| --- | ---: |
| Total images | 84,898 |
| Trainable images | 84,898 |
| Subjects | 37 |
| Closed images | 41,946 |
| Open images | 42,952 |
| Unreadable images | 0 |
| Unparseable filenames | 0 |

The annotation check confirmed `0 = closed` and `1 = open`.

Important Stage 8 outputs:

- `artifacts/mappings/mrl_eye_all_images.csv`
- `artifacts/mappings/mrl_eye_trainable.csv`
- `artifacts/mappings/mrl_eye_trainable_with_split.csv`
- `artifacts/splits/mrl_eye_subject_split.csv`
- `reports/mrl_eye_dataset_report.md`
- `reports/mrl_eye_split_report.md`
- `artifacts/visual_checks/mrl_eye_closed_contact_sheet.jpg`
- `artifacts/visual_checks/mrl_eye_open_contact_sheet.jpg`
- `artifacts/visual_checks/mrl_eye_by_split_contact_sheet.jpg`

### 6.3 Subject-Level Split

Source report: `reports/mrl_eye_split_report.md`

The MRL Eye split is subject-level, not random image-level.

| Split | Subjects | Images | Closed | Open |
| --- | ---: | ---: | ---: | ---: |
| train | 25 | 58,982 | 29,310 | 29,672 |
| val | 6 | 13,029 | 6,333 | 6,696 |
| test | 6 | 12,887 | 6,303 | 6,584 |

Checks passed:

- Leakage check result: `True`
- Missing split label check result: `True`
- Every image receives exactly one split: `True`
- Every split contains closed and open: `True`
- Missing file check result: `True`

### 6.4 Stage 9 Training Setup

Source plan: `reports/mrl_eye_stage9_training_plan.md`

Stage 9 trained:

- ResNet18
- MobileNetV2
- EfficientNet-B0

Training setup:

| Setting | Value |
| --- | --- |
| Framework | PyTorch / torchvision |
| Input manifest | `artifacts/mappings/mrl_eye_trainable_with_split.csv` |
| Image size | 224 |
| Batch size | 64 |
| Epochs | 10 |
| Freeze epochs | 1 |
| Early stopping patience | 3 |
| Learning rate | `1e-4` |
| Loss | Weighted cross entropy from training split |
| Scheduler | ReduceLROnPlateau |
| Checkpoint metric | Validation macro F1 |
| Pretrained weights | Loaded for all three models |
| Mixed precision | Enabled when CUDA was available |

Stage 9 reports accuracy, macro precision/recall/F1, weighted F1, per-class metrics, confusion matrices, false-open counts, false-closed counts, and threshold sweeps for `p_eye_closed`.

### 6.5 Stage 9 Results

Source file: `outputs/mrl_eye/results/mrl_eye_initial_results.csv`

| Model | Train Accuracy | Validation Accuracy | Test Accuracy | Test Macro F1 | Test Closed Recall | False Open | False Closed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet18 | 99.16% | 98.37% | 98.46% | 98.46% | 98.59% | 89 | 109 |
| MobileNetV2 | 99.33% | 97.91% | 98.63% | 98.63% | 98.52% | 93 | 84 |
| EfficientNet-B0 | 99.44% | 97.91% | 98.62% | 98.62% | 98.24% | 111 | 67 |

Definitions:

- `false_open`: ground truth closed, predicted open. This is safety-critical because the eye module misses a closed-eye frame.
- `false_closed`: ground truth open, predicted closed. This is a false alarm tendency.

### 6.6 Stage 9B Error Analysis and Model Selection

Source files:

- `reports/mrl_eye_stage9b_error_analysis.md`
- `outputs/mrl_eye/results/mrl_eye_stage9b_model_selection.json`
- `outputs/mrl_eye/results/mrl_eye_stage9b_model_comparison.csv`

Stage 9B selected:

| Item | Selection |
| --- | --- |
| Primary selected model | MobileNetV2 |
| Recommended default threshold | argmax / `p_eye_closed >= 0.50` |
| Safety-prioritized reference | ResNet18 with validation-selected threshold `0.30` |
| Stage 10 readiness status recorded in selection JSON | `READY` |

Threshold summary:

| Model | Val-selected threshold | Test Macro F1 at 0.50 | Closed Recall at 0.50 | False Open at 0.50 | False Closed at 0.50 | Test Macro F1 at selected threshold | Closed Recall at selected threshold | False Open at selected threshold | False Closed at selected threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet18 | 0.30 | 98.46% | 98.59% | 89 | 109 | 97.60% | 99.08% | 58 | 251 |
| MobileNetV2 | 0.30 | 98.63% | 98.52% | 93 | 84 | 98.48% | 98.79% | 76 | 120 |
| EfficientNet-B0 | 0.30 | 98.62% | 98.24% | 111 | 67 | 98.52% | 98.65% | 85 | 106 |

Interpretation:

- MobileNetV2 is preferred as the primary model because it has the best overall default test accuracy/macro F1 and is lightweight.
- ResNet18 at threshold `0.30` is a conservative safety reference because it reduces false-open errors and improves closed-eye recall.
- The ResNet18 `0.30` threshold also increases false-closed errors substantially, so it is not the default setting.

### 6.7 Current MRL Eye Module Status

The MRL Eye open/closed module is completed through Stage 9B. Full local artifacts are present under `outputs/mrl_eye/`, including results, reports, figures, error-analysis contact sheets, and checkpoints.

Selected runtime checkpoint:

```text
outputs/mrl_eye/checkpoints/best_mobilenet_v2_mrl_eye.pt
```

## 7. Current Best Models

| Module | Primary model | Selection basis | Default decision rule |
| --- | --- | --- | --- |
| YawDD/YawDD+ Dash mouth/yawn | ResNet18 | Highest Stage 7 test accuracy among the three baselines | argmax over `no_yawn` / `yawn` |
| MRL Eye open/closed | MobileNetV2 | Best balance of default test accuracy, macro F1, closed-eye recall, false-open/false-closed trade-off, and real-time suitability | argmax / `p_eye_closed >= 0.50` |

Reference option:

| Module | Reference option | Why it is kept |
| --- | --- | --- |
| MRL Eye open/closed | ResNet18 at `p_eye_closed >= 0.30` | Safety-prioritized setting with higher closed-eye recall and fewer false-open errors, but more false alarms. |

## 8. Important Thresholds and Safety Notes

MRL Eye threshold behavior:

- Default: argmax / `p_eye_closed >= 0.50`
- Safety reference: ResNet18 with `p_eye_closed >= 0.30`

Safety interpretation:

- False-open errors are safety-critical because they miss closed-eye frames.
- False-closed errors are false alarms and can make the system too sensitive.
- Lowering the threshold can improve closed-eye recall but may increase false alarms.
- Threshold candidates were selected from validation sweeps only; test metrics are final reporting values.

YawDD safety interpretation:

- The YawDD mouth/yawn module identifies yawning behavior but does not alone prove drowsiness.
- The mouth/yawn output should be fused later with eye-state probabilities over time.

## 9. Important Files and Artifacts

| File or folder | Purpose |
| --- | --- |
| `docs/PROJECT_STRUCTURE.md` | Repository structure guide. |
| `docs/PROJECT_CURRENT_STATUS.md` | Current status and experimental summary. |
| `reports/yawdd_dash_mouth_crop_report.md` | YawDD mouth ROI preprocessing summary. |
| `reports/yawdd_dash_split_report.md` | YawDD subject-level split summary. |
| `colab_file/stage7_yawdd_training_r.ipynb` | Completed Stage 7 YawDD training run output. |
| `reports/mrl_eye_dataset_report.md` | MRL Eye dataset inspection report. |
| `reports/mrl_eye_split_report.md` | MRL Eye subject split report. |
| `outputs/mrl_eye/results/mrl_eye_initial_results.csv` | Main Stage 9 MRL Eye metrics table. |
| `outputs/mrl_eye/results/mrl_eye_metrics_summary.json` | Stage 9 MRL Eye metrics summary. |
| `reports/mrl_eye_stage9b_error_analysis.md` | Final MRL Eye model-selection report. |
| `outputs/mrl_eye/results/mrl_eye_stage9b_model_selection.json` | Machine-readable Stage 9B selection summary. |
| `outputs/mrl_eye/artifact_inventory.md` | Confirms complete local MRL Eye output set. |
| `outputs/mrl_eye/checkpoints/best_mobilenet_v2_mrl_eye.pt` | Selected MRL Eye model checkpoint. Ignored by Git. |

## 10. Reporting Notes and Limitations

Use careful wording in reports and presentations:

- Say these are specialist-module results.
- Do not claim final system-level driver drowsiness accuracy.
- YawDD/YawDD+ Dash results are mouth/yawn classification results.
- MRL Eye results are eye open/closed classification results.
- Subject-level split was used for the current YawDD and MRL Eye modules.
- Fusion and final fatigue scoring are future work.

Known artifact caveat:

- Final Stage 7 YawDD results are visible in `colab_file/stage7_yawdd_training_r.ipynb`.
- The local `artifacts/results/initial_results.csv` currently contains stale `not_run` values and should not be used for final Stage 7 reporting unless refreshed.

## 11. Maintenance Notes

- Keep `dataset/`, zip files, and checkpoint binaries out of normal Git.
- Metrics CSV/JSON, reports, figures, and error-analysis images may be committed for reproducibility.
- If checkpoints need to be versioned, use Git LFS rather than normal Git.
- When adding new results, cite the exact local artifact used as the source of truth.
- Keep the NTHUDDD2 work documented as an explored branch, not as the active final system direction.
