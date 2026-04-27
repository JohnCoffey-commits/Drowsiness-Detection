# Project Structure Guide

## 1. Purpose of This Document

This document explains how the repository is organized for the modular driver drowsiness project. It is intended as a living guide for teammates who need to find datasets, preprocessing outputs, training scripts, reports, model outputs, and current analysis artifacts quickly.

The repository currently supports a modular driver monitoring system rather than one monolithic drowsiness classifier:

- YawDD/YawDD+ Dash mouth/yawn specialist -> `p_yawn`
- MRL Eye open/closed specialist -> `p_eye_closed`

## 2. High-Level Project Architecture

| Module | Dataset source | Specialist task | Output concept | Current status |
| --- | --- | --- | --- | --- |
| Mouth/yawn module | Original YawDD Dash videos plus YawDD+ annotation files | Binary mouth/yawn classification | `p_yawn` | Stage 7 completed |
| Eye open/closed module | MRL Eye | Binary eye-state classification | `p_eye_closed` | Stage 9 and Stage 9B completed |
| NTHUDDD2 branch | Official NTHU considered; Kaggle extracted-frame version explored | Drowsy/not-drowsy frame classification | Not part of final module direction | Not main direction |

The future system should fuse the specialist module outputs over time. Current reported accuracies are specialist-module results, not final system-level driver drowsiness accuracy.

## 3. Repository Layout

Important top-level locations:

```text
Drowsiness_Detection/
  artifacts/
  checkpoints/
  colab_file/
  dataset/
  docs/
  outputs/
  reports/
  src/
  .gitignore
  README_*.md
  requirements.txt
```

| Path | Purpose |
| --- | --- |
| `dataset/` | Raw or locally reconstructed datasets. This is large local data and is ignored by Git. |
| `artifacts/` | Preprocessing outputs, manifests, split files, visual checks, and intermediate results. |
| `reports/` | Human-readable reports for dataset inspection, preprocessing, split validation, training summaries, and model selection. |
| `src/` | Python source code for dataset preparation, preprocessing, and training. |
| `colab_file/` | Google Colab notebooks used for GPU training and Colab workflows. |
| `outputs/` | Synced final training outputs, currently focused on MRL Eye Stage 9. |
| `checkpoints/` | Legacy or local model checkpoint location. Large checkpoint files should not be committed to normal Git. |
| `docs/` | GitHub-friendly project structure and current-status documentation. |

## 4. Dataset Folders

`dataset/` is for local raw or reconstructed data. It should generally not be committed.

Observed local dataset folders:

| Path | Purpose |
| --- | --- |
| `dataset/YawDD_raw/` | Original YawDD Dash video data. Used as the video source for reconstructed Dash frames. |
| `dataset/YawDD+/` | YawDD+ annotation files. These provide frame indices and class labels. |
| `dataset/YawDD_plus_reconstructed/` | Reconstructed Dash full frames and generated mouth crops derived from YawDD raw videos plus YawDD+ annotations. |
| `dataset/mrlEyes_2018_01/` | MRL Eye subject folders and annotation/stat files. Used for the eye open/closed module. |
| `dataset/NTHUDDD2/` | Kaggle extracted-frame NTHUDDD2 exploration data, not the current main system direction. |

Dataset notes:

- YawDD/YawDD+ uses `0 = no_yawn`, `1 = yawn`.
- MRL Eye uses `0 = closed`, `1 = open`.
- NTHUDDD2 Kaggle uses `notdrowsy = 0`, `drowsy = 1`, but this branch is not the current main module direction.

## 5. Artifacts and Manifests

`artifacts/` stores reproducible intermediate files from inspection, reconstruction, preprocessing, and splitting.

Important subfolders:

| Path | Purpose |
| --- | --- |
| `artifacts/mappings/` | CSV manifests produced during preprocessing and dataset preparation. |
| `artifacts/splits/` | Subject-level train/validation/test split files. |
| `artifacts/visual_checks/` | Contact sheets and visual sanity-check images. |
| `artifacts/results/` | Earlier baseline result outputs. Note that the local YawDD `initial_results.csv` currently appears stale and should not be treated as the final Stage 7 source. |
| `artifacts/preprocessed/` | Regenerable preprocessing outputs. Ignored by Git. |
| `artifacts/cache/` | Local cache files. Ignored by Git. |
| `artifacts/models/` | Local model artifacts. Ignored by Git. |

Key manifest examples:

| File | Meaning |
| --- | --- |
| `artifacts/mappings/yawdd_dash_all_labeled_frames.csv` | Reconstructed YawDD+ Dash labeled-frame manifest. |
| `artifacts/mappings/yawdd_dash_all_mouth_crops.csv` | All attempted YawDD mouth-crop rows, including failures. |
| `artifacts/mappings/yawdd_dash_all_mouth_crops_trainable.csv` | Trainable YawDD mouth-crop rows with split labels. |
| `artifacts/splits/yawdd_dash_subject_split.csv` | Leakage-safe subject-level YawDD split. |
| `artifacts/mappings/mrl_eye_all_images.csv` | Full MRL Eye image manifest. |
| `artifacts/mappings/mrl_eye_trainable.csv` | Trainable MRL Eye rows. |
| `artifacts/mappings/mrl_eye_trainable_with_split.csv` | MRL Eye trainable manifest with subject-level split labels. |
| `artifacts/splits/mrl_eye_subject_split.csv` | Subject-level MRL Eye split. |
| `artifacts/mappings/nthuddd2_kaggle_all_images*.csv` | Kaggle NTHUDDD2 exploration manifests. |

## 6. Reports

`reports/` contains human-readable Markdown and CSV summaries. These are useful for GitHub because they explain the decisions and checks behind the data pipeline.

Important reports:

| File | Purpose |
| --- | --- |
| `reports/yawdd_raw_dash_report.md` | Inspection of original YawDD Dash videos. |
| `reports/yawdd_plus_annotation_format_report.md` | Interpretation of YawDD+ annotation format and class IDs. |
| `reports/yawdd_dash_reconstruction_report.md` | Dash frame reconstruction summary. |
| `reports/yawdd_dash_visual_sanity_check.md` | Visual confirmation that YawDD+ class `1` corresponds to yawning and class `0` to non-yawning. |
| `reports/yawdd_dash_mouth_crop_report.md` | MediaPipe mouth-crop preprocessing summary and readiness result. |
| `reports/yawdd_dash_split_report.md` | YawDD subject-level split report and leakage checks. |
| `reports/mrl_eye_dataset_report.md` | MRL Eye Stage 8 inspection report. |
| `reports/mrl_eye_split_report.md` | MRL Eye subject-level split report. |
| `reports/mrl_eye_stage9_training_plan.md` | Stage 9 MRL Eye training design. |
| `reports/mrl_eye_stage9b_error_analysis.md` | Final Stage 9B MRL Eye model-selection report. |
| `reports/nthuddd2_kaggle_dataset_report.md` | Kaggle NTHUDDD2 exploration report and limitations. |
| `reports/nthu_dataset_report.md` | Earlier NTHUDDD2 inspection notes. |

## 7. Source Code

`src/` is organized by function.

### `src/data/`

Dataset inspection, manifest-building, splitting, validation, and spot-check scripts.

Important examples:

| File | Purpose |
| --- | --- |
| `src/data/inspect_mrl_eye.py` | Inspect raw MRL Eye files and produce dataset report. |
| `src/data/build_mrl_eye_manifest.py` | Build MRL Eye all-image and trainable manifests. |
| `src/data/split_mrl_eye_subjects.py` | Create leakage-safe subject-level MRL Eye split. |
| `src/data/spotcheck_mrl_eye.py` | Generate visual MRL Eye contact sheets. |
| `src/data/build_yawdd_dash_mapping.py` | Build YawDD/YawDD+ Dash frame mapping. |
| `src/data/extract_yawdd_dash_labeled_frames.py` | Reconstruct labeled Dash frames from original videos. |
| `src/data/build_yawdd_split.py` | Build YawDD subject-level split. |
| `src/data/build_nthuddd2_kaggle_manifest.py` | Build Kaggle NTHUDDD2 exploration manifest. |
| `src/data/split_nthuddd2_kaggle_subject.py` | Build Kaggle NTHUDDD2 subject split and LOSO folds. |

### `src/preprocessing/`

Mouth ROI crop generation for the YawDD mouth/yawn module.

| File | Purpose |
| --- | --- |
| `src/preprocessing/generate_yawdd_mouth_crops.py` | Generate mouth crops using MediaPipe Face Mesh landmarks. |
| `src/preprocessing/precompute_yawdd_mouth_crops.py` | Earlier mouth-crop preprocessing entrypoint. |

### `src/training/`

Training scripts for CNN baselines.

| File | Purpose |
| --- | --- |
| `src/training/train_classifier.py` | YawDD mouth/yawn classifier training helper. |
| `src/training/run_initial_baselines.py` | YawDD three-model baseline runner. |
| `src/training/train_mrl_eye_baselines.py` | MRL Eye Stage 9 baseline training pipeline. |

## 8. Colab Notebooks

`colab_file/` stores notebooks for GPU-based training and audit runs.

| Notebook | Purpose |
| --- | --- |
| `colab_file/stage7_yawdd_training.ipynb` | Stage 7 YawDD training notebook template. |
| `colab_file/stage7_yawdd_training_r.ipynb` | Completed Stage 7 YawDD run notebook with output metrics. |
| `colab_file/stage8_nthuddd2_kaggle_training.ipynb` | Kaggle NTHUDDD2 exploration notebook. Not current main direction. |
| `colab_file/stage9_mrl_eye_training.ipynb` | Stage 9 MRL Eye Colab training notebook. |
| `colab_file/stage9_mrl_eye_training_r.ipynb` | Completed Stage 9 MRL Eye run notebook. |

The completed Stage 7 YawDD result values are present in `colab_file/stage7_yawdd_training_r.ipynb`. The local `artifacts/results/initial_results.csv` currently appears stale and contains `not_run`, so teammates should not use that CSV as the final Stage 7 result source unless it is refreshed.

## 9. Outputs

`outputs/` is for synced final outputs from cloud/Colab runs.

The main current output tree is:

```text
outputs/mrl_eye/
  README.md
  artifact_inventory.md
  results/
  reports/
  figures/
  error_analysis/
  checkpoints/
```

Important MRL Eye output groups:

| Path | Purpose |
| --- | --- |
| `outputs/mrl_eye/results/` | Metrics CSV/JSON, histories, threshold sweeps, and Stage 9B model-selection machine-readable summaries. |
| `outputs/mrl_eye/reports/` | Colab-generated Stage 9 experiment summary. |
| `outputs/mrl_eye/figures/` | Training curves, confusion matrices, and closed-class precision-recall curves. |
| `outputs/mrl_eye/error_analysis/` | False-open and false-closed contact sheets. |
| `outputs/mrl_eye/checkpoints/` | Best model checkpoint files for ResNet18, MobileNetV2, and EfficientNet-B0. |

`outputs/mrl_eye/artifact_inventory.md` confirms the expected local MRL Eye output set is complete.

## 10. Model Checkpoints

Checkpoint locations:

| Path | Meaning |
| --- | --- |
| `outputs/mrl_eye/checkpoints/best_resnet18_mrl_eye.pt` | Best MRL Eye ResNet18 checkpoint. |
| `outputs/mrl_eye/checkpoints/best_mobilenet_v2_mrl_eye.pt` | Best MRL Eye MobileNetV2 checkpoint. This is the selected primary eye model. |
| `outputs/mrl_eye/checkpoints/best_efficientnet_b0_mrl_eye.pt` | Best MRL Eye EfficientNet-B0 checkpoint. |
| `checkpoints/` | Legacy/local checkpoint folder for earlier training scripts. |

`outputs/mrl_eye/checkpoints/` is needed locally for later model loading, but checkpoint binaries should generally not be committed to normal Git. Use Git LFS if checkpoint versioning is required.

## 11. Current Module Status

| Area | Status |
| --- | --- |
| YawDD/YawDD+ Dash mouth/yawn module | Completed and stable as the mouth/yawn specialist. |
| MRL Eye open/closed module | Stage 8 preparation, Stage 9 training, and Stage 9B model selection completed. |
| Selected MRL Eye model | MobileNetV2, default argmax / `p_eye_closed >= 0.50`. |
| Safety-prioritized MRL Eye reference | ResNet18 with validation-selected threshold around `0.30`. |
| NTHUDDD2 branch | Explored but no longer the main system direction. |

## 12. What Should and Should Not Be Committed to GitHub

Generally useful to commit:

- Documentation in `docs/`
- Human-readable reports in `reports/`
- Lightweight CSV/JSON metrics and manifests in `artifacts/mappings/`, `artifacts/splits/`, and `outputs/mrl_eye/results/`
- Figures in `outputs/mrl_eye/figures/`
- Error-analysis contact sheets in `outputs/mrl_eye/error_analysis/`
- Colab notebooks in `colab_file/`
- Source code in `src/`

Generally do not commit to normal Git:

- Raw datasets under `dataset/`
- Dataset zip files (`*.zip`)
- Model checkpoints (`*.pt`, `*.pth`, `*.ckpt`)
- `outputs/**/checkpoints/`
- Local caches and preprocessed bulk data
- Python virtual environments and cache folders

The current `.gitignore` protects datasets, zip files, model checkpoints, `outputs/**/checkpoints/`, cache folders, and Python/Jupyter cache artifacts.

## 13. How to Update This Document

Update this document whenever:

- A new major folder is added.
- A dataset is adopted or retired.
- A preprocessing stage creates a new canonical artifact.
- A training stage creates a new output directory.
- A checkpoint location or selected model changes.
- Git tracking policy changes for outputs or large files.

When adding a new result, prefer citing the exact local CSV/JSON/report path used as the source of truth.
