# Initial Experiment Pipeline

This project stage implements a reproducible first-pass image-classification pipeline for mouth-focused yawning detection.

## Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Inspect Datasets

```bash
python3 -m src.data.inspect_datasets
```

Outputs:

- `reports/yawdd_dash_dataset_report.md`
- `reports/nthu_dataset_report.md`
- `reports/dataset_inspection_summary.csv`

The initial experiment uses `dataset/YawDD+/dataset/Dash` as the intended training source. `NTHUDDD2/train_data` is inspected only and is not substituted for YawDD+ Dash in this initial stage.

## 2. Preprocess YawDD+ Dash Mouth Crops

```bash
python3 -m src.preprocessing.precompute_yawdd_mouth_crops
```

Outputs:

- `artifacts/preprocessed/yawdd_dash_mouth/`
- `artifacts/preprocessed/yawdd_dash_mouth/manifest.csv`
- `artifacts/preprocessed/yawdd_dash_mouth/preprocessing_failures.csv`
- `reports/preprocessing_report.md`

The preprocessing script uses MediaPipe Face Mesh lip landmarks to crop the mouth region. If landmarks fail, it uses a lower-face fallback crop and logs the event.

## 3. Build Subject-Level Split

```bash
python3 -m src.data.build_yawdd_split
```

Output:

- `artifacts/splits/yawdd_dash_subject_split.csv`

The split is subject-level, approximately 70% train, 15% validation, and 15% test.

## 4. Train All Three Baselines

```bash
python3 -m src.training.run_initial_baselines
```

Models:

- CNN-1: ResNet18
- CNN-2: MobileNetV2
- CNN-3: EfficientNet-B0

Training defaults:

- PyTorch
- Adam
- learning rate `1e-4`
- weighted cross entropy
- batch size `32`, with practical fallback to `16`
- `12` epochs
- early stopping patience `3`
- `ReduceLROnPlateau`
- mild image augmentation
- freeze backbone first, then fine-tune the full model

Outputs:

- `artifacts/results/initial_results.csv`
- `artifacts/results/metrics_summary.json`
- `artifacts/figures/*_training_curve.png`
- `artifacts/figures/*_confusion_matrix.png`
- `reports/initial_experiment_summary.md`

## Current Dataset Note

If the local `YawDD+` folder contains only YOLO-style `.txt` annotation files and no image files, preprocessing and training cannot run yet. Add the paired Dash image frames under `dataset/YawDD+/dataset/Dash`, rerun the commands above, and the reports/results will be generated from the real data.
