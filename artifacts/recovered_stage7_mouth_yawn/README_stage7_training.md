# Stage 7 - Initial CNN Training on YawDD+ Dash Mouth Crops

## What this stage uses

Stage 7 consumes the existing Stage 6 subject-level split directly. It does not rebuild the split and does not perform any image-level randomization.

- Split manifest: `artifacts/splits/yawdd_dash_subject_split.csv`
- Trainable mouth-crop manifest: `artifacts/mappings/yawdd_dash_all_mouth_crops_trainable.csv`
- Split report: `reports/yawdd_dash_split_report.md`

The split manifest is the source of truth during training. Each row already contains:

- `split` (`train`, `val`, `test`)
- `processed_path` (the mouth-crop image path)
- `label` (`no_yawn` or `yawn`)
- `subject_id`

## How labels are read

Training reads labels from the split manifest rather than inferring them from folder names.

- current canonical labels: `no_yawn`, `yawn`
- backward-compatibility alias accepted by the loader: `no-yawn` -> `no_yawn`

This keeps Stage 7 compatible with the verified Stage 5 and Stage 6 outputs.

## How to launch the three baselines

Run all three initial baselines:

```bash
.venv/bin/python -m src.training.run_initial_baselines
```

Optional example with explicit settings:

```bash
.venv/bin/python -m src.training.run_initial_baselines \
  --split-manifest artifacts/splits/yawdd_dash_subject_split.csv \
  --epochs 12 \
  --batch-size 32 \
  --num-workers 0 \
  --seed 42
```

Run a single model directly:

```bash
.venv/bin/python -m src.training.train_classifier \
  --model resnet18 \
  --split-manifest artifacts/splits/yawdd_dash_subject_split.csv
```

Supported models:

- `resnet18`
- `mobilenet_v2`
- `efficientnet_b0`

## Where outputs are saved

Metrics and tables:

- `artifacts/results/initial_results.csv`
- `artifacts/results/metrics_summary.json`
- `artifacts/results/<model>_metrics.json`
- `artifacts/results/<model>_history.json`

Figures:

- `artifacts/figures/<model>_training_curve.png`
- `artifacts/figures/<model>_test_confusion_matrix.png`

Checkpoints:

- `checkpoints/<model>_best.pt`

Written summary:

- `reports/initial_experiment_summary.md`
