# MRL Eye Stage 9 Training Plan

## Purpose

Stage 9 trains the MRL Eye module as an eye open/closed specialist for the modular driver monitoring system.

The module is not a full drowsiness classifier. Its job is to estimate per-frame eye-state probabilities:

- `p_eye_closed`
- `p_eye_open`

These probabilities can later be smoothed over time and fused with the completed YawDD mouth/yawn module.

## Input and Output

Input:

- `artifacts/mappings/mrl_eye_trainable_with_split.csv`
- MRL Eye image files from `dataset/mrlEyes_2018_01/`

Labels:

- `0 = closed`
- `1 = open`

Output:

- Best model checkpoint per architecture
- Train/validation/test metrics
- Test threshold sweeps for `p_eye_closed`
- Confusion matrices
- Closed-class precision-recall curves
- False-open and false-closed visual error sheets

## Dataset and Split Summary

Stage 8 confirmed:

- Total images: 84,898
- Trainable images: 84,898
- Subjects: 37
- Closed: 41,946
- Open: 42,952

Subject-level split:

- Train: 25 subjects, 58,982 images, closed 29,310, open 29,672
- Validation: 6 subjects, 13,029 images, closed 6,333, open 6,696
- Test: 6 subjects, 12,887 images, closed 6,303, open 6,584

Leakage checks passed in Stage 8. Stage 9 must preserve this subject-level split and must not create random image-level splits.

## Models

Stage 9 supports three PyTorch/torchvision baselines:

- `resnet18`: strong simple baseline
- `mobilenet_v2`: lightweight real-time deployment candidate
- `efficientnet_b0`: strong transfer-learning baseline

Each model uses ImageNet-style preprocessing and replaces the final classification head with a 2-class head.

## Training Defaults

- Image size: 224
- Batch size: 64
- Epochs: 10
- Freeze epochs: 1
- Early stopping patience: 3
- Learning rate: 1e-4
- Loss: weighted cross entropy using training-split class weights
- Scheduler: ReduceLROnPlateau
- Checkpoint metric: validation macro F1

CUDA mixed precision is enabled when CUDA is available. CPU execution falls back cleanly.

## Metrics

Accuracy alone is not enough for this module. Stage 9 reports:

- Accuracy
- Macro precision, recall, and F1
- Weighted F1
- Closed-class precision, recall, and F1
- Open-class precision, recall, and F1
- Confusion matrix
- False-open count: true closed, predicted open
- False-closed count: true open, predicted closed

Closed-eye recall matters because false-open predictions on closed-eye samples are safety-critical: they can hide periods of eye closure that a later PERCLOS-like temporal module needs to detect.

## Threshold Tuning

The trained model outputs `p_eye_closed`. Stage 9 evaluates thresholds:

`0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70`

Prediction rule:

```text
if p_eye_closed >= threshold:
    predicted label = closed
else:
    predicted label = open
```

The goal is to expose the trade-off between closed-eye recall and false-closed errors. Stage 9 does not automatically choose an aggressive threshold that predicts almost everything as closed.

## Future Temporal Fusion

MRL Eye is frame-level. Real driver drowsiness depends on temporal behavior, not a single frame. Stage 10/11 should use the per-frame `p_eye_closed` output for smoothing or PERCLOS-like logic, then fuse it with the YawDD mouth/yawn specialist.

## Expected Next Step

Run full Stage 9 training in Colab with GPU:

```bash
python src/training/train_mrl_eye_baselines.py \
  --models resnet18 mobilenet_v2 efficientnet_b0 \
  --epochs 10 \
  --batch-size 64 \
  --image-size 224 \
  --manifest artifacts/mappings/mrl_eye_trainable_with_split.csv \
  --output-dir outputs/mrl_eye
```

Then inspect:

- `outputs/mrl_eye/results/mrl_eye_initial_results.csv`
- `outputs/mrl_eye/reports/mrl_eye_experiment_summary.md`
- Confusion matrices and threshold sweeps
- False-open and false-closed contact sheets
