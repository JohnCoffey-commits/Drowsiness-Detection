# Stage 14 Mouth/Yawn Model Availability Audit

Date: 2026-05-09

## Purpose

Stage 14 is intended to be the mouth/yawn runtime equivalent of Stage 10 eye ROI validation: full-face video input, mouth ROI extraction, mouth/yawn checkpoint loading, `p_yawn` timeline generation, and qualitative ROI evidence.

This audit checks whether the repository currently contains enough local mouth/yawn model information to implement that runtime step safely.

No model was trained. No checkpoint was modified. No Stage 8/9/10/11/12/13 source code was modified.

## Search Scope

Searched:

- `src/`
- `docs/`
- `reports/`
- `outputs/`
- `artifacts/`
- `checkpoints/`
- Stage 7 notebooks were also inspected for path clues because project documentation names `colab_file/stage7_yawdd_training_r.ipynb` as the Stage 7 source of truth.

Ignored:

- `.git/`
- `.venv/`
- `.venv-stage10/`
- `dataset/`
- `data/`
- `SystemUI/`
- `__pycache__/`
- `.ipynb_checkpoints/`
- `*.zip`

Checkpoint files were searched read-only.

## Audit Answers

### 1. Is there a usable selected mouth/yawn checkpoint?

**No.**

No trained local YawDD/YawDD+ Dash mouth/yawn checkpoint was found.

The only local `.pt`/`.pth` model files found were:

- `outputs/mrl_eye/checkpoints/best_resnet18_mrl_eye.pt`
- `outputs/mrl_eye/checkpoints/best_mobilenet_v2_mrl_eye.pt`
- `outputs/mrl_eye/checkpoints/best_efficientnet_b0_mrl_eye.pt`
- `artifacts/cache/torch/checkpoints/resnet18-f37072fd.pth`
- `artifacts/cache/torch/checkpoints/mobilenet_v2-7ebf99e0.pth`
- `artifacts/cache/torch/checkpoints/efficientnet_b0_rwightman-7f5810bc.pth`

The `outputs/mrl_eye/checkpoints/*` files are eye open/closed checkpoints, not mouth/yawn checkpoints. The `artifacts/cache/torch/checkpoints/*` files are generic torchvision pretrained backbone weights, not trained YawDD mouth/yawn classifiers.

### 2. Exact checkpoint path

**Blocked: no local selected mouth/yawn checkpoint path exists.**

The Stage 7 notebook code would save model checkpoints as:

```text
checkpoints/<model_name>_best.pt
```

and copy them to a Google Drive `outputs/checkpoints/` folder, but no corresponding trained local checkpoint was found in this repository.

### 3. What model architecture is required?

Known from `docs/PROJECT_CURRENT_STATUS.md`, `src/training/train_classifier.py`, and `colab_file/stage7_yawdd_training_r.ipynb`:

- Stage 7 trained `resnet18`, `mobilenet_v2`, and `efficientnet_b0`.
- Project status records ResNet18 as the current primary YawDD/YawDD+ Dash mouth/yawn specialist because it achieved the strongest Stage 7 test accuracy.
- The training code constructs ResNet18 with torchvision and replaces the classifier head:

```python
model = models.resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 2)
```

However, this architecture cannot be used for runtime inference without the trained Stage 7 mouth/yawn checkpoint.

### 4. What preprocessing/eval transform is required?

Known from `src/training/train_classifier.py` and Stage 7 notebook snippets:

- Input images are opened and converted with `Image.open(...).convert("RGB")`.
- Default image size is `224`.
- Evaluation transform is deterministic:

```python
transforms.Resize((image_size, image_size))
transforms.ToTensor()
transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
```

The Stage 7 notebook uses the same normalization and deterministic resize for evaluation.

### 5. What is the label mapping?

Known from project documentation and training code:

```text
0 = no_yawn
1 = yawn
```

Evidence:

- `src/training/train_classifier.py` defines `LABEL_TO_INDEX = {"no_yawn": 0, "yawn": 1}`.
- `docs/PROJECT_CURRENT_STATUS.md` records `0 = no_yawn`, `1 = yawn`.
- `reports/yawdd_dash_visual_sanity_check.md` confirms class `1` is yawning and class `0` is not yawning in sampled frames.

### 6. Which class index corresponds to yawn?

Class index `1` corresponds to `yawn`.

### 7. How should `p_yawn` be computed?

If the trained Stage 7 mouth/yawn checkpoint were available and loaded into the matching two-class model:

```python
probs = softmax(logits, dim=1)
p_no_yawn = probs[0]
p_yawn = probs[1]
```

This follows the confirmed mapping `0 = no_yawn`, `1 = yawn`.

### 8. Is a runtime full-video mouth/yawn script already present?

**No.**

Existing mouth/yawn code is preprocessing/training-oriented:

- `src/preprocessing/generate_yawdd_mouth_crops.py`
- `src/preprocessing/precompute_yawdd_mouth_crops.py`
- `src/data/verify_yawdd_mouth_crops.py`
- `src/training/train_classifier.py`
- `src/training/run_initial_baselines.py`

No runtime script was found that reads a full-face video and emits timestamped model-generated `p_yawn` predictions.

### 9. Is any blocking information missing?

**Yes. Stage 14 is blocked by the missing trained selected mouth/yawn checkpoint.**

The following are known:

- Architecture: ResNet18 with a two-class head.
- Eval transform: RGB, resize to 224 x 224, tensor conversion, ImageNet normalization.
- Label mapping: `0 = no_yawn`, `1 = yawn`.
- `p_yawn`: `softmax(logits)[1]`.

The following is missing:

- A local trained selected Stage 7 mouth/yawn checkpoint, expected to be a file like `checkpoints/resnet18_best.pt` or equivalent copied from the completed Stage 7 run.

Without that checkpoint, implementing Stage 14 runtime inference would require fabricating model weights or using generic pretrained ImageNet weights, which would be invalid for `p_yawn`.

## Decision

Stage 14 runtime mouth/yawn inference is **blocked** until the trained Stage 7 mouth/yawn checkpoint is restored locally.

Do not proceed to runtime `p_yawn` generation or Stage 15 real synchronized mouth-eye fusion until the selected mouth/yawn checkpoint is available and verified.
