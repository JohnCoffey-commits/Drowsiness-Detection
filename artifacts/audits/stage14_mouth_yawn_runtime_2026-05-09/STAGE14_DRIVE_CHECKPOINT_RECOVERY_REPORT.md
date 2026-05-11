# Stage 14 Drive Checkpoint Recovery Report

Date: 2026-05-10

## Section 1 - Best Checkpoint Candidate

**Best candidate:** `resnet18_best.pt`

| Field | Value |
| --- | --- |
| Drive path | `My Drive/Drowsiness_Detection_Colab/outputs/checkpoints/resnet18_best.pt` |
| Drive URL | `https://drive.google.com/file/d/13hQ7HqoKG6CXpkvnmjlJNSHSpp0d8dTD/view?usp=drivesdk` |
| Drive file ID | `13hQ7HqoKG6CXpkvnmjlJNSHSpp0d8dTD` |
| File size | Not exposed by the Google Drive connector metadata response for this binary file |
| Created time | `2026-04-24T13:12:05.814Z` |
| Modified time | `2026-04-24T12:54:20.000Z` |
| Confidence | High |
| Recommended local target | `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/checkpoints/resnet18_best.pt` |
| Should download | Yes |

Why this is likely the correct Stage 7 mouth/yawn checkpoint:

- It is located under the Stage 7 Colab output tree: `Drowsiness_Detection_Colab/outputs/checkpoints/`.
- The same folder contains the expected Stage 7 checkpoint set:
  - `resnet18_best.pt`
  - `mobilenet_v2_best.pt`
  - `efficientnet_b0_best.pt`
- The matching Stage 7 results folder contains `initial_results.csv`, `metrics_summary.json`, and `resnet18_metrics.json`.
- `initial_results.csv` shows `CNN-1 (ResNet18)` achieved the strongest test accuracy: `0.9937242244934552`.
- `initial_experiment_summary.md` states that CNN-1 ResNet18 achieved the strongest test accuracy in the completed Stage 7 run.
- `README_stage7_training.md` says Stage 7 checkpoints are saved as `checkpoints/<model>_best.pt`.
- The checkpoint modified time matches the `resnet18_metrics.json` modified time: `2026-04-24T12:54:20.000Z`.

Important caveat:

- The Drive connector search/list/metadata path identified the checkpoint, but the binary checkpoint content was not downloaded or inspected during this recovery task.
- After download, Stage 14 should verify `torch.load(...)` contents locally before runtime inference.
- Expected compatibility check after download: checkpoint payload should contain a ResNet18-compatible state dict, with final `fc` classifier weights shaped for two classes, and metadata such as `class_to_index = {"no_yawn": 0, "yawn": 1}` if present.

Additional checkpoint candidates:

| Candidate | Drive path | File ID | Modified time | Confidence | Evidence | Should download |
| --- | --- | --- | --- | --- | --- | --- |
| `resnet18_best_single.pt` | `My Drive/Drowsiness_Detection_Colab/outputs/checkpoints/resnet18_best_single.pt` | `1h_tIQeqwUgY9ZHFKjU2sbqR_71pPRqtT` | `2026-04-24T12:43:42.826Z` | Low | Same checkpoint folder, but name suggests an earlier single/smoke run. Results folder also contains `*_smoketest` files around this time. | No, unless the final checkpoint fails |
| `mobilenet_v2_best.pt` | `My Drive/Drowsiness_Detection_Colab/outputs/checkpoints/mobilenet_v2_best.pt` | `18JtrDrwCy2YHAecc8EzFPR8z4oWoTL1W` | `2026-04-24T13:03:38.000Z` | Medium as a Stage 7 checkpoint, low as selected model | Same Stage 7 checkpoint folder, but ResNet18 was selected by test accuracy. | Optional only |
| `efficientnet_b0_best.pt` | `My Drive/Drowsiness_Detection_Colab/outputs/checkpoints/efficientnet_b0_best.pt` | `14OkU3u0bbqBzPJrgCnZkFQzMWNqkHy5H` | `2026-04-24T13:11:32.000Z` | Medium as a Stage 7 checkpoint, low as selected model | Same Stage 7 checkpoint folder, but ResNet18 was selected by test accuracy. | Optional only |

Rejected/invalid candidates:

- `outputs/mrl_eye/checkpoints/*`: MRL Eye open/closed checkpoints, not YawDD mouth/yawn.
- `artifacts/cache/torch/checkpoints/*`: generic torchvision pretrained backbone weights, not trained Stage 7 mouth/yawn classifiers.

## Section 2 - Files the User Should Download

### A. Required

| File | Drive path | Recommended local destination | Reason |
| --- | --- | --- | --- |
| `resnet18_best.pt` | `My Drive/Drowsiness_Detection_Colab/outputs/checkpoints/resnet18_best.pt` | `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/checkpoints/resnet18_best.pt` | Required selected trained Stage 7 mouth/yawn checkpoint for Stage 14 runtime inference |

### B. Recommended

| File | Drive path | Recommended local destination | Reason |
| --- | --- | --- | --- |
| `initial_results.csv` | `My Drive/Drowsiness_Detection_Colab/outputs/results/initial_results.csv` | `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/artifacts/recovered_stage7_mouth_yawn/initial_results.csv` | Shows ResNet18 has the strongest Stage 7 test accuracy |
| `metrics_summary.json` | `My Drive/Drowsiness_Detection_Colab/outputs/results/metrics_summary.json` | `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/artifacts/recovered_stage7_mouth_yawn/metrics_summary.json` | Contains model metrics, image size, and pretrained flag for all Stage 7 models |
| `resnet18_metrics.json` | `My Drive/Drowsiness_Detection_Colab/outputs/results/resnet18_metrics.json` | `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/artifacts/recovered_stage7_mouth_yawn/resnet18_metrics.json` | Contains selected model metrics, best epoch, image size, and test confusion matrix |
| `initial_experiment_summary.md` | `My Drive/Drowsiness_Detection_Colab/outputs/reports/initial_experiment_summary.md` | `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/artifacts/recovered_stage7_mouth_yawn/initial_experiment_summary.md` | Human-readable Stage 7 result summary stating ResNet18 achieved strongest test accuracy |
| `README_stage7_training.md` | `My Drive/Drowsiness_Detection_Colab/repo/README_stage7_training.md` | `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/artifacts/recovered_stage7_mouth_yawn/README_stage7_training.md` | Documents Stage 7 input manifests, labels, supported models, and checkpoint naming |

### C. Optional

| File | Drive path | Recommended local destination | Reason |
| --- | --- | --- | --- |
| `resnet18_history.json` | `My Drive/Drowsiness_Detection_Colab/outputs/results/resnet18_history.json` | `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/artifacts/recovered_stage7_mouth_yawn/resnet18_history.json` | Training history for selected model |
| `resnet18_test_confusion_matrix.png` | `My Drive/Drowsiness_Detection_Colab/outputs/figures/resnet18_test_confusion_matrix.png` | `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/artifacts/recovered_stage7_mouth_yawn/resnet18_test_confusion_matrix.png` | Visual evidence of selected model test confusion matrix |
| `resnet18_training_curve.png` | `My Drive/Drowsiness_Detection_Colab/outputs/figures/resnet18_training_curve.png` | `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/artifacts/recovered_stage7_mouth_yawn/resnet18_training_curve.png` | Visual evidence of selected model training curve |
| `stage7_yawdd_training.ipynb` | `My Drive/Drowsiness_Detection_Colab/repo/colab_file/stage7_yawdd_training.ipynb` | `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/artifacts/recovered_stage7_mouth_yawn/stage7_yawdd_training.ipynb` | Notebook source evidence; optional because local repo already has Stage 7 notebook files |

Known Stage 7 runtime semantics after recovery:

- Expected architecture: torchvision `resnet18` with `model.fc = nn.Linear(model.fc.in_features, 2)`.
- Label mapping: `0 = no_yawn`, `1 = yawn`.
- `p_yawn = softmax(logits)[1]`.
- Evaluation transform: RGB image, resize `224 x 224`, `ToTensor`, ImageNet normalization.

## Section 3 - If No Valid Checkpoint Is Found

A high-confidence selected Stage 7 checkpoint candidate was found in Drive:

```text
My Drive/Drowsiness_Detection_Colab/outputs/checkpoints/resnet18_best.pt
```

Therefore Stage 14 does not need retraining if this checkpoint downloads and passes local compatibility checks.

If the checkpoint fails local validation after download, Stage 14 remains blocked. The required next action would be to recover the completed Stage 7 Colab output again or rerun Stage 7 training. Do not use generic ImageNet weights and do not use an MRL Eye checkpoint for mouth/yawn runtime inference.

## Remaining Checks After Download

Before Stage 14 runtime inference is considered unblocked, run a local checkpoint check:

1. `torch.load("checkpoints/resnet18_best.pt", map_location="cpu")`.
2. Confirm payload contains either `model_state_dict` or a directly loadable state dict.
3. Confirm keys resemble torchvision ResNet18.
4. Confirm classifier head is compatible with two output classes, ideally `fc.weight` shaped `[2, 512]`.
5. Confirm metadata if present:
   - `model` or `model_name`: `resnet18`
   - `class_to_index`: `{"no_yawn": 0, "yawn": 1}`
   - `image_size`: `224`
6. Only then rerun the Stage 14 audit/preflight.
