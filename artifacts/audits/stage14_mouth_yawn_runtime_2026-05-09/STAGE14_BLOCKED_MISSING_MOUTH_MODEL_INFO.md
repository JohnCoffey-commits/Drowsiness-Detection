# Stage 14 Blocked: Missing Mouth/Yawn Checkpoint

Stage 14 runtime mouth/yawn inference was not implemented because a usable trained selected YawDD/YawDD+ Dash mouth/yawn checkpoint was not found locally.

## Blocking Item

Missing required artifact:

```text
trained Stage 7 mouth/yawn checkpoint
```

The expected checkpoint is likely equivalent to:

```text
checkpoints/resnet18_best.pt
```

or the completed Stage 7 Google Drive output:

```text
outputs/checkpoints/resnet18_best.pt
```

but no such trained mouth/yawn checkpoint exists in the local repository.

## What Is Known

- Selected mouth/yawn architecture in project status: ResNet18.
- Model construction: `torchvision.models.resnet18(...)` with `model.fc = nn.Linear(model.fc.in_features, 2)`.
- Label mapping: `0 = no_yawn`, `1 = yawn`.
- `p_yawn` class index: `1`.
- `p_yawn` computation, once a valid checkpoint is available: `softmax(logits)[1]`.
- Evaluation preprocessing: RGB image, resize to `224 x 224`, `ToTensor`, ImageNet normalization.

## What Is Missing

- Local trained mouth/yawn checkpoint.
- Verified checkpoint loading path for the selected Stage 7 model.

## Why Runtime Inference Was Not Implemented

Using generic torchvision pretrained weights or an eye checkpoint would not produce a valid mouth/yawn probability. Creating a Stage 14 runtime script with a fabricated or unverified checkpoint would make downstream `p_yawn` timelines invalid.

## Required Next Action

Restore or provide the completed Stage 7 selected mouth/yawn checkpoint locally, then rerun the Stage 14 audit.

Recommended target path:

```text
checkpoints/resnet18_best.pt
```

or another explicit path documented as the selected Stage 7 YawDD/YawDD+ Dash mouth/yawn checkpoint.

After the checkpoint is available, Stage 14 can implement runtime mouth ROI consistency validation and automatic `p_yawn` timeline generation.

## Warning

No model was trained. No checkpoint was modified. No final system-level drowsiness accuracy is claimed.
