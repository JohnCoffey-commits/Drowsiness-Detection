# Stage 14 Recovered Stage 7 Metadata Summary

## Source Files

- `artifacts/recovered_stage7_mouth_yawn/initial_results.csv`
- `artifacts/recovered_stage7_mouth_yawn/metrics_summary.json`
- `artifacts/recovered_stage7_mouth_yawn/resnet18_metrics-2.json`
- `artifacts/recovered_stage7_mouth_yawn/initial_experiment_summary.md`
- `artifacts/recovered_stage7_mouth_yawn/README_stage7_training.md`

## Selected Model Evidence

`initial_results.csv` ranks ResNet18 highest by test accuracy.

| Model | Test accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| ResNet18 | 0.993724 | 0.964744 | 0.978862 | 0.971751 |

The recovered `initial_experiment_summary.md` states that CNN-1 ResNet18 achieved the strongest test accuracy in the completed Stage 7 run.

## Model Metadata

| Field | Value |
| --- | --- |
| Selected model | `resnet18` |
| Best epoch | `4` |
| Image size | `224` |
| Batch size | `32` |
| Pretrained backbone used during training | `True` |
| Test confusion matrix | `[[9880, 44], [26, 1204]]` |

## Label Mapping

The recovered README records canonical Stage 7 labels as `no_yawn` and `yawn`. The local training source and checkpoint metadata define:

```text
{"no_yawn": 0, "yawn": 1}
```

Therefore:

```text
p_yawn = softmax(logits)[1]
```

## Transform Assumptions

From the Stage 7 training source and notebook:

- RGB conversion with `Image.open(...).convert("RGB")`
- Evaluation resize to `224 x 224`
- `ToTensor()`
- ImageNet normalization: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`

## Caveats

- This metadata supports runtime inference setup; it does not validate real-world mouth/yawn performance on the A/B/C/D videos by itself.
- Stage 14 must still perform runtime mouth ROI consistency checks and visual inspection.
- This is not final system-level drowsiness accuracy.
