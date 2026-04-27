# MRL Eye Stage 9 Experiment Summary

The MRL Eye module is an eye open/closed specialist. It reports per-frame `p_eye_closed` and `p_eye_open` for later temporal fusion; it is not a full drowsiness classifier.

## Results

| Model | Pretrained | Val macro F1 | Test macro F1 (argmax) | Test closed recall (argmax) | Val-selected threshold | Test macro F1 (fixed threshold) | Test closed recall (fixed threshold) | False open (fixed threshold) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| resnet18 | True | 0.9837 | 0.9846 | 0.9859 | 0.30 | 0.9760 | 0.9908 | 58 |
| mobilenet_v2 | True | 0.9791 | 0.9863 | 0.9852 | 0.30 | 0.9848 | 0.9879 | 76 |
| efficientnet_b0 | True | 0.9791 | 0.9862 | 0.9824 | 0.30 | 0.9852 | 0.9865 | 85 |

## Threshold Notes

Threshold candidates are selected from the validation sweep only. The test sweep is saved for final reporting and audit, not for choosing the threshold.

- `resnet18`: Candidate threshold: 0.30. It keeps macro F1 at 0.9792 and closed recall at 0.9953. This is only a candidate for later temporal fusion. Selected from validation: 0.30. Applied unchanged to test: macro F1 0.9760, closed recall 0.9908, false-open count 58.
- `mobilenet_v2`: Candidate threshold: 0.30. It keeps macro F1 at 0.9755 and closed recall at 0.9871. This is only a candidate for later temporal fusion. Selected from validation: 0.30. Applied unchanged to test: macro F1 0.9848, closed recall 0.9879, false-open count 76.
- `efficientnet_b0`: Candidate threshold: 0.30. It keeps macro F1 at 0.9757 and closed recall at 0.9904. This is only a candidate for later temporal fusion. Selected from validation: 0.30. Applied unchanged to test: macro F1 0.9852, closed recall 0.9865, false-open count 85.

Closed-eye recall should be reviewed alongside macro F1. A threshold that predicts nearly everything as closed is not useful for deployment even if it reduces false-open errors in isolation.

Next step: run the selected Stage 9 models in Colab, inspect error contact sheets, then pass `p_eye_closed` into Stage 10/11 temporal smoothing or PERCLOS-like fusion with the YawDD mouth/yawn module.
