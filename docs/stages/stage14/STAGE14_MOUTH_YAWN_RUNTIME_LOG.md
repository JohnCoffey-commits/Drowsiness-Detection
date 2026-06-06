# Stage 14 Mouth/Yawn Runtime Log

## Purpose

Stage 14 verifies runtime mouth ROI extraction and mouth/yawn specialist inference on A/B/C/D full-face videos. It is not final system-level drowsiness accuracy and not mouth-eye fusion.

## Checkpoint

- Source recovered checkpoint: `artifacts/recovered_stage7_mouth_yawn/resnet18_best.pt`
- Runtime checkpoint: `checkpoints/resnet18_best.pt`
- Verification: `CHECKPOINT_VERIFICATION_PASSED`
- Architecture: ResNet18 with two-class `fc` head
- Label mapping: `0 = no_yawn`, `1 = yawn`
- `p_yawn = softmax(logits)[1]`

## Validation Commands

```bash
source .venv-stage10/bin/activate
python -m py_compile src/runtime/stage14_mouth_yawn_runtime.py
python src/runtime/stage14_mouth_yawn_runtime.py --preflight
```

Both validation commands passed.

## Video Runs

| Video | Successful crops | Failures | Yawn events | Result |
| --- | ---: | ---: | ---: | --- |
| `A_normal_open_baseline` | 70 | 0 | 0 | passed |
| `B_realistic_drowsy_simulation` | 103 | 0 | 14 | passed |
| `C_mild_head_motion` | 89 | 6 | 0 | passed |
| `D_controlled_long_open_closed` | 119 | 0 | 0 | passed |

## B Manual Interval Check

For `B_realistic_drowsy_simulation`, the user manually confirmed yawning from approximately `14.3s` to `16.8s`. Stage 14 produced 12/12 yawn-event rows in that sampled interval, with mean `p_yawn = 0.981091` and max `p_yawn = 0.997966`.

## Outputs

- `outputs/stage14_mouth_yawn_runtime_multi_video_summary.csv`
- `outputs/stage14_mouth_yawn_runtime_multi_video_summary.json`
- `reports/stage14_mouth_yawn_runtime_validation_report.md`

## Next Step

Human visual inspection of mouth contact sheets and debug frames is required. If accepted, Stage 15 can use the real Stage 14 `p_yawn` timelines for synchronized mouth-eye fusion validation.

Warning: this is not final system-level drowsiness accuracy.
