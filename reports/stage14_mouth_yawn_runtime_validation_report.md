# Stage 14 Mouth/Yawn Runtime Validation Report

## 1. Purpose

Stage 14 validates runtime mouth ROI extraction and mouth/yawn specialist inference on full-face videos. It produces timestamped `p_yawn` timelines from the recovered Stage 7 YawDD/YawDD+ Dash mouth/yawn checkpoint.

This is not final system-level drowsiness accuracy and not mouth-eye fusion yet.

## 2. Checkpoint Verification Result

- Checkpoint copied to: `checkpoints/resnet18_best.pt`
- Verification result: `CHECKPOINT_VERIFICATION_PASSED`
- Payload key: `model_state_dict`
- Classifier head: `fc.weight` shape `(2, 512)`, `fc.bias` shape `(2,)`
- Dummy inference output: logits/probabilities shape `(1, 2)`

## 3. Model Architecture and Semantics

- Architecture: torchvision ResNet18 with `model.fc = nn.Linear(model.fc.in_features, 2)`.
- Label mapping: `0 = no_yawn`, `1 = yawn`.
- `p_yawn` class index: `1`.
- `p_yawn = softmax(logits)[1]`.
- Evaluation transform: RGB image, resize `224 x 224`, `ToTensor`, ImageNet normalization.

## 4. Method

- Input: full-face A/B/C/D videos.
- Face landmarks: MediaPipe Tasks FaceLandmarker.
- ROI: mouth/lip landmarks from the same MediaPipe topology used in Stage 5 mouth crop generation.
- Inference: recovered Stage 7 ResNet18 mouth/yawn specialist.
- Output: `p_yawn`, `p_no_yawn`, `yawn_event`, `recent_yawn_event`, crops, debug frames, contact sheets, and figures.

## 5. Per-Video Results

| Video | Sampled frames | Successful crops | Failures | No-face | Mean p_yawn | Max p_yawn | Yawn events | Recent-yawn rows | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `A_normal_open_baseline` | 70 | 70 | 0 | 0 | 0.216834 | 0.239631 | 0 | 0 | passed |
| `B_realistic_drowsy_simulation` | 103 | 103 | 0 | 0 | 0.149097 | 0.997966 | 14 | 36 | passed |
| `C_mild_head_motion` | 95 | 89 | 6 | 6 | 0.224402 | 0.289967 | 0 | 0 | passed |
| `D_controlled_long_open_closed` | 119 | 119 | 0 | 0 | 0.250691 | 0.270475 | 0 | 0 | passed |

## 6. B Manual Expectation Comparison

The user manually confirmed yawning in `B_realistic_drowsy_simulation.mp4` from approximately `14.3s` to `16.8s`.

- Sampled rows in 14.3-16.8s window: 12
- Yawn-event rows in that window: 12/12
- Mean `p_yawn` in that window: 0.981091
- Min `p_yawn` in that window: 0.950278
- Max `p_yawn` in that window: 0.997966

The model clearly elevated `p_yawn` in and immediately around the manually confirmed interval. It also triggered two yawn-event rows just before 14.3s (`13.964501s` and `14.172926s`), so visual review should confirm whether those sampled frames show the start of the yawn.

Selected B timeline rows around the manual interval:

| timestamp_sec | frame_index | p_yawn | predicted_label | yawn_event | recent_yawn_event |
| ---: | ---: | ---: | --- | --- | --- |
| 13.130799 | 315 | 0.243770 | `no_yawn` | False | False |
| 13.339225 | 320 | 0.043308 | `no_yawn` | False | False |
| 13.547650 | 325 | 0.077681 | `no_yawn` | False | False |
| 13.756076 | 330 | 0.485093 | `no_yawn` | False | False |
| 13.964501 | 335 | 0.739645 | `yawn` | True | True |
| 14.172926 | 340 | 0.921275 | `yawn` | True | True |
| 14.381352 | 345 | 0.950278 | `yawn` | True | True |
| 14.589777 | 350 | 0.956470 | `yawn` | True | True |
| 14.798203 | 355 | 0.959170 | `yawn` | True | True |
| 15.006628 | 360 | 0.994702 | `yawn` | True | True |
| 15.215053 | 365 | 0.996596 | `yawn` | True | True |
| 15.423479 | 370 | 0.995108 | `yawn` | True | True |
| 15.631904 | 375 | 0.997966 | `yawn` | True | True |
| 15.840329 | 380 | 0.996374 | `yawn` | True | True |
| 16.048755 | 385 | 0.987613 | `yawn` | True | True |
| 16.257180 | 390 | 0.988446 | `yawn` | True | True |
| 16.465606 | 395 | 0.961307 | `yawn` | True | True |
| 16.674031 | 400 | 0.989060 | `yawn` | True | True |
| 16.882456 | 405 | 0.015969 | `no_yawn` | False | True |
| 17.090882 | 410 | 0.000444 | `no_yawn` | False | True |
| 17.299307 | 415 | 0.001058 | `no_yawn` | False | True |
| 17.507733 | 420 | 0.004663 | `no_yawn` | False | True |
| 17.716158 | 425 | 0.012930 | `no_yawn` | False | True |
| 17.924583 | 430 | 0.005789 | `no_yawn` | False | True |
| 18.133009 | 435 | 0.007728 | `no_yawn` | False | True |
| 18.341434 | 440 | 0.001819 | `no_yawn` | False | True |

## 7. Visual Inspection Requirement

A human must inspect the Stage 14 contact sheets and debug frames before treating the runtime mouth signal as accepted. Mouth crops must show the mouth region, and high `p_yawn` crops should visually correspond to yawning or mouth-open/yawn-like frames.

Key folders:

- `outputs/stage14_mouth_yawn_runtime_A_normal_open_baseline/contact_sheets/`
- `outputs/stage14_mouth_yawn_runtime_A_normal_open_baseline/debug_frames/`
- `outputs/stage14_mouth_yawn_runtime_B_realistic_drowsy_simulation/contact_sheets/`
- `outputs/stage14_mouth_yawn_runtime_B_realistic_drowsy_simulation/debug_frames/`
- `outputs/stage14_mouth_yawn_runtime_C_mild_head_motion/contact_sheets/`
- `outputs/stage14_mouth_yawn_runtime_C_mild_head_motion/debug_frames/`
- `outputs/stage14_mouth_yawn_runtime_D_controlled_long_open_closed/contact_sheets/`
- `outputs/stage14_mouth_yawn_runtime_D_controlled_long_open_closed/debug_frames/`

## 8. Limitations

- Small validation set.
- No final drowsiness labels.
- Not final system-level drowsiness accuracy.
- Not mouth-eye fusion yet.
- Runtime mouth ROI may fail under occlusion, head pose, or low light.
- The mouth/yawn specialist was trained on YawDD/YawDD+ Dash mouth crops, not necessarily these runtime videos.

## 9. Next Step

If the user accepts the visual crop/debug-frame quality, Stage 15 can rerun Stage 13 using real Stage 14 `p_yawn` timelines for synchronized mouth-eye fusion validation.
