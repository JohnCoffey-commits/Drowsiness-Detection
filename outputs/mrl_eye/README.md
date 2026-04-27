# MRL Eye Stage 9 Outputs

These outputs were generated in Google Colab and synced into the local project repository.

- Original cloud location: `MyDrive/Drowsiness_Detection_Colab/outputs/mrl_eye/`
- Local location: `outputs/mrl_eye/`
- Stage: Stage 9 MRL Eye full training + Stage 9B model selection
- Dataset: MRL Eye
- Task: eye open/closed image classification
- Labels: `0 = closed`, `1 = open`
- Models: ResNet18, MobileNetV2, EfficientNet-B0
- Training environment: Google Colab, PyTorch / torchvision. A GPU runtime was used in Colab; the exact GPU model should be checked from the Colab runtime logs because it is not recorded in the synced report.

## Selected Model

- Primary selected model: MobileNetV2
- Default threshold: argmax / `p_eye_closed >= 0.50`
- Safety-prioritized reference: ResNet18 with validation-selected threshold around `0.30`

## Important Warning

These are specialist eye-state classification outputs, not final system-level driver drowsiness accuracy. The final driver monitoring system still needs runtime eye ROI consistency testing and later temporal fusion with the YawDD/YawDD+ Dash mouth/yawn module.

## Version-Control Note

Metrics, reports, figures, and error-analysis images may be committed if desired. Large checkpoints (`*.pt`) should generally remain local or be tracked with Git LFS, not normal Git.
