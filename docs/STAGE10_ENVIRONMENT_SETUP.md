# Stage 10 Environment Setup

## 1. Purpose

This document records the local Python environment setup used to validate the Stage 10 runtime eye ROI consistency preflight. The setup is for dependency validation and runtime readiness only.

No model training was run. No checkpoints were modified. No Stage 8 or Stage 9 outputs were modified. SystemUI was not modified or connected to the Python runtime.

## 2. Interpreter Selected

Selected interpreter:

```text
/Users/zhengpeixian/miniforge3/bin/python3.12
Python 3.12.11
```

Python 3.12 was selected because it was available and preferred over the existing repository `.venv`, which was reported to use Python 3.14.

## 3. Environment Path

Dedicated Stage 10 virtual environment:

```text
/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/.venv-stage10
```

## 4. Install Commands Used

```bash
python3.12 -m venv .venv-stage10
source .venv-stage10/bin/activate
.venv-stage10/bin/python -m pip install --upgrade pip setuptools wheel
.venv-stage10/bin/pip install mediapipe opencv-python pillow numpy pandas matplotlib scikit-learn torch torchvision
```

The first pip upgrade attempt inside the sandbox could not reach PyPI due DNS/network restrictions. The same install was rerun with approved network access and completed successfully.

## 5. Installed Package Versions

Import check result from `.venv-stage10`:

```text
Python: 3.12.11
mediapipe: 0.10.35
opencv-python / cv2: 4.13.0.92 / 4.13.0
pillow: 12.2.0
numpy: 2.4.4
pandas: 3.0.2
matplotlib: 3.10.9
scikit-learn: 1.8.0
torch: 2.11.0
torchvision: 0.26.0
```

Full package freeze:

```text
artifacts/audits/stage10_environment_setup_2026-05-09/pip_freeze_stage10.txt
```

## 6. CUDA Availability

```text
cuda_available False
```

## 7. MPS Availability

```text
mps_available False
```

Stage 10 preflight selected CPU.

## 8. py_compile Result

Command:

```bash
.venv-stage10/bin/python -m py_compile src/runtime/stage10_eye_roi_consistency.py
```

Result:

```text
exit_code=0
```

## 9. Stage 10 Preflight Result

Command:

```bash
.venv-stage10/bin/python src/runtime/stage10_eye_roi_consistency.py --preflight
```

Result:

```text
exit_code=0
```

The preflight validated dependency imports, required assets, MobileNetV2 construction, checkpoint loading from `payload["state_dict"]`, Stage 9 evaluation transform construction, and the existing 0=closed / 1=open label mapping used for `p_eye_closed = softmax(logits)[0]`.

## 10. summary.json Path

```text
/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/outputs/stage10_eye_roi_consistency/summary.json
```

The summary mode is `preflight_only`; no images or videos were processed.

## 11. Warnings

- The first pip upgrade attempt failed under the restricted sandbox network, then succeeded after approved network access.
- pip reported that `/Users/zhengpeixian/Library/Caches/pip` was not writable, so pip caching was disabled.
- Matplotlib reported that `/Users/zhengpeixian/.matplotlib` was not writable and used a temporary cache directory. This did not block imports or preflight.
- `setuptools` was installed as `81.0.0` after dependency resolution for the selected package set.

## 12. Audit Evidence

Environment setup artifacts:

```text
/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/artifacts/audits/stage10_environment_setup_2026-05-09/stage10_environment_commands.log
/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/artifacts/audits/stage10_environment_setup_2026-05-09/stage10_import_check.txt
/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/artifacts/audits/stage10_environment_setup_2026-05-09/stage10_preflight_output.txt
/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/artifacts/audits/stage10_environment_setup_2026-05-09/pip_freeze_stage10.txt
```

This setup proves that the local Stage 10 runtime environment can import the required packages, construct and load the selected MRL Eye MobileNetV2 checkpoint, and complete preflight validation. It does not prove runtime ROI quality, demo behavior, fusion accuracy, fatigue scoring, or final system-level drowsiness accuracy.
