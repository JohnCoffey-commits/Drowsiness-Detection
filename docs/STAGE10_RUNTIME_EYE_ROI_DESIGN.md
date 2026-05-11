# Stage 10 Runtime Eye ROI Design

## Why Stage 10 Exists

The project is modular. The completed eye module is an MRL Eye open/closed specialist that outputs `p_eye_closed`; it is not a complete driver drowsiness classifier.

Stage 10 exists to test whether eye crops extracted at runtime from full-face frames or video are consistent enough to feed into the selected MRL Eye MobileNetV2 checkpoint before fusion or demo work begins.

## Why ROI Consistency Comes Before Fusion

The MRL Eye model was trained on eye crop images. A runtime system starts from full frames or video, so the eye region must first be detected and cropped. If runtime crops have different framing, scale, aspect ratio, contrast, or background context from the MRL Eye training crops, later temporal fusion can receive unstable `p_eye_closed` values even if the specialist model itself is valid.

Stage 10 therefore checks the interface between runtime ROI extraction and the trained eye specialist.

## MediaPipe FaceLandmarker

Stage 10 uses MediaPipe Tasks `FaceLandmarker`, consistent with the existing Stage 5 mouth ROI implementation in `src/preprocessing/generate_yawdd_mouth_crops.py`.

The default model asset is:

`artifacts/models/face_landmarker.task`

The runtime script detects one face per frame or image. If no face landmarks are found, the frame/image is logged as a failure row rather than silently dropped.

## Eye Crop Generation

The runtime script defines explicit MediaPipe Face Mesh landmark ID sets for left and right eyes. For each detected face, it:

1. Reads the landmark coordinates for one eye.
2. Computes the raw landmark bounding box.
3. Expands the box using `--eye-margin`, default `0.35`.
4. Clamps the box to frame boundaries.
5. Validates that the crop is non-empty.
6. Runs the selected MRL Eye model on each valid crop.

Runtime eye crops are not horizontally flipped. MediaPipe side names are anatomical, so a mirrored camera preview may visually reverse left and right. For Stage 10, deterministic crop geometry and logging are more important than perfect semantic side naming.

## Preprocessing Match To Stage 9

The Stage 10 model input preprocessing matches the Stage 9 MRL Eye evaluation transform:

```text
image.convert("RGB")
Resize(image_size + 16)
CenterCrop(image_size)
ToTensor()
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

The default `image_size` is `224`.

## Probability Semantics

The MRL Eye label mapping is:

```text
0 = closed
1 = open
```

The selected MobileNetV2 model outputs two logits in that order. Therefore:

```text
p_eye_closed = softmax(logits)[0]
p_eye_open = softmax(logits)[1]
```

The default decision rule remains:

```text
argmax / p_eye_closed >= 0.50
```

The runtime script exposes `--closed-threshold`, default `0.50`, for consistency checks.

## Not A Final Fatigue Score

Stage 10 does not produce final drowsiness accuracy. It does not produce a final fatigue score. It only verifies whether runtime eye ROIs can be extracted, transformed, and passed through the selected MRL Eye specialist while preserving inspectable crop geometry and probability traces.

Later fusion should combine `p_eye_closed` over time with the mouth/yawn module output `p_yawn`, using temporal smoothing or PERCLOS-like logic.

## Artifacts To Inspect Before Fusion Or Demo

The runtime output directory is:

`outputs/stage10_eye_roi_consistency/`

Inspect these artifacts before moving to fusion/demo:

- `runtime_eye_roi_predictions.csv`
- `failures.csv`
- `summary.json`
- `STAGE10_RUNTIME_EYE_ROI_REPORT.md`
- `contact_sheets/left_eye_samples.jpg`
- `contact_sheets/right_eye_samples.jpg`
- `contact_sheets/high_p_eye_closed.jpg`
- `contact_sheets/low_p_eye_closed.jpg`
- `contact_sheets/mixed_runtime_eye_samples.jpg`
- `debug_frames/` when `--save-debug-frames` is used
- `crops/` when `--save-crops` is used

## Known Risks

- Crop domain gap between MRL Eye training crops and runtime MediaPipe crops.
- Glasses and reflections, already observed in Stage 9B error analysis.
- Low light, blur, and poor contrast.
- Profile faces or partial faces.
- Motion blur in video.
- MediaPipe landmark failures.
- Runtime crops that include too much eyebrow, skin, glasses frame, or background.
- Runtime crops that are too tight and lose eyelid boundary context.

