# Stage 10 Repository Audit

Audit date: 2026-05-09  
Scope: read-only repository audit for Stage 10 runtime eye ROI consistency planning.  
Note: the repository snapshot below was taken before writing this audit report and the four `codex_stage10_*` supporting audit files.

## 1. Repository health snapshot

Current working directory:

`/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection`

Git status summary:

```text
?? SystemUI/
?? docs/PROJECT_LEARNING_GUIDE.md
?? git_log_last10.txt
?? git_status.txt
?? repo_dir_list.txt
?? repo_file_list.txt
?? repo_pwd.txt
?? repo_tree.txt
```

Last 10 commits requested; this local history currently shows 8 commits:

```text
4e7b5e3 Add MRL eye stage 9b outputs, Colab notebooks, and project docs
cc9cbb7 Update stage9 MRL Eye Colab notebook
bc348c0 Update stage9 MRL Eye Colab training notebook
1d280ba Refine MRL Eye Stage 9 training pipeline
4e9d36e Add MRL Eye Stage 8 preparation artifacts
1aab023 Add MRL Eye and NTHUDDD2 Kaggle manifests, splits, and Colab training notebooks
55d91fc Stage 7: training docs and classifier/baseline training updates
06db338 Backup core pipeline before Stage 7
```

Repo dirty: YES. The working tree has untracked items.

Untracked item relevance to Stage 10:

| Item | Stage 10 relevance |
| --- | --- |
| `SystemUI/` | Potentially related to a future demo UI, but it is untracked and appears separate from the Python ML pipeline. It uses mock dashboard data and does not load the MobileNetV2 checkpoint. |
| `docs/PROJECT_LEARNING_GUIDE.md` | Relevant background documentation. It summarizes the modular project state and Stage 10 direction, but it is untracked. |
| `git_log_last10.txt`, `git_status.txt`, `repo_dir_list.txt`, `repo_file_list.txt`, `repo_pwd.txt`, `repo_tree.txt` | Snapshot artifacts from prior repository inspection. Useful for audit context, not Stage 10 runtime implementation code. |

SystemUI assessment: `SystemUI/` appears to be a separate untracked Next.js dashboard/demo UI. It has a `live-monitor` page, mock `p_eye_closed`-style settings, ROI preview cards, and a static video card, but no Python runtime bridge, no MediaPipe execution, and no checkpoint loading. It is potentially part of a later demo surface, not the Stage 10 Python ML runtime pipeline.

## 2. Required Stage 10 prerequisite files

| Path | Status | Size bytes |
| --- | --- | ---: |
| `outputs/mrl_eye/checkpoints/best_mobilenet_v2_mrl_eye.pt` | FOUND | 9156491 |
| `reports/mrl_eye_stage9b_error_analysis.md` | FOUND | 10016 |
| `docs/PROJECT_STRUCTURE.md` | FOUND | 12736 |
| `docs/PROJECT_CURRENT_STATUS.md` | FOUND | 16620 |
| `requirements.txt` | FOUND | 86 |
| `outputs/mrl_eye/results/mrl_eye_stage9b_model_selection.json` | FOUND | 1038 |
| `outputs/mrl_eye/results/mrl_eye_stage9b_model_comparison.csv` | FOUND | 827 |
| `outputs/mrl_eye/results/mobilenet_v2_metrics.json` | FOUND | 4111 |
| `outputs/mrl_eye/results/mobilenet_v2_val_threshold_sweep.csv` | FOUND | 1606 |
| `outputs/mrl_eye/results/mobilenet_v2_test_threshold_sweep.csv` | FOUND | 1597 |
| `artifacts/mappings/mrl_eye_trainable_with_split.csv` | FOUND | 24606502 |
| `artifacts/splits/mrl_eye_subject_split.csv` | FOUND | 2349 |

Additional Stage 10-relevant asset found:

| Path | Status | Size bytes |
| --- | --- | ---: |
| `artifacts/models/face_landmarker.task` | FOUND | 3758596 |

## 3. Existing inference/demo/runtime code

Exact path-name matches found:

| Path | Purpose |
| --- | --- |
| `SystemUI/src/app/live-monitor/` | Next.js route directory for a live-monitor dashboard page. It presents tracking status, model signals, ROI previews, and runtime events from mock data. |
| `SystemUI/src/app/live-monitor/page.tsx` | Live Monitor dashboard implementation. It displays left/right eye ROI status and an eye-closure probability, but reads `liveMonitorData` from mocks. |
| `SystemUI/src/components/dashboard/LiveVideoCard.tsx` | UI video card with a static Unsplash image, drawn face/eye/mouth boxes, FPS display, and no real video/model integration. |

Relevant content-only matches:

| Path | Purpose |
| --- | --- |
| `SystemUI/src/lib/mockData.ts` | Defines mock live-monitor data, demo-video/webcam settings, selected eye model `MobileNetV2`, and threshold text `argmax / p_eye_closed >= 0.50`. |
| `SystemUI/src/app/page.tsx` | Imports the mock `LiveVideoCard` on the dashboard entry page. |
| `SystemUI/src/app/session-review/page.tsx` | Contains a session video placeholder, not a runtime inference path. |
| `src/training/train_mrl_eye_baselines.py` | Has `predict(...)` for offline train/val/test evaluation and threshold sweeps. It is not a runtime CLI for full-face frames or videos. |
| `src/training/train_classifier.py` | Has offline classifier evaluation helpers for earlier training work. |
| `src/data/inspect_yawdd_raw_dash.py` | Offline inspection of YawDD raw video files without frame decoding. |
| `src/data/build_yawdd_dash_mapping.py` | Offline mapping from YawDD+ annotations to raw YawDD Dash videos. |
| `src/data/audit_yawdd_dash_framecounts.py` | Offline video frame-count audit. |
| `src/data/validate_yawdd_dash_frames.py` | Offline selected-frame validation from raw videos. |
| `src/data/extract_yawdd_dash_labeled_frames.py` | Offline YawDD Dash labeled-frame extraction. |
| `src/data/build_yawdd_split.py` | Carries raw video metadata into YawDD split artifacts. |

No existing Python inference/demo/runtime pipeline found.

## 4. MediaPipe dependency and usage

MediaPipe in `requirements.txt`: FOUND.

Pinned version: none. The line is plain `mediapipe`, so the exact version is not pinned by the repository dependency file.

Every source file using the MediaPipe API:

| Source file | API style | Functions/classes around use | ROI support |
| --- | --- | --- | --- |
| `src/preprocessing/generate_yawdd_mouth_crops.py` | `mediapipe.tasks.python.vision.FaceLandmarker` | Imports `mediapipe as mp`, `BaseOptions`, `FaceLandmarker`, `FaceLandmarkerOptions`, `RunningMode`; defines `MOUTH_LANDMARK_IDS`, `mouth_bbox_from_landmarks(...)`, `build_landmarker(...)`, and uses `mp.Image(...); landmarker.detect(...)`. | Mouth ROI only. |
| `src/preprocessing/precompute_yawdd_mouth_crops.py` | `mp.solutions.face_mesh.FaceMesh` | Imports `mediapipe as mp`; defines `MOUTH_LANDMARKS`, `mouth_box_from_landmarks(...)`, `lower_face_fallback_box(...)`, `detect_face_box(...)`, and calls `face_mesh.process(image_rgb)`. | Mouth ROI only. |

Important code locations:

- `src/preprocessing/generate_yawdd_mouth_crops.py:43-49` imports MediaPipe Tasks classes.
- `src/preprocessing/generate_yawdd_mouth_crops.py:55-62` defines only mouth landmark IDs.
- `src/preprocessing/generate_yawdd_mouth_crops.py:111-132` computes a mouth bbox from landmarks.
- `src/preprocessing/generate_yawdd_mouth_crops.py:157-166` builds a `FaceLandmarker`.
- `src/preprocessing/generate_yawdd_mouth_crops.py:324-339` runs Face Mesh detection and calls `mouth_bbox_from_landmarks(...)`.
- `src/preprocessing/precompute_yawdd_mouth_crops.py:12-35` defines only mouth landmarks.
- `src/preprocessing/precompute_yawdd_mouth_crops.py:84-94` computes a mouth box.
- `src/preprocessing/precompute_yawdd_mouth_crops.py:172-187` uses `mp.solutions.face_mesh.FaceMesh`.

Current code does not implement eye ROI extraction. The available MediaPipe code can detect full-face landmarks, but all implemented landmark sets and bbox functions target the mouth/lower face.

## 5. MRL Eye training preprocessing audit

This section is based on `src/training/train_mrl_eye_baselines.py`, the executed Stage 9 Colab notebook record, and the MRL Eye manifest/report files.

Input image size:

- Training CLI default is `--image-size 224` in `src/training/train_mrl_eye_baselines.py:1038`.
- The Stage 9 Colab run record uses `--image-size 224` in `colab_file/stage9_mrl_eye_training_r.ipynb` grep output.
- Therefore the trained Stage 9 MRL Eye run used 224 x 224 model input.

RGB vs grayscale:

- `MRLEyeDataset.__getitem__` opens each image and calls `.convert("RGB")` at `src/training/train_mrl_eye_baselines.py:254`.
- There is no grayscale transform in the training pipeline.

Train transforms:

From `src/training/train_mrl_eye_baselines.py:260-273`:

```text
transforms.RandomResizedCrop(image_size, scale=(0.88, 1.0), ratio=(0.90, 1.10))
transforms.RandomRotation(10)
transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.06))
transforms.RandomHorizontalFlip(p=0.5)
transforms.ColorJitter(brightness=0.20, contrast=0.20)
transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.15)
transforms.ToTensor()
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

Val/test transforms:

From `src/training/train_mrl_eye_baselines.py:275-281`:

```text
transforms.Resize(image_size + 16)
transforms.CenterCrop(image_size)
transforms.ToTensor()
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

Normalization:

- Mean: `[0.485, 0.456, 0.406]`
- Std: `[0.229, 0.224, 0.225]`
- Location: `src/training/train_mrl_eye_baselines.py:261`.

Class mapping and label order:

- `LABEL_TO_NAME = {0: "closed", 1: "open"}` and `NAME_TO_LABEL = {"closed": 0, "open": 1}` in `src/training/train_mrl_eye_baselines.py:84-85`.
- The file docstring fixes labels as `0 = closed`, `1 = open` in `src/training/train_mrl_eye_baselines.py:9-12`.
- The MRL Eye manifest builder also defines `LABEL_NAME_MAP = {0: "closed", 1: "open"}` in `src/data/build_mrl_eye_manifest.py:47`.
- `reports/mrl_eye_dataset_report.md:3-4` states that `annotation.txt` confirms `0=closed and 1=open`.

How `p_eye_closed` is computed:

- `predict(...)` computes `probs = torch.softmax(logits, dim=1)` and default predictions with `probs.argmax(axis=1)` at `src/training/train_mrl_eye_baselines.py:449-450`.
- Threshold sweeps set `p_closed = probs[:, 0]` and predict closed when `p_closed >= threshold` at `src/training/train_mrl_eye_baselines.py:502-506`.
- `metrics_at_closed_threshold(...)` also uses `np.where(probs[:, 0] >= threshold, 0, 1)` at `src/training/train_mrl_eye_baselines.py:564-570`.
- Error contact sheets write `p_eye_closed = predictions["probs"][:, 0]` and `p_eye_open = predictions["probs"][:, 1]` at `src/training/train_mrl_eye_baselines.py:710-713`.

Conclusion: label 0 means closed, label 1 means open, and closed probability is `softmax(logits)[0]`, not index 1.

## 6. MRL Eye model architecture and checkpoint loading audit

Exact torchvision model name for MobileNetV2:

- `models.mobilenet_v2(weights=weights)` at `src/training/train_mrl_eye_baselines.py:338-340`.

Pretrained weights:

- `try_torchvision_weights("mobilenet_v2", pretrained=True)` returns `models.MobileNet_V2_Weights.DEFAULT` at `src/training/train_mrl_eye_baselines.py:316-324`.
- Stage 9 executed with `--require-pretrained`, and `outputs/mrl_eye/results/mobilenet_v2_metrics.json` reports `"pretrained_requested": true`, `"pretrained_required": true`, and `"pretrained_loaded": true`.
- `outputs/mrl_eye/results/mrl_eye_stage9b_model_comparison.csv` also reports `pretrained_loaded=True` for MobileNetV2.

Classifier head replacement and output classes:

- MobileNetV2 replaces the last classifier layer with `nn.Linear(model.classifier[-1].in_features, 2)` at `src/training/train_mrl_eye_baselines.py:338-340`.
- Number of output classes: 2, ordered `[closed, open]`.

Checkpoint save format:

- `checkpoint_payload(...)` returns a dict with `model_name`, `epoch`, `best_val_macro_f1`, `state_dict`, `label_mapping`, `image_size`, and `outputs` at `src/training/train_mrl_eye_baselines.py:734-749`.
- The checkpoint is not a plain state_dict.
- It is not shown to contain optimizer state, scheduler state, or training metadata beyond the fields above.
- The key is `state_dict`, not `model_state_dict`.

Checkpoint selection metric:

- During training, a checkpoint is saved when `val_metrics["macro_f1"] > best_val_macro_f1` at `src/training/train_mrl_eye_baselines.py:832-839`.
- Selection metric: validation macro F1.

Expected checkpoint path:

- Local selected checkpoint: `outputs/mrl_eye/checkpoints/best_mobilenet_v2_mrl_eye.pt`.
- Stage 9B model-selection JSON records this as `stage10_checkpoint_required` and confirms `stage10_checkpoint_found: true`.

Existing checkpoint reload code:

- In `src/training/train_mrl_eye_baselines.py:849-851`, the training script reloads an existing checkpoint with `payload = torch.load(checkpoint_path, map_location=device)` and `model.load_state_dict(payload["state_dict"])`.
- There is no separate runtime loader script for Stage 10 yet.

## 7. Stage 9B model-selection audit

Files inspected:

- `reports/mrl_eye_stage9b_error_analysis.md`
- `outputs/mrl_eye/results/mrl_eye_stage9b_model_selection.json`
- `outputs/mrl_eye/results/mrl_eye_stage9b_model_comparison.csv`

Selected primary model:

- `mobilenet_v2`.
- Report location: `reports/mrl_eye_stage9b_error_analysis.md:119-123`.
- JSON field: `"primary_selected_model": "mobilenet_v2"`.

Default threshold/rule:

- Default threshold: `0.50`.
- Default rule: `argmax / p_eye_closed >= 0.50`.
- Report location: `reports/mrl_eye_stage9b_error_analysis.md:121-124`.
- JSON field: `"recommended_default_threshold": 0.5`, `"recommended_default_rule": "argmax / p_eye_closed >= 0.50"`.

Safety reference model and threshold:

- Safety-prioritized reference: ResNet18 with `p_eye_closed >= 0.30`.
- Report location: `reports/mrl_eye_stage9b_error_analysis.md:125`.
- JSON field: `"safety_prioritized_reference": {"model": "resnet18", "threshold": 0.3, ...}`.

Stage 10 readiness status:

- `READY`.
- Report location: `reports/mrl_eye_stage9b_error_analysis.md:143-151`.
- JSON field: `"stage10_status": "READY"`.

False-open and false-closed definitions:

- `false_open`: true closed, predicted open. This is safety-critical because a closed-eye frame is missed.
- `false_closed`: true open, predicted closed. This is a false alarm tendency.
- Report location: `reports/mrl_eye_stage9b_error_analysis.md:11-14`.
- Training metrics implement these as `cm[0, 1]` and `cm[1, 0]` at `src/training/train_mrl_eye_baselines.py:482-498`.

Specialist-only warning:

- `reports/mrl_eye_stage9b_error_analysis.md:7` states these are specialist eye-state classification results, not final system-level driver drowsiness accuracy.
- `reports/mrl_eye_stage9b_error_analysis.md:159-161` repeats that these are MRL Eye specialist results only.
- `outputs/mrl_eye/README.md` also warns that final driver monitoring still needs runtime eye ROI consistency testing and later temporal fusion.

## 8. Runtime eye ROI design constraints inferred from existing code

Reusable directly:

- MediaPipe Tasks setup from `src/preprocessing/generate_yawdd_mouth_crops.py`: `FaceLandmarkerOptions`, `RunningMode.IMAGE`, `mp.Image`, `landmarker.detect(...)`, model-asset path handling, and face-detection fallback logging patterns.
- Bbox clamping from `clamp_bbox(...)`.
- MRL Eye transform semantics from `build_transforms(...)`: RGB input, 224 image size, eval resize to 240 then center crop to 224, ImageNet normalization.
- MobileNetV2 architecture construction pattern from `build_model(...)`.
- Checkpoint load pattern using `payload["state_dict"]`.

Must be newly implemented:

- Eye landmark ID sets for left and right eye, including a documented bbox policy and margin.
- Eye crop extraction from full-face frames/videos.
- A Stage 10 inference/evaluation CLI that reads image/video sources, runs FaceLandmarker, crops eyes, applies the exact eval transform, loads the selected MobileNetV2 checkpoint, and emits per-eye `p_eye_closed` values.
- Runtime output artifacts: CSV, contact sheets, visual overlays, failure logs, and summary JSON/Markdown.
- MPS device support if desired; existing Stage 9 code supports CUDA then CPU only.

Eye landmarks currently absent or present:

- Present indirectly: FaceLandmarker returns full face landmarks that include eye regions.
- Absent in code: explicit left-eye/right-eye landmark ID constants, eye bbox functions, eye crop writer, and any runtime eye ROI consistency evaluator.

Recommended Stage 10 output artifacts:

- `outputs/stage10_eye_roi_consistency/summary.json`
- `outputs/stage10_eye_roi_consistency/runtime_eye_roi_predictions.csv`
- `outputs/stage10_eye_roi_consistency/failures.csv`
- `outputs/stage10_eye_roi_consistency/contact_sheets/*.jpg`
- `outputs/stage10_eye_roi_consistency/debug_frames/*.jpg`
- `outputs/stage10_eye_roi_consistency/STAGE10_RUNTIME_EYE_ROI_REPORT.md`

Top ROI consistency risks:

- MRL Eye training images are already eye crops; runtime full-face crops may include too much eyebrow/skin/glasses frame or miss eyelid boundaries.
- Runtime crop aspect ratio and margin may differ from MRL Eye crop geometry; the eval transform will resize and center crop, which can hide systematic framing errors.
- MRL Eye images are opened as RGB but may visually be close to grayscale; runtime full-frame color/white balance can introduce a domain gap.
- Left/right eye orientation and horizontal flipping need a consistent policy. Training used random horizontal flip, but runtime output should still record side and bbox.
- Glasses/reflections were noted as common high-confidence error patterns in Stage 9B.
- MediaPipe face failures, partial faces, profile views, motion blur, and low-light frames need explicit failure logging instead of silent model calls.

## 9. Recommended Stage 10 implementation plan for Codex

Do not write code yet. Proposed implementation plan:

Proposed new files:

- `src/runtime/stage10_eye_roi_consistency.py`
- `src/runtime/__init__.py`
- `docs/STAGE10_RUNTIME_EYE_ROI_DESIGN.md` if design notes are wanted separately
- `tests/` only if a lightweight non-model unit test surface is added later

Proposed output directory:

- `outputs/stage10_eye_roi_consistency/`

Suggested CLI arguments:

```text
--input-images <path or glob>
--input-video <path>
--sample-every-n-frames <int>
--max-frames <int>
--checkpoint outputs/mrl_eye/checkpoints/best_mobilenet_v2_mrl_eye.pt
--face-landmarker artifacts/models/face_landmarker.task
--output-dir outputs/stage10_eye_roi_consistency
--image-size 224
--device auto
--closed-threshold 0.50
--save-crops
--save-debug-frames
--contact-sheet-max 64
```

CSV schema:

```text
source_id
source_path
source_type
frame_index
timestamp_sec
face_detected
num_faces
eye_side
eye_bbox_xyxy
eye_crop_path
debug_frame_path
crop_width
crop_height
crop_aspect_ratio
landmark_ids
crop_method
model_name
checkpoint_path
device
p_eye_closed
p_eye_open
pred_label
closed_threshold
decision_rule
status
error
```

Contact sheet outputs:

- `contact_sheets/left_eye_samples.jpg`
- `contact_sheets/right_eye_samples.jpg`
- `contact_sheets/high_p_eye_closed.jpg`
- `contact_sheets/low_p_eye_closed.jpg`
- `contact_sheets/failures_or_low_confidence.jpg`

Visual debug outputs:

- Full frame overlays with face landmarks sparsely sampled, left/right eye bboxes, side labels, and `p_eye_closed`.
- Optional crop panels showing raw crop and transformed 224 x 224 view.

Model loading behavior:

- Instantiate `models.mobilenet_v2(weights=None)`.
- Replace classifier head with `nn.Linear(model.classifier[-1].in_features, 2)`.
- Load `torch.load(checkpoint, map_location=device)`.
- Require dict payload with `state_dict`; validate optional metadata fields `model_name`, `label_mapping`, `image_size`, and `outputs`.
- Load `payload["state_dict"]`, set `model.eval()`, and run under `torch.no_grad()`.

CPU/MPS/CUDA handling:

- `--device auto` should prefer CUDA, then MPS when `torch.backends.mps.is_available()`, then CPU.
- Use CPU-compatible inference by default; Stage 10 should be able to run without training hardware.
- Avoid AMP unless explicitly requested; deterministic audit output is more important than speed.

Failure handling:

- Missing checkpoint: hard fail before processing.
- Missing FaceLandmarker model: hard fail with download/setup instructions.
- No face landmarks: emit a failure row with `status=no_face`.
- Invalid eye bbox or empty crop: emit `status=invalid_crop`.
- Image/video decode failure: emit `status=decode_failed`.
- Model inference exception: emit `status=inference_failed` and preserve error text.
- Never silently drop rows from the summary counts.

Acceptance criteria:

- The selected MobileNetV2 checkpoint loads without modifying it.
- Full-face image and/or sampled video frames produce left/right eye crops and per-eye `p_eye_closed`.
- Output CSV contains one row per attempted eye crop or logged failure.
- Contact sheets and debug overlays make ROI geometry inspectable.
- The report explicitly states this is a runtime eye ROI consistency test, not final drowsiness accuracy.
- No Stage 8/9 outputs or checkpoints are modified.
- No training is performed.

## 10. Hard constraints

Stage 10 should:

- Not train any model.
- Not modify Stage 8/9 outputs.
- Not modify checkpoints.
- Not revive NTHUDDD2.
- Not claim final system-level drowsiness accuracy.
- Focus only on runtime eye ROI consistency testing.

Supporting audit files created with this report:

- `codex_stage10_file_existence.tsv`
- `codex_stage10_mediapipe_grep.txt`
- `codex_stage10_inference_grep.txt`
- `codex_stage10_dependency_summary.txt`

