# VisionGuard Data Preprocessing Technical Guide

Last updated: 2026-05-26

## 1. Scope of This Document

This document explains VisionGuard's data preprocessing workflow from raw datasets up to the point immediately before model training.

It covers:

- YawDD / YawDD+ Dash preprocessing for the mouth/yawn specialist.
- MRL Eye preprocessing for the eye open/closed specialist.
- The generated manifests, trainable manifests, visual checks, and leakage-safe subject-level splits.
- The technical reasons for each preprocessing step.

It does not cover CNN architecture design, training loops, model selection, runtime inference, temporal fusion, frontend behavior, backend APIs, alert logic, deployment, or final system evaluation.

VisionGuard should be understood as a modular driver drowsiness detection and monitoring system. It is not a single drowsy/not-drowsy classifier. The preprocessing work prepares specialist datasets for two visual evidence signals:

- Mouth/yawn evidence, later represented by `p_yawn`.
- Eye open/closed evidence, later represented by `p_eye_closed`.

## 2. High-Level Preprocessing Pipeline

The project preprocessing flow is:

```text
Raw datasets
-> dataset inspection
-> label interpretation
-> manifest construction
-> frame/image extraction or filtering
-> ROI crop generation where needed
-> visual sanity checks
-> trainable manifest creation
-> subject-level train/validation/test split
-> ready-for-training inputs
```

Preprocessing is not just file conversion. It protects the project from several common failure modes:

- Incorrect label interpretation, such as reversing yawn/non-yawn or open/closed labels.
- Non-reproducible folder scanning during training.
- Training on invalid or failed crops.
- Identity leakage caused by random frame-level splitting.
- Misleading model metrics caused by repeated frames or the same subject appearing in multiple splits.
- Silent drift between annotation files, extracted frames, crop paths, and training inputs.

The preprocessing artifacts are therefore part of the experiment design. A training run should consume documented manifests and split files, not discover files ad hoc from raw dataset folders.

## 3. Main Dataset 1: YawDD / YawDD+ Dash

YawDD / YawDD+ Dash is used for the mouth/yawn specialist.

Project role:

| Item | Meaning |
| --- | --- |
| Specialist task | `no_yawn` vs `yawn` classification |
| Label mapping | `0 = no_yawn`, `1 = yawn` |
| Later runtime evidence concept | `p_yawn` |
| Main raw input | Original YawDD Dash `.avi` videos |
| Annotation input | YawDD+ Dash label files |
| Final training input form | Mouth ROI crop images with train/val/test split labels |

The raw Dash videos are full driver frames. They are not the final model input for the mouth specialist. The yawn cue is localized around the mouth, so the project reconstructs labeled full frames first, then generates mouth region-of-interest crops using face landmarks.

Relevant paths:

- `dataset/YawDD_raw/`
- `dataset/YawDD+/`
- `dataset/YawDD_plus_reconstructed/`
- `src/data/build_yawdd_dash_mapping.py`
- `src/data/extract_yawdd_dash_labeled_frames.py`
- `src/preprocessing/generate_yawdd_mouth_crops.py`
- `src/preprocessing/precompute_yawdd_mouth_crops.py`
- `src/data/build_yawdd_split.py`
- `reports/yawdd_raw_dash_report.md`
- `reports/yawdd_plus_annotation_format_report.md`
- `reports/yawdd_dash_reconstruction_report.md`
- `reports/yawdd_dash_visual_sanity_check.md`
- `reports/yawdd_dash_mouth_crop_report.md`
- `reports/yawdd_dash_split_report.md`

### Raw Dash Videos

The raw YawDD Dash source is delivered as `.avi` videos, one continuous clip per subject. The inspected local raw Dash data contains 29 videos: 13 female subjects and 16 male subjects.

The filenames encode subject index, gender, and glasses state. Some raw filenames have small irregularities, such as duplicated `.avi.avi` extensions or stray whitespace. The preprocessing mapping step normalizes these names before pairing videos with YawDD+ folders.

### YawDD+ Annotations

The YawDD+ Dash folders contain annotation text files under per-subject `labels/` folders. Each annotation file name contains the frame index:

```text
<8-digit frame index>_<object index>.txt
```

Each file contains a YOLO-style row:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Important interpretation:

- The class id is the useful label.
- `0` means non-yawning.
- `1` means yawning.
- The frame index is a native 0-based raw-video frame index.
- The YOLO geometry is preserved for traceability, but it is not used as the mouth crop.

The visual sanity check found that the YawDD+ box geometry is not a reliable mouth region. In sampled frames, the box often covered the torso or lower body area rather than the mouth. This is why the project reconstructs frames and then computes fresh mouth ROIs.

### All Attempts vs Trainable Rows

The YawDD mouth-crop preprocessing distinguishes between all crop attempts and training-ready rows:

- `artifacts/mappings/yawdd_dash_all_mouth_crops.csv` records every attempted crop row, including failed rows.
- `artifacts/mappings/yawdd_dash_all_mouth_crops_trainable.csv` excludes failed crop rows and adds split labels.

This distinction matters because failed crops are useful for quality auditing, but they should not be fed into the CNN training dataset.

## 4. YawDD / YawDD+ Step-by-Step Preprocessing

### 4.1 Raw Video and Annotation Inspection

The first step is to inspect the raw Dash videos and annotation folders without doing training.

Reports:

- `reports/yawdd_raw_dash_report.md`
- `reports/yawdd_plus_annotation_format_report.md`

What this verifies:

- The raw source contains 29 Dash `.avi` videos.
- The YawDD+ Dash annotation folders also cover 29 subjects.
- Raw video canonical tokens match YawDD+ subject folder tokens after normalizing whitespace and repeated `.avi` suffixes.
- Annotation files follow the expected `<frame>_<object>.txt` naming pattern.
- Annotation rows contain binary class ids and YOLO-normalized geometry.
- Some subjects have frame-index gaps, so frame indices must be respected exactly instead of using a running counter.

Why this is necessary:

- It prevents mismatching a subject's annotations to the wrong raw video.
- It confirms that frame extraction can be driven by YawDD+ frame indices.
- It flags multi-object `_1` annotation files before bulk extraction.

### 4.2 Building the Dash Frame Mapping

Script:

- `src/data/build_yawdd_dash_mapping.py`

Default output:

- `artifacts/mappings/yawdd_dash_mapping.csv`
- `reports/yawdd_dash_mapping_report.md`

This script pairs every YawDD+ Dash subject folder with the matching raw YawDD Dash `.avi` file.

Important code meaning:

- It indexes raw `.avi` files under the Dash female/male folders.
- It normalizes raw filenames by stripping whitespace and repeated `.avi` suffixes.
- It uses the normalized raw video stem and the YawDD+ folder name as the canonical subject token.
- It writes one row per YawDD+ subject folder.
- It records mapping confidence and notes.

Expected mapping fields include:

- `subject_id`
- `annotation_folder`
- `annotation_txt_path`
- `raw_source_path`
- `mapping_confidence`
- `mapping_notes`

Why this step exists:

- Later extraction must know exactly which raw `.avi` supplies frames for each annotation folder.
- The mapping table makes that pairing reproducible instead of relying on manual path matching.
- Filename anomalies are handled once and documented.

### 4.3 Reconstructing Labeled Dash Frames

Script:

- `src/data/extract_yawdd_dash_labeled_frames.py`

Default input:

- `artifacts/mappings/yawdd_dash_mapping.csv`

Main outputs:

- `dataset/YawDD_plus_reconstructed/Dash/full_frames/<subject_id>/<frame_index>.jpg`
- `dataset/YawDD_plus_reconstructed/Dash/labels_csv/<subject_id>.csv`
- `artifacts/mappings/yawdd_dash_all_labeled_frames.csv`

This step reconstructs the actual image frames referenced by the YawDD+ annotations. YawDD+ provides label files, not image files, so the source image must be decoded from the original raw Dash video.

Important code meaning:

- It scans each subject's `labels/` folder.
- It groups annotation files by frame index.
- It prefers the `_0` annotation file for each frame.
- It records whether duplicate `_1` files existed using `had_duplicate_box`.
- It opens the matching `.avi` and iterates frames sequentially.
- It saves only requested frame indices.
- It writes per-frame provenance into the output manifest.

The reconstructed labeled-frame manifest has columns such as:

- `subject_id`
- `frame_index`
- `image_path`
- `raw_video_path`
- `annotation_txt_path`
- `class_id`
- `binary_label`
- `kept_object_id`
- `had_duplicate_box`
- `yawdd_bbox_raw`
- `extraction_status`
- `notes`

Documented result:

- 64,378 labeled frames.
- 57,347 `no_yawn` rows.
- 7,031 `yawn` rows.
- 10 duplicate `_1` boxes flagged and ignored.
- 0 missing extracted JPEGs.

Why reconstruction is needed:

- The annotations alone are not model inputs.
- The mouth cropper needs actual pixels.
- The exact YawDD+ frame index must be connected to the matching raw video frame.

### 4.4 Visual Sanity Check

Report:

- `reports/yawdd_dash_visual_sanity_check.md`

The visual sanity check verifies that the labels and frame alignment mean what the project thinks they mean.

Confirmed findings:

- YawDD+ frame indices align with raw `.avi` frame indices.
- Class `1` corresponds to visible yawning.
- Class `0` corresponds to non-yawning.
- `_1` files are spurious duplicate detections in the checked samples.
- The YawDD+ YOLO boxes should not be used as mouth crops because they do not reliably localize the mouth.

Why this step is required:

- Binary labels can be easy to reverse if the dataset documentation is incomplete.
- A model trained with reversed yawn labels would still produce numbers, but the model would be wrong.
- Visual evidence is the safest way to confirm that class `1` really means yawning and class `0` really means non-yawning.

### 4.5 Mouth ROI Crop Generation

Primary script:

- `src/preprocessing/generate_yawdd_mouth_crops.py`

Earlier/legacy preprocessing entrypoint:

- `src/preprocessing/precompute_yawdd_mouth_crops.py`

Default input to the current Stage 5 cropper:

- `artifacts/mappings/yawdd_dash_all_labeled_frames.csv`

Main outputs:

- `dataset/YawDD_plus_reconstructed/Dash/mouth_crops/<subject_id>/<frame_index>.jpg`
- `artifacts/mappings/yawdd_dash_all_mouth_crops.csv`

The mouth cropper turns reconstructed full frames into mouth ROI images. This is the key preprocessing step that makes the mouth/yawn specialist a focused visual model rather than a full-frame driver classifier.

Technical flow:

1. Read a row from the labeled-frame manifest.
2. Load the reconstructed full-frame image.
3. Run MediaPipe Face Landmarker / Face Mesh.
4. Use fixed outer-lip and inner-lip landmark indices.
5. Build a mouth bounding box from the landmark coordinates.
6. Expand the box with padding.
7. Clip the box to image boundaries.
8. Save the mouth crop.
9. Record the crop path, crop method, crop bounding box, label, source frame, annotation path, and notes.

Primary crop method:

- `face_mesh`: MediaPipe landmarks are found and a mouth ROI is computed from lip landmarks.

Fallback method:

- `fallback_lower_face`: if Face Mesh fails, an OpenCV Haar face detector attempts to find a face and uses the lower portion of the face as a fallback crop.

Failure method:

- `failed`: neither Face Mesh nor the fallback produces a usable crop, or the image/crop cannot be saved.

Documented Stage 5 result:

| Metric | Value |
| --- | ---: |
| Total frames processed | 64,378 |
| MediaPipe Face Mesh crops | 64,093 |
| Fallback lower-face crops | 109 |
| Failed crops | 176 |
| Saved trainable crops | 64,202 |
| Success rate | 99.73% |

Why mouth ROI crops are better than full frames for this specialist:

- The yawn label is visually concentrated around the mouth.
- Full driver frames include irrelevant background, steering wheel, lighting, clothing, and subject identity cues.
- Mouth crops reduce unnecessary variation and make the classifier focus on the evidence cue.
- The crop manifest allows failed crops to be audited instead of silently entering training.

The earlier `precompute_yawdd_mouth_crops.py` script is useful historical context. It also uses MediaPipe-style mouth cropping and fallback behavior, but the current documented YawDD Dash Stage 5 output is represented by `generate_yawdd_mouth_crops.py` and `artifacts/mappings/yawdd_dash_all_mouth_crops.csv`.

### 4.6 Trainable Mouth-Crop Manifest

Artifacts:

- `artifacts/mappings/yawdd_dash_all_mouth_crops.csv`
- `artifacts/mappings/yawdd_dash_all_mouth_crops_trainable.csv`

Difference:

| File | Meaning |
| --- | --- |
| `yawdd_dash_all_mouth_crops.csv` | All crop attempts, including `face_mesh`, `fallback_lower_face`, and `failed` rows. |
| `yawdd_dash_all_mouth_crops_trainable.csv` | Training-ready rows after filtering out failed crops and invalid labels/paths, with a `split` column added. |

Rows may be excluded from trainable data when:

- `crop_method == failed`
- `binary_label` is not one of `no_yawn` or `yawn`
- `mouth_crop_path` is missing
- the referenced crop file does not exist

The trainable manifest preserves source provenance. It still links back to the full frame, raw video, annotation file, class id, and YawDD+ raw box. This makes later training and error analysis traceable.

### 4.7 Subject-Level Split

Script:

- `src/data/build_yawdd_split.py`

Artifacts:

- `artifacts/mappings/yawdd_dash_all_mouth_crops_trainable.csv`
- `artifacts/splits/yawdd_dash_subject_split.csv`
- `reports/yawdd_dash_split_report.md`

The YawDD split is by `subject_id`, not by individual frames. The script searches for a subject assignment that balances:

- train/validation/test subject counts,
- image proportions,
- yawn rate,
- gender distribution,
- glasses/no-glasses distribution,
- the requirement that every split contains both classes.

Documented split:

| Split | Subjects | Images | `no_yawn` | `yawn` | Yawn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `train` | 20 | 44,156 | 39,345 | 4,811 | 10.90% |
| `val` | 4 | 8,892 | 7,902 | 990 | 11.13% |
| `test` | 5 | 11,154 | 9,924 | 1,230 | 11.03% |

Verification checks:

- 29 unique trainable subjects.
- No subject appears in more than one split.
- No failed crop rows in trainable data.
- Every split contains both classes.
- All referenced mouth-crop files exist.

Why random frame-level splitting is invalid:

- Adjacent frames from the same video are visually similar.
- A random frame split would place the same person's face and nearby frames in both training and test sets.
- That would inflate evaluation results by testing on identities and contexts the model has already seen.

The split file is the correct boundary before mouth/yawn model training.

## 5. Main Dataset 2: MRL Eye

MRL Eye is used for the eye open/closed specialist.

Project role:

| Item | Meaning |
| --- | --- |
| Specialist task | Closed vs open eye classification |
| Label mapping | `0 = closed`, `1 = open` |
| Later runtime evidence concept | `p_eye_closed` |
| Main raw input | Eye crop images organized by subject |
| Final training input form | Trainable image manifest with subject-level split labels |

Relevant paths:

- `dataset/mrlEyes_2018_01/`
- `src/data/inspect_mrl_eye.py`
- `src/data/build_mrl_eye_manifest.py`
- `src/data/split_mrl_eye_subjects.py`
- `src/data/spotcheck_mrl_eye.py`
- `reports/mrl_eye_dataset_report.md`
- `reports/mrl_eye_split_report.md`

MRL Eye is already an eye-image dataset, so it does not require the same full-frame reconstruction and mouth ROI crop generation used for YawDD. The critical preprocessing work is inspection, label parsing, manifest construction, image-readability filtering, visual spot checks, and leakage-safe subject splitting.

The filename structure encodes metadata such as:

- subject id,
- image id,
- gender,
- glasses,
- eye state,
- reflections,
- lighting,
- sensor id.

The project verifies the annotation mapping before building training manifests. The important class mapping is:

| Class id | Label |
| ---: | --- |
| 0 | `closed` |
| 1 | `open` |

Subject-level splitting matters for MRL Eye because each subject contributes many similar eye images. A random image split would again risk identity leakage.

## 6. MRL Eye Step-by-Step Preprocessing

### 6.1 Raw Dataset Inspection

Script:

- `src/data/inspect_mrl_eye.py`

Report:

- `reports/mrl_eye_dataset_report.md`

This script is read-only with respect to the raw dataset. It scans the MRL Eye folder, parses filenames, verifies image readability, and writes a dataset inspection report.

What the inspection verifies:

- Dataset root exists.
- `annotation.txt` confirms `0 = closed` and `1 = open`.
- Filenames match the expected MRL format.
- Subject ids can be parsed.
- Images can be opened.
- Both closed and open labels are present.
- Metadata distributions are recorded for gender, glasses, lighting, reflections, and sensor id.

Documented result:

| Metric | Value |
| --- | ---: |
| Total images | 84,898 |
| Total subjects | 37 |
| Unreadable images | 0 |
| Unparseable filenames | 0 |
| Closed images | 41,946 |
| Open images | 42,952 |

Why this step exists:

- It confirms the class mapping before model preparation.
- It prevents malformed filenames or unreadable images from entering training.
- It gives the project a stable count of subjects and classes before splitting.

### 6.2 Building the Full Image Manifest

Script:

- `src/data/build_mrl_eye_manifest.py`

Artifact:

- `artifacts/mappings/mrl_eye_all_images.csv`

A manifest is a structured CSV index of dataset samples. It is used instead of scanning folders ad hoc during training.

The full MRL Eye manifest records columns such as:

- `image_path`
- `relative_path`
- `filename`
- `subject_id`
- `image_id`
- `gender`
- `glasses`
- `eye_state`
- `label`
- `label_name`
- `reflections`
- `lighting`
- `sensor_id`
- `parse_ok`
- `width`
- `height`
- `extension`
- `is_valid`
- `read_ok`
- `error`

Why a manifest is used:

- It freezes the exact sample list.
- It records parsing and validation status.
- It makes training reproducible.
- It keeps labels and metadata available for later analysis.

### 6.3 Creating the Trainable Manifest

Artifacts:

- `artifacts/mappings/mrl_eye_trainable.csv`
- `artifacts/mappings/mrl_eye_trainable_with_split.csv`

The trainable manifest is the filtered subset of the full manifest that is safe to use for training. In the current documented MRL Eye preprocessing result, all 84,898 images are trainable.

Readiness criteria include:

- filename parsed successfully,
- image is readable,
- file exists,
- label is valid and in `{0, 1}`,
- subject id is present,
- row is marked valid.

`mrl_eye_trainable.csv` contains the trainable image rows before split labels are attached.

`mrl_eye_trainable_with_split.csv` contains the same trainable rows plus a `split` column. This is the main training-ready MRL Eye manifest.

### 6.4 Visual Spot Checks

Script:

- `src/data/spotcheck_mrl_eye.py`

Inputs:

- `artifacts/mappings/mrl_eye_trainable_with_split.csv`

Outputs:

- `artifacts/visual_checks/mrl_eye_closed_contact_sheet.jpg`
- `artifacts/visual_checks/mrl_eye_open_contact_sheet.jpg`
- `artifacts/visual_checks/mrl_eye_by_split_contact_sheet.jpg`

The script builds contact sheets from sampled rows. A contact sheet is a grid of small images with labels printed next to them.

Why contact sheets are useful:

- They quickly reveal if `closed` and `open` labels are reversed.
- They show whether images are visually plausible eye crops.
- They help verify that training, validation, and test splits contain sensible examples.
- They can reveal unreadable, corrupted, or unexpected images before training.

Visual spot checks do not replace quantitative checks, but they catch mistakes that CSV summaries cannot.

### 6.5 Subject-Level Split

Script:

- `src/data/split_mrl_eye_subjects.py`

Artifacts:

- `artifacts/splits/mrl_eye_subject_split.csv`
- `artifacts/mappings/mrl_eye_trainable_with_split.csv`
- `reports/mrl_eye_split_report.md`

The split script reads `artifacts/mappings/mrl_eye_trainable.csv` and assigns each subject to exactly one of:

- `train`
- `val`
- `test`

The split search attempts to match approximate 70/15/15 image ratios while keeping subject groups intact and ensuring each split contains both classes.

Documented split:

| Split | Subjects | Images | Closed | Open |
| --- | ---: | ---: | ---: | ---: |
| `train` | 25 | 58,982 | 29,310 | 29,672 |
| `val` | 6 | 13,029 | 6,333 | 6,696 |
| `test` | 6 | 12,887 | 6,303 | 6,584 |

Verification checks:

- leakage check passed,
- no missing split labels,
- every image receives exactly one split,
- every split contains closed and open samples,
- referenced image files exist.

Why subject separation is important:

- MRL Eye contains many samples per subject.
- Eye appearance can encode identity, lighting, camera, and sensor conditions.
- Evaluation is more honest when the held-out split contains subjects not seen during training.

The split manifest is the boundary immediately before eye specialist model training.

## 7. Brief Note on NTHUDDD2

NTHUDDD2 / Kaggle extracted-frame data was explored, but it is not the current main system direction.

Relevant paths:

- `reports/nthuddd2_kaggle_dataset_report.md`
- `reports/nthu_dataset_report.md`
- `src/data/build_nthuddd2_kaggle_manifest.py`
- `src/data/split_nthuddd2_kaggle_subject.py`
- `artifacts/mappings/nthuddd2_kaggle_all_images.csv`
- `artifacts/mappings/nthuddd2_kaggle_all_images_trainable.csv`
- `artifacts/mappings/nthuddd2_kaggle_all_images_trainable_with_split.csv`
- `artifacts/splits/nthuddd2_kaggle_subject_split.csv`

The explored Kaggle branch is a binary extracted-frame dataset with labels:

- `notdrowsy = 0`
- `drowsy = 1`

It is not the official raw-video NTHU-DDD protocol, and only four parsed subjects are available locally. It should not be presented as the final VisionGuard evidence pipeline. The current main system uses MRL Eye for the eye specialist, YawDD / YawDD+ Dash for the mouth/yawn specialist, and rule-based runtime fusion after model inference.

This distinction matters because describing VisionGuard around NTHUDDD2 would make the project look like a single binary drowsy/not-drowsy image classifier, which is not the current architecture.

## 8. Generated Artifacts and Their Meanings

The following table lists the main preprocessing artifacts found in the current repository.

| Artifact path | Dataset/module | What it contains | Why it exists | Used before training for what purpose |
| --- | --- | --- | --- | --- |
| `artifacts/mappings/yawdd_dash_all_labeled_frames.csv` | YawDD mouth/yawn | 64,378 reconstructed full-frame rows with subject, frame index, image path, raw video path, annotation path, class id, binary label, duplicate-box flag, and extraction status | Connects YawDD+ labels to actual decoded Dash frames | Source manifest for mouth ROI crop generation |
| `artifacts/mappings/yawdd_dash_all_mouth_crops.csv` | YawDD mouth/yawn | 64,378 crop-attempt rows including `face_mesh`, `fallback_lower_face`, and `failed` crop methods | Audits every crop attempt and preserves failure information | Source for trainable crop filtering and split construction |
| `artifacts/mappings/yawdd_dash_all_mouth_crops_trainable.csv` | YawDD mouth/yawn | 64,202 trainable mouth-crop rows with valid crop paths and a `split` column | Removes failed crop rows while preserving provenance | Training-ready row list for mouth/yawn CNN training |
| `artifacts/splits/yawdd_dash_subject_split.csv` | YawDD mouth/yawn | Row-level split manifest with subject, split, gender, glasses, full-frame path, crop path, label, class id, crop method, and provenance fields | Makes the subject-level train/val/test assignment explicit | Confirms leakage-safe split membership and supplies split-aware sample rows |
| `artifacts/mappings/mrl_eye_all_images.csv` | MRL Eye | 84,898 full image manifest rows with parsed filename metadata, label, image size, validity, and read status | Freezes the complete parsed image list | Source for filtering trainable MRL Eye samples |
| `artifacts/mappings/mrl_eye_trainable.csv` | MRL Eye | 84,898 trainable rows after validation and filtering | Provides the clean eye image set before splitting | Input to the MRL subject split script |
| `artifacts/mappings/mrl_eye_trainable_with_split.csv` | MRL Eye | 84,898 trainable rows plus `split` labels | Combines clean samples with train/val/test assignment | Training-ready row list for eye open/closed CNN training |
| `artifacts/splits/mrl_eye_subject_split.csv` | MRL Eye | 37 subject-level split rows with image counts and class ratios per subject | Documents which subjects belong to train, validation, and test | Verifies subject separation and split balance |

No required artifact in this table was missing in the current repository at inspection time.

## 9. Technical Concepts Behind the Preprocessing

### Manifest

A manifest is a CSV index of samples. It records paths, labels, metadata, validation flags, and provenance. Training uses a manifest so that the sample list is reproducible.

### Trainable Manifest

A trainable manifest is the subset of a full manifest that has passed filtering. It excludes rows that are unreadable, malformed, missing labels, missing files, or failed crops.

### Label Mapping

Label mapping defines what each numeric class id means. In this project:

- YawDD: `0 = no_yawn`, `1 = yawn`.
- MRL Eye: `0 = closed`, `1 = open`.
- NTHUDDD2 exploratory branch: `0 = notdrowsy`, `1 = drowsy`.

Correct label mapping is required before any model training.

### ROI Crop

ROI means region of interest. For YawDD mouth/yawn training, the ROI is the mouth crop. It focuses the model input on the visual evidence relevant to yawning.

### MediaPipe Landmarks

MediaPipe landmarks are facial keypoints detected on an image. The mouth cropper uses lip landmarks to build a mouth bounding box. If landmarks fail, the project tries a lower-face fallback crop.

### Visual Sanity Check

A visual sanity check is a small manual or semi-manual review of decoded images and labels. It confirms that the labels and frame indices mean what the code assumes.

### Contact Sheet

A contact sheet is a grid of image samples with labels. It helps humans quickly check label correctness, sample quality, and split plausibility.

### Subject-Level Split

A subject-level split assigns each person to only one split. This prevents the model from seeing the same person's images during training and testing.

### Data Leakage

Data leakage happens when information from the validation or test set appears in training. In video and face/eye datasets, random frame-level splitting is a common leakage source because nearby frames and the same subject can appear in multiple splits.

### Class Balance

Class balance describes how many samples belong to each class. YawDD mouth/yawn is imbalanced because yawn frames are a minority. The split process attempts to preserve a similar yawn rate across train, validation, and test.

### Failed Crop / Invalid Row

A failed crop is a row where no usable ROI image was produced. An invalid row may have a bad label, missing subject id, missing file, unreadable image, or failed parsing. These rows should be audited, not trained on.

### Reproducible Preprocessing Artifact

A reproducible preprocessing artifact is a saved CSV, report, or contact sheet that records what the preprocessing step produced. It allows teammates to inspect the pipeline without rerunning expensive or destructive operations.

## 10. What Is Ready Immediately Before Model Training

At the preprocessing/training boundary, the project has two main ready-for-training inputs.

### Mouth/Yawn Specialist Input

Training should consume the YawDD mouth/yawn trainable mouth-crop data:

- `artifacts/mappings/yawdd_dash_all_mouth_crops_trainable.csv`
- or the split-equivalent rows in `artifacts/splits/yawdd_dash_subject_split.csv`

Each training row should provide:

- mouth crop path,
- label (`no_yawn` or `yawn`),
- class id (`0` or `1`),
- subject id,
- split (`train`, `val`, or `test`),
- crop method and provenance for later analysis.

The training boundary starts after this file exists and passes split/leakage checks.

### Eye Open/Closed Specialist Input

Training should consume the MRL Eye trainable image data:

- `artifacts/mappings/mrl_eye_trainable_with_split.csv`

Each training row should provide:

- eye image path,
- label name (`closed` or `open`),
- class id (`0` or `1`),
- subject id,
- split (`train`, `val`, or `test`),
- parsed metadata such as lighting, reflections, glasses, and sensor id.

The training boundary starts after this file exists and `reports/mrl_eye_split_report.md` confirms leakage-safe splitting.

This document stops at that boundary. Model architectures, optimization settings, metrics, and checkpoint selection belong to training documentation.

## 11. Common Mistakes and How This Project Avoids Them

| Mistake | How the project avoids it |
| --- | --- |
| Treating specialist-model metrics as final system-level driver drowsiness accuracy | The docs keep specialist datasets and runtime warning-candidate analysis separate. |
| Calling VisionGuard a single drowsy/not-drowsy classifier | The project uses separate mouth/yawn and eye-state specialists plus rule-based temporal fusion. |
| Using NTHUDDD2 as the main project direction | NTHUDDD2 is documented as an explored branch, while the current main direction is YawDD plus MRL Eye. |
| Reversing YawDD labels | Visual sanity checks confirm `0 = no_yawn` and `1 = yawn`. |
| Reversing MRL Eye labels | `annotation.txt` and inspection scripts confirm `0 = closed` and `1 = open`. |
| Using YawDD+ YOLO boxes as mouth crops | Visual checks showed the boxes are not reliable mouth ROIs, so MediaPipe mouth landmarks are used instead. |
| Training on failed crops | Failed crop rows are preserved in the all-attempt manifest but excluded from the trainable manifest. |
| Random frame-level splitting across the same subject | Subject-level split scripts assign each subject to one split only. |
| Relying on stale result files | Current reports and manifest headers should be checked before interpreting outputs. |
| Committing raw datasets, generated bulk crops, checkpoints, or large outputs to normal Git | Raw and generated large assets should stay in ignored dataset/artifact/output/checkpoint locations unless explicitly curated. |

## 12. Reproducibility and Verification Checklist

Use this checklist before starting or reviewing model training.

### Source Files Inspected

- `docs/PROJECT_STRUCTURE.md`
- `docs/PROJECT_CURRENT_STATUS.md`
- `docs/tech_learning/PROJECT_LEARNING_GUIDE.md`
- `src/data/build_yawdd_dash_mapping.py`
- `src/data/extract_yawdd_dash_labeled_frames.py`
- `src/preprocessing/generate_yawdd_mouth_crops.py`
- `src/preprocessing/precompute_yawdd_mouth_crops.py`
- `src/data/build_yawdd_split.py`
- `src/data/inspect_mrl_eye.py`
- `src/data/build_mrl_eye_manifest.py`
- `src/data/split_mrl_eye_subjects.py`
- `src/data/spotcheck_mrl_eye.py`

### Reports Checked

- `reports/yawdd_raw_dash_report.md`
- `reports/yawdd_plus_annotation_format_report.md`
- `reports/yawdd_dash_reconstruction_report.md`
- `reports/yawdd_dash_visual_sanity_check.md`
- `reports/yawdd_dash_mouth_crop_report.md`
- `reports/yawdd_dash_split_report.md`
- `reports/mrl_eye_dataset_report.md`
- `reports/mrl_eye_split_report.md`
- `reports/nthuddd2_kaggle_dataset_report.md`
- `reports/nthu_dataset_report.md`

### Manifest Headers Checked

- `artifacts/mappings/yawdd_dash_all_labeled_frames.csv`
- `artifacts/mappings/yawdd_dash_all_mouth_crops.csv`
- `artifacts/mappings/yawdd_dash_all_mouth_crops_trainable.csv`
- `artifacts/splits/yawdd_dash_subject_split.csv`
- `artifacts/mappings/mrl_eye_all_images.csv`
- `artifacts/mappings/mrl_eye_trainable.csv`
- `artifacts/mappings/mrl_eye_trainable_with_split.csv`
- `artifacts/splits/mrl_eye_subject_split.csv`

### Verification Questions

- Are YawDD labels confirmed as `0 = no_yawn`, `1 = yawn`?
- Are MRL Eye labels confirmed as `0 = closed`, `1 = open`?
- Does YawDD frame reconstruction use the annotation frame index, not a running counter?
- Are YawDD+ `_1` duplicate object files ignored for labeling but recorded for traceability?
- Are YawDD mouth crops generated from MediaPipe mouth landmarks or fallback lower-face logic, not from the original YawDD+ YOLO boxes?
- Are failed mouth crops excluded from trainable rows?
- Do YawDD and MRL split files keep each subject in exactly one split?
- Does every split contain both classes?
- Do referenced image/crop paths exist according to the reports?
- Was no model training accidentally run during preprocessing review?
- Were no source code, dataset, checkpoint, runtime, frontend, backend, report, or existing documentation files modified during review?

If any answer is uncertain, inspect the relevant report and artifact header before training.
