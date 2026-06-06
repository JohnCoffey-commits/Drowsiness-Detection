# YawDD Dash — Stage 4B (Labeled Full-Frame Reconstruction)

Stages 1 – 4A are assumed complete. Those stages established:

- the YawDD+ `.txt` files are YOLO-style with binary class labels
  (`0 = no_yawn`, `1 = yawn`),
- frame indices in YawDD+ filenames match the native 0-based index of the
  raw `.avi` videos,
- `_1` annotation siblings are spurious duplicate detections and must be
  ignored,
- the YawDD+ bounding box is **not** a usable mouth/face crop — it must
  be carried forward only as traceability metadata.

Stage 4B takes those facts and **reconstructs a clean, labeled, full-frame
dataset** from the raw videos. No mouth-ROI cropping, no splits, no
classifier training.

---

## 1. What the pipeline does

Three small scripts, run in order:

1. `src/data/extract_yawdd_dash_labeled_frames.py` — for every subject in
   the Stage-3 mapping CSV, decodes **only the frames that YawDD+ annotated**
   and writes one JPEG per frame plus a per-subject label CSV.
2. `src/data/merge_yawdd_dash_labels.py` — concatenates all per-subject
   CSVs into one manifest at
   `artifacts/mappings/yawdd_dash_all_labeled_frames.csv`.
3. `src/data/verify_yawdd_dash_reconstruction.py` — audits the manifest
   against the YawDD+ source (row counts, class balance, on-disk image
   presence, duplicate accounting) and writes
   `reports/yawdd_dash_reconstruction_report.md`.

### 1.1 Frame selection

For each subject the extractor reads every file in
`YawDD+/dataset/Dash/<subject>/labels/` matching `\d{8}_\d+\.txt` and
groups them by the 8-digit frame index. Only these frames are decoded.
Indices are **not** assumed contiguous; the extractor iterates the raw
video sequentially with `cap.grab()` and calls `cap.read()` only on
frames whose index is in the target set.

### 1.2 Handling `_1` duplicate annotations

If a frame has both `<idx>_0.txt` and `<idx>_1.txt`:

- only the `_0` file is used for the class label and bbox metadata,
- the row in the per-subject CSV has `kept_object_id = 0` and
  `had_duplicate_box = true`,
- the `_1` file is **not** re-processed and is counted as a duplicate in
  the final report.

Stage 4A confirmed that in this corpus the `_0` and `_1` boxes of the
same frame overlap by >95 % IoU, agree on class, and correspond to a
single driver — i.e. they are NMS duplicates, not multi-face
annotations.

### 1.3 YawDD+ bbox as traceability metadata

The extractor stores the raw YOLO line verbatim in the
`yawdd_bbox_raw` column so downstream code can audit it if needed. The
line **must not** be used to crop the image for the classifier: the box
consistently covers the driver's torso, not the face / mouth (verified
in Stage 4A).

---

## 2. Where things land

```
Drowsiness_Detection/
├── dataset/
│   └── YawDD_plus_reconstructed/
│       └── Dash/
│           ├── full_frames/
│           │   ├── 1-FemaleNoGlasses/
│           │   │   ├── 00000000.jpg
│           │   │   ├── 00000001.jpg
│           │   │   └── ...
│           │   └── ...  (one folder per subject, 29 total)
│           └── labels_csv/
│               ├── 1-FemaleNoGlasses.csv
│               ├── 1-MaleGlasses.csv
│               └── ...  (one CSV per subject, 29 total)
├── artifacts/
│   └── mappings/
│       └── yawdd_dash_all_labeled_frames.csv   # merged manifest
└── reports/
    └── yawdd_dash_reconstruction_report.md     # verification report
```

Each per-subject CSV and the merged manifest share the same columns:

| column | description |
|--------|-------------|
| `subject_id`          | YawDD+ subject folder name, e.g. `1-FemaleNoGlasses` |
| `frame_index`         | zero-padded 8-digit frame index (`00001661`) |
| `image_path`          | absolute path to the extracted JPEG |
| `raw_video_path`      | absolute path to the source `.avi` |
| `annotation_txt_path` | absolute path to the `_0` `.txt` used for the label |
| `class_id`            | `0` or `1` |
| `binary_label`        | `no_yawn` or `yawn` |
| `kept_object_id`      | always `0` in this corpus |
| `had_duplicate_box`   | `true` iff a `_1` sibling existed and was ignored |
| `yawdd_bbox_raw`      | the raw 5-token YOLO line (traceability only) |
| `extraction_status`   | `extracted` / `skipped_existing` / failure reason |
| `notes`               | free-form diagnostics |

---

## 3. How to run

All scripts use the project's `.venv` interpreter. OpenCV is required:

```bash
python -m venv .venv
.venv/bin/pip install opencv-python numpy
```

Then:

```bash
# 1. Extract annotated frames + per-subject CSVs  (~2 minutes, ~4 GB JPEGs)
.venv/bin/python src/data/extract_yawdd_dash_labeled_frames.py

# 2. Merge per-subject CSVs into a single manifest
.venv/bin/python src/data/merge_yawdd_dash_labels.py

# 3. Verify the reconstruction + generate report
.venv/bin/python src/data/verify_yawdd_dash_reconstruction.py
```

Useful extraction flags (all optional):

| flag              | purpose |
|-------------------|---------|
| `--subjects a,b`  | only process these YawDD+ folder names (for quick trials) |
| `--limit N`       | stop after the first N target frames per subject |
| `--force`         | overwrite existing JPEGs instead of skipping them |
| `--jpeg-quality`  | override default JPEG quality of 90 |

The extractor is **idempotent**: rerunning it skips frames whose JPEG
already exists (rows get `extraction_status = skipped_existing`). Use
`--force` if you want to regenerate images.

---

## 4. How to verify the reconstructed dataset

Read `reports/yawdd_dash_reconstruction_report.md`. It covers:

- global totals (subjects, rows, yawn / no_yawn counts),
- extraction-status breakdown (any non-`extracted` rows are flagged),
- a per-subject cross-check table comparing manifest row count with the
  number of YawDD+ `_0` files (they must match),
- `_1` duplicate accounting (the number of `had_duplicate_box=true` rows
  must equal the number of `_n≥1` files in the YawDD+ source),
- disk footprint per subject and globally,
- a final readiness line that says whether Stage 5 (mouth-ROI generation)
  is safe to run.

Re-run the verifier any time after modifying the manifest or the
extracted images. Expected clean-run output:

```
[verify] report written to .../yawdd_dash_reconstruction_report.md
[verify] ready_for_stage5=True
```

### Random spot-check (human inspection)

To sample a few random **class 0** / **class 1** frames per subject, build
contact sheets, and write a small CSV + markdown summary:

```bash
.venv/bin/python src/data/spotcheck_yawdd_reconstructed.py
```

Outputs go to `artifacts/visual_checks/spotcheck/` (per-subject folders,
`contact_sheet_<subject>.jpg`, copied `c0_*.jpg` / `c1_*.jpg`,
`spotcheck_samples.csv`, `spotcheck_report.md`). Requires Pillow
(`pip install pillow`).

---

## 5. What Stage 5 will consume

Stage 5 (mouth-ROI generation) should:

- read `artifacts/mappings/yawdd_dash_all_labeled_frames.csv`,
- load `image_path` for each row,
- run a fresh face/mouth detector (e.g. MediaPipe FaceMesh, already in
  `requirements.txt`) to obtain the actual mouth crop,
- keep `class_id` / `binary_label` as the target label,
- treat `yawdd_bbox_raw` as read-only traceability metadata.

Stage 4B does **not** do any of that. Nothing here has been cropped to
the mouth, nothing has been split into train/val/test, and no model has
been trained.
