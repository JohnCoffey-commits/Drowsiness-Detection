# YawDD+ Dash reconstruction - Stage 5: mouth ROI crops

Stage 5 turns the **full labeled frames** produced in Stage 4B into **mouth
crop images** that can be fed to a binary yawn / no-yawn CNN.

It only generates crops - it does NOT split, balance or train anything.
Splitting and training are handled in later stages.

---

## 1. What the pipeline does

For every row of the Stage 4B manifest

`artifacts/mappings/yawdd_dash_all_labeled_frames.csv`

the generator performs the following steps on the full frame image:

1. **Primary - MediaPipe Face Mesh**
   * Runs the MediaPipe Tasks `FaceLandmarker` on the full frame.
   * Uses a fixed set of 40 lip landmarks (outer lip + inner lip) from the
     canonical MediaPipe Face Mesh topology.
   * Computes a raw mouth bbox from the landmark extrema.
   * Expands it with a per-axis margin of
     `max(10 px, 10% of bbox_width)` on x and
     `max(10 px, 10% of bbox_height)` on y.
   * Clips the bbox to the image boundaries.
   * Crops and saves the mouth region.
   * Sets `crop_method = face_mesh`.

2. **Fallback A - lower-face from Haar face detector**
   * If Face Mesh returns no face, the generator runs the OpenCV
     `haarcascade_frontalface_default` detector on the grayscale image.
   * If a face box is found, the lower 40% of that face box
     (`y1 = fy1 + 0.60 * face_height`) is used as the crop.
   * Sets `crop_method = fallback_lower_face`.

3. **Fallback B - failed**
   * If neither Face Mesh nor Haar finds anything, **no crop image is saved**.
   * The manifest row is kept with `crop_method = failed` and the reason in
     `notes`, so the sample can still be traced back to its source frame.

The original YawDD+ YOLO bounding box (`yawdd_bbox_raw`) is preserved **only as
traceability metadata**.  It is NEVER used as a cropping source - the Stage 4A
visual sanity check showed that those boxes target the torso, not the mouth.

---

## 2. File layout

```
Drowsiness_Detection/
├── artifacts/
│   ├── models/
│   │   └── face_landmarker.task          # downloaded MediaPipe model
│   ├── mappings/
│   │   └── yawdd_dash_all_mouth_crops.csv
│   └── visual_checks/
│       └── mouth_crops/                  # QC side-by-side panels
├── dataset/
│   └── YawDD_plus_reconstructed/
│       └── Dash/
│           ├── full_frames/              # Stage 4B (unchanged)
│           ├── labels_csv/               # Stage 4B (unchanged)
│           └── mouth_crops/
│               └── <subject_id>/<frame_index>.jpg
├── reports/
│   └── yawdd_dash_mouth_crop_report.md
└── src/
    ├── preprocessing/
    │   └── generate_yawdd_mouth_crops.py
    └── data/
        └── verify_yawdd_mouth_crops.py
```

---

## 3. Prerequisites

Python venv and packages:

```bash
.venv/bin/pip install mediapipe opencv-python numpy pillow
```

MediaPipe face landmark model (≈3.6 MB):

```bash
mkdir -p artifacts/models
curl -sSL -o artifacts/models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

Note: On macOS the MediaPipe Tasks API needs access to the system's Metal / GL
context.  It will refuse to start inside a strict sandbox
(`Could not create an NSOpenGLPixelFormat`).  Run the generator from a normal
terminal.

---

## 4. How to run

From the repo root:

```bash
# (1) Generate mouth crops for every labeled frame (~6 min on an M-series Mac).
.venv/bin/python src/preprocessing/generate_yawdd_mouth_crops.py

# (2) Verify, count, write the report and render QC panels.
.venv/bin/python src/data/verify_yawdd_mouth_crops.py
```

Useful flags on `generate_yawdd_mouth_crops.py`:

* `--limit N` - process only the first `N` rows (smoke test).
* `--subjects 1-FemaleNoGlasses,13-MaleNoGlasses` - restrict to a subject whitelist.
* `--resume` - skip frames whose crop file already exists on disk (emits a
  `crop_method = resumed` row in the manifest to keep it complete).
* `--crop-root` / `--output-manifest` / `--model-path` - override defaults.
* `--jpeg-quality 95` - JPEG quality for saved crops.

---

## 5. Merged manifest schema

`artifacts/mappings/yawdd_dash_all_mouth_crops.csv`

| column | meaning |
|---|---|
| `subject_id` | e.g. `1-FemaleNoGlasses` |
| `frame_index` | zero-padded frame number, e.g. `00001661` |
| `image_path` | absolute path to the original full frame (Stage 4B) |
| `mouth_crop_path` | absolute path to the saved mouth crop (empty for `failed`) |
| `class_id` | `0` or `1` (raw YawDD+ class id) |
| `binary_label` | `no_yawn` or `yawn` |
| `crop_method` | `face_mesh`, `fallback_lower_face`, `resumed`, or `failed` |
| `mouth_bbox_xyxy` | `x1,y1,x2,y2` in pixel coords of `image_path` |
| `raw_video_path` | traceability back to the raw YawDD `.avi` |
| `annotation_txt_path` | traceability back to the YawDD+ `.txt` |
| `yawdd_bbox_raw` | original YawDD+ YOLO bbox (traceability only, do NOT crop with this) |
| `notes` | failure reason or resume flag when applicable |

---

## 6. Verification and readiness

`src/data/verify_yawdd_mouth_crops.py` produces
`reports/yawdd_dash_mouth_crop_report.md`.  The report includes:

* total frames processed,
* Face Mesh / fallback / failed counts and overall success rate,
* class distribution after cropping,
* per-subject method counts,
* file-integrity checks (missing crop files, unreadable JPEG sample),
* a handful of random successful crop paths for manual inspection,
* a list of rendered QC panels in
  `artifacts/visual_checks/mouth_crops/`,
* a final **READY / NOT READY** verdict for Stage 6 (subject-level split).

The readiness rule is:

1. success rate ≥ 95%,
2. every subject has at least one usable crop,
3. both classes (`no_yawn` and `yawn`) are non-empty,
4. no missing or unreadable crop files in the integrity checks.

With the default run the dataset reached **99.73 %** success
(64,093 Face Mesh + 109 fallback, 176 failed out of 64,378), all 29 subjects
covered, and both classes populated (57,171 `no_yawn` / 7,031 `yawn`).

---

## 7. What Stage 5 does NOT do

* It does not create `train` / `val` / `test` splits.
* It does not train any classifier.
* It does not use the YawDD+ YOLO bbox as a crop source.
* It does not touch Stage 4B outputs (`full_frames/`, `labels_csv/`).
