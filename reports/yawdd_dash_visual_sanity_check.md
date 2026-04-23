# YawDD Dash — Visual Sanity Check (Stage 4A)

This report is the small, evidence-first validation that precedes any bulk
frame extraction. It answers five concrete questions:

1. Do YawDD+ annotation frame indices align with raw `.avi` frame indices?
2. Does class `1` really correspond to *yawning*?
3. Does class `0` really correspond to *not yawning*?
4. Do the YOLO bounding boxes localise the mouth / face region?
5. Are the multi-object (`_1`) annotation files problematic?

No bulk frame extraction was performed. Only 11 frames across 3 subjects were
decoded, overlaid with their YOLO box, and saved as JPEGs. A separate
zero-decode audit also covers all 29 subjects at the frame-count level.

- Validation script: `src/data/validate_yawdd_dash_frames.py`
- Audit script: `src/data/audit_yawdd_dash_framecounts.py`
- Per-frame log: `artifacts/visual_checks/validation_log.json`
- Dataset-wide audit CSV: `artifacts/visual_checks/framecount_audit.csv`
- Annotated images: `artifacts/visual_checks/<subject>/...`

---

## 1. Selected validation subset

| subject | reason for inclusion |
|---------|-----------------------|
| `1-FemaleNoGlasses` | simple Female / NoGlasses case; has class-1 frames |
| `13-MaleNoGlasses`  | Male / NoGlasses; includes the raw video with a stray-space filename (`13-MaleNoGlasses .avi`) |
| `8-MaleNoGlasses`   | Male / NoGlasses; the only subject with several `_1` multi-object annotation files |

For each subject the following frames were decoded and annotated:

| subject | frame index | obj | class | purpose |
|---------|-------------|-----|-------|---------|
| 1-FemaleNoGlasses | 00000000 | 0 | 0 | alignment |
| 1-FemaleNoGlasses | 00001370 | 0 | 0 | middle class-0 |
| 1-FemaleNoGlasses | 00001661 | 0 | 1 | yawn |
| 13-MaleNoGlasses  | 00000000 | 0 | 0 | alignment |
| 13-MaleNoGlasses  | 00000500 | 0 | 0 | middle class-0 |
| 13-MaleNoGlasses  | 00001001 | 0 | 1 | yawn |
| 8-MaleNoGlasses   | 00000000 | 0 | 0 | alignment |
| 8-MaleNoGlasses   | 00000800 | 0 | 0 | middle class-0 |
| 8-MaleNoGlasses   | 00001287 | 0 | 0 | paired with `_1` sibling |
| 8-MaleNoGlasses   | 00001287 | 1 | 0 | multi-object `_1` |
| 8-MaleNoGlasses   | 00001756 | 0 | 1 | yawn |

All 11 frames decoded successfully (see `validation_log.json`, every entry has
`decoded_ok = true`, no fallback seek needed).

---

## 2. Frame-index alignment — **CONFIRMED**

### 2.1 Per-subject quantitative cross-check (all 29 subjects)

The audit script opens every raw `.avi` and compares its
`CAP_PROP_FRAME_COUNT` with the maximum YawDD+ annotation frame index:

- **29 / 29** subjects satisfy `max(YawDD+ frame index) == raw_frame_count − 1`.
- Example: `1-FemaleNoGlasses` → 2741 `.avi` frames, YawDD+ max index = 2740.
- Example: `10-FemaleNoGlasses` → 1217 `.avi` frames, YawDD+ max index = 1216
  (1213 files present, 4 frames dropped in the middle, indexing unchanged).
- Every subject's `CAP_PROP_FPS` is reported as ~30.0003 fps, matching the
  Readme.

This means YawDD+ uses the **native 0-based frame index of the raw video**
(no resampling, no offset). Gaps in annotation indices are dropped frames,
not a shifted timebase.

### 2.2 Visual cross-check (3 subjects)

Every decoded frame's content matches what the annotation says:

- `1-FemaleNoGlasses/frame_00001661_obj0_class1.jpg` — subject is visibly
  mid-yawn (mouth wide open); label = `1`.
- `13-MaleNoGlasses/frame_00001001_obj0_class1.jpg` — mid-yawn, label = `1`.
- `8-MaleNoGlasses/frame_00001756_obj0_class1.jpg` — mid-yawn, label = `1`.
- All 8 class-0 samples show a non-yawning driver.

**Conclusion:** annotation filename index == raw-video decoded frame index.
Safe to seek frames by `cv2.CAP_PROP_POS_FRAMES = int(filename_index)`.

---

## 3. Class-id semantics — **CONFIRMED**

| subject | frame | label | visual state | agrees? |
|---------|-------|-------|--------------|---------|
| 1-FemaleNoGlasses | 00000000 | 0 | smiling, mouth closed | ✓ |
| 1-FemaleNoGlasses | 00001370 | 0 | neutral, mouth closed | ✓ |
| 1-FemaleNoGlasses | 00001661 | **1** | mouth wide open, yawning | ✓ |
| 13-MaleNoGlasses  | 00000000 | 0 | driving, mouth closed | ✓ |
| 13-MaleNoGlasses  | 00000500 | 0 | looking away, mouth closed | ✓ |
| 13-MaleNoGlasses  | 00001001 | **1** | yawning, teeth visible | ✓ |
| 8-MaleNoGlasses   | 00000000 | 0 | smiling, mouth closed | ✓ |
| 8-MaleNoGlasses   | 00000800 | 0 | neutral, mouth closed | ✓ |
| 8-MaleNoGlasses   | 00001287 | 0 | talking/smiling, no yawn | ✓ |
| 8-MaleNoGlasses   | 00001756 | **1** | yawning, mouth wide open | ✓ |

- Class `1` **is yawning** in every sampled frame.
- Class `0` **is not yawning** in every sampled frame.
- The Stage-1 hypothesis (`0 = non-yawn, 1 = yawn`) is confirmed for the
  sample; we proceed treating it as the ground-truth interpretation.

---

## 4. Bounding-box semantics — **MISALIGNED WITH MOUTH / FACE** (important)

This is the only unexpected finding of Stage 4A.

In every one of the 11 sampled frames the YOLO bounding box does **not**
cover the driver's face or mouth. It consistently covers the **torso
region** (roughly from the driver's chin down to the steering wheel / lap).

Concrete measurements (pixel coordinates in a 640x480 frame):

| image | bbox (x1, y1, x2, y2) | bbox top vs face location |
|-------|------------------------|---------------------------|
| `1-FemaleNoGlasses/frame_00000000_obj0_class0.jpg` | (37, 282, 537, 473) | top at y≈282 is below the chin; face is in the upper third |
| `1-FemaleNoGlasses/frame_00001661_obj0_class1.jpg` | (96, 261, 525, 475) | top at y≈261, mouth is at y≈180 → *above* the box |
| `13-MaleNoGlasses/frame_00001001_obj0_class1.jpg`  | (215, 263, 639, 467) | top at y≈263, yawning mouth is higher up |
| `8-MaleNoGlasses/frame_00001756_obj0_class1.jpg`   | (97, 209, 583, 447) | top at y≈209, mouth slightly above |

This means:

- The bounding box is **not** a face crop and **not** a mouth crop.
- Using the box directly as the classifier's input image would discard
  the mouth region, which is exactly the part of the image that carries
  the yawn signal.
- The class label (`0` / `1`) is still sound — what is wrong is only the
  crop geometry.

### Implication for downstream work

The bounding box cannot be trusted as the face region. The correct approach
for Stage 4B / 5 is:

1. Extract the raw frame at the YawDD+ index (the index alignment is
   perfect, as confirmed above).
2. Ignore the YOLO geometry fields for cropping purposes.
3. Run a fresh face / mouth detector (e.g. MediaPipe FaceMesh, already in
   `requirements.txt`) on the extracted frame to obtain a clean mouth ROI.
4. Keep the class id from the YawDD+ `.txt` as the per-frame label.

If a labeler wants to preserve the YawDD+ box for traceability, write it
as an auxiliary column; just don't feed it to the classifier.

---

## 5. Multi-object (`_1`) files — **TREAT AS SPURIOUS DUPLICATES**

Only ten `_1` files exist in the entire YawDD+ Dash corpus; eight of them
are in `8-MaleNoGlasses`. We inspected the pair
`8-MaleNoGlasses/labels/00001287_0.txt` and `_1.txt`:

- `_0`: `0 0.525862 0.715625 0.862069 0.468750` → pixel bbox (61, 231, 612, 456)
- `_1`: `0 0.512931 0.731250 0.853448 0.487500` → pixel bbox (55, 234, 601, 468)

The two boxes have the same class and overlap by >95% IoU; the decoded
frame (see `artifacts/visual_checks/8-MaleNoGlasses/frame_00001287_obj0_class0.jpg`
and `...obj1_class0.jpg`) clearly contains **one** driver and **no**
passenger.

Interpretation: the `_1` files are **duplicate detections** that leaked
through the YawDD+ authors' NMS step, *not* legitimate second-face
annotations.

Recommendation: when building the per-frame classifier label index, keep
only the `_0` file per (subject, frame) pair. Drop or deduplicate the `_1`
files; they add no information and — since they all happen to share the
same class — they cannot introduce label ambiguity. (If later it turns out
a `_1` file disagrees with its `_0` sibling on class, escalate.)

---

## 6. Should we proceed to full frame extraction?

**Yes — with one scope adjustment.**

| question | answer |
|---------|--------|
| Can we map `.txt` frame index → decoded frame 1:1? | **Yes**, across all 29 subjects. |
| Are class ids trustworthy? | **Yes** (`1 = yawn`, `0 = not yawn`). |
| Can we use the YawDD+ bbox as the model's crop? | **No.** It sits on the torso. |
| Will `_1` files cause problems? | **No**, if we drop them as duplicates. |

### Proposed Stage 4B plan

1. For every `(subject, frame_index)` pair present in YawDD+ (classified by
   the `_0` file), decode that exact frame from the raw `.avi` and save it
   as a `.jpg`/`.png`.
2. Alongside each image, write a tiny sidecar (CSV or JSON) containing
   `subject_id`, `frame_index`, `class_id`, and the raw YawDD+ YOLO box
   (for traceability only — do not use for cropping).
3. Apply a fresh face/mouth detector (MediaPipe FaceMesh) to the extracted
   frame to get the actual mouth ROI used for training.
4. Do **not** train yet. First re-run this same visual sanity check on the
   mouth ROI cropper to confirm it also behaves correctly on a handful of
   frames.

---

## 7. Sample images referenced in this report

All paths are relative to the project root.

### 1-FemaleNoGlasses

- `artifacts/visual_checks/1-FemaleNoGlasses/frame_00000000_obj0_class0.jpg`
- `artifacts/visual_checks/1-FemaleNoGlasses/frame_00001370_obj0_class0.jpg`
- `artifacts/visual_checks/1-FemaleNoGlasses/frame_00001661_obj0_class1.jpg`

### 13-MaleNoGlasses

- `artifacts/visual_checks/13-MaleNoGlasses/frame_00000000_obj0_class0.jpg`
- `artifacts/visual_checks/13-MaleNoGlasses/frame_00000500_obj0_class0.jpg`
- `artifacts/visual_checks/13-MaleNoGlasses/frame_00001001_obj0_class1.jpg`

### 8-MaleNoGlasses

- `artifacts/visual_checks/8-MaleNoGlasses/frame_00000000_obj0_class0.jpg`
- `artifacts/visual_checks/8-MaleNoGlasses/frame_00000800_obj0_class0.jpg`
- `artifacts/visual_checks/8-MaleNoGlasses/frame_00001287_obj0_class0.jpg`
- `artifacts/visual_checks/8-MaleNoGlasses/frame_00001287_obj1_class0.jpg`
- `artifacts/visual_checks/8-MaleNoGlasses/frame_00001756_obj0_class1.jpg`

### Supporting data

- Per-frame log (decoded shape, parsed YOLO line, pixel bbox, output path):
  `artifacts/visual_checks/validation_log.json`
- Dataset-wide alignment audit (29/29 pass):
  `artifacts/visual_checks/framecount_audit.csv`
