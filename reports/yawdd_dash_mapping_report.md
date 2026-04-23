# YawDD Dash — Annotation ↔ Raw Video Mapping Report (Stage 3)

This report is produced by `src/data/build_yawdd_dash_mapping.py`. It pairs every subject folder under `YawDD+/dataset/Dash/` with the matching raw `.avi` under `YawDD_raw/Dash/`. The CSV output is

`/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/artifacts/mappings/yawdd_dash_mapping.csv`

- YawDD+ root: `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/dataset/YawDD+/dataset/Dash`
- Raw root:    `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/dataset/YawDD_raw`
- Subjects processed: **29**
- Confidence: high=29, medium=0, low=0, none=0

## 1. Matching logic

The algorithm normalises each raw `.avi` filename by repeatedly stripping a trailing `.avi` (to collapse names like `11-FemaleGlasses.avi.avi`) and removing stray whitespace. The resulting string — for example `13-MaleNoGlasses` — is the **canonical token**.

Each YawDD+ subject folder is already in canonical form, so the primary key is an exact string match between the folder name and the canonical token of a raw `.avi`.

Confidence levels used in the CSV:

- **high** — exact canonical-token match (no ambiguity, no heuristic). Safe to use for frame extraction as-is.
- **medium** — match obtained by a case-insensitive comparison, or by dropping the `Glasses` / `NoGlasses` / `SunGlasses` suffix because the raw file and the YawDD+ folder disagree on it. The subject is the same person but the attribute label must be reviewed by hand.
- **low** — multiple candidates remain even after the heuristics above. The script refuses to pick one.
- **none** — no raw video could be associated with this YawDD+ folder. Frame extraction is blocked for this subject.

## 2. Concrete rules applied in code

1. `canonical_token = name.strip().removesuffix('.avi').removesuffix('.avi').strip()`
2. Exact token match → confidence **high**.
3. Case-insensitive token match → confidence **medium** (note the raw token).
4. Match on `(subject_index, gender)` ignoring `GlassesState` →
   - exactly one candidate → **medium** with a note describing the disagreement.
   - more than one candidate → **low** with the list of candidates.
5. No candidate → **none**.

## 3. Per-subject mapping

| subject_id | YawDD+ folder | raw video (basename) | confidence | notes |
|------------|---------------|-----------------------|------------|-------|
| 1-Female | `1-FemaleNoGlasses` | `1-FemaleNoGlasses.avi` | high | exact canonical-token match |
| 1-Male | `1-MaleGlasses` | `1-MaleGlasses.avi` | high | exact canonical-token match |
| 10-Female | `10-FemaleNoGlasses` | `10-FemaleNoGlasses.avi` | high | exact canonical-token match |
| 10-Male | `10-MaleGlasses` | `10-MaleGlasses.avi` | high | exact canonical-token match |
| 11-Female | `11-FemaleGlasses` | `11-FemaleGlasses.avi.avi` | high | exact canonical-token match |
| 11-Male | `11-MaleGlasses` | `11-MaleGlasses.avi` | high | exact canonical-token match |
| 12-Female | `12-FemaleGlasses` | `12-FemaleGlasses.avi.avi` | high | exact canonical-token match |
| 12-Male | `12-MaleGlasses` | `12-MaleGlasses.avi` | high | exact canonical-token match |
| 13-Female | `13-FemaleGlasses` | `13-FemaleGlasses.avi.avi` | high | exact canonical-token match |
| 13-Male | `13-MaleNoGlasses` | `13-MaleNoGlasses .avi` | high | exact canonical-token match |
| 14-Male | `14-MaleNoGlasses` | `14-MaleNoGlasses.avi` | high | exact canonical-token match |
| 15-Male | `15-MaleGlasses` | `15-MaleGlasses.avi` | high | exact canonical-token match |
| 16-Male | `16-MaleNoGlasses` | `16-MaleNoGlasses.avi` | high | exact canonical-token match |
| 2-Female | `2-FemaleNoGlasses` | `2-FemaleNoGlasses.avi` | high | exact canonical-token match |
| 2-Male | `2-MaleGlasses` | `2-MaleGlasses.avi` | high | exact canonical-token match |
| 3-Female | `3-FemaleGlasses` | `3-FemaleGlasses.avi` | high | exact canonical-token match |
| 3-Male | `3-MaleGlasses` | `3-MaleGlasses.avi` | high | exact canonical-token match |
| 4-Female | `4-FemaleNoGlasses` | `4-FemaleNoGlasses.avi` | high | exact canonical-token match |
| 4-Male | `4-MaleNoGlasses` | `4-MaleNoGlasses.avi` | high | exact canonical-token match |
| 5-Female | `5-FemaleNoGlasses` | `5-FemaleNoGlasses.avi` | high | exact canonical-token match |
| 5-Male | `5-MaleGlasses` | `5-MaleGlasses.avi` | high | exact canonical-token match |
| 6-Female | `6-FemaleNoGlasses` | `6-FemaleNoGlasses.avi` | high | exact canonical-token match |
| 6-Male | `6-MaleGlasses` | `6-MaleGlasses.avi` | high | exact canonical-token match |
| 7-Female | `7-FemaleNoGlasses` | `7-FemaleNoGlasses.avi` | high | exact canonical-token match |
| 7-Male | `7-MaleGlasses` | `7-MaleGlasses.avi` | high | exact canonical-token match |
| 8-Female | `8-FemaleGlasses` | `8-FemaleGlasses.avi.avi` | high | exact canonical-token match |
| 8-Male | `8-MaleNoGlasses` | `8-MaleNoGlasses.avi` | high | exact canonical-token match |
| 9-Female | `9-FemaleNoGlasses` | `9-FemaleNoGlasses.avi` | high | exact canonical-token match |
| 9-Male | `9-MaleNoGlasses` | `9-MaleNoGlasses.avi` | high | exact canonical-token match |

## 4. Recommended Next Step Before Frame Extraction

All 29 subjects mapped with **high** confidence. It is safe to proceed to Stage 4 (frame extraction from the raw videos) using the `raw_source_path` column of the CSV as the input list.

Before running a bulk decode, do a **tiny sanity check** on one subject first:

1. Decode frame `00000000` from the raw video for, e.g., `1-FemaleNoGlasses.avi`.
2. Load the matching annotation `1-FemaleNoGlasses/labels/00000000_0.txt` and draw its bounding box on the decoded image.
3. Confirm visually that the box frames the driver's face.
4. Repeat for a frame whose label is class `1` (e.g. the `00001661_0.txt` file under `1-FemaleNoGlasses`) and verify the driver is visibly yawning on that frame.

Only after those two checks pass should you run the bulk extractor. That will simultaneously confirm the frame-index convention and the class-id semantics that Stage 1 flagged as uncertain.

