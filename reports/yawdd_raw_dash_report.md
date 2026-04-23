# YawDD Raw — Dash Source Inspection Report (Stage 2)

This report describes the raw, as-delivered YawDD Dash data that we will later use to reconstruct frames for the YawDD+ annotations. No videos were decoded — we only inspected file names, sizes and the accompanying PDFs.

- Scanned root: `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/dataset/YawDD_raw`
- Female directory: `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/dataset/YawDD_raw/Dash/Dash/Female`
- Male directory:   `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/dataset/YawDD_raw/Dash/Dash/Male`
- Raw files found: **29** (Female: 13, Male: 16)

## 1. File types found

| extension | count |
|-----------|-------|
| `.avi` | 29 |

All raw source files are **`.avi` videos** (single container per subject). No image sequences, no per-frame folders.

## 2. Naming patterns observed

The canonical name of each raw video matches the pattern `<subject_index>-<Gender><GlassesState>.avi` where

- `<subject_index>` is an integer 1..16 (males) or 1..13 (females),
- `<Gender>` is literally `Female` or `Male`,
- `<GlassesState>` is `Glasses`, `NoGlasses`, or `SunGlasses`.

### Observed deviations from the canonical pattern

- **4 file(s)** have a duplicated extension (`.avi.avi`):
  - `Female/11-FemaleGlasses.avi.avi`
  - `Female/12-FemaleGlasses.avi.avi`
  - `Female/13-FemaleGlasses.avi.avi`
  - `Female/8-FemaleGlasses.avi.avi`
- **1 file(s)** have whitespace inside the name, typically a stray space before `.avi`:
  - `Male/13-MaleNoGlasses .avi`

### Full Female listing

| file | size | subject | gender | glasses | canonical token |
|------|------|---------|--------|---------|------------------|
| `1-FemaleNoGlasses.avi` | 162.4 MB | 1 | Female | NoGlasses | `1-FemaleNoGlasses` |
| `2-FemaleNoGlasses.avi` | 129.6 MB | 2 | Female | NoGlasses | `2-FemaleNoGlasses` |
| `3-FemaleGlasses.avi` | 178.0 MB | 3 | Female | Glasses | `3-FemaleGlasses` |
| `4-FemaleNoGlasses.avi` | 90.3 MB | 4 | Female | NoGlasses | `4-FemaleNoGlasses` |
| `5-FemaleNoGlasses.avi` | 126.4 MB | 5 | Female | NoGlasses | `5-FemaleNoGlasses` |
| `6-FemaleNoGlasses.avi` | 90.7 MB | 6 | Female | NoGlasses | `6-FemaleNoGlasses` |
| `7-FemaleNoGlasses.avi` | 223.5 MB | 7 | Female | NoGlasses | `7-FemaleNoGlasses` |
| `8-FemaleGlasses.avi.avi` | 136.0 MB | 8 | Female | Glasses | `8-FemaleGlasses` |
| `9-FemaleNoGlasses.avi` | 93.0 MB | 9 | Female | NoGlasses | `9-FemaleNoGlasses` |
| `10-FemaleNoGlasses.avi` | 71.0 MB | 10 | Female | NoGlasses | `10-FemaleNoGlasses` |
| `11-FemaleGlasses.avi.avi` | 98.9 MB | 11 | Female | Glasses | `11-FemaleGlasses` |
| `12-FemaleGlasses.avi.avi` | 147.0 MB | 12 | Female | Glasses | `12-FemaleGlasses` |
| `13-FemaleGlasses.avi.avi` | 148.2 MB | 13 | Female | Glasses | `13-FemaleGlasses` |

### Full Male listing

| file | size | subject | gender | glasses | canonical token |
|------|------|---------|--------|---------|------------------|
| `1-MaleGlasses.avi` | 153.8 MB | 1 | Male | Glasses | `1-MaleGlasses` |
| `2-MaleGlasses.avi` | 142.8 MB | 2 | Male | Glasses | `2-MaleGlasses` |
| `3-MaleGlasses.avi` | 118.8 MB | 3 | Male | Glasses | `3-MaleGlasses` |
| `4-MaleNoGlasses.avi` | 120.6 MB | 4 | Male | NoGlasses | `4-MaleNoGlasses` |
| `5-MaleGlasses.avi` | 142.4 MB | 5 | Male | Glasses | `5-MaleGlasses` |
| `6-MaleGlasses.avi` | 97.0 MB | 6 | Male | Glasses | `6-MaleGlasses` |
| `7-MaleGlasses.avi` | 139.5 MB | 7 | Male | Glasses | `7-MaleGlasses` |
| `8-MaleNoGlasses.avi` | 129.7 MB | 8 | Male | NoGlasses | `8-MaleNoGlasses` |
| `9-MaleNoGlasses.avi` | 122.7 MB | 9 | Male | NoGlasses | `9-MaleNoGlasses` |
| `10-MaleGlasses.avi` | 142.3 MB | 10 | Male | Glasses | `10-MaleGlasses` |
| `11-MaleGlasses.avi` | 105.2 MB | 11 | Male | Glasses | `11-MaleGlasses` |
| `12-MaleGlasses.avi` | 113.9 MB | 12 | Male | Glasses | `12-MaleGlasses` |
| `13-MaleNoGlasses .avi` | 128.7 MB | 13 | Male | NoGlasses | `13-MaleNoGlasses` |
| `14-MaleNoGlasses.avi` | 180.9 MB | 14 | Male | NoGlasses | `14-MaleNoGlasses` |
| `15-MaleGlasses.avi` | 156.1 MB | 15 | Male | Glasses | `15-MaleGlasses` |
| `16-MaleNoGlasses.avi` | 144.5 MB | 16 | Male | NoGlasses | `16-MaleNoGlasses` |

## 3. Metadata inferable from folder/file names

From the folder and file names alone we can recover, for every raw video: the subject index, the gender, and the glasses state. No other per-video metadata (session id, clip id, time range, yawning time stamps, ...) is present in the filenames — the raw Dash videos are delivered as a single continuous clip per subject containing the three scripted segments (Normal / Talking / Yawning) back-to-back.

## 4. Compatibility with YawDD+ subject folders

- YawDD+ Dash subject folders: **29**
- Raw Dash videos (canonical tokens): **29**
- Tokens present in **both** YawDD+ and raw: **29**
- Tokens **only** in YawDD+ (no raw video): **0** — none
- Tokens **only** in raw (no YawDD+ labels): **0** — none

**The canonical tokens match one-to-one.** Every YawDD+ subject folder has a raw `.avi` with the same `<index>-<Gender><Glasses>` token, after normalising the `.avi.avi` and whitespace anomalies above. Subject-level mapping is therefore fully determined by folder name == canonical raw-video stem.

## 5. Value added by the supplied PDFs

The three PDFs under `dataset/YawDD_raw/` were read for context; none of them add filename-level information that was not already recoverable from the folder structure.

| PDF | present | summary of relevance |
|-----|---------|-----------------------|
| `Readme_YawDD.pdf` | yes | Confirms the Dash dataset has **29 videos** (1 per participant), 30 fps, 640x480 24-bit RGB AVI, no audio, with each participant performing Normal / Talking / Yawning segments inside a single continuous clip. |
| `Table1.pdf` | yes | Describes the **Mirror** dataset (camera under front mirror, 322 short clips). Not directly relevant to Dash, kept here only for completeness. |
| `Table2.pdf` | yes | Per-subject metadata for the **Dash** dataset (16 males + 13 females = 29 subjects), matching the 29 raw videos and 29 YawDD+ folders. Useful later for fairness/ethnicity reporting, but not needed for the frame mapping. |

## 6. Preliminary judgment on subject-level mapping

**Feasible with high confidence.** The 29 YawDD+ annotation folders and the 29 raw Dash videos share identical canonical subject tokens after trivial whitespace / double-extension normalisation. Stage 3 can safely build a 1-to-1 mapping table keyed on that token.

