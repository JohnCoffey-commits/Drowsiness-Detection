# YawDD / YawDD+ Dash — Stages 1 – 3

This document covers **only** the first three stages of the YawDD reconstruction
workflow:

1. **Stage 1** — inspect the YawDD+ Dash annotation format
2. **Stage 2** — inspect the raw YawDD Dash source data
3. **Stage 3** — build a mapping between YawDD+ annotation folders and raw `.avi` videos

No frames are extracted, no labels are aligned, no images are preprocessed,
and no CNN training happens here. Those come later.

---

## 0. Expected dataset layout

Before running anything, the local dataset tree must look like this:

```
Drowsiness_Detection/
├── dataset/
│   ├── YawDD+/
│   │   └── dataset/
│   │       └── Dash/
│   │           ├── 1-FemaleNoGlasses/labels/00000000_0.txt ...
│   │           ├── 1-MaleGlasses/labels/...
│   │           └── ... (29 subject folders total)
│   └── YawDD_raw/
│       ├── Readme_YawDD.pdf
│       ├── Table1.pdf
│       ├── Table2.pdf
│       └── Dash/
│           ├── Female/    (some installs nest an extra Dash/ here)
│           │   ├── 1-FemaleNoGlasses.avi
│           │   └── ...
│           └── Male/
│               ├── 1-MaleGlasses.avi
│               └── ...
└── src/data/
    ├── inspect_yawdd_plus_annotations.py   # Stage 1
    ├── inspect_yawdd_raw_dash.py           # Stage 2
    └── build_yawdd_dash_mapping.py         # Stage 3
```

Both `YawDD_raw/Dash/{Female,Male}/` and `YawDD_raw/Dash/Dash/{Female,Male}/`
are supported; the scripts auto-detect either layout.

The only dependency is the Python standard library — no `pip install` needed
for Stages 1 – 3.

---

## 1. Stage 1 — YawDD+ annotation inspection

```bash
python src/data/inspect_yawdd_plus_annotations.py
```

Optional flags:

| flag | default |
|------|---------|
| `--root`   | `dataset/YawDD+/dataset/Dash` |
| `--report` | `reports/yawdd_plus_annotation_format_report.md` |

What it does:

- Walks every `labels/*.txt` under every subject folder.
- Validates the filename against `^\d{8}_\d+\.txt$`.
- Parses each line as `class cx cy w h` (YOLO-normalised), counts classes,
  rows-per-file, object-index suffixes, and flags out-of-range coordinates.
- Detects frame-index gaps (subjects where indices are non-contiguous).
- Writes a markdown report with:
  - annotation-format summary and real-file examples,
  - per-subject table,
  - **Anomalies** (format, multi-object `_1` files, frame-index skips),
  - **Known Uncertainties**.

Output:

- `reports/yawdd_plus_annotation_format_report.md`

---

## 2. Stage 2 — Raw Dash inspection

```bash
python src/data/inspect_yawdd_raw_dash.py
```

Optional flags:

| flag | default |
|------|---------|
| `--raw-root` | `dataset/YawDD_raw` |
| `--report`   | `reports/yawdd_raw_dash_report.md` |

What it does:

- Lists every file under `Dash/{Female,Male}/` (and the nested
  `Dash/Dash/{Female,Male}/` variant).
- Records file type, size, filename anomalies (`.avi.avi` double extension,
  stray whitespace before `.avi`, ...).
- Derives a **canonical token** for each video: `<idx>-<Gender><GlassesState>`.
- Cross-checks the token set against the YawDD+ subject folders and reports
  any mismatches.
- Notes what the three PDFs under `dataset/YawDD_raw/` add (spoiler:
  background context, not filename info).

Output:

- `reports/yawdd_raw_dash_report.md`

---

## 3. Stage 3 — Build the mapping table

```bash
python src/data/build_yawdd_dash_mapping.py
```

Optional flags:

| flag | default |
|------|---------|
| `--plus-root` | `dataset/YawDD+/dataset/Dash` |
| `--raw-root`  | `dataset/YawDD_raw` |
| `--csv-out`   | `artifacts/mappings/yawdd_dash_mapping.csv` |
| `--report`    | `reports/yawdd_dash_mapping_report.md` |

What it does:

- Indexes every raw `.avi` by canonical token (strips `.avi.avi` and
  whitespace).
- For every YawDD+ subject folder, finds the matching raw video using a
  cascade of rules:
  1. exact token match → `high` confidence,
  2. case-insensitive match → `medium`,
  3. match on `(index, gender)` but disagreeing glasses suffix → `medium`
     (with an explanatory note),
  4. multiple candidates → `low` (refuses to pick),
  5. nothing matched → `none`.
- Writes the CSV (one row per YawDD+ subject folder) and a matching
  human-readable report that calls out all medium/low/none subjects and
  gives an explicit **Recommended Next Step Before Frame Extraction**.

Outputs:

- `artifacts/mappings/yawdd_dash_mapping.csv` with columns:

  ```
  subject_id,
  annotation_folder,
  annotation_txt_path,
  raw_source_path,
  mapping_confidence,
  mapping_notes
  ```

- `reports/yawdd_dash_mapping_report.md`

---

## 4. Where outputs land

| kind        | path                                                          |
|-------------|----------------------------------------------------------------|
| Stage 1 report | `reports/yawdd_plus_annotation_format_report.md`            |
| Stage 2 report | `reports/yawdd_raw_dash_report.md`                          |
| Stage 3 CSV    | `artifacts/mappings/yawdd_dash_mapping.csv`                 |
| Stage 3 report | `reports/yawdd_dash_mapping_report.md`                      |

---

## 5. Before proceeding to Stage 4 (frame extraction)

Read **section 4** of `reports/yawdd_dash_mapping_report.md` (the
"Recommended Next Step Before Frame Extraction" section). It describes the
small visual sanity check that must pass before a bulk decode is run —
in particular, confirming that the YawDD+ frame index matches the index
produced by the raw-video decoder, and that class `1` really does mean
"yawning".
