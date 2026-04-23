# YawDD Dash — Reconstruction Verification Report (Stage 4B)

This report verifies the labeled-frame dataset that was rebuilt from the raw `.avi` videos using the YawDD+ annotation filenames as the source of frame indices and class labels.

- Manifest: `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/artifacts/mappings/yawdd_dash_all_labeled_frames.csv`
- YawDD+ source: `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/dataset/YawDD+/dataset/Dash`

## 1. Global totals

- Subjects reconstructed: **29**
- Total labeled frames (rows in manifest): **64378**
- `no_yawn` (class 0): **57347** (89.08%)
- `yawn`    (class 1): **7031** (10.92%)
- `had_duplicate_box = true` rows (i.e. frames that also had a `_1` sibling, which we ignored): **10**

### Extraction-status breakdown

| status | count |
|--------|-------|
| `extracted` | 64378 |

- Missing JPEGs (manifest says extracted but file not on disk): **0**
- Total disk footprint of extracted frames: **3.9 GB**

## 2. Per-subject cross-check against YawDD+ source

Expected behaviour: `manifest rows == number of `_0` files in YawDD+/labels/` for every subject. Any other result means the extractor dropped or duplicated frames.

| subject | YawDD+ `.txt` files | of which `_0` | manifest rows | yawn | no_yawn | dup flagged | missing JPEG | disk |
|---------|---------------------|---------------|---------------|------|---------|-------------|--------------|------|
| 1-FemaleNoGlasses | 2741 | 2741 | 2741 | 529 | 2212 | 0 | 0 | 175.6 MB |
| 1-MaleGlasses | 2576 | 2576 | 2576 | 410 | 2166 | 0 | 0 | 158.1 MB |
| 10-FemaleNoGlasses | 1213 | 1213 | 1213 | 43 | 1170 | 0 | 0 | 77.9 MB |
| 10-MaleGlasses | 2418 | 2418 | 2418 | 281 | 2137 | 0 | 0 | 164.0 MB |
| 11-FemaleGlasses | 1683 | 1683 | 1683 | 236 | 1447 | 0 | 0 | 109.8 MB |
| 11-MaleGlasses | 1760 | 1760 | 1760 | 204 | 1556 | 0 | 0 | 112.1 MB |
| 12-FemaleGlasses | 2514 | 2514 | 2514 | 162 | 2352 | 0 | 0 | 148.8 MB |
| 12-MaleGlasses | 1957 | 1957 | 1957 | 264 | 1693 | 0 | 0 | 127.4 MB |
| 13-FemaleGlasses | 2488 | 2488 | 2488 | 95 | 2393 | 0 | 0 | 155.8 MB |
| 13-MaleNoGlasses | 2140 | 2140 | 2140 | 198 | 1942 | 0 | 0 | 151.0 MB |
| 14-MaleNoGlasses | 3010 | 3010 | 3010 | 328 | 2682 | 0 | 0 | 184.0 MB |
| 15-MaleGlasses | 2640 | 2639 | 2639 | 208 | 2431 | 1 | 0 | 164.5 MB |
| 16-MaleNoGlasses | 2503 | 2503 | 2503 | 173 | 2330 | 0 | 0 | 165.3 MB |
| 2-FemaleNoGlasses | 2180 | 2180 | 2180 | 224 | 1956 | 0 | 0 | 133.2 MB |
| 2-MaleGlasses | 2330 | 2329 | 2329 | 353 | 1976 | 1 | 0 | 145.6 MB |
| 3-FemaleGlasses | 3057 | 3057 | 3057 | 482 | 2575 | 0 | 0 | 182.5 MB |
| 3-MaleGlasses | 2027 | 2027 | 2027 | 186 | 1841 | 0 | 0 | 113.7 MB |
| 4-FemaleNoGlasses | 1494 | 1494 | 1494 | 92 | 1402 | 0 | 0 | 86.7 MB |
| 4-MaleNoGlasses | 1998 | 1998 | 1998 | 319 | 1679 | 0 | 0 | 115.9 MB |
| 5-FemaleNoGlasses | 2149 | 2149 | 2149 | 154 | 1995 | 0 | 0 | 130.2 MB |
| 5-MaleGlasses | 2395 | 2395 | 2395 | 125 | 2270 | 0 | 0 | 152.8 MB |
| 6-FemaleNoGlasses | 1477 | 1477 | 1477 | 348 | 1129 | 0 | 0 | 85.3 MB |
| 6-MaleGlasses | 1633 | 1633 | 1633 | 238 | 1395 | 0 | 0 | 96.1 MB |
| 7-FemaleNoGlasses | 3618 | 3618 | 3618 | 218 | 3400 | 0 | 0 | 226.7 MB |
| 7-MaleGlasses | 2398 | 2398 | 2398 | 310 | 2088 | 0 | 0 | 156.9 MB |
| 8-FemaleGlasses | 2297 | 2297 | 2297 | 327 | 1970 | 0 | 0 | 148.6 MB |
| 8-MaleNoGlasses | 2140 | 2132 | 2132 | 143 | 1989 | 8 | 0 | 139.9 MB |
| 9-FemaleNoGlasses | 1532 | 1532 | 1532 | 67 | 1465 | 0 | 0 | 93.7 MB |
| 9-MaleNoGlasses | 2020 | 2020 | 2020 | 314 | 1706 | 0 | 0 | 125.6 MB |

## 3. Diagnostics

### 3.1 Non-success extraction statuses

- None. Every row has `extraction_status ∈ {extracted, skipped_existing}`.

### 3.2 Manifest-vs-YawDD+ consistency

- All 29 subjects: manifest row count equals the number of `_0` annotation files (i.e. exactly one row per unique (subject, frame) pair, after dropping `_1` duplicates). ✓

### 3.3 Duplicate-box accounting

- YawDD+ source contains **10** `.txt` files with a non-zero object-index suffix (`_1`, `_2`, …). These are the spurious duplicate detections the Stage-4A visual check identified.
- Manifest flags **10** frames as `had_duplicate_box = true`, i.e. frames where at least one `_n≥1` sibling existed in the source and was ignored for labelling.

- The numbers match: every `_1` file in the source corresponds to exactly one flagged frame in the manifest.

## 4. Readiness for Stage 5 (mouth-ROI generation)

**Ready.** The reconstructed labeled-frame dataset passes every check:
- Every YawDD+ `_0` annotation has exactly one extracted JPEG on disk.
- Every row has a valid class id in {0, 1} and a parseable bounding-box line.
- No extraction failures.
- No missing images.
- `_1` duplicates are accounted for via `had_duplicate_box`.

Stage 5 can consume `artifacts/mappings/yawdd_dash_all_labeled_frames.csv` as the definitive per-frame input list. The mouth-ROI detector (e.g. MediaPipe FaceMesh) should read `image_path` and ignore the `yawdd_bbox_raw` column except for traceability.

