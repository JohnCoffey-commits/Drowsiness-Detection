# YawDD+ Dash — Annotation Format Report (Stage 1)

This report is the result of a forensic inspection of every `.txt` file under `dataset/YawDD+/dataset/Dash/<subject>/labels/`. No frames were extracted, no labels were aligned, and no images were touched.

- Scanned root: `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/dataset/YawDD+/dataset/Dash`
- Subject folders: **29**
- Total annotation files: **64388**

## 1. Summary of the annotation structure

Each subject folder contains exactly one sub-folder called `labels/`. Every file inside `labels/` is a small plain-text file named like

```
<8-digit frame index>_<object index>.txt
```

Each file contains **one line per detected object**, in the canonical YOLO-v5 / Darknet bounding-box format:

```
<class_id> <x_center> <y_center> <width> <height>
```

All four geometry fields are normalised to `[0, 1]` relative to the (unseen) source image. The class id is an integer.

## 2. Filename pattern

Across **all** scanned files, the filename conforms to the regex `^\d{8}_\d+\.txt$`. The first group is a zero-padded, monotonically increasing frame counter starting at `00000000`. The second group is an object index suffix.

Object-index suffix distribution (across the whole corpus):

| suffix | count |
|--------|-------|
| `_0` | 64378 |
| `_1` | 10 |

## 3. Rows per file

| rows in file | number of files |
|--------------|-----------------|
| 1 | 64388 |

**Every** scanned file contains exactly one bounding box. This confirms the interpretation: one file == one annotated frame == one face bounding box.

## 4. Class-id distribution

| class id | row count | share |
|----------|-----------|-------|
| 0 | 57357 | 89.08% |
| 1 | 7031 | 10.92% |

Only two class ids appear in the whole corpus: `0` and `1`. Given the purpose of YawDD (a yawning-detection dataset) the most plausible interpretation — **subject to confirmation against a few sample frames once we extract them** — is:

- `0` — face present, **not yawning** (normal / talking)

- `1` — face present, **yawning**

## 5. Per-subject statistics

| subject | files | frame min | frame max | frames unique | class 0 | class 1 |
|---------|-------|-----------|-----------|---------------|---------|---------|
| 1-FemaleNoGlasses | 2741 | 0 | 2740 | 2741 | 2212 | 529 |
| 1-MaleGlasses | 2576 | 0 | 2575 | 2576 | 2166 | 410 |
| 10-FemaleNoGlasses | 1213 | 0 | 1216 | 1213 | 1170 | 43 |
| 10-MaleGlasses | 2418 | 0 | 2417 | 2418 | 2137 | 281 |
| 11-FemaleGlasses | 1683 | 0 | 1682 | 1683 | 1447 | 236 |
| 11-MaleGlasses | 1760 | 0 | 1759 | 1760 | 1556 | 204 |
| 12-FemaleGlasses | 2514 | 0 | 2517 | 2514 | 2352 | 162 |
| 12-MaleGlasses | 1957 | 0 | 1956 | 1957 | 1693 | 264 |
| 13-FemaleGlasses | 2488 | 0 | 2487 | 2488 | 2393 | 95 |
| 13-MaleNoGlasses | 2140 | 0 | 2189 | 2140 | 1942 | 198 |
| 14-MaleNoGlasses | 3010 | 0 | 3009 | 3010 | 2682 | 328 |
| 15-MaleGlasses | 2640 | 0 | 2638 | 2639 | 2432 | 208 |
| 16-MaleNoGlasses | 2503 | 0 | 2502 | 2503 | 2330 | 173 |
| 2-FemaleNoGlasses | 2180 | 0 | 2179 | 2180 | 1956 | 224 |
| 2-MaleGlasses | 2330 | 0 | 2363 | 2329 | 1977 | 353 |
| 3-FemaleGlasses | 3057 | 0 | 3056 | 3057 | 2575 | 482 |
| 3-MaleGlasses | 2027 | 22 | 2085 | 2027 | 1841 | 186 |
| 4-FemaleNoGlasses | 1494 | 0 | 1495 | 1494 | 1402 | 92 |
| 4-MaleNoGlasses | 1998 | 0 | 1997 | 1998 | 1679 | 319 |
| 5-FemaleNoGlasses | 2149 | 0 | 2148 | 2149 | 1995 | 154 |
| 5-MaleGlasses | 2395 | 0 | 2394 | 2395 | 2270 | 125 |
| 6-FemaleNoGlasses | 1477 | 0 | 1490 | 1477 | 1129 | 348 |
| 6-MaleGlasses | 1633 | 0 | 1632 | 1633 | 1395 | 238 |
| 7-FemaleNoGlasses | 3618 | 0 | 3617 | 3618 | 3400 | 218 |
| 7-MaleGlasses | 2398 | 0 | 2397 | 2398 | 2088 | 310 |
| 8-FemaleGlasses | 2297 | 0 | 2296 | 2297 | 1970 | 327 |
| 8-MaleNoGlasses | 2140 | 0 | 2178 | 2132 | 1997 | 143 |
| 9-FemaleNoGlasses | 1532 | 0 | 1531 | 1532 | 1465 | 67 |
| 9-MaleNoGlasses | 2020 | 0 | 2019 | 2020 | 1706 | 314 |

## 6. Real examples copied from disk

Below are three verbatim samples (first file, middle file, last file) from four different subjects, chosen to cover both the Female/Male and Glasses/NoGlasses axes.

### 1-FemaleNoGlasses

`00000000_0.txt`:
```
0 0.448905 0.786070 0.781022 0.398010
```

`00001370_0.txt`:
```
0 0.481618 0.784264 0.683824 0.380711
```

`00002740_0.txt`:
```
0 0.435606 0.783422 0.704545 0.368984
```

### 1-MaleGlasses

`00000000_0.txt`:
```
0 0.518987 0.733831 0.670886 0.353234
```

`00001288_0.txt`:
```
0 0.524691 0.760465 0.679012 0.367442
```

`00002575_0.txt`:
```
0 0.515244 0.782222 0.664634 0.355556
```

### 13-MaleNoGlasses

`00000000_0.txt`:
```
0 0.486207 0.805851 0.696552 0.335106
```

`00001070_0.txt`:
```
0 0.697080 0.797872 0.605839 0.340426
```

`00002189_0.txt`:
```
0 0.485401 0.771930 0.722628 0.397661
```

### 11-FemaleGlasses

`00000000_0.txt`:
```
0 0.481752 0.788265 0.656934 0.352041
```

`00000841_0.txt`:
```
0 0.503521 0.769036 0.654930 0.370558
```

`00001682_0.txt`:
```
0 0.517857 0.762255 0.750000 0.387255
```

## 7. Anomalies and data-quality notes

### 7.1 Format-level

- Every filename matches the canonical pattern, every line parses as a YOLO bounding box, and every coordinate is within `[0, 1]`. The format itself is clean.

### 7.2 Multi-object frames (`_1` suffix)

- **10** file(s) carry a non-zero object-index suffix. That means YawDD+ occasionally stores **two (or more) faces per frame** — the same frame index appears as `<frame>_0.txt` *and* `<frame>_1.txt`, with one bounding box each. Treat these as multi-face frames, not as duplicates.

| subject | filename |
|---------|----------|
| 15-MaleGlasses | `00002070_1.txt` |
| 2-MaleGlasses | `00000314_1.txt` |
| 8-MaleNoGlasses | `00001287_1.txt` |
| 8-MaleNoGlasses | `00001289_1.txt` |
| 8-MaleNoGlasses | `00001291_1.txt` |
| 8-MaleNoGlasses | `00001293_1.txt` |
| 8-MaleNoGlasses | `00001295_1.txt` |
| 8-MaleNoGlasses | `00001298_1.txt` |
| 8-MaleNoGlasses | `00001302_1.txt` |
| 8-MaleNoGlasses | `00001318_1.txt` |

### 7.3 Frame-index skips

- **8** subject(s) have gaps between the minimum and maximum frame index. The YawDD+ authors appear to have **dropped some frames** (likely frames where the face detector failed). Keep the frame index, not a running counter, as the canonical key when extracting frames from the raw video.

| subject | frame_min | frame_max | expected if contiguous | actually present |
|---------|-----------|-----------|------------------------|-------------------|
| 10-FemaleNoGlasses | 0 | 1216 | 1217 | 1213 |
| 12-FemaleGlasses | 0 | 2517 | 2518 | 2514 |
| 13-MaleNoGlasses | 0 | 2189 | 2190 | 2140 |
| 2-MaleGlasses | 0 | 2363 | 2364 | 2329 |
| 3-MaleGlasses | 22 | 2085 | 2064 | 2027 |
| 4-FemaleNoGlasses | 0 | 1495 | 1496 | 1494 |
| 6-FemaleNoGlasses | 0 | 1490 | 1491 | 1477 |
| 8-MaleNoGlasses | 0 | 2178 | 2179 | 2132 |

## 8. Safe assumptions going forward

- Each `.txt` file annotates **one image / one frame** (filename is a zero-padded frame index).
- Each file contains **exactly one bounding box** per line, in YOLO-normalised `<class> <cx> <cy> <w> <h>` format.
- The class id is a **binary yawn vs non-yawn label** (`0` = not yawning, `1` = yawning) — pending visual confirmation on a few frames.
- The annotation format is independent of gender / glasses; the subject-folder name is the only place that encodes those attributes.
- The `_<n>` suffix is an **object index** within a frame. In this corpus it is almost always `_0` (one face per frame), but a handful of frames also have a `_1.txt` companion when a second face was annotated.
- Frame indices are **not guaranteed to be contiguous**: some subjects skip indices because the YawDD+ authors dropped frames where no reliable detection was possible. Use the index in the filename, never a running counter, when matching to decoded frames.

## 9. Known Uncertainties

- **Class semantics are not written down anywhere in the YawDD+ folder.** The `0`/`1` meaning above is inferred from the dataset's stated purpose and from which files carry class `1` (a small minority, consistent with yawn-positive frames). This must be visually verified once we extract the matching images.
- **Images themselves are not present** under `YawDD+/dataset/Dash/`; only the `labels/` folder exists. The source frames must therefore be reconstructed from the raw YawDD Dash videos (Stage 2 and Stage 3).
- **Frame-rate / frame indexing convention** of the YawDD+ authors is not documented. We assume they decoded each raw `.avi` at its native 30 fps and wrote one `.txt` per decoded frame, but this must be verified by counting actual video frames during Stage 4.
- **Frame-index gaps.** We do not yet know *why* the authors skipped certain indices. The two most likely causes are (a) the YawDD+ face detector failed / was filtered on those frames, or (b) they decoded the raw video at a different effective rate. This matters: when we decode the raw video in Stage 4 we must index frames the same way the annotators did, otherwise labels and images will drift. A small alignment experiment on one subject is required before bulk extraction.
- **Object-index suffix (`_1`).** Ten files carry a `_1` suffix. Until we visualise one, we do not know whether the second box is a passenger's face, a reflection, or something else. For the binary yawn classifier this is unlikely to matter, but the labeling has to be decided (drop, keep, or treat specially).

