# YawDD+ Dash - Stage 5 Mouth-Crop Report

- Merged manifest: `artifacts/mappings/yawdd_dash_all_mouth_crops.csv`
- Crop root: `dataset/YawDD_plus_reconstructed/Dash/mouth_crops/`
- QC samples: `artifacts/visual_checks/mouth_crops`

## Processing statistics

- Total frames processed: **64378**
- MediaPipe Face Mesh crops (`face_mesh`): **64093**
- Fallback lower-face crops (`fallback_lower_face`): **109**
- Resumed from a prior run (`resumed`): **0**
- Failed (no crop saved, `failed`): **176**
- Success rate: **99.73%**

## Class distribution across saved crops

| Class | Count |
|---|---|
| `no_yawn` | 57171 |
| `yawn` | 7031 |

## Per-method class breakdown

| Method | no_yawn | yawn |
|---|---|---|
| `face_mesh` | 57065 | 7028 |
| `fallback_lower_face` | 106 | 3 |
| `resumed` | 0 | 0 |
| `failed` | 176 | 0 |

## Per-subject method counts

| subject_id | face_mesh | fallback | resumed | failed |
|---|---|---|---|---|
| `1-FemaleNoGlasses` | 2741 | 0 | 0 | 0 |
| `1-MaleGlasses` | 2576 | 0 | 0 | 0 |
| `10-FemaleNoGlasses` | 1205 | 1 | 0 | 7 |
| `10-MaleGlasses` | 2418 | 0 | 0 | 0 |
| `11-FemaleGlasses` | 1683 | 0 | 0 | 0 |
| `11-MaleGlasses` | 1713 | 11 | 0 | 36 |
| `12-FemaleGlasses` | 2375 | 86 | 0 | 53 |
| `12-MaleGlasses` | 1954 | 0 | 0 | 3 |
| `13-FemaleGlasses` | 2488 | 0 | 0 | 0 |
| `13-MaleNoGlasses` | 2140 | 0 | 0 | 0 |
| `14-MaleNoGlasses` | 3010 | 0 | 0 | 0 |
| `15-MaleGlasses` | 2639 | 0 | 0 | 0 |
| `16-MaleNoGlasses` | 2503 | 0 | 0 | 0 |
| `2-FemaleNoGlasses` | 2180 | 0 | 0 | 0 |
| `2-MaleGlasses` | 2312 | 0 | 0 | 17 |
| `3-FemaleGlasses` | 3057 | 0 | 0 | 0 |
| `3-MaleGlasses` | 2027 | 0 | 0 | 0 |
| `4-FemaleNoGlasses` | 1492 | 0 | 0 | 2 |
| `4-MaleNoGlasses` | 1998 | 0 | 0 | 0 |
| `5-FemaleNoGlasses` | 2149 | 0 | 0 | 0 |
| `5-MaleGlasses` | 2395 | 0 | 0 | 0 |
| `6-FemaleNoGlasses` | 1473 | 4 | 0 | 0 |
| `6-MaleGlasses` | 1615 | 0 | 0 | 18 |
| `7-FemaleNoGlasses` | 3618 | 0 | 0 | 0 |
| `7-MaleGlasses` | 2398 | 0 | 0 | 0 |
| `8-FemaleGlasses` | 2297 | 0 | 0 | 0 |
| `8-MaleNoGlasses` | 2085 | 7 | 0 | 40 |
| `9-FemaleNoGlasses` | 1532 | 0 | 0 | 0 |
| `9-MaleNoGlasses` | 2020 | 0 | 0 | 0 |

## File-integrity checks

- Crop rows whose image file is MISSING on disk: **0**
- Readability sample size: 500 (of 64202 saved crops)
- Unreadable JPEGs in the sample: **0**

## Random example crops (manual inspection)

- `12-MaleGlasses` / `00000355` / `no_yawn` / `face_mesh` -> `dataset/YawDD_plus_reconstructed/Dash/mouth_crops/12-MaleGlasses/00000355.jpg`
- `6-FemaleNoGlasses` / `00000555` / `no_yawn` / `face_mesh` -> `dataset/YawDD_plus_reconstructed/Dash/mouth_crops/6-FemaleNoGlasses/00000555.jpg`
- `6-FemaleNoGlasses` / `00000984` / `yawn` / `face_mesh` -> `dataset/YawDD_plus_reconstructed/Dash/mouth_crops/6-FemaleNoGlasses/00000984.jpg`
- `6-FemaleNoGlasses` / `00000696` / `no_yawn` / `face_mesh` -> `dataset/YawDD_plus_reconstructed/Dash/mouth_crops/6-FemaleNoGlasses/00000696.jpg`
- `3-FemaleGlasses` / `00001520` / `no_yawn` / `face_mesh` -> `dataset/YawDD_plus_reconstructed/Dash/mouth_crops/3-FemaleGlasses/00001520.jpg`
- `13-FemaleGlasses` / `00001508` / `no_yawn` / `face_mesh` -> `dataset/YawDD_plus_reconstructed/Dash/mouth_crops/13-FemaleGlasses/00001508.jpg`
- `13-MaleNoGlasses` / `00001716` / `no_yawn` / `face_mesh` -> `dataset/YawDD_plus_reconstructed/Dash/mouth_crops/13-MaleNoGlasses/00001716.jpg`
- `13-MaleNoGlasses` / `00000592` / `no_yawn` / `face_mesh` -> `dataset/YawDD_plus_reconstructed/Dash/mouth_crops/13-MaleNoGlasses/00000592.jpg`

## Visual QC samples

A side-by-side (full-frame with bbox | crop) image was rendered for up to 4 examples per (method, class) bucket.
Total QC panels written: **15**.
Location: `artifacts/visual_checks/mouth_crops/`.

- `artifacts/visual_checks/mouth_crops/face_mesh__no_yawn__1-MaleGlasses__00002566.jpg`
- `artifacts/visual_checks/mouth_crops/face_mesh__no_yawn__3-MaleGlasses__00000632.jpg`
- `artifacts/visual_checks/mouth_crops/face_mesh__no_yawn__3-MaleGlasses__00001564.jpg`
- `artifacts/visual_checks/mouth_crops/face_mesh__no_yawn__9-MaleNoGlasses__00001719.jpg`
- `artifacts/visual_checks/mouth_crops/face_mesh__yawn__1-FemaleNoGlasses__00001661.jpg`
- `artifacts/visual_checks/mouth_crops/face_mesh__yawn__10-MaleGlasses__00000871.jpg`
- `artifacts/visual_checks/mouth_crops/face_mesh__yawn__1-MaleGlasses__00002346.jpg`
- `artifacts/visual_checks/mouth_crops/face_mesh__yawn__4-MaleNoGlasses__00000938.jpg`

## Readiness for subject-level splitting

The Stage 5 output is ready for subject-level train/val/test splitting when ALL of the following hold:

1. Success rate >= 95%.
2. No subject has 0 usable crops (`face_mesh + fallback > 0`).
3. Both classes (`no_yawn` and `yawn`) are non-empty.
4. No missing/unreadable files in the file-integrity checks.

### Verdict

**READY** - the mouth-crop dataset passes all readiness checks and can be used as input to subject-level splitting (Stage 6).

