# YawDD+ Dash - Stage 6 Subject-Level Split Report

## Inputs and outputs

- Source mouth-crop manifest: `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/artifacts/mappings/yawdd_dash_all_mouth_crops.csv`
- Trainable manifest: `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/artifacts/mappings/yawdd_dash_all_mouth_crops_trainable.csv`
- Split manifest: `/Users/zhengpeixian/ZPX/UTS/DeepLearning/Group/Drowsiness_Detection/artifacts/splits/yawdd_dash_subject_split.csv`
- Split unit: subject_id (no image-level randomization)
- Split search seed: 42
- Split search iterations: 200000
- Split search score: 0.427063

## Filtering

- Source rows: **64378**
- Excluded rows where `crop_method == failed`: **176**
- Trainable rows: **64202**
- Missing crop files among trainable rows: **0**

## Overall class distribution

| Class | Images | Percent |
|---|---:|---:|
| `no_yawn` | 57171 | 89.05% |
| `yawn` | 7031 | 10.95% |

## Split distribution

| Split | Subjects | Images | Image % | no_yawn | no_yawn % | yawn | yawn % | Yawn rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `train` | 20 | 44156 | 68.78% | 39345 | 89.10% | 4811 | 10.90% | 10.90% |
| `val` | 4 | 8892 | 13.85% | 7902 | 88.87% | 990 | 11.13% | 11.13% |
| `test` | 5 | 11154 | 17.37% | 9924 | 88.97% | 1230 | 11.03% | 11.03% |

## Subject attribute distribution

| Split | Female subjects | Male subjects | Glasses subjects | NoGlasses subjects |
|---|---:|---:|---:|---:|
| `train` | 9 | 11 | 10 | 10 |
| `val` | 2 | 2 | 2 | 2 |
| `test` | 2 | 3 | 3 | 2 |

## Leakage check

- Unique trainable subjects: **29**
- Requested subject counts met: **YES**
- No subject appears in more than one split: **YES**
- No failed crop rows in trainable data: **YES**
- Every split contains both classes: **YES**
- All referenced mouth-crop files exist: **YES**

## Subject assignments

| Split | subject_id | Gender | Glasses | Images | no_yawn | yawn |
|---|---|---|---|---:|---:|---:|
| `train` | `1-FemaleNoGlasses` | Female | NoGlasses | 2741 | 2212 | 529 |
| `train` | `10-FemaleNoGlasses` | Female | NoGlasses | 1206 | 1163 | 43 |
| `train` | `10-MaleGlasses` | Male | Glasses | 2418 | 2137 | 281 |
| `train` | `11-FemaleGlasses` | Female | Glasses | 1683 | 1447 | 236 |
| `train` | `11-MaleGlasses` | Male | Glasses | 1724 | 1520 | 204 |
| `train` | `12-MaleGlasses` | Male | Glasses | 1954 | 1690 | 264 |
| `train` | `13-FemaleGlasses` | Female | Glasses | 2488 | 2393 | 95 |
| `train` | `13-MaleNoGlasses` | Male | NoGlasses | 2140 | 1942 | 198 |
| `train` | `15-MaleGlasses` | Male | Glasses | 2639 | 2431 | 208 |
| `train` | `16-MaleNoGlasses` | Male | NoGlasses | 2503 | 2330 | 173 |
| `train` | `2-MaleGlasses` | Male | Glasses | 2312 | 1959 | 353 |
| `train` | `3-FemaleGlasses` | Female | Glasses | 3057 | 2575 | 482 |
| `train` | `3-MaleGlasses` | Male | Glasses | 2027 | 1841 | 186 |
| `train` | `4-MaleNoGlasses` | Male | NoGlasses | 1998 | 1679 | 319 |
| `train` | `5-FemaleNoGlasses` | Female | NoGlasses | 2149 | 1995 | 154 |
| `train` | `6-FemaleNoGlasses` | Female | NoGlasses | 1477 | 1129 | 348 |
| `train` | `7-FemaleNoGlasses` | Female | NoGlasses | 3618 | 3400 | 218 |
| `train` | `7-MaleGlasses` | Male | Glasses | 2398 | 2088 | 310 |
| `train` | `8-MaleNoGlasses` | Male | NoGlasses | 2092 | 1949 | 143 |
| `train` | `9-FemaleNoGlasses` | Female | NoGlasses | 1532 | 1465 | 67 |
| `val` | `2-FemaleNoGlasses` | Female | NoGlasses | 2180 | 1956 | 224 |
| `val` | `5-MaleGlasses` | Male | Glasses | 2395 | 2270 | 125 |
| `val` | `8-FemaleGlasses` | Female | Glasses | 2297 | 1970 | 327 |
| `val` | `9-MaleNoGlasses` | Male | NoGlasses | 2020 | 1706 | 314 |
| `test` | `1-MaleGlasses` | Male | Glasses | 2576 | 2166 | 410 |
| `test` | `12-FemaleGlasses` | Female | Glasses | 2461 | 2299 | 162 |
| `test` | `14-MaleNoGlasses` | Male | NoGlasses | 3010 | 2682 | 328 |
| `test` | `4-FemaleNoGlasses` | Female | NoGlasses | 1492 | 1400 | 92 |
| `test` | `6-MaleGlasses` | Male | Glasses | 1615 | 1377 | 238 |

## Verdict

**READY** - the train/validation/test split is leakage-safe and ready for Stage 7 CNN training.
