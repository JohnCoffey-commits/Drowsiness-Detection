# NTHUDDD2 Kaggle Extracted-Frame Dataset Report

Generated for the local preprocessing/data-preparation branch for NTHUDDD2 Kaggle JPG frames.

## Dataset Scope

- Dataset path: `dataset/NTHUDDD2/train_data/`
- Expected class folders found:
  - `dataset/NTHUDDD2/train_data/drowsy/`
  - `dataset/NTHUDDD2/train_data/notdrowsy/`
- This is the Kaggle extracted-frame JPG version of NTHUDDD2, not the official raw-video NTHUDDD2 protocol.
- Task type: binary image classification.
- Labels:
  - `notdrowsy = 0`
  - `drowsy = 1`
- Source dataset tag in manifest: `nthuddd2_kaggle`

No video frame extraction, face cropping, permanent resizing, or local CNN training was performed.

This work should be reported as a Kaggle extracted-frame binary image-classification baseline, not as the official NTHU-DDD video benchmark protocol.

## Filename Parsing

All 66,521 JPG filenames follow a five-token underscore-separated pattern:

`subject_id_condition_state_token_frame_index_label.jpg`

Examples:

| Class folder | Example filenames |
|---|---|
| `drowsy` | `001_glasses_sleepyCombination_1000_drowsy.jpg`, `001_glasses_sleepyCombination_1001_drowsy.jpg`, `001_glasses_sleepyCombination_1002_drowsy.jpg`, `001_glasses_sleepyCombination_1003_drowsy.jpg`, `001_glasses_sleepyCombination_1004_drowsy.jpg` |
| `notdrowsy` | `001_glasses_nonsleepyCombination_0_notdrowsy.jpg`, `001_glasses_nonsleepyCombination_1000_notdrowsy.jpg`, `001_glasses_nonsleepyCombination_1001_notdrowsy.jpg`, `001_glasses_nonsleepyCombination_1002_notdrowsy.jpg`, `001_glasses_nonsleepyCombination_1003_notdrowsy.jpg` |

Parsing audit:

| Check | Result |
|---|---:|
| Files with 5 underscore-separated stem tokens | 66,521 |
| Subject ID parse failures | 0 |
| Frame index parse failures | 0 |
| Parsed subjects | 4 |

The first token is parseable as the subject ID for every image. The second token is parseable as condition, the third token as state token, and the last numeric token as frame index.

## Manifest Outputs

| Output | Description |
|---|---|
| `artifacts/mappings/nthuddd2_kaggle_all_images.csv` | Full image manifest, including unreadable-image flags and dimensions. |
| `artifacts/mappings/nthuddd2_kaggle_all_images_trainable.csv` | Trainable manifest excluding unreadable images. |
| `artifacts/mappings/nthuddd2_kaggle_all_images_trainable_with_split.csv` | Trainable manifest plus subject-level split label. |
| `artifacts/splits/nthuddd2_kaggle_subject_split.csv` | One row per subject with split and class counts. |

Required manifest columns were written:

`image_path`, `relative_path`, `filename`, `label`, `class_id`, `subject_id`, `condition`, `state_token`, `frame_index`, `source_dataset`, `image_ok`, `width`, `height`.

## Image Counts

| Metric | Count |
|---|---:|
| Total JPG images | 66,521 |
| Final trainable images | 66,521 |
| Unreadable images | 0 |
| Parsed subjects | 4 |

Only 4 subjects were parsed from the Kaggle extracted-frame dataset: `001`, `002`, `005`, and `006`.

Class counts:

| Label | Class ID | Images |
|---|---:|---:|
| `drowsy` | 1 | 36,030 |
| `notdrowsy` | 0 | 30,491 |

## Subject Summary

Image count per subject:

| Subject ID | Images |
|---|---:|
| `001` | 19,016 |
| `002` | 18,833 |
| `005` | 21,933 |
| `006` | 6,739 |

Class count per subject:

| Subject ID | Drowsy | Not drowsy | Total |
|---|---:|---:|---:|
| `001` | 9,584 | 9,432 | 19,016 |
| `002` | 10,596 | 8,237 | 18,833 |
| `005` | 13,087 | 8,846 | 21,933 |
| `006` | 2,763 | 3,976 | 6,739 |

## Parsed Metadata

Parsed conditions:

| Condition | Images |
|---|---:|
| `glasses` | 37,050 |
| `noglasses` | 29,471 |

Parsed state tokens:

| State token | Images |
|---|---:|
| `nonsleepyCombination` | 20,918 |
| `sleepyCombination` | 19,958 |
| `slowBlinkWithNodding` | 13,147 |
| `yawning` | 12,498 |

## Image Size Summary

All readable images are 640 x 480.

| Statistic | Width | Height |
|---|---:|---:|
| Count | 66,521 | 66,521 |
| Mean | 640 | 480 |
| Min | 640 | 480 |
| 25% | 640 | 480 |
| 50% | 640 | 480 |
| 75% | 640 | 480 |
| Max | 640 | 480 |

Dynamic resizing should be handled later inside the training pipeline or dataloader. The original JPG files were not resized or overwritten.

## Subject-Level Split

The split was created by `subject_id`, not individual images. This subject-level split prevents identity leakage by ensuring no subject appears in more than one split.

Only 4 subject groups are available, so exact 70/15/15 image-level ratios are not possible while keeping subjects grouped. The selected split is the closest practical grouped split that keeps every split non-empty and includes both classes.

Validation and test each contain only one subject, so evaluation has limited subject diversity and should be interpreted cautiously.

Subject split assignment:

| Subject ID | Split | Images | Drowsy | Not drowsy |
|---|---|---:|---:|---:|
| `001` | `train` | 19,016 | 9,584 | 9,432 |
| `005` | `train` | 21,933 | 13,087 | 8,846 |
| `002` | `val` | 18,833 | 10,596 | 8,237 |
| `006` | `test` | 6,739 | 2,763 | 3,976 |

Split summary:

| Split | Subjects | Images | Drowsy | Not drowsy |
|---|---:|---:|---:|---:|
| `train` | 2 | 40,949 | 22,671 | 18,278 |
| `val` | 1 | 18,833 | 10,596 | 8,237 |
| `test` | 1 | 6,739 | 2,763 | 3,976 |

Leakage check result: passed. No subject appears in more than one split.

## LOSO Recommendation

A leave-one-subject-out split file was added at:

`artifacts/splits/nthuddd2_kaggle_loso_folds.csv`

Each LOSO fold assigns exactly one subject to `test`; the remaining subjects are assigned to `train` and `val` deterministically with zero subject leakage within each fold. With only four parsed subjects, LOSO evaluation is recommended as a follow-up sensitivity check alongside the fixed subject split. It gives every subject one turn as the held-out test subject, but it still remains a Kaggle extracted-frame binary image-classification baseline rather than the official raw-video NTHU-DDD benchmark protocol.

## Visual Spot-Check

Contact sheets were generated without modifying original images:

- `artifacts/visual_checks/nthuddd2_kaggle_drowsy_contact_sheet.jpg`
- `artifacts/visual_checks/nthuddd2_kaggle_notdrowsy_contact_sheet.jpg`

The sheets sample images with a fixed seed and display filename, label, subject ID, and split.

## Limitations

- This dataset is an extracted-frame Kaggle version, not the original NTHUDDD2 raw videos.
- Temporal order may be incomplete.
- Full multi-task head, mouth, and eye labels are not guaranteed.
- The current baseline should be reported as single-frame binary drowsiness image classification.
- Only four parsed subject IDs are present locally, so subject-level validation and test splits are subject-limited and image ratios are coarse.
- Exact 70/15/15 image-level split ratios are not possible under subject-level grouping because there are only four subject groups.
- Validation and test each contain only one subject in the fixed split, limiting subject diversity during evaluation.

## Recommended Next Step

Train ResNet18, MobileNetV2, and EfficientNet-B0 in a new Colab notebook named:

`stage8_nthuddd2_kaggle_training.ipynb`
