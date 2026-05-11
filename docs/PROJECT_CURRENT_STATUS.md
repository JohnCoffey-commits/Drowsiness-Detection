# Project Current Status and Experimental Summary

## 1. Project Goal

The project goal is to build a driver drowsiness detection and monitoring system using deep learning. The current design is modular: each model specializes in one visible driver behavior, and the current Stage 17 system combines the specialist outputs with rule-based temporal fusion for uploaded-video warning-candidate analysis.

Current specialist modules:

- YawDD/YawDD+ Dash mouth/yawn module -> `p_yawn`
- MRL Eye open/closed module -> `p_eye_closed`
- Stage 17 rule-based fusion -> warning-candidate states for uploaded videos

The results in this document are specialist-module metrics and rule-based warning-candidate analysis outputs. They should not be reported as final system-level driver drowsiness accuracy.

## 2. Current System Design

| Module | Input type | Specialist task | Labels | Output concept | Current state |
| --- | --- | --- | --- | --- | --- |
| YawDD/YawDD+ Dash mouth/yawn | Mouth crops from reconstructed Dash video frames | No-yawn vs yawn | `0 = no_yawn`, `1 = yawn` | `p_yawn` | Completed |
| MRL Eye | Eye crop images | Closed vs open | `0 = closed`, `1 = open` | `p_eye_closed` | Completed through Stage 9B |
| Stage 10-15 runtime/fusion pipeline | Sampled full-face video frames | Runtime eye ROI, mouth ROI, temporal signals, and rule-based fusion | Not trained labels | Warning-candidate timeline | Completed as controlled-validation prototype |
| Stage 17 video-upload MVP | Uploaded short video | Rule-based warning-candidate analysis with keyframes and technical evidence | Not final truth labels | Upload-session summary, intervals, figures, keyframes | Completed through Stage 17.4 local acceptance/demo documentation |

The current Stage 17 output is a rule-based drowsiness warning-candidate analysis for uploaded videos. It is not final system-level drowsiness accuracy, not deployment readiness, not a trained fusion classifier, and not webcam monitoring. Learned fusion, final fatigue scoring, and real-time webcam feasibility remain future work.

## 3. Dataset Strategy

The final direction uses complementary specialist datasets:

| Dataset | Used for | Reason |
| --- | --- | --- |
| YawDD Dash + YawDD+ annotations | Mouth/yawn specialist | Provides driver Dash videos and frame-level yawn labels. |
| MRL Eye | Eye open/closed specialist | Provides eye-state labels and subject folders suitable for subject-level splitting. |
| NTHUDDD2 official | Considered but not used | Official access required additional institutional/laboratory approval. |
| NTHUDDD2 Kaggle extracted frames | Explored only | Random image-level splitting can be misleading; subject-level evaluation was limited by only four parsed subjects and weak cross-subject generalization risk. |

Subject-level splitting is used for the current specialist datasets to reduce identity/frame leakage.

## 4. YawDD/YawDD+ Dash Mouth/Yawn Module

### 4.1 Dataset Source and Reconstruction

Source materials:

- Original YawDD Dash videos under `dataset/YawDD_raw/`
- YawDD+ annotation files under `dataset/YawDD+/`

Frames were reconstructed from the original YawDD Dash videos using YawDD+ annotation frame indices. Reconstruction outputs are under `dataset/YawDD_plus_reconstructed/`.

Relevant source reports:

- `reports/yawdd_raw_dash_report.md`
- `reports/yawdd_plus_annotation_format_report.md`
- `reports/yawdd_dash_reconstruction_report.md`
- `reports/yawdd_dash_visual_sanity_check.md`

Reconstruction summary from `reports/yawdd_dash_reconstruction_report.md`:

| Class | Count |
| --- | ---: |
| `no_yawn` | 57,347 |
| `yawn` | 7,031 |

### 4.2 Annotation Interpretation

YawDD+ annotation files provided frame labels and frame indices. Visual sanity checks confirmed:

- Class `0` corresponds to non-yawning frames.
- Class `1` corresponds to yawning frames.

Class mapping:

| Class ID | Label |
| ---: | --- |
| 0 | `no_yawn` |
| 1 | `yawn` |

The original YawDD+ bounding boxes were not used as final training mouth crops because visual checks showed that they were not reliable mouth ROIs. They often covered larger face regions and did not consistently isolate the mouth.

### 4.3 Mouth ROI Generation

Mouth crops were generated using MediaPipe Face Mesh lip landmarks, with a lower-face fallback crop when landmarks failed.

Source report: `reports/yawdd_dash_mouth_crop_report.md`

Processing summary:

| Metric | Value |
| --- | ---: |
| Total frames processed | 64,378 |
| MediaPipe Face Mesh crops | 64,093 |
| Fallback lower-face crops | 109 |
| Failed crops | 176 |
| Saved trainable crops | 64,202 |
| Success rate | 99.73% |

Saved crop class distribution:

| Class | Count |
| --- | ---: |
| `no_yawn` | 57,171 |
| `yawn` | 7,031 |

The Stage 5 mouth-crop report verdict was `READY`.

### 4.4 Subject-Level Split

Source report: `reports/yawdd_dash_split_report.md`

The YawDD mouth/yawn split is subject-level, not random image-level. This prevents the same subject from appearing across train, validation, and test splits.

| Split | Subjects | Images | `no_yawn` | `yawn` | Yawn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 20 | 44,156 | 39,345 | 4,811 | 10.90% |
| val | 4 | 8,892 | 7,902 | 990 | 11.13% |
| test | 5 | 11,154 | 9,924 | 1,230 | 11.03% |

Leakage checks passed:

- No subject appears in more than one split.
- Every split contains both classes.
- All referenced mouth-crop files exist.

### 4.5 Stage 7 Training Setup

Stage 7 trained three CNN baselines:

- ResNet18
- MobileNetV2
- EfficientNet-B0

Training settings documented in `README_stage7_training.md` and `colab_file/stage7_yawdd_training_r.ipynb`:

| Setting | Value |
| --- | --- |
| Framework | PyTorch / torchvision |
| Input | Mouth crops from `artifacts/splits/yawdd_dash_subject_split.csv` |
| Labels | `no_yawn`, `yawn` |
| Image size | 224 x 224 |
| Optimizer | Adam |
| Learning rate | `1e-4` |
| Loss | Weighted cross entropy |
| Batch size | 32, with practical fallback to 16 |
| Epochs | 12 |
| Early stopping patience | 3 |
| Scheduler | ReduceLROnPlateau |
| Augmentation | Mild rotation, brightness/contrast jitter, slight affine scaling |
| Transfer learning | Freeze backbone first, then fine-tune full model |

### 4.6 Stage 7 Results

Source of truth for completed local results: `colab_file/stage7_yawdd_training_r.ipynb`.

Important note: `artifacts/results/initial_results.csv` currently appears stale and reports `not_run`; do not use it as the final Stage 7 result source unless it is refreshed from the completed Colab output.

| CNN Architecture | Train Accuracy | Validation Accuracy | Test Accuracy | Yawn Precision | Yawn Recall | Yawn F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CNN-1: ResNet18 | 98.92% | 98.85% | 99.37% | 96.47% | 97.89% | 97.18% |
| CNN-2: MobileNetV2 | 98.97% | 98.48% | 98.75% | 91.74% | 97.48% | 94.52% |
| CNN-3: EfficientNet-B0 | 98.76% | 99.08% | 99.20% | 94.82% | 98.13% | 96.44% |

ResNet18 achieved the strongest Stage 7 test accuracy in the completed run. EfficientNet-B0 had the strongest validation accuracy.

### 4.7 Current YawDD Module Status

The YawDD/YawDD+ Dash mouth/yawn module is completed and should be treated as the stable mouth/yawn specialist. It should not be retrained or modified unless a later project decision explicitly changes the module.

## 5. NTHUDDD2 Branch Status

NTHUDDD2 is no longer the main system direction.

Summary:

- The official NTHU dataset was considered but could not be obtained within the project timeframe because access required institutional/laboratory approval.
- The Kaggle extracted-frame NTHUDDD2 version was explored under `dataset/NTHUDDD2/`.
- The Kaggle version contains 66,521 JPG frames and only four parsed subjects: `001`, `002`, `005`, and `006`.
- Random image-level splitting was considered misleading because visually similar or adjacent frames can appear across splits.
- Subject-level evaluation is more appropriate but is limited by the small number of parsed subjects and weak cross-subject generalization risk.
- MRL Eye replaced NTHUDDD2 as the eye open/closed specialist because it complements the YawDD mouth/yawn module more directly.

Source report: `reports/nthuddd2_kaggle_dataset_report.md`.

## 6. MRL Eye Open/Closed Module

### 6.1 Dataset Source

MRL Eye is used as an eye-state specialist dataset.

Local dataset root:

```text
dataset/mrlEyes_2018_01/
```

Expected structure includes `annotation.txt`, `stats_2018_01.ods`, and subject folders such as `s0001/` through `s0037/`.

Label mapping:

| Label | Meaning |
| ---: | --- |
| 0 | `closed` |
| 1 | `open` |

### 6.2 Stage 8 Dataset Preparation

Source report: `reports/mrl_eye_dataset_report.md`

Stage 8 confirmed:

| Metric | Value |
| --- | ---: |
| Total images | 84,898 |
| Trainable images | 84,898 |
| Subjects | 37 |
| Closed images | 41,946 |
| Open images | 42,952 |
| Unreadable images | 0 |
| Unparseable filenames | 0 |

The annotation check confirmed `0 = closed` and `1 = open`.

Important Stage 8 outputs:

- `artifacts/mappings/mrl_eye_all_images.csv`
- `artifacts/mappings/mrl_eye_trainable.csv`
- `artifacts/mappings/mrl_eye_trainable_with_split.csv`
- `artifacts/splits/mrl_eye_subject_split.csv`
- `reports/mrl_eye_dataset_report.md`
- `reports/mrl_eye_split_report.md`
- `artifacts/visual_checks/mrl_eye_closed_contact_sheet.jpg`
- `artifacts/visual_checks/mrl_eye_open_contact_sheet.jpg`
- `artifacts/visual_checks/mrl_eye_by_split_contact_sheet.jpg`

### 6.3 Subject-Level Split

Source report: `reports/mrl_eye_split_report.md`

The MRL Eye split is subject-level, not random image-level.

| Split | Subjects | Images | Closed | Open |
| --- | ---: | ---: | ---: | ---: |
| train | 25 | 58,982 | 29,310 | 29,672 |
| val | 6 | 13,029 | 6,333 | 6,696 |
| test | 6 | 12,887 | 6,303 | 6,584 |

Checks passed:

- Leakage check result: `True`
- Missing split label check result: `True`
- Every image receives exactly one split: `True`
- Every split contains closed and open: `True`
- Missing file check result: `True`

### 6.4 Stage 9 Training Setup

Source plan: `reports/mrl_eye_stage9_training_plan.md`

Stage 9 trained:

- ResNet18
- MobileNetV2
- EfficientNet-B0

Training setup:

| Setting | Value |
| --- | --- |
| Framework | PyTorch / torchvision |
| Input manifest | `artifacts/mappings/mrl_eye_trainable_with_split.csv` |
| Image size | 224 |
| Batch size | 64 |
| Epochs | 10 |
| Freeze epochs | 1 |
| Early stopping patience | 3 |
| Learning rate | `1e-4` |
| Loss | Weighted cross entropy from training split |
| Scheduler | ReduceLROnPlateau |
| Checkpoint metric | Validation macro F1 |
| Pretrained weights | Loaded for all three models |
| Mixed precision | Enabled when CUDA was available |

Stage 9 reports accuracy, macro precision/recall/F1, weighted F1, per-class metrics, confusion matrices, false-open counts, false-closed counts, and threshold sweeps for `p_eye_closed`.

### 6.5 Stage 9 Results

Source file: `outputs/mrl_eye/results/mrl_eye_initial_results.csv`

| Model | Train Accuracy | Validation Accuracy | Test Accuracy | Test Macro F1 | Test Closed Recall | False Open | False Closed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet18 | 99.16% | 98.37% | 98.46% | 98.46% | 98.59% | 89 | 109 |
| MobileNetV2 | 99.33% | 97.91% | 98.63% | 98.63% | 98.52% | 93 | 84 |
| EfficientNet-B0 | 99.44% | 97.91% | 98.62% | 98.62% | 98.24% | 111 | 67 |

Definitions:

- `false_open`: ground truth closed, predicted open. This is safety-critical because the eye module misses a closed-eye frame.
- `false_closed`: ground truth open, predicted closed. This is a false alarm tendency.

### 6.6 Stage 9B Error Analysis and Model Selection

Source files:

- `reports/mrl_eye_stage9b_error_analysis.md`
- `outputs/mrl_eye/results/mrl_eye_stage9b_model_selection.json`
- `outputs/mrl_eye/results/mrl_eye_stage9b_model_comparison.csv`

Stage 9B selected:

| Item | Selection |
| --- | --- |
| Primary selected model | MobileNetV2 |
| Recommended default threshold | argmax / `p_eye_closed >= 0.50` |
| Safety-prioritized reference | ResNet18 with validation-selected threshold `0.30` |
| Stage 10 readiness status recorded in selection JSON | `READY` |

Threshold summary:

| Model | Val-selected threshold | Test Macro F1 at 0.50 | Closed Recall at 0.50 | False Open at 0.50 | False Closed at 0.50 | Test Macro F1 at selected threshold | Closed Recall at selected threshold | False Open at selected threshold | False Closed at selected threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet18 | 0.30 | 98.46% | 98.59% | 89 | 109 | 97.60% | 99.08% | 58 | 251 |
| MobileNetV2 | 0.30 | 98.63% | 98.52% | 93 | 84 | 98.48% | 98.79% | 76 | 120 |
| EfficientNet-B0 | 0.30 | 98.62% | 98.24% | 111 | 67 | 98.52% | 98.65% | 85 | 106 |

Interpretation:

- MobileNetV2 is preferred as the primary model because it has the best overall default test accuracy/macro F1 and is lightweight.
- ResNet18 at threshold `0.30` is a conservative safety reference because it reduces false-open errors and improves closed-eye recall.
- The ResNet18 `0.30` threshold also increases false-closed errors substantially, so it is not the default setting.

### 6.7 Current MRL Eye Module Status

The MRL Eye open/closed module is completed through Stage 9B. Full local artifacts are present under `outputs/mrl_eye/`, including results, reports, figures, error-analysis contact sheets, and checkpoints.

Selected runtime checkpoint:

```text
outputs/mrl_eye/checkpoints/best_mobilenet_v2_mrl_eye.pt
```

### 6.8 Stage 10 Runtime Eye ROI Consistency

Stage 10 implemented a Python runtime eye ROI consistency test in `src/runtime/stage10_eye_roi_consistency.py`. The dedicated `.venv-stage10` environment was validated with Python 3.12.11 and the required runtime dependencies.

Controlled video `IMG_4901.mp4` was processed successfully from a normal macOS Terminal using `.venv-stage10`. The earlier Codex/sandbox run failed during MediaPipe FaceLandmarker graphics/Metal service initialization, but the manual Terminal smoke and controlled runs succeeded.

Successful controlled run summary:

| Metric | Value |
| --- | ---: |
| Sampled frames | 119 |
| Successful eye crops | 238 |
| Failure count | 0 |
| No-face count | 0 |
| Invalid-crop count | 0 |
| Inference-failed count | 0 |
| Device | `mps` |
| Model | `mobilenet_v2` |

The user manually inspected the contact sheets and debug frames and reported that the eye ROIs were basically accurate. Stage 10 is accepted for this controlled video only. This is still not final drowsiness accuracy.

Historical next work from this point was Stage 11 temporal smoothing / PERCLOS-like eye signal analysis; that work is now completed and has been followed by Stage 12-17 runtime/fusion/upload work.

### 6.9 Stage 11 Eye-Only Temporal Analysis

Stage 11 implemented eye-only temporal smoothing / PERCLOS-like signal analysis in `src/runtime/stage11_eye_temporal_analysis.py`. It uses the successful Stage 10 controlled-video predictions as input and does not train models or perform mouth-eye fusion.

Stage 11 output summary for `IMG_4901.mp4`:

| Metric | Value |
| --- | ---: |
| Frame count | 119 |
| Prediction rows | 238 |
| Closed threshold | 0.50 |
| Rolling window | 5 sampled frames |
| Mean of frame-level `mean_p_eye_closed` | 0.467611 |
| `mean_closed_binary` frames | 67 |
| `either_eye_closed_binary` frames | 71 |
| `both_eyes_closed_binary` frames | 67 |
| Candidate eye-closure events | 2 |

Stage 11 is completed for this controlled video as eye-only temporal analysis. It is not final drowsiness accuracy, not final fatigue scoring, and not mouth/yawn fusion by itself. The subsequent Stage 12-17 work now provides rule-based warning-candidate fusion and uploaded-video review.

### 6.10 Stage 10/11 Multi-Video Validation

Multi-video Stage 10/11 validation was run on four newly recorded controlled-realistic videos:

- `A_normal_open_baseline.mp4`
- `B_realistic_drowsy_simulation.mp4`
- `C_mild_head_motion.mp4`
- `D_controlled_long_open_closed.mp4`

All four videos completed Stage 10 runtime ROI processing and Stage 11 eye-only temporal analysis. A/B/D had 0 Stage 10 failure rows. C mild head motion completed but produced 6 no-face failure rows, so head-motion robustness should be reviewed before treating the runtime signal as stable.

Summary artifacts:

- `outputs/stage11_multi_video_validation_summary.csv`
- `outputs/stage11_multi_video_validation_summary.json`
- `reports/stage11_multi_video_temporal_validation_report.md`
- `docs/STAGE10_11_MULTI_VIDEO_VALIDATION_LOG.md`

This remains runtime eye ROI and eye-only temporal behavior validation only. It is not final drowsiness accuracy and not mouth/yawn fusion by itself. The A/B/C/D findings were later used by Stage 12 eye-only rule design and Stage 15 synchronized rule-based fusion.

### 6.11 Stage 12 Eye-Only Alert Rule Design

Stage 12 implemented eye-only alert rule comparison in `src/runtime/stage12_eye_alert_rule_analysis.py`. It compared rolling probability, PERCLOS-like rolling ratios, event-duration rules, and quality-gated PERCLOS-like rules across the A/B/C/D validation videos.

Recommended conservative eye-only rule:

```text
quality_gated_perclos_mean_ge_0.60_consec
```

Rule parameters:

- Use `rolling_perclos_mean_binary >= 0.60`.
- Require at least 2 consecutive sampled frames.
- Mark recent no-face ratio `> 0.20` over a 5-sampled-frame window as `signal_unreliable`.
- Treat no-face / tracking failure as signal quality, not drowsiness.

Stage 12 result summary:

| Check | Result |
| --- | --- |
| A normal-open short false event suppressed | True |
| B realistic drowsy simulation produced warning candidates | True |
| C mixed fatigue/head-motion/occlusion no-face rows handled as signal quality issues | True |
| D controlled long closure produced warning candidates | True |


Human review note for `C_mild_head_motion`: the user clarified that C is a mixed fatigue-like eye closure, mild head motion, and partial occlusion scenario, not a pure normal-open robustness negative. Short alert markers are plausible during visible simulated eye closures, while no-face/tracking failures should remain signal quality issues rather than drowsiness evidence.

Stage 12 is eye-only alert rule design only. It is not final system-level drowsiness accuracy, not mouth/yawn fusion, and not deployment readiness.

### 6.12 Stage 13 Mouth-Eye Fusion Design

Stage 13 implemented a rule-based mouth-eye fusion design/prototype in `src/runtime/stage13_mouth_eye_fusion_design.py`.

The Stage 13 mouth/yawn runtime audit did not find real synchronized `p_yawn` timelines for the A/B/C/D videos. Therefore, the current Stage 13 run uses synthetic mouth timelines for design demonstration only. It is not validated synchronized runtime fusion.

Recommended fusion rule:

```text
F5_tiered_quality_aware_fusion
```

Rule behavior:

- Preserve `signal_unreliable` when the eye signal is unreliable and no recent yawn exists.
- Emit `mouth_warning_candidate` when a recent yawn exists but the eye signal is unreliable.
- Emit `high_confidence_drowsiness_candidate` only when an eye warning candidate and recent yawn co-occur.
- Otherwise allow eye-only or mouth-only warning candidates, or return `normal`.

Stage 13 artifacts:

- `docs/STAGE13_MOUTH_EYE_FUSION_DESIGN.md`
- `artifacts/audits/stage13_mouth_eye_fusion_design_2026-05-09/stage13_mouth_runtime_audit.md`
- `outputs/stage13_mouth_eye_fusion_design/`
- `reports/stage13_mouth_eye_fusion_design_report.md`

Stage 13 manual annotation sanity check:

- `B_realistic_drowsy_simulation.mp4` has a user-confirmed yawning interval from `14.3s` to `16.8s`.
- A manual mouth timeline was generated at `artifacts/audits/stage13_manual_B_yawn_annotation_2026-05-09/manual_mouth_timeline_B_yawn_14p3_16p8.csv`.
- Stage 13 was rerun with this manual annotation in `outputs/stage13_mouth_eye_fusion_manual_B_yawn_annotation/`.
- This uses manual mouth annotation, not runtime mouth model output.
- True synchronized fusion validation still requires runtime mouth/yawn inference to produce automatic `p_yawn` timelines.

Stage 13 is design/prototype work only. It is not final system-level drowsiness accuracy, not a trained fusion classifier, and not deployment readiness. The missing real runtime mouth/yawn inference was later addressed in Stage 14, and synchronized rule-based fusion was validated in Stage 15.

### 6.13 Stage 14 Mouth/Yawn Runtime Inference

Stage 14 was unblocked after recovering the completed Stage 7 ResNet18 mouth/yawn checkpoint from Google Drive and copying it to:

```text
checkpoints/resnet18_best.pt
```

Checkpoint verification passed:

- Architecture: ResNet18 with a two-class classifier head.
- Label mapping: `0 = no_yawn`, `1 = yawn`.
- `p_yawn = softmax(logits)[1]`.
- Evaluation preprocessing: RGB image, resize to `224 x 224`, `ToTensor`, and ImageNet normalization.

Stage 14 implemented runtime mouth ROI extraction and mouth/yawn inference in `src/runtime/stage14_mouth_yawn_runtime.py`. It processed A/B/C/D controlled-realistic videos using MediaPipe FaceLandmarker mouth/lip landmarks and the recovered ResNet18 checkpoint.

Stage 14 result summary:

| Video | Successful mouth crops | Failures | Yawn events | Notes |
| --- | ---: | ---: | ---: | --- |
| `A_normal_open_baseline` | 70 | 0 | 0 | Baseline stayed below yawn threshold. |
| `B_realistic_drowsy_simulation` | 103 | 0 | 14 | `p_yawn` rose strongly around the manually confirmed 14.3s-16.8s yawn interval. |
| `C_mild_head_motion` | 89 | 6 | 0 | Six no-face rows; no yawn fabricated from occlusion/head motion. |
| `D_controlled_long_open_closed` | 119 | 0 | 0 | No yawn events detected. |

For `B_realistic_drowsy_simulation`, the sampled 14.3s-16.8s interval produced 12/12 yawn-event rows, with mean `p_yawn` approximately `0.981`.

Stage 14 artifacts:

- `artifacts/audits/stage14_mouth_yawn_runtime_2026-05-09/stage14_mouth_model_audit.md`
- `artifacts/audits/stage14_mouth_yawn_runtime_2026-05-09/STAGE14_CHECKPOINT_LOCAL_COPY.md`
- `artifacts/audits/stage14_mouth_yawn_runtime_2026-05-09/STAGE14_RECOVERED_CHECKPOINT_VERIFICATION.md`
- `outputs/stage14_mouth_yawn_runtime_multi_video_summary.csv`
- `outputs/stage14_mouth_yawn_runtime_multi_video_summary.json`
- `reports/stage14_mouth_yawn_runtime_validation_report.md`
- `docs/STAGE14_MOUTH_YAWN_RUNTIME_LOG.md`

Stage 14 is runtime mouth/yawn validation only. It is not final system-level drowsiness accuracy and not fusion by itself. Human visual inspection of mouth contact sheets and debug frames remains useful for ROI quality review. Stage 15 later used these `p_yawn` timelines for real synchronized rule-based mouth-eye fusion validation.

### 6.14 Stage 15 Real Synchronized Rule-Based Mouth-Eye Fusion

Stage 15 completed the first real synchronized rule-based mouth-eye fusion validation using:

- Stage 12 real eye alert timelines.
- Stage 14 real model-generated `p_yawn` timelines.

Stage 15 did not use synthetic mouth timelines and did not use manual mouth annotation timelines for fusion decisions. The validated fusion rule was the Stage 13 recommended `F5_tiered_quality_aware_fusion` rule.

Stage 15 result summary:

| Video | Normal frames | Eye-warning frames | Mouth-warning frames | High-confidence candidate frames | Signal-unreliable frames | Scenario match |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `A_normal_open_baseline` | 70 | 0 | 0 | 0 | 0 | Yes |
| `B_realistic_drowsy_simulation` | 49 | 18 | 30 | 6 | 0 | Yes |
| `C_mild_head_motion` | 76 | 7 | 0 | 0 | 12 | Yes |
| `D_controlled_long_open_closed` | 54 | 65 | 0 | 0 | 0 | Yes |

For `B_realistic_drowsy_simulation`, Stage 14 model output produced 12/12 yawn-event rows in the manually observed 14.3s-16.8s yawn interval, with mean `p_yawn` approximately `0.981`. Stage 15 produced high-confidence candidate frames when recent-yawn evidence overlapped with eye-warning evidence.

Stage 15 artifacts:

- `outputs/stage15_real_mouth_eye_fusion/combined_stage14_real_mouth_timeline.csv`
- `outputs/stage15_real_mouth_eye_fusion/stage15_real_fusion_rule_comparison.csv`
- `outputs/stage15_real_mouth_eye_fusion/stage15_real_fusion_summary.json`
- `outputs/stage15_real_mouth_eye_fusion/STAGE15_REAL_MOUTH_EYE_FUSION_REPORT.md`
- `reports/stage15_real_mouth_eye_fusion_validation_report.md`
- `docs/STAGE15_REAL_MOUTH_EYE_FUSION_LOG.md`

Stage 15 is not final system-level drowsiness accuracy, not deployment readiness, and not a trained fusion classifier. It validates rule behavior on a small controlled-realistic set. Stage 16 integration summary and Stage 17 uploaded-video warning-candidate MVP now build on this evidence, with any learned fusion still deferred until synchronized annotated data exists.

### 6.15 Stage 16 Final Integration Summary and Evidence Package

Stage 16 completed final documentation, evidence packaging, and demo planning for the controlled-validation prototype.

Stage 16 also fixed the Stage 15 figure-title issue: Stage 15 figures now use Stage 15-specific titles instead of the reused Stage 13 wording. No fusion logic or numeric outputs were changed by the title fix.

Stage 16 artifacts:

- `reports/stage16_final_integration_summary_report.md`
- `docs/STAGE16_FINAL_EVIDENCE_PACKAGE.md`
- `docs/STAGE16_DEMO_AND_PRESENTATION_OUTLINE.md`
- `docs/PROJECT_FINAL_STATUS_STAGE16.md`
- `artifacts/audits/stage16_final_integration_2026-05-09/STAGE15_FIGURE_TITLE_FIX.md`
- `artifacts/audits/stage16_final_integration_2026-05-09/final_repo_artifact_audit.md`

Stage 16 status is now superseded by Stage 17.4. The Stage 16 evidence package remains useful historical integration evidence, but the current project state is the Stage 17.4 local video-upload warning-candidate MVP with acceptance and demo documentation.

### 6.16 Stage 17 Video Upload Warning-Candidate MVP

Stage 17 implemented and stabilized a video-upload warning-candidate MVP.

The MVP lets a user upload a short video, run the existing eye-mouth rule-based warning-candidate pipeline, view summary results, inspect interval-level evidence, review figures/keyframes, and access technical evidence files through safe backend URLs.

Stage 17 implementation files:

- `src/runtime/system_video_upload_pipeline.py`
- `src/runtime/keyframe_extractor.py`
- `src/backend/app.py`
- `src/backend/static/upload_test.html`
- `SystemUI/src/app/video-upload/page.tsx`
- `SystemUI/src/components/video-upload/`
- `SystemUI/src/lib/videoUploadTypes.ts`
- `SystemUI/src/lib/videoUploadUtils.ts`
- `scripts/start_stage17_ui.sh`
- `Makefile`

The pipeline runs the uploaded video through:

1. Stage 10 eye ROI inference.
2. Stage 11 eye temporal analysis.
3. A Stage 12-style eye alert adapter using `quality_gated_perclos_mean_ge_0.60_consec`.
4. Stage 14 mouth/yawn inference.
5. F5 tiered quality-aware rule-based fusion.
6. Keyframe extraction for warning-candidate intervals.

Stage 17 backend validation on `B_realistic_drowsy_simulation.mp4` passed:

| Metric | Value |
| --- | ---: |
| Total sampled frames | 103 |
| Eye-warning candidate frames | 18 |
| Mouth-warning candidate frames | 30 |
| High-confidence warning candidate frames | 6 |
| Signal-unreliable frames | 0 |
| Yawn-event count | 14 |
| Keyframes extracted | 3 |

Stage 17.1 added a sustained-eye gate for high-confidence escalation:

- High-confidence warning candidates require recent mouth/yawn evidence.
- High-confidence warning candidates also require sustained eye-warning evidence.
- Brief blink-like activity overlapping recent-yawn evidence is suppressed from high-confidence escalation.

Stage 17.2 added interpretation-layer documentation:

- Eye-warning evidence is not automatically described as sustained full eye closure.
- Safer terms include reduced eye openness, blink-like activity, brief eye-closure evidence, fatigue-like eye-warning evidence, and ROI-sensitive cases.
- High-confidence warning candidates remain rule-based candidates, not final drowsiness truth.

Stage 17.3 implemented the dedicated SystemUI page at:

```text
http://127.0.0.1:3000/video-upload
```

The page includes:

- Upload card with backend URL setting.
- Seven-step processing indicator.
- Result header and permanent warning text.
- Summary metric cards.
- `Warning-candidate intervals` table.
- Stage 17.1 / Stage 17.2 interpretation card.
- Fusion timeline, `p_eye_closed`, and `p_yawn` figures.
- Keyframe evidence gallery with metadata.
- Technical evidence links using safe backend API paths.

Stage 17.4 added local acceptance/demo stabilization:

- One-command launcher: `make stage17-ui`
- Launcher script: `scripts/start_stage17_ui.sh`
- Local backend URL: `http://127.0.0.1:8000`
- Local frontend URL: `http://127.0.0.1:3000/video-upload`
- Acceptance checklist: `docs/STAGE17_4_VIDEO_UPLOAD_UI_ACCEPTANCE_CHECKLIST.md`
- Demo script: `docs/STAGE17_4_DEMO_SCRIPT.md`
- Stabilization report: `reports/stage17_4_video_upload_mvp_stabilization_report.md`

Stage 17.3/17.4 manual upload validation on `upload_test/C_upload_test.mp4` produced these expected UI markers:

| Marker | Value |
| --- | ---: |
| High-confidence warning candidate frames | 9 |
| Suppressed brief-eye escalation frames | 8 |
| Keyframes | 4 |
| Figures | 3 |
| Interval table | Present |

Backend upload validation passed using `.venv-stage10` with the required backend dependencies.

Stage 17.4 is the current project status. It remains a local video-upload warning-candidate MVP. It is not final system-level drowsiness accuracy, not deployment readiness, not webcam monitoring, and not a trained fusion classifier.

## 7. Current Best Models

| Module | Primary model | Selection basis | Default decision rule |
| --- | --- | --- | --- |
| YawDD/YawDD+ Dash mouth/yawn | ResNet18 | Highest Stage 7 test accuracy among the three baselines | argmax over `no_yawn` / `yawn` |
| MRL Eye open/closed | MobileNetV2 | Best balance of default test accuracy, macro F1, closed-eye recall, false-open/false-closed trade-off, and real-time suitability | argmax / `p_eye_closed >= 0.50` |

Reference option:

| Module | Reference option | Why it is kept |
| --- | --- | --- |
| MRL Eye open/closed | ResNet18 at `p_eye_closed >= 0.30` | Safety-prioritized setting with higher closed-eye recall and fewer false-open errors, but more false alarms. |

## 8. Important Thresholds and Safety Notes

MRL Eye threshold behavior:

- Default: argmax / `p_eye_closed >= 0.50`
- Safety reference: ResNet18 with `p_eye_closed >= 0.30`

Safety interpretation:

- False-open errors are safety-critical because they miss closed-eye frames.
- False-closed errors are false alarms and can make the system too sensitive.
- Lowering the threshold can improve closed-eye recall but may increase false alarms.
- Threshold candidates were selected from validation sweeps only; test metrics are final reporting values.

YawDD safety interpretation:

- The YawDD mouth/yawn module identifies yawning behavior but does not alone prove drowsiness.
- The mouth/yawn output should be fused later with eye-state probabilities over time.

## 9. Important Files and Artifacts

| File or folder | Purpose |
| --- | --- |
| `docs/PROJECT_STRUCTURE.md` | Repository structure guide. |
| `docs/PROJECT_CURRENT_STATUS.md` | Current status and experimental summary. |
| `reports/yawdd_dash_mouth_crop_report.md` | YawDD mouth ROI preprocessing summary. |
| `reports/yawdd_dash_split_report.md` | YawDD subject-level split summary. |
| `colab_file/stage7_yawdd_training_r.ipynb` | Completed Stage 7 YawDD training run output. |
| `reports/mrl_eye_dataset_report.md` | MRL Eye dataset inspection report. |
| `reports/mrl_eye_split_report.md` | MRL Eye subject split report. |
| `outputs/mrl_eye/results/mrl_eye_initial_results.csv` | Main Stage 9 MRL Eye metrics table. |
| `outputs/mrl_eye/results/mrl_eye_metrics_summary.json` | Stage 9 MRL Eye metrics summary. |
| `reports/mrl_eye_stage9b_error_analysis.md` | Final MRL Eye model-selection report. |
| `reports/stage10_runtime_eye_roi_acceptance_report.md` | Stage 10 controlled-video runtime eye ROI acceptance report. |
| `outputs/mrl_eye/results/mrl_eye_stage9b_model_selection.json` | Machine-readable Stage 9B selection summary. |
| `outputs/mrl_eye/artifact_inventory.md` | Confirms complete local MRL Eye output set. |
| `outputs/mrl_eye/checkpoints/best_mobilenet_v2_mrl_eye.pt` | Selected MRL Eye model checkpoint. Ignored by Git. |
| `outputs/stage10_eye_roi_consistency_IMG_4901_controlled_terminal/` | Successful Stage 10 controlled-video runtime output directory. |
| `src/runtime/system_video_upload_pipeline.py` | Stage 17 uploaded-video analysis pipeline. |
| `src/backend/app.py` | FastAPI backend for upload analysis and constrained session file serving. |
| `SystemUI/src/app/video-upload/page.tsx` | Stage 17.3 Video Upload Analysis route. |
| `SystemUI/src/components/video-upload/` | Stage 17.3 modular UI components for upload, results, intervals, figures, keyframes, and technical evidence. |
| `scripts/start_stage17_ui.sh` | One-command local launcher for backend and frontend. |
| `Makefile` | Project-level command target `make stage17-ui`. |
| `docs/STAGE17_VIDEO_UPLOAD_RESULT_SCHEMA.md` | Stage 17 response, summary, interval, timeline, and keyframe schema. |
| `docs/STAGE17_3_LOCAL_LAUNCH_GUIDE.md` | Local launcher and troubleshooting guide. |
| `docs/STAGE17_3_VIDEO_UPLOAD_UI_PAGE_REPORT.md` | Stage 17.3 UI implementation report. |
| `docs/STAGE17_4_VIDEO_UPLOAD_UI_ACCEPTANCE_CHECKLIST.md` | Manual acceptance checklist for the video-upload MVP. |
| `docs/STAGE17_4_DEMO_SCRIPT.md` | Safe-worded demo script for Stage 17.4. |
| `reports/stage17_4_video_upload_mvp_stabilization_report.md` | Current Stage 17.4 stabilization report. |

## 10. Reporting Notes and Limitations

Use careful wording in reports and presentations:

- Say these are specialist-module results.
- Do not claim final system-level driver drowsiness accuracy.
- YawDD/YawDD+ Dash results are mouth/yawn classification results.
- MRL Eye results are eye open/closed classification results.
- Subject-level split was used for the current YawDD and MRL Eye modules.
- Stage 15 and Stage 17 use rule-based fusion to produce warning-candidate states.
- Learned fusion, final fatigue scoring, final system-level accuracy, and deployment readiness remain future work.
- Stage 17.4 outputs are uploaded-video warning-candidate analysis results, not final drowsiness truth.

Known artifact caveat:

- Final Stage 7 YawDD results are visible in `colab_file/stage7_yawdd_training_r.ipynb`.
- The local `artifacts/results/initial_results.csv` currently contains stale `not_run` values and should not be used for final Stage 7 reporting unless refreshed.

## 11. Maintenance Notes

- Keep `dataset/`, zip files, and checkpoint binaries out of normal Git.
- Metrics CSV/JSON, reports, figures, and error-analysis images may be committed for reproducibility.
- If checkpoints need to be versioned, use Git LFS rather than normal Git.
- When adding new results, cite the exact local artifact used as the source of truth.
- Keep the NTHUDDD2 work documented as an explored branch, not as the active final system direction.
- Keep the Stage 17 launcher, schema, acceptance checklist, demo script, and stabilization report synchronized whenever the upload UI or backend API changes.
- Recommended next stage: Stage 18 real-time webcam warning-candidate feasibility prototype, only after Stage 17.4 acceptance is complete.
