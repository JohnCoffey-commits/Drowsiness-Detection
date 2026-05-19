# Project Current Status and Experimental Summary

Last updated: 2026-05-19

Current snapshot: the repository has moved beyond the historical Stage 16 final integration package, Stage 17.4 uploaded-video MVP stabilization, and Stage 18 frontend-only history page. The current working system state is:

- Stage 17.5 `/video-upload` evidence-review UI polish and interpretation cleanup.
- Stage 18 `/history-48h` frontend-only warning-candidate history page with demo/local `localStorage` data.
- Stage 17 FastAPI uploaded-video backend and rule-based fusion remain the current real backend-connected analysis path.
- Stage 19.7C `/` Live Monitor realtime webcam warning-candidate feasibility prototype now supports a clean product-style webcam frame, automatic 2 FPS sampling from Start Camera, realtime single-frame backend evidence, session-local realtime rule-based warning-candidate state, product-style yawn/eye/critical-eye warning overlays, a face-not-visible signal-quality overlay, default-on sound alerts after the camera user gesture, a Drowsiness Risk card bound to frontend warning-candidate severity state, dashboard widgets bound to stable Live Monitor events/score samples, and a product-style current-session state-derived display severity waveform chart.
- Latest Stage 19 work redesigns the Drowsiness Level chart into a Low/Medium/High waveform using real timestamp-based X-axis positioning, stable realtime warning-candidate states, bounded display-only variation, and no raw frame-level plotting.
- Stage 20A adds a local MVP login gate with the assigned local account, current-user profile menu, manual Day/Night theme toggle, light/day and coherent/night app shell styling, and a local notification center.
- Stage 20B connects stable Live Monitor frontend warning-candidate events into `/history-48h` frontend-local history under `visionguard.history48h.v1`, scoped to the current local MVP user where auth is available.
- Stage 21A separates event history from analytics: `/history-48h` is now the event-level local warning-candidate history and review workstation, while `/insights` is the user-scoped local analytics page derived from the same frontend-local history records.
- Latest May 19 UI polish balances the Live Monitor desktop grid when the sidebar is expanded, keeps selected sidebar labels/icons white, aligns the collapsed sidebar logo and expand control on one row, and removes the extra top explanatory notice block from `/history-48h` while preserving the warning-candidate boundary elsewhere.

This is still a local warning-candidate MVP. The Live Monitor is a realtime webcam warning-candidate feasibility prototype, not browser notification, not backend database history ingestion, not production authentication, not raw image/video storage, not final system-level drowsiness accuracy, not a trained fusion classifier, and not deployment readiness.

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
| Stage 17 video-upload MVP | Uploaded short video | Rule-based warning-candidate analysis with keyframes and technical evidence | Not final truth labels | Upload-session summary, intervals, figures, keyframes | Backend-connected local MVP completed through Stage 17.5 evidence-review UI polish |
| Stage 18/20B/21A 48h History UI | Browser-local demo/local history events plus stable Live Monitor local events | Frontend warning-candidate history review | Frontend-local records, not backend truth labels | Event filters, compact operational summary, review queue, event timeline, event details, recent sessions | Frontend-only page completed at `/history-48h`; Stage 20B adds Live Monitor local ingestion; Stage 21A focuses the page on event review |
| Stage 21A Insights UI | Same frontend-local 48h history records | User-scoped local analytics and review-priority insights | Aggregate local warning-candidate patterns, not backend truth labels | Trend, event composition, time-of-day pattern, session comparison, signal-quality insights, review recommendations | Frontend-only page completed at `/insights`; read-only analytics derived from current user local history |
| Stage 19 Live Monitor | Browser webcam sampled frames | Realtime single-frame evidence, rule-based temporal warning-candidate state, product-style warning overlays, face visibility cue, default-on sound alerts after camera start, dynamic risk gauge binding, dashboard data binding, and current-session state-derived waveform rendering | Not final truth labels | `p_eye_closed`, `p_yawn`, ROI/signal quality, realtime temporal candidate state, product overlays, stable frontend warning-candidate events, frontend warning-candidate severity score samples | Stage 19.7C realtime webcam warning-candidate feasibility prototype |

The current backend-connected Stage 17 output is a rule-based drowsiness warning-candidate analysis for uploaded videos. The `/history-48h` page is frontend-only local history storage and review workflow. The `/insights` page is frontend-only aggregate analytics derived from the same current-user local history records. The Stage 19.7C Live Monitor is a realtime webcam warning-candidate feasibility prototype with visual alert debounce, stable frontend alert events, product-style warning overlays, automatic sampling from Start Camera, default-on sound alerts after the camera user gesture, a dynamic frontend warning-candidate severity gauge, current-drive EYE/YAWN event counts, today-only Recent Events, and a real timestamp-based current-session state-derived display severity waveform chart. Stage 20B now normalizes those stable Live Monitor events into `/history-48h` frontend-local records for the current local MVP user. Stage 21A intentionally avoids duplicating the same charts and event lists between History and Insights. It has no browser notification, no backend database, no production authentication, no raw image/video storage, and no backend user-owned history persistence. None of these pages report final system-level drowsiness accuracy, deployment readiness, or trained fusion. Learned fusion, final fatigue scoring, real backend history storage, alert output beyond the current frontend layer, and authenticated persisted webcam history remain future work.

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

Stage 16 status is now superseded by Stage 17 and Stage 18 frontend work. The Stage 16 evidence package remains useful historical integration evidence, but the current project state is the Stage 17.5 local video-upload warning-candidate review UI plus the Stage 18 frontend-only 48h history page.

### 6.16 Stage 17 Video Upload Warning-Candidate MVP

Stage 17 implemented and stabilized a video-upload warning-candidate MVP. Stage 17.5 then refined both backend-provided eye evidence interpretation and the `/video-upload` UI presentation of warning-candidate evidence.

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
| Yawn warning candidate frames | 30 |
| Critical eye warning candidate frames | 6 |
| Signal quality issue frames | 0 |
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

Stage 17.5 added conservative eye evidence calibration and UI evidence-review cleanup:

- Backend schema supports Stage 17.5 eye evidence fields such as weak/moderate/strong eye evidence, interval eye-strength gate status, and weak-eye suppression counts.
- `/video-upload` keeps missing optional fields from being shown as false.
- Keyframe cards no longer repeat long missing-field fallback text.
- Recent-yawn evidence is explained as temporal-window evidence rather than necessarily a yawn in the exact current frame.
- Interval rows are compact, with long backend reasons moved into expandable details.
- Fusion state and descriptive eye evidence are clearly separated; the UI does not recompute or override backend fusion state.
- Evidence figures are tabbed, with the fusion timeline visible by default.

Stage 17.5 UI validation completed with `npm run lint`, `npm run build`, `/video-upload` browser checks, `/` route checks, and a backend-connected upload of `upload_test/C_upload_test.mp4`.

Stage 17.5 remains a local uploaded-video warning-candidate MVP. It is not final system-level drowsiness accuracy, not deployment readiness, not webcam monitoring, and not a trained fusion classifier.

### 6.17 Stage 18 48h History Frontend Page

Stage 18 added a frontend-only 48-hour history page:

```text
http://127.0.0.1:3000/history-48h
```

The page summarizes recent warning-candidate history using demo/local browser data. It does not add webcam capture, backend history storage, model changes, Python inference changes, or Stage 17 fusion changes.

Stage 18 implementation files:

- `SystemUI/src/app/history-48h/page.tsx`
- `SystemUI/src/components/history-48h/`
- `SystemUI/src/lib/history48hTypes.ts`
- `SystemUI/src/lib/history48hMockData.ts`
- `SystemUI/src/lib/history48hStorage.ts`
- `SystemUI/src/lib/history48hUtils.ts`
- `SystemUI/next.config.ts`
- `docs/STAGE18_HISTORY_48H_UI_PAGE_PLAN.md`

Stage 18 data strategy:

- Uses localStorage key `visionguard.history48h.v1`.
- Seeds realistic demo history across the recent 48 hours on first load.
- Filters out records older than 48 hours.
- Supports reset demo data, clear history, copy summary, local review-state updates, and session filtering.

Stage 18 page sections:

- Header and boundary notice.
- Filters and controls.
- Summary cards.
- Candidate severity trend chart.
- Warning-candidate distribution chart.
- State breakdown chart.
- High-risk warning candidates.
- Event timeline with expandable details.
- Recent sessions.
- Manual review queue.
- Interpretation note.

Stage 21A later split the chart-heavy history review surface into a focused `/history-48h` review page and a separate `/insights` analytics page. The current History page keeps filters, compact summary cards, event review/detail flows, recent sessions, and the interpretation note, while aggregate charts now live under Insights.

The current sidebar intentionally shows only:

1. Live Monitor
2. Video Upload Analysis
3. 48h History
4. Insights

Stage 18 validation completed with `npm run lint`, `npm run build`, `/history-48h` browser checks, localStorage reset/clear persistence checks, filter checks, manual-review local update checks, `/video-upload` route check, and `/` route check.

Stage 18 was a frontend evidence-review/history page only. It was not backend history storage, final system-level drowsiness accuracy, or deployment readiness. Stage 20B later added frontend-local Live Monitor warning-candidate ingestion into browser history without adding backend persistence.

### 6.18 Stage 19 Live Monitor Realtime Webcam Prototype

Stage 19 develops `/` from the former Dashboard concept into the Live Monitor local webcam prototype:

```text
http://127.0.0.1:3000/
```

Current Live Monitor implementation:

- Renames the visible Dashboard navigation and page header to `Live Monitor`.
- Uses real browser webcam capture through `navigator.mediaDevices.getUserMedia`.
- Keeps webcam preview mirrored for frontend display only; backend sampled frames remain unmirrored.
- Starts active webcam sampling automatically at 2 FPS through a hidden canvas and JPEG `Blob` extraction after Start Camera.
- Starts and stops an in-memory realtime backend session with the camera lifecycle.
- Sends sampled frames to `POST /api/realtime/frame`.
- Returns frame-level `p_eye_closed`, `p_yawn`, face/ROI status, signal quality, device, and latency.
- Maintains session-local realtime temporal warning-candidate state.
- Maps backend `fusion_state` into product-style yawn and eye warning overlays after 1.0 second stable-state debounce.
- Escalates critical eye warnings at the UI layer from existing `high_confidence` events, sustained eye-warning fields, or repeated debounced eye-warning events in a recent frontend window.
- Shows a face-not-visible signal-quality overlay when existing face/ROI/signal fields indicate an unreliable camera signal.
- Emits a small frontend risk state to the parent page so the Drowsiness Risk card can reflect the current realtime warning-candidate state without duplicate realtime API calls.
- Updates the Drowsiness Risk card as a frontend warning-candidate severity score covering idle, normal monitoring, yawn warning, eye warning, critical eye warning, and signal-quality states.
- Animates the Drowsiness Risk score and needle smoothly when the target frontend severity score changes.
- Emits stable frontend dashboard events only when the existing debounce/cooldown visual alert layer creates an alert event, with critical-eye escalation counted once as a critical eye warning-candidate event.
- Counts top EYE/YAWN cards from stable warning-candidate events in the current browser-run drive session, not from raw frames, p-values, pending debounce, or display-only reminders.
- Stores lightweight dashboard events and candidate severity score points in `visionguard.liveMonitorDashboard.v1`; no raw image, video, blob, or frame payload is stored.
- Shows Recent Events from real local Live Monitor events for the current local calendar day only and uses the fixed `Today` badge instead of mock count text.
- Keeps older local dashboard records at midnight; the visible Recent Events list changes because the local-date filter changes, not because records are deleted.
- Replaces the mock Drowsiness Level chart with current-session warning-candidate severity points derived from the same stable frontend warning-candidate states used by the Drowsiness Risk card.
- Uses real numeric timestamps for the chart X-axis and keeps formatted `HH:mm` strings only as display labels, avoiding repeated minute-label X-axis collisions.
- Sorts, validates, and filters candidate severity points to the current dynamic session window; corrupted non-finite or out-of-range scores are not plotted.
- Compacts risk points into small timestamp buckets and keeps the latest representative point per bucket so repeated frame-level or near-duplicate updates do not create vertical clusters.
- Appends risk points on material target changes and every 10 seconds while monitoring is active, avoiding per-frame chart writes and avoiding camera-off fake history.
- Adds a separate `displaySeverityScore` for chart rendering only; this state-derived display severity trend uses small deterministic bounded waveform variation for readability and does not alter backend evidence, warning overlays, sound alerts, event counts, or `/history-48h`.
- Renders the chart as a product-style waveform with Low green, Medium orange, and High red colors matching the Drowsiness Risk gauge.
- Applies normal-clear debounce and per-kind cooldown before adding session-local alert events.
- Keeps Stage 19.7C dashboard data in a separate Live Monitor localStorage key for now; `/history-48h` remains on its existing demo/local schema and is not broken by Live Monitor writes.
- Keeps the Live Monitor component mounted inside `AppShell` during normal in-app navigation so camera/sampling state and current-drive counts survive visits to `/video-upload` and `/history-48h`.
- Expands the Recent Events card as a smooth right-column overlay that grows upward over the EYE/YAWN and Drowsiness Risk panels; this is UI polish only and does not change alert/event data logic or navigate to `/history-48h`.
- Removes default in-video diagnostic metric panels, sound controls, test sound control, and separate Start/Stop Sampling controls from the product UI.
- Adds default-on sound alerts through the Web Audio API after the Start Camera user gesture. Normal warning-candidate events play a short gentle one-shot cue, while an unacknowledged critical-eye modal repeats a more urgent multi-beep cue until the user confirms it.
- Stops sampling, camera tracks, realtime sessions, active warning overlays, and sound repeat timers on Stop Camera and full app/component unmount; normal route navigation hides Live Monitor without stopping it.

Realtime backend endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /api/realtime/health` | Realtime service health and checkpoint path availability without expensive inference. |
| `POST /api/realtime/session/start` | Creates a lightweight in-memory realtime session. |
| `POST /api/realtime/frame` | Accepts one JPEG webcam frame and returns frame-level evidence plus temporal state. |
| `POST /api/realtime/session/stop` | Stops the in-memory realtime session without unloading model singletons. |

Realtime temporal state currently separates yawn and eye semantics:

| Concept | Current behavior |
| --- | --- |
| `mouth_active` | Current or near-current mouth/yawn activity; drives `mouth_warning_candidate`. |
| `recent_yawn_event` | 4.0 second fusion context only; does not by itself keep mouth-warning active. |
| `recent_yawn_reminder` | 8.0 second display-only reminder; does not affect fusion state. |
| `current_eye_evidence` | Current frame eye evidence category from `mean_p_eye_closed`. |
| `eye_warning_active` | Temporal eye-warning candidate state; enters after rolling evidence, exits with hysteresis. |
| `recent_eye_warning_reminder` | 4.0 second display-only note after a sustained moderate/strong eye-warning interval ends. |

Live Monitor eye temporal parameters:

| Parameter | Value |
| --- | ---: |
| `eye_closed_threshold` | `0.50` |
| `eye_warning_enter_rolling_mean` | `0.60` |
| `eye_warning_enter_consecutive_frames` | `2` |
| `eye_warning_exit_rolling_mean` | `0.40` |
| `eye_warning_exit_consecutive_frames` | `2` |
| `sustained_eye_warning_min_seconds` | `1.0` |
| `sustained_eye_warning_min_frames` | `5` |
| `recent_eye_warning_reminder_seconds` | `4.0` |

Live Monitor implementation files:

- `SystemUI/src/app/page.tsx`
- `SystemUI/src/components/dashboard/AppShell.tsx`
- `SystemUI/src/components/dashboard/DrowsinessLevelChart.tsx`
- `SystemUI/src/components/dashboard/LiveMonitorPage.tsx`
- `SystemUI/src/components/dashboard/LiveVideoCard.tsx`
- `SystemUI/src/components/dashboard/RecentEventsList.tsx`
- `SystemUI/src/components/dashboard/StatusMetricCard.tsx`
- `SystemUI/src/lib/liveMonitorDashboardStore.ts`
- `SystemUI/src/lib/liveMonitorDashboardTypes.ts`
- `SystemUI/src/lib/liveMonitorAlertUtils.ts`
- `SystemUI/src/lib/liveMonitorSoundUtils.ts`
- `SystemUI/src/components/dashboard/Sidebar.tsx`
- `SystemUI/src/components/dashboard/AppShell.tsx`
- `src/backend/app.py`
- `src/runtime/realtime_frame_inference.py`
- `src/runtime/realtime_temporal_state.py`

Stage 19.5 visual alert parameters:

| Parameter | Value |
| --- | ---: |
| Stable-state debounce for `eye_warning_candidate` | `1.0s` |
| Stable-state debounce for `mouth_warning_candidate` | `1.0s` |
| Stable-state debounce for `high_confidence_drowsiness_candidate` | `1.0s` |
| Stable-state debounce for `signal_unreliable` | `1.0s` |
| Normal-clear debounce | `2.0s` |
| Same-kind `eye_warning` event cooldown | `8s` |
| Same-kind `mouth_warning` event cooldown | `8s` |
| Same-kind `high_confidence` event cooldown | `10s` |
| Same-kind `signal_quality` event cooldown | `5s` |

Stage 19.6A sound alert behavior:

| Area | Current behavior |
| --- | --- |
| Default sound setting | On after Start Camera initializes or resumes Web Audio from the user gesture. |
| Sound trigger | Only when a Stage 19.5 debounced/cooldown session-local visual alert event is created, plus the critical-eye modal repeat timer while unacknowledged. |
| Test sound | Removed from the default Live Monitor product UI. |
| Eye-warning sound | Gentle two-tone one-shot reminder. |
| Mouth-warning sound | Soft two-tone one-shot reminder at a lower pitch. |
| High-confidence warning-candidate sound | More urgent multi-beep cue repeated by the critical-eye modal until confirmation. |
| Signal quality sound | Neutral low beep; signal quality remains a quality cue, not driver-state evidence. |
| Storage | React state only; no localStorage and no `/history-48h` write. |

Stage 19.6B Drowsiness Risk gauge mapping:

| State | Frontend score | Label | Helper |
| --- | ---: | --- | --- |
| Camera off or not sampling | `0` | `Idle` | `Start camera to monitor` |
| Normal monitoring | `20` | `Low` | `Monitoring` |
| Yawn warning candidate | `55` | `Medium` | `Yawn warning candidate` |
| Eye warning candidate | `74` | `High` | `Eye warning candidate` |
| Critical eye warning candidate | `92` | `Critical` | `Stop and rest when safe` |
| Face not visible / signal unreliable | `30` | `Signal Check` | `Center face in frame` |

Stage 19.7C dashboard binding and waveform chart:

| Dashboard area | Real data source | Counting/filtering rule |
| --- | --- | --- |
| EYE summary card | Stable `eye_warning` and `critical_eye_warning` dashboard events in the current browser-run drive session | Counts user-facing warning-candidate events once per stable alert event; does not count raw frames, raw `p_eye_closed`, pending debounce, or display-only reminders. |
| YAWN summary card | Stable `yawn_warning` dashboard events in the current browser-run drive session | Counts user-facing warning-candidate events once per stable alert event; does not count raw frames, raw `p_yawn`, pending debounce, or recent yawn reminders. |
| Recent Events | Lightweight local Live Monitor dashboard events | Filters by local calendar day for display; badge says `Today`; older records are not deleted at midnight. |
| Drowsiness Level chart | Current browser-run session state-derived display severity points from stable realtime warning-candidate states | Uses numeric timestamp X-axis coordinates, display-only `HH:mm` tick labels, a session-scoped dynamic start-to-1-hour window with no artificial minimum duration, 5-second bucket compaction, bounded point storage, render-only gray `Idle` baseline only outside active monitoring segments, and Low/Medium/High waveform colors matching the Drowsiness Risk gauge. |

Stage 19.7C — Product-Style Drowsiness Level Waveform Chart:

- The Drowsiness Level chart now uses a state-derived display severity trend, not a true model risk score, clinical fatigue curve, or final drowsiness truth.
- The waveform is derived from stable realtime warning-candidate states and the same frontend state source used by the Drowsiness Risk gauge.
- Small deterministic bounded visual variation is used only to make the product chart readable; it does not affect alert logic, sound logic, event counts, backend state, or stored warning-candidate event classification.
- Low, Medium, and High chart colors match the Drowsiness Risk gauge: green, orange, and red.
- The chart is scoped to the current browser-run Live Monitor session like the current-drive EYE/YAWN counts, so stale stored points from earlier sessions do not render after a fresh app load.
- The visible time window starts at the current Live Monitor session start with no artificial minimum duration, grows smoothly toward 1 hour, then rolls forward as a standard 1-hour window.
- Camera-off or pre-camera gaps are filled by a render-only gray `Idle` baseline at the Low-height display level; those idle scaffold points are not alert events, backend evidence, or `/history-48h` records.
- The chart avoids raw frame-level plotting, formatted-time X-axis collisions, duplicate timestamp clusters, and mock data.
- The Stage 19 implementation remained frontend-only at the time: no backend database, no raw image/video storage, no browser notification, and no final system-level drowsiness accuracy claim.

Stage 19 validation completed with Python compile checks, backend preflight, realtime health/start/frame/stop HTTP checks, inline temporal tests, `npm run lint`, `npm run build`, and browser checks for `/`, `/video-upload`, and `/history-48h`. Stage 19.7C frontend validation should keep `npm run lint`, `npm run build`, and route checks for `/`, `/video-upload`, and `/history-48h` synchronized with Live Monitor UI changes.

Stage 19.7C keeps realtime dashboard data binding for current-drive EYE/YAWN cards, today-only Recent Events, and the current-session state-derived display severity chart, with the chart now rendered as a product-style Idle/Low/Medium/High waveform. It keeps product-style Live Monitor UI polish, automatic sampling from Start Camera, default-on sound alerts after the camera user gesture, Drowsiness Risk gauge binding, and no default in-card diagnostic panel. It does not implement backend 48h history ingestion, database storage, raw image/video storage, final system-level drowsiness accuracy, deployment readiness, or final system-level performance claims.

May 19 Live Monitor layout polish:

- The desktop Live Monitor grid now reserves a practical right-column width when the sidebar is expanded, so EYES CLOSED, YAWN, Drowsiness Risk, and Recent Events are readable without letting the video/chart column dominate the page.
- Below the desktop split, the page stacks into a scrollable single-column layout with practical minimum heights for the video and chart areas.
- The Drowsiness Level chart keeps its existing state-derived data and thresholds; only layout/margin polish was applied so axis labels remain readable after the left column narrows.

Stage 20A — Local Account, Theme, and Notification Foundation:

- Added a local MVP login gate for SystemUI routes, including Live Monitor, Video Upload Analysis, 48h History, and Insights navigation. The current MVP gate uses one assigned local username/password account rather than open self-registration.
- The top-right profile menu now reflects the current local user, with name, email, role, local account status, and logout.
- Local user scoping is applied to frontend Live Monitor dashboard records, current-session chart points, and notifications where implemented; legacy local records remain supported for the first local user.
- Added a manual Day/Night theme toggle with localStorage persistence and no automatic system theme switching.
- The day sidebar now uses a light design; night mode uses a coherent dark app-shell, sidebar, header, card, chart, and page treatment.
- Sidebar active navigation now renders both the label and icon in white for stronger selected-state contrast, and the collapsed sidebar keeps the logo and expand control aligned in one row with complete, uncut touch targets.
- Added a clickable notification center for warning-candidate, system, and review notifications, with unread state and per-user localStorage filtering.
- This does not add production authentication, backend user-owned persisted history, database-backed history, model changes, final system-level claims, or deployment readiness. True multi-user isolation requires backend persistence and server-side ownership checks in a future stage.

Stage 20B — Live Monitor Local History Ingestion:

- Stable Live Monitor frontend warning-candidate events are now normalized into `/history-48h` frontend-local history under `visionguard.history48h.v1`.
- Records are scoped to the current local MVP user where Stage 20A auth is available.
- Ingestion uses stable debounced/cooldown events only, not raw frames, raw probabilities, pending debounce states, display-only reminders, or chart display points.
- No raw image, video, blob, or frame payloads are stored.
- This is frontend-local history ingestion only. It does not add backend database storage, production authentication, backend user-owned persisted history, browser notification API, model changes, final system-level claims, or deployment readiness.

Stage 21A — History / Insights Separation and User-Scoped Local Insights Dashboard:

- `/history-48h` is focused on event-level filtering, details, recent sessions, and manual review workflow.
- `/insights` is the read-only aggregate analytics page derived from the same current-user frontend-local history records.
- The latest History header polish removes the extra top explanatory notice block from `/history-48h`; lower-level safe interpretation notes and warning-candidate wording remain in place.
- This remains frontend-local warning-candidate review and analytics only. It does not add backend database storage, production authentication, raw image/video storage, deployment readiness, or final system-level claims.

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
| `SystemUI/src/app/page.tsx` | Lightweight Stage 19.7C route placeholder for `/`; persistent Live Monitor rendering is owned by `AppShell`. |
| `SystemUI/src/components/dashboard/AppShell.tsx` | Persistent application shell that keeps `LiveMonitorPage` mounted across normal route navigation so camera/session state is not reset by visiting other pages. |
| `SystemUI/src/components/dashboard/LiveMonitorPage.tsx` | Stage 19.7C/20B Live Monitor route content; stores the latest frontend warning-candidate risk state, writes lightweight dashboard events/state-derived display severity points, appends stable warning-candidate events to frontend-local 48h history, skips camera-off fake stored chart points, passes real current-drive/today/current-session data into the dashboard widgets, and owns the smooth right-column Recent Events overlay state. |
| `SystemUI/src/components/dashboard/LiveVideoCard.tsx` | Stage 19.7A clean Live Monitor product UI with single Start/Stop Camera button, automatic frame sampling, product yawn/eye overlays, concise critical-eye modal, face visibility cue, default-on sound after camera start, risk-state callback emission, and stable dashboard event emission. |
| `SystemUI/src/components/dashboard/DrowsinessRiskCard.tsx` | Stage 19.6B presentational Drowsiness Risk gauge bound to realtime frontend warning-candidate severity state with smooth score/needle animation. |
| `SystemUI/src/components/dashboard/StatusMetricCard.tsx` | Stage 19.7A top EYE/YAWN cards bound to current-drive stable warning-candidate event counts. |
| `SystemUI/src/components/dashboard/RecentEventsList.tsx` | Stage 19.7A today-only Recent Events card using real local Live Monitor dashboard events, fixed `Today` badge, and compact/expanded rendering for the right-column overlay interaction. |
| `SystemUI/src/components/dashboard/DrowsinessLevelChart.tsx` | Stage 19.7C product-style current-session state-derived display severity waveform chart with numeric timestamp X-axis binding, gray Idle baseline, Low/Medium/High segmented line colors, soft area fill, dynamic start-to-1-hour windowing with no artificial minimum duration, validation/filtering, and bucket compaction. |
| `SystemUI/src/lib/liveMonitorAlertUtils.ts` | Stage 19.5 frontend visual alert mapping, debounce, normal-clear, cooldown, and session-local alert event helper. |
| `SystemUI/src/lib/liveMonitorSoundUtils.ts` | Stage 19.6A Web Audio sound pattern and playback helper for Live Monitor warning sounds after the camera user gesture, with gentle one-shot normal warning cues and repeated urgent critical-eye cues until confirmation. |
| `SystemUI/src/lib/liveMonitorRiskUtils.ts` | Stage 19.6B frontend mapping from camera/sampling, alert kind, temporal state, and face/signal quality into a warning-candidate severity score. |
| `SystemUI/src/lib/liveMonitorDashboardTypes.ts` | Stage 19.7C typed lightweight local dashboard event and state-derived display severity point model. |
| `SystemUI/src/lib/liveMonitorDashboardStore.ts` | Stage 19.7C localStorage helpers for `visionguard.liveMonitorDashboard.v1`, today filtering, current-session risk-point filtering, display severity derivation, risk-point validation, 5-second bucket compaction, bounded chart point retention, and current-drive EYE/YAWN summaries. |
| `SystemUI/src/app/video-upload/page.tsx` | Stage 17.3+ Video Upload Analysis route. |
| `SystemUI/src/components/video-upload/` | Stage 17.5 modular UI components for upload, compact result overview, summary metrics, expandable intervals, tabbed figures, keyframes, and technical evidence. |
| `SystemUI/src/app/history-48h/page.tsx` | Stage 18/20B 48h History route. |
| `SystemUI/src/components/history-48h/` | Stage 18/20B modular UI components for demo/local and Live Monitor local history summary, charts, timeline, sessions, source badges, and manual review queue. |
| `SystemUI/src/lib/history48hTypes.ts` | Stage 18/20B strongly typed history event/session model with frontend-local source and optional user scoping fields. |
| `SystemUI/src/lib/history48hMockData.ts` | Stage 18/20B seeded demo/local warning-candidate history data, with user-scoped demo generation when requested. |
| `SystemUI/src/lib/history48hStorage.ts` | Stage 18/20B localStorage load/save/reset/clear/append helpers using `visionguard.history48h.v1`, including pruning, deduping, compatibility normalization, and current-user merge behavior. |
| `SystemUI/src/lib/history48hUtils.ts` | Stage 18/20B filters, aggregations, chart helpers, safe labels, source labels, and copy-summary text. |
| `SystemUI/src/lib/liveMonitorHistoryIngestion.ts` | Stage 20B pure mapper and append helper that converts stable Live Monitor dashboard events into frontend-local 48h history records without raw frame/image/video payloads. |
| `scripts/start_stage17_ui.sh` | One-command local launcher for backend and frontend. |
| `Makefile` | Project-level command target `make stage17-ui`. |
| `src/runtime/realtime_frame_inference.py` | Stage 19 realtime single-frame webcam evidence inference service. |
| `src/runtime/realtime_temporal_state.py` | Stage 19 session-local realtime warning-candidate temporal state. |
| `docs/STAGE17_VIDEO_UPLOAD_RESULT_SCHEMA.md` | Stage 17 response, summary, interval, timeline, and keyframe schema. |
| `docs/STAGE17_3_LOCAL_LAUNCH_GUIDE.md` | Local launcher and troubleshooting guide. |
| `docs/STAGE17_3_VIDEO_UPLOAD_UI_PAGE_REPORT.md` | Stage 17.3 UI implementation report. |
| `docs/STAGE17_4_VIDEO_UPLOAD_UI_ACCEPTANCE_CHECKLIST.md` | Manual acceptance checklist for the video-upload MVP. |
| `docs/STAGE17_4_DEMO_SCRIPT.md` | Safe-worded demo script for Stage 17.4. |
| `reports/stage17_4_video_upload_mvp_stabilization_report.md` | Historical Stage 17.4 stabilization report, superseded by Stage 17.5 UI cleanup and Stage 18 history-page work. |
| `docs/STAGE17_5_EYE_EVIDENCE_CALIBRATION.md` | Stage 17.5 eye evidence calibration and strength-gate documentation. |
| `reports/stage17_5_eye_evidence_calibration_report.md` | Stage 17.5 implementation report. |
| `docs/STAGE17_5_VIDEO_UPLOAD_UI_EVIDENCE_REVIEW_PAGE.md` | Stage 17.5 evidence review UI page report. |
| `docs/STAGE17_5_VIDEO_UPLOAD_UI_FALLBACK_POLISH.md` | Stage 17.5 keyframe fallback and optional-field UI polish report. |
| `docs/STAGE17_5_VIDEO_UPLOAD_UI_SECOND_PASS_CLEANUP.md` | Stage 17.5 second-pass `/video-upload` evidence review cleanup report. |
| `docs/STAGE18_HISTORY_48H_UI_PAGE_PLAN.md` | Stage 18 frontend-only 48h History page implementation and validation record. |

## 10. Reporting Notes and Limitations

Use careful wording in reports and presentations:

- Say these are specialist-module results.
- Do not claim final system-level driver drowsiness accuracy.
- YawDD/YawDD+ Dash results are mouth/yawn classification results.
- MRL Eye results are eye open/closed classification results.
- Subject-level split was used for the current YawDD and MRL Eye modules.
- Stage 15 and Stage 17 use rule-based fusion to produce warning-candidate states.
- Learned fusion, final fatigue scoring, final system-level performance scoring, and deployment readiness remain future work.
- Stage 17.5 outputs are uploaded-video warning-candidate analysis results, not final drowsiness truth.
- Stage 18/20B `/history-48h` data is frontend-local browser history; Stage 20B adds stable Live Monitor local ingestion, not backend database storage.
- Stage 19 Live Monitor outputs are realtime rule-based warning-candidate states from webcam frame evidence, not final system-level drowsiness accuracy.
- Stage 19 recent yawn context and recent sustained eye-warning reminders are temporal/contextual UI evidence; display-only reminders do not drive high-confidence escalation.
- Stage 19.5 visual alert events remain the stable source for user-facing Live Monitor warning-candidate dashboard events.
- Stage 19.6A sound alerts default On after Start Camera and are tied to debounced warning-candidate event creation, plus the unacknowledged critical-eye modal repeat timer.
- Stage 19.6B Drowsiness Risk values are frontend warning-candidate severity scores for display only, not final drowsiness truth.
- Stage 19.7C dashboard widgets use stable frontend events and timestamp-clean, bucketed state-derived display severity samples, not raw frame counts or raw probability timelines.
- Stage 19.7C stores lightweight Live Monitor dashboard records under `visionguard.liveMonitorDashboard.v1`; Stage 20B also writes stable warning-candidate history records under `visionguard.history48h.v1`.

Known artifact caveat:

- Final Stage 7 YawDD results are visible in `colab_file/stage7_yawdd_training_r.ipynb`.
- The local `artifacts/results/initial_results.csv` currently contains stale `not_run` values and should not be used for final Stage 7 reporting unless refreshed.

## 11. Maintenance Notes

- Keep `dataset/`, zip files, and checkpoint binaries out of normal Git.
- Metrics CSV/JSON, reports, figures, and error-analysis images may be committed for reproducibility.
- If checkpoints need to be versioned, use Git LFS rather than normal Git.
- When adding new results, cite the exact local artifact used as the source of truth.
- Keep the NTHUDDD2 work documented as an explored branch, not as the active final system direction.
- Keep the Stage 17 launcher, schema, Stage 17.5 UI reports, Stage 18/20B/21A history and insights documentation, and Stage 19/20A Live Monitor/app-shell realtime API/state/visual-alert/sound/layout notes synchronized whenever the upload UI, history UI, insights UI, Live Monitor UI, app shell, or backend API changes.
- Recommended next technical stages: design any browser notification or persisted history ingestion separately; keep those future stages behind the warning-candidate boundary.
