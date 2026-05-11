# Project Structure Guide

## 1. Purpose of This Document

This document explains how the repository is organized for the modular driver drowsiness project. It is intended as a living guide for teammates who need to find datasets, preprocessing outputs, training scripts, reports, model outputs, and current analysis artifacts quickly.

The repository currently supports a modular driver monitoring system rather than one monolithic drowsiness classifier:

- YawDD/YawDD+ Dash mouth/yawn specialist -> `p_yawn`
- MRL Eye open/closed specialist -> `p_eye_closed`
- Runtime video analysis and rule-based fusion -> warning-candidate states
- FastAPI + Next.js Stage 17 video-upload analysis workstation

## 2. High-Level Project Architecture

| Module | Dataset source | Specialist task | Output concept | Current status |
| --- | --- | --- | --- | --- |
| Mouth/yawn module | Original YawDD Dash videos plus YawDD+ annotation files | Binary mouth/yawn classification | `p_yawn` | Stage 7 completed |
| Eye open/closed module | MRL Eye | Binary eye-state classification | `p_eye_closed` | Stage 9 and Stage 9B completed |
| Runtime temporal analysis | Controlled A/B/C/D videos and upload videos | Eye/mouth ROI extraction, timeline generation, rule-based fusion | warning-candidate timelines | Stage 10-15 completed as controlled-validation prototype |
| Video Upload Analysis MVP | Uploaded local videos through FastAPI + SystemUI | Professional warning-candidate review page | summary, intervals, figures, keyframes, technical files | Stage 17.4 local MVP stabilized |
| NTHUDDD2 branch | Official NTHU considered; Kaggle extracted-frame version explored | Drowsy/not-drowsy frame classification | Not part of final module direction | Not main direction |

Stage 17 currently produces rule-based drowsiness warning-candidate analysis for uploaded videos. Current reported accuracies are specialist-module results, not final system-level driver drowsiness accuracy. The project does not currently claim webcam support, deployment readiness, final system-level performance, or final drowsiness truth.

## 3. Repository Layout

Important top-level locations:

```text
Drowsiness_Detection/
  artifacts/
  checkpoints/
  colab_file/
  dataset/
  docs/
  outputs/
  reports/
  scripts/
  src/
  SystemUI/
  upload_test/
  .gitignore
  Makefile
  README_*.md
  requirements.txt
```

| Path | Purpose |
| --- | --- |
| `dataset/` | Raw or locally reconstructed datasets. This is large local data and is ignored by Git. |
| `artifacts/` | Preprocessing outputs, manifests, split files, visual checks, and intermediate results. |
| `reports/` | Human-readable reports for dataset inspection, preprocessing, split validation, training summaries, and model selection. |
| `src/` | Python source code for dataset preparation, preprocessing, training, and runtime checks. |
| `src/backend/` | FastAPI backend for Stage 17 upload analysis and safe artifact serving. |
| `SystemUI/` | Independent Next.js App Router frontend for the dashboard and Stage 17 video-upload analysis page. |
| `scripts/` | Local helper scripts, currently including the Stage 17 one-command launcher. |
| `upload_test/` | Local short videos for upload UI/backend validation. |
| `colab_file/` | Google Colab notebooks used for GPU training and Colab workflows. |
| `outputs/` | Synced final training outputs and runtime evidence outputs, including MRL Eye Stage 9/9B and Stage 10-17 runtime/upload evidence. |
| `checkpoints/` | Legacy or local model checkpoint location. Large checkpoint files should not be committed to normal Git. |
| `docs/` | GitHub-friendly project structure and current-status documentation. |

## 4. Dataset Folders

`dataset/` is for local raw or reconstructed data. It should generally not be committed.

Observed local dataset folders:

| Path | Purpose |
| --- | --- |
| `dataset/YawDD_raw/` | Original YawDD Dash video data. Used as the video source for reconstructed Dash frames. |
| `dataset/YawDD+/` | YawDD+ annotation files. These provide frame indices and class labels. |
| `dataset/YawDD_plus_reconstructed/` | Reconstructed Dash full frames and generated mouth crops derived from YawDD raw videos plus YawDD+ annotations. |
| `dataset/mrlEyes_2018_01/` | MRL Eye subject folders and annotation/stat files. Used for the eye open/closed module. |
| `dataset/NTHUDDD2/` | Kaggle extracted-frame NTHUDDD2 exploration data, not the current main system direction. |

Dataset notes:

- YawDD/YawDD+ uses `0 = no_yawn`, `1 = yawn`.
- MRL Eye uses `0 = closed`, `1 = open`.
- NTHUDDD2 Kaggle uses `notdrowsy = 0`, `drowsy = 1`, but this branch is not the current main module direction.

## 5. Artifacts and Manifests

`artifacts/` stores reproducible intermediate files from inspection, reconstruction, preprocessing, and splitting.

Important subfolders:

| Path | Purpose |
| --- | --- |
| `artifacts/mappings/` | CSV manifests produced during preprocessing and dataset preparation. |
| `artifacts/splits/` | Subject-level train/validation/test split files. |
| `artifacts/visual_checks/` | Contact sheets and visual sanity-check images. |
| `artifacts/results/` | Earlier baseline result outputs. Note that the local YawDD `initial_results.csv` currently appears stale and should not be treated as the final Stage 7 source. |
| `artifacts/preprocessed/` | Regenerable preprocessing outputs. Ignored by Git. |
| `artifacts/cache/` | Local cache files. Ignored by Git. |
| `artifacts/models/` | Local model artifacts. Ignored by Git. |

Key manifest examples:

| File | Meaning |
| --- | --- |
| `artifacts/mappings/yawdd_dash_all_labeled_frames.csv` | Reconstructed YawDD+ Dash labeled-frame manifest. |
| `artifacts/mappings/yawdd_dash_all_mouth_crops.csv` | All attempted YawDD mouth-crop rows, including failures. |
| `artifacts/mappings/yawdd_dash_all_mouth_crops_trainable.csv` | Trainable YawDD mouth-crop rows with split labels. |
| `artifacts/splits/yawdd_dash_subject_split.csv` | Leakage-safe subject-level YawDD split. |
| `artifacts/mappings/mrl_eye_all_images.csv` | Full MRL Eye image manifest. |
| `artifacts/mappings/mrl_eye_trainable.csv` | Trainable MRL Eye rows. |
| `artifacts/mappings/mrl_eye_trainable_with_split.csv` | MRL Eye trainable manifest with subject-level split labels. |
| `artifacts/splits/mrl_eye_subject_split.csv` | Subject-level MRL Eye split. |
| `artifacts/mappings/nthuddd2_kaggle_all_images*.csv` | Kaggle NTHUDDD2 exploration manifests. |

## 6. Reports

`reports/` contains human-readable Markdown and CSV summaries. These are useful for GitHub because they explain the decisions and checks behind the data pipeline.

Important reports:

| File | Purpose |
| --- | --- |
| `reports/yawdd_raw_dash_report.md` | Inspection of original YawDD Dash videos. |
| `reports/yawdd_plus_annotation_format_report.md` | Interpretation of YawDD+ annotation format and class IDs. |
| `reports/yawdd_dash_reconstruction_report.md` | Dash frame reconstruction summary. |
| `reports/yawdd_dash_visual_sanity_check.md` | Visual confirmation that YawDD+ class `1` corresponds to yawning and class `0` to non-yawning. |
| `reports/yawdd_dash_mouth_crop_report.md` | MediaPipe mouth-crop preprocessing summary and readiness result. |
| `reports/yawdd_dash_split_report.md` | YawDD subject-level split report and leakage checks. |
| `reports/mrl_eye_dataset_report.md` | MRL Eye Stage 8 inspection report. |
| `reports/mrl_eye_split_report.md` | MRL Eye subject-level split report. |
| `reports/mrl_eye_stage9_training_plan.md` | Stage 9 MRL Eye training design. |
| `reports/mrl_eye_stage9b_error_analysis.md` | Final Stage 9B MRL Eye model-selection report. |
| `reports/stage10_runtime_eye_roi_acceptance_report.md` | Stage 10 controlled-video runtime eye ROI acceptance report. |
| `reports/stage13_mouth_eye_fusion_design_report.md` | Stage 13 rule-based mouth-eye fusion design/prototype report. |
| `reports/stage14_mouth_yawn_runtime_validation_report.md` | Stage 14 runtime mouth/yawn validation report using the recovered Stage 7 checkpoint. |
| `reports/stage15_real_mouth_eye_fusion_validation_report.md` | Stage 15 real synchronized rule-based mouth-eye fusion validation report. |
| `reports/stage16_final_integration_summary_report.md` | Stage 16 final integration package summary and conservative claim boundary. |
| `reports/stage17_video_upload_detection_mvp_report.md` | Stage 17 video-upload backend/pipeline MVP report. |
| `reports/stage17_2_manual_review_interpretation_report.md` | Stage 17.2 manual interpretation report for safe warning-candidate wording. |
| `reports/stage17_4_video_upload_mvp_stabilization_report.md` | Stage 17.4 stabilization report covering launcher, acceptance, demo, and current limitations. |
| `reports/nthuddd2_kaggle_dataset_report.md` | Kaggle NTHUDDD2 exploration report and limitations. |
| `reports/nthu_dataset_report.md` | Earlier NTHUDDD2 inspection notes. |

## 7. Source Code

`src/` is organized by function.

### `src/data/`

Dataset inspection, manifest-building, splitting, validation, and spot-check scripts.

Important examples:

| File | Purpose |
| --- | --- |
| `src/data/inspect_mrl_eye.py` | Inspect raw MRL Eye files and produce dataset report. |
| `src/data/build_mrl_eye_manifest.py` | Build MRL Eye all-image and trainable manifests. |
| `src/data/split_mrl_eye_subjects.py` | Create leakage-safe subject-level MRL Eye split. |
| `src/data/spotcheck_mrl_eye.py` | Generate visual MRL Eye contact sheets. |
| `src/data/build_yawdd_dash_mapping.py` | Build YawDD/YawDD+ Dash frame mapping. |
| `src/data/extract_yawdd_dash_labeled_frames.py` | Reconstruct labeled Dash frames from original videos. |
| `src/data/build_yawdd_split.py` | Build YawDD subject-level split. |
| `src/data/build_nthuddd2_kaggle_manifest.py` | Build Kaggle NTHUDDD2 exploration manifest. |
| `src/data/split_nthuddd2_kaggle_subject.py` | Build Kaggle NTHUDDD2 subject split and LOSO folds. |

### `src/preprocessing/`

Mouth ROI crop generation for the YawDD mouth/yawn module.

| File | Purpose |
| --- | --- |
| `src/preprocessing/generate_yawdd_mouth_crops.py` | Generate mouth crops using MediaPipe Face Mesh landmarks. |
| `src/preprocessing/precompute_yawdd_mouth_crops.py` | Earlier mouth-crop preprocessing entrypoint. |

### `src/training/`

Training scripts for CNN baselines.

| File | Purpose |
| --- | --- |
| `src/training/train_classifier.py` | YawDD mouth/yawn classifier training helper. |
| `src/training/run_initial_baselines.py` | YawDD three-model baseline runner. |
| `src/training/train_mrl_eye_baselines.py` | MRL Eye Stage 9 baseline training pipeline. |

### `src/runtime/`

Runtime scripts for safe post-training checks.

| File | Purpose |
| --- | --- |
| `src/runtime/stage10_eye_roi_consistency.py` | Stage 10 runtime eye ROI consistency test using MediaPipe FaceLandmarker and the selected MRL Eye MobileNetV2 checkpoint. |
| `src/runtime/stage11_eye_temporal_analysis.py` | Stage 11 eye-only temporal smoothing / PERCLOS-like analysis. |
| `src/runtime/stage12_eye_alert_rule_analysis.py` | Stage 12 eye-only alert rule comparison and recommendation. |
| `src/runtime/stage13_mouth_eye_fusion_design.py` | Stage 13 rule-based mouth-eye fusion design/prototype. |
| `src/runtime/stage14_mouth_yawn_runtime.py` | Stage 14 runtime mouth ROI consistency and mouth/yawn inference using the recovered Stage 7 ResNet18 checkpoint. |
| `src/runtime/stage15_real_mouth_eye_fusion_validation.py` | Stage 15 real synchronized rule-based mouth-eye fusion validation using Stage 12 eye timelines and Stage 14 model-generated `p_yawn` timelines. |
| `src/runtime/system_video_upload_pipeline.py` | Stage 17 single-video upload analysis pipeline that runs eye branch, mouth branch, F5 fusion, and keyframe extraction. |
| `src/runtime/keyframe_extractor.py` | Stage 17 helper for saving warning-candidate keyframe screenshots. |

### `src/backend/`

FastAPI backend for local upload analysis and safe artifact serving.

| File | Purpose |
| --- | --- |
| `src/backend/app.py` | Stage 17 backend entrypoint. Provides `POST /api/analyze-video` and safe session file URLs under `/api/runs/{session_id}/...`. |
| `src/backend/static/upload_test.html` | Minimal standalone backend-hosted upload test page. The primary frontend is now SystemUI `/video-upload`. |

## 8. SystemUI Frontend

`SystemUI/` is an independent Next.js App Router frontend using TypeScript, Tailwind CSS, shadcn/base-ui style components, `lucide-react`, and `recharts`.

Important Stage 17 frontend files:

| Path | Purpose |
| --- | --- |
| `SystemUI/src/app/video-upload/page.tsx` | Route entry for `/video-upload`. |
| `SystemUI/src/components/dashboard/Sidebar.tsx` | Sidebar menu; Video Upload Analysis is placed directly under Dashboard. |
| `SystemUI/src/components/dashboard/PageChrome.tsx` | Shared dashboard page chrome/layout wrapper. |
| `SystemUI/src/components/video-upload/VideoUploadAnalysis.tsx` | Main Stage 17 upload analysis workstation component. |
| `SystemUI/src/components/video-upload/AnalysisSummaryCards.tsx` | Summary metric cards for duration, sampled frames, warning-candidate frame counts, yawn events, and suppressed escalation. |
| `SystemUI/src/components/video-upload/IntervalReviewTable.tsx` | Warning-candidate interval review table. |
| `SystemUI/src/components/video-upload/KeyframeEvidenceGallery.tsx` | Keyframe evidence gallery with timestamp, frame index, fusion state, probabilities, and reason metadata. |
| `SystemUI/src/components/video-upload/TechnicalEvidencePanel.tsx` | Collapsible technical evidence/download links. |
| `SystemUI/src/components/video-upload/InterpretationNotice.tsx` | Permanent safe interpretation warning and Stage 17 explanation text. |
| `SystemUI/src/lib/videoUploadTypes.ts` | TypeScript types for backend response, summary, intervals, figures, and keyframes. |
| `SystemUI/src/lib/videoUploadUtils.ts` | URL, formatting, interval, figure, keyframe, and copy-summary helpers. |

Current Stage 17 route:

```text
http://127.0.0.1:3000/video-upload
```

Permanent wording boundary for the page:

```text
This output is a rule-based drowsiness warning-candidate analysis, not final system-level drowsiness accuracy.
```

## 9. Colab Notebooks

`colab_file/` stores notebooks for GPU-based training and audit runs.

| Notebook | Purpose |
| --- | --- |
| `colab_file/stage7_yawdd_training.ipynb` | Stage 7 YawDD training notebook template. |
| `colab_file/stage7_yawdd_training_r.ipynb` | Completed Stage 7 YawDD run notebook with output metrics. |
| `colab_file/stage8_nthuddd2_kaggle_training.ipynb` | Kaggle NTHUDDD2 exploration notebook. Not current main direction. |
| `colab_file/stage9_mrl_eye_training.ipynb` | Stage 9 MRL Eye Colab training notebook. |
| `colab_file/stage9_mrl_eye_training_r.ipynb` | Completed Stage 9 MRL Eye run notebook. |

The completed Stage 7 YawDD result values are present in `colab_file/stage7_yawdd_training_r.ipynb`. The local `artifacts/results/initial_results.csv` currently appears stale and contains `not_run`, so teammates should not use that CSV as the final Stage 7 result source unless it is refreshed.

## 10. Outputs

`outputs/` is for synced final outputs from cloud/Colab runs.

The main current output tree is:

```text
outputs/mrl_eye/
  README.md
  artifact_inventory.md
  results/
  reports/
  figures/
  error_analysis/
  checkpoints/
```

Important MRL Eye output groups:

| Path | Purpose |
| --- | --- |
| `outputs/mrl_eye/results/` | Metrics CSV/JSON, histories, threshold sweeps, and Stage 9B model-selection machine-readable summaries. |
| `outputs/mrl_eye/reports/` | Colab-generated Stage 9 experiment summary. |
| `outputs/mrl_eye/figures/` | Training curves, confusion matrices, and closed-class precision-recall curves. |
| `outputs/mrl_eye/error_analysis/` | False-open and false-closed contact sheets. |
| `outputs/mrl_eye/checkpoints/` | Best model checkpoint files for ResNet18, MobileNetV2, and EfficientNet-B0. |

`outputs/mrl_eye/artifact_inventory.md` confirms the expected local MRL Eye output set is complete.

Stage 10 runtime output:

| Path | Purpose |
| --- | --- |
| `outputs/stage10_eye_roi_consistency_IMG_4901_controlled_terminal/` | Successful controlled-video runtime eye ROI consistency evidence package for `IMG_4901.mp4`, including predictions, failures CSV, contact sheets, debug frames, crops, temporal summary, figures, and acceptance JSON. |

Stage 10 supporting docs:

| File | Purpose |
| --- | --- |
| `docs/STAGE10_RUNTIME_EYE_ROI_DESIGN.md` | Stage 10 design notes and runtime ROI constraints. |
| `docs/STAGE10_IMPLEMENTATION_LOG.md` | Stage 10 implementation log. |
| `docs/STAGE10_ENVIRONMENT_SETUP.md` | Dedicated `.venv-stage10` environment setup and validation evidence. |
| `docs/STAGE10_CONTROLLED_VIDEO_TEST_LOG.md` | Controlled-video test log covering Codex/sandbox failure and successful manual Terminal run. |

Stage 13 fusion design outputs:

| Path | Purpose |
| --- | --- |
| `docs/STAGE13_MOUTH_EYE_FUSION_DESIGN.md` | Stage 13 fusion states, schema, and recommended tiered rule design. |
| `artifacts/audits/stage13_mouth_eye_fusion_design_2026-05-09/stage13_mouth_runtime_audit.md` | Audit confirming that real synchronized mouth/yawn timelines for A/B/C/D are not currently available. |
| `outputs/stage13_mouth_eye_fusion_design/` | Stage 13 design/prototype outputs, including synthetic mouth timelines, fusion timelines, rule comparison CSV, summary JSON, report, and figures. |
| `reports/stage13_mouth_eye_fusion_design_report.md` | Human-readable Stage 13 fusion design/prototype report. |

Stage 14 mouth/yawn runtime audit:

| Path | Purpose |
| --- | --- |
| `artifacts/audits/stage14_mouth_yawn_runtime_2026-05-09/stage14_mouth_model_audit.md` | Audit of mouth/yawn checkpoint, architecture, transform, and label mapping availability. |
| `artifacts/audits/stage14_mouth_yawn_runtime_2026-05-09/STAGE14_BLOCKED_MISSING_MOUTH_MODEL_INFO.md` | Historical blocking report from before the Stage 7 checkpoint was recovered locally. |
| `artifacts/audits/stage14_mouth_yawn_runtime_2026-05-09/STAGE14_DRIVE_CHECKPOINT_RECOVERY_REPORT.md` | Google Drive recovery report identifying the completed Stage 7 mouth/yawn checkpoint candidate. |
| `artifacts/audits/stage14_mouth_yawn_runtime_2026-05-09/STAGE14_CHECKPOINT_LOCAL_COPY.md` | Local copy record for the recovered checkpoint. |
| `artifacts/audits/stage14_mouth_yawn_runtime_2026-05-09/STAGE14_RECOVERED_CHECKPOINT_VERIFICATION.md` | Local checkpoint payload and ResNet18 compatibility verification. |
| `outputs/stage14_mouth_yawn_runtime_A_normal_open_baseline/` | Stage 14 runtime mouth/yawn output for A baseline video. |
| `outputs/stage14_mouth_yawn_runtime_B_realistic_drowsy_simulation/` | Stage 14 runtime mouth/yawn output for B realistic drowsy simulation video. |
| `outputs/stage14_mouth_yawn_runtime_C_mild_head_motion/` | Stage 14 runtime mouth/yawn output for C mixed fatigue/head-motion/occlusion video. |
| `outputs/stage14_mouth_yawn_runtime_D_controlled_long_open_closed/` | Stage 14 runtime mouth/yawn output for D controlled long open/closed reference video. |
| `reports/stage14_mouth_yawn_runtime_validation_report.md` | Human-readable Stage 14 multi-video runtime mouth/yawn validation report. |
| `docs/STAGE14_MOUTH_YAWN_RUNTIME_LOG.md` | Stage 14 implementation and run log. |

Stage 15 real synchronized fusion validation:

| Path | Purpose |
| --- | --- |
| `artifacts/audits/stage15_real_mouth_eye_fusion_2026-05-09/stage15_input_audit.md` | Input audit confirming Stage 12 eye timelines and Stage 14 model-generated mouth timelines were available and aligned. |
| `outputs/stage15_real_mouth_eye_fusion/` | Stage 15 real synchronized rule-based fusion outputs, including combined real mouth timeline, fusion timelines, rule comparison CSV, summary JSON, reports, and figures. |
| `reports/stage15_real_mouth_eye_fusion_validation_report.md` | Human-readable Stage 15 real mouth-eye fusion validation report. |
| `docs/STAGE15_REAL_MOUTH_EYE_FUSION_LOG.md` | Stage 15 run log and evidence summary. |

Stage 16 final integration package:

| Path | Purpose |
| --- | --- |
| `reports/stage16_final_integration_summary_report.md` | Final high-level integration summary, architecture, evidence inventory, claim boundaries, and demo plan. |
| `docs/STAGE16_FINAL_EVIDENCE_PACKAGE.md` | Structured checklist of final evidence files. |
| `docs/STAGE16_DEMO_AND_PRESENTATION_OUTLINE.md` | Conservative demo and presentation outline. |
| `docs/PROJECT_FINAL_STATUS_STAGE16.md` | Concise final Stage 16 status snapshot. |
| `artifacts/audits/stage16_final_integration_2026-05-09/STAGE15_FIGURE_TITLE_FIX.md` | Audit note for Stage 15 figure-title correction. |
| `artifacts/audits/stage16_final_integration_2026-05-09/final_repo_artifact_audit.md` | Non-destructive final repository artifact audit. |

Stage 17 video-upload MVP:

| Path | Purpose |
| --- | --- |
| `src/backend/app.py` | FastAPI backend for video upload analysis and safe session artifact serving. |
| `src/backend/static/upload_test.html` | Standalone backend-hosted upload test page. |
| `SystemUI/src/app/video-upload/page.tsx` | SystemUI video-upload analysis page. |
| `SystemUI/src/components/video-upload/` | Modular Stage 17 UI components for upload, summary cards, interval table, keyframes, technical evidence, and interpretation notice. |
| `SystemUI/src/lib/videoUploadTypes.ts` | TypeScript response and evidence types. |
| `SystemUI/src/lib/videoUploadUtils.ts` | Safe URL construction, formatting, interval merging, figure/keyframe grouping, and copy-summary helpers. |
| `outputs/system_video_upload_runs/` | Per-session Stage 17 upload-analysis outputs. |
| `reports/stage17_video_upload_detection_mvp_report.md` | Stage 17 implementation and validation report. |
| `reports/stage17_2_manual_review_interpretation_report.md` | Stage 17.2 interpretation-layer report for conservative manual review wording. |
| `reports/stage17_4_video_upload_mvp_stabilization_report.md` | Stage 17.4 current stabilization report. |
| `docs/STAGE17_VIDEO_UPLOAD_RESULT_SCHEMA.md` | Result JSON, timeline, API, and keyframe schema. |
| `docs/STAGE17_VIDEO_UPLOAD_DETECTION_LOG.md` | Stage 17 command log and validation summary. |
| `docs/STAGE17_2_RESULT_INTERPRETATION_SCHEMA_ADDENDUM.md` | Result interpretation schema addendum for Stage 17.2. |
| `docs/STAGE17_2_MANUAL_REVIEW_INTERPRETATION_NOTES.md` | Manual review interpretation notes for safe result discussion. |
| `docs/STAGE17_3_VIDEO_UPLOAD_UI_PAGE_REPORT.md` | Stage 17.3 UI page report and safe wording notes. |
| `docs/STAGE17_3_LOCAL_LAUNCH_GUIDE.md` | One-command local launch guide for backend and frontend. |
| `docs/STAGE17_4_VIDEO_UPLOAD_UI_ACCEPTANCE_CHECKLIST.md` | Manual acceptance checklist for the Stage 17.3/17.4 video-upload MVP. |
| `docs/STAGE17_4_DEMO_SCRIPT.md` | Demo script for presenting the Stage 17.4 warning-candidate MVP. |
| `scripts/start_stage17_ui.sh` | Starts FastAPI backend and Next.js frontend, and stops both on Ctrl+C. |
| `Makefile` | Includes `make stage17-ui` target for the one-command launcher. |
| `artifacts/audits/stage17_video_upload_mvp_2026-05-09/stage17_systemui_backend_audit.md` | SystemUI/backend audit for Stage 17. |

Expected C upload validation markers recorded for Stage 17.4:

| Marker | Expected value |
| --- | ---: |
| High-confidence warning-candidate frames | 9 |
| Suppressed brief-eye escalation frames | 8 |
| Keyframes | 4 |
| Figures | 3 |
| Interval table | present |

## 11. Model Checkpoints

Checkpoint locations:

| Path | Meaning |
| --- | --- |
| `outputs/mrl_eye/checkpoints/best_resnet18_mrl_eye.pt` | Best MRL Eye ResNet18 checkpoint. |
| `outputs/mrl_eye/checkpoints/best_mobilenet_v2_mrl_eye.pt` | Best MRL Eye MobileNetV2 checkpoint. This is the selected primary eye model. |
| `outputs/mrl_eye/checkpoints/best_efficientnet_b0_mrl_eye.pt` | Best MRL Eye EfficientNet-B0 checkpoint. |
| `checkpoints/resnet18_best.pt` | Recovered Stage 7 ResNet18 mouth/yawn checkpoint used by Stage 14+ mouth runtime. |
| `checkpoints/` | Legacy/local checkpoint folder for earlier training scripts and recovered local checkpoint copies. |

`outputs/mrl_eye/checkpoints/` and `checkpoints/resnet18_best.pt` are needed locally for model loading, but checkpoint binaries should generally not be committed to normal Git. Use Git LFS if checkpoint versioning is required.

## 12. Current Module Status

| Area | Status |
| --- | --- |
| YawDD/YawDD+ Dash mouth/yawn module | Completed and stable as the mouth/yawn specialist. |
| MRL Eye open/closed module | Stage 8 preparation, Stage 9 training, and Stage 9B model selection completed. |
| Selected MRL Eye model | MobileNetV2, default argmax / `p_eye_closed >= 0.50`. |
| Stage 10 runtime eye ROI consistency | Accepted for controlled video `IMG_4901.mp4`; not final drowsiness accuracy. |
| Stage 11 eye temporal analysis | Completed controlled-validation timeline analysis. |
| Stage 12 eye alert rule analysis | Completed eye-only rule comparison; not final system output. |
| Stage 13 mouth-eye fusion design | Completed rule-based fusion design/prototype. |
| Stage 14 mouth/yawn runtime | Completed runtime validation using recovered Stage 7 mouth/yawn checkpoint. |
| Stage 15 real synchronized fusion | Completed rule-based validation using synchronized eye and mouth timelines. |
| Stage 16 final integration package | Completed but superseded by Stage 17.4 as the current local MVP status. |
| Stage 17.1 sustained-eye gate | Completed; high-confidence warning candidate requires recent mouth/yawn evidence plus sustained eye-warning evidence. |
| Stage 17.2 interpretation wording | Completed; eye-warning evidence is not automatically described as verified sustained full eye closure. |
| Stage 17.3 Video Upload Analysis UI | Completed in SystemUI route `/video-upload`. |
| Stage 17.4 launcher and acceptance/demo docs | Completed; `make stage17-ui` starts local backend and frontend. |
| Safety-prioritized MRL Eye reference | ResNet18 with validation-selected threshold around `0.30`. |
| NTHUDDD2 branch | Explored but no longer the main system direction. |
| Current claim boundary | Rule-based uploaded-video warning-candidate analysis only; no webcam, final system-level performance, or deployment-readiness claim. |

## 13. What Should and Should Not Be Committed to GitHub

Generally useful to commit:

- Documentation in `docs/`
- Human-readable reports in `reports/`
- Lightweight CSV/JSON metrics and manifests in `artifacts/mappings/`, `artifacts/splits/`, and `outputs/mrl_eye/results/`
- Figures in `outputs/mrl_eye/figures/`
- Error-analysis contact sheets in `outputs/mrl_eye/error_analysis/`
- Colab notebooks in `colab_file/`
- Source code in `src/`
- Frontend source code in `SystemUI/src/`
- Local launcher scripts in `scripts/`
- Lightweight Stage 17 UI/launcher documentation

Generally do not commit to normal Git:

- Raw datasets under `dataset/`
- Dataset zip files (`*.zip`)
- Model checkpoints (`*.pt`, `*.pth`, `*.ckpt`)
- `outputs/**/checkpoints/`
- `SystemUI/node_modules/`
- `SystemUI/.next/`
- Python virtual environments such as `.venv-stage10/`
- Browser/test caches such as `.playwright-cli/`
- Local caches and preprocessed bulk data
- Generated upload-session evidence under `outputs/system_video_upload_runs/`, unless a specific lightweight report or selected demo artifact is intentionally being versioned.

The current `.gitignore` protects datasets, zip files, model checkpoints, `outputs/**/checkpoints/`, cache folders, and Python/Jupyter cache artifacts.

## 14. How to Update This Document

Update this document whenever:

- A new major folder is added.
- A dataset is adopted or retired.
- A preprocessing stage creates a new canonical artifact.
- A training stage creates a new output directory.
- A checkpoint location or selected model changes.
- The FastAPI upload API, SystemUI route structure, or launcher command changes.
- A Stage 17 acceptance checklist, demo script, or stabilization report is superseded.
- Git tracking policy changes for outputs or large files.

When adding a new result, prefer citing the exact local CSV/JSON/report path used as the source of truth.
