# Project Structure

This document gives a compact map of the repository and the main system modules.

## System Summary

VisionGuard is a modular driver drowsiness warning-candidate prototype. It uses two specialist visual models and a rule-based temporal fusion layer:

```text
YawDD/YawDD+ mouth crops -> mouth/yawn model -> p_yawn
MRL Eye crops            -> eye model        -> p_eye_closed
full-face video frames   -> ROI extraction   -> runtime evidence
runtime evidence         -> temporal fusion  -> warning-candidate states
```

The system is intentionally not a single end-to-end drowsy/not-drowsy classifier. Specialist metrics are reported separately from runtime warning-candidate behavior.

## Top-Level Directories

| Path | Purpose |
| --- | --- |
| `src/` | Python preprocessing, training, runtime inference, and backend code. |
| `src/backend/` | FastAPI service for uploaded-video analysis, realtime frame inference, and local archive APIs. |
| `src/runtime/` | Runtime ROI extraction, eye/mouth inference, temporal logic, video-upload pipeline, and keyframe extraction. |
| `src/training/` | Training helpers and model baseline code. |
| `SystemUI/` | Next.js frontend application on the UI branch. |
| `artifacts/` | Dataset mappings, split manifests, recovered metrics, and selected non-source evidence artifacts. |
| `outputs/` | Selected model outputs, result JSON/CSV files, figures, and validation artifacts. |
| `colab_file/` | Colab notebooks used for GPU model training and result records. |
| `docs/` | Minimal project documentation plus technical learning notes. |
| `scripts/` | Local launch, deployment, and utility scripts. |
| `tests/` | Lightweight regression tests. |

## Data And Model Evidence

| Area | Main files or directories | Notes |
| --- | --- | --- |
| Mouth/yawn data | `artifacts/mappings/`, `artifacts/splits/`, `colab_file/stage7_yawdd_training_r.ipynb` | YawDD/YawDD+ Dash reconstruction, mouth crops, subject-level split, and Stage 7 training record. |
| Mouth/yawn metrics | `artifacts/recovered_stage7_mouth_yawn/metrics_summary.json`, `artifacts/recovered_stage7_mouth_yawn/initial_results.csv` | ResNet18 is selected for runtime mouth/yawn inference. |
| Eye data | MRL Eye manifests and split artifacts under `artifacts/` and `outputs/mrl_eye/` | Eye open/closed specialist data and model-selection evidence. |
| Eye metrics | `outputs/mrl_eye/results/` | MobileNetV2 is selected for runtime eye inference. |
| Runtime validation | selected JSON/CSV/figure outputs under `outputs/` | Used to inspect ROI extraction, temporal behavior, and fusion outputs. |

## Runtime Pipeline

The uploaded-video analysis path is centered on:

```text
src/runtime/system_video_upload_pipeline.py
```

It combines these stages:

1. Full-frame video sampling.
2. Face and ROI extraction.
3. Eye specialist inference for `p_eye_closed`.
4. Mouth/yawn specialist inference for `p_yawn`.
5. Temporal quality checks and warning-candidate fusion.
6. Keyframe and summary artifact generation.
7. FastAPI response formatting.

The realtime path uses:

```text
src/runtime/realtime_frame_inference.py
src/runtime/realtime_temporal_state.py
src/backend/app.py
```

## Frontend Surface

The frontend branch contains `SystemUI/`, a Next.js application with:

- Live Monitor.
- Video Upload Analysis.
- History.
- Insights.
- Local settings and archive-aware summary views.

The frontend calls the local FastAPI backend for runtime evidence. The frontend does not retrain models and does not recompute the backend fusion decisions.

## Documentation Kept On GitHub

| File or directory | Purpose |
| --- | --- |
| `README.md` | Repository entrypoint and quick setup. |
| `docs/final/final_report_en.md` | Final technical report. |
| `docs/PROJECT_STRUCTURE.md` | This compact structure and system map. |
| `docs/DEPLOYMENT_RUNBOOK.md` | External-access deployment workflow. |
| `docs/tech_learning/` | Technical learning notes kept for project understanding. |

## Files Not Included In Normal Git

The following are expected to be local or provided separately:

```text
dataset/
checkpoints/*.pt
outputs/mrl_eye/checkpoints/*.pt
artifacts/models/face_landmarker.task
large generated runtime sessions
local virtual environments
local SQLite databases
```

## Claim Boundary

The repository supports a local research and demonstration prototype. It reports specialist model metrics and rule-based warning-candidate behavior. It does not claim final system-level drowsiness accuracy, production readiness, clinical validation, or road-level safety certification.
