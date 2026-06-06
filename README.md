# Driver Drowsiness Detection Prototype

This repository contains a driver drowsiness warning-candidate prototype for a deep learning group project. The system does not use one end-to-end drowsy/not-drowsy classifier. Instead, it uses two specialist visual modules:

- a mouth/yawn classifier trained from YawDD/YawDD+ Dash evidence
- an eye open/closed classifier trained from MRL Eye evidence

The runtime pipeline extracts face regions from video frames, runs the specialist models, and combines their outputs with rule-based temporal logic. The output should be interpreted as warning-candidate evidence for review, not final system-level drowsiness accuracy.

## Branches

The repository currently has two important branches:

- `main`: core machine learning pipeline, FastAPI backend code, experiment reports, manifests, and selected output evidence.
- `codex/visionguard-github-update`: includes the full `SystemUI/` Next.js frontend, Live Monitor prototype, History/Insights pages, local archive support, and updated deployment documentation.

If reviewing the frontend demo, please use the `codex/visionguard-github-update` branch or a pull request based on that branch. The `main` branch is mainly the core ML/backend submission state.

## What Is Included

Implemented project components include:

- YawDD/YawDD+ Dash mouth/yawn data preparation and model training evidence
- MRL Eye open/closed data preparation and model selection evidence
- subject-level train/validation/test splits to reduce identity leakage
- runtime eye and mouth ROI extraction scripts
- rule-based temporal warning-candidate fusion
- FastAPI uploaded-video analysis backend
- human-readable reports and selected evaluation artifacts
- SystemUI frontend work on the `codex/visionguard-github-update` branch

## Repository Structure

```text
src/          Python data processing, training, runtime, and backend code
reports/      Current human-readable experiment and validation reports
docs/         Project docs, stage notes, learning guides, final report, and archive
artifacts/    Dataset manifests, split files, mappings, and audit evidence
outputs/      Selected experiment results, figures, and model-selection summaries
colab_file/   Colab notebooks used for GPU training and result records
scripts/      Local helper scripts
tests/        Lightweight regression tests
SystemUI/     Frontend application, available on the UI branch
```

Large raw datasets, model checkpoints, generated runtime outputs, and local environment folders are intentionally not committed to normal Git.

## Setup

Create a Python environment and install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some historical runtime scripts were validated with a local environment named `.venv-stage10`. If a script or launcher expects that path, either create that environment or adapt the command to the active Python environment.

For the frontend branch, install Node dependencies separately:

```bash
cd SystemUI
npm install
```

## Running The Backend

From the repository root:

```bash
python src/backend/app.py --host 127.0.0.1 --port 8000
```

The backend provides uploaded-video analysis and serves generated session artifacts. The static backend upload test page is available at:

```text
http://127.0.0.1:8000/static/upload_test.html
```

On the branch that includes `SystemUI/`, the frontend can be started with:

```bash
cd SystemUI
npm run dev
```

The frontend normally expects the backend at `http://127.0.0.1:8000`.

## Required Local Assets

The following files are required for full runtime inference but are not committed to normal Git because they are large or generated assets:

```text
dataset/
outputs/mrl_eye/checkpoints/best_mobilenet_v2_mrl_eye.pt
checkpoints/resnet18_best.pt
artifacts/models/face_landmarker.task
```

The model checkpoint paths are referenced by the runtime scripts. Without these files, the repository can still be inspected and reports can be read, but full video inference will not run.

The MediaPipe face landmarker asset can be downloaded with:

```bash
mkdir -p artifacts/models
curl -L -o artifacts/models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

Model checkpoints should be provided separately, for example through a course submission attachment, GitHub Release asset, Google Drive link, or Git LFS if checkpoint versioning is required.

## Results Summary

The main reported model results are specialist-module metrics:

- Mouth/yawn specialist: ResNet18 achieved 99.37% test accuracy and 97.18% yawn F1 in the completed Stage 7 run.
- Eye open/closed specialist: MobileNetV2 achieved 98.63% test accuracy and 98.63% macro F1 in the Stage 9/9B evaluation.

These values are not final system-level driver drowsiness accuracy. They evaluate specialist tasks on their own datasets and splits. The runtime system adds ROI extraction, temporal smoothing, signal quality handling, and rule-based fusion, which is a separate evaluation problem.

## Limitations

This project should be understood as a local research and demonstration prototype. Current limitations include:

- no final road-level or clinical validation
- no final system-level drowsiness accuracy claim
- no trained end-to-end temporal fusion classifier
- no production authentication
- no cloud database or production backend deployment in the submitted core branch
- raw datasets and model checkpoints are not included in normal Git

The safest wording is warning-candidate analysis rather than final drowsiness detection truth.

## Useful Files For Review

- `docs/PROJECT_STRUCTURE.md`: repository structure and module map
- `docs/PROJECT_CURRENT_STATUS.md`: current project status and claim boundaries
- `docs/README.md`: documentation map
- `reports/mrl_eye_stage9b_error_analysis.md`: eye model comparison and selection
- `reports/stage15_real_mouth_eye_fusion_validation_report.md`: synchronized fusion validation
- `reports/stage17_video_upload_detection_mvp_report.md`: uploaded-video backend and pipeline report
- `tests/test_stage17_5_eye_evidence_calibration.py`: lightweight regression test for Stage 17.5 evidence calibration
