# Driver Drowsiness Detection Prototype

This repository contains a modular driver drowsiness warning-candidate prototype for a deep learning group project. The system does not use one end-to-end drowsy/not-drowsy classifier. It separates the problem into two specialist visual evidence channels and combines them with runtime signal-quality checks and temporal rules.

## What The System Does

- Mouth/yawn specialist: classifies mouth crops as `no_yawn` or `yawn` and outputs `p_yawn`.
- Eye open/closed specialist: classifies eye crops as `closed` or `open` and outputs `p_eye_closed`.
- Runtime layer: extracts face, eye, and mouth regions from full-face video frames.
- Fusion layer: combines eye evidence, mouth evidence, and signal quality into warning-candidate states.
- Application layer: supports uploaded-video analysis and a frontend monitoring interface on the UI branch.

The output should be interpreted as warning-candidate evidence for review. It is not final system-level driver drowsiness accuracy, not a clinical diagnosis, and not a production safety certification.

## Repository Layout

```text
src/          Python preprocessing, training, runtime inference, and FastAPI backend code
SystemUI/     Next.js frontend application on the UI branch
artifacts/    Dataset mappings, split files, recovered metrics, and selected evidence artifacts
outputs/      Selected model outputs, figures, and evaluation summaries
colab_file/   Colab notebooks used for GPU training and result records
docs/         Minimal project documentation and technical learning notes
scripts/      Local helper scripts
tests/        Lightweight regression tests
```

Large raw datasets, model checkpoints, generated runtime outputs, and local environment folders are intentionally not committed to normal Git.

## Setup

Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some runtime scripts were validated with a local environment named `.venv-stage10`. If a launcher expects that path, either restore that environment or adapt the command to the active Python environment.

For the frontend branch:

```bash
cd SystemUI
npm install
npm run dev
```

## Running The Backend

From the repository root:

```bash
python src/backend/app.py --host 127.0.0.1 --port 8000
```

The uploaded-video backend is then available through the FastAPI routes under `http://127.0.0.1:8000`.

## Required Local Assets

The full runtime pipeline requires local assets that are not committed to normal Git:

```text
dataset/
outputs/mrl_eye/checkpoints/best_mobilenet_v2_mrl_eye.pt
checkpoints/resnet18_best.pt
artifacts/models/face_landmarker.task
```

The MediaPipe face landmarker asset can be downloaded with:

```bash
mkdir -p artifacts/models
curl -L -o artifacts/models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

Model checkpoints should be provided separately through the course submission package, release assets, cloud storage, or Git LFS if checkpoint versioning is required.

## Main Results

The reported metrics are specialist-module metrics, not full-system drowsiness accuracy.

| Specialist | Selected model | Key result |
| --- | --- | --- |
| Mouth/yawn | ResNet18 | 99.37% test accuracy, 97.18% yawn F1 |
| Eye open/closed | MobileNetV2 | 98.63% test accuracy, 98.63% macro F1 |

The final warning-candidate system adds ROI extraction, signal-quality handling, temporal smoothing, and rule-based fusion on top of these specialist modules.

## Main Documentation

- `docs/final/final_report_en.md`: final technical report.
- `docs/PROJECT_STRUCTURE.md`: compact repository and system map.
- `docs/DEPLOYMENT_RUNBOOK.md`: remote frontend plus local backend deployment workflow.
- `docs/tech_learning/`: technical learning notes kept for project understanding.

## Limitations

- No final road-level or clinical validation.
- No final system-level drowsiness accuracy claim.
- No trained end-to-end temporal fusion classifier.
- No production authentication or production backend in normal Git.
- Raw datasets and model checkpoints are not included in normal Git.
