# Verified environment and troubleshooting

Recorded 11 November 2026. Host inspection identified a Mac mini Mac18,5, Apple M6, arm64, 12 logical CPUs, 32 GiB RAM, macOS 27.0.1 (26A434). Preparation had approximately 657 GiB available storage. Python 3.11.18, Node 24.22.2, npm 11.19.1 and Git 2.54.0 were used. System Python was not replaced; `.venv` was created specifically for this checkout.

Docker Desktop 4.98.0, Engine/CLI 29.9.2 and Compose 5.6.1 were verified. Official hello-world completed. The Linux container platform is arm64, distinct from macOS/MPS. Model execution uses `VISIONGUARD_DEVICE=cpu` for comparison. CUDA and MPS acceleration are not claimed in containers.

Runtime pins include torch 2.15.0, torchvision 0.29.2, MediaPipe 1.2.2, OpenCV contrib 5.0.0.93, FastAPI 0.162.2, pandas 3.1.1 and NumPy 2.4.6. Do not install `opencv-python` alongside `opencv-contrib-python`; they both provide cv2. Full resolved lists are versioned under `requirements/`. Dependencies were installed from those final locks before local backend rechecking.

Base images use version and manifest digest: Python `3.11.18-slim-bookworm` at `sha256:a83ba18aef0f865b1a4a611c2cf460fe446f2795673bbb9fe43be570439e3b39`; Node `24.22.2-bookworm-slim` at `sha256:01d566e8d94d8c619591d78652babfccd3c6992a3dffa9ed5ab39e11cf6c70e9`. APT packages come from Debian Bookworm at build time; this is not a byte-identical, offline OS package lock. Record the resulting image ID/size, not just its tag.

## Observed problems and fixes

1. The local Git HTTPS push dry-run could not obtain credentials. The already connected GitHub plugin can publish reviewed tree contents and commits without granting new access. Exact remote commit/tree identity must then be fetched and verified locally.
2. Regression tests found dot-segment session traversal, unbounded realtime read, nonfinite/truncated frame metadata and rejected uploads left on disk. Input/path checks now fail as client errors and incomplete uploads are removed. Original failure logs are retained locally.
3. The first CPU Docker build resolved torch from its dedicated index but could not find the pinned filelock 4.2.9 there. Fixed support packages are now installed from PyPI first, followed by CPU torch/vision from the official CPU index. This preserves pins instead of silently relaxing them.
4. Frontend security repairs required updating Next and its ESLint config, retaining baseline shadcn CSS locally, and a targeted brace-expansion patch override. Seven moderate developer-tool findings remain documented.

Build logs, measured durations, image metadata, health responses and clean-clone results are retained in the separate local evidence pack. An initial frontend container build completed in 86.59 seconds. Final build/verification results supersede preliminary runs and must include the tested commit.
