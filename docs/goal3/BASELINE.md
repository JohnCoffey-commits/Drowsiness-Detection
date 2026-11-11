# Goal 3 baseline and contribution boundary

Recorded 11 November 2026 (Australia/Sydney); preparation checks were performed 9 November.

- Repository: https://github.com/JohnCoffey-commits/Drowsiness-Detection
- Baseline and PR target: `codex/visionguard-github-update` at `ae5e7c84213fb8c3c9352d4ee8e85347bce9ddd2`.
- Learning branch: `learning/reproducible-delivery`.
- Remote `main` was `6d34e6c0277273f3b5c8f1a4e0dd14adf95b8b93` when rechecked on 11 November.

The complete existing SystemUI frontend, upload pipeline, realtime model service, temporal rules and SQLite archive are on the baseline branch. Targeting that branch keeps their historical implementation out of this learning PR. Existing model training, datasets, evaluation, experiments and team contributions remain attributed to the original repository history. This task did not train a new model or build those application features.

The baseline contained three tests in `tests/test_stage17_5_eye_evidence_calibration.py`. They passed during preparation. Frontend lint and production build also passed. Model checkpoint loading and four existing image-crop inference checks passed; a blank-image MediaPipe check returned zero faces. These were preparation checks, not complete video or camera verification. Preparation audit found 32 npm vulnerabilities; a fresh 11 November audit found 31, including two critical findings. The changing audit database is part of the evidence context.

Initial local changes were `.gitignore` additions and untracked preparation records. The original source was unchanged. Private machine records, downloaded assets and videos stay in ignored `preparation/`; they are not public repository deliverables.

Goal 3 additions cover deterministic failure-boundary tests, small API input/path fixes, CPU configuration, distinct health endpoints, pinned dependencies, scoped quality checks, CI, container configuration and delivery documentation. AI performed implementation and machine verification. Student operation and understanding checks are recorded separately in `AI_ASSISTANCE.md` and the local evidence pack.

The original A1 PDF page 8 was read and visually checked on 11 November. It plans Goal 3 for 28 September to 23 October 2026, 45 hours, 2 CP. Actual implementation dates are in November; no hours or CP completion are inferred. This evidence covers Goal 3 only, not the whole A3 or all 6 CP. Prior assessed project content is background; any reuse of that content itself for assessment needs the student's course approval.
