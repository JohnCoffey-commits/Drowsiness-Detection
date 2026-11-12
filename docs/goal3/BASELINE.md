# Goal 3 baseline and contribution boundary

Recorded 11 November 2026 (Australia/Sydney); preparation checks were performed 9 November.

- Repository: https://github.com/JohnCoffey-commits/Drowsiness-Detection
- Baseline and PR target: `codex/visionguard-github-update` at `ae5e7c84213fb8c3c9352d4ee8e85347bce9ddd2`.
- Learning branch: `learning/reproducible-delivery`.
- Remote `main` was `6d34e6c0277273f3b5c8f1a4e0dd14adf95b8b93` when rechecked on 11 November.

The complete existing SystemUI frontend, upload pipeline, realtime model service, temporal rules and SQLite archive are on the baseline branch. Targeting that branch keeps their historical implementation out of this learning PR. Existing model training, datasets, evaluation, experiments and team contributions remain attributed to the original repository history. This task did not train a new model or build those application features.

The baseline contained three tests in `tests/test_stage17_5_eye_evidence_calibration.py`. They passed during preparation. Frontend lint and production build also passed. Model checkpoint loading and four existing image-crop inference checks passed; a blank-image MediaPipe check returned zero faces. These were preparation checks, not complete video or camera verification. Preparation audit found 32 npm vulnerabilities; a fresh 11 November audit found 31, including two critical findings. The changing audit database is part of the evidence context.

Initial local changes were `.gitignore` additions and untracked preparation records. The original source was unchanged. Private machine records, downloaded assets and videos stay in ignored `preparation/`; they are not public repository deliverables.

Goal 3 additions cover deterministic failure-boundary tests, small API input/path fixes, CPU configuration, distinct health endpoints, pinned dependencies, scoped quality checks, CI, container configuration and delivery documentation. Codex implemented and ran the technical checks. I later ran pytest and the Docker/API observation script myself. `AI_ASSISTANCE.md` and my private portfolio describe these contributions.

My A1 plan on page 8 allocated 45 hours and 2 CP to Goal 3 from 28 September to 23 October 2026. The recorded work took place in November. I have no reliable record of actual hours, and external credit recognition remains to be confirmed. This evidence covers Goal 3. I use the earlier assessed project as the starting point and identify the new work separately.
