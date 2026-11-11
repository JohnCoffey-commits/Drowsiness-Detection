# Checks and their limits

`bash scripts/check.sh backend|frontend|audit|all` is the common local/CI entry point. Use the environment setup in `RUNNING.md`.

## Backend behavior

There are 35 local passing tests as of 11 November: three historical eye-evidence calibration tests and 32 new Goal 3 parameterized cases. The historical assertions are preserved; only an import-order annotation documents their existing direct-script path bootstrap.

New tests exercise session path and symlink boundaries, invalid/empty/oversized upload handling and cleanup, structured pipeline failure, finite and correctly typed frame metadata, missing model feedback, frame limits, separate session state, stop/reset behavior, temporary SQLite upsert/query/review/export, media rejection, optional write token, disabled archive, liveness/readiness distinction and explicit CPU selection. Inference substitutes intentionally avoid private assets. Temporary databases never write to the user's real archive.

The first new regression run produced 12 failed cases: dot session IDs, empty/oversized file handling, and invalid frame numbers. After fixes, 29 tests passed. Five later tests added runtime health and temporal contracts. An incorrect test assumption about a nonexistent response field was corrected to the actual documented signal fields; no failing production behavior was hidden.

The latest measured coverage is 73% across backend API, SQLite archive and temporal-state modules. It is diagnostic evidence, not a target or an accuracy metric. Models, browser camera collection and every historical experiment are outside this suite.

Ruff checks fatal syntax/import/name mistakes (E4/E7/E9/F) in affected API/archive/realtime modules, tests and asset verification. Black checks affected API/archive/realtime files, new tests and the asset script. Mypy checks the SQLite archive and asset verification script, not the entire untyped ML pipeline. One existing httpx TestClient deprecation warning remains visible.

## Frontend and dependency checks

Frontend ESLint, TypeScript `--noEmit`, and Next production build passed locally. Next and eslint-config-next moved from 16.2.5 to 16.4.6 in response to the actual audit. The shadcn generator CLI was removed from application dependencies: only its original 4.7.0 CSS was used. That exact stylesheet is retained with MIT attribution, preserving the baseline's styles without shipping the CLI dependency graph. A narrow compatible `brace-expansion` 1.x override pins 1.1.21 after automatic audit repair left an affected version.

Fresh npm audit: 31 findings before changes (including 2 critical and 14 high), then 7 moderate development-tool findings and zero high/critical. The remaining chain is typescript-eslint to minimatch 10; its currently installed latest typescript-eslint 8.72.2 still depends on that major. A forced incompatible transitive major upgrade was not applied. The gate fails on high/critical findings; moderate findings remain visible for review. Runtime `npm audit --omit=dev` must be recorded separately if claimed.

Python dependencies have exact transitive pins in `requirements/dev.txt` and `runtime.txt`. CPU torch/vision come from the official PyTorch CPU index; their support packages are pinned in `torch-constraints.txt` and installed from PyPI first. The full native environment audit found no known vulnerabilities at execution time. Advisory databases change: a passing audit does not imply absence of vulnerabilities.

## GitHub Actions

Three jobs cover backend behavior/quality, frontend lint/types/build and dependency audits. Versions are explicit; actions are pinned to verified full commit SHAs. Permissions are `contents: read`, checkout does not retain credentials, cache keys depend on lockfiles, jobs have timeouts, and test/audit artifacts are retained for 14 days even on failure. No secrets, models or private videos are required.

[Run 42967521499](https://github.com/JohnCoffey-commits/Drowsiness-Detection/actions/runs/42967521499) passed all three jobs for runtime commit `08a4da1f3b2d58effdca61c2973fc7529de21451` (35 tests). The earlier [run 42966445209](https://github.com/JohnCoffey-commits/Drowsiness-Detection/actions/runs/42966445209) passed at `4773a8727b08b7794514aea046f15dc79e8c64f3` (34 tests). At the initial verification on 11 November, the base branch had no required status checks.

A 35th test executes the Stage 11 CLI with synthetic CSV input and obstructed legacy global-report storage. It verifies session-scoped reporting, covering the real non-root container PermissionError. Production npm audit (`--omit=dev`) separately reported zero findings.

## Required checks

On 11 November 2026 the owner authorised `Backend behavior and quality`, `Frontend lint types and build`, and `Dependency audit` as the required GitHub Actions checks for `codex/visionguard-github-update`. GitHub requested account reauthentication when saving the ruleset. On 12 November 2026, [ruleset 27044329](https://github.com/JohnCoffey-commits/Drowsiness-Detection/rules/27044329) was saved and verified as Active. It matches only `codex/visionguard-github-update` and requires `Backend behavior and quality`, `Frontend lint types and build`, and `Dependency audit`, all sourced to GitHub Actions. No bypass actors or additional rules are configured. The raw API response is retained locally in `sources/E08-active-ruleset.json`.
