# Committed runtime verification

Actual activity date: 11 November 2026, Australia/Sydney. Runtime commit: `08a4da1f3b2d58effdca61c2973fc7529de21451`. Baseline: `ae5e7c84213fb8c3c9352d4ee8e85347bce9ddd2`. Later evidence-only documentation changes do not constitute a different tested implementation.

## Independent checkout

A new clone initially checked out `4773a87`, created a new Python 3.11.18 virtual environment, installed `requirements/dev.txt`, ran a fresh `npm ci`, and passed `bash scripts/check.sh all`. It then fetched and detached at `08a4da1`, repeated the affected backend checks (35 passed), and built/started both containers from that checkout. No old virtual environment, node_modules or hidden configuration was copied. Three authorized model files were copied as data and verified by SHA-256. Global download and Docker layer caches remained available; this was not an offline or cache-empty test.

The independent Compose project used explicit ports and build settings:

```bash
API_PORT=8010 WEB_PORT=3010 \
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8010 \
FRONTEND_ORIGIN=http://127.0.0.1:3010 \
docker compose -p visionguard-goal3-clean up --build -d --wait
```

Browser observation confirmed local login, original-video preview and playback, upload to port 8010, completed results, and the message “Saved compact video analysis summary.” Backend logs recorded successful analyze-video and archive/video-run requests. SQLite export contained the matching video-run session. The History screen only displays live-monitor events by design. Browser console errors were not separately audited. Camera capture and live end-to-end latency were not tested.

## Actual results

| Execution | Sampled frames | Pipeline time | Result |
| --- | ---: | ---: | --- |
| Native macOS CPU at runtime SHA | 166 | 35.00 s | Completed; command wall time 35.49 s |
| Primary Linux arm64 CPU container | 166 | 60.79 s | HTTP 200; API wall time 61.45 s |
| Independent-clone browser to Linux CPU | 166 | 64.00 s | Completed; UI displayed 64.5 s; 7 keyframes and compact archive row |

One authorized 27.64-second portrait video was used without conversion. Private input, facial crops and full inference outputs are retained locally, not published. One unlabeled clip demonstrates execution, not accuracy, representativeness or driving-safety performance. Native candidate counts were normal 26, eye 76, mouth 0, combined 64; Linux counts were 27, 75, 13, 51 respectively. Both Linux runs agreed on those counts. Cross-platform numerical parity is not established.

## Container identity and storage

| Image in independent clone | Bytes | Image ID |
| --- | ---: | --- |
| Backend, Linux arm64 | 2673894446 | `sha256:9566e745060e59c5d2eb93eb73b4523d52605989e6b5990ac3f8f7a9cf5071c8` |
| Frontend, Linux arm64, API port 8010 baked in | 406290176 | `sha256:8a23b22b83bbb44e74acd17b05a0599550ebb7613a4bcdb94e9dd132180e7999` |

Runtime identity was UID/GID 10001 for backend and 1000 for frontend. All three model binds had `RW=false`; the data volume had `RW=true`. Liveness returned `ok: true`; model readiness separately returned `ready: true, device: cpu`.

The primary project retained a synthetic archive record and completed video output across stop/start and forced container recreation. The independent project also retained its browser-created video-run row and output across forced recreation, completed in 11.88 seconds. No volume deletion was used. Backup/restore and disk exhaustion were not tested.

Measured build history: initial frontend 86.59 s; first backend attempt failed after 216.46 s; support-dependency repair built in 237.81 s; EGL and GLES rebuilds took 61.90 s and 58.02 s. These runs had different cache states and are not a benchmark. The final report-path rebuild and independent build succeeded; their total wall times were not separately measured. Original logs retain layer timing.

## Remote checks and review

All three jobs in [Actions run 42967521499](https://github.com/JohnCoffey-commits/Drowsiness-Detection/actions/runs/42967521499) passed at the runtime SHA. Full ML inference is a separate local validation; CI uses inference substitutes on Ubuntu x86_64. [Draft PR 5](https://github.com/JohnCoffey-commits/Drowsiness-Detection/pull/5) remains unmerged. The linked private [Project](https://github.com/users/JohnCoffey-commits/projects/1) tracks four issues. I approved adding the required status checks after this runtime verification. [Ruleset 27044329](https://github.com/JohnCoffey-commits/Drowsiness-Detection/rules/27044329) was activated and verified on 12 November 2026. Two peers reviewed the documentation and linked CI results on 12 November; my reply is linked in `review-record.md`. Their review covered documentation commit `b9f47fd`, not later revisions.
