# VisionGuard production readiness checklist

11 November 2026. Runtime SHA `08a4da1`. This checklist applies production-ML concepts to the prototype. It records engineering checks; model performance and road safety require separate evaluation.

| Area | Verified evidence | Remaining condition |
| --- | --- | --- |
| Reproducibility | Exact pins/model hashes; independent clone; Linux arm64 CPU browser flow | Other platforms; offline and OS-package reproducibility |
| Software checks | 35 tests; scoped quality checks; frontend build; three CI jobs pass. [Ruleset 27044329](https://github.com/JohnCoffey-commits/Drowsiness-Detection/rules/27044329) enforces the three checks on the PR base from 12 November 2026 | Wider regression coverage |
| Model performance | Real models complete one authorized video | Labelled representative evaluation, error analysis and thresholds; platform output differences |
| Configuration | CPU and writable paths explicit; browser API URL built correctly | Managed deployment settings, TLS and secrets before remote use |
| Input handling | Size/format/empty/metadata checks; directory confinement; readiness errors | Rate limits, concurrency/queue limits, hostile decoder inputs |
| Access and privacy | Localhost ports; non-root users; readonly models; compact archive | Server authentication/authorization; retention and authorized deletion |
| Storage and recovery | SQLite and outputs survive stop/start and recreation | Tested backups/restores, storage limits and disk-full recovery |
| Health and logging | Liveness, model readiness and pipeline logs distinguished | Metrics, alert routing and availability objectives |
| ML monitoring | Signal quality and rule outputs recorded | Drift/ROI trends, versioned evaluation and rollback criteria |
| Release governance | Exact version, draft PR, linked Issues, private Project and two documentation/CI reviews | Further independent practice and release sign-off |

My next priorities are more independent practice with the workflow, labelled failure cases, authenticated access and recovery testing. Keep human interpretation of warning candidates. No diagnostic or safety guarantee follows from these tests.

Reference: [Google MLCC Production ML Systems](https://developers.google.com/machine-learning/crash-course/production-ml-systems). The passing-result screenshot I supplied is kept in my private portfolio; external-course recognition remains to be confirmed.
