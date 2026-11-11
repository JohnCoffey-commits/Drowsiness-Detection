# VisionGuard production readiness checklist

11 November 2026. This checklist applies production-ML systems thinking to an existing research prototype. It does not certify production or road-safety readiness. The student's completion of the Google MLCC module is not established by this document.

| Area | Current evidence | Remaining condition |
| --- | --- | --- |
| Reproducibility | Pinned packages, versioned configuration, model hashes, container definitions | Final committed clean-clone execution and supported-platform record |
| Software checks | API, archive and temporal tests; scoped lint, format and types; frontend build | Real Actions result; broader application/browser regression coverage |
| Model identity | Three verified assets, safe checkpoint loading, CPU selection | Dataset/version provenance and redistribution rights must remain documented |
| Model performance | One authorized video completes the native pipeline | Labelled representative evaluation, thresholds, false positive/negative analysis and subgroup conditions |
| Input failures | Extension/size/empty/metadata validation, directory confinement, readiness 503 | Rate limits, concurrency/queue limits and resilient video decoding for hostile inputs |
| Configuration | Browser URL is a build argument; CPU and persistent paths explicit | Validated deployment configuration, TLS and secret management for remote use |
| Access and privacy | Localhost binding, non-root containers, readonly models, raw-media archive rejection | Authenticated reads/writes, authorization, privacy notice and retention/deletion process before multiuser access |
| Persistence | Named volume planned for SQLite, inputs, outputs and logs | Restart evidence, tested backup/restore, retention bounds and disk-full behavior |
| Health and logs | Separate process liveness/model readiness, pipeline logs and summaries | Central metrics, alert routing and availability targets; no monitoring service is installed |
| ML monitoring | Rule outputs and signal quality are observable | Input drift, missing ROI rates, quality trends, versioned evaluation and rollback criteria |
| Release and review | Reviewable changes and acceptance criteria | Human peer review, required-check policy decision and release sign-off |

Priority next work is to complete independent reproduction and human review, then evaluate labelled failure cases and recovery. Keep human interpretation of warning candidates; no output is a medical diagnosis or a validated decision about a driver's safety.

Reference: [Google MLCC Production ML Systems](https://developers.google.com/machine-learning/crash-course/production-ml-systems). This document is an engineering application, not evidence of course completion.
