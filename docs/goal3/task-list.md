# Goal 3 task status

Updated 11 November 2026. Implementation, machine verification and personal learning are recorded separately.

| Work item | Acceptance evidence | Status |
| --- | --- | --- |
| Baseline and scope | Exact starting SHA and historical contribution boundary | Verified |
| Issue backlog and Project | Issues 1-4; private Project with acceptance criteria | Created and updated |
| API/archive/temporal regression checks | 3 preserved historical tests and 32 new cases | 35 pass locally and in CI |
| Dependencies and quality | Locks, Ruff, Black, scoped mypy, frontend lint/types/build | Pass; 7 moderate npm dev findings remain |
| Actions | Three actual jobs on runtime commit 08a4da1 | All passed. See the delivery manifest for the required-check ruleset status. |
| Containers | CPU, non-root, readonly models, persistent volume | Built and verified on Linux arm64 |
| Native video | Real checkpoints, authorized video at runtime SHA | Completed |
| Clean clone | New environment/npm installation; exact runtime SHA; browser to archive | Verified, including recreation persistence |
| Draft PR | Complete-app base and linked issues | PR 5 open and draft |
| Human peer review | Actual comments and responses | Pending student arrangement |
| Evidence pack | E05-E09 editable Word and PDF | Assembled separately; see local delivery QA record |
| Student learning | Own operations, understanding and reflection | Partial; new toolchain exercise and hours pending |

See `VERIFICATION.md`, `evidence-index.md` and `review-record.md` for evidence and limits. Machine runtimes are not student learning hours.
