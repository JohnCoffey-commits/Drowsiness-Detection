# Goal 3 task status

Updated 12 November 2026. This table records the project checks and my own activities.

| Work item | Acceptance evidence | Status |
| --- | --- | --- |
| Baseline and scope | Exact starting SHA and historical contribution boundary | Verified |
| Issue backlog and Project | Issues 1-4; private Project with acceptance criteria | Created and updated |
| API/archive/temporal regression checks | 3 preserved historical tests and 32 new cases | 35 pass locally and in CI |
| Dependencies and quality | Locks, Ruff, Black, scoped mypy, frontend lint/types/build | Pass; 7 moderate npm dev findings remain |
| Actions | Three actual jobs on runtime commit 08a4da1 | All passed. [Ruleset 27044329](https://github.com/JohnCoffey-commits/Drowsiness-Detection/rules/27044329) enforces the three checks on the PR base from 12 November 2026. |
| Containers | CPU, non-root, readonly models, persistent volume | Built and verified on Linux arm64 |
| Native video | Real checkpoints, authorized video at runtime SHA | Completed |
| Clean clone | New environment/npm installation; exact runtime SHA; browser to archive | Verified, including recreation persistence |
| Draft PR | Complete-app base and linked issues | PR 5 open and draft |
| Peer review | Two documentation/CI reviews and my reply | Recorded on 12 November; see review-record.md |
| Evidence pack | E05-E09 editable Word and PDF | Assembled separately; see local delivery QA record |
| My learning | Pytest and Docker/API operation records; reflection | Guided operations recorded; further independent practice and external recognition remain; actual hours unknown |

See `VERIFICATION.md`, `evidence-index.md` and `review-record.md` for evidence and limits. Machine runtimes are not student learning hours.
