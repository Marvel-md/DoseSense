# Submission corrections — verification record

Checked on 14 September 2026 against the corrections based on upstream commit `4515b72b98834413eb322b61e3d1f2753a8b439f`.

## Completed

| Check | Result |
|---|---|
| Full reproduction | Generated 600 patients, trained the ensemble, evaluated all ablations, regenerated reports |
| Python tests | 81 passed in 24.17 seconds |
| Cross-process determinism | Cohort fingerprints match under different `PYTHONHASHSEED` values |
| Generated-report check | README, presentation result blocks and evaluation report match metrics.json |
| API startup | Full 600-patient cohort loaded successfully |
| Live functional render harness | 22 checks passed; five assessment states and all eight walkthrough cases present |
| Patient portal | Normal, partially filled and empty-history views rendered; risk-language check passed |
| Syntax and patch whitespace | Python parsing, JavaScript syntax and `git diff --check` passed |

Commands used:

```bash
OMP_NUM_THREADS=2 bash scripts/reproduce.sh
OMP_NUM_THREADS=2 python scripts/check_demo.py
python scripts/make_report.py --check
node --check tests/render_harness.js
```

Runtime: Python 3.12.14, Node.js 24.19.0 on Linux. Dependency versions are recorded in `requirements-lock.txt` and `artifacts/metrics.json`. Dependencies were installed into the available runtime; this was not a separate clean virtual-environment installation. These timings are test-run observations, not laptop performance guarantees.

## Why the figures changed

The old patient generator seeded NumPy from Python's process-randomised `hash()`. The public seed alone therefore did not reproduce the stored cohort across process restarts. Patient seeds now derive from a stable SHA-256 digest, with a subprocess regression test. This necessarily creates a new cohort/model; the regenerated figures supersede the old submission numbers.

The new run reports PR-AUC 0.900 versus 0.670 for the baseline and a 49.7% relative reduction in threshold-level false alerts. Those are synthetic-data measurements. Current, higher-precision values and definitions are in `artifacts/metrics.json` and `docs/evaluation.md`.

No predictive feature family or clinical policy was added or tuned to improve the test result. Changes address stable generation, metric interpretation, reporting, test setup and presentation preparation.

## Remaining checks outside this run

- Visual browser/accessibility review and rehearsal on the actual presentation machine.
- Docker build/runtime and migration of existing Docker feedback were not exercised here. The new feedback volume avoids persisting model artifacts across image rebuilds; old feedback is retained in the old volume and is not automatically migrated.
- Historical multi-seed benchmarks were not rerun and are not current submission evidence.
- The exact official-template PDF, final video duration, registered-member appearances, identifiers and submission-link access have not been checked in this run.
- No live website was deployed and no hackathon entry was submitted.

See `docs/submission-checklist.md` for the final artifact gates.
