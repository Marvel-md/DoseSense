# DoseSense

**Detect possible medication-taking difficulties from routine care records, without requiring patients to log missed doses.**

DoseSense combines pharmacy dispensing, prescription changes, appointment attendance, existing symptom questionnaires, laboratory trends and optional activity records. It builds a personal baseline, distinguishes temporary irregularities from sustained changes, and produces an explained worklist for clinician review.

**Hackathon prototype; all data is synthetic.** It does not establish whether a dose was taken, and it has no clinical validation. The doctor/patient role selector is a demo view switch, not authentication.

## What is implemented

- A bagged LightGBM ensemble with isotonic calibration on a separate patient split.
- 52 features in seven engineering families: refill, prescription, appointment, symptom, laboratory, behavioural and context. These families differ from the six record sources above.
- Personal-baseline change detection, including temporary, persistent and provisional shifts.
- Evidence completeness, ensemble disagreement and an explicit insufficient-evidence state.
- Priority based on likelihood, an illustrative consequence policy and confidence.
- Grouped SHAP explanations and rule-based barrier hypotheses that may abstain.
- A FastAPI backend, doctor dashboard, patient portal and SQLite feedback storage.
- Synthetic-data generation, patient-level evaluation, ablation, subgroup analysis and automated tests.

### What “without asking” means

Detection does not require manual dose logs. **New reports submitted through the patient portal are excluded from model inputs.** They are stored separately and shown to the clinician as patient-provided context. Existing symptom questionnaires in routine records do feed symptom features. We therefore do not claim that every model input is free of patient-reported information.

The API keeps `observed`, `inferred`, `hypothesis` and `action` separate. The portal view omits risk estimates, but the open demo API has no identity enforcement. It must only contain synthetic records.

## Measured results

<!-- BEGIN GENERATED RESULTS -->
**Synthetic-data evaluation only.** 148 held-out patients, 1,924 snapshots; seed 20260910. Generated from `artifacts/metrics.json` (2026-09-14 13:32:34 UTC).

| Metric | DoseSense | PDC / refill-gap baseline |
|---|---|---|
| ROC-AUC | 0.923 | 0.767 |
| PR-AUC | 0.900 | 0.670 |
| Precision | 0.886 | 0.763 |
| Recall | 0.787 | 0.647 |
| F1 | 0.834 | 0.700 |
| False alerts / 100 patient-months | 3.95 | 7.85 |

Relative change in false alerts: 49.7% reduction (a negative value means an increase). Expected calibration error: 0.0132. The evaluated model uses 52 features.

These are model-threshold results. The dashboard additionally applies uncertainty, state and priority rules. These measurements do not establish clinical outcomes, real-world accuracy or end-to-end workflow savings.
<!-- END GENERATED RESULTS -->

Full denominators, subgroup recall, detection-timing definitions and limitations: [evaluation](docs/evaluation.md). The numbers above and in the presentation are generated from the same artifact. Historical benchmark runs in `artifacts/benchmark.json` predate the stable patient-seed fix and are not current submission claims.

## Setup

Use Python 3.12 and Node.js 18 or newer for the dashboard check. On macOS, LightGBM may also need the OpenMP runtime (`brew install libomp`).

```bash
git clone https://github.com/Marvel-md/DoseSense.git
cd DoseSense
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements-lock.txt
bash scripts/reproduce.sh
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

On Windows, activate with `venv\Scripts\activate` and run the individual Python steps below if Bash is unavailable. Open http://127.0.0.1:8000 after startup completes. The API scores the cohort before becoming ready; startup time depends on the machine.

Individual reproduction steps:

```bash
python scripts/generate_data.py --patients 600
python ml/train.py
python ml/evaluate.py
python scripts/make_report.py
python -m pytest tests/ -q
```

`requirements.txt` declares supported lower bounds; `requirements-lock.txt` pins the environment used for the recorded run. Results can change with dependencies or seeds. The model, panel and generated cohort are intentionally not committed and must be generated before starting the API.

### Verify the demo

With the API running, in another terminal with the environment activated:

```bash
node tests/render_harness.js
python scripts/make_report.py --check
```

The harness retrieves current API payloads itself: no `/tmp` files or manual fixture preparation. It renders every assessment state present in the cohort, walkthrough cases, empty-series cases and the patient portal. Set `DOSESENSE_BASE_URL` for a different local URL. It exits nonzero on missing data, failed requests or render errors. It is a functional render check, not a visual browser review.

For an automated API startup and render check after generating the cohort/model:

```bash
python scripts/check_demo.py
```

The script starts a temporary local API, waits for readiness, runs the harness and stops the API. It uses a temporary feedback database and does not change clinician feedback.

A faster, smaller live panel can be selected with `DOSESENSE_PATIENT_LIMIT=120`, but walkthrough availability depends on the selected cohort. Rehearse with the exact panel you will present. Evaluation always uses the full generated dataset.

### Docker

```bash
docker compose up --build
```

Docker builds the cohort and model. Feedback persists separately from model artifacts, so rebuilding does not keep a stale model from an older image. This uses a new `dosesense-feedback-v2` volume. If upgrading an existing Docker demo, the old volume is retained but its feedback is not migrated automatically; copy the existing SQLite file into the new feedback volume before relying on prior entries. Docker is an alternate setup path; see [verification](docs/verification.md) for what was actually tested.

## Demonstration

Use the walkthrough to show:

1. Concern despite healthy dispensing coverage: explain the other recorded signals, without claiming confirmed missed doses.
2. Temporary irregularity: show the return toward baseline and its current assessment.
3. Insufficient evidence: contrast abstention with a well-supported low-concern case.
4. Clinician feedback and the patient portal: feedback is stored; it does not retrain the model.

Use live values rather than memorised patient IDs or probabilities. See [five-person script and presentation notes](docs/presentation.md), [business proposal](docs/business-plan.md) and [submission checklist](docs/submission-checklist.md).

## Project structure

| Path | Purpose |
|---|---|
| `dosesense/` | Simulator, features, ensemble, change detection, explanations and evaluation |
| `backend/app/main.py` | API and SQLite feedback/portal storage |
| `frontend/index.html` | Dashboard and patient portal; no frontend build step or CDN |
| `ml/` | Training and evaluation entry points |
| `scripts/` | Reproduction, generated reporting, benchmarking and demo checks |
| `tests/` | Python tests and JavaScript render harness |
| `docs/` | Architecture, methodology, results, limitations and submission preparation |

Interactive API documentation is at `/docs`. Key endpoints include `/api/health`, `/api/patients`, `/api/patients/{id}`, `/api/demo`, `/api/config` and `/api/metrics`.

## Limits

All performance is measured on our simulator; patient-level separation does not remove simulator bias. Disease progression can resemble medication-taking difficulty. Barrier rules and consequence weights are illustrative. The API has no authentication or real record connectors, and stored feedback does not yet recalibrate the model. No clinical outcomes, savings, customers or partnerships are claimed. Read [limitations](docs/limitations.md) before reusing the results.
