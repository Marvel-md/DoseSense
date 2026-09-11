# DoseSense

**Detecting possible medication non-adherence from routine care records — without asking the patient.**

A patient can be on exactly the right medication and the treatment still fails, because doses are
missed, taken at the wrong time, or quietly stopped. Between one appointment and the next, a
clinician usually has no visibility into what actually happened. The clues exist — pharmacy
dispensing records, prescription changes, symptom questionnaires, laboratory drift, missed
follow-ups — but they sit in different systems and are rarely read together.

DoseSense reads them together. It estimates the probability that a patient is experiencing a
medication-taking problem, distinguishes a temporary irregularity from a sustained pattern, offers a
hypothesis about *why*, says how much it trusts its own estimate, and orders the result as a
worklist a pharmacist can actually work through.

It is a prioritisation and decision-support tool. It is not a diagnostic device, it never asserts
that a dose was missed, and it never labels a patient.

---

## The problem with the metric everyone currently uses

Health systems measure adherence with **Proportion of Days Covered** — the share of days in a
window covered by a dispensed supply, with 0.80 as the conventional cut-off for "adherent". It is
computed from pharmacy records, which creates two blind spots that no amount of threshold tuning
fixes:

**It cannot see the patient who collects and doesn't take.** PDC measures *collection*, not
ingestion. A patient who picks up every prescription on schedule and takes none of it scores near
1.0. This is not a hypothetical — it is the single most common form of intermittent non-adherence,
and it is the case this project is named after. In our evaluation the PDC rule detects **10.5%** of
those genuinely non-adherent snapshots. DoseSense detects **18.9%** — still low, but nearly twice
as many on a pattern the standard metric structurally cannot address.

**It cannot tell a holiday from a decline.** PDC is a population threshold applied to an individual.
A 34-day gap is unremarkable for a patient who has always run late and is a red flag for one who has
never been more than a day out. On four behavioural patterns that are adherent for their entire
record by construction, the PDC rule raises a false alert on **6.8%** of snapshots. DoseSense raises
one on **3.4%**.

---

## Results

On 148 patients held out **at the patient level** — no patient contributes rows to more than one
split, and calibration was fitted on a third disjoint set of patients.

| | ROC-AUC | PR-AUC | Precision | Recall | F1 | Brier |
|---|---|---|---|---|---|---|
| **DoseSense** | **0.924** | **0.894** | 0.927 | 0.745 | **0.826** | 0.095 |
| PDC / refill-gap rule | 0.807 | 0.723 | 0.810 | 0.699 | 0.750 | 0.227 |

- **PR-AUC +0.172** over the metric in current clinical use
- **False alerts down 64.5%** — 2.2 vs 6.3 per 100 patient-months, at comparable total alert volume
- **Expected calibration error 0.024**, so a stated 70% means roughly seventy in a hundred
- **1.8× detection** on silent non-adherence, the case PDC cannot see
- **50% fewer false alerts** on adversarially-constructed adherent patients

Full tables, calibration curve, ablation, subgroup analysis and operating-point sweep:
**[`docs/evaluation.md`](docs/evaluation.md)** — generated directly from `artifacts/metrics.json` by
a script. No figure in this repository is transcribed by hand.

### What did not work as hoped

Two results are weaker than the pitch would like, and both are in the docs rather than buried.

**Detection lead time.** Median time to detection is one snapshot interval for both methods.
DoseSense does *not* beat PDC on the median. It does catch a larger share of cases at or before the
onset snapshot (24.3% vs 14.7%) with a third of the false alerts, but the honest claim is precision
and hard-case coverage, not speed.

**Dispensing data alone carries most of the signal.** The ablation puts refill-only PR-AUC at 0.873
against 0.907 for all seven signal families. Multi-signal fusion is worth +0.034 — real and
monotone, with the largest jump from adding symptoms, but not the transformation a slide deck would
claim. The extra signals earn their place by reaching cases refill data cannot and by enabling
barrier inference, not by dramatically lifting AUC. We found this out because the ablation
contradicted our own hypothesis, which is the reason it is in the pipeline.

---

## How it works

```
Pharmacy · prescriptions · appointments · symptoms · labs · activity
                              │
                    normalise to patient timeline
                              │
              ┌───────────────┴───────────────┐
      personal baseline              causal feature windows
   (this patient's own history)      (trailing only, never future)
              └───────────────┬───────────────┘
                              │
                   change-point detection
              stable / temporary / persistent
                              │
                  bagged calibrated ensemble
                   probability + disagreement
                              │
         ┌────────────────────┼────────────────────┐
   uncertainty          barrier inference     consequence
   & abstention         (rules, abstains)     (declared policy)
         └────────────────────┼────────────────────┘
                              │
              priority = likelihood × consequence × confidence
                              │
                    explainable alert → clinician
                              │
                      feedback → recalibration
```

### The five ideas that matter

**Personal baselines, not population thresholds.** Every feature is expressed against the patient's
own history. Adding a clinical effect floor to the change detector — a shift must exceed 3.5 days in
absolute terms, not just be statistically detectable — cut false positives on genuinely stable
patients from **18% to 0.8%**. Statistical detectability and clinical meaning are different tests
and the code applies both.

**Temporary versus persistent as an explicit output.** After finding a change point, the detector
tests whether the level came back. A shift found too recently to judge is marked `provisional` and
the confidence layer downgrades it, rather than the detector guessing.

**Abstention as a real state.** Where evidence is too thin, the system says so instead of producing
a number that looks like knowledge. Critically, *"no evidence of a concern"* and *"not enough
evidence to tell"* are kept separate: a patient with eighteen months of clean records is a confident
negative, not an unknown. Conflating those two was a bug we shipped and then caught — it marked 51%
of the panel unknown, which would have trained the care team to ignore the column.

**Priority is not probability.** An 88% concern on levothyroxine should not outrank a 61% concern on
warfarin. Priority is `likelihood × clinical consequence × confidence`, and the consequence weight
comes from a declared policy table in `dosesense/config.py` — auditable and changeable by a care
team without retraining anything.

**Barriers, not blame.** The output is not "non-adherent" but a hypothesis: access, cost,
tolerability, regimen complexity, care disengagement — or explicitly *none*, when nothing points
clearly. Roughly a third of alerts return no barrier. That is the system declining to guess, because
a clinician acting on a wrong reason asks the wrong question and may close the case.

---

## Setup

```bash
git clone <this-repo> && cd dose-sense
pip install -r requirements.txt

python scripts/generate_data.py --patients 600   # ~2s
python ml/train.py                               # ~25s
python ml/evaluate.py                            # ~60s, writes artifacts/metrics.json
python scripts/make_report.py                    # regenerates docs/evaluation.md

uvicorn backend.app.main:app --reload
# open http://127.0.0.1:8000
```

Or all of it at once:

```bash
bash scripts/reproduce.sh
```

**Docker:**

```bash
docker compose up --build   # http://localhost:8000
```

**Faster start for a live demo.** Scoring 600 patients takes about 35 seconds at boot.
`DOSESENSE_PATIENT_LIMIT=120 uvicorn backend.app.main:app` starts in about 9 seconds and shows
identical behaviour. Evaluation always runs on the full cohort.

**Tests:**

```bash
python -m pytest tests/ -q          # 67 tests
node tests/render_harness.js        # dashboard render checks (needs the API running)
```

---

## Demo walkthrough

The dashboard has a **Walkthrough** panel that selects cases from the live scored cohort *by
behaviour* rather than by hard-coded ID, so it stays valid if you regenerate the data. Eight slots,
each showing something different:

1. **Sustained concern, high consequence** — several signal families converging on a patient where a
   gap matters most. Priority CRITICAL.
2. **Dispensing looks perfect, clinical picture does not** — the money slide. PDC is healthy,
   DoseSense flags a concern because the clinical signals moved while the pharmacy record stayed
   immaculate.
3. **Temporary irregularity, deliberately not escalated** — one departure from baseline that
   returned. The system names it and stands down. This is the requirement to distinguish temporary
   from meaningful, made visible.
4. **Access hypothesis** — barrier inference with the evidence it rests on.
5. **Tolerability hypothesis** — symptoms rose after a regimen change, then collection became less
   timely.
6. **Concern with no reason offered** — evidence supports a concern, nothing points to a cause, the
   system declines to invent one.
7. **Insufficient evidence** — a short or sparse record. Abstention.
8. **Confident negative** — a long complete record with nothing worrying in it. Not the same state
   as "we cannot tell".

### Reading the interface

The dashboard's one structural device carries the brief's core requirement. Three kinds of
statement are given visually distinct left rules, so they can never be mistaken for one another:

| | Meaning |
|---|---|
| solid ink rule | **Observed** — recorded in the health system, with its measured value |
| dashed violet rule | **Inferred** — DoseSense's estimate, its uncertainty, its attribution |
| dotted violet rule, italic | **Hypothesis** — a possible reason, explicitly not a finding |

It survives greyscale printing, which a colour-coded badge does not.

---

## Data

**All synthetic. No real patient record is used anywhere in this project.**

`dosesense/datagen.py` is a causal simulator, not a table of random numbers. For each patient it
samples a hidden day-by-day adherence state trajectory, then generates the observable record
conditioned on it. The hidden trajectory is written to a separate file, is loaded only by the
evaluation path, and is never available to the feature pipeline or the API — enforced by a test.

Twelve behavioural archetypes, including four adherent-by-construction and two deliberately
adversarial:

| Archetype | What it tests |
|---|---|
| `ADHERENT_STABLE` | baseline specificity |
| `OCCASIONAL_IRREGULARITY` | brief self-correcting dips must not alert |
| `TEMPORARY_INTERRUPTION` | a real lapse that resolves — alert, then stand down |
| `GRADUAL_DECLINE` | slow drift detection |
| `REPEATED_REFILL_DELAY` | the straightforward case |
| `ACCESS_BARRIER` | barrier inference, access |
| `COST_BARRIER` | barrier inference, cost |
| `SIDE_EFFECT` | barrier inference, tolerability |
| `COMPLEX_REGIMEN` | barrier inference, complexity |
| **`SILENT_NONADHERENCE`** | **collects on time, doesn't take — PDC is blind** |
| **`MISLEADING_ANOMALY`** | **one dramatic gap, fully adherent — must not alert** |
| **`DISEASE_PROGRESSION`** | **deteriorates clinically, dispensing flawless — must not alert** |

The last two exist because a system that cannot separate *"the treatment isn't working"* from
*"the treatment isn't being taken"* sends the care team after the wrong problem.

Record richness varies the way a real panel does: about a quarter of patients enrolled part-way
through, a fifth never complete symptom questionnaires, some attend laboratory monitoring far less
often than protocol. This is the main source of genuine uncertainty in the system, and without it
the confidence layer was decorative — 93% of patients came out "High".

---

## Model

Bagged LightGBM over 47 engineered features in seven families, isotonic-calibrated on a held-out
patient split.

Gradient-boosted trees rather than a sequence model, because the constraints favour them on every
axis that matters: a few thousand patient-snapshots of tabular data with heavy missingness, a hard
requirement that every alert be explainable to a pharmacist, and inference fast enough to rescore a
panel on a laptop. A sequence model would need an order of magnitude more patients to beat this and
would make the explanation layer a research problem of its own.

**Bagging does double duty.** Five models are trained on bootstrap resamples of distinct *patients*
(not rows — neighbouring snapshots of one person would otherwise land in and out of the same bag and
understate real disagreement). Their spread is an honest signal about how much the estimate depends
on who happened to be in the training set, and it feeds the confidence assessment. The system can
say *"the models disagree about this person"*.

**Excluded from the feature matrix, deliberately:** sex and insurance tier (protected or
near-protected), distance to pharmacy (a plausible access signal but also a socioeconomic proxy —
used only as evidence shown to a human in barrier inference, where it is visible), and clinical
consequence (severity is declared policy applied in the priority score, not a learned correlate of
behaviour).

---

## API

Interactive docs at `/docs` when running. `GET /api/config` exposes the declared policy behind every
score, so the consequence table and thresholds can be inspected rather than taken on trust.

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | readiness, patient count, disclaimer |
| `GET /api/overview` | cohort counts, state mix, barrier mix, feedback summary |
| `GET /api/patients` | the priority worklist, filterable by state, tier, barrier, search |
| `GET /api/patients/{id}` | full assessment, trajectory, timeline, raw series |
| `GET /api/patients/{id}/timeline` | evidence timeline |
| `GET /api/patients/{id}/risk` | probability, uncertainty, hypothesis, priority |
| `GET /api/alerts` | alerting patients only |
| `POST /api/alerts/{id}/feedback` | clinician adjudication → SQLite |
| `GET /api/metrics` | the full evaluation output |
| `GET /api/demo` | behaviour-selected walkthrough cases |
| `GET /api/config` | declared clinical policy and thresholds |

Every assessment is served with the three-way separation enforced in the schema: `observed`,
`inferred`, `hypothesis`, `action`.

---

## Repository

```
dose-sense/
├── dosesense/            core engine (importable, no web or CLI dependencies)
│   ├── config.py          declared clinical policy, kept out of the ML code
│   ├── datagen.py         causal simulator, 12 archetypes, hidden ground truth
│   ├── features.py        47 causal features in 7 families
│   ├── changepoint.py     personal-baseline segmentation, temporary vs persistent
│   ├── baseline.py        faithful PDC / MPR comparator
│   ├── panel.py           snapshot panel, labelling, patient-level splits
│   ├── model.py           bagged calibrated ensemble
│   ├── barriers.py        scored barrier hypotheses with abstention
│   ├── risk.py            uncertainty, abstention, priority
│   ├── explain.py         SHAP grouped by family, plain-language names
│   ├── evaluate.py        metrics, calibration, lead time, ablation, subgroups
│   └── pipeline.py        assessment assembly and serving engine
├── backend/app/main.py   FastAPI, feedback store
├── frontend/index.html   zero-build dashboard, hand-rolled SVG charts
├── ml/                   train.py, evaluate.py
├── scripts/              generate_data.py, make_report.py, reproduce.sh
├── tests/                67 pytest tests + JS render harness
└── docs/                 architecture, methodology, evaluation, limitations, presentation
```

### A note on the frontend

The dashboard is vanilla JS with hand-written SVG charts and **no build step, no npm install and no
CDN**. This is a deliberate trade against the more conventional React + TypeScript choice: a
demonstration venue's network fails, and an interface that renders from a single file with zero
external requests is worth more than a nicer component model nobody will inspect. Charts are a few
lines of arithmetic each.

Because there is no compiler to catch a typo in a template literal, `tests/render_harness.js` loads
the real script into a minimal fake DOM and renders every assessment state against live API
payloads, plus a degenerate patient with no plottable data. It fails on any undefined property or
leaked placeholder.

---

## Limitations

Read **[`docs/limitations.md`](docs/limitations.md)** before believing any number here. In short:

- **Synthetic data throughout.** Performance is measured against our own simulator's ground truth.
  It establishes that the method works on data with the structure we believe real data has. It does
  not establish real-world performance, and no claim here should be read as clinical validation.
- **The largest subgroup disparities are by age band and condition**, not sex. We tested removing
  age from the model: it cost 0.003 PR-AUC and did not close the gap, so the disparity is not the
  model reading age directly. It is unaddressed.
- **Barrier inference is expert rules, not a learned model**, because no ground truth for barriers
  exists in a real record. A supervised model there would be fitting our own generative assumptions
  and reporting its accuracy as if it meant something.
- **No prospective validation, no clinical outcomes, no cost savings are claimed**, because none
  have been measured.
- The test set is 148 patients. Small-cohort variance is real and metrics move with the seed.

## Responsible use

The system is built so that a clinician always sees observation before inference, and inference
before hypothesis. It uses the language of possibility throughout — "evidence suggests", "possible
concern", "requires review" — and the test suite fails the build if prohibited phrasing
(`non-compliant`, `did not take`, `confirmed non-adherence`) appears anywhere in any output.

The purpose is to help a care team decide **who to talk to and what to ask about**. Finding a
barrier is more useful than assigning fault, and a patient who is struggling to afford or obtain
their medication is not a patient who has failed.
