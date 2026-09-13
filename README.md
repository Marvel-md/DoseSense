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
and it is the case this project is named after. In our evaluation the PDC rule detects **8.4%** of
those genuinely non-adherent snapshots. DoseSense detects **32.2%** — still low, but nearly twice
as many on a pattern the standard metric structurally cannot address.

**It cannot tell a holiday from a decline.** PDC is a population threshold applied to an individual.
A 34-day gap is unremarkable for a patient who has always run late and is a red flag for one who has
never been more than a day out. On four behavioural patterns that are adherent for their entire
record by construction, the PDC rule raises a false alert on **6.8%** of snapshots. DoseSense raises
one on **3.4%**.

---

## "Without asking" is an architectural guarantee, not a slogan

The brief asks for detection *without relying on patients to report every missed
dose*. DoseSense includes a patient portal, and it would be reasonable to
suspect that undermines the premise. It does not, and the reason is enforced in
code rather than promised in prose.

**No self-reported data is ever a model input.** Self-reports are written to a
separate `self_report` table, are not read by `features.py`, do not appear in
`FEATURE_COLUMNS`, and cannot reach `model.predict`. Every write to the
self-report endpoint returns `used_for_prediction: false` so the guarantee is
visible at the API surface, and `test_self_reports_never_reach_the_model`
fails the build if a path ever opens.

The portal is a real screen, not a described one: a role selector in the top bar switches between
**Doctor workspace** and **Patient portal**, and the portal renders a person's own regimen, their
next repeat date, and a structured self-report form — dose recall, a how-are-you-feeling slider,
tap-to-select barrier tags, a free-text note and a date. `tests/render_harness.js` renders it
against live payloads and asserts no risk language appears on it.

So what is the portal *for*? Detection happens entirely from indirect signals.
The portal lets a patient **confirm, correct, or explain a hypothesis the system
already formed about them** — turning an inferred guess about cost into a stated
fact about cost. The clinician sees it as a fourth, separately-sourced kind of
evidence: what was recorded, what the model inferred, what it hypothesised, and
what the patient said.

**The portal also never shows the patient their own risk estimate.** No
probability, no priority tier, no barrier hypothesis. Telling someone there is a
78% chance they are not taking their medicine is the accusation this project
exists to avoid, and a figure derived from indirect signals is not a thing to
put in front of the person it is about. The clinician sees the estimate; the
patient sees their own record and an invitation to say how things are going.
`test_portal_never_exposes_the_estimate` enforces it.

---

## Results

On 148 patients held out **at the patient level** — no patient contributes rows to more than one
split, and calibration was fitted on a third disjoint set of patients.

| | ROC-AUC | PR-AUC | Precision | Recall | F1 | Brier |
|---|---|---|---|---|---|---|
| **DoseSense** | **0.904** | **0.873** | 0.864 | 0.738 | **0.796** | 0.109 |
| PDC / refill-gap rule | 0.786 | 0.691 | 0.786 | 0.676 | 0.727 | 0.239 |

- **PR-AUC +0.182** over the metric in current clinical use
- **False alerts down 37%** — 4.5 vs 7.1 per 100 patient-months, at comparable total alert volume
- **Expected calibration error 0.015**, so a stated 70% means roughly seventy in a hundred
- **1.8× detection** on silent non-adherence, the case PDC cannot see
- **50% fewer false alerts** on adversarially-constructed adherent patients

Full tables, calibration curve, ablation, subgroup analysis and operating-point sweep:
**[`docs/evaluation.md`](docs/evaluation.md)** — generated directly from `artifacts/metrics.json` by
a script. No figure in this repository is transcribed by hand.

### Behaviour on each pattern, pooled across seven seeds

A rare archetype contributes only ~20 snapshots from two patients in any single
run, so a rate from one seed says nothing. These are pooled counts across seven
independent cohorts, with a Wilson 95% interval on the model's rate.

| Pattern | Snapshots | True rate | DoseSense | 95% CI | PDC rule |
|---|---|---|---|---|---|
| Complex regimen | 182 | 0.88 | **0.81** | 0.74–0.86 | 0.70 |
| **Silent non-adherence** | 364 | 0.80 | **0.27** | 0.22–0.31 | 0.03 |
| Repeated refill delay | 546 | 0.77 | **0.67** | 0.63–0.71 | 0.65 |
| Access barrier | 455 | 0.76 | **0.64** | 0.60–0.69 | 0.63 |
| Gradual decline | 546 | 0.75 | **0.67** | 0.63–0.71 | 0.65 |
| Side effect | 364 | 0.70 | **0.64** | 0.59–0.69 | 0.58 |
| Cost barrier | 273 | 0.70 | **0.55** | 0.49–0.61 | 0.52 |
| Temporary interruption | 546 | 0.10 | **0.19** | 0.16–0.23 | 0.36 |
| Occasional irregularity | 728 | 0.00 | **0.07** | 0.05–0.09 | 0.12 |
| Adherent, stable | 1456 | 0.00 | **0.04** | 0.03–0.05 | 0.06 |
| Misleading anomaly | 91 | 0.00 | **0.18** | 0.11–0.27 | 0.30 |
| Disease progression | 182 | 0.00 | 0.10 | 0.06–0.15 | **0.06** |

For the seven mostly-non-adherent patterns, higher is better. For the five
mostly-adherent ones, lower is better. DoseSense wins on eleven of twelve.

**It loses on one, and the reason is structural.** `DISEASE_PROGRESSION` is a
patient whose condition worsens while their dispensing record stays flawless.
PDC only reads dispensing records, so it almost never fires — 0.06. DoseSense
reads laboratory drift and symptoms too, which *are* moving, so it fires at 0.10.

This is the direct cost of the thing that makes the system work. The extra
signals are exactly what catch silent non-adherence (0.27 against PDC's 0.03,
a nine-fold difference); the same signals are what make progression harder to
dismiss. You cannot have one without the other, and the honest framing is a
trade: roughly four extra false alerts per hundred progression snapshots, in
exchange for catching a pattern the conventional metric is blind to.

### Making the baseline harder to beat

PDC is conventionally measured over 90 days. Once the simulated panel spanned 15- to 90-day supply
periods, that window quietly became unfair to long-supply patients: a single fill falling just
outside a 90-day window collapses the score, and the baseline was false-alarming on **30% of
maintenance-supply patients against 11% of standard ones**. That disparity was an artefact we had
introduced, and it was inflating our own improvement.

The comparator now gets a window sized to each patient's own supply period, and the seven-day gap
rule scales with it too. The maintenance false-alarm rate fell to 22.5%, and our reported advantage
fell with it. That is the correct direction: an improvement measured against a crippled baseline is
not an improvement.

### Four refinements, and one rejection

**Adaptive dispensing intervals.** A real panel does not run on a single supply period: acute
courses come in 15 days, maintenance therapy in 90. A fixed "more than seven days late" rule is
simultaneously too sensitive for one and useless for the other. Every lateness measure is now also
expressed as a fraction of that patient's own expected interval, and the change detector's clinical
floor scales with it — `max(3.5 days, 12% of the supply period)`. The simulator was extended to
generate the full 15-to-90-day range so the feature has something real to adapt to.

**Coverage gaps are drawn, not just implied.** The clinician timeline now shows stretches where
*nothing was recorded* as hatched bands, and a strip of channel-coverage indicators sits above the
assessment. A quiet record and a stable patient look identical until you can see which channels
were actually measured, and that ambiguity is what most low-confidence scores are really about.

**Barriers now carry concrete next steps.** `barriers.py` maps each hypothesis to ordered workflow
actions with an effort level and an owner — check local stock, offer delivery, synchronise repeats,
check subsidy eligibility, book a structured medication review. None of them alters a prescription:
every one is a conversation, a check or a referral, because deciding what a patient should take is
a clinician's job. Where two hypotheses compete, the cheapest step from each is offered rather than
committing a clinician's time to the wrong one.

**Equity audited by data richness, not just demographics.** The more pressing fairness question
here is not whether the model treats men and women differently — it is whether it treats *thinly
recorded* patients differently, since record richness tracks who has time, transport and money. See
the audit below.

**Cross-signal velocity: built, measured, rejected.** The idea was to catch declining trajectories
earlier by correlating the rate of clinical drift against the rate of refill drift — the signature
of silent non-adherence being deterioration without any collection signal. Across three seeds it
moved PR-AUC by +0.0007 (noise), silent-non-adherence recall by +0.011, and disease-progression
false alerts by **+0.025 — worse**. It helps the case it was designed for slightly and makes the
confounder meaningfully worse. It is kept in `features.py` as `EXPERIMENTAL_FAMILIES`, excluded
from the shipped model, and still reported in the ablation table, because a measured negative
result is worth more than a deleted branch.

### Equity across data-richness strata

Snapshots are stratified by observable coverage only — fills, appointments, laboratory results,
wearable presence — so the same labelling could be applied in deployment where no ground truth
exists. The question is whether abstention *protects* thin records or quietly *hides failures* on
them.

| Stratum | Snapshots | Concern rate | Abstains | Alerts | Recall where it commits |
|---|---|---|---|---|---|
| sparse | 324 | 0.37 | **74.4%** | 0.105 | **0.882** |
| partial | 294 | 0.32 | 0.0% | 0.272 | 0.699 |
| good | 179 | 0.49 | 0.0% | 0.430 | 0.830 |
| rich | 1127 | 0.38 | 0.0% | 0.382 | 0.825 |

Abstention is doing its job: on the thinnest records the system declines to commit three times in
four, and where it does commit its recall is the *highest* of any stratum. The number that would
have indicated a hidden problem — committed-recall falling as records thin — does the opposite.

The weakest stratum is `partial`, not `sparse`: enough record to trigger a commitment, not enough
to be reliable. That is the honest gap (0.183 recall spread), and it suggests the abstention
threshold is currently set slightly too low rather than too high.

### What did not work as hoped

Two results are weaker than the pitch would like, and both are in the docs rather than buried.

**Detection lead time.** Median time to detection is one snapshot interval for both methods.
DoseSense does *not* beat PDC on the median. It does catch a larger share of cases at or before the
onset snapshot (24.3% vs 14.7%) with a third of the false alerts, but the honest claim is precision
and hard-case coverage, not speed.

**One archetype is worse than the baseline.** Disease progression, for the
reason above. It is a genuine weakness and it is reported rather than omitted.

**Dispensing data alone carries most of the signal.** The ablation puts refill-only PR-AUC at 0.834 against 0.871 for all seven signal families. Multi-signal fusion is worth +0.037 — real and
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

**Benchmark** — seed stability, latency and cohort scaling:

```bash
python scripts/benchmark.py                    # ~2 minutes, writes artifacts/benchmark.json
python scripts/benchmark.py --seeds 3 --quick  # faster sanity check
BENCHMARK=1 bash scripts/reproduce.sh          # include it in the full run
```

The stability run is the answer to "your test set is only 148 patients". Five independent
seeds — fresh cohort, fresh split, fresh model each time — give PR-AUC **0.9019 ± 0.0048**
against a baseline of **0.7064 ± 0.0096**. The model beat the baseline on **5 of 5 seeds**,
worst-case advantage **+0.1849 PR-AUC**. The headline figure is not a lucky split.

Latency: **54 ms** per patient for a complete assessment including SHAP, **12,500**
predictions/second in batch. Rescoring a 10,000-patient panel projects to about **9 minutes**.

Cohort scaling is flat — PR-AUC moves −0.0026 going from 150 to 600 patients — so the method
is not data-starved at this scale and more synthetic data would not improve the result.

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

A patient opens as a single column you read top to bottom, like a note a colleague left on your
desk. The one structural device carries the brief's core requirement: three kinds of statement get
left rules that differ in style rather than colour, so they can never be mistaken for one another.

| | Meaning |
|---|---|
| solid rule | **What the record shows** — recorded in the health system, with its measured value |
| dashed rule | **What DoseSense estimates** — the figure, its uncertainty, its attribution |
| dotted rule, italic | **A possible reason** — a hypothesis, explicitly not a finding |

Because the distinction is structural rather than chromatic, it survives greyscale printing and
does not spend a colour. Colour itself does exactly one job in the interface: clay means this needs
attention, sage means it has settled. Nothing else is coloured.

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
