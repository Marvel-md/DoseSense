# Presentation

Twelve slides, a 3-minute pitch script and a 5-minute technical script. Every figure below appears
in `artifacts/metrics.json` and can be regenerated with `bash scripts/reproduce.sh`.

---

## Slides

### 1 — The gap between prescription and outcome

A patient can be on exactly the right medication and the treatment still fails. Between one
appointment and the next, a clinician has almost no visibility into what actually happened.

*Visual:* a timeline from prescription to next appointment with a blank space in the middle.
One line of text. Nothing else.

### 2 — Why the current metric misses it

Health systems measure adherence with Proportion of Days Covered, computed from pharmacy
dispensing records. Two structural blind spots:

- **It measures collection, not ingestion.** A patient who collects every prescription and takes
  none of it scores near 1.0.
- **It is a population threshold applied to an individual.** A 34-day gap means nothing for a
  patient who always runs late, and is a red flag for one who never does.

*Visual:* two patient records side by side, identical PDC of 0.94, radically different situations.

### 3 — What we built

DoseSense reads six signal families together, estimates the probability of a medication-taking
problem, separates a temporary irregularity from a sustained pattern, offers a hypothesis about
*why*, states how much it trusts itself, and hands a clinician a ranked worklist.

A prioritisation tool. Not a diagnostic device. It never asserts a dose was missed.

*Visual:* the pipeline diagram from `docs/architecture.md`, simplified to seven boxes.

### 4 — Personal baselines, not population thresholds

Every feature is measured against the patient's own history. Then a change-point detector asks
whether a shift happened and, critically, whether it **came back**.

Adding a clinical effect floor — a shift must exceed 3.5 days in absolute terms, not merely be
statistically significant — cut false positives on genuinely stable patients from **18% to 0.8%**.

*Visual:* one patient's refill-lateness series with the detected change point marked, next to a
second patient whose single spike reverted.

### 5 — The demo: dispensing looks perfect, the patient does not

**PDC 0.94. Every prescription collected on time. DoseSense flags a possible concern at 71%.**

Because HbA1c has drifted 2.4 standard deviations from this patient's own earlier range, symptom
burden has risen, and a follow-up was missed — while the pharmacy record stayed immaculate.

This is the case the standard metric structurally cannot see. PDC detects **10.5%** of them.
DoseSense detects **18.9%**.

*Visual:* live dashboard, patient detail view. This is the centrepiece — spend time here.

### 6 — And the demo it *doesn't* fire on

The same system, a different patient. One 24-day refill gap, a missed appointment, an off-trend
laboratory result. Dramatic-looking record. Fully adherent.

> *"A departure from this patient's baseline was detected and has since returned. No sustained
> pattern is evident."*

On four patterns that are adherent by construction, the PDC rule false-alarms on **6.8%** of
snapshots. DoseSense on **3.4%**.

*Visual:* the same interface, the calm teal verdict block. The contrast with slide 5 is the point.

### 7 — Saying "I don't know" as a designed behaviour

Three separate states, which most systems collapse into one:

| State | Meaning |
|---|---|
| Confident negative | Long complete record, nothing worrying. We know a lot and it's fine. |
| Insufficient evidence | Sparse record, or an elevated estimate resting on one signal. We abstain. |
| Sustained concern | Multiple families corroborate. Review requested. |

Confidence combines how many independent signal families carry evidence, how much a five-model
ensemble disagrees about *this specific patient*, and how complete the record is.

*Visual:* three worklist rows, one per state, with the same probability and different confidence.

### 8 — Priority is not probability

```
priority = likelihood  ×  clinical consequence  ×  confidence
```

An 88% concern on levothyroxine should not outrank a 61% concern on warfarin.

The consequence weight is **declared policy** in a config table, exposed over the API — not
learned. A care team can change what the system cares about without retraining anything.

*Visual:* two rows where sorting by probability and sorting by priority give opposite orders.

### 9 — Barriers, not blame

The output is not a label. It is a hypothesis with the evidence behind it: access, cost,
tolerability, regimen complexity, care disengagement — or **explicitly none**.

About a third of alerts return no barrier. That is the system declining to guess, because a
clinician who acts on a wrong reason asks the wrong question and may close the case.

*Visual:* the dotted-rule hypothesis block from the interface, evidence bullets visible.

### 10 — Results, including what didn't work

| | PR-AUC | F1 | False alerts / 100 patient-months |
|---|---|---|---|
| **DoseSense** | **0.894** | **0.826** | **2.2** |
| PDC rule | 0.723 | 0.750 | 6.3 |

Calibration error 0.024. Held out at the patient level; calibrated on a third disjoint split.

**Two things did not work as hoped:**

- Detection lead time does **not** beat the baseline on the median. Both detect within one 30-day
  rescoring interval. We catch more cases at or before onset, with a third of the false alerts —
  but "detects sooner" is not a claim the data supports.
- Dispensing data alone reaches PR-AUC 0.873; all seven families reach 0.907. Fusion is worth
  +0.034 — real and monotone, not a transformation.

*Visual:* the comparison table, then the ablation table directly beneath it. Do not hide the second.

### 11 — How we found that out

Our first ablation showed multi-signal fusion added **nothing**: 0.9561 refill-only versus 0.9572
all-signals.

The fault was our simulator, not the model. We had made non-adherence always show up as late
refills — which makes the pharmacy record a near-sufficient statistic and is exactly the assumption
the real problem violates. We added the silent-non-adherence case, and the ablation became honest.

The ablation is in the pipeline because it overruled us once.

*Visual:* the two ablation tables, before and after, side by side.

### 12 — What it would take to be real

**Built:** a causally clean pipeline, a faithful baseline, calibrated uncertainty, abstention,
barrier inference, 67 tests, a report generated from metrics so no number is hand-typed.

**Not built, and not claimed:** prospective validation on real records, clinical governance,
authentication and audit, group-wise fairness calibration, and an answer to the non-stationarity
the system's own alerts would introduce.

Where it could go: hospital chains and chronic-care clinics, payers, and pharmacy-led adherence
programmes — all of whom already fund this work manually. The same engine transfers to medicine
supply-shortage detection with the domain logic swapped.

*Visual:* two columns, built and not built. End on the second.

---

## 3-minute pitch

> A patient can be on exactly the right medication, and the treatment still fails — because the
> doses aren't being taken. Between one appointment and the next, a clinician has almost no
> visibility into that.
>
> The way health systems measure this today is Proportion of Days Covered: what share of days were
> covered by a prescription the patient collected. It has two blind spots that no amount of
> threshold tuning fixes.
>
> It measures collection, not ingestion. A patient who picks up every prescription on time and
> takes none of it scores a perfect 1.0. And it applies a population threshold to an individual — a
> month-long gap means nothing for someone who always runs late, and is serious for someone who
> never does.
>
> **[dashboard]** This is DoseSense. It reads six signal families together — pharmacy, prescriptions,
> appointments, symptoms, labs, activity — and gives a care team a ranked worklist.
>
> **[open patient, slide 5]** Look at this patient. PDC is 0.94. Every prescription collected on
> schedule. By the standard metric they are exemplary. DoseSense flags a possible concern, because
> their HbA1c has drifted from their own earlier range, symptoms have risen, and a follow-up was
> missed — while the pharmacy record stayed immaculate. This is the case the current metric
> structurally cannot see.
>
> **[open second patient, slide 6]** Now this one. A 24-day refill gap, a missed appointment, an
> off-trend lab result. A far more dramatic-looking record. The system says: *a departure from this
> patient's baseline was detected and has since returned; no sustained pattern is evident.* It
> stands down. Separating those two cases is the entire problem.
>
> Three numbers. Against the metric in clinical use we improve PR-AUC from 0.72 to 0.89, we cut
> false alerts by 65%, and our probabilities are calibrated to within 0.024 — which matters,
> because we multiply that probability by clinical severity to set the work order. A 61% concern on
> warfarin outranks an 88% concern on a thyroid supplement.
>
> Two things didn't work. We don't detect earlier than the baseline on the median. And refill data
> alone carries most of the signal — fusion is worth 0.034 PR-AUC, not a transformation. Both are
> in the docs, because we found the second one when our own ablation contradicted our pitch.
>
> And the output is never a label. It's a hypothesis about *why* — access, cost, tolerability — or
> explicitly nothing, when nothing points clearly. Because a patient who can't afford or can't
> obtain their medication hasn't failed. Finding the barrier is more useful than assigning fault.

---

## 5-minute technical presentation

**Framing (30s).** Unit of prediction is a patient-snapshot: every patient rescored every 30 days,
judged on the trailing window only. Non-adherence is a state that comes and goes, not a property of
a person — so patients who lapse and recover are positive during the lapse and negative after. This
also makes false-alerts-per-patient-month and lead time computable, which one-row-per-patient
cannot.

**Data (60s).** Fully synthetic causal simulator. Sample a hidden day-by-day adherence trajectory,
then generate the observable record conditioned on it. Hidden trajectory in a separate file, loaded
only by evaluation, stripped from every API payload, enforced by a test.

Twelve archetypes, four adherent-by-construction, two adversarial: `MISLEADING_ANOMALY` produces a
dramatic record while fully adherent; `DISEASE_PROGRESSION` deteriorates clinically with a flawless
dispensing record. The second exists so the model must distinguish *"the treatment isn't working"*
from *"the treatment isn't being taken"* — confuse those and you send the care team after the wrong
problem.

Record richness varies as a real panel does: a quarter enrolled part-way through, a fifth never
complete symptom questionnaires. Before we added that, 93% of patients came out High confidence and
the uncertainty layer was decorative.

**Causality (45s).** Every window trailing. Enforced structurally, verified by a test that
recomputes each snapshot against a record physically truncated at *t* and asserts all 47 features
are identical. Splits are by patient, never by row — adjacent snapshots share almost all their
history. Leakage here is invisible in the metrics, which is why it gets a dedicated test rather than
a comment.

**Change detection (60s).** Exact binary segmentation, Gaussian mean-shift cost, BIC-style penalty.
Not PELT: on twelve-to-eighteen-point series it buys nothing and costs the interpretability the
alert copy depends on.

Two fixes worth mentioning. First, we required a shift of 1.1 baseline SDs — but refill delays are
roughly exponential, so SD ≈ mean and a shift from 1 to 3 days late cleared it. Adding an absolute
clinical floor of 3.5 days cut false positives on stable patients from 18% to 0.8%. Statistical
detectability and clinical meaning are different tests. Second, our `provisional` flag for
too-recent shifts was dead code — the search range guaranteed three points after any change point.
Fixing it exposed a degenerate case where a single-point segment has zero variance and always wins
the split, so segment variance is now floored against the whole series.

**Model (45s).** Bagged LightGBM, 47 features, isotonic-calibrated on a disjoint patient split.
Trees not sequences: a few thousand tabular snapshots with heavy missingness, a hard explainability
requirement, laptop-speed inference.

Bagging is for uncertainty, not accuracy. Five models on bootstrap resamples of distinct *patients* —
not rows, or neighbouring snapshots of one person land in and out of the same bag and understate
disagreement. The spread feeds confidence, so the system can say the models disagree about this
person.

Excluded deliberately: sex and insurance tier as protected; distance to pharmacy as a socioeconomic
proxy, used only as evidence shown to a human in barrier inference; consequence as declared policy
rather than a learned correlate.

**Uncertainty (45s).** Our first version abstained whenever fewer than two families carried
evidence — marking 51% of the panel unknown, including patients with eighteen months of clean
records. The error was conceptual: *no evidence of a concern* and *not enough evidence to tell* are
different states, and a queue half-filled with "unknown" teaches the team to ignore the column.
Abstention is now reserved for a record too sparse for a baseline, or an elevated estimate on a
single family. 3% of the panel.

**Evaluation (75s).** PR-AUC 0.894 vs 0.723. F1 0.826 vs 0.750. False alerts 2.2 vs 6.3 per 100
patient-months, a 65% reduction. ECE 0.024.

The per-archetype table is where the brief is actually tested — four archetypes are negative by
construction, so the alert rate on them is a false-positive rate against a known adversarial truth,
directly comparable to the PDC rule on the same rows. 3.4% vs 6.8%.

Two honest weaknesses. Median lead time is one snapshot interval for both methods — we do not detect
earlier, we detect more precisely and cover harder cases. And the ablation puts refill-only at 0.873
against 0.907 for everything, so fusion is +0.034.

That ablation is the reason two archetypes exist. The first version showed 0.9561 versus 0.9572 —
fusion added nothing — because we'd made non-adherence always express as late refills, which makes
the pharmacy record a near-sufficient statistic and is precisely the assumption the real problem
violates. We added silent non-adherence and the result became honest.

**Engineering (30s).** 67 tests including the leakage check, a hand-worked PDC verification so the
baseline is the real metric, and a language guard that fails the build if "non-compliant" or "did
not take" appears anywhere in any output. The dashboard has no build step and no CDN, so a JS render
harness loads it into a fake DOM and renders every assessment state against live payloads.
`docs/evaluation.md` is generated from `metrics.json` by a script — no number in the repository is
transcribed by hand.

---

## Questions we expect

**"Isn't this just PDC with extra steps?"** PDC is implemented in the repo and hand-verified in the
tests, and we beat it by 0.17 PR-AUC with 65% fewer false alerts. More to the point, PDC is
structurally blind to the patient who collects and doesn't take — 10.5% detection against our 18.9%.

**"Your data is synthetic, so what does the number mean?"** That the method works on data with the
structure we believe real data has, and nothing more. `docs/limitations.md` lists six specific ways
the simulator is probably wrong. The circularity risk is real and we mitigated it with adversarial
archetypes, a confounder the model has to survive, and by letting the ablation overrule our own
pitch.

**"Why rules for barriers instead of a model?"** Because no ground truth for barriers exists in a
real record — nobody labels "this gap was caused by cost". A supervised model there would fit our
simulator's generative assumptions and report its accuracy as though it meant something.

**"What about bias?"** Sex and insurance tier are out of the feature matrix, and we measure subgroup
performance anyway because a correlated feature can reproduce a disparity. Sex gaps are negligible.
The largest are age band and condition. We tested removing age: it cost 0.003 PR-AUC and made the
age gap slightly worse, so the disparity isn't the model reading age — it's base-rate and
record-richness differences. Unfixed, and reported as unfixed.

**"Could this harm a patient?"** The failure mode we take most seriously isn't a missed case, it's a
patient wrongly treated as non-compliant. That's why the output is a probability with confidence
rather than a label, why hypotheses are marked as hypotheses, why there's a hard abstention state,
and why a test fails the build on prohibited phrasing.
