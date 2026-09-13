# Methodology

This document explains the substantive choices and, where a choice was between defensible
alternatives, why we went the way we did. Where a decision turned out to be wrong and was changed,
that is recorded too — two of the more important design details in this system exist because an
earlier version failed.

---

## 1. The unit of prediction

The obvious framing is one row per patient: *is this patient non-adherent?* We rejected it.

Non-adherence is a **state that comes and goes**, not a property of a person. A patient can lapse
for five weeks and recover. Labelling that patient "non-adherent" is both clinically wrong and
ethically objectionable — it brands someone on the basis of an episode.

So the unit is a **patient-snapshot**: the system re-scores every patient every 30 days, and each
snapshot is judged on the trailing window only. This has three consequences that all turned out to
matter:

- Patients who lapse and recover are positive during the lapse and negative afterwards. The model is
  scored on *tracking a state*, not on classifying a person.
- Two metrics become computable that cannot be computed from one row per patient: **false alerts per
  patient-month** (the alert-fatigue budget) and **detection lead time**.
- It matches deployment. A care team runs a panel review on a cadence; they do not classify a cohort
  once.

600 patients × 540 days at a 30-day cadence with a 180-day warm-up gives 7,800 snapshots.

## 2. The label

A snapshot is labelled a concern when the hidden adherence state was non-adherent for **at least 34%
of the trailing 90 days**.

This threshold is a *definition of "meaningful"*, not a tuning knob, and it is the single most
consequential number in the project. The brief asks the system to distinguish temporary irregularity
from a meaningful pattern; you cannot do that without committing to where the line sits. At 34% of a
quarter:

- a single missed week does not clear it
- a sustained partial pattern does
- a five-week complete interruption does clear it *while it is happening* — which we think is
  correct, because six weeks without an anticoagulant is a real clinical event, not a blip

The alternative we considered was labelling only *persistent* non-adherence positive and treating
resolved lapses as negative throughout. We rejected it because it would have trained the model to
ignore acute interruptions, which are exactly the cases where a phone call helps most.

## 3. Causality, and the test that protects it

Every window is a trailing window. A snapshot on day *t* may only use records dated on or before *t*.

This is enforced structurally — each patient's series is sliced once per snapshot rather than
filtered after the fact — and it is verified by `test_no_future_leakage`, which recomputes a
snapshot against a record physically truncated at *t* and asserts all 47 features are identical.

That test exists because leakage in an adherence model is easy to introduce and **invisible in the
metrics**. A centred rolling window or a feature accidentally computed over the full series produces
a model that looks superb and performs badly, and nothing in the confusion matrix will tell you.

Splitting is by **patient**, never by row, for the same reason. Two snapshots of the same person 30
days apart share almost all of their history; a random row split leaks that history across the
train/test boundary and inflates everything.

## 4. Personal baselines

Almost every feature is expressed against the patient's own history rather than a population average.

> A 34-day gap is unremarkable for a patient who has always run late. It is a red flag for one who
> has never been more than a day out.

Concretely: `refill_delay_excess` is recent lateness minus this patient's own baseline lateness;
`lab_last_z` is the latest result against this patient's own earlier range; the behavioural features
are all z-scores against the individual's own distribution.

Two guards had to be added after seeing real output:

**A dispersion floor.** A patient whose first three HbA1c results land within 0.05% of each other
has a near-zero sample SD, and dividing by it turns ordinary assay noise into a twelve-sigma event.
We saw z-scores of 1,143 in the first run. The denominator is now floored at 5% of the measurement
level — roughly the analytical variability of the markers involved — and z-scores are clipped at ±8.

**A clinical effect floor.** See §5.

## 5. Change-point detection

Exact binary segmentation with a Gaussian mean-shift cost and a BIC-style penalty, applied to refill
lateness, symptom burden and the laboratory marker.

**Why not PELT, or a Bayesian online detector?** On the series a pharmacy record actually produces —
typically twelve to eighteen refill intervals — the heavier methods give no measurable benefit and
cost interpretability, which the alert copy depends on. The alert says *"timing shifted from a
baseline of 1.1 days late to 14.9 days and stayed there"*, and that sentence is generated from the
segmentation's own parameters.

**Three statuses, not two.** `STABLE`, `TEMPORARY_ANOMALY`, `PERSISTENT_SHIFT`. After locating a
change point the detector tests whether the tail came back toward the old level. This is what stops
the system firing on the patient who went on holiday once.

### Two fixes that mattered

**The clinical effect floor.** Our first version required a shift of 1.1 baseline standard
deviations. Because adherent patients' refill delays are roughly exponential, SD ≈ mean, so a shift
from 1 day late to 3 days late cleared the bar. It was a real shift and clinically meaningless.
Adding an absolute floor — 3.5 days for refill timing, 0.8 points on the 0–10 symptom scale — cut
false positives on genuinely stable patients **from 17.9% to 0.8%**. Statistical detectability and
clinical meaning are different tests, and the code now applies both. The floors are declared
constants, stated openly so they can be argued with.

**Making `provisional` reachable.** We added a `provisional` flag for shifts detected too recently
to judge reversion — and then a test revealed it could never fire. The search range required at
least three points either side of the change point, so there were always at least three points after
it. The flag was dead code.

Fixing it required a second fix. Extending the search so the right-hand segment can be one or two
points exposed a degenerate case: a single-point segment has zero variance, log of which is unbounded
below, so an unfloored cost makes the final point the most attractive split in *every* series. The
segment variance is now floored at 5% of the whole series' variance, which keeps short tail segments
comparable to long ones. `provisional` now fires, and the confidence layer downgrades those
assessments instead of asserting a sustained shift on two observations.

## 6. Features

47 features in seven families: refill, prescription, appointment, symptom, laboratory, behavioural,
context. Family grouping is not cosmetic — it drives the ablation, the evidence-strength count used
for abstention, and the grouped attribution shown in the interface.

### Deliberate exclusions from the model matrix

| Excluded | Why |
|---|---|
| sex, insurance tier | Protected or near-protected. Held back for subgroup evaluation, never predicted from. |
| distance to pharmacy | A plausible access signal, but also a socioeconomic proxy. Used *only* as evidence shown to a human in barrier inference, where a clinician sees it and can discount it. Keeping it out of the model means a postcode proxy cannot silently drive the score. |
| clinical consequence | Severity is declared policy applied in the priority score, not a learned correlate of behaviour. Leaving it in risks the model learning that sicker patients are less adherent. |

`test_protected_attributes_are_not_model_inputs` fails the build if any of these reappear.

Age *is* included. It is clinically relevant and costs little to keep — see §11 for what happened
when we tried removing it.

## 7. Model choice

Bagged LightGBM, isotonic-calibrated. Not a sequence model.

The constraints favour trees on every axis that matters here: a few thousand patient-snapshots of
tabular data with heavy missingness, a hard requirement that every alert be explainable to a
pharmacist, and inference fast enough to rescore a panel on a laptop. A transformer over the raw
event stream is the interesting research direction and would need an order of magnitude more patients
to beat this, while making the explanation layer a research problem of its own.

**Bagging serves uncertainty, not accuracy.** Five models on bootstrap resamples of distinct
*patients* — not rows, because neighbouring snapshots of one person would otherwise land in and out
of the same bag and understate the real disagreement. The spread of their predictions is an honest
signal about how much the estimate depends on who happened to be in the training set: high for a
patient whose pattern is unlike anything in the data, low for a familiar one.

**Calibration on a third disjoint patient split.** Raw boosted-tree scores are ranking scores that
cluster near 0 and 1, not probabilities. Since the output is shown as a percentage *and multiplied by
a consequence weight to set priority*, it has to mean what it says — a ranking-correct but
miscalibrated model produces a wrong work order that no AUC would reveal. Isotonic regression,
fitted on patients seen by neither the trained models nor the evaluation. Resulting ECE: 0.024.

## 8. Uncertainty and abstention

Three inputs: how many independent signal families carry evidence, how much the ensemble disagrees
about this specific patient, and how complete the record is.

### The bug worth recording

Our first version abstained whenever fewer than two signal families carried evidence. That marked
**51% of the panel "insufficient evidence"**, including patients with eighteen months of flawless
records.

The error was conceptual, not numerical. *"No evidence of a concern"* and *"not enough evidence to
tell"* are different states, and collapsing them is both wrong and corrosive — a queue where half
the column reads "unknown" teaches the care team to ignore the column. A patient with a long
complete record and nothing worrying in it is a **confident negative**: the system knows a great
deal about them and none of it is concerning.

Abstention is now reserved for two situations that genuinely defeat the method:

1. the record is too sparse to establish any personal baseline (completeness < 0.34)
2. the probability is high enough to matter but rests on a single signal family

That is 3% of the panel, not 51%. For low-probability snapshots, confidence rests on how many
families were *available to examine*; for elevated ones, on how many *corroborate*.

### Making uncertainty real rather than decorative

Even after the fix, 93% of patients came out "High" confidence. The ensemble barely disagreed,
because the simulated signal was clean.

The honest fix was in the data, not the confidence formula. Real panels contain recently-enrolled
patients with four months of history, patients who never complete symptom questionnaires, and
patients who attend laboratory monitoring half as often as protocol. Adding that variation —
about a quarter, a fifth, and a broad distribution respectively — produced genuine spread:
High 112, Medium 29, Low 2, Insufficient 9 on a 150-patient sample. Late enrolment is not a
data-quality problem to be cleaned away; it is the normal condition of a panel and the main source
of real uncertainty in this system.

## 9. Priority

```
priority = calibrated probability × clinical consequence × confidence weight
```

An 88% concern on levothyroxine should not outrank a 61% concern on warfarin. Sorting a worklist by
probability alone ignores that a pharmacist with two hours has to weigh likelihood against harm.

The consequence weight comes from a declared table in `dosesense/config.py`, exposed at
`GET /api/config`. **Clinical severity is a governance decision, not a statistical one**, and a care
team must be able to change it without retraining anything. Keeping it out of the model and in
policy is what makes that possible.

## 10. Barrier inference

Scored expert rules over the observed evidence, producing a ranked hypothesis with the specific
observations supporting it — or an explicit abstention.

**Why rules rather than a second model?** There is no ground truth for barriers in a real health
record. Nobody labels "this gap was caused by cost". A supervised model here would be fitting our own
simulator's generative assumptions and reporting its accuracy as if it meant something about reality.
Rules are honest about being an expert prior, and a clinician can read them and disagree.

Three commitments:

- **Hypothesis, never attribution.** Always "consistent with", never a determination about a person.
- **It abstains.** Below a score floor the answer is that nothing points clearly. Where the top two
  hypotheses are within 0.08 of each other, both are shown rather than the system picking a winner it
  cannot justify. Roughly a third of alerts return no barrier. Guessing a reason and being wrong is
  worse than saying nothing, because a clinician acting on a wrong reason asks the wrong question and
  may close the case.
- **It uses context the model is not allowed to.** Insurance tier and distance to pharmacy appear
  here as evidence *shown to a human alongside a stated hypothesis*, where they are useful and
  visible, rather than as silent model inputs.

## 11. What we tested and did not adopt

**Removing age from the feature matrix.** Age was the fourth most important feature and age band
showed the largest subgroup recall gap (0.173). We removed it and retrained: PR-AUC fell 0.003 and
the age-band gap got slightly *worse* (0.181). So the disparity is **not** the model reading age
directly — it reflects genuine differences in base rate and record richness across bands. We kept
age and documented the gap as unaddressed, rather than removing it and claiming a fix we had not
achieved.

**A deep sequence model.** See §7. Not justified at this data scale, and costly to the explanation
layer.

**Postgres.** The cohort is CSV on disk and the only mutable state is a clinician-feedback table.
A database container would add an orchestration dependency and a failure mode at a demonstration
without buying anything. Persistence is behind functions in `backend/app/main.py`; swapping it is a
contained change.

**React + TypeScript for the dashboard.** See the README. A single-file interface with no build step
and no external requests is worth more at a venue with unreliable network than a nicer component
model nobody will inspect. The trade is real and the render harness compensates for the missing
compiler.

## 12. The simulator, and the flaw that changed the design

`dosesense/datagen.py` samples a hidden day-by-day adherence trajectory per patient, then generates
the observable record conditioned on it. Twelve archetypes.

Two archetypes exist specifically because the first version of the simulator was too easy, and the
ablation caught it.

**The ablation contradicted our own hypothesis.** The pitch was "fusing multiple signals is the
innovation". The first ablation showed refill-only PR-AUC of 0.9561 against 0.9572 for all seven
families — fusion added **nothing**, and false alerts got *worse* as signals were added. A judge
reading that table would have taken the project apart.

The cause was the simulator, not the model. We had made non-adherence always express itself as late
refills, which makes the dispensing record a near-sufficient statistic. But the clinical phenomenon
this problem is named after is the opposite: the patient who collects every prescription on schedule
and does not take it. PDC is computed from dispensing records and is structurally blind to them.

Adding `SILENT_NONADHERENCE` — perfect primary adherence, poor secondary adherence — made the
ablation honest: PR-AUC 0.873 → 0.907, monotone, with the largest jump from symptoms, which is
exactly where those cases become visible.

`DISEASE_PROGRESSION` was added for the mirror-image reason: a patient who deteriorates clinically
with a flawless dispensing record. Without it, nothing in the cohort produced worsening labs and
symptoms *without* non-adherence, so the model had no reason to learn the distinction between
*"the treatment isn't working"* and *"the treatment isn't being taken"*. A system that confuses those
two sends the care team after the wrong problem. Its presence is also why symptom importance fell
from an implausible 0.71 to a realistic 0.15.

## 12b. Adaptive intervals

A fixed day-count threshold for "late" assumes every patient is on the same dispensing schedule.
They are not: the simulated panel now spans 15-day acute courses to 90-day maintenance supplies,
matching normal pharmacy practice, and being a week late means something entirely different at each
end.

Two changes follow. Every lateness measure is also expressed as a fraction of the patient's own
expected interval, taken from the prescription rather than assumed — the raw day counts are kept
because they are what a clinician reads, while the ratios are what generalise across regimens. And
the change detector's clinical floor became `max(3.5 days, 12% of the supply period)`, so a slip
that is trivial on a quarterly supply no longer clears the same bar as one on a fortnightly course.

## 12c. An experiment that failed

Cross-signal velocity was built to catch declining trajectories earlier by correlating the rate of
clinical drift against the rate of refill drift. The reasoning was sound: silent non-adherence is
deterioration with no collection signal, so the *divergence* between the channels should be the
detector. Measured across three seeds it moved PR-AUC by +0.0007, silent-non-adherence recall by
+0.011, and disease-progression false alerts by +0.025 in the wrong direction.

It is excluded from the shipped model and retained in `EXPERIMENTAL_FAMILIES` with the numbers
written into the source. The failure is informative: the divergence signature is genuinely shared
between the pattern we want to catch and the confounder we must not, so separating them needs
something other than relative velocity. We do not know what.

## 13. Evaluation philosophy

Reported: PR-AUC, false alerts per patient-month, calibration, per-archetype behaviour, subgroup
performance, an operating-point sweep, and an ablation.

Not reported as a headline: accuracy. With a 38% concern rate it is easy to score well on and it
says nothing about the two things that decide adoption — unnecessary calls generated, and whether a
stated probability means what it says.

The per-archetype table is where the brief is actually tested. Four archetypes are
adherent-by-construction, so the alert rate on them is a false-positive rate against a *known,
adversarial* truth, comparable directly against the PDC rule on the same rows.

`docs/evaluation.md` is generated by `scripts/make_report.py` from `artifacts/metrics.json`. No
figure in this repository is transcribed by hand, so there is no path by which a stale or invented
number can survive in the write-up.
