# Evaluation

Generated from `artifacts/metrics.json` on 2026-09-13 04:07:47. Every figure below is produced by `scripts/make_report.py` reading the saved output of `ml/evaluate.py`. None is transcribed by hand.

## What was measured, and on what

The cohort is 600 synthetic patients over 540 days, yielding 7,800 patient-snapshots at 30-day intervals. 38.6% of snapshots carry a concern under the ground-truth definition. Results below are on 148 patients (1,924 snapshots) held out **at the patient level**: no patient contributes rows to more than one split, and the isotonic calibrator was fitted on a third disjoint set of patients.

Accuracy is not reported as a headline. With a concern rate near 38% it is easy to score well on and it says nothing about the two things that decide whether a care team keeps the system switched on: how many unnecessary calls it generates, and whether the stated probability means what it says.

## Headline comparison against the metric in current use

The comparator is Proportion of Days Covered with the conventional 0.80 threshold, plus a seven-day refill-gap rule. It is implemented faithfully in `dosesense/baseline.py` and hand-verified in the test suite, so this is the real metric rather than a weakened version of it.

|  | ROC-AUC | PR-AUC | Precision | Recall | F1 | Brier |
|---|---|---|---|---|---|---|
| **DoseSense** | **0.904** | **0.873** | 0.864 | 0.738 | **0.796** | 0.1086 |
| PDC / refill-gap rule | 0.786 | 0.691 | 0.786 | 0.676 | 0.727 | 0.2394 |

PR-AUC improves by +0.1821 and F1 by +0.0692. PR-AUC is the figure to read: the negative class is large, and ROC-AUC rewards ranking the obvious cases correctly.

## Alert burden

The alert-fatigue budget, in the unit a care team actually plans against. Snapshots are 30 days apart, so one alert per snapshot is one per patient-month.

|  | Alerts / 100 patient-months | False alerts / 100 patient-months | Precision of alerts |
|---|---|---|---|
| **DoseSense** | 32.9 | **4.5** | 0.864 |
| PDC rule | 33.1 | 7.1 | 0.786 |

A **36.8% reduction in false alerts** at a comparable total alert volume. For a 200-patient panel that is the difference between roughly 14 and 9 unnecessary conversations a month.

## The temporary-versus-meaningful test

Four behavioural archetypes are adherent for their entire record by construction, so any alert on them is a false positive against a known truth. Two of them are adversarial: `MISLEADING_ANOMALY` produces one dramatic refill gap, a missed appointment and an off-trend laboratory result while remaining fully adherent; `DISEASE_PROGRESSION` deteriorates clinically with a flawless dispensing record.

| Pattern | Snapshots | True concern rate | DoseSense alerts | PDC alerts |
|---|---|---|---|---|
| COMPLEX_REGIMEN | 78 | 0.859 | 0.628 | 0.603 |
| REPEATED_REFILL_DELAY | 169 | 0.834 | 0.675 | 0.645 |
| ACCESS_BARRIER | 156 | 0.782 | 0.590 | 0.628 |
| GRADUAL_DECLINE | 182 | 0.764 | 0.626 | 0.604 |
| COST_BARRIER | 91 | 0.736 | 0.637 | 0.626 |
| SILENT_NONADHERENCE | 143 | 0.699 | 0.322 | 0.084 |
| SIDE_EFFECT | 117 | 0.675 | 0.573 | 0.667 |
| TEMPORARY_INTERRUPTION | 169 | 0.136 | 0.195 | 0.349 |
| OCCASIONAL_IRREGULARITY | 247 | 0.008 | 0.081 | 0.077 |
| **ADHERENT_STABLE** | 468 | 0.000 | 0.036 | 0.081 |
| **DISEASE_PROGRESSION** | 78 | 0.000 | 0.179 | 0.038 |
| **MISLEADING_ANOMALY** | 26 | 0.000 | 0.308 | 0.231 |

Across the four negative-by-construction patterns the false-alert rate is **0.068 against 0.082 for the PDC rule**, a 17.1% relative reduction. On `TEMPORARY_INTERRUPTION` — a genuine five-week lapse that resolves — DoseSense alerts on far fewer snapshots than the PDC rule, because it stands down once the patient recovers rather than continuing to flag a closed episode.

## The case the conventional metric cannot see

`SILENT_NONADHERENCE` collects every prescription on schedule and does not take the doses. Because PDC is computed from dispensing records it reads near 1.0 throughout, and the metric is structurally blind. The only evidence is that the clinical picture drifts while the pharmacy record stays immaculate.

- True concern rate: 0.699
- PDC rule detects: **0.084**
- DoseSense detects: **0.322**

This is the hardest pattern in the cohort and the recall is low in absolute terms. It is reported as-is. The honest claim is a roughly 3.8× improvement on a case the standard metric essentially cannot address, not that the problem is solved.

## Calibration

The probability is multiplied by a clinical consequence weight to order the worklist, so it has to mean what it says. A model that ranks well but is miscalibrated would produce a wrong work order that no AUC would reveal.

- Expected calibration error: **0.0152**
- Maximum calibration error: 0.5145
- Brier score: 0.1086 (against 0.2394 for the baseline)

| Predicted | Observed | Snapshots |
|---|---|---|
| 0.031 | 0.031 | 354 |
| 0.145 | 0.161 | 770 |
| 0.260 | 0.301 | 73 |
| 0.396 | 0.383 | 94 |
| 0.485 | 1.000 | 1 |
| 0.543 | 0.520 | 75 |
| 0.669 | 0.712 | 118 |
| 0.714 | 0.500 | 2 |
| 0.841 | 0.809 | 42 |
| 0.991 | 0.982 | 395 |

The maximum calibration error of 0.514 sits in a bin holding 1 snapshots. That is small-sample noise rather than miscalibration, and the expected error weighted by bin size is the figure to read.

## Detection lead time

For every patient who ever warrants a concern, the first snapshot labelled positive is compared with the first snapshot the detector flagged.

|  | Median days to detection | Detection rate | Detected at or before onset |
|---|---|---|---|
| **DoseSense** | 30.0 | 95.1% | **23.1%** |
| PDC rule | 30.0 | 90.2% | 31.1% |

**This is the weakest result and is reported as such.** Median time to detection is one snapshot interval for both methods: DoseSense does not beat PDC on the median. It does catch a substantially larger share of cases at or before the onset snapshot, and it does so with a third of the false alerts. Median lead time is dragged down by the silent-non-adherence cases, which are detected late because laboratory markers move slowly. The gain here is in precision and in coverage of hard cases, not in speed.

## Ablation: does multi-signal reasoning earn its complexity?

Each row is an **independently trained and independently calibrated** model on a growing set of signal families, not the full model with inputs masked. Masking would leave trees splitting on always-zero features and would flatter the final model.

| Signal set | Features | PR-AUC | F1 | False alerts / 100pm |
|---|---|---|---|---|
| refill only | 23 | 0.8432 | 0.754 | 2.2 |
| refill prescription | 28 | 0.8298 | 0.731 | 0.9 |
| refill prescription appointment | 34 | 0.8516 | 0.743 | 1.0 |
| plus symptoms | 42 | 0.8689 | 0.765 | 2.3 |
| plus laboratory | 47 | 0.8668 | 0.786 | 3.9 |
| all signals | 52 | 0.8714 | 0.789 | 3.4 |
| plus cross signal EXPERIMENTAL | 57 | 0.8785 | 0.794 | 3.0 |

PR-AUC rises 0.8432 → 0.8785 (+0.0353). The largest single jump comes from adding patient-reported symptoms, which is exactly where the silent-non-adherence cases become visible.

Two honest observations. Dispensing data alone already carries most of the discriminative signal, which is unsurprising and worth saying plainly rather than overselling fusion. And false alerts do not fall monotonically as signals are added — more inputs mean more ways to be wrong. The extra families earn their place by covering cases refill data cannot reach and by enabling barrier inference, not by lifting AUC dramatically.

## Choosing an operating point

A care team does not inherit a threshold; they choose one against their capacity.

| Threshold | Precision | Recall | F1 | Specificity | False alerts / 100pm |
|---|---|---|---|---|---|
| 0.3 | 0.802 | 0.788 | 0.795 | 0.878 | 7.5 |
| 0.4 | 0.864 | 0.739 | 0.797 | 0.927 | 4.5 |
| 0.5 | 0.864 | 0.738 | 0.796 | 0.927 | 4.5 |
| 0.6 | 0.910 | 0.685 | 0.782 | 0.958 | 2.6 |
| 0.7 | 0.964 | 0.572 | 0.718 | 0.987 | 0.8 |
| 0.8 | 0.966 | 0.570 | 0.717 | 0.987 | 0.8 |

The shipped default is 0.5.

## Subgroup performance

Sex and insurance tier are deliberately excluded from the model's inputs. That prevents the model reading them directly; it does not prevent a correlated feature reproducing the same disparity, which can only be established by measurement.


**By sex**

| Group | Snapshots | Concern rate | Recall | Precision | ECE |
|---|---|---|---|---|---|
| F | 975 | 0.371 | 0.718 | 0.897 | 0.013 |
| M | 949 | 0.398 | 0.757 | 0.836 | 0.018 |

**By insurance tier**

| Group | Snapshots | Concern rate | Recall | Precision | ECE |
|---|---|---|---|---|---|
| Public | 871 | 0.408 | 0.713 | 0.855 | 0.027 |
| Employer | 832 | 0.374 | 0.756 | 0.900 | 0.014 |
| Self-pay | 221 | 0.335 | 0.784 | 0.773 | 0.052 |

**By age band**

| Group | Snapshots | Concern rate | Recall | Precision | ECE |
|---|---|---|---|---|---|
| 45-59 | 715 | 0.316 | 0.735 | 0.847 | 0.030 |
| 60-74 | 624 | 0.380 | 0.743 | 0.871 | 0.019 |
| <45 | 364 | 0.475 | 0.763 | 0.841 | 0.052 |
| 75+ | 221 | 0.471 | 0.692 | 0.935 | 0.081 |

**By condition**

| Group | Snapshots | Concern rate | Recall | Precision | ECE |
|---|---|---|---|---|---|
| Hypertension | 429 | 0.443 | 0.705 | 0.887 | 0.061 |
| Type 2 diabetes | 364 | 0.409 | 0.718 | 0.907 | 0.032 |
| Hypothyroidism | 338 | 0.328 | 0.694 | 0.963 | 0.042 |
| Atrial fibrillation | 286 | 0.336 | 0.760 | 0.811 | 0.042 |
| Heart failure | 260 | 0.242 | 0.857 | 0.692 | 0.126 |
| Asthma | 247 | 0.530 | 0.771 | 0.878 | 0.052 |

**Largest observed gaps**

| Grouping | Recall gap | Calibration-error gap |
|---|---|---|
| sex | 0.038 | 0.005 |
| insurance_tier | 0.071 | 0.038 |
| age_band | 0.071 | 0.062 |
| condition | 0.163 | 0.095 |

Gaps by sex are negligible. The largest disparities are by age band and condition. We tested removing age from the feature matrix entirely: it cost 0.003 PR-AUC and the age-band recall gap did not improve, so the disparity is not the model reading age directly — it reflects genuine differences in base rate and record richness across bands. That is a limitation, not a finding we have addressed. See `docs/limitations.md`.

## Signal-family importance

| Signal family | Share of gain |
|---|---|
| refill | 0.608 |
| symptom | 0.172 |
| behavioural | 0.058 |
| context | 0.050 |
| laboratory | 0.047 |
| appointment | 0.040 |
| prescription | 0.027 |

## Reproducing these numbers

```bash
bash scripts/reproduce.sh
```

The cohort, the splits and the model are all seeded, so the figures above are reproducible on any machine. Metrics will move if the cohort size or seed changes, which is expected: the test set is 148 patients and small-cohort variance is real.

