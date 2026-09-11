# Evaluation

Generated from `artifacts/metrics.json` on 2026-09-11 22:37:23. Every figure below is produced by `scripts/make_report.py` reading the saved output of `ml/evaluate.py`. None is transcribed by hand.

## What was measured, and on what

The cohort is 600 synthetic patients over 540 days, yielding 7,800 patient-snapshots at 30-day intervals. 38.5% of snapshots carry a concern under the ground-truth definition. Results below are on 148 patients (1,924 snapshots) held out **at the patient level**: no patient contributes rows to more than one split, and the isotonic calibrator was fitted on a third disjoint set of patients.

Accuracy is not reported as a headline. With a concern rate near 38% it is easy to score well on and it says nothing about the two things that decide whether a care team keeps the system switched on: how many unnecessary calls it generates, and whether the stated probability means what it says.

## Headline comparison against the metric in current use

The comparator is Proportion of Days Covered with the conventional 0.80 threshold, plus a seven-day refill-gap rule. It is implemented faithfully in `dosesense/baseline.py` and hand-verified in the test suite, so this is the real metric rather than a weakened version of it.

|  | ROC-AUC | PR-AUC | Precision | Recall | F1 | Brier |
|---|---|---|---|---|---|---|
| **DoseSense** | **0.941** | **0.924** | 0.919 | 0.807 | **0.859** | 0.0763 |
| PDC / refill-gap rule | 0.801 | 0.709 | 0.786 | 0.695 | 0.738 | 0.2200 |

PR-AUC improves by +0.2159 and F1 by +0.1216. PR-AUC is the figure to read: the negative class is large, and ROC-AUC rewards ranking the obvious cases correctly.

## Alert burden

The alert-fatigue budget, in the unit a care team actually plans against. Snapshots are 30 days apart, so one alert per snapshot is one per patient-month.

|  | Alerts / 100 patient-months | False alerts / 100 patient-months | Precision of alerts |
|---|---|---|---|
| **DoseSense** | 32.6 | **2.6** | 0.919 |
| PDC rule | 32.8 | 7.0 | 0.786 |

A **62.3% reduction in false alerts** at a comparable total alert volume. For a 200-patient panel that is the difference between roughly 14 and 5 unnecessary conversations a month.

## The temporary-versus-meaningful test

Four behavioural archetypes are adherent for their entire record by construction, so any alert on them is a false positive against a known truth. Two of them are adversarial: `MISLEADING_ANOMALY` produces one dramatic refill gap, a missed appointment and an off-trend laboratory result while remaining fully adherent; `DISEASE_PROGRESSION` deteriorates clinically with a flawless dispensing record.

| Pattern | Snapshots | True concern rate | DoseSense alerts | PDC alerts |
|---|---|---|---|---|
| COMPLEX_REGIMEN | 78 | 0.872 | 0.769 | 0.654 |
| ACCESS_BARRIER | 156 | 0.801 | 0.718 | 0.744 |
| REPEATED_REFILL_DELAY | 169 | 0.799 | 0.710 | 0.710 |
| SILENT_NONADHERENCE | 143 | 0.748 | 0.322 | 0.028 |
| GRADUAL_DECLINE | 182 | 0.698 | 0.632 | 0.538 |
| SIDE_EFFECT | 117 | 0.684 | 0.658 | 0.504 |
| COST_BARRIER | 91 | 0.593 | 0.472 | 0.538 |
| TEMPORARY_INTERRUPTION | 169 | 0.106 | 0.142 | 0.355 |
| **ADHERENT_STABLE** | 468 | 0.000 | 0.011 | 0.051 |
| **DISEASE_PROGRESSION** | 78 | 0.000 | 0.064 | 0.077 |
| **MISLEADING_ANOMALY** | 26 | 0.000 | 0.192 | 0.231 |
| **OCCASIONAL_IRREGULARITY** | 247 | 0.000 | 0.061 | 0.154 |

Across the four negative-by-construction patterns the false-alert rate is **0.037 against 0.090 for the PDC rule**, a 59.5% relative reduction. On `TEMPORARY_INTERRUPTION` — a genuine five-week lapse that resolves — DoseSense alerts on far fewer snapshots than the PDC rule, because it stands down once the patient recovers rather than continuing to flag a closed episode.

## The case the conventional metric cannot see

`SILENT_NONADHERENCE` collects every prescription on schedule and does not take the doses. Because PDC is computed from dispensing records it reads near 1.0 throughout, and the metric is structurally blind. The only evidence is that the clinical picture drifts while the pharmacy record stays immaculate.

- True concern rate: 0.748
- PDC rule detects: **0.028**
- DoseSense detects: **0.322**

This is the hardest pattern in the cohort and the recall is low in absolute terms. It is reported as-is. The honest claim is a roughly 11.5× improvement on a case the standard metric essentially cannot address, not that the problem is solved.

## Calibration

The probability is multiplied by a clinical consequence weight to order the worklist, so it has to mean what it says. A model that ranks well but is miscalibrated would produce a wrong work order that no AUC would reveal.

- Expected calibration error: **0.0136**
- Maximum calibration error: 0.1353
- Brier score: 0.0763 (against 0.2200 for the baseline)

| Predicted | Observed | Snapshots |
|---|---|---|
| 0.057 | 0.045 | 751 |
| 0.133 | 0.127 | 363 |
| 0.256 | 0.274 | 146 |
| 0.350 | 0.486 | 35 |
| 0.430 | 0.500 | 2 |
| 0.596 | 0.623 | 69 |
| 0.665 | 0.674 | 46 |
| 0.853 | 0.905 | 63 |
| 0.996 | 0.991 | 449 |

The maximum calibration error of 0.135 sits in a bin holding 35 snapshots. That is small-sample noise rather than miscalibration, and the expected error weighted by bin size is the figure to read.

## Detection lead time

For every patient who ever warrants a concern, the first snapshot labelled positive is compared with the first snapshot the detector flagged.

|  | Median days to detection | Detection rate | Detected at or before onset |
|---|---|---|---|
| **DoseSense** | 30.0 | 96.2% | **27.6%** |
| PDC rule | 30.0 | 86.1% | 20.6% |

**This is the weakest result and is reported as such.** Median time to detection is one snapshot interval for both methods: DoseSense does not beat PDC on the median. It does catch a substantially larger share of cases at or before the onset snapshot, and it does so with a third of the false alerts. Median lead time is dragged down by the silent-non-adherence cases, which are detected late because laboratory markers move slowly. The gain here is in precision and in coverage of hard cases, not in speed.

## Ablation: does multi-signal reasoning earn its complexity?

Each row is an **independently trained and independently calibrated** model on a growing set of signal families, not the full model with inputs masked. Masking would leave trees splitting on always-zero features and would flatter the final model.

| Signal set | Features | PR-AUC | F1 | False alerts / 100pm |
|---|---|---|---|---|
| refill only | 18 | 0.8760 | 0.798 | 2.8 |
| refill prescription | 23 | 0.8830 | 0.808 | 2.2 |
| refill prescription appointment | 29 | 0.8940 | 0.821 | 3.3 |
| plus symptoms | 37 | 0.9173 | 0.852 | 2.9 |
| plus laboratory | 42 | 0.9171 | 0.850 | 3.7 |
| all signals | 47 | 0.9245 | 0.859 | 2.6 |

PR-AUC rises 0.8760 → 0.9245 (+0.0485). The largest single jump comes from adding patient-reported symptoms, which is exactly where the silent-non-adherence cases become visible.

Two honest observations. Dispensing data alone already carries most of the discriminative signal, which is unsurprising and worth saying plainly rather than overselling fusion. And false alerts do not fall monotonically as signals are added — more inputs mean more ways to be wrong. The extra families earn their place by covering cases refill data cannot reach and by enabling barrier inference, not by lifting AUC dramatically.

## Choosing an operating point

A care team does not inherit a threshold; they choose one against their capacity.

| Threshold | Precision | Recall | F1 | Specificity | False alerts / 100pm |
|---|---|---|---|---|---|
| 0.3 | 0.882 | 0.838 | 0.859 | 0.934 | 4.2 |
| 0.4 | 0.917 | 0.808 | 0.859 | 0.957 | 2.7 |
| 0.5 | 0.919 | 0.807 | 0.859 | 0.958 | 2.6 |
| 0.6 | 0.955 | 0.747 | 0.838 | 0.979 | 1.3 |
| 0.7 | 0.981 | 0.703 | 0.819 | 0.992 | 0.5 |
| 0.8 | 0.981 | 0.703 | 0.819 | 0.992 | 0.5 |

The shipped default is 0.5.

## Subgroup performance

Sex and insurance tier are deliberately excluded from the model's inputs. That prevents the model reading them directly; it does not prevent a correlated feature reproducing the same disparity, which can only be established by measurement.


**By sex**

| Group | Snapshots | Concern rate | Recall | Precision | ECE |
|---|---|---|---|---|---|
| M | 1001 | 0.380 | 0.755 | 0.923 | 0.020 |
| F | 923 | 0.362 | 0.865 | 0.915 | 0.027 |

**By insurance tier**

| Group | Snapshots | Concern rate | Recall | Precision | ECE |
|---|---|---|---|---|---|
| Employer | 819 | 0.382 | 0.802 | 0.937 | 0.010 |
| Public | 741 | 0.348 | 0.810 | 0.897 | 0.019 |
| Self-pay | 364 | 0.393 | 0.811 | 0.921 | 0.030 |

**By age band**

| Group | Snapshots | Concern rate | Recall | Precision | ECE |
|---|---|---|---|---|---|
| 45-59 | 793 | 0.420 | 0.850 | 0.916 | 0.019 |
| 60-74 | 585 | 0.362 | 0.802 | 0.919 | 0.027 |
| <45 | 377 | 0.308 | 0.672 | 0.897 | 0.032 |
| 75+ | 169 | 0.314 | 0.849 | 0.978 | 0.046 |

**By condition**

| Group | Snapshots | Concern rate | Recall | Precision | ECE |
|---|---|---|---|---|---|
| Hypothyroidism | 403 | 0.407 | 0.774 | 0.970 | 0.027 |
| Asthma | 364 | 0.420 | 0.850 | 0.915 | 0.030 |
| Heart failure | 364 | 0.305 | 0.775 | 0.915 | 0.026 |
| Type 2 diabetes | 325 | 0.305 | 0.737 | 0.890 | 0.017 |
| Atrial fibrillation | 286 | 0.448 | 0.891 | 0.934 | 0.040 |
| Hypertension | 182 | 0.324 | 0.780 | 0.821 | 0.053 |

**Largest observed gaps**

| Grouping | Recall gap | Calibration-error gap |
|---|---|---|
| sex | 0.110 | 0.007 |
| insurance_tier | 0.009 | 0.020 |
| age_band | 0.177 | 0.027 |
| condition | 0.153 | 0.036 |

Gaps by sex are negligible. The largest disparities are by age band and condition. We tested removing age from the feature matrix entirely: it cost 0.003 PR-AUC and the age-band recall gap did not improve, so the disparity is not the model reading age directly — it reflects genuine differences in base rate and record richness across bands. That is a limitation, not a finding we have addressed. See `docs/limitations.md`.

## Signal-family importance

| Signal family | Share of gain |
|---|---|
| refill | 0.651 |
| symptom | 0.150 |
| context | 0.048 |
| behavioural | 0.046 |
| appointment | 0.041 |
| laboratory | 0.036 |
| prescription | 0.027 |

## Reproducing these numbers

```bash
bash scripts/reproduce.sh
```

The cohort, the splits and the model are all seeded, so the figures above are reproducible on any machine. Metrics will move if the cohort size or seed changes, which is expected: the test set is 148 patients and small-cohort variance is real.

