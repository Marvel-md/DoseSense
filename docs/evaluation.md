# Evaluation

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

## Protocol

Patients are disjoint across training, calibration and test sets. Windows use only records available at the snapshot date. The comparator combines PDC with a refill-gap rule. No real patient records were used.

Cohort: 600 patients and 7,800 snapshots. Snapshot spacing: 30 days. Model threshold: 0.5. The label uses a trailing 90-day window and a minimum non-adherent fraction of 0.34.

## Alert burden

One threshold crossing per snapshot counts as an alert, including repeated alerts on the same patient. This is not a measured count of calls made or avoided.

| Method | Alerts / 100 patient-months | False alerts / 100 patient-months |
|---|---|---|
| DoseSense | 34.67 | 3.95 |
| PDC rule | 33.11 | 7.85 |

## Behaviour by pattern

**Alert rate** divides all flagged snapshots by all snapshots in that pattern. **Recall** divides true-positive flags by positive snapshots only. A pattern can include both positive and negative snapshots; its alert rate is not its recall. Recall is undefined (—) when there are no positive snapshots.

| Pattern | Patients | Snapshots | True concern rate | Model alert rate | PDC alert rate | Model recall | PDC recall |
|---|---|---|---|---|---|---|---|
| REPEATED_REFILL_DELAY | 13 | 169 | 0.846 | 0.740 | 0.716 | 0.860 | 0.804 |
| COMPLEX_REGIMEN | 6 | 78 | 0.808 | 0.679 | 0.590 | 0.825 | 0.730 |
| SILENT_NONADHERENCE | 11 | 143 | 0.797 | 0.455 | 0.126 | 0.535 | 0.123 |
| COST_BARRIER | 7 | 91 | 0.758 | 0.681 | 0.593 | 0.884 | 0.739 |
| GRADUAL_DECLINE | 14 | 182 | 0.758 | 0.604 | 0.538 | 0.775 | 0.667 |
| ACCESS_BARRIER | 12 | 156 | 0.756 | 0.635 | 0.641 | 0.839 | 0.797 |
| SIDE_EFFECT | 9 | 117 | 0.684 | 0.650 | 0.581 | 0.938 | 0.800 |
| TEMPORARY_INTERRUPTION | 13 | 169 | 0.148 | 0.213 | 0.343 | 0.480 | 0.360 |
| OCCASIONAL_IRREGULARITY | 19 | 247 | 0.004 | 0.073 | 0.117 | 1.000 | 1.000 |
| ADHERENT_STABLE | 36 | 468 | 0.000 | 0.030 | 0.073 | — | — |
| DISEASE_PROGRESSION | 6 | 78 | 0.000 | 0.038 | 0.103 | — | — |
| MISLEADING_ANOMALY | 2 | 26 | 0.000 | 0.231 | 0.115 | — | — |

3 patterns contain no positive labels in this test set: ADHERENT_STABLE, DISEASE_PROGRESSION, MISLEADING_ANOMALY. Their pooled false-alert rates are 0.040 (DoseSense) and 0.079 (PDC). The number of all-negative groups is determined from labels, not assumed.

## Silent non-adherence

This simulated pattern decouples medication collection from the hidden adherence state. It tests whether other records can contribute evidence when refills look normal.

Alert rates across all 143 snapshots: 45.5% (model), 12.6% (baseline).

Recall among 114 positive snapshots: 61/114 = 53.5% (model), 14/114 = 12.3% (baseline). These are repeated observations of a small patient subgroup, not independent clinical cases. Higher alert volume alone does not prove better detection.

## Calibration

ECE: 0.0132; maximum bin error: 0.5000; Brier score: 0.0956.

| Mean prediction | Observed rate | Snapshots |
|---|---|---|
| 0.053 | 0.039 | 536 |
| 0.142 | 0.155 | 418 |
| 0.255 | 0.236 | 296 |
| 0.333 | 0.571 | 7 |
| 0.500 | 1.000 | 2 |
| 0.649 | 0.633 | 150 |
| 0.800 | 0.714 | 14 |
| 0.965 | 0.966 | 501 |

ECE is a bin-weighted average, not a bound on each prediction. Small bins are unstable; their errors cannot simply be dismissed as noise. This calibration has not been tested on clinical data.

## Detection timing

The current calculation selects the first flagged snapshot at or after onset; if none exists, it uses the last pre-onset flag. Consequently, a pre-onset flag does not take precedence over a later flag. Timing statistics are conditional on patients with a selected flag; never-flagged patients are reported separately. This retrospective convention is not evidence of prospective early warning.

| Method | Onset patients | Never flagged | Median days | Any flag / onset patients | At/before onset / flagged patients |
|---|---|---|---|---|---|
| DoseSense | 83 | 2 | 30.0 | 97.6% | 32.1% |
| PDC rule | 83 | 7 | 30.0 | 91.6% | 30.3% |

## Ablation

Each row is independently trained and calibrated on the same patient splits. The experimental row is excluded from the shipped feature set.

| Signal set | Features | PR-AUC | F1 | False alerts / 100pm |
|---|---|---|---|---|
| refill_only | 23 | 0.8262 | 0.749 | 3.38 |
| refill_prescription | 28 | 0.8486 | 0.752 | 4.31 |
| refill_prescription_appointment | 34 | 0.8624 | 0.763 | 2.91 |
| plus_symptoms | 42 | 0.8938 | 0.817 | 2.86 |
| plus_laboratory | 47 | 0.8981 | 0.820 | 2.60 |
| all_signals | 52 | 0.9000 | 0.834 | 3.95 |
| plus_cross_signal_EXPERIMENTAL | 57 | 0.9106 | 0.834 | 2.44 |

Shipped-family ablation versus refill-only: 0.8262 → 0.9000 (+0.0738 PR-AUC). Intermediate additions need not improve performance monotonically. The separately trained ablation is not the headline ensemble.

## Operating points

This is a descriptive test-set sweep. Choose thresholds on development data and evaluate on new held-out data; do not select a winner from this table and treat its test performance as an unbiased estimate.

| Threshold | Precision | Recall | F1 | Specificity | False alerts / 100pm |
|---|---|---|---|---|---|
| 0.3 | 0.883 | 0.792 | 0.835 | 0.933 | 4.11 |
| 0.4 | 0.886 | 0.787 | 0.834 | 0.935 | 3.95 |
| 0.5 | 0.886 | 0.787 | 0.834 | 0.935 | 3.95 |
| 0.6 | 0.886 | 0.784 | 0.832 | 0.935 | 3.95 |
| 0.7 | 0.959 | 0.658 | 0.780 | 0.982 | 1.09 |
| 0.8 | 0.959 | 0.658 | 0.780 | 0.982 | 1.09 |

## Subgroups

Excluding sex and insurance tier from features does not eliminate proxy effects or establish fairness. These synthetic subgroup measurements do not explain causation.

### sex

| Group | Snapshots | Recall | Precision | ECE |
|---|---|---|---|---|
| F | 1053 | 0.790 | 0.844 | 0.029 |
| M | 871 | 0.783 | 0.933 | 0.027 |

### insurance_tier

| Group | Snapshots | Recall | Precision | ECE |
|---|---|---|---|---|
| Public | 988 | 0.783 | 0.931 | 0.020 |
| Employer | 663 | 0.789 | 0.818 | 0.035 |
| Self-pay | 273 | 0.798 | 0.879 | 0.026 |

### age_band

| Group | Snapshots | Recall | Precision | ECE |
|---|---|---|---|---|
| 45-59 | 767 | 0.794 | 0.919 | 0.027 |
| 60-74 | 650 | 0.826 | 0.881 | 0.045 |
| <45 | 299 | 0.761 | 0.953 | 0.056 |
| 75+ | 208 | 0.692 | 0.692 | 0.080 |

### condition

| Group | Snapshots | Recall | Precision | ECE |
|---|---|---|---|---|
| Hypothyroidism | 481 | 0.745 | 0.854 | 0.025 |
| Hypertension | 312 | 0.780 | 0.907 | 0.023 |
| Atrial fibrillation | 299 | 0.831 | 0.784 | 0.052 |
| Asthma | 286 | 0.832 | 0.972 | 0.038 |
| Heart failure | 286 | 0.860 | 0.942 | 0.025 |
| Type 2 diabetes | 260 | 0.678 | 0.910 | 0.019 |

| Grouping | Recall gap | ECE gap |
|---|---|---|
| sex | 0.007 | 0.002 |
| insurance_tier | 0.016 | 0.015 |
| age_band | 0.134 | 0.053 |
| condition | 0.182 | 0.033 |

## Approximate data-richness audit

This audit uses fixed ensemble disagreement (0.02), simplified family counts and no change-detection context. It is not the serving abstention policy evaluated end to end. Recall among committed cases excludes abstentions and must not be interpreted as recall across all patients.

| Stratum | Snapshots | Abstention rate | Committed recall |
|---|---|---|---|
| sparse | 301 | 65.5% | 0.636 |
| partial | 269 | 0.0% | 0.785 |
| good | 196 | 0.0% | 0.759 |
| rich | 1158 | 0.0% | 0.839 |

## Feature importance

| Family | Share of gain |
|---|---|
| refill | 0.610 |
| symptom | 0.155 |
| appointment | 0.070 |
| context | 0.058 |
| behavioural | 0.047 |
| laboratory | 0.039 |
| prescription | 0.021 |

Importance describes how this ensemble uses simulated features; it is not causal evidence.

## Reproduction

```bash
bash scripts/reproduce.sh
python scripts/make_report.py --check
```

Seeds control cohort generation and splits. Library versions can change results; use requirements-lock.txt for the verified environment. Reproducing simulator results does not validate the simulator or a clinical deployment.

```json
{
  "python": "3.12.14",
  "packages": {
    "numpy": "2.3.5",
    "pandas": "2.2.3",
    "scipy": "1.17.0",
    "scikit-learn": "1.8.0",
    "lightgbm": "4.7.0",
    "shap": "0.52.0"
  }
}
```
