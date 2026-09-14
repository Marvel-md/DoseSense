# Methodology

## Prediction unit and labels

The unit is a patient-snapshot, not a permanent label assigned to a person. Evaluation rescores at 30-day intervals with trailing features. The default synthetic cohort contains 600 patients over 540 days, with a 180-day warm-up. The hidden simulator trajectory defines whether non-adherence occupies at least 34% of the trailing 90-day label window. This is an illustrative operational definition, not a clinically validated threshold.

A behavioural archetype can contain positive and negative snapshots. Whether a group is entirely negative is checked from its actual labels. The hidden trajectory is used for training/evaluation labels and is excluded from serving.

## Causal windows and splits

A snapshot on day t can use only records dated on or before t. Tests compare feature extraction on the full record with extraction after physically truncating the record. Training, calibration and test sets are split by patient, because adjacent snapshots share history. Bagging also resamples whole patients.

These controls address record leakage. They do not remove optimism caused by developing a model and simulator together or repeatedly inspecting the same test set. Independent external validation remains necessary.

## Personal baselines and change detection

Refill lateness, symptom change and lab trends are compared with the patient's own history. The model also contains absolute, availability and context features; not every feature is baseline-normalised.

Mean-shift segmentation distinguishes stable, temporary and persistent changes. Short recent changes can be provisional. Variance and effect-size floors prevent tiny baseline variation from exaggerating deviations. Refill timing includes raw days and interval-relative measures, and its change floor scales with the supply period. The constants are explicit in code and remain illustrative policy choices.

## Features and record sources

The shipped model uses **52 features in seven families**: refill, prescription, appointment, symptom, laboratory, behavioural and context. The six main record sources are pharmacy, prescription, appointment, symptom, laboratory and activity records; context is an engineering grouping, not an extra wearable or questionnaire requirement.

New patient-portal submissions are not feature inputs. Existing symptom questionnaires already in the routine records are feature inputs. The claim is detection without requiring manual missed-dose logs, not complete exclusion of patient-reported information.

Sex, insurance tier, pharmacy distance and clinical consequence are excluded from the model matrix. Insurance/distance can appear in a separately displayed barrier hypothesis; age is included. Excluding attributes does not eliminate proxies or establish fairness.

## Ensemble and explanations

Five LightGBM classifiers are trained on patient bootstrap samples. An isotonic calibrator is fitted to their mean prediction on a separate calibration set. Ensemble spread contributes to uncertainty; it is not itself a calibrated confidence interval.

Grouped SHAP values explain contributions to the tree model. They do not establish causality or directly represent percentage-point changes in the final calibrated probability. The dashboard also offers comparisons that replace a signal family with training medians; these are model sensitivity checks, not intervention predictions.

Tree models fit the current tabular representation and support fast scoring and explanations. No unimplemented sequence-model comparison is claimed.

## Confidence, priority and hypotheses

Confidence uses record completeness, evidence families, model disagreement and change context. Insufficient evidence is kept separate from a well-supported low-concern assessment. The dashboard combines likelihood with illustrative clinical consequence and confidence weights for priority; severity is declared policy rather than a learned adherence feature.

Barrier hypotheses use scored rules and can return no proposed cause. They are not validated explanations of why an individual is struggling. The display separates observed records, model inference, hypotheses and suggested human review.

Clinician feedback and new portal reports persist in SQLite. They do not automatically retrain or recalibrate the ensemble.

## Evaluation denominators

| Quantity | Numerator / denominator |
|---|---|
| Alert rate | All model flags / all snapshots in the evaluated group |
| Recall | True-positive flags / positive snapshots; undefined if none are positive |
| Precision | True-positive flags / all flags |
| False alerts per 100 patient-months | False-positive flags / observed patient-months × 100 |
| ECE | Bin-size-weighted absolute difference between average predictions and observed rates |

The headline comparison thresholds model probabilities. It does not evaluate every dashboard policy rule end to end. Repeated flags count separately and do not measure unique calls. Per-pattern snapshots are correlated within patients; their counts must not be treated as independent clinical sample sizes.

The timing function takes the first flagged snapshot at or after onset, or the last before onset if no later flag exists. Timing summaries condition on a selected flag; never-flagged patients are reported separately. This convention does not justify a prospective early-detection claim.

## Ablation, fairness and uncertainty audit

Ablation trains/calibrates separate models using successively larger feature sets. Compare `refill_only` with `all_signals` for shipped-family gain; the experimental cross-signal family is excluded from the shipped model. Intermediate performance may rise or fall. Historical experiment comments are not current submission evidence.

Subgroup metrics describe the simulator results and cannot establish the cause of disparities. The data-richness audit approximates confidence using fixed disagreement and simplified evidence counts, so it is not a full evaluation of serving abstention. Recall on committed cases excludes abstentions.

## Reporting and reproducibility

`ml/evaluate.py` writes metrics, subgroup recall counts and runtime versions. `scripts/make_report.py` creates the full report plus marked result blocks in the README and presentation. `--check` detects stale generated blocks. Narrative outside those blocks still needs human review. The current submission uses one evaluation artifact; historical benchmark results are separately scoped and not mixed into the pitch.

Patient random streams use a stable SHA-256-derived seed. The earlier Python `hash()` seed was randomised across processes; its saved results cannot be reproduced from the public seed alone. A cross-process test now checks independence from `PYTHONHASHSEED`.

The lock file records the tested dependency versions. Reproduction verifies a computational result on the simulator, not clinical validity. See [limitations.md](limitations.md).
