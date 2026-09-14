# Limitations

DoseSense is a working hackathon prototype evaluated on synthetic data. It has no prospective or external clinical validation.

## Simulator bias

We built both the simulator and the model. Patient-level train/calibration/test separation prevents patients leaking between splits; it does not validate the simulator's assumptions. Simulated laboratory responses, symptom trends, record completeness and adherence trajectories may differ substantially from real care records. The generator models one primary condition per patient and does not reproduce real multimorbidity, messy pharmacy transfers or the effect of clinician interventions.

## Difficult patterns and metric interpretation

Disease progression can produce worsening labs and symptoms despite healthy dispensing coverage. The system may confuse that pattern with medication-taking difficulty. Consult the current per-pattern table in [evaluation.md](evaluation.md), including cases where the baseline performs better.

A subgroup alert rate counts every flagged snapshot, including false positives. Recall counts flags only among positive snapshots. Earlier versions of the documentation conflated them; the evaluation now reports both with explicit denominators. Repeated snapshots are correlated observations, so small patient subgroups do not support precise generalisation claims.

The timing function selects the first flag at or after onset, or the last pre-onset flag if no later flag exists. It does not always select the earliest flag. Medians and timing shares exclude never-flagged patients, who are reported separately. This retrospective convention and the snapshot cadence do not establish prospective early-warning benefit.

The headline alert burden is evaluated at the model threshold. The dashboard additionally uses confidence, change state and priority rules. A reduction in synthetic threshold flags is not proof of fewer calls, saved money or better clinical outcomes.

## Calibration and fairness

ECE is an average across bins, not a guarantee that every probability is close to the true risk. Sparse calibration bins are unstable. Calibration on synthetic patients does not transfer automatically to clinical data.

Sex and insurance tier are excluded from model inputs, but correlated features can retain proxy effects. Age and condition subgroup gaps are measured, not resolved. Excluding an attribute alone does not establish fairness or explain a disparity.

The richness audit is approximate: it uses fixed ensemble disagreement, simplified family counts and no change-detection context. Its committed-recall values exclude abstained cases. It must not be presented as end-to-end serving-policy validation or evidence that all sparse records are handled safely.

## Feature ablation and experimental work

Compare the refill-only row with `all_signals` for the shipped-family ablation. The separately trained ablation ensemble need not exactly match the headline model. An `EXPERIMENTAL` row remains excluded from the shipped feature set. Gains are conditional on the simulator; intermediate additions need not help monotonically.

Historical numbers in source comments and `artifacts/benchmark.json` are separate development experiments. They are not the current submission results and were not all rerun as part of the submission corrections. Current claims come from `artifacts/metrics.json` and its generated report.

## Human input and hypotheses

Barrier inference uses illustrative expert rules, not a validated causal model. Clinical consequence weights are also illustrative policy values. Neither establishes the cause of a patient's difficulty or the correct clinical response.

New patient-portal reports are stored separately and do not affect predictions. Existing symptom questionnaires in routine records do feed the model. Portal adoption, report reliability and the effect of showing reports to clinicians have not been evaluated.

Clinician feedback is stored, but automated recalibration/retraining is not implemented. Any future use of feedback would need reliable labels, governance and a fresh evaluation.

## Engineering and deployment

- The demo API has no authentication, authorisation or verified audit identity. Switching views is not access control. Use synthetic records only.
- The engine scores and holds the cohort in memory at startup. Larger panels require a measured batch-scoring and persistence design.
- There is no live hospital/pharmacy/FHIR connector. Mapping real records requires substantial data validation.
- A functional render harness does not replace browser, accessibility or presentation-machine testing.
- Seeds improve repeatability, but dependencies and hardware can change timing and results. The recorded environment is pinned separately.

A real pilot requires agreed record access, privacy/security controls, clinical oversight, and appropriate approvals. No deployment readiness, regulatory classification, customer, revenue, savings or health improvement is claimed.
