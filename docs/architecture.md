# Architecture

## Layers

```mermaid
flowchart TD
    subgraph SRC["Routine care records (synthetic)"]
        A1[Pharmacy dispensing]
        A2[Prescriptions & changes]
        A3[Appointment attendance]
        A4[Symptom questionnaires]
        A5[Laboratory results]
        A6[Activity & sleep summaries]
        A7[Care events: supply, insurance, admissions]
    end

    SRC --> NORM[Normalise to a per-patient timeline<br/>features.PatientSeries]

    NORM --> SNAP["Snapshot at day t<br/><b>trailing windows only</b>"]

    SNAP --> BASE[Personal baseline<br/>this patient's own history]
    SNAP --> FEAT[47 causal features<br/>7 signal families]

    BASE --> CP[Change-point detection<br/>stable / temporary / persistent<br/>+ provisional]
    CP --> FEAT

    FEAT --> ENS[Bagged LightGBM x5<br/>resampled by patient]
    ENS --> CALIB[Isotonic calibration<br/>disjoint patient split]
    ENS --> SD[Ensemble disagreement]

    CALIB --> PROB[Calibrated probability]
    SD --> UNC
    FEAT --> UNC[Uncertainty & abstention<br/>evidence families x disagreement x completeness]

    PROB --> DEC{Alert?}
    UNC --> DEC
    CP --> DEC

    FEAT --> BAR[Barrier hypothesis<br/>scored rules, abstains]
    CTX[Patient context<br/>insurance, pharmacy distance] --> BAR

    POL[["Declared clinical policy<br/>config.py consequence table"]] --> PRIO
    PROB --> PRIO[Priority<br/>likelihood x consequence x confidence]
    UNC --> PRIO

    FEAT --> SHAP[SHAP attribution<br/>grouped by family]

    DEC --> ASSESS
    BAR --> ASSESS
    PRIO --> ASSESS
    SHAP --> ASSESS
    ASSESS[["Assessment<br/>observed | inferred | hypothesis | action"]]

    ASSESS --> API[FastAPI]
    API --> UI[Dashboard worklist]
    UI --> FB[Clinician adjudication]
    FB --> STORE[(SQLite)]
    STORE -.->|observed false-positive rate<br/>designed, not yet wired| UNC

    GT[(Hidden adherence state)] -.->|evaluation only,<br/>never served| EVAL[Evaluation]
    PROB -.-> EVAL

    classDef policy fill:#F3F1F9,stroke:#5B4C9A
    classDef hidden fill:#F1F4F7,stroke:#78889A,stroke-dasharray:4 3
    class POL policy
    class GT,EVAL hidden
```

## The boundary that matters most

Three things are kept structurally separate, and the separation is enforced rather than documented:

**Observation vs inference vs hypothesis.** Every assessment is a four-key object — `observed`,
`inferred`, `hypothesis`, `action` — and the keys are rendered in that order in the interface with
visually distinct left rules. `test_observed_inferred_hypothesis_are_separate_keys` fails the build
if a model estimate appears inside the observed block. The purpose is to make it structurally
difficult for a reader to mistake an inference for a fact.

**Learned vs declared.** The model estimates likelihood. Clinical severity, alert thresholds and
priority tiers live in `dosesense/config.py` and are exposed at `GET /api/config`. A care team must
be able to change what the system cares about without retraining it, and an auditor must be able to
read the policy without reading the model.

**Ground truth vs serving.** The hidden adherence trajectory is written to a separate file, loaded
only by the evaluation path, and the API loads the cohort with `include_ground_truth=False`. The
generator's archetype label is stripped from every patient payload.
`test_ground_truth_never_served` walks every string in 25 patient payloads asserting no latent
state name appears.

## Module responsibilities

| Module | Responsibility | Depends on |
|---|---|---|
| `config.py` | Declared clinical policy, thresholds, taxonomies, language guardrails | nothing |
| `datagen.py` | Causal simulator, 12 archetypes, hidden trajectories | config |
| `changepoint.py` | Personal-baseline segmentation, temporary vs persistent | config |
| `baseline.py` | Faithful PDC / MPR comparator | — |
| `features.py` | 47 causal features, observed facts, family coverage | config, baseline, changepoint |
| `panel.py` | Snapshot panel, labelling from hidden state, patient-level splits | config, baseline, features |
| `model.py` | Bagged calibrated ensemble, persistence, importance | config, features |
| `barriers.py` | Scored barrier hypotheses with abstention | config |
| `risk.py` | Evidence strength, confidence, abstention, priority | config |
| `explain.py` | SHAP grouped by family, plain-language names | features |
| `evaluate.py` | Metrics, calibration, lead time, burden, ablation, subgroups | config, features, panel |
| `pipeline.py` | Assessment assembly, cohort scoring, serving engine | all of the above |

`dosesense/` has no dependency on FastAPI, the dashboard, or any CLI. The engine is importable on its
own, which is what makes the test suite able to exercise it without a server.

## Request path

The engine scores the cohort once at start-up and holds current assessments, probability
trajectories and evidence timelines in memory. Scoring 600 patients takes ~35 seconds; doing it per
request would make the interface feel broken.

This does not scale to a real panel and is called out in `docs/limitations.md`. The production shape
is batch scoring to a store with the API reading from it — a contained change, since `pipeline.py`
already separates scoring from querying.

## Reuse beyond adherence

The generic spine — multi-signal ingestion, personal baselines, change detection, evidence
timelines, calibrated uncertainty with abstention, consequence-weighted prioritisation,
grouped explanation, human feedback — carries no adherence-specific logic. Domain specifics are
confined to `config.py` (conditions, medications, consequence weights), the feature definitions, and
the barrier rules.

The shape recurs across problems where indirect signals must be fused into a triage decision under
uncertainty: medicine supply-chain shortage detection, insider-threat behavioural analytics,
procurement anomaly review. In each, the task is to separate a meaningful shift from normal
variation, quantify confidence, and hand a human a ranked, explained queue.
