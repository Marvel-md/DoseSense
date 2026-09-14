# Presentation and demo script

This is content for the official hackathon template, not a replacement slide layout. Follow the supplied template, anonymity rules and final PDF requirements in [submission-checklist.md](submission-checklist.md).

## Results to use consistently

<!-- BEGIN GENERATED RESULTS -->
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
<!-- END GENERATED RESULTS -->

Use these current values wherever numbers appear. Do not mix them with historical benchmark runs. PR-AUC is not “accuracy”; ECE is not a maximum error bound; subgroup alert rate is not recall. See [evaluation.md](evaluation.md) for denominators and timing limitations.

## Five-person Round 1 script

Target a final video below 3:00, including transitions and the prototype demonstration. The times below are rehearsal allocations, not a measured recording duration. Every registered member must appear on camera. Manuaditya operates and explains the prototype; assign the other four segments to the remaining registered members. Narration is in English.

### Speaker 1 — problem (0:00–0:25)

A prescription tells us what treatment should happen. It does not show what happens between appointments. A patient may struggle with cost, access or side effects. Routine care records contain clues, but those clues are scattered. DoseSense brings them together to help a care team decide who needs a conversation.

### Speaker 2 — approach (0:25–0:50)

We compare changes with each patient's own history. DoseSense combines refill records, prescription changes, appointments, existing symptoms, laboratory trends and optional activity data. It distinguishes temporary irregularities from sustained changes and says when evidence is insufficient. It estimates a possible concern; it never proves that a dose was missed.

### Manuaditya — prototype (0:50–1:45)

[Show your face, then screen-share the running dashboard. Prepare the three cases before recording.]

This is our clinician worklist. Let me open a case where dispensing coverage looks healthy. The other recorded signals help explain why the model still raises a possible concern. The display separates recorded facts, the model estimate and a possible barrier.

Now compare a temporary irregularity: the pattern returned toward baseline. Next, this sparse record shows why insufficient evidence is a separate state.

Behind the dashboard is a calibrated LightGBM ensemble. Its features use trailing windows, and patients are separated across training and testing. Clinicians can record feedback. Patients can add context through the portal, but those new reports do not change the prediction. Existing symptom questionnaires remain part of the routine records.

[Read only what the displayed cases actually show. If a walkthrough case is unavailable, use a verified alternative and adjust the narration before recording.]

### Speaker 4 — evidence (1:45–2:15)

<!-- BEGIN GENERATED PITCH RESULTS -->
On 148 held-out synthetic patients, DoseSense achieved a PR-AUC of 0.900, compared with 0.670 for our refill baseline. The relative reduction in model-threshold false alerts was 49.7%. These are simulator results, not clinical validation.
<!-- END GENERATED PITCH RESULTS -->

### Speaker 5 — feasibility, business and close (2:15–2:50)

Our proposed first customer is a chronic-care clinic with a pharmacist reviewing repeat prescriptions. We would begin with an approved evaluation and measure review time, false alerts and useful follow-ups. Our proposed business model is a clinic subscription with integration support, tested through pilot feedback. Those are plans, not existing customers or proven savings. DoseSense helps teams investigate possible barriers with evidence, uncertainty and human review.

Reserve the remaining time for transitions. Record a complete rehearsal and check the actual exported duration. A script alone cannot verify timing, camera participation, audio quality or video-link access.

## Slide content mapping

Place this content into the matching sections of the official template; do not add custom layouts or slides if the template forbids it.

| Topic | Content |
|---|---|
| Problem | Limited visibility between visits; collection does not prove ingestion |
| Solution | Combine indirect records, personal baselines and explicit uncertainty |
| Innovation | Temporary/persistent distinction, barrier hypotheses, abstention and separated evidence |
| Prototype | Running worklist, patient timeline, explanation, feedback and portal |
| Architecture | Python/LightGBM → FastAPI → vanilla JS; SQLite for feedback |
| Evaluation | Generated results above; synthetic data and patient-level splits |
| Feasibility | Current laptop prototype; staged record integration and approved validation |
| Marketing | Target clinic decision-makers with a synthetic demo and workflow interviews |
| Monetisation | Proposed subscription and integration support; pricing requires validation |
| Limitations | Simulator bias, confounders, open demo API, no clinical validation |

## Technical questions

**Why LightGBM?** The current inputs are tabular features with missingness, and the ensemble supports efficient scoring and grouped explanations. We have not established superiority over an untested sequence model.

**What does “without asking” mean?** No manual dose log is required. New portal self-reports do not enter the model. Existing routine symptom questionnaires do.

**How do you avoid leakage?** Feature windows end at the snapshot date. Training, calibration and test patients are disjoint. Tests compare full and truncated records and check hidden labels are not served.

**Can the system know a dose was missed?** No. The result is an inference from indirect records and requires human review.

**Does it detect earlier?** Do not claim prospective early detection. The timing calculation is retrospective, has a defined flag-selection convention and uses a coarse snapshot cadence. Show the actual table if asked.

**Does the portal have secure patient accounts?** No. The role selector demonstrates two interfaces. Authentication and authorisation are deployment work still to do.

**Does feedback train the model?** No. Feedback and portal reports are stored and displayed; automatic retraining is not implemented.

**What does the alert reduction measure?** Thresholded model flags on synthetic snapshots. It is not a measured reduction in clinician calls or a complete evaluation of dashboard policy.

**How will this make money?** See [business-plan.md](business-plan.md): proposed buyer, outreach, subscription hypothesis, costs and pilot milestones. No paying customer or financial outcome is claimed.

## Live rehearsal

1. Run the reproduction pipeline and `python scripts/check_demo.py` on the presentation machine.
2. Start the app and wait for `/api/health` to return `status: ok`.
3. Inspect `/api/demo`; open the selected cases and confirm their descriptions match the screen.
4. Use synthetic records only. Show the observed facts before explaining the estimate.
5. Record an uninterrupted rehearsal with the actual five members and screen share.
6. Review the export and follow the PDF/video checklist before submission.
