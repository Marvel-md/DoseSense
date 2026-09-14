# Proposed business and pilot plan

All commercial statements below are hypotheses for the hackathon. No customer, partnership, price acceptance, revenue or cost saving has been validated.

## Customer and value proposition

**Proposed initial buyer:** the clinical lead or operations manager of a chronic-care clinic with repeat-prescription patients and a pharmacist/care coordinator doing follow-up.

**Daily user:** the pharmacist or care coordinator reviewing the worklist, evidence and possible barriers. Patients benefit indirectly from better-informed conversations; the prototype has not measured that benefit.

**Problem to validate:** staff currently inspect several records manually and need a manageable review list. Interview staff to establish their actual workflow, available records, time cost and acceptable false-alert burden before assuming demand.

## Outreach and media

1. Invite clinic decision-makers and pharmacist networks to short workflow interviews. Use a synthetic demo, never real patient screenshots in outreach.
2. Publish a concise demo video explaining the three visible behaviours: concern, temporary irregularity and insufficient evidence. Include the synthetic-data qualifier with every performance claim.
3. Offer a supervised evaluation proposal to interested clinics once data/security/clinical prerequisites are met.
4. Track interview acceptance, qualified demonstration requests, pilot interest and reasons for rejection. No conversion-rate target is presented as achieved.

## Revenue hypothesis

A clinic subscription could cover a defined panel size, with a separate fee for record mapping, onboarding and support. Panel-size bands could align price with workload. Validate willingness to pay and procurement constraints through interviews; do not assign an arbitrary price and call it market evidence.

The economic test is whether clinic value exceeds deployment and support cost. Proposed cost categories are hosting, integration work, monitoring, security, training and ongoing support. No gross margin or saving is claimed until these are measured.

## Staged roadmap

The time windows are proposed planning estimates, dependent on staffing and approvals.

| Stage | Proposed timing | Deliverable and decision gate |
|---|---|---|
| Workflow discovery | Weeks 1–2 | Interviews with clinic staff; documented data availability and review workflow; proceed only with a clear use case |
| Integration and governance design | Weeks 3–6 | Approved access plan, identity/access controls, audit design, data mapping and agreed evaluation protocol |
| Offline evaluation | After approvals | Evaluate authorised records with independent clinician adjudication; assess missing data, false alerts and subgroup gaps |
| Shadow-mode pilot | Proposed 4–6 weeks | Staff inspect outputs without autonomous treatment changes; measure usefulness and review burden |
| Commercial decision | After pilot | Decide whether to iterate, stop or offer a paid service based on validated value, cost and governance requirements |

## Pilot measurements

Agree acceptance thresholds with the participating clinic before the evaluation. Measure:

- Review time per record compared with its existing workflow.
- Alert precision and recall where a defensible adjudicated reference is available.
- False alerts, repeated alerts, abstentions and missed concerns.
- Useful follow-ups and clinician disagreement with barrier hypotheses.
- Performance across data-completeness and relevant patient subgroups.
- Integration effort, uptime, support demand and willingness to pay.

Separate outcomes that have a valid reference label from clinician opinion. An adherence label inferred from the same records is not direct proof of ingestion. A pilot must not turn the simulator's hidden labels into a claim about real-world ground truth.

## Inclusivity and scaling

Detection does not require a patient smartphone, wearable or manual dose log. Missing records can still reduce performance or trigger abstention; that is a limitation to measure. Multilingual support and accessible patient communication are proposed improvements, not completed features.

Start with one agreed record format and workflow. Scale only after measuring scoring cost and operational performance. The current in-memory startup design is a prototype; a larger service would use scheduled batch scoring, persisted assessments and monitored data pipelines.
