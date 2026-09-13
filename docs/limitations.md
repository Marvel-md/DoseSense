# Limitations

Read this before believing any number in this repository.

The project is a working demonstration of a method, evaluated on data we generated ourselves. That
is a meaningful thing and it is also a limited thing, and the difference matters.

---

## 1. The data is synthetic, and that bounds every claim

**What the evaluation establishes:** that the method works on data with the structure we believe real
data has. The pipeline is causally clean, the splits are honest, the baseline is faithfully
implemented, and the model beats it substantially on that data.

**What it does not establish:** real-world performance. Nothing here is clinical validation.

The specific risk is circularity. We wrote the simulator *and* the model. Where the simulator's
assumptions are wrong, the model has been optimised against a fiction. We have taken three
deliberate steps against that, and they reduce the problem without removing it:

- **Adversarial archetypes.** Four patterns are adherent for their entire record by construction,
  two of them designed specifically to fool the system (`MISLEADING_ANOMALY`, `DISEASE_PROGRESSION`).
  The false-alert rate on them is measured against a known truth.
- **A confounder the model must survive.** Disease progression produces worsening symptoms and labs
  with a flawless dispensing record. A model that cannot separate *"the treatment isn't working"*
  from *"the treatment isn't being taken"* fails on these rows.
- **We let the evaluation overrule the pitch.** The first ablation showed multi-signal fusion added
  nothing (PR-AUC 0.9561 → 0.9572). Rather than quietly dropping the ablation, we found the
  simulator flaw causing it. That is the process working, but it is also proof that simulator
  artefacts can produce confident wrong conclusions — and we only caught that one because we
  happened to run the right diagnostic.

**Specific ways the simulator is likely wrong:**

| Simplification | Likely direction of error |
|---|---|
| Latent state is a three-level discrete variable | Real adherence is continuous and dose-specific |
| Symptom and lab response to non-adherence is a smooth lagged exposure curve | Real responses are threshold-like, drug-specific and far noisier |
| Exactly one condition per patient | Real patients have multimorbidity and competing regimens |
| No care-team intervention in the loop | In reality a clinician intervenes, which changes the trajectory the model is predicting |
| Dispensing records are complete and accurate | Real pharmacy data has duplicates, transfers between pharmacies, cash purchases and missing days-supply |
| Non-adherence is unrelated to prior detection | In deployment the system's own alerts change behaviour, so the data becomes non-stationary |

The last of those is the most serious for a real deployment and is not modelled at all.

## 2. Detection lead time does not beat the baseline

This is our weakest result and the one most at odds with the intuition behind the project.

Median time to detection is **one snapshot interval (30 days) for both DoseSense and the PDC rule**.
We do not detect earlier on the median. We catch a larger share of cases at or before the onset
snapshot (24.3% vs 14.7%), with roughly a third of the false alerts, but "detects problems sooner"
is not a claim the data supports.

The cause is partly structural: a 30-day rescoring cadence quantises lead time into 30-day
increments, so the median is a coarse instrument. It is also partly real — silent non-adherence is
detected late because laboratory markers move slowly, and those cases drag the median. A shorter
cadence would resolve the measurement issue and would increase alert volume, which we have not
evaluated.

## 3. Recall on silent non-adherence is low

The archetype the project is named after is detected at **0.189** against **0.105** for the PDC rule.
That is a 1.8× improvement on a pattern the standard metric essentially cannot address, and it is
still a minority of cases caught.

This is genuinely hard. When collection behaviour is normal, the only evidence is that clinical
markers drift — and clinical markers also drift from disease progression, dosing errors, drug
interactions and measurement noise. Improving here means either more signal (dispensing-event
granularity, smart packaging, pharmacy interaction notes) or accepting more false alerts on
progression cases. We chose not to trade away specificity for a headline recall number.

## 3b. The system is worse than the baseline on disease progression

Pooled across seven seeds, `DISEASE_PROGRESSION` — a patient deteriorating
clinically with a flawless dispensing record — draws a **0.10 alert rate
(95% CI 0.06–0.15) against 0.06 for the PDC rule**. Every one of those is a
false alert against a known truth.

The mechanism is not a bug and cannot be tuned away without giving up the
system's main advantage. PDC reads only dispensing records, which are clean for
these patients, so it stays quiet by being blind. DoseSense reads laboratory
drift and symptom trends, which genuinely are moving, so it sometimes concludes
a medication-taking problem when the real answer is that the disease is
advancing. The same sensitivity is what produces a 0.27 detection rate on silent
non-adherence where PDC manages 0.03.

Reducing it would mean either giving the dispensing record veto power over the
clinical signals — which reintroduces exactly the blindness we set out to fix —
or adding a progression-versus-adherence discriminator we have not built. A
real deployment would want the second. We have not attempted it.

## 4. Multi-signal fusion is a modest gain, not a transformation

Refill data alone reaches PR-AUC 0.873; all seven signal families reach 0.907. The +0.034 is real,
monotone and concentrated in the symptom family, but dispensing data carries most of the
discriminative signal and we say so rather than overselling fusion.

False alerts also do **not** fall monotonically as signals are added — more inputs mean more ways to
be wrong. The extra families earn their place by reaching cases refill data cannot and by enabling
barrier inference, not by lifting AUC dramatically.

## 5. Subgroup disparities are measured and unaddressed

Sex and insurance tier are excluded from the model's inputs. That prevents the model reading them
directly; it does not prevent a correlated feature reproducing the same disparity, which is why we
measure rather than assume.

- Gaps by **sex** are negligible (recall gap 0.011).
- Gaps by **insurance tier** are modest (0.103) and favour self-pay patients, likely an artefact of
  how the simulator couples cost barriers to that tier.
- The largest gaps are by **age band** (0.119) and **condition** (0.169).

We tested removing age from the feature matrix entirely. It cost 0.003 PR-AUC and the age-band gap
got slightly *worse*. So the disparity is not the model reading age directly — it reflects genuine
differences in base rate and record richness across bands. **We have not fixed it**, and we are
reporting a measurement rather than a solution.

A real deployment would need group-wise threshold calibration and a standing monitoring
requirement, neither of which is implemented here.

## 6. Barrier inference is an expert prior, not a validated model

The barrier rules have **no ground-truth validation of any kind**, and cannot have one from this
data. Nobody labels "this gap was caused by cost" in a real health record.

We chose rules over a learned model precisely because a supervised barrier classifier trained on our
simulator would be fitting our own generative assumptions and reporting its accuracy as if it meant
something. The rules are transparent and a clinician can disagree with them, which is the honest
posture. But the reported barrier distributions describe our rules operating on our simulator, and
say nothing about whether the barriers are real.

## 6b. The patient portal is outside the evaluated system

Self-reported data is stored, shown to clinicians, and never used for
prediction. That means none of the reported metrics say anything about whether
self-reports are accurate, whether patients would use the portal, or whether
seeing their own record changes behaviour. The portal is a design proposal with
a working implementation, not a validated intervention.

The deliberate exclusion also has a cost worth naming: a patient who tells us
plainly that they stopped taking a medicine cannot move the estimate, even
though that is strong evidence. Wiring it in would require re-establishing which
self-reports are reliable, which is a research question and not one we can
answer from synthetic data.

## 6c. The abstention threshold is probably set slightly too low

The data-richness audit shows the weakest performance in the `partial` stratum — records with
enough content to trigger a commitment but not enough to be reliable. Committed recall there is
0.699 against 0.882 on `sparse` records, where the system abstains three times in four.

That pattern says the abstention rule is protecting the thinnest records correctly and letting
through a band just above them that it should not. We have not retuned the threshold, because doing
so on the same synthetic data that produced the observation would be fitting to our own simulator.
It is flagged as the most actionable known defect.

## 6d. Cross-signal velocity was tried and rejected

Documented in the README and retained in `features.py` as `EXPERIMENTAL_FAMILIES`. It is not part
of the shipped model. The reason to mention it under limitations as well: the approach we hoped
would separate silent non-adherence from disease progression did not work, so that separation
remains unsolved rather than merely unattempted.

## 7. Statistical power

The test set is **148 patients / 1,924 snapshots**. Rarer archetypes have very few test patients —
`MISLEADING_ANOMALY` contributes 26 snapshots from roughly two patients. Per-archetype rates for the
rare patterns are indicative, not precise, and they move noticeably with the random seed.

The maximum calibration error of 0.423 sits in a bin holding one snapshot. That is small-sample
noise, not miscalibration; the size-weighted expected error of 0.024 is the figure to read.

## 8. The label threshold is a judgement call

"Non-adherent for at least 34% of the trailing 90 days" defines what counts as meaningful. Reasonable
clinicians would set it differently, and every reported metric is conditional on it. We have not run
a sensitivity analysis across label definitions, which is the most obvious missing robustness check
in the evaluation.

## 9. Engineering limits

- **No authentication, authorisation, audit log or encryption.** The API is open. This is a
  demonstration, not a deployable clinical system, and it would fail any health-information
  governance review as it stands.
- **The engine scores the cohort in memory at start-up** (~35 seconds for 600 patients). This does
  not scale to a real panel of tens of thousands and would need batch scoring with a persistence
  layer.
- **No FHIR ingestion.** The data model is FHIR-*compatible* in shape but there is no real connector,
  and mapping real messy pharmacy data is substantially harder than the schema suggests.
- **Feedback is recorded but does not yet retrain anything.** The loop is closed as far as storage
  and an observed false-positive rate; using it to adjust thresholds is designed and not built.
- **The clinical consequence table is illustrative.** The values are our judgement, not validated
  severity weights, and a real deployment would need them set by a clinical committee.

## 10. What is explicitly not claimed

- No clinical outcome improvement. None was measured.
- No cost saving. None was measured.
- No prospective or external validation.
- No regulatory assessment. A system that influences clinical attention plausibly falls within
  medical-device software regimes in several jurisdictions; we have not analysed this.
- No claim that any output establishes whether a patient took a medicine. The system cannot know
  that, and the test suite fails the build if any output implies otherwise.

---

## The honest summary

This is a well-engineered demonstration of a defensible method, with an evaluation designed to
surface its own weaknesses rather than hide them. It beats the metric currently in clinical use by a
substantial margin *on our data*, it is calibrated, it abstains when it should, and it distinguishes
a temporary irregularity from a sustained pattern.

Turning it into something that could touch a real patient would require prospective validation on
real records, clinical governance, security and privacy engineering, group-wise fairness
calibration, and a serious answer to the non-stationarity introduced by the system's own alerts.
None of that is here.
