"""
Uncertainty, abstention and clinical prioritisation.

Two separate problems live here, and conflating them is the most common way a
model like this fails in practice.

*How sure are we?* A calibrated probability is necessary but not sufficient.
0.62 derived from four converging signal families over eighteen months of
history means something quite different from 0.62 derived from a single late
refill on a patient with four months of record. Both would appear identically in
a ranked list sorted by probability. So confidence is built from three things:
how many independent signal families carry evidence, how much a bagged ensemble
disagrees with itself about this specific patient, and how complete the record
is. Where evidence is too thin, the system abstains rather than producing a
number that looks like knowledge.

*Who should be contacted first?* Not the highest probability. A pharmacist with
two hours has to weigh how likely the concern is, how much harm a sustained gap
would do for that condition and drug, and how much the estimate can be trusted.
An 88% concern on a thyroid supplement should not outrank a 61% concern on
warfarin. So priority is likelihood x consequence x confidence, and the
consequence term comes from the declared policy table in config, not from the
model - clinical severity is a governance decision that a care team must be able
to change without retraining anything.
"""

from __future__ import annotations

import numpy as np

from . import config as C

# Record-completeness expectations used to judge whether a snapshot is
# well-evidenced or merely quiet.
# The ensemble on this cohort disagrees by roughly 0.01-0.05 in probability, so
# scaling disagreement against 0.18 would make the term a constant. It is scaled
# against the spread actually observed, which is what makes it informative.
SD_SCALE = 0.07

EXPECTED_MIN = {
    "refill_n_fills_365": 6.0,
    "appt_n_365": 2.0,
    "lab_n_results": 3.0,
}


def evidence_strength(families: dict[str, bool], facts: list[dict]) -> dict:
    """Count independent informative signal families and total evidence weight."""
    informative = [k for k, v in families.items() if v]
    fact_families = {fa["family"] for fa in facts}
    # A family counts toward evidence strength only if it produced an actual
    # observation, not merely if data existed. Having six months of normal
    # laboratory results is data, not evidence of a concern.
    active = sorted(fact_families & set(informative))
    weight = float(sum(min(1.0, fa["weight"]) for fa in facts))
    return {
        "families_available": informative,
        "families_with_evidence": active,
        "n_families_available": len(informative),
        "n_families_with_evidence": len(active),
        "total_evidence_weight": round(weight, 3),
    }


def record_completeness(features: dict) -> float:
    """0-1 measure of how much history this snapshot actually rests on."""
    parts = []
    for key, expected in EXPECTED_MIN.items():
        parts.append(min(1.0, float(features.get(key, 0.0)) / expected))
    return float(np.mean(parts)) if parts else 0.0


def assess_confidence(probability: float, ensemble_sd: float, strength: dict,
                      features: dict, changes: dict | None = None) -> dict:
    """Decide how much weight to place on this estimate, or whether to abstain."""
    changes = changes or {}
    n_ev = strength["n_families_with_evidence"]
    completeness = record_completeness(features)
    provisional = any((c or {}).get("provisional") for c in changes.values())

    reasons: list[str] = []

    # "No evidence of a concern" and "not enough evidence to tell" are different
    # states and must not collapse into one. A patient with eighteen months of
    # complete records, every refill on time and stable markers is a *confident
    # negative*: the system knows a great deal about them and none of it is
    # worrying. Reporting that as "insufficient evidence" would be both wrong and
    # corrosive - a queue where half the panel is marked unknown teaches the care
    # team to ignore the column.
    #
    # Abstention is reserved for two situations that genuinely defeat the method:
    # a record too sparse to establish any personal baseline, or a probability
    # high enough to matter resting on a single signal family.

    if completeness < 0.34:
        reasons.append("The record is too sparse to establish a reliable personal baseline")
        return {
            "confidence": C.CONFIDENCE_INSUFFICIENT, "abstained": True,
            "completeness": round(completeness, 3),
            "ensemble_sd": round(float(ensemble_sd), 4), "reasons": reasons,
        }

    if probability >= C.ALERT_PROBABILITY_THRESHOLD and n_ev < C.MIN_FAMILIES_FOR_ALERT:
        reasons.append(
            f"The estimate is elevated but rests on {n_ev} signal "
            f"{'family' if n_ev == 1 else 'families'}; at least "
            f"{C.MIN_FAMILIES_FOR_ALERT} are required before raising a concern")
        return {
            "confidence": C.CONFIDENCE_INSUFFICIENT, "abstained": True,
            "completeness": round(completeness, 3),
            "ensemble_sd": round(float(ensemble_sd), 4), "reasons": reasons,
        }

    # For a low-probability snapshot, confidence rests on how much of the record
    # was available to look at. For an elevated one, it rests on how many
    # families independently corroborate the finding.
    if probability >= C.ALERT_PROBABILITY_THRESHOLD:
        corroboration = min(1.0, n_ev / 4.0)
    else:
        corroboration = min(1.0, strength["n_families_available"] / 4.0)
        if strength["n_families_available"] >= 4:
            reasons.append(
                f"{strength['n_families_available']} signal families were available to "
                f"examine and none indicates a departure from baseline")

    score = 0.0
    score += corroboration * 0.45
    score += (1.0 - min(1.0, float(ensemble_sd) / SD_SCALE)) * 0.30
    score += completeness * 0.25

    if provisional:
        score -= 0.14
        reasons.append("A change was detected too recently to tell whether it will persist")
    if float(ensemble_sd) > 0.12:
        reasons.append("Models trained on different data subsets disagree about this patient")
    if n_ev >= 4 and probability >= C.ALERT_PROBABILITY_THRESHOLD:
        reasons.append(f"{n_ev} independent signal families point the same way")
    if completeness > 0.85:
        reasons.append("A long, complete record supports the personal baseline")

    if score >= 0.78:
        conf = C.CONFIDENCE_HIGH
    elif score >= 0.52:
        conf = C.CONFIDENCE_MEDIUM
    else:
        conf = C.CONFIDENCE_LOW

    return {
        "confidence": conf,
        "abstained": False,
        "confidence_score": round(float(score), 3),
        "completeness": round(completeness, 3),
        "ensemble_sd": round(float(ensemble_sd), 4),
        "reasons": reasons,
    }


def priority(probability: float, consequence: float, confidence: str) -> dict:
    """Combine likelihood, declared clinical consequence and confidence into a tier."""
    weight = C.CONFIDENCE_WEIGHT.get(confidence, 0.5)
    score = float(probability) * float(consequence) * weight
    tier = C.PRIORITY_TIERS[-1][0]
    for name, floor in C.PRIORITY_TIERS:
        if score >= floor:
            tier = name
            break
    return {
        "priority_score": round(score, 4),
        "priority_tier": tier,
        "components": {
            "likelihood": round(float(probability), 4),
            "clinical_consequence": round(float(consequence), 3),
            "confidence_weight": weight,
        },
    }


def should_alert(probability: float, confidence: str, change_status: str | None = None) -> dict:
    """Decide whether this snapshot becomes a queued alert or stays a watch item.

    A high probability supported by thin evidence does not become an alert; it
    becomes a request for more information. This is where the brief's
    requirement to communicate uncertainty stops being a display choice and
    starts changing what the system does.
    """
    if confidence == C.CONFIDENCE_INSUFFICIENT:
        return {"alert": False, "state": "INSUFFICIENT_EVIDENCE",
                "message": ("Signals are present but too thin to raise a concern. "
                            "Monitoring; no action requested.")}
    if probability < C.ALERT_PROBABILITY_THRESHOLD:
        if change_status == C.CHANGE_TEMPORARY:
            return {"alert": False, "state": "TEMPORARY_IRREGULARITY",
                    "message": ("A departure from this patient's baseline was detected and has "
                                "since returned. No sustained pattern is evident.")}
        return {"alert": False, "state": "WITHIN_BASELINE",
                "message": "Behaviour is consistent with this patient's established pattern."}
    if change_status == C.CHANGE_TEMPORARY:
        return {"alert": True, "state": "RESOLVING",
                "message": ("Evidence suggests a recent adherence concern that appears to be "
                            "resolving. Worth a light-touch check rather than escalation.")}
    return {"alert": True, "state": "SUSTAINED_CONCERN",
            "message": ("Evidence suggests a sustained departure from this patient's own "
                        "medication-taking pattern. Recommended for clinician or pharmacist review.")}
