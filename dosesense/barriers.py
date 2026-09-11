"""
Barrier inference: moving from "this looks like a problem" to "this looks like
a problem of a particular kind".

This is the layer that changes what a care team actually does. A probability
that a patient's medication is not being taken as prescribed tells a pharmacist
to phone someone. A hypothesis that the pattern is consistent with a supply
problem at the dispensing pharmacy tells them what to ask about, and answers a
different question - why - which the brief explicitly asks for and which the
conventional refill-gap metric cannot address at all.

Three design commitments.

*It is inference, not attribution.* The output is a hypothesis with supporting
evidence, always presented as "consistent with", never as a determination about
a person's behaviour or motives.

*It abstains.* When no hypothesis clears a floor, the answer is that there is
not enough evidence to suggest one. When the top two hypotheses are close, both
are shown rather than the system picking a winner it cannot justify. Guessing a
reason and being wrong is worse than saying nothing, because a clinician acting
on a wrong reason asks the wrong question and may close the case.

*Rules, not a second model.* These are transparent scored rules rather than a
learned classifier. There is no ground truth for barriers in a real health
record - nobody labels "this gap was caused by cost" - so a supervised model
here would be fitting the simulator's own generative assumptions and reporting
its accuracy as if it meant something. Rules are honest about being an expert
prior, and a clinician can read and disagree with them.

Note on inputs: patient context that is deliberately excluded from the model
matrix - insurance tier, distance to the dispensing pharmacy - is used here.
That is intentional. As supporting evidence shown to a human alongside a stated
hypothesis it is useful context; as a silent model input it would let
socioeconomic proxies drive the score with no visibility. See docs/limitations.md.
"""

from __future__ import annotations

import numpy as np

from . import config as C


def _clip(v: float) -> float:
    return float(np.clip(v, 0.0, 1.0))


def infer_barriers(features: dict, patient: dict, events: list[dict], snapshot_day: int,
                   changes: dict | None = None) -> dict:
    """Score every barrier hypothesis and return a ranked, evidence-backed result."""
    changes = changes or {}
    f = features
    recent_events = [e for e in events if snapshot_day - 240 <= e.get("day", -1) <= snapshot_day]

    def had(event_type: str, within: int = 240) -> list[dict]:
        return [e for e in recent_events
                if e.get("event_type") == event_type and snapshot_day - within <= e["day"]]

    scores: dict[str, float] = {}
    evidence: dict[str, list[str]] = {b: [] for b in C.BARRIERS}

    # ---------------- ACCESS ----------------
    s = 0.0
    supply = had("pharmacy_supply_issue")
    if supply:
        s += 0.42
        evidence["ACCESS"].append(
            f"{len(supply)} dispensing-pharmacy stock issue(s) recorded in the last 8 months")
    if f.get("refill_overdue_ratio", 0) > 1.4:
        s += 0.22
        evidence["ACCESS"].append(
            f"Currently {f['refill_overdue_ratio']:.1f} supply-periods past the expected refill date")
    if f.get("refill_delay_excess", 0) > 6:
        s += 0.20
        evidence["ACCESS"].append(
            f"Recent dispensing runs {f['refill_delay_excess']:.0f} days later than this patient's own baseline")
    dist = float(patient.get("distance_to_pharmacy_km", 0) or 0)
    if dist > 8 and f.get("refill_last_delay", 0) > 7:
        s += 0.14
        evidence["ACCESS"].append(
            f"Dispensing pharmacy is {dist:.1f} km away, alongside late collection")
    if f.get("refill_consecutive_late", 0) >= 3:
        s += 0.18
        evidence["ACCESS"].append(
            f"{f['refill_consecutive_late']:.0f} consecutive collections were late, "
            f"a pattern more consistent with obtaining the medicine than with forgetting a dose")
    if had("life_disruption"):
        s += 0.10
        evidence["ACCESS"].append("Travel or disruption away from the usual pharmacy is noted in the record")
    scores["ACCESS"] = _clip(s)

    # ---------------- COST ----------------
    s = 0.0
    if had("insurance_change", within=300):
        s += 0.46
        evidence["COST"].append("Insurance or plan change recorded before the behaviour changed")
    if had("prescription_abandoned"):
        s += 0.34
        evidence["COST"].append("A prescription was issued but not collected")
    if patient.get("insurance_tier") == "Self-pay" and f.get("refill_delay_excess", 0) > 4:
        s += 0.18
        evidence["COST"].append("Self-pay patient with dispensing later than their own baseline")
    if f.get("rx_n_medications", 0) >= 2 and f.get("refill_consecutive_late", 0) >= 2:
        s += 0.10
        evidence["COST"].append("Repeated late collection across a multi-medication regimen")
    scores["COST"] = _clip(s)

    # ---------------- SIDE EFFECT / TOLERABILITY ----------------
    s = 0.0
    starts = had("medication_added", within=200) + had("dose_change", within=200)
    sym_rise = f.get("sym_delta", 0.0)
    if starts and sym_rise > 1.0:
        s += 0.52
        newest = max(e["day"] for e in starts)
        evidence["SIDE_EFFECT"].append(
            f"Symptom burden rose by {sym_rise:.1f} points after a regimen change "
            f"{snapshot_day - newest} days ago")
    elif starts and sym_rise > 0.4:
        s += 0.22
        evidence["SIDE_EFFECT"].append("Mild symptom increase follows a recent regimen change")
    if starts and f.get("refill_delay_excess", 0) > 3:
        s += 0.20
        evidence["SIDE_EFFECT"].append("Collection became less timely after the regimen change")
    sym_cp = changes.get("symptom") or {}
    if sym_cp.get("status") == C.CHANGE_PERSISTENT and starts:
        s += 0.16
        evidence["SIDE_EFFECT"].append("The symptom increase has persisted rather than settled")
    scores["SIDE_EFFECT"] = _clip(s)

    # ---------------- REGIMEN COMPLEXITY ----------------
    s = 0.0
    doses = f.get("rx_total_daily_doses", 0)
    if doses >= 5:
        s += 0.40
        evidence["REGIMEN_COMPLEXITY"].append(f"{doses:.0f} scheduled doses per day")
    elif doses >= 3:
        s += 0.20
        evidence["REGIMEN_COMPLEXITY"].append(f"{doses:.0f} scheduled doses per day")
    if f.get("rx_n_medications", 0) >= 3:
        s += 0.22
        evidence["REGIMEN_COMPLEXITY"].append(
            f"{f['rx_n_medications']:.0f} concurrent tracked medications")
    if f.get("rx_changes_180", 0) >= 3:
        s += 0.22
        evidence["REGIMEN_COMPLEXITY"].append(
            f"{f['rx_changes_180']:.0f} regimen changes in the last 180 days")
    if f.get("refill_interval_cv", 0) > 0.35 and doses >= 3:
        s += 0.12
        evidence["REGIMEN_COMPLEXITY"].append("Irregular collection intervals on a frequent-dosing regimen")
    scores["REGIMEN_COMPLEXITY"] = _clip(s)

    # ---------------- CARE ENGAGEMENT ----------------
    s = 0.0
    if f.get("appt_consecutive_missed", 0) >= 2:
        s += 0.44
        evidence["CARE_ENGAGEMENT"].append(
            f"{f['appt_consecutive_missed']:.0f} consecutive follow-ups not attended")
    elif f.get("appt_missed_rate_180", 0) > 0.4:
        s += 0.22
        evidence["CARE_ENGAGEMENT"].append(
            f"{f['appt_missed_rate_180'] * 100:.0f}% of follow-ups missed over 180 days")
    if f.get("appt_days_since_attended", 0) > 210:
        s += 0.26
        evidence["CARE_ENGAGEMENT"].append(
            f"{f['appt_days_since_attended']:.0f} days since the last attended review")
    if f.get("appt_missed_rate_excess", 0) > 0.25:
        s += 0.18
        evidence["CARE_ENGAGEMENT"].append("Attendance has worsened against this patient's own history")
    scores["CARE_ENGAGEMENT"] = _clip(s)

    # ---------------- resolve ----------------
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top, top_score = ranked[0]
    runner, runner_score = ranked[1]

    if top_score < C.BARRIER_MIN_SCORE:
        return {
            "primary": "UNKNOWN",
            "primary_label": C.BARRIER_LABELS["UNKNOWN"],
            "primary_score": round(top_score, 3),
            "confidence": C.CONFIDENCE_INSUFFICIENT,
            "evidence": [],
            "competing": [],
            "all_scores": {k: round(v, 3) for k, v in scores.items()},
            "note": ("The available signals do not point clearly to any one barrier. "
                     "Ask the patient rather than assuming a reason."),
        }

    competing = []
    if top_score - runner_score < C.BARRIER_MIN_MARGIN and runner_score >= C.BARRIER_MIN_SCORE:
        competing = [{
            "barrier": runner,
            "label": C.BARRIER_LABELS[runner],
            "score": round(runner_score, 3),
            "evidence": evidence[runner],
        }]

    if top_score >= 0.66 and not competing:
        conf = C.CONFIDENCE_HIGH
    elif top_score >= 0.48:
        conf = C.CONFIDENCE_MEDIUM
    else:
        conf = C.CONFIDENCE_LOW

    return {
        "primary": top,
        "primary_label": C.BARRIER_LABELS[top],
        "primary_score": round(top_score, 3),
        "confidence": conf,
        "evidence": evidence[top],
        "competing": competing,
        "all_scores": {k: round(v, 3) for k, v in scores.items()},
        "note": ("Presented as a hypothesis consistent with the recorded signals. "
                 "It is not a determination about the patient."),
    }
