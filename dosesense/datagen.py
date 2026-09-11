"""
Synthetic longitudinal healthcare record generator.

DoseSense is trained and evaluated entirely on synthetic data. No real patient
record is used anywhere in this project.

The generator is a *causal* simulator, not a table of random numbers. For each
patient it first samples a hidden day-by-day adherence state trajectory, then
generates the observable record (pharmacy fills, appointments, patient-reported
symptoms, laboratory results, activity summaries, care events) conditioned on
that hidden trajectory. The hidden trajectory is written to a separate file and
is never available to the feature pipeline or the model at inference time. It
exists so that every claim about detection performance can be checked.

Why this matters for the problem statement: the challenge is to separate
temporary irregularity from a meaningful pattern. You can only demonstrate that
you have done so if you know, independently of your own model, which was which.

Nine behavioural archetypes are simulated, including one - MISLEADING_ANOMALY -
in which the patient is adherent for the entire record but produces a single
dramatic refill gap and a coincidental laboratory blip. A system that fires on
that patient has not solved the problem.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from . import config as C

INDEX_DATE = dt.date(2025, 3, 10)


def day_to_date(day: int) -> dt.date:
    return INDEX_DATE + dt.timedelta(days=int(day))


def date_to_day(d: dt.date) -> int:
    return (d - INDEX_DATE).days


# ---------------------------------------------------------------------------
# Archetypes
# ---------------------------------------------------------------------------
# Each archetype is a recipe for a latent state trajectory plus the contextual
# events that plausibly accompany it. Weight is the share of the cohort.

ARCHETYPES = {
    "ADHERENT_STABLE": {
        "weight": 0.26,
        "description": "Consistently adherent for the whole record.",
    },
    "OCCASIONAL_IRREGULARITY": {
        "weight": 0.14,
        "description": "Adherent with brief, self-correcting dips of one to two weeks.",
    },
    "TEMPORARY_INTERRUPTION": {
        "weight": 0.10,
        "description": "One interruption of three to six weeks, then a full return to baseline.",
    },
    "GRADUAL_DECLINE": {
        "weight": 0.11,
        "description": "Slow drift from adherent through partial to lapsed.",
    },
    "REPEATED_REFILL_DELAY": {
        "weight": 0.10,
        "description": "Persistently partial adherence expressed mainly as late refills.",
    },
    "ACCESS_BARRIER": {
        "weight": 0.09,
        "description": "Partial to lapsed adherence alongside pharmacy and travel access events.",
    },
    "COST_BARRIER": {
        "weight": 0.06,
        "description": "Adherence falls after an insurance change raises out-of-pocket cost.",
    },
    "SIDE_EFFECT": {
        "weight": 0.07,
        "description": "A new medication is started, symptoms rise, then adherence falls.",
    },
    "COMPLEX_REGIMEN": {
        "weight": 0.05,
        "description": "Many medications and frequent dosing, producing patchy partial adherence.",
    },
    "MISLEADING_ANOMALY": {
        "weight": 0.02,
        "description": "Adherent throughout, but with one large isolated refill gap and a lab blip.",
    },
    "DISEASE_PROGRESSION": {
        "weight": 0.05,
        "description": ("Fully adherent, but symptoms and laboratory markers worsen anyway as the "
                        "underlying condition progresses."),
    },
    "SILENT_NONADHERENCE": {
        "weight": 0.08,
        "description": ("Collects every prescription on time and does not take the doses. The "
                        "dispensing record looks exemplary; only the clinical signals move."),
    },
}


@dataclass
class PatientRecord:
    """Everything generated for one patient."""

    patient: dict
    medications: list = field(default_factory=list)
    refills: list = field(default_factory=list)
    appointments: list = field(default_factory=list)
    symptoms: list = field(default_factory=list)
    labs: list = field(default_factory=list)
    activity: list = field(default_factory=list)
    events: list = field(default_factory=list)
    state_segments: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Latent trajectory construction
# ---------------------------------------------------------------------------


def _segments_to_daily(segments: Iterable[dict], n_days: int) -> np.ndarray:
    """Expand run-length segments into a per-day array of state indices."""
    daily = np.zeros(n_days, dtype=np.int8)
    for seg in segments:
        lo = max(0, int(seg["start_day"]))
        hi = min(n_days, int(seg["end_day"]) + 1)
        daily[lo:hi] = C.STATES.index(seg["state"])
    return daily


def _merge(segments: list[dict]) -> list[dict]:
    """Collapse adjacent segments that share a state."""
    if not segments:
        return []
    segments = sorted(segments, key=lambda s: s["start_day"])
    out = [dict(segments[0])]
    for seg in segments[1:]:
        if seg["state"] == out[-1]["state"] and seg["start_day"] <= out[-1]["end_day"] + 1:
            out[-1]["end_day"] = max(out[-1]["end_day"], seg["end_day"])
        else:
            out.append(dict(seg))
    return out


def _encode_runs(daily: np.ndarray) -> list[dict]:
    """Run-length encode a per-day state array into contiguous tiled segments."""
    runs: list[dict] = []
    start = 0
    for i in range(1, len(daily) + 1):
        if i == len(daily) or daily[i] != daily[start]:
            runs.append({"start_day": int(start), "end_day": int(i - 1),
                         "state": C.STATES[int(daily[start])]})
            start = i
    return runs


def build_trajectory(archetype: str, rng: np.random.Generator, n_days: int) -> tuple[list[dict], list[dict]]:
    """Return (state_segments, context_events) for an archetype.

    Context events are things a health system would actually record - an
    insurance change, a pharmacy stockout, a new prescription - which give the
    barrier-inference layer something real to reason over.
    """
    segs: list[dict] = [{"start_day": 0, "end_day": n_days - 1, "state": C.STATE_ADHERENT}]
    events: list[dict] = []

    def overlay(start: int, length: int, state: str) -> None:
        start = int(np.clip(start, 0, n_days - 1))
        end = int(np.clip(start + length - 1, 0, n_days - 1))
        segs.append({"start_day": start, "end_day": end, "state": state})

    if archetype == "ADHERENT_STABLE":
        pass

    elif archetype == "OCCASIONAL_IRREGULARITY":
        for _ in range(int(rng.integers(2, 5))):
            overlay(rng.integers(60, n_days - 20), int(rng.integers(7, 16)), C.STATE_PARTIAL)

    elif archetype == "TEMPORARY_INTERRUPTION":
        start = int(rng.integers(200, n_days - 150))
        overlay(start, int(rng.integers(24, 44)), C.STATE_LAPSED)
        events.append({"day": start - int(rng.integers(0, 6)), "event_type": "life_disruption",
                       "detail": "Travel away from usual pharmacy recorded at reception"})

    elif archetype == "GRADUAL_DECLINE":
        onset = int(rng.integers(160, 300))
        overlay(onset, 90, C.STATE_PARTIAL)
        overlay(onset + 90, n_days, C.STATE_LAPSED)

    elif archetype == "REPEATED_REFILL_DELAY":
        onset = int(rng.integers(150, 280))
        overlay(onset, n_days, C.STATE_PARTIAL)

    elif archetype == "ACCESS_BARRIER":
        onset = int(rng.integers(150, 300))
        overlay(onset, 70, C.STATE_PARTIAL)
        overlay(onset + 70, n_days, C.STATE_LAPSED)
        for k in range(int(rng.integers(2, 4))):
            events.append({"day": onset + 20 + k * int(rng.integers(45, 80)),
                           "event_type": "pharmacy_supply_issue",
                           "detail": "Dispensing pharmacy recorded the item as out of stock"})

    elif archetype == "COST_BARRIER":
        onset = int(rng.integers(180, 320))
        events.append({"day": onset - 12, "event_type": "insurance_change",
                       "detail": "Plan change increased the patient's out-of-pocket share"})
        overlay(onset, 60, C.STATE_PARTIAL)
        overlay(onset + 60, n_days, C.STATE_LAPSED)
        events.append({"day": onset + 8, "event_type": "prescription_abandoned",
                       "detail": "Prescription printed but not collected from the pharmacy"})

    elif archetype == "SIDE_EFFECT":
        onset = int(rng.integers(170, 300))
        events.append({"day": onset, "event_type": "medication_added",
                       "detail": "Second agent added to the regimen"})
        overlay(onset + int(rng.integers(18, 34)), n_days, C.STATE_PARTIAL)

    elif archetype == "COMPLEX_REGIMEN":
        onset = int(rng.integers(120, 240))
        overlay(onset, n_days, C.STATE_PARTIAL)
        for k in range(int(rng.integers(2, 5))):
            events.append({"day": int(rng.integers(60, n_days - 30)),
                           "event_type": "dose_change",
                           "detail": "Dosing schedule revised"})

    elif archetype == "SILENT_NONADHERENCE":
        # The case the problem statement is named for. Primary adherence is
        # perfect - every prescription is collected on schedule - and secondary
        # adherence is poor: the doses are not taken. Proportion of days covered
        # is computed from dispensing records, so it reads near 1.0 throughout
        # and the conventional metric is structurally blind here. The only
        # evidence is that the clinical picture drifts while the pharmacy record
        # stays immaculate, and that inconsistency between signal families is
        # the whole detection.
        onset = int(rng.integers(150, 300))
        overlay(onset, 90, C.STATE_PARTIAL)
        overlay(onset + 90, n_days, C.STATE_LAPSED)

    elif archetype == "DISEASE_PROGRESSION":
        # The latent adherence state never leaves ADHERENT. The clinical picture
        # deteriorates regardless, because the condition itself is advancing.
        # This is the confounder that separates "the treatment is not working"
        # from "the treatment is not being taken", and a system that cannot
        # tell them apart sends the care team after the wrong problem.
        onset = int(rng.integers(140, 300))
        events.append({"day": onset, "event_type": "clinical_progression",
                       "detail": "Deterioration recorded with dispensing history unchanged"})

    elif archetype == "MISLEADING_ANOMALY":
        # Adherent the whole way through. The record still contains one very
        # visible irregularity, because real life produces those.
        blip = int(rng.integers(300, n_days - 120))
        events.append({"day": blip, "event_type": "life_disruption",
                       "detail": "Extended trip abroad noted in the record"})
        events.append({"day": blip, "event_type": "isolated_refill_gap",
                       "detail": "Single long gap between dispensing events"})
        events.append({"day": blip + 30, "event_type": "lab_blip",
                       "detail": "One off-trend laboratory result, repeated in range"})

    else:  # pragma: no cover - guarded by the caller
        raise ValueError(f"unknown archetype {archetype!r}")

    return _merge(segs), events


# ---------------------------------------------------------------------------
# Observation generators
# ---------------------------------------------------------------------------


def _taking_fraction(state_idx: int, rng: np.random.Generator) -> float:
    lo, hi = C.STATE_TAKING_FRACTION[C.STATES[state_idx]]
    return float(rng.uniform(lo, hi))


def _refill_delay(state: str, archetype: str, rng: np.random.Generator) -> float | None:
    """Days late for one dispensing event, or None if no fill occurs at all.

    Two mechanisms are folded together here, both of which produce late fills in
    a real pharmacy record: stretching a supply by skipping doses, and simply
    not going to collect the refill.
    """
    if archetype == "SILENT_NONADHERENCE":
        # Collection behaviour is decoupled from taking behaviour: this patient
        # keeps every pharmacy appointment regardless of latent state.
        return float(rng.exponential(1.4))
    if state == C.STATE_ADHERENT:
        return float(rng.exponential(1.6))
    if state == C.STATE_PARTIAL:
        base = float(rng.gamma(shape=3.0, scale=3.4))       # centred near 10 days
        if archetype in ("REPEATED_REFILL_DELAY", "ACCESS_BARRIER"):
            base *= 1.35
        return base
    # LAPSED: often no fill at all
    if rng.random() < 0.42:
        return None
    return float(rng.gamma(shape=4.0, scale=8.5))           # centred near 34 days


def _generate_refills(rec: PatientRecord, daily_state: np.ndarray, rng: np.random.Generator,
                      n_days: int) -> None:
    archetype = rec.patient["archetype"]
    for med in rec.medications:
        days_supply = med["days_supply"]
        day = int(med["start_day"])
        fill_index = 0
        stopped_at: int | None = None
        while day < min(n_days, med["end_day"]):
            state = C.STATES[daily_state[min(day, n_days - 1)]]
            delay = _refill_delay(state, archetype, rng)

            if rec.patient["archetype"] == "MISLEADING_ANOMALY":
                # Insert the single isolated gap at the scripted blip.
                blip_days = [e["day"] for e in rec.events if e["event_type"] == "isolated_refill_gap"]
                if blip_days and abs(day - blip_days[0]) <= days_supply // 2:
                    delay = float(rng.uniform(21, 27))

            if delay is None:
                stopped_at = day
                break

            expected = day + days_supply
            actual = expected + delay
            # Round before the bounds check, not after: a fill at day 539.6
            # rounds to 540 and would otherwise land outside the record.
            if int(round(actual)) >= n_days:
                break
            rec.refills.append({
                "patient_id": rec.patient["patient_id"],
                "medication_id": med["medication_id"],
                "fill_index": fill_index,
                "prescribed_day": int(day),
                "expected_day": int(expected),
                "actual_day": int(round(actual)),
                "days_supplied": days_supply,
                "quantity": days_supply * med["doses_per_day"],
                "delay_days": round(float(delay), 2),
            })
            fill_index += 1
            day = int(round(actual))

        if stopped_at is not None:
            rec.events.append({
                "patient_id": rec.patient["patient_id"],
                "day": int(stopped_at),
                "event_type": "dispensing_stopped",
                "detail": f"No further dispensing recorded for {med['name']}",
            })


def _exposure_curve(daily_state: np.ndarray, lag_days: int) -> np.ndarray:
    """Lagged cumulative exposure to non-adherence, in 'month equivalents'.

    A patient does not deteriorate the instant a dose is missed. This builds a
    smoothed, lagged burden signal that drives symptoms and laboratory drift,
    and which decays when the patient returns to adherence.
    """
    burden = np.where(daily_state == 2, 1.0, np.where(daily_state == 1, 0.45, -0.28))
    out = np.zeros_like(burden)
    acc = 0.0
    alpha = 1.0 / max(lag_days, 1)
    for i, b in enumerate(burden):
        acc += alpha * (b - 0.0)
        acc = max(0.0, acc * 0.995)   # decays toward control when adherent
        out[i] = acc
    return out / 3.0                  # scale into month-equivalent units


def _progression_curve(archetype: str, events: list[dict], rng: np.random.Generator,
                       n_days: int) -> np.ndarray:
    """Adherence-independent clinical deterioration, in the same units as exposure.

    Every chronic condition progresses to some degree while perfectly treated,
    and some patients progress fast. Without this term the simulated symptom and
    laboratory series are a near-deterministic readout of the hidden adherence
    state, a model trained on them looks superb, and the reported performance is
    an artefact of the simulator rather than a property of the method. It also
    creates the confounder the system has to survive: deterioration that is not
    a medication-taking problem at all.
    """
    curve = np.zeros(n_days, dtype=float)
    if archetype == "DISEASE_PROGRESSION":
        rate = float(rng.uniform(0.55, 1.05))
        onsets = [e["day"] for e in events if e["event_type"] == "clinical_progression"]
        onset = onsets[0] if onsets else int(rng.integers(140, 300))
    else:
        # A long tail: most patients are near-stable, a minority progress.
        rate = float(abs(rng.normal(0, 0.16)))
        onset = int(rng.integers(60, max(70, n_days - 60)))
    days = np.arange(n_days, dtype=float)
    ramp = np.clip((days - onset) / 180.0, 0.0, None)
    curve = rate * ramp

    # Transient flares unrelated to medication: infection, seasonal change, an
    # unrelated comorbidity. These are what make a single worsening result
    # uninformative on its own.
    for _ in range(int(rng.poisson(1.6))):
        centre = int(rng.integers(0, n_days))
        width = float(rng.uniform(12, 40))
        amp = float(rng.uniform(0.18, 0.7))
        curve += amp * np.exp(-0.5 * ((days - centre) / width) ** 2)
    return curve


def _generate_symptoms(rec: PatientRecord, exposure: np.ndarray, progression: np.ndarray,
                       rng: np.random.Generator, n_days: int) -> None:
    """Weekly patient-reported symptom burden on a 0-10 scale.

    Patient-reported outcomes are a weak, noisy and irregularly collected
    signal. The random intercept, the wide sensitivity range and the high
    residual noise are all deliberate: a patient who habitually reports 6/10 and
    a patient who habitually reports 1/10 must not be separable by their
    absolute score, only by movement against their own baseline.
    """
    if not rec.patient.get("reports_symptoms", True):
        return
    base = float(rng.uniform(0.8, 5.0))          # habitual reporting level
    sens = float(rng.uniform(0.5, 1.9))          # how much adherence affects them
    for day in range(7, n_days, 7):
        if rng.random() < 0.34:      # reporting is patchy
            continue
        val = (base
               + sens * exposure[day]
               + 1.5 * progression[day]
               + rng.normal(0, 1.15))
        if rec.patient["archetype"] == "SIDE_EFFECT":
            added = [e["day"] for e in rec.events if e["event_type"] == "medication_added"]
            if added and added[0] <= day <= added[0] + 80:
                val += 2.6 * np.exp(-(day - added[0]) / 55.0)
        rec.symptoms.append({
            "patient_id": rec.patient["patient_id"],
            "day": day,
            "score": round(float(np.clip(val, 0, 10)), 2),
        })


def _generate_labs(rec: PatientRecord, exposure: np.ndarray, progression: np.ndarray,
                   rng: np.random.Generator, n_days: int) -> None:
    lab = C.CONDITIONS[rec.patient["condition"]]["lab"]
    baseline = float(rng.uniform(*lab["baseline"]))
    # A patient who attends half as often as protocol yields half the results,
    # and a personal laboratory baseline built from two points is a weak thing.
    interval = lab["interval_days"] / max(rec.patient.get("lab_attendance", 1.0), 0.2)
    blip_days = [e["day"] for e in rec.events if e["event_type"] == "lab_blip"]
    day = int(rng.integers(10, interval))
    while day < n_days:
        # Adherence-driven drift and progression-driven drift are
        # indistinguishable in the value itself. Only their relationship to the
        # dispensing record tells them apart, which is the whole point.
        drift = lab["worsen_per_month"] * (exposure[day] + progression[day])
        val = baseline + drift + rng.normal(0, lab["noise"] * 1.35)
        if blip_days and abs(day - blip_days[0]) <= interval // 2:
            val += lab["direction"] * 2.9 * lab["noise"]   # a single off-trend result
        rec.labs.append({
            "patient_id": rec.patient["patient_id"],
            "day": int(day),
            "marker": lab["name"],
            "value": round(float(np.clip(val, lab["floor"], lab["ceiling"])), 2),
            "unit": lab["unit"],
            "direction": lab["direction"],
        })
        day += int(interval + rng.normal(0, interval * 0.12))


def _generate_appointments(rec: PatientRecord, daily_state: np.ndarray, rng: np.random.Generator,
                           n_days: int) -> None:
    engagement = rec.patient["baseline_engagement"]
    interval = int(rng.integers(55, 95))
    day = int(rng.integers(20, interval))
    while day < n_days:
        state = C.STATES[daily_state[min(day, n_days - 1)]]
        penalty = {C.STATE_ADHERENT: 0.0, C.STATE_PARTIAL: 0.13, C.STATE_LAPSED: 0.30}[state]
        attend_p = float(np.clip(engagement - penalty, 0.35, 0.99))
        attended = bool(rng.random() < attend_p)
        if rec.patient["archetype"] == "MISLEADING_ANOMALY":
            blip = [e["day"] for e in rec.events if e["event_type"] == "isolated_refill_gap"]
            attended = not (blip and abs(day - blip[0]) <= 25)
        rec.appointments.append({
            "patient_id": rec.patient["patient_id"],
            "scheduled_day": int(day),
            "attended": int(attended),
            "appointment_type": "Routine review" if rng.random() < 0.8 else "Medication review",
        })
        day += int(interval + rng.normal(0, 9))


def _generate_activity(rec: PatientRecord, exposure: np.ndarray, progression: np.ndarray,
                       rng: np.random.Generator, n_days: int) -> None:
    """Weekly wearable summaries. Deliberately a weak signal.

    Activity and sleep are only loosely coupled to medication behaviour. They
    are included because the brief lists them as available, and because a system
    that treats a weak signal as strong is the failure mode we are guarding
    against. The ablation study reports how little they add.
    """
    if rng.random() < 0.35:      # only a subset of patients wear a device
        return
    steps0 = float(rng.uniform(3800, 9200))
    sleep0 = float(rng.uniform(6.1, 7.9))
    hr0 = float(rng.uniform(58, 76))
    for day in range(7, n_days, 7):
        e = exposure[day] + progression[day]
        rec.activity.append({
            "patient_id": rec.patient["patient_id"],
            "day": day,
            "steps": int(max(300, rng.normal(steps0 * (1 - 0.10 * e), steps0 * 0.17))),
            "sleep_hours": round(float(np.clip(rng.normal(sleep0 - 0.20 * e, 0.62), 3.0, 10.5)), 2),
            "resting_hr": int(np.clip(rng.normal(hr0 + 2.1 * e, 3.4), 44, 108)),
        })


def _generate_prescription_events(rec: PatientRecord, rng: np.random.Generator, n_days: int) -> None:
    n = int(rng.poisson(1.1))
    for _ in range(n):
        rec.events.append({
            "patient_id": rec.patient["patient_id"],
            "day": int(rng.integers(40, n_days - 20)),
            "event_type": rng.choice(["dose_change", "medication_switch", "prescription_renewed"]),
            "detail": "Regimen amended at review",
        })
    # Hospitalisation risk rises with clinical consequence.
    if rng.random() < 0.10 + 0.12 * rec.patient["consequence"]:
        rec.events.append({
            "patient_id": rec.patient["patient_id"],
            "day": int(rng.integers(60, n_days - 10)),
            "event_type": "hospitalisation",
            "detail": "Short inpatient admission",
        })


# ---------------------------------------------------------------------------
# Cohort assembly
# ---------------------------------------------------------------------------


def generate_patient(patient_id: str, archetype: str, rng: np.random.Generator,
                     n_days: int = C.OBSERVATION_DAYS) -> PatientRecord:
    condition = str(rng.choice(list(C.CONDITIONS.keys())))
    spec = C.CONDITIONS[condition]

    n_meds = len(spec["medications"])
    if archetype == "COMPLEX_REGIMEN":
        n_meds = len(spec["medications"])
    else:
        n_meds = int(rng.integers(1, n_meds + 1))
    chosen = spec["medications"][:n_meds]

    doses_per_day = sum(m["doses_per_day"] for m in chosen)
    if archetype == "COMPLEX_REGIMEN":
        doses_per_day += int(rng.integers(2, 5))     # additional non-tracked medicines
    consequence = max(m["consequence"] for m in chosen)

    patient = {
        "patient_id": patient_id,
        "age": int(np.clip(rng.normal(58, 14), 21, 92)),
        "sex": str(rng.choice(["F", "M"])),
        "condition": condition,
        "symptom_scale": spec["symptom_scale"],
        "lab_marker": spec["lab"]["name"],
        "lab_unit": spec["lab"]["unit"],
        "lab_direction": spec["lab"]["direction"],
        "archetype": archetype,
        "n_medications": n_meds,
        "total_daily_doses": doses_per_day,
        "consequence": round(float(consequence), 3),
        "insurance_tier": str(rng.choice(["Public", "Employer", "Self-pay"], p=[0.45, 0.4, 0.15])),
        "distance_to_pharmacy_km": round(float(abs(rng.normal(3.4, 3.0))) + 0.3, 2),
        "baseline_engagement": round(float(np.clip(rng.normal(0.87, 0.09), 0.5, 0.99)), 3),
        "index_date": day_to_date(0).isoformat(),
        "observation_days": n_days,
        # Record richness varies enormously across a real panel, and a system
        # that assumes a complete eighteen-month history for everyone will
        # quietly produce its most confident output on its thinnest evidence.
        # Roughly a quarter of patients here enrolled part-way through, a fifth
        # never complete symptom questionnaires, and some attend for laboratory
        # monitoring far less often than the protocol suggests.
        "enrolled_day": int(rng.integers(150, 400)) if rng.random() < 0.24 else 0,
        "reports_symptoms": bool(rng.random() > 0.20),
        "lab_attendance": round(float(np.clip(rng.normal(1.0, 0.45), 0.35, 1.6)), 2),
    }

    rec = PatientRecord(patient=patient)

    for i, med in enumerate(chosen):
        rec.medications.append({
            "patient_id": patient_id,
            "medication_id": f"{patient_id}-M{i + 1}",
            "name": med["name"],
            "doses_per_day": med["doses_per_day"],
            "days_supply": med["days_supply"],
            "start_day": int(rng.integers(0, 14)),
            "end_day": n_days,
            "consequence": med["consequence"],
        })

    segments, ctx_events = build_trajectory(archetype, rng, n_days)
    daily_state = _segments_to_daily(segments, n_days)
    # build_trajectory layers overlays on top of a base segment, so its output
    # overlaps. Flatten to the resolved per-day states and re-encode as tiled,
    # non-overlapping runs before storing: a ground-truth file that has to be
    # read in layer order is a file that will eventually be read wrongly.
    rec.state_segments = [{"patient_id": patient_id, **s}
                          for s in _encode_runs(daily_state)]
    rec.events = [{"patient_id": patient_id, **e} for e in ctx_events]
    exposure = _exposure_curve(daily_state, C.CONDITIONS[condition]["symptom_lag_days"])
    progression = _progression_curve(archetype, rec.events, rng, n_days)

    _generate_refills(rec, daily_state, rng, n_days)
    _generate_symptoms(rec, exposure, progression, rng, n_days)
    _generate_labs(rec, exposure, progression, rng, n_days)
    _generate_appointments(rec, daily_state, rng, n_days)
    _generate_activity(rec, exposure, progression, rng, n_days)
    _generate_prescription_events(rec, rng, n_days)

    _apply_enrolment_cut(rec)

    # Side-effect archetype gets a second medication actually recorded.
    if archetype == "SIDE_EFFECT" and n_meds < len(spec["medications"]):
        added = [e["day"] for e in rec.events if e["event_type"] == "medication_added"]
        if added:
            extra = spec["medications"][n_meds]
            rec.medications.append({
                "patient_id": patient_id,
                "medication_id": f"{patient_id}-M{n_meds + 1}",
                "name": extra["name"],
                "doses_per_day": extra["doses_per_day"],
                "days_supply": extra["days_supply"],
                "start_day": int(added[0]),
                "end_day": n_days,
                "consequence": extra["consequence"],
            })
            _generate_refills(
                PatientRecord(patient=patient, medications=[rec.medications[-1]],
                              refills=rec.refills, events=rec.events),
                daily_state, rng, n_days,
            )

    return rec


def _apply_enrolment_cut(rec: PatientRecord) -> None:
    """Drop observations from before the patient joined the panel.

    Late enrolment is not a data-quality problem to be cleaned away; it is the
    normal condition of a real panel and the main source of genuine uncertainty
    in this system. A patient with four months of history cannot have a
    trustworthy personal baseline, and the confidence layer needs to see that.
    """
    cut = int(rec.patient.get("enrolled_day", 0) or 0)
    if cut <= 0:
        return
    rec.refills = [r for r in rec.refills if r["actual_day"] >= cut]
    rec.appointments = [a for a in rec.appointments if a["scheduled_day"] >= cut]
    rec.symptoms = [s_ for s_ in rec.symptoms if s_["day"] >= cut]
    rec.labs = [l for l in rec.labs if l["day"] >= cut]
    rec.activity = [a for a in rec.activity if a["day"] >= cut]
    rec.events = [e for e in rec.events if e.get("day", 0) >= cut]


def generate_cohort(n_patients: int = 600, seed: int = C.RANDOM_SEED,
                    n_days: int = C.OBSERVATION_DAYS) -> dict[str, pd.DataFrame]:
    """Generate a full synthetic cohort and return it as a dict of DataFrames."""
    rng = np.random.default_rng(seed)

    names = list(ARCHETYPES.keys())
    weights = np.array([ARCHETYPES[a]["weight"] for a in names], dtype=float)
    weights = weights / weights.sum()
    counts = np.floor(weights * n_patients).astype(int)
    while counts.sum() < n_patients:            # give remainder to the majority class
        counts[int(np.argmax(weights))] += 1

    assignments: list[str] = []
    for name, k in zip(names, counts):
        assignments.extend([name] * int(k))
    # Guarantee at least four of every archetype so the demo and the subgroup
    # metrics always have something to show.
    for name in names:
        if assignments.count(name) < 4:
            for _ in range(4 - assignments.count(name)):
                assignments.append(name)
    rng.shuffle(assignments)

    records: list[PatientRecord] = []
    for i, archetype in enumerate(assignments):
        pid = f"P{1000 + i}"
        # Per-patient generator keeps a patient's record stable even if the
        # cohort size changes, which makes demo cases reproducible.
        prng = np.random.default_rng(abs(hash((seed, pid))) % (2**32))
        records.append(generate_patient(pid, archetype, prng, n_days))

    def frame(key: str) -> pd.DataFrame:
        rows: list[dict] = []
        for r in records:
            rows.extend(getattr(r, key))
        return pd.DataFrame(rows)

    tables = {
        "patients": pd.DataFrame([r.patient for r in records]),
        "medications": frame("medications"),
        "refills": frame("refills"),
        "appointments": frame("appointments"),
        "symptoms": frame("symptoms"),
        "labs": frame("labs"),
        "activity": frame("activity"),
        "events": frame("events"),
        "ground_truth_states": frame("state_segments"),
    }
    for name, df in tables.items():
        if not df.empty and "day" in df.columns:
            df["date"] = [day_to_date(d).isoformat() for d in df["day"]]
    if not tables["refills"].empty:
        tables["refills"]["date"] = [day_to_date(d).isoformat()
                                     for d in tables["refills"]["actual_day"]]
    if not tables["appointments"].empty:
        tables["appointments"]["date"] = [day_to_date(d).isoformat()
                                          for d in tables["appointments"]["scheduled_day"]]
    return tables
