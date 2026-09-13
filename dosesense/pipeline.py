"""
Assessment assembly.

This module turns a scored snapshot into the object a clinician reads. The
ordering of the keys in that object is not incidental: observed facts come
before the model's estimate, and the estimate comes before any hypothesis about
cause. The interface renders them in that order, so a reader meets the evidence
before meeting the inference.

Every assessment carries an explicit three-way separation, which the brief
requires and which the API schema enforces:

  observed          things recorded in the health system, with their values
  inferred          the model's estimate, its uncertainty and its attribution
  hypothesis        a possible barrier, marked as a hypothesis, or an abstention

The trajectory of probabilities across past snapshots is included too. A single
number cannot show a clinician whether a concern is emerging, sustained or
resolving, and that distinction is the difference between a phone call today and
a note for the next review.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import barriers as BAR
from . import risk as RISK
from .datagen import day_to_date
from .features import build_series, extract_snapshot, FAMILY_OF, FAMILY_LABELS
from .panel import snapshot_days


def _timeline(ps, snap: dict, assessments_history: list[dict]) -> list[dict]:
    """Chronological evidence timeline for the interface.

    Dispensing events, attendance, notable symptom moves and recorded care
    events, merged and dated. Only entries with something to say are kept: a
    timeline of eighteen uneventful refills buries the two that matter.
    """
    t = snap["day"]
    entries: list[dict] = []

    m = ps.fill_day <= t
    for day, delay in zip(ps.fill_day[m], ps.fill_delay[m]):
        if delay <= 3:
            kind, text = "routine", "Refill collected on schedule"
        elif delay <= 10:
            kind, text = "notable", f"Refill collected {delay:.0f} days late"
        else:
            kind, text = "adverse", f"Refill collected {delay:.0f} days late"
        entries.append({"day": int(day), "date": day_to_date(int(day)).isoformat(),
                        "channel": "Pharmacy", "kind": kind, "text": text, "observed": True})

    am = ps.appt_day <= t
    for day, attended in zip(ps.appt_day[am], ps.appt_attended[am]):
        entries.append({
            "day": int(day), "date": day_to_date(int(day)).isoformat(), "channel": "Clinic",
            "kind": "routine" if attended else "adverse",
            "text": "Follow-up attended" if attended else "Follow-up not attended",
            "observed": True})

    if ps.ev_day.size:
        em = ps.ev_day <= t
        for day, ev, detail in zip(ps.ev_day[em], ps.ev_type[em], ps.ev_detail[em]):
            entries.append({"day": int(day), "date": day_to_date(int(day)).isoformat(),
                            "channel": "Care record", "kind": "context",
                            "text": f"{str(ev).replace('_', ' ').capitalize()}: {detail}",
                            "observed": True})

    # Change points are model output, not observations, and are marked as such.
    for series_name, cr in (snap["changes"] or {}).items():
        if cr and cr.get("status") in (C.CHANGE_PERSISTENT, C.CHANGE_TEMPORARY) and cr.get("change_position"):
            day = int(cr["change_position"])
            label = {"refill_delay": "refill timing", "symptom": "symptom burden",
                     "laboratory": "laboratory marker"}.get(series_name, series_name)
            verb = ("shifted and has stayed shifted" if cr["status"] == C.CHANGE_PERSISTENT
                    else "shifted and later returned to baseline")
            entries.append({
                "day": day, "date": day_to_date(day).isoformat(), "channel": "DoseSense",
                "kind": "inference", "observed": False,
                "text": f"Detected change point: {label} {verb}"})

    # Coverage gaps. A timeline that shows only what happened hides the thing
    # that most often drives uncertainty: the months where nothing was recorded.
    # A clinician looking at a low-confidence estimate needs to see immediately
    # whether the record is quiet because the patient is stable or because
    # nobody measured anything, and those look identical until the gaps are
    # drawn.
    for channel, days, expected, label in (
        ("Laboratory", ps.lab_day, 120, "laboratory results"),
        ("Clinic", ps.appt_day, 150, "scheduled follow-ups"),
        ("Symptoms", ps.sym_day, 60, "symptom questionnaires"),
        ("Pharmacy", ps.fill_day, 120, "dispensing events"),
    ):
        d = np.asarray(days)[np.asarray(days) <= t]
        if not d.size:
            entries.append({
                "day": int(max(0, t - 30)), "date": day_to_date(int(max(0, t - 30))).isoformat(),
                "channel": channel, "kind": "gap", "observed": True,
                "text": f"No {label} on record at all",
                "gap_days": None})
            continue
        bounds = np.concatenate(([d[0]], d, [t]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            span = int(b - a)
            if span >= expected:
                entries.append({
                    "day": int(a), "date": day_to_date(int(a)).isoformat(),
                    "channel": channel, "kind": "gap", "observed": True,
                    "text": (f"{span} days with no {label}"
                             f"{' — still open' if int(b) >= t else ''}"),
                    "gap_days": span})

    entries.sort(key=lambda e: e["day"])
    return entries


def leave_one_family_out(model, x: np.ndarray, families: dict[str, bool],
                         full_probability: float) -> list[dict]:
    """What would the estimate have been without each signal family?

    For every family that carried evidence, the model is re-run with that
    family's features replaced by their training medians, and the change in the
    estimate reported. This answers the question a clinician actually asks -
    "is this resting on one thing, or do several agree?" - which SHAP does not,
    because SHAP attributes a single prediction rather than simulating its
    absence.

    Two honest limits on how this should be read.

    It is *not* a causal counterfactual. Replacing a family with typical values
    is not the same as that family never having been recorded, and the features
    are correlated, so removing symptoms partially removes information the
    laboratory features also carry. The number says how much the model leans on
    a family, not what would have happened in a world without it.

    And it is not additive. The deltas will not sum to the prediction, because
    families overlap. A patient where every family individually changes the
    estimate very little is a patient with genuinely redundant evidence, which
    is the strongest kind.
    """
    if getattr(model, "feature_medians", None) is None:
        return []
    cols = model.feature_columns
    med = model.feature_medians
    out = []
    for fam, present in families.items():
        if not present:
            continue
        idx = [i for i, c in enumerate(cols) if FAMILY_OF.get(c) == fam]
        if not idx:
            continue
        alt = x.copy()
        alt[idx] = med[idx]
        p_without = float(model.predict(alt.reshape(1, -1))["probability"][0])
        out.append({
            "family": fam,
            "label": FAMILY_LABELS.get(fam, fam),
            "probability_without": round(p_without, 4),
            "delta": round(full_probability - p_without, 4),
            "n_features": len(idx),
        })
    return sorted(out, key=lambda r: -abs(r["delta"]))


def assess_snapshot(ps, snap: dict, model, explainer, patient: dict,
                    events: list[dict]) -> dict:
    """Produce a complete assessment for one patient-snapshot."""
    feats = snap["features"]
    x = np.array([feats[c] for c in model.feature_columns], dtype=float)
    pred = model.predict(x.reshape(1, -1))
    prob = float(pred["probability"][0])
    sd = float(pred["ensemble_sd"][0])

    strength = RISK.evidence_strength(snap["families"], snap["facts"])
    conf = RISK.assess_confidence(prob, sd, strength, feats, snap["changes"])
    refill_cp = (snap["changes"] or {}).get("refill_delay") or {}
    change_status = refill_cp.get("status")

    decision = RISK.should_alert(prob, conf["confidence"], change_status)
    prio = RISK.priority(prob, float(patient.get("consequence", 0.5)), conf["confidence"])
    barrier = BAR.infer_barriers(feats, patient, events, snap["day"], snap["changes"])

    attribution = explainer.explain_row(x) if explainer is not None else None
    counterfactual = leave_one_family_out(model, x, snap["families"], prob)

    # How much the conclusion depends on any single family. A concern that
    # survives the removal of every family individually is corroborated; one
    # that collapses when a single family is neutralised is a one-legged
    # finding, and the interface should say so.
    worst = max((abs(c["delta"]) for c in counterfactual), default=0.0)
    robustness = ("corroborated" if counterfactual and worst < 0.15
                  else "rests on one signal" if worst >= 0.35
                  else "partly corroborated")

    return {
        "patient_id": patient["patient_id"],
        "day": snap["day"],
        "date": day_to_date(snap["day"]).isoformat(),

        # 1. What was recorded.
        "observed": {
            "facts": snap["facts"],
            "signal_families_available": strength["families_available"],
            "signal_families_with_evidence": strength["families_with_evidence"],
            "pdc_90": round(feats["refill_pdc_90"], 3),
            "mpr_90": round(feats["refill_mpr_90"], 3),
            "days_since_last_dispensing": int(feats["refill_days_since_last"]),
            "last_refill_delay_days": round(feats["refill_last_delay"], 1),
            "missed_followups_180d_rate": round(feats["appt_missed_rate_180"], 3),
        },

        # 2. What the model estimated, and how much to trust it.
        "inferred": {
            "adherence_concern_probability": round(prob, 4),
            "confidence": conf["confidence"],
            "abstained": conf["abstained"],
            "uncertainty": {
                "ensemble_disagreement": round(sd, 4),
                "record_completeness": conf["completeness"],
                "independent_evidence_families": strength["n_families_with_evidence"],
                "total_evidence_weight": strength["total_evidence_weight"],
                "notes": conf["reasons"],
            },
            "change_detection": snap["changes"],
            "pattern": change_status or C.CHANGE_STABLE,
            "attribution": attribution,
            "counterfactual": counterfactual,
            "evidence_robustness": robustness,
        },

        # 3. A possible reason, clearly marked as a hypothesis.
        "hypothesis": barrier,

        # 4. What the system asks a human to do.
        "action": {
            "alert": decision["alert"],
            "state": decision["state"],
            "message": decision["message"],
            "priority_tier": prio["priority_tier"],
            "priority_score": prio["priority_score"],
            "priority_components": prio["components"],
        },
    }


class DoseSenseEngine:
    """Scores a cohort and holds the results for serving.

    Built once at start-up. Scoring 600 patients across their full snapshot
    history takes a few seconds, so the engine precomputes the trajectory and
    the current assessment for each patient and keeps them in memory rather than
    recomputing on every request.
    """

    def __init__(self, tables: dict[str, pd.DataFrame], model, explainer=None,
                 history_snapshots: int = 8):
        self.tables = tables
        self.model = model
        self.explainer = explainer
        self.series = build_series(tables)
        self.patients = {r["patient_id"]: r for r in tables["patients"].to_dict("records")}
        self.events_by_patient: dict[str, list[dict]] = {}
        if len(tables["events"]):
            for pid, grp in tables["events"].groupby("patient_id"):
                self.events_by_patient[pid] = grp.to_dict("records")

        self.current: dict[str, dict] = {}
        self.trajectory: dict[str, list[dict]] = {}
        self.timeline: dict[str, list[dict]] = {}
        self._score_all(history_snapshots)

    def _score_all(self, history_snapshots: int) -> None:
        for pid, ps in self.series.items():
            n_days = int(ps.p["observation_days"])
            days = snapshot_days(n_days)
            events = self.events_by_patient.get(pid, [])
            traj: list[dict] = []
            last_snap = None
            last_assess = None

            for t in days[-history_snapshots:]:
                snap = extract_snapshot(ps, t)
                a = assess_snapshot(ps, snap, self.model, self.explainer, ps.p, events)
                traj.append({
                    "day": t,
                    "date": day_to_date(t).isoformat(),
                    "probability": a["inferred"]["adherence_concern_probability"],
                    "confidence": a["inferred"]["confidence"],
                    "pattern": a["inferred"]["pattern"],
                    "state": a["action"]["state"],
                    "priority_tier": a["action"]["priority_tier"],
                    "pdc_90": a["observed"]["pdc_90"],
                })
                last_snap, last_assess = snap, a

            self.trajectory[pid] = traj
            self.current[pid] = last_assess
            self.timeline[pid] = _timeline(ps, last_snap, traj) if last_snap else []

    # -- queries -----------------------------------------------------------

    def queue(self) -> list[dict]:
        """The priority worklist: every patient, ordered as a clinician should work it."""
        rows = []
        for pid, a in self.current.items():
            p = self.patients[pid]
            traj = self.trajectory[pid]
            probs = [s["probability"] for s in traj]
            rows.append({
                "patient_id": pid,
                "age": p["age"],
                "sex": p["sex"],
                "condition": p["condition"],
                "medications": self._med_names(pid),
                "probability": a["inferred"]["adherence_concern_probability"],
                "confidence": a["inferred"]["confidence"],
                "abstained": a["inferred"]["abstained"],
                "pattern": a["inferred"]["pattern"],
                "state": a["action"]["state"],
                "alert": a["action"]["alert"],
                "priority_tier": a["action"]["priority_tier"],
                "priority_score": a["action"]["priority_score"],
                "clinical_consequence": p["consequence"],
                "barrier": a["hypothesis"]["primary"],
                "barrier_label": a["hypothesis"]["primary_label"],
                "barrier_confidence": a["hypothesis"]["confidence"],
                "evidence_families": a["inferred"]["uncertainty"]["independent_evidence_families"],
                "pdc_90": a["observed"]["pdc_90"],
                "spark": probs,
                "top_fact": a["observed"]["facts"][0]["text"] if a["observed"]["facts"] else
                            "No notable departure from this patient's baseline",
            })
        rows.sort(key=lambda r: -r["priority_score"])
        return rows

    def _med_names(self, pid: str) -> list[str]:
        meds = self.tables["medications"]
        sel = meds[meds["patient_id"] == pid]
        return sel["name"].tolist()

    def patient_detail(self, pid: str) -> dict | None:
        if pid not in self.current:
            return None
        p = dict(self.patients[pid])
        p.pop("archetype", None)      # the hidden generator label is never served
        return {
            "patient": p,
            "coverage": self._coverage(pid),
            "medications": self.tables["medications"][
                self.tables["medications"]["patient_id"] == pid].to_dict("records"),
            "assessment": self.current[pid],
            "trajectory": self.trajectory[pid],
            "timeline": self.timeline[pid],
            "series": self._patient_series(pid),
        }

    def _coverage(self, pid: str) -> list[dict]:
        """Per-channel record coverage, so the interface can show what is missing.

        ``status`` is deliberately three-valued rather than a percentage: a
        clinician needs to know whether a channel is usable, thin, or absent,
        and a number invites false precision about data that is simply not
        there.
        """
        ps = self.series[pid]
        t = int(ps.p["observation_days"])
        out = []
        for name, days, expected_gap, unit in (
            ("Pharmacy dispensing", ps.fill_day, 120, "collections"),
            ("Clinic attendance", ps.appt_day, 150, "appointments"),
            ("Symptom reports", ps.sym_day, 60, "questionnaires"),
            ("Laboratory results", ps.lab_day, 120, "results"),
            ("Activity and sleep", ps.act_day, 60, "weeks of data"),
        ):
            d = np.asarray(days)[np.asarray(days) <= t]
            n = int(d.size)
            if n == 0:
                status, note = "absent", "Nothing on record"
            else:
                since = int(t - d[-1])
                gaps = np.diff(np.concatenate(([d[0]], d, [t]))) if n else np.array([0])
                worst = int(gaps.max()) if gaps.size else 0
                if since >= expected_gap or worst >= expected_gap * 1.5:
                    status = "sparse"
                    note = f"{n} {unit}, last {since} days ago"
                else:
                    status = "covered"
                    note = f"{n} {unit}, last {since} days ago"
            out.append({"channel": name, "status": status, "n": n, "note": note})
        return out

    def _patient_series(self, pid: str) -> dict:
        ps = self.series[pid]
        t = int(ps.p["observation_days"])
        m = ps.fill_day <= t
        return {
            "refills": [{"day": int(d), "date": day_to_date(int(d)).isoformat(),
                         "delay_days": round(float(x), 1)}
                        for d, x in zip(ps.fill_day[m], ps.fill_delay[m])],
            "symptoms": [{"day": int(d), "date": day_to_date(int(d)).isoformat(),
                          "score": round(float(v), 2)}
                         for d, v in zip(ps.sym_day, ps.sym_val) if d <= t],
            "labs": [{"day": int(d), "date": day_to_date(int(d)).isoformat(),
                      "value": round(float(v), 2)}
                     for d, v in zip(ps.lab_day, ps.lab_val) if d <= t],
            "lab_marker": ps.p.get("lab_marker"),
            "lab_unit": ps.p.get("lab_unit"),
            "symptom_scale": ps.p.get("symptom_scale"),
        }

    def overview(self) -> dict:
        q = self.queue()
        alerts = [r for r in q if r["alert"]]
        return {
            "patients_monitored": len(q),
            "requiring_review": len(alerts),
            "critical": sum(1 for r in alerts if r["priority_tier"] == "CRITICAL"),
            "high": sum(1 for r in alerts if r["priority_tier"] == "HIGH"),
            "persistent_shifts": sum(1 for r in q if r["pattern"] == C.CHANGE_PERSISTENT),
            "temporary_irregularities": sum(1 for r in q if r["pattern"] == C.CHANGE_TEMPORARY),
            "insufficient_evidence": sum(1 for r in q if r["abstained"]),
            "mean_probability": round(float(np.mean([r["probability"] for r in q])), 4) if q else 0.0,
            "barrier_mix": _count(r["barrier"] for r in alerts),
            "state_mix": _count(r["state"] for r in q),
        }


def _count(values) -> dict:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
