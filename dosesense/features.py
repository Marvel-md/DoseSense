"""
Feature engineering.

Two rules govern this module.

*Causality.* A snapshot taken on day t may only use records dated on or before
day t. Every window is a trailing window. This is enforced structurally by
slicing each patient's series once per snapshot rather than by filtering after
the fact, because leakage in an adherence model is easy to introduce and
impossible to see in the metrics.

*Personal baselines.* Nearly every feature is expressed relative to the
patient's own history rather than to a population average. "Refill was 19 days
late" is weak evidence on its own; "refill was 19 days late for a patient whose
last eleven fills were within two days of schedule" is strong evidence.

Deliberate exclusions from the model matrix, and why:

  sex, insurance tier      protected or near-protected attributes. Held back for
                           subgroup evaluation, never used to predict.
  distance to pharmacy     a plausible access signal but also a socioeconomic
                           proxy. Used as supporting evidence in barrier
                           inference, where a human sees it, and kept out of the
                           model so it cannot silently drive the score.
  clinical consequence     severity is applied as declared policy in the
                           priority score, not learned as a correlate of
                           behaviour.

The extractor returns both a numeric feature vector and a structured record of
observed facts. The second is what the interface shows the clinician, and it
contains only things that were actually measured.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import baseline as B
from .changepoint import detect_change

# ---------------------------------------------------------------------------
# Feature registry
# ---------------------------------------------------------------------------
# Grouping features by signal family is not cosmetic: it drives the ablation
# study, the evidence-strength count used for abstention, and the grouped
# attribution shown in the interface.

FEATURE_FAMILIES: dict[str, list[str]] = {
    "refill": [
        "refill_n_fills_365",
        "refill_interval_mean",
        "refill_interval_cv",
        "refill_last_delay",
        "refill_delay_recent_mean",
        "refill_delay_baseline_mean",
        "refill_delay_excess",
        "refill_delay_slope",
        "refill_consecutive_late",
        "refill_days_since_last",
        "refill_overdue_ratio",
        "refill_pdc_90",
        "refill_pdc_adaptive",
        "refill_pdc_180",
        "refill_pdc_delta",
        "refill_mpr_90",
        "refill_cp_effect_sd",
        "refill_cp_persistent",
        "refill_cp_temporary",
        # Interval-relative measures. A 10-day delay is trivial on a 90-day
        # maintenance supply and severe on a 15-day acute course, so the same
        # quantity is expressed as a fraction of the expected interval as well
        # as in days. The raw day-counts are kept because they are what a
        # clinician reads; the ratios are what generalise across regimens.
        "refill_supply_days",
        "refill_last_delay_ratio",
        "refill_delay_excess_ratio",
        "refill_recent_delay_ratio",
    ],
    "prescription": [
        "rx_n_medications",
        "rx_total_daily_doses",
        "rx_changes_180",
        "rx_days_since_change",
        "rx_stopped_count",
    ],
    "appointment": [
        "appt_n_365",
        "appt_missed_rate_365",
        "appt_missed_rate_180",
        "appt_missed_rate_excess",
        "appt_consecutive_missed",
        "appt_days_since_attended",
    ],
    "symptom": [
        "sym_last",
        "sym_mean_90",
        "sym_baseline_mean",
        "sym_delta",
        "sym_slope_90",
        "sym_variability",
        "sym_cp_effect_sd",
        "sym_cp_persistent",
    ],
    "laboratory": [
        "lab_last_z",
        "lab_slope_per_90d",
        "lab_cp_effect_sd",
        "lab_cp_persistent",
        "lab_n_results",
    ],
    "behavioural": [
        "beh_has_wearable",
        "beh_steps_z",
        "beh_sleep_z",
        "beh_rhr_z",
    ],
    "context": [
        "ctx_age",
    ],
}

# ---------------------------------------------------------------------------
# Evaluated and not shipped
# ---------------------------------------------------------------------------
# Cross-signal velocity. The idea: the families above each describe one channel,
# while these describe how two channels move *relative to each other*, which is
# where the two hardest patterns are supposed to live. Silent non-adherence is
# clinical markers deteriorating while collection timing stays immaculate -
# neither channel alarming alone, the divergence between them being the whole
# signal. Disease progression produces the same divergence, so the terms were
# kept separate rather than collapsed, to let the model weigh them.
#
# It did not work. Measured across three seeds against the shipped feature set:
#
#   PR-AUC              0.8781 -> 0.8788   (+0.0007, noise)
#   silent recall        0.311 -> 0.322    (+0.011, marginal)
#   progression false alerts  0.090 -> 0.115  (+0.025, worse)
#   median lead time       40d -> 40d      (unchanged)
#
# It slightly improves the case it was designed for and makes the confounder
# meaningfully worse, which is the wrong trade. The features are kept here, out
# of the shipped model, because a measured negative result is worth more than a
# deleted branch: the ablation still reports them, and anyone repeating the idea
# can see it was tried and what happened.

EXPERIMENTAL_FAMILIES: dict[str, list[str]] = {
    "cross_signal": [
        "xs_clinical_velocity",
        "xs_refill_velocity",
        "xs_divergence",
        "xs_concordance",
        "xs_symptom_lab_agreement",
    ],
}

ALL_FAMILIES = {**FEATURE_FAMILIES, **EXPERIMENTAL_FAMILIES}

# What the shipped model sees.
FEATURE_COLUMNS: list[str] = [f for fam in FEATURE_FAMILIES.values() for f in fam]

# Everything computed, including experiments, so the ablation can measure them.
ALL_FEATURE_COLUMNS: list[str] = [f for fam in ALL_FAMILIES.values() for f in fam]

FAMILY_OF = {f: fam for fam, feats in ALL_FAMILIES.items() for f in feats}

FAMILY_LABELS = {
    "refill": "Pharmacy dispensing",
    "prescription": "Prescription changes",
    "appointment": "Follow-up attendance",
    "symptom": "Reported symptoms",
    "laboratory": "Laboratory results",
    "behavioural": "Activity and sleep",
    "context": "Demographics",
    "cross_signal": "Cross-signal velocity (experimental, not in the shipped model)",
}


# ---------------------------------------------------------------------------
# Per-patient series bundle
# ---------------------------------------------------------------------------


class PatientSeries:
    """Pre-sliced numpy views of one patient's record, built once per patient."""

    def __init__(self, patient: dict, medications: pd.DataFrame, refills: pd.DataFrame,
                 appointments: pd.DataFrame, symptoms: pd.DataFrame, labs: pd.DataFrame,
                 activity: pd.DataFrame, events: pd.DataFrame):
        self.p = patient
        self.meds = medications.sort_values("start_day").to_dict("records")

        r = refills.sort_values("actual_day")
        self.fill_day = r["actual_day"].to_numpy(dtype=float)
        self.fill_delay = r["delay_days"].to_numpy(dtype=float)
        self.fill_supply = r["days_supplied"].to_numpy(dtype=float)
        self.fill_med = r["medication_id"].to_numpy() if len(r) else np.array([])

        a = appointments.sort_values("scheduled_day")
        self.appt_day = a["scheduled_day"].to_numpy(dtype=float)
        self.appt_attended = a["attended"].to_numpy(dtype=float)

        s = symptoms.sort_values("day")
        self.sym_day = s["day"].to_numpy(dtype=float)
        self.sym_val = s["score"].to_numpy(dtype=float)

        l = labs.sort_values("day")
        self.lab_day = l["day"].to_numpy(dtype=float)
        self.lab_val = l["value"].to_numpy(dtype=float)
        self.lab_direction = int(patient.get("lab_direction", 1))

        w = activity.sort_values("day")
        self.act_day = w["day"].to_numpy(dtype=float)
        self.act_steps = w["steps"].to_numpy(dtype=float) if len(w) else np.array([])
        self.act_sleep = w["sleep_hours"].to_numpy(dtype=float) if len(w) else np.array([])
        self.act_rhr = w["resting_hr"].to_numpy(dtype=float) if len(w) else np.array([])

        e = events.sort_values("day") if len(events) else events
        self.ev_day = e["day"].to_numpy(dtype=float) if len(e) else np.array([])
        self.ev_type = e["event_type"].to_numpy() if len(e) else np.array([])
        self.ev_detail = e["detail"].to_numpy() if len(e) else np.array([])


def _slope(x: np.ndarray, y: np.ndarray, per: float = 1.0) -> float:
    """Least-squares slope, scaled to units per ``per`` x-units."""
    if x.size < 3 or np.ptp(x) == 0:
        return 0.0
    return float(np.polyfit(x, y, 1)[0] * per)


def _safe(v: float) -> float:
    return float(v) if np.isfinite(v) else 0.0


Z_CLIP = 8.0

# Smallest shift in each series that is worth a clinician's attention, in the
# series' own units. These are judgement calls, stated openly so they can be
# argued with, rather than thresholds buried inside a model.
CLINICAL_FLOOR = {
    "refill_delay_days": 3.5,            # a supply collected three days late is normal life
    "refill_delay_fraction_of_supply": 0.12,  # ...or 12% of the interval, whichever is larger
    "symptom_points": 0.8,               # on a 0-10 self-reported scale
}


def _sd_floor(sd: float, mean: float) -> float:
    """A dispersion floor that scales with the measurement.

    A patient whose first three HbA1c results land within 0.05% of each other
    has a near-zero sample SD, and dividing by it turns ordinary assay noise
    into a twelve-sigma event. Flooring the denominator at 5% of the level -
    roughly the analytical variability of the markers involved - keeps the
    z-score interpretable instead of letting an accident of sampling dominate
    the model.
    """
    return max(float(sd), 0.05 * abs(float(mean)), 1e-3)


def _z(value: float, mean: float, sd: float) -> float:
    return float(np.clip((value - mean) / _sd_floor(sd, mean), -Z_CLIP, Z_CLIP))


# ---------------------------------------------------------------------------
# Snapshot extraction
# ---------------------------------------------------------------------------


def extract_snapshot(ps: PatientSeries, t: int) -> dict:
    """Compute features and observed facts for one patient at day ``t``.

    Returns a dict with keys ``features`` (numeric, model-ready), ``facts``
    (human-readable observations with values), ``families`` (which signal
    families carried usable information) and ``changes`` (change-point results).
    """
    f: dict[str, float] = {k: 0.0 for k in ALL_FEATURE_COLUMNS}
    facts: list[dict] = []
    families: dict[str, bool] = {k: False for k in C.SIGNAL_FAMILIES}
    changes: dict[str, dict] = {}

    # ---------------- refill / dispensing ----------------
    m = self_mask = ps.fill_day <= t
    fd, dl, sup = ps.fill_day[m], ps.fill_delay[m], ps.fill_supply[m]

    f["refill_n_fills_365"] = float(np.sum(fd >= t - 365))
    if fd.size >= 2:
        intervals = np.diff(fd)
        f["refill_interval_mean"] = _safe(np.mean(intervals))
        f["refill_interval_cv"] = _safe(np.std(intervals) / max(np.mean(intervals), 1e-6))
    # Expected dispensing interval for this patient, used to scale everything
    # below. Taken from the prescription rather than assumed.
    supply_days = float(np.mean([m_["days_supply"] for m_ in ps.meds])) if ps.meds else 30.0
    f["refill_supply_days"] = supply_days

    if fd.size >= 1:
        families["refill"] = True
        f["refill_last_delay"] = _safe(dl[-1])
        f["refill_days_since_last"] = float(t - fd[-1])
        expected_supply = float(sup[-1]) if sup.size else 30.0
        f["refill_overdue_ratio"] = float((t - fd[-1]) / max(expected_supply, 1.0))
        recent, base = dl[-3:], dl[:-3]
        f["refill_delay_recent_mean"] = _safe(np.mean(recent))
        f["refill_delay_baseline_mean"] = _safe(np.mean(base)) if base.size else _safe(np.mean(dl))
        f["refill_delay_excess"] = f["refill_delay_recent_mean"] - f["refill_delay_baseline_mean"]
        f["refill_last_delay_ratio"] = f["refill_last_delay"] / supply_days
        f["refill_recent_delay_ratio"] = f["refill_delay_recent_mean"] / supply_days
        f["refill_delay_excess_ratio"] = f["refill_delay_excess"] / supply_days
        f["refill_delay_slope"] = _slope(fd, dl, per=90.0)
        late = dl > 5
        run = 0
        for v in late[::-1]:
            if v:
                run += 1
            else:
                break
        f["refill_consecutive_late"] = float(run)

        # The clinical floor adapts to the regimen. Three and a half days is
        # the right threshold for a monthly supply and far too sensitive for a
        # ninety-day one, where the same slip is proportionally trivial.
        adaptive_floor = max(CLINICAL_FLOOR["refill_delay_days"],
                             CLINICAL_FLOOR["refill_delay_fraction_of_supply"] * supply_days)
        cr = detect_change(dl, fd, higher_is_worse=True,
                           min_absolute_effect=adaptive_floor)
        changes["refill_delay"] = cr.to_dict()
        f["refill_cp_effect_sd"] = cr.effect_sd
        f["refill_cp_persistent"] = 1.0 if cr.status == C.CHANGE_PERSISTENT else 0.0
        f["refill_cp_temporary"] = 1.0 if cr.status == C.CHANGE_TEMPORARY else 0.0

        # Facts, one per genuinely notable observation.
        if dl[-1] > max(3.0, 0.1 * supply_days):
            ratio = dl[-1] / supply_days
            qualifier = (f" ({ratio * 100:.0f}% of a {supply_days:.0f}-day supply)"
                         if abs(supply_days - 30) > 10 else "")
            facts.append({"family": "refill", "weight": min(1.0, ratio / 0.8),
                          "text": (f"Most recent dispensing was {dl[-1]:.0f} days later than "
                                   f"scheduled{qualifier}"),
                          "day": int(fd[-1])})
        if run >= 2:
            facts.append({"family": "refill", "weight": min(1.0, run / 4.0),
                          "text": f"{run} consecutive dispensing events were more than 5 days late",
                          "day": int(fd[-1])})
        if f["refill_overdue_ratio"] > 1.25:
            facts.append({"family": "refill", "weight": min(1.0, f["refill_overdue_ratio"] / 2.5),
                          "text": (f"{t - fd[-1]:.0f} days since the last dispensing, against a "
                                   f"{expected_supply:.0f}-day supply"),
                          "day": int(fd[-1])})
        if cr.status == C.CHANGE_PERSISTENT:
            facts.append({"family": "refill", "weight": min(1.0, abs(cr.effect_sd) / 3.0),
                          "text": (f"Dispensing timing shifted from a personal baseline of "
                                   f"{cr.baseline_mean:.1f} days late to {cr.recent_mean:.1f} days "
                                   f"and stayed there"),
                          "day": int(cr.change_position or fd[-1])})
        elif cr.status == C.CHANGE_TEMPORARY:
            facts.append({"family": "refill", "weight": 0.15,
                          "text": (f"Dispensing timing moved away from baseline and returned "
                                   f"within {cr.n_points - (cr.change_index or 0)} fills"),
                          "day": int(cr.change_position or fd[-1])})

    # PDC / MPR, averaged across medications the way payers compute them.
    # PDC is conventionally measured over 90 days, which is fine for a monthly
    # supply and unfair to a quarterly one: a single fill falling just outside
    # the window collapses the score. Since the baseline is the comparator we
    # claim to beat, it gets a window sized to the patient's own supply period
    # as well, so the comparison is against PDC at its best rather than against
    # a windowing artefact we introduced.
    adaptive_window = int(max(90.0, round(supply_days * 1.5)))
    pdc_90s, pdc_180s, pdc_prior, mpr_90s, pdc_adapt = [], [], [], [], []
    for med in ps.meds:
        mid = med["medication_id"]
        sel = (ps.fill_med == mid) & (ps.fill_day <= t) if ps.fill_med.size else np.array([], bool)
        d = ps.fill_day[sel] if sel.size else np.array([])
        s = ps.fill_supply[sel] if sel.size else np.array([])
        if int(med["start_day"]) > t - 60:
            continue
        pdc_90s.append(B.pdc(d, s, t - 90, t))
        pdc_adapt.append(B.pdc(d, s, t - adaptive_window, t))
        pdc_180s.append(B.pdc(d, s, t - 180, t))
        pdc_prior.append(B.pdc(d, s, t - 180, t - 90))
        mpr_90s.append(B.mpr(d, s, t - 90, t))
    if pdc_90s:
        f["refill_pdc_90"] = _safe(np.nanmean(pdc_90s))
        f["refill_pdc_adaptive"] = _safe(np.nanmean(pdc_adapt))
        f["refill_pdc_180"] = _safe(np.nanmean(pdc_180s))
        f["refill_mpr_90"] = _safe(np.nanmean(mpr_90s))
        f["refill_pdc_delta"] = f["refill_pdc_90"] - _safe(np.nanmean(pdc_prior))
        if f["refill_pdc_90"] < B.PDC_ADHERENT_THRESHOLD:
            facts.append({"family": "refill", "weight": min(1.0, (0.8 - f["refill_pdc_90"]) / 0.5),
                          "text": (f"Proportion of days covered over 90 days is "
                                   f"{f['refill_pdc_90']:.2f}, below the 0.80 threshold"),
                          "day": int(t)})

    # ---------------- prescription ----------------
    active = [m_ for m_ in ps.meds if int(m_["start_day"]) <= t]
    f["rx_n_medications"] = float(len(active))
    f["rx_total_daily_doses"] = float(sum(m_["doses_per_day"] for m_ in active)) or 0.0
    if ps.ev_day.size:
        em = ps.ev_day <= t
        types = ps.ev_type[em]
        days = ps.ev_day[em]
        change_types = {"dose_change", "medication_switch", "medication_added", "prescription_renewed"}
        is_change = np.array([ty in change_types for ty in types], dtype=bool)
        recent_changes = is_change & (days >= t - 180)
        f["rx_changes_180"] = float(recent_changes.sum())
        if is_change.any():
            families["prescription"] = True
            f["rx_days_since_change"] = float(t - days[is_change].max())
        f["rx_stopped_count"] = float(np.sum(types == "dispensing_stopped"))
        if f["rx_changes_180"] >= 3:
            facts.append({"family": "prescription", "weight": 0.45,
                          "text": f"{f['rx_changes_180']:.0f} regimen changes recorded in the last 180 days",
                          "day": int(t)})
        if f["rx_stopped_count"] > 0:
            facts.append({"family": "prescription", "weight": 0.7,
                          "text": "Dispensing has stopped for at least one prescribed medication",
                          "day": int(t)})
    if f["rx_total_daily_doses"] >= 4:
        families["prescription"] = True
        facts.append({"family": "prescription", "weight": 0.35,
                      "text": f"Regimen requires {f['rx_total_daily_doses']:.0f} doses per day across "
                              f"{len(active)} medications",
                      "day": int(t)})

    # ---------------- appointments ----------------
    am = ps.appt_day <= t
    ad, aa = ps.appt_day[am], ps.appt_attended[am]
    if ad.size:
        families["appointment"] = True
        f["appt_n_365"] = float(np.sum(ad >= t - 365))
        for col, win in (("appt_missed_rate_365", 365), ("appt_missed_rate_180", 180)):
            sel = ad >= t - win
            f[col] = float(1.0 - np.mean(aa[sel])) if sel.any() else 0.0
        older = ad < t - 180
        f["appt_missed_rate_excess"] = f["appt_missed_rate_180"] - (
            float(1.0 - np.mean(aa[older])) if older.any() else f["appt_missed_rate_180"])
        run = 0
        for v in aa[::-1]:
            if v == 0:
                run += 1
            else:
                break
        f["appt_consecutive_missed"] = float(run)
        attended_days = ad[aa == 1]
        f["appt_days_since_attended"] = float(t - attended_days[-1]) if attended_days.size else float(t)
        if run >= 2:
            facts.append({"family": "appointment", "weight": min(1.0, run / 3.0),
                          "text": f"{run} consecutive scheduled follow-ups were not attended",
                          "day": int(ad[-1])})
        elif run == 1:
            facts.append({"family": "appointment", "weight": 0.25,
                          "text": "The most recent scheduled follow-up was not attended",
                          "day": int(ad[-1])})
        if f["appt_days_since_attended"] > 200:
            facts.append({"family": "appointment", "weight": 0.5,
                          "text": f"{f['appt_days_since_attended']:.0f} days since the last attended review",
                          "day": int(t)})

    # ---------------- symptoms ----------------
    sm = ps.sym_day <= t
    sd, sv = ps.sym_day[sm], ps.sym_val[sm]
    if sv.size >= 3:
        families["symptom"] = True
        recent = sv[sd >= t - 90]
        base = sv[sd < t - 90]
        f["sym_last"] = _safe(sv[-1])
        f["sym_mean_90"] = _safe(np.mean(recent)) if recent.size else _safe(np.mean(sv))
        f["sym_baseline_mean"] = _safe(np.mean(base)) if base.size else f["sym_mean_90"]
        f["sym_delta"] = f["sym_mean_90"] - f["sym_baseline_mean"]
        f["sym_variability"] = _safe(np.std(sv[-12:]))
        if recent.size >= 3:
            f["sym_slope_90"] = _slope(sd[sd >= t - 90], recent, per=90.0)
        cs = detect_change(sv, sd, higher_is_worse=True,
                           min_absolute_effect=CLINICAL_FLOOR["symptom_points"])
        changes["symptom"] = cs.to_dict()
        f["sym_cp_effect_sd"] = cs.effect_sd
        f["sym_cp_persistent"] = 1.0 if cs.status == C.CHANGE_PERSISTENT else 0.0
        if f["sym_delta"] > 0.8:
            facts.append({"family": "symptom", "weight": min(1.0, f["sym_delta"] / 3.0),
                          "text": (f"Reported symptom burden rose from {f['sym_baseline_mean']:.1f} "
                                   f"to {f['sym_mean_90']:.1f} on a 0-10 scale"),
                          "day": int(sd[-1])})

    # ---------------- laboratory ----------------
    lm = ps.lab_day <= t
    ld, lv = ps.lab_day[lm], ps.lab_val[lm]
    if lv.size >= 3:
        families["laboratory"] = True
        f["lab_n_results"] = float(lv.size)
        early = lv[:max(2, lv.size // 2)]
        mu = float(np.mean(early))
        sd_ = _sd_floor(float(np.std(early, ddof=1)) if early.size > 1 else 0.0, mu)
        f["lab_last_z"] = float(np.clip(ps.lab_direction * (lv[-1] - mu) / sd_, -Z_CLIP, Z_CLIP))
        f["lab_slope_per_90d"] = float(np.clip(
            ps.lab_direction * _slope(ld, lv, per=90.0) / sd_, -Z_CLIP, Z_CLIP))
        cl = detect_change(lv, ld, higher_is_worse=(ps.lab_direction > 0))
        changes["laboratory"] = cl.to_dict()
        f["lab_cp_effect_sd"] = cl.effect_sd
        f["lab_cp_persistent"] = 1.0 if cl.status == C.CHANGE_PERSISTENT else 0.0
        if f["lab_last_z"] > 1.2:
            marker = ps.p.get("lab_marker", "laboratory marker")
            facts.append({"family": "laboratory", "weight": min(1.0, f["lab_last_z"] / 3.5),
                          "text": (f"{marker} has moved {abs(f['lab_last_z']):.1f} standard deviations "
                                   f"from this patient's own earlier range"),
                          "day": int(ld[-1])})

    # ---------------- behavioural ----------------
    wm = ps.act_day <= t
    if ps.act_day.size and int(np.sum(wm)) >= 8:
        families["behavioural"] = True
        f["beh_has_wearable"] = 1.0
        for col, arr in (("beh_steps_z", ps.act_steps), ("beh_sleep_z", ps.act_sleep),
                         ("beh_rhr_z", ps.act_rhr)):
            a = arr[wm]
            base, recent = a[:-4], a[-4:]
            if base.size >= 4:
                z = _z(float(np.mean(recent)), float(np.mean(base)),
                       float(np.std(base, ddof=1)))
                # Sign every behavioural feature so that positive means "worse":
                # fewer steps, less sleep, higher resting heart rate.
                f[col] = z if col == "beh_rhr_z" else -z

    # ---------------- cross-signal velocity ----------------
    # Rates are put on a common scale - standard deviations of change per 90
    # days - so that a symptom scale, a laboratory marker and a dispensing
    # interval can be compared without one unit dominating by accident.
    clin_terms = []
    if families["symptom"] and f["sym_variability"] > 0:
        clin_terms.append(float(np.clip(f["sym_slope_90"] / max(f["sym_variability"], 0.3), -6, 6)))
    if families["laboratory"]:
        clin_terms.append(float(np.clip(f["lab_slope_per_90d"], -6, 6)))
    clinical_v = float(np.mean(clin_terms)) if clin_terms else 0.0

    supply_ref = max(f["refill_supply_days"], 7.0)
    refill_v = float(np.clip(f["refill_delay_slope"] / supply_ref, -6, 6))

    f["xs_clinical_velocity"] = clinical_v
    f["xs_refill_velocity"] = refill_v
    # Deteriorating clinically while collection timing holds steady. Positive
    # only when the two genuinely disagree.
    f["xs_divergence"] = float(max(0.0, clinical_v) * max(0.0, 0.25 - refill_v) * 4.0)
    # Both channels worsening together: the ordinary, legible case.
    f["xs_concordance"] = float(max(0.0, clinical_v) * max(0.0, refill_v))
    if len(clin_terms) == 2:
        a, b = clin_terms
        f["xs_symptom_lab_agreement"] = float(np.sign(a) * np.sign(b) * min(abs(a), abs(b)))

    if f["xs_divergence"] > 0.8:
        facts.append({"family": "refill", "weight": min(1.0, f["xs_divergence"] / 3.0),
                      "text": ("Clinical measures are drifting while dispensing has stayed on "
                               "schedule, a pattern the days-covered metric cannot show"),
                      "day": int(t)})

    # ---------------- context ----------------
    f["ctx_age"] = float(ps.p["age"])

    return {
        "features": f,
        "facts": sorted(facts, key=lambda x: -x["weight"]),
        "families": families,
        "changes": changes,
        "day": int(t),
    }


def build_series(tables: dict[str, pd.DataFrame]) -> dict[str, PatientSeries]:
    """Index the cohort tables into one PatientSeries per patient."""
    out: dict[str, PatientSeries] = {}
    by = {name: (df.groupby("patient_id") if len(df) else None)
          for name, df in tables.items() if name != "patients"}

    def grp(name: str, pid: str, cols: list[str]) -> pd.DataFrame:
        g = by.get(name)
        if g is None:
            return pd.DataFrame(columns=cols)
        try:
            return g.get_group(pid)
        except KeyError:
            return pd.DataFrame(columns=cols)

    for patient in tables["patients"].to_dict("records"):
        pid = patient["patient_id"]
        out[pid] = PatientSeries(
            patient=patient,
            medications=grp("medications", pid, ["medication_id", "name", "doses_per_day",
                                                 "days_supply", "start_day", "end_day", "consequence"]),
            refills=grp("refills", pid, ["medication_id", "actual_day", "delay_days",
                                         "days_supplied", "expected_day", "prescribed_day"]),
            appointments=grp("appointments", pid, ["scheduled_day", "attended", "appointment_type"]),
            symptoms=grp("symptoms", pid, ["day", "score"]),
            labs=grp("labs", pid, ["day", "value", "marker", "unit", "direction"]),
            activity=grp("activity", pid, ["day", "steps", "sleep_hours", "resting_hr"]),
            events=grp("events", pid, ["day", "event_type", "detail"]),
        )
    return out
