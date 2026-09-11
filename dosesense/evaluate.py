"""
Evaluation.

The headline number for this problem is not accuracy and not ROC-AUC. A panel
where a third of snapshots carry a concern makes accuracy easy, and ROC-AUC is
dominated by the large negative class and rewards ranking the obvious cases
correctly. The numbers that decide whether a care team keeps using the system
are these:

  PR-AUC                  performance on the class that triggers work
  false alerts per        the alert-fatigue budget. A tool that produces four
    clinician-month       unnecessary calls a week is switched off in a month
                          regardless of its AUC
  detection lead time     how many days earlier than PDC the concern surfaces,
                          because earlier is the entire clinical value
  calibration             whether a stated 70% means seventy in a hundred, which
                          matters because the number is multiplied by a
                          consequence weight to set priority
  archetype behaviour     per-pattern false-alert rates, which is the only place
                          the "temporary versus meaningful" requirement is
                          actually tested

Everything here is computed on patients held out at the patient level, and the
calibration split is disjoint from both training and test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)

from . import config as C
from .features import FEATURE_FAMILIES
from .panel import onset_day


# ---------------------------------------------------------------------------
# Core discrimination metrics
# ---------------------------------------------------------------------------


def classification_metrics(y_true, score, threshold: float = 0.5, flag=None) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    yhat = np.asarray(flag, dtype=int) if flag is not None else (score >= threshold).astype(int)
    out = {
        "n": int(y_true.size),
        "positive_rate": round(float(y_true.mean()), 4),
        "threshold": threshold if flag is None else None,
        "precision": round(float(precision_score(y_true, yhat, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, yhat, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, yhat, zero_division=0)), 4),
        "alert_rate": round(float(yhat.mean()), 4),
    }
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = round(float(roc_auc_score(y_true, score)), 4)
        out["pr_auc"] = round(float(average_precision_score(y_true, score)), 4)
        out["brier"] = round(float(brier_score_loss(y_true, np.clip(score, 0, 1))), 4)
    tp = int(((yhat == 1) & (y_true == 1)).sum())
    fp = int(((yhat == 1) & (y_true == 0)).sum())
    fn = int(((yhat == 0) & (y_true == 1)).sum())
    tn = int(((yhat == 0) & (y_true == 0)).sum())
    out["confusion"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    out["specificity"] = round(tn / max(tn + fp, 1), 4)
    return out


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibration_curve(y_true, prob, n_bins: int = 10) -> dict:
    """Reliability table plus expected and maximum calibration error.

    Reported because the probability is not decoration: it is multiplied by a
    clinical consequence weight to order the worklist. If a stated 0.70 is
    really 0.40, the ordering is wrong in a way no AUC would reveal.
    """
    y_true = np.asarray(y_true, dtype=float)
    prob = np.clip(np.asarray(prob, dtype=float), 0, 1)
    edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (prob >= lo) & (prob < hi) if i < n_bins - 1 else (prob >= lo) & (prob <= hi)
        n = int(sel.sum())
        if n == 0:
            bins.append({"bin_low": round(lo, 2), "bin_high": round(hi, 2), "n": 0,
                         "mean_predicted": None, "observed_rate": None})
            continue
        mp = float(prob[sel].mean())
        obs = float(y_true[sel].mean())
        gap = abs(mp - obs)
        ece += (n / prob.size) * gap
        mce = max(mce, gap)
        bins.append({"bin_low": round(lo, 2), "bin_high": round(hi, 2), "n": n,
                     "mean_predicted": round(mp, 4), "observed_rate": round(obs, 4),
                     "gap": round(gap, 4)})
    return {
        "bins": bins,
        "expected_calibration_error": round(float(ece), 4),
        "maximum_calibration_error": round(float(mce), 4),
        "brier": round(float(brier_score_loss(y_true, prob)), 4),
    }


# ---------------------------------------------------------------------------
# Alert burden and lead time
# ---------------------------------------------------------------------------


def alert_burden(panel: pd.DataFrame, flags: np.ndarray) -> dict:
    """False alerts expressed in the unit a care team budgets in.

    Snapshots are 30 days apart, so one false alert per patient-snapshot is one
    per patient-month. Scaled to a 200-patient panel this is the number that
    decides adoption: a clinician will tolerate a handful of unnecessary calls a
    month and will abandon a tool that generates dozens.
    """
    flags = np.asarray(flags, dtype=int)
    y = panel["label"].to_numpy(dtype=int)
    n_snap = len(panel)
    n_patients = panel["patient_id"].nunique()
    months = n_snap * (C.SNAPSHOT_INTERVAL_DAYS / 30.0)
    fp = int(((flags == 1) & (y == 0)).sum())
    tp = int(((flags == 1) & (y == 1)).sum())
    return {
        "n_snapshots": n_snap,
        "n_patients": int(n_patients),
        "patient_months": round(months, 1),
        "false_alerts": fp,
        "true_alerts": tp,
        "false_alerts_per_patient_month": round(fp / max(months, 1e-9), 4),
        "false_alerts_per_100_patient_months": round(100 * fp / max(months, 1e-9), 2),
        "alerts_per_100_patient_months": round(100 * (fp + tp) / max(months, 1e-9), 2),
        "precision_of_alerts": round(tp / max(tp + fp, 1), 4),
    }


def detection_lead_time(panel: pd.DataFrame, flags: np.ndarray) -> dict:
    """Days between a concern becoming true and the system first raising it.

    For every patient who ever warrants a concern, compare the first snapshot
    labelled positive with the first snapshot the detector flagged. Negative
    values mean the detector moved first. Only patients with a genuine onset are
    included; patients flagged who never had an onset are false positives and
    are counted in the alert burden instead.
    """
    panel = panel.copy()
    panel["_flag"] = np.asarray(flags, dtype=int)
    deltas, detected, missed = [], 0, 0

    for pid, grp in panel.groupby("patient_id"):
        grp = grp.sort_values("day")
        onset = onset_day(grp)
        if onset is None:
            continue
        flagged = grp.loc[grp["_flag"] == 1, "day"]
        if not len(flagged):
            missed += 1
            continue
        # First flag at or after the onset, or the last flag before it if the
        # detector moved early.
        after = flagged[flagged >= onset]
        before = flagged[flagged < onset]
        first = int(after.min()) if len(after) else int(before.max())
        deltas.append(first - onset)
        detected += 1

    if not deltas:
        return {"n_with_onset": detected + missed, "n_detected": 0, "n_never_detected": missed}
    d = np.array(deltas, dtype=float)
    return {
        "n_with_onset": detected + missed,
        "n_detected": detected,
        "n_never_detected": missed,
        "detection_rate": round(detected / max(detected + missed, 1), 4),
        "median_days_to_detection": round(float(np.median(d)), 1),
        "mean_days_to_detection": round(float(np.mean(d)), 1),
        "share_detected_at_or_before_onset": round(float(np.mean(d <= 0)), 4),
        "share_detected_within_30d": round(float(np.mean(d <= 30)), 4),
    }


# ---------------------------------------------------------------------------
# Behaviour by archetype: the "temporary versus meaningful" test
# ---------------------------------------------------------------------------


def archetype_behaviour(panel: pd.DataFrame, prob: np.ndarray,
                        baseline_flag: np.ndarray | None = None) -> list[dict]:
    """Per-pattern alert rates, which is where the brief is actually tested.

    ADHERENT_STABLE, MISLEADING_ANOMALY, DISEASE_PROGRESSION and
    OCCASIONAL_IRREGULARITY are all labelled negative by construction. The alert
    rate on them is a false-positive rate against a known, adversarial truth,
    and comparing it to the PDC rule on the same rows is the clearest single
    piece of evidence that multi-signal reasoning is worth the complexity.
    """
    prob = np.asarray(prob, dtype=float)
    flags = (prob >= C.ALERT_PROBABILITY_THRESHOLD).astype(int)
    out = []
    for archetype, grp in panel.groupby("archetype"):
        idx = panel.index.get_indexer(grp.index)
        row = {
            "archetype": archetype,
            "n_snapshots": len(grp),
            "n_patients": int(grp["patient_id"].nunique()),
            "true_concern_rate": round(float(grp["label"].mean()), 4),
            "model_alert_rate": round(float(flags[idx].mean()), 4),
            "model_mean_probability": round(float(prob[idx].mean()), 4),
        }
        if baseline_flag is not None:
            row["baseline_alert_rate"] = round(
                float(np.asarray(baseline_flag, dtype=int)[idx].mean()), 4)
        # For the negative-by-construction archetypes, the alert rate *is* the
        # false-positive rate.
        row["is_negative_by_construction"] = bool(grp["label"].max() == 0)
        out.append(row)
    return sorted(out, key=lambda r: -r["true_concern_rate"])


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------

ABLATIONS = {
    "refill_only": ["refill"],
    "refill_prescription": ["refill", "prescription"],
    "refill_prescription_appointment": ["refill", "prescription", "appointment"],
    "plus_symptoms": ["refill", "prescription", "appointment", "symptom"],
    "plus_laboratory": ["refill", "prescription", "appointment", "symptom", "laboratory"],
    "all_signals": ["refill", "prescription", "appointment", "symptom", "laboratory",
                    "behavioural", "context"],
}


def run_ablation(panel: pd.DataFrame, splits: dict, train_fn) -> list[dict]:
    """Retrain from scratch on progressively larger signal sets.

    Each row is an independently trained and independently calibrated model, not
    the full model with inputs masked at inference. Masking would leave the
    trees splitting on features that are always zero and understate what the
    smaller signal set can achieve, which would flatter the final model.
    """
    test = panel[splits["test"]]
    y = test["label"].to_numpy(dtype=int)
    results = []
    for name, families in ABLATIONS.items():
        cols = [f for fam in families for f in FEATURE_FAMILIES[fam]]
        model = train_fn(panel, splits, feature_columns=cols)
        prob = model.predict(test[cols].to_numpy(dtype=float))["probability"]
        m = classification_metrics(y, prob, threshold=C.ALERT_PROBABILITY_THRESHOLD)
        burden = alert_burden(test, (prob >= C.ALERT_PROBABILITY_THRESHOLD).astype(int))
        lead = detection_lead_time(test, (prob >= C.ALERT_PROBABILITY_THRESHOLD).astype(int))
        results.append({
            "name": name,
            "families": families,
            "n_features": len(cols),
            "roc_auc": m.get("roc_auc"),
            "pr_auc": m.get("pr_auc"),
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "brier": m.get("brier"),
            "false_alerts_per_100_patient_months": burden["false_alerts_per_100_patient_months"],
            "median_days_to_detection": lead.get("median_days_to_detection"),
        })
    return results


# ---------------------------------------------------------------------------
# Subgroups
# ---------------------------------------------------------------------------


def subgroup_metrics(panel: pd.DataFrame, prob: np.ndarray, by: str,
                     min_n: int = 60) -> list[dict]:
    """Performance and calibration within subgroups.

    Sex and insurance tier are deliberately excluded from the model's inputs,
    which prevents the model using them directly but does not prevent a
    correlated feature reproducing the same disparity. That can only be checked
    by measurement, so it is measured. Groups below ``min_n`` snapshots are
    reported with a note rather than a number, because a precision computed on
    twenty rows is noise presented as a finding.
    """
    prob = np.asarray(prob, dtype=float)
    flags = (prob >= C.ALERT_PROBABILITY_THRESHOLD).astype(int)
    out = []
    for value, grp in panel.groupby(by, dropna=False):
        idx = panel.index.get_indexer(grp.index)
        y = grp["label"].to_numpy(dtype=int)
        row = {"group": str(value), "n_snapshots": len(grp),
               "n_patients": int(grp["patient_id"].nunique()),
               "positive_rate": round(float(y.mean()), 4),
               "alert_rate": round(float(flags[idx].mean()), 4)}
        if len(grp) < min_n or len(np.unique(y)) < 2:
            row["note"] = "Too few snapshots or only one outcome class for a stable estimate"
        else:
            m = classification_metrics(y, prob[idx], threshold=C.ALERT_PROBABILITY_THRESHOLD)
            cal = calibration_curve(y, prob[idx], n_bins=5)
            row.update({"roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"],
                        "precision": m["precision"], "recall": m["recall"],
                        "expected_calibration_error": cal["expected_calibration_error"]})
        out.append(row)
    return sorted(out, key=lambda r: -r["n_snapshots"])


def fairness_summary(subgroups: dict[str, list[dict]]) -> dict:
    """Largest observed gap in recall and calibration across each grouping."""
    summary = {}
    for name, rows in subgroups.items():
        usable = [r for r in rows if "recall" in r]
        if len(usable) < 2:
            summary[name] = {"note": "Fewer than two subgroups met the minimum sample size"}
            continue
        recalls = [r["recall"] for r in usable]
        eces = [r["expected_calibration_error"] for r in usable]
        summary[name] = {
            "n_groups_assessed": len(usable),
            "recall_min": min(recalls), "recall_max": max(recalls),
            "recall_gap": round(max(recalls) - min(recalls), 4),
            "ece_min": min(eces), "ece_max": max(eces),
            "ece_gap": round(max(eces) - min(eces), 4),
        }
    return summary


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------


def threshold_sweep(panel: pd.DataFrame, prob: np.ndarray,
                    thresholds=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8)) -> list[dict]:
    """Precision, recall and alert burden across operating points.

    A care team does not inherit a threshold; they choose one against their
    capacity. This table is what that conversation needs.
    """
    y = panel["label"].to_numpy(dtype=int)
    prob = np.asarray(prob, dtype=float)
    rows = []
    for t in thresholds:
        flags = (prob >= t).astype(int)
        m = classification_metrics(y, prob, threshold=t)
        b = alert_burden(panel, flags)
        rows.append({
            "threshold": t, "precision": m["precision"], "recall": m["recall"],
            "f1": m["f1"], "specificity": m["specificity"],
            "alerts_per_100_patient_months": b["alerts_per_100_patient_months"],
            "false_alerts_per_100_patient_months": b["false_alerts_per_100_patient_months"],
        })
    return rows
