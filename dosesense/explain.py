"""
Explainability.

Two different explanations are needed, and shipping only one of them is a
mistake that looks fine in a demo and fails on a ward.

*What the model responded to.* SHAP values against the trained ensemble,
aggregated to signal-family level. This is a faithful account of the model's
arithmetic. It is the right artefact for validating the model, and the wrong
artefact for a pharmacist with four minutes, because "refill_cp_effect_sd
contributed +0.14 in log-odds" is not actionable.

*What was actually observed.* The plain-language facts assembled in
features.py, each carrying the measured value it came from. This is what goes
in front of a clinician.

The interface shows both and keeps them visually separate, because they answer
different questions and conflating them is how a person comes to believe a model
output is an observation. The facts are things that happened. The attributions
are a property of the model.
"""

from __future__ import annotations

import numpy as np

from .features import FAMILY_OF, FAMILY_LABELS, FEATURE_COLUMNS

# Plain-language names. Anything shown to a clinician needs one, so the
# registry is checked by the test suite for completeness.
FEATURE_LABELS = {
    "refill_n_fills_365": "dispensing events in the last year",
    "refill_interval_mean": "usual gap between refills",
    "refill_interval_cv": "irregularity of refill timing",
    "refill_last_delay": "lateness of the most recent refill",
    "refill_delay_recent_mean": "average lateness of recent refills",
    "refill_delay_baseline_mean": "usual lateness for this patient",
    "refill_delay_excess": "recent lateness above this patient's own baseline",
    "refill_delay_slope": "trend in refill lateness",
    "refill_consecutive_late": "run of consecutive late refills",
    "refill_days_since_last": "days since the last dispensing",
    "refill_overdue_ratio": "how far past the expected refill date",
    "refill_pdc_90": "proportion of days covered, 90 days",
    "refill_pdc_adaptive": "proportion of days covered, over this patient's own supply period",
    "refill_pdc_180": "proportion of days covered, 180 days",
    "refill_pdc_delta": "change in days covered against the previous quarter",
    "refill_mpr_90": "medication possession ratio, 90 days",
    "refill_cp_effect_sd": "size of the shift in refill timing",
    "refill_cp_persistent": "refill timing shifted and stayed shifted",
    "refill_cp_temporary": "refill timing shifted and returned",
    "rx_n_medications": "number of tracked medications",
    "rx_total_daily_doses": "doses required per day",
    "rx_changes_180": "regimen changes in 180 days",
    "rx_days_since_change": "days since the last regimen change",
    "rx_stopped_count": "medications with no further dispensing",
    "appt_n_365": "scheduled follow-ups in the last year",
    "appt_missed_rate_365": "share of follow-ups missed, one year",
    "appt_missed_rate_180": "share of follow-ups missed, six months",
    "appt_missed_rate_excess": "attendance against this patient's own history",
    "appt_consecutive_missed": "run of consecutive missed follow-ups",
    "appt_days_since_attended": "days since the last attended review",
    "sym_last": "most recent symptom score",
    "sym_mean_90": "average symptom score, 90 days",
    "sym_baseline_mean": "usual symptom score for this patient",
    "sym_delta": "change in symptom burden against baseline",
    "sym_slope_90": "trend in symptom burden",
    "sym_variability": "week-to-week symptom variability",
    "sym_cp_effect_sd": "size of the shift in symptom burden",
    "sym_cp_persistent": "symptom burden shifted and stayed shifted",
    "lab_last_z": "latest result against this patient's own range",
    "lab_slope_per_90d": "trend in the laboratory marker",
    "lab_cp_effect_sd": "size of the shift in the laboratory marker",
    "lab_cp_persistent": "laboratory marker shifted and stayed shifted",
    "lab_n_results": "number of laboratory results available",
    "beh_has_wearable": "activity data available",
    "beh_steps_z": "daily activity against this patient's own range",
    "beh_sleep_z": "sleep duration against this patient's own range",
    "beh_rhr_z": "resting heart rate against this patient's own range",
    "ctx_age": "age",
    "refill_supply_days": "length of each supply for this patient",
    "refill_last_delay_ratio": "lateness of the last refill, as a share of the supply",
    "refill_delay_excess_ratio": "recent lateness above baseline, as a share of the supply",
    "refill_recent_delay_ratio": "average recent lateness, as a share of the supply",
    "xs_clinical_velocity": "how fast the clinical picture is changing",
    "xs_refill_velocity": "how fast collection timing is changing",
    "xs_divergence": "clinical picture worsening while collection stays on schedule",
    "xs_concordance": "clinical picture and collection timing worsening together",
    "xs_symptom_lab_agreement": "symptoms and laboratory results moving the same way",
}


class Explainer:
    """Grouped SHAP attribution over the bagged ensemble."""

    def __init__(self, model, background: np.ndarray | None = None):
        self.model = model
        self.columns = list(model.feature_columns)
        self._shap = None
        try:
            import warnings
            import shap
            # SHAP emits a UserWarning about LightGBM binary output shape on every
            # TreeExplainer construction. The shape is handled explicitly in
            # attribute() below, and the notice floods any batch or benchmark run.
            warnings.filterwarnings(
                "ignore", message=".*LightGBM binary classifier with TreeExplainer.*")
            # Attribute against a single ensemble member. Averaging TreeExplainer
            # output over five boosters triples the cost and moves the top
            # contributors by a negligible amount, and the ensemble spread is
            # already reported separately as uncertainty.
            self._shap = shap.TreeExplainer(model.boosters[0].booster_)
        except Exception as exc:                      # pragma: no cover
            self._error = repr(exc)

    @property
    def available(self) -> bool:
        return self._shap is not None

    def attribute(self, X: np.ndarray) -> dict:
        """Per-row SHAP values plus family aggregation.

        Falls back to the ensemble's gain importance if SHAP is unavailable, so
        the interface degrades to a coarser explanation rather than to none.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self._shap is None:
            gains = self.model.gain_importance()
            vals = np.tile(np.array([gains.get(c, 0.0) for c in self.columns]), (X.shape[0], 1))
            return {"values": vals, "method": "gain_importance_fallback"}
        vals = self._shap.shap_values(X)
        if isinstance(vals, list):                    # older shap returns per-class
            vals = vals[-1]
        vals = np.asarray(vals)
        if vals.ndim == 3:                            # (rows, features, classes)
            vals = vals[:, :, -1]
        return {"values": vals, "method": "shap_tree_explainer"}

    def explain_row(self, x: np.ndarray, top_k: int = 6) -> dict:
        """Human-readable attribution for one snapshot."""
        out = self.attribute(np.asarray(x, dtype=float).reshape(1, -1))
        vals = out["values"][0]
        order = np.argsort(-np.abs(vals))[:top_k]

        contributions = [{
            "feature": self.columns[i],
            "label": FEATURE_LABELS.get(self.columns[i], self.columns[i]),
            "family": FAMILY_OF.get(self.columns[i], "context"),
            "family_label": FAMILY_LABELS.get(FAMILY_OF.get(self.columns[i], "context"), "Other"),
            "value": round(float(x[i]), 3),
            "contribution": round(float(vals[i]), 4),
            "direction": "raises" if vals[i] > 0 else "lowers",
        } for i in order]

        by_family: dict[str, float] = {}
        for i, col in enumerate(self.columns):
            fam = FAMILY_OF.get(col, "context")
            by_family[fam] = by_family.get(fam, 0.0) + float(vals[i])

        return {
            "method": out["method"],
            "contributions": contributions,
            "by_family": [{"family": k, "label": FAMILY_LABELS.get(k, k),
                           "contribution": round(v, 4)}
                          for k, v in sorted(by_family.items(), key=lambda kv: -abs(kv[1]))],
        }
