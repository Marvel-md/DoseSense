"""
The prediction model.

Gradient-boosted trees over engineered features, not a deep sequence model. The
reasoning is not that transformers are unfashionable but that the constraints
here favour trees on every axis that matters: a few thousand patient-snapshots
of tabular data with heavy missingness, a hard requirement that every alert be
explainable to a pharmacist, and inference fast enough to rescore a panel of
thousands on a laptop. A sequence model would need an order of magnitude more
patients to beat this, and would make the explanation layer a research problem
of its own.

Two choices carry real weight.

*Bagging for uncertainty.* Five models are trained on bootstrap resamples of
distinct patients with different feature subsamples. The spread of their
predictions is an honest signal about how much the estimate depends on which
patients happened to be in the training set - high for a patient whose pattern
is unlike anything in the data, low for a familiar one. This feeds the
confidence assessment, so the system can say "the models disagree about this
person" rather than reporting a single confident-looking number.

*Calibration on a held-out patient split.* Raw boosted-tree scores are not
probabilities; they are ranking scores that cluster near 0 and 1. Since the
output is shown to a clinician as a percentage and multiplied by a consequence
weight to set priority, it has to mean what it says. An isotonic regression is
fitted on a calibration split of patients seen by neither the trained models nor
the final evaluation.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config as C
from .features import FEATURE_COLUMNS, FEATURE_FAMILIES

N_ENSEMBLE = 5

LGB_PARAMS = dict(
    objective="binary",
    n_estimators=260,
    learning_rate=0.055,
    num_leaves=20,
    min_child_samples=28,
    subsample=0.82,
    subsample_freq=1,
    colsample_bytree=0.72,
    reg_lambda=1.4,
    verbose=-1,
)


@dataclass
class AdherenceModel:
    """A calibrated bagged ensemble plus the metadata needed to reproduce it."""

    boosters: list
    calibrator: object | None
    feature_columns: list[str]
    metadata: dict

    # -- inference ---------------------------------------------------------

    def _raw(self, X: np.ndarray) -> np.ndarray:
        """Per-model probabilities, shape (n_models, n_rows)."""
        return np.vstack([m.predict_proba(X)[:, 1] for m in self.boosters])

    def predict(self, X: np.ndarray) -> dict:
        """Return calibrated probability and ensemble disagreement per row."""
        X = np.asarray(X, dtype=float)
        raw = self._raw(X)
        mean_raw = raw.mean(axis=0)
        sd = raw.std(axis=0)
        if self.calibrator is not None:
            prob = np.asarray(self.calibrator.predict(mean_raw), dtype=float)
        else:
            prob = mean_raw
        return {
            "probability": np.clip(prob, 0.0, 1.0),
            "raw_probability": mean_raw,
            "ensemble_sd": sd,
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.predict(X)["probability"]

    # -- introspection -----------------------------------------------------

    def gain_importance(self) -> dict[str, float]:
        total = np.zeros(len(self.feature_columns), dtype=float)
        for m in self.boosters:
            total += np.asarray(m.booster_.feature_importance(importance_type="gain"), dtype=float)
        total = total / max(total.sum(), 1e-9)
        return {c: round(float(v), 5) for c, v in
                sorted(zip(self.feature_columns, total), key=lambda kv: -kv[1])}

    def family_importance(self) -> dict[str, float]:
        gains = self.gain_importance()
        out: dict[str, float] = {}
        for fam, feats in FEATURE_FAMILIES.items():
            out[fam] = round(float(sum(gains.get(f, 0.0) for f in feats)), 5)
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    # -- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        with open(path.with_suffix(".meta.json"), "w") as fh:
            json.dump(self.metadata, fh, indent=2)

    @staticmethod
    def load(path: str | Path) -> "AdherenceModel":
        with open(path, "rb") as fh:
            return pickle.load(fh)


def train_model(panel, splits: dict, feature_columns: list[str] | None = None,
                seed: int = C.RANDOM_SEED, n_ensemble: int = N_ENSEMBLE,
                calibrate: bool = True) -> AdherenceModel:
    """Train the bagged ensemble and fit calibration on the held-out calibration split."""
    from lightgbm import LGBMClassifier
    from sklearn.isotonic import IsotonicRegression

    cols = feature_columns or FEATURE_COLUMNS
    rng = np.random.default_rng(seed)

    tr = panel[splits["train"]]
    X_tr = tr[cols].to_numpy(dtype=float)
    y_tr = tr["label"].to_numpy(dtype=int)
    groups = tr["patient_id"].to_numpy()
    unique_patients = np.unique(groups)

    boosters = []
    for k in range(n_ensemble):
        # Resample whole patients, not rows: bootstrapping rows would place
        # neighbouring snapshots of the same person in and out of the same bag
        # and understate the real disagreement between models.
        picked = rng.choice(unique_patients, size=unique_patients.size, replace=True)
        idx = np.concatenate([np.flatnonzero(groups == p) for p in np.unique(picked)])
        clf = LGBMClassifier(**LGB_PARAMS, random_state=seed + k * 17)
        clf.fit(X_tr[idx], y_tr[idx])
        boosters.append(clf)

    model = AdherenceModel(
        boosters=boosters, calibrator=None, feature_columns=list(cols),
        metadata={
            "n_ensemble": n_ensemble,
            "n_features": len(cols),
            "n_train_rows": int(len(tr)),
            "n_train_patients": int(unique_patients.size),
            "train_positive_rate": round(float(y_tr.mean()), 4),
            "lgb_params": LGB_PARAMS,
            "seed": seed,
            "calibration": "none",
        },
    )

    if calibrate and splits["calib"].sum() > 40:
        ca = panel[splits["calib"]]
        raw = model._raw(ca[cols].to_numpy(dtype=float)).mean(axis=0)
        y_ca = ca["label"].to_numpy(dtype=int)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw, y_ca)
        model.calibrator = iso
        model.metadata["calibration"] = "isotonic"
        model.metadata["n_calib_rows"] = int(len(ca))
        model.metadata["n_calib_patients"] = int(ca["patient_id"].nunique())

    return model
