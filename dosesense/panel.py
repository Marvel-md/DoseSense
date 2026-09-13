"""
Panel construction: turning a longitudinal record into a supervised dataset.

DoseSense re-scores every patient on a fixed cadence, so the unit of prediction
is a *patient-snapshot*: one row per patient per 30 days, built only from
records available on that day. This shape is what makes the two most important
evaluation numbers possible - false alerts per clinician-month, and detection
lead time - neither of which can be computed from a single row per patient.

Labelling. A snapshot is labelled a concern when the patient's hidden state was
non-adherent for at least 34% of the trailing 90 days. The threshold is a
deliberate definition of "meaningful", not a tuning knob: a single missed week
inside a quarter does not clear it, a sustained partial pattern does. Patients
who lapse and then recover are labelled positive during the lapse and negative
afterwards, so the model is scored on tracking a state rather than branding a
person.

Splitting is by patient, never by row. Two snapshots of the same patient thirty
days apart share almost all of their history; splitting rows at random would
leak that history across the train/test boundary and inflate every metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import baseline as B
from .features import (PatientSeries, extract_snapshot, build_series,
                       FEATURE_COLUMNS, ALL_FEATURE_COLUMNS)

STATE_WEIGHT = {C.STATE_ADHERENT: 0.0, C.STATE_PARTIAL: 1.0, C.STATE_LAPSED: 1.0}


def state_array(segments: pd.DataFrame, n_days: int) -> np.ndarray:
    """Per-day non-adherence weight from the hidden ground-truth segments."""
    arr = np.zeros(n_days, dtype=float)
    for row in segments.itertuples():
        lo = max(0, int(row.start_day))
        hi = min(n_days, int(row.end_day) + 1)
        arr[lo:hi] = STATE_WEIGHT[row.state]
    return arr


def snapshot_days(n_days: int = C.OBSERVATION_DAYS) -> list[int]:
    return list(range(C.SNAPSHOT_WARMUP_DAYS, n_days + 1, C.SNAPSHOT_INTERVAL_DAYS))


def build_panel(tables: dict[str, pd.DataFrame], keep_detail: bool = False):
    """Build the supervised panel.

    Returns ``(panel_df, detail)`` where ``detail`` maps ``(patient_id, day)`` to
    the full snapshot dict (facts, families, change points) when
    ``keep_detail`` is set. The interface needs the detail; training does not,
    and holding it for 7,800 snapshots is wasteful, so it is opt-in.
    """
    series = build_series(tables)
    gt = tables["ground_truth_states"]
    gt_by = gt.groupby("patient_id") if len(gt) else None

    rows: list[dict] = []
    detail: dict[tuple[str, int], dict] = {}

    for pid, ps in series.items():
        n_days = int(ps.p["observation_days"])
        try:
            segs = gt_by.get_group(pid)
        except (KeyError, AttributeError):
            segs = pd.DataFrame(columns=["start_day", "end_day", "state"])
        nonadh = state_array(segs, n_days)

        for t in snapshot_days(n_days):
            snap = extract_snapshot(ps, t)
            lo = max(0, t - C.LABEL_WINDOW_DAYS)
            window = nonadh[lo:t]
            frac = float(window.mean()) if window.size else 0.0
            label = int(frac >= C.LABEL_MIN_NONADHERENT_FRACTION)

            fam = snap["families"]
            # The comparator is scored on the fairer adaptive window. Giving the
            # baseline its best shot is the only way the improvement we report
            # means anything.
            base = B.baseline_score(
                snap["features"]["refill_pdc_adaptive"] if fam["refill"] else float("nan"),
                snap["features"]["refill_last_delay"] if fam["refill"] else None,
                supply_days=snap["features"].get("refill_supply_days") or 30.0,
            )

            row = {
                "patient_id": pid,
                "day": t,
                "archetype": ps.p["archetype"],
                "condition": ps.p["condition"],
                "sex": ps.p["sex"],
                "age": ps.p["age"],
                "insurance_tier": ps.p["insurance_tier"],
                "consequence": ps.p["consequence"],
                "label": label,
                "nonadherent_fraction": round(frac, 4),
                "n_families": int(sum(fam.values())),
                "baseline_flag": base["flag"],
                "baseline_score": base["score"],
                **snap["features"],
            }
            rows.append(row)
            if keep_detail:
                detail[(pid, t)] = snap

    panel = pd.DataFrame(rows)
    return panel, detail


def split_patients(panel: pd.DataFrame, seed: int = C.RANDOM_SEED,
                   train: float = 0.60, calib: float = 0.15) -> dict[str, np.ndarray]:
    """Patient-level split, stratified by archetype so every split sees every pattern."""
    rng = np.random.default_rng(seed)
    meta = panel[["patient_id", "archetype"]].drop_duplicates()
    assign: dict[str, str] = {}
    for archetype, grp in meta.groupby("archetype"):
        pids = grp["patient_id"].to_numpy()
        rng.shuffle(pids)
        n = len(pids)
        n_tr = max(1, int(round(n * train)))
        n_ca = max(1, int(round(n * calib))) if n - n_tr > 1 else 0
        for p in pids[:n_tr]:
            assign[p] = "train"
        for p in pids[n_tr:n_tr + n_ca]:
            assign[p] = "calib"
        for p in pids[n_tr + n_ca:]:
            assign[p] = "test"
    split = panel["patient_id"].map(assign).to_numpy()
    return {
        "assignment": assign,
        "train": split == "train",
        "calib": split == "calib",
        "test": split == "test",
    }


def onset_day(panel_patient: pd.DataFrame) -> int | None:
    """First snapshot day at which the hidden state warranted a concern label."""
    pos = panel_patient.loc[panel_patient["label"] == 1, "day"]
    return int(pos.min()) if len(pos) else None


def matrix(panel: pd.DataFrame, columns: list[str] | None = None) -> np.ndarray:
    cols = columns or FEATURE_COLUMNS
    return panel[cols].to_numpy(dtype=float)
