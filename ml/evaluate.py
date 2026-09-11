#!/usr/bin/env python3
"""Evaluate the model against the PDC baseline and write artifacts/metrics.json.

Every number quoted in the README, the docs and the presentation comes from this
script. Nothing is typed in by hand.

Usage:
    python ml/evaluate.py [--artifacts artifacts] [--skip-ablation]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                     # noqa: E402
import pandas as pd                                    # noqa: E402

from dosesense import config as C                       # noqa: E402
from dosesense import evaluate as E                     # noqa: E402
from dosesense import io, model as M                    # noqa: E402


def load_panel(out: Path) -> pd.DataFrame:
    if (out / "panel.parquet").exists():
        return pd.read_parquet(out / "panel.parquet")
    return pd.read_csv(out / "panel.csv")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", type=str, default=str(ROOT / "artifacts"))
    ap.add_argument("--skip-ablation", action="store_true")
    args = ap.parse_args()
    out = Path(args.artifacts)

    pan = load_panel(out)
    splits = {k: np.load(out / f"split_{k}.npy") for k in ("train", "calib", "test")}
    mdl = M.AdherenceModel.load(out / "model.pkl")

    test = pan[splits["test"]].reset_index(drop=True)
    y = test["label"].to_numpy(dtype=int)
    prob = mdl.predict(test[mdl.feature_columns].to_numpy(dtype=float))["probability"]
    flags = (prob >= C.ALERT_PROBABILITY_THRESHOLD).astype(int)

    thr = C.ALERT_PROBABILITY_THRESHOLD
    results: dict = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "alert_threshold": thr,
            "snapshot_interval_days": C.SNAPSHOT_INTERVAL_DAYS,
            "label_window_days": C.LABEL_WINDOW_DAYS,
            "label_min_nonadherent_fraction": C.LABEL_MIN_NONADHERENT_FRACTION,
            "min_families_for_alert": C.MIN_FAMILIES_FOR_ALERT,
        },
        "model_metadata": mdl.metadata,
        "dataset": {
            "n_patients": int(pan["patient_id"].nunique()),
            "n_snapshots": int(len(pan)),
            "concern_rate": round(float(pan["label"].mean()), 4),
            "n_test_patients": int(test["patient_id"].nunique()),
            "n_test_snapshots": int(len(test)),
        },
    }

    print("=" * 74)
    print("DoseSense evaluation - held-out patients")
    print("=" * 74)
    print(f"test set: {len(test):,} snapshots from {test['patient_id'].nunique()} patients, "
          f"concern rate {y.mean():.3f}\n")

    # ---------------- headline comparison ----------------
    results["model"] = E.classification_metrics(y, prob, threshold=thr)
    results["baseline_pdc"] = E.classification_metrics(
        y, test["baseline_score"].to_numpy(dtype=float), flag=test["baseline_flag"].to_numpy())

    print(f"{'':26s}{'ROC-AUC':>9s}{'PR-AUC':>9s}{'Prec':>8s}{'Recall':>8s}{'F1':>8s}{'Brier':>8s}")
    for label, key in (("DoseSense", "model"), ("PDC / refill-gap rule", "baseline_pdc")):
        r = results[key]
        print(f"{label:26s}{r.get('roc_auc', float('nan')):>9.4f}"
              f"{r.get('pr_auc', float('nan')):>9.4f}{r['precision']:>8.3f}"
              f"{r['recall']:>8.3f}{r['f1']:>8.3f}{r.get('brier', float('nan')):>8.4f}")
    d_pr = results["model"]["pr_auc"] - results["baseline_pdc"]["pr_auc"]
    d_f1 = results["model"]["f1"] - results["baseline_pdc"]["f1"]
    print(f"\n  PR-AUC improvement {d_pr:+.4f}   F1 improvement {d_f1:+.4f}")
    results["improvement"] = {"pr_auc_delta": round(d_pr, 4), "f1_delta": round(d_f1, 4)}

    # ---------------- alert burden ----------------
    results["alert_burden"] = {
        "model": E.alert_burden(test, flags),
        "baseline_pdc": E.alert_burden(test, test["baseline_flag"].to_numpy()),
    }
    mb, bb = results["alert_burden"]["model"], results["alert_burden"]["baseline_pdc"]
    print("\nAlert burden per 100 patient-months")
    print(f"  DoseSense  total {mb['alerts_per_100_patient_months']:6.1f}   "
          f"false {mb['false_alerts_per_100_patient_months']:6.1f}")
    print(f"  PDC rule   total {bb['alerts_per_100_patient_months']:6.1f}   "
          f"false {bb['false_alerts_per_100_patient_months']:6.1f}")
    red = 1 - mb["false_alerts_per_100_patient_months"] / max(
        bb["false_alerts_per_100_patient_months"], 1e-9)
    print(f"  reduction in false alerts: {red * 100:.1f}%")
    results["alert_burden"]["false_alert_reduction"] = round(float(red), 4)

    # ---------------- lead time ----------------
    results["lead_time"] = {
        "model": E.detection_lead_time(test, flags),
        "baseline_pdc": E.detection_lead_time(test, test["baseline_flag"].to_numpy()),
    }
    ml, bl = results["lead_time"]["model"], results["lead_time"]["baseline_pdc"]
    print("\nDetection lead time (days from true onset to first alert; lower is earlier)")
    for label, r in (("DoseSense", ml), ("PDC rule", bl)):
        print(f"  {label:12s} median {r.get('median_days_to_detection', float('nan')):6.1f}   "
              f"detected {r.get('detection_rate', 0) * 100:5.1f}%   "
              f"at or before onset {r.get('share_detected_at_or_before_onset', 0) * 100:5.1f}%")
    if ml.get("median_days_to_detection") is not None and bl.get("median_days_to_detection") is not None:
        gain = bl["median_days_to_detection"] - ml["median_days_to_detection"]
        print(f"  DoseSense detects a median {gain:+.1f} days earlier")
        results["lead_time"]["median_days_earlier_than_baseline"] = round(float(gain), 1)

    # ---------------- calibration ----------------
    results["calibration"] = E.calibration_curve(y, prob)
    cal = results["calibration"]
    print(f"\nCalibration: ECE {cal['expected_calibration_error']:.4f}   "
          f"max {cal['maximum_calibration_error']:.4f}   Brier {cal['brier']:.4f}")
    print(f"  {'predicted':>12s}{'observed':>12s}{'n':>8s}")
    for b in cal["bins"]:
        if b["n"]:
            print(f"  {b['mean_predicted']:>12.3f}{b['observed_rate']:>12.3f}{b['n']:>8d}")

    # ---------------- archetype behaviour ----------------
    results["archetype_behaviour"] = E.archetype_behaviour(
        test, prob, test["baseline_flag"].to_numpy())
    print("\nBehaviour by pattern. Rows marked * are negative by construction:")
    print(f"  {'pattern':28s}{'n':>6s}{'true':>8s}{'model':>8s}{'PDC':>8s}")
    for r in results["archetype_behaviour"]:
        star = "*" if r["is_negative_by_construction"] else " "
        print(f" {star}{r['archetype']:28s}{r['n_snapshots']:>6d}"
              f"{r['true_concern_rate']:>8.3f}{r['model_alert_rate']:>8.3f}"
              f"{r.get('baseline_alert_rate', float('nan')):>8.3f}")

    neg = [r for r in results["archetype_behaviour"] if r["is_negative_by_construction"]]
    if neg:
        mfp = float(np.average([r["model_alert_rate"] for r in neg],
                               weights=[r["n_snapshots"] for r in neg]))
        bfp = float(np.average([r["baseline_alert_rate"] for r in neg],
                               weights=[r["n_snapshots"] for r in neg]))
        print(f"\n  false-alert rate on adversarial negatives: "
              f"DoseSense {mfp:.3f} vs PDC {bfp:.3f}")
        results["adversarial_negatives"] = {
            "model_false_alert_rate": round(mfp, 4),
            "baseline_false_alert_rate": round(bfp, 4),
            "relative_reduction": round(1 - mfp / max(bfp, 1e-9), 4),
        }

    # ---------------- threshold sweep ----------------
    results["threshold_sweep"] = E.threshold_sweep(test, prob)
    print("\nOperating points")
    print(f"  {'thr':>5s}{'prec':>8s}{'recall':>8s}{'F1':>8s}{'false/100pm':>13s}")
    for r in results["threshold_sweep"]:
        print(f"  {r['threshold']:>5.2f}{r['precision']:>8.3f}{r['recall']:>8.3f}"
              f"{r['f1']:>8.3f}{r['false_alerts_per_100_patient_months']:>13.1f}")

    # ---------------- subgroups ----------------
    subs = {}
    test = test.copy()
    test["age_band"] = pd.cut(test["age"], [0, 45, 60, 75, 200],
                              labels=["<45", "45-59", "60-74", "75+"]).astype(str)
    for by in ("sex", "insurance_tier", "age_band", "condition"):
        subs[by] = E.subgroup_metrics(test, prob, by)
    results["subgroups"] = subs
    results["fairness_summary"] = E.fairness_summary(subs)
    print("\nSubgroup performance (sex and insurance tier are excluded from model inputs)")
    for by in ("sex", "insurance_tier"):
        for r in subs[by]:
            if "recall" in r:
                print(f"  {by:16s}{r['group']:12s} n={r['n_snapshots']:5d}  "
                      f"recall {r['recall']:.3f}  prec {r['precision']:.3f}  "
                      f"ECE {r['expected_calibration_error']:.3f}")
            else:
                print(f"  {by:16s}{r['group']:12s} n={r['n_snapshots']:5d}  {r['note']}")
    for name, s in results["fairness_summary"].items():
        if "recall_gap" in s:
            print(f"  largest recall gap by {name}: {s['recall_gap']:.3f}   "
                  f"largest ECE gap: {s['ece_gap']:.3f}")

    # ---------------- ablation ----------------
    if not args.skip_ablation:
        print("\nAblation: independently trained models on growing signal sets")
        t0 = time.time()
        results["ablation"] = E.run_ablation(pan, splits, M.train_model)
        print(f"  ({time.time() - t0:.1f}s)")
        print(f"  {'signal set':34s}{'feat':>6s}{'PR-AUC':>9s}{'F1':>8s}{'false/100pm':>13s}{'lead':>7s}")
        for r in results["ablation"]:
            print(f"  {r['name']:34s}{r['n_features']:>6d}{r['pr_auc']:>9.4f}{r['f1']:>8.3f}"
                  f"{r['false_alerts_per_100_patient_months']:>13.1f}"
                  f"{r['median_days_to_detection'] if r['median_days_to_detection'] is not None else float('nan'):>7.0f}")

    # ---------------- importance ----------------
    results["family_importance"] = mdl.family_importance()
    results["feature_importance_top20"] = dict(list(mdl.gain_importance().items())[:20])

    io.save_json(results, out / "metrics.json")
    print(f"\nWritten {out / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
