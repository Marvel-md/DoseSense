#!/usr/bin/env python3
"""
Benchmark: is the result stable, and is it fast enough to deploy?

The evaluation in ml/evaluate.py answers "how good is the model on this test
set". It cannot answer the two questions that follow, and both get asked:

1. *Is that number real, or did you get a lucky split?* The test set is roughly
   150 patients. Retraining across independent seeds - each one a fresh cohort,
   a fresh split and a fresh model - shows how much the headline figures move.
   If PR-AUC swings by 0.1 across seeds, a single reported value means little.
   This is the honest answer to "your sample is small".

2. *Can it run on a real panel?* Scoring latency per patient, and how the
   pipeline scales with cohort size. A model that needs a minute per patient is
   a research artefact regardless of its AUC.

Usage:
    python scripts/benchmark.py                    # full run, a few minutes
    python scripts/benchmark.py --seeds 3 --quick  # faster sanity check
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np   # noqa: E402

from dosesense import config as C          # noqa: E402
from dosesense import datagen, evaluate as E, explain, io, model as M, panel as P, pipeline  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Seed stability
# ---------------------------------------------------------------------------


def run_one_seed(seed: int, n_patients: int) -> dict:
    """Full generate -> panel -> split -> train -> evaluate cycle on one seed."""
    tables = datagen.generate_cohort(n_patients=n_patients, seed=seed)
    pan, _ = P.build_panel(tables)
    splits = P.split_patients(pan, seed=seed)
    mdl = M.train_model(pan, splits, seed=seed)

    te = pan[splits["test"]].reset_index(drop=True)
    y = te["label"].to_numpy(dtype=int)
    prob = mdl.predict(te[mdl.feature_columns].to_numpy(dtype=float))["probability"]
    flags = (prob >= C.ALERT_PROBABILITY_THRESHOLD).astype(int)

    m = E.classification_metrics(y, prob, threshold=C.ALERT_PROBABILITY_THRESHOLD)
    b = E.classification_metrics(y, te["baseline_score"].to_numpy(dtype=float),
                                 flag=te["baseline_flag"].to_numpy())
    burden = E.alert_burden(te, flags)
    base_burden = E.alert_burden(te, te["baseline_flag"].to_numpy())
    cal = E.calibration_curve(y, prob)

    arch = {r["archetype"]: r for r in E.archetype_behaviour(
        te, prob, te["baseline_flag"].to_numpy())}
    neg = [r for r in arch.values() if r["is_negative_by_construction"]]
    w = [r["n_snapshots"] for r in neg] or [1]

    # Raw counts per archetype. Rates on 20-odd snapshots from two patients are
    # not interpretable on their own; pooling the counts across seeds is the
    # only way to say anything about a rare pattern.
    per_arch = {}
    base_flags = te["baseline_flag"].to_numpy(dtype=int)
    for name, grp in te.groupby("archetype"):
        idx = te.index.get_indexer(grp.index)
        per_arch[name] = {
            "n": int(len(grp)),
            "n_patients": int(grp["patient_id"].nunique()),
            "model_alerts": int(flags[idx].sum()),
            "baseline_alerts": int(base_flags[idx].sum()),
            "true_positives": int(grp["label"].sum()),
        }

    return {
        "seed": seed,
        "n_test_patients": int(te["patient_id"].nunique()),
        "model_pr_auc": m["pr_auc"],
        "model_roc_auc": m["roc_auc"],
        "model_f1": m["f1"],
        "model_precision": m["precision"],
        "model_recall": m["recall"],
        "baseline_pr_auc": b["pr_auc"],
        "baseline_f1": b["f1"],
        "pr_auc_delta": round(m["pr_auc"] - b["pr_auc"], 4),
        "ece": cal["expected_calibration_error"],
        "false_alerts_per_100pm": burden["false_alerts_per_100_patient_months"],
        "baseline_false_alerts_per_100pm": base_burden["false_alerts_per_100_patient_months"],
        "adversarial_fp_model": round(float(np.average(
            [r["model_alert_rate"] for r in neg] or [0], weights=w)), 4),
        "adversarial_fp_baseline": round(float(np.average(
            [r.get("baseline_alert_rate", 0) for r in neg] or [0], weights=w)), 4),
        "silent_recall_model": arch.get("SILENT_NONADHERENCE", {}).get("model_alert_rate"),
        "silent_recall_baseline": arch.get("SILENT_NONADHERENCE", {}).get("baseline_alert_rate"),
        "per_archetype": per_arch,
    }


def summarise(runs: list[dict], key: str) -> dict:
    vals = [r[key] for r in runs if r.get(key) is not None]
    if not vals:
        return {}
    return {
        "mean": round(float(stats.mean(vals)), 4),
        "sd": round(float(stats.stdev(vals)), 4) if len(vals) > 1 else 0.0,
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
        "spread": round(max(vals) - min(vals), 4),
    }


# ---------------------------------------------------------------------------
# 2. Latency
# ---------------------------------------------------------------------------


def benchmark_latency(n_patients: int = 200, seed: int = C.RANDOM_SEED) -> dict:
    """Time each stage, and the end-to-end cost of assessing one patient."""
    out: dict = {}

    t0 = time.perf_counter()
    tables = datagen.generate_cohort(n_patients=n_patients, seed=seed)
    out["generate_cohort_s"] = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    pan, _ = P.build_panel(tables)
    dt = time.perf_counter() - t0
    out["build_panel_s"] = round(dt, 3)
    out["feature_extraction_ms_per_snapshot"] = round(1000 * dt / max(len(pan), 1), 3)

    splits = P.split_patients(pan, seed=seed)
    t0 = time.perf_counter()
    mdl = M.train_model(pan, splits, seed=seed)
    out["train_s"] = round(time.perf_counter() - t0, 3)

    X = pan[mdl.feature_columns].to_numpy(dtype=float)
    mdl.predict(X[:32])                                    # warm up
    t0 = time.perf_counter()
    for _ in range(5):
        mdl.predict(X)
    dt = (time.perf_counter() - t0) / 5
    out["batch_predict_s"] = round(dt, 4)
    out["predict_us_per_row"] = round(1e6 * dt / len(X), 1)
    out["predict_rows_per_second"] = int(len(X) / max(dt, 1e-9))

    # Full assessment: features + ensemble + SHAP + barriers + priority.
    ex = explain.Explainer(mdl)
    t0 = time.perf_counter()
    eng = pipeline.DoseSenseEngine(tables, mdl, ex, history_snapshots=8)
    dt = time.perf_counter() - t0
    out["score_full_cohort_s"] = round(dt, 2)
    out["full_assessment_ms_per_patient"] = round(1000 * dt / n_patients, 1)
    # Each patient gets 8 snapshots scored, so per-assessment cost is lower.
    out["full_assessment_ms_per_snapshot"] = round(1000 * dt / (n_patients * 8), 1)

    t0 = time.perf_counter()
    for _ in range(20):
        eng.queue()
    out["queue_build_ms"] = round(1000 * (time.perf_counter() - t0) / 20, 2)

    # What a real panel would cost, extrapolated from measured throughput.
    per_patient = dt / n_patients
    out["projected_rescore_10k_panel_minutes"] = round(10_000 * per_patient / 60, 1)
    return out


# ---------------------------------------------------------------------------
# 3. Cohort-size scaling
# ---------------------------------------------------------------------------


def benchmark_scaling(sizes=(150, 300, 600), seed: int = C.RANDOM_SEED) -> list[dict]:
    """Does more training data help, and where does it stop helping?

    Relevant because the honest question about a synthetic cohort is whether 600
    patients is enough. A curve that has flattened suggests the method is not
    data-starved at this scale; one still climbing steeply suggests the reported
    figures understate what the approach could do.
    """
    rows = []
    for n in sizes:
        t0 = time.perf_counter()
        r = run_one_seed(seed, n)
        r["n_patients"] = n
        r["wall_s"] = round(time.perf_counter() - t0, 1)
        rows.append(r)
    return rows


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5, help="independent seeds for the stability run")
    ap.add_argument("--patients", type=int, default=600)
    ap.add_argument("--quick", action="store_true", help="smaller cohorts, skip scaling")
    ap.add_argument("--out", type=str, default=str(ROOT / "artifacts" / "benchmark.json"))
    args = ap.parse_args()

    n = 250 if args.quick else args.patients
    results: dict = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "n_patients": n, "n_seeds": args.seeds}

    print("=" * 78)
    print(" DoseSense benchmark")
    print("=" * 78)

    # -- 1. stability -------------------------------------------------------
    print(f"\n[1/3] Seed stability — {args.seeds} independent cohorts of {n} patients")
    print("      Each seed regenerates the data, resplits and retrains from scratch.\n")
    runs = []
    for i in range(args.seeds):
        seed = C.RANDOM_SEED + i * 1009
        t0 = time.perf_counter()
        r = run_one_seed(seed, n)
        runs.append(r)
        print(f"   seed {seed:<10} PR-AUC {r['model_pr_auc']:.4f} "
              f"(base {r['baseline_pr_auc']:.4f}, +{r['pr_auc_delta']:.4f})   "
              f"F1 {r['model_f1']:.3f}   ECE {r['ece']:.3f}   "
              f"[{time.perf_counter() - t0:.0f}s]")
    results["seed_runs"] = runs

    keys = ["model_pr_auc", "baseline_pr_auc", "pr_auc_delta", "model_f1", "baseline_f1",
            "model_precision", "model_recall", "ece", "false_alerts_per_100pm",
            "baseline_false_alerts_per_100pm", "adversarial_fp_model",
            "adversarial_fp_baseline", "silent_recall_model", "silent_recall_baseline"]
    results["stability"] = {k: summarise(runs, k) for k in keys}

    print(f"\n   {'metric':38s}{'mean':>9s}{'sd':>8s}{'min':>8s}{'max':>8s}")
    for k in keys:
        s = results["stability"].get(k)
        if s:
            print(f"   {k:38s}{s['mean']:>9.4f}{s['sd']:>8.4f}{s['min']:>8.4f}{s['max']:>8.4f}")

    deltas = [r["pr_auc_delta"] for r in runs]
    beat = sum(1 for d in deltas if d > 0)
    print(f"\n   The model beat the PDC baseline on {beat}/{len(runs)} seeds.")
    print(f"   Smallest advantage observed: +{min(deltas):.4f} PR-AUC.")
    results["beat_baseline_seeds"] = f"{beat}/{len(runs)}"
    results["worst_case_pr_auc_delta"] = min(deltas)

    # -- 1b. pooled per-archetype behaviour ---------------------------------
    pooled: dict[str, dict] = {}
    for r in runs:
        for name, c in r["per_archetype"].items():
            d = pooled.setdefault(name, {"n": 0, "n_patients": 0, "model_alerts": 0,
                                         "baseline_alerts": 0, "true_positives": 0})
            for k in d:
                d[k] += c[k]
    for name, d in pooled.items():
        d["model_alert_rate"] = round(d["model_alerts"] / max(d["n"], 1), 4)
        d["baseline_alert_rate"] = round(d["baseline_alerts"] / max(d["n"], 1), 4)
        d["true_concern_rate"] = round(d["true_positives"] / max(d["n"], 1), 4)
        d["negative_by_construction"] = d["true_positives"] == 0
        # Wilson 95% interval on the model rate, so a rare archetype is reported
        # with its uncertainty rather than as a bare fraction.
        n, k = d["n"], d["model_alerts"]
        if n:
            ph, z = k / n, 1.96
            den = 1 + z * z / n
            centre = (ph + z * z / (2 * n)) / den
            half = z * ((ph * (1 - ph) / n + z * z / (4 * n * n)) ** .5) / den
            d["model_ci95"] = [round(max(0, centre - half), 4), round(min(1, centre + half), 4)]
    results["pooled_archetypes"] = pooled

    print(f"\n[1b] Behaviour per pattern, pooled across all {args.seeds} seeds")
    print("     Rows marked * are adherent by construction, so any alert is a false one.\n")
    print(f"   {'pattern':26s}{'snaps':>7s}{'pts':>5s}{'true':>7s}{'model':>7s}"
          f"{'95% CI':>15s}{'PDC':>7s}{'verdict':>12s}")
    for name, d in sorted(pooled.items(), key=lambda kv: -kv[1]["true_concern_rate"]):
        star = "*" if d["negative_by_construction"] else " "
        ci = f"[{d['model_ci95'][0]:.2f},{d['model_ci95'][1]:.2f}]" if "model_ci95" in d else ""
        # For a mostly-adherent pattern, fewer alerts is the better outcome; for
        # a mostly-non-adherent one, more. Judge each against its own true rate
        # rather than against a single positive snapshot.
        if d["true_concern_rate"] < 0.5:
            worse = d["baseline_alert_rate"] < d["model_ci95"][0]
            verdict = ("WORSE" if worse else
                       "better" if d["model_alert_rate"] < d["baseline_alert_rate"] else "no better")
        else:
            verdict = "better" if d["model_alert_rate"] > d["baseline_alert_rate"] else "no better"
        print(f" {star}{name:26s}{d['n']:>7d}{d['n_patients']:>5d}"
              f"{d['true_concern_rate']:>7.2f}{d['model_alert_rate']:>7.2f}{ci:>15s}"
              f"{d['baseline_alert_rate']:>7.2f}{verdict:>12s}")

    # -- 2. latency ---------------------------------------------------------
    print(f"\n[2/3] Latency — cohort of {min(n, 200)} patients")
    lat = benchmark_latency(n_patients=min(n, 200))
    results["latency"] = lat
    for k in ["generate_cohort_s", "build_panel_s", "feature_extraction_ms_per_snapshot",
              "train_s", "predict_us_per_row", "predict_rows_per_second",
              "full_assessment_ms_per_patient", "queue_build_ms",
              "projected_rescore_10k_panel_minutes"]:
        print(f"   {k:42s} {lat[k]}")

    # -- 3. scaling ---------------------------------------------------------
    if not args.quick:
        print("\n[3/3] Cohort-size scaling — does more data still help?\n")
        sc = benchmark_scaling()
        results["scaling"] = sc
        print(f"   {'patients':>10s}{'PR-AUC':>10s}{'F1':>8s}{'ECE':>8s}"
              f"{'false/100pm':>13s}{'wall':>8s}")
        for r in sc:
            print(f"   {r['n_patients']:>10d}{r['model_pr_auc']:>10.4f}{r['model_f1']:>8.3f}"
                  f"{r['ece']:>8.3f}{r['false_alerts_per_100pm']:>13.1f}{r['wall_s']:>7.0f}s")
        gain = sc[-1]["model_pr_auc"] - sc[0]["model_pr_auc"]
        print(f"\n   PR-AUC gain from {sc[0]['n_patients']} to {sc[-1]['n_patients']} "
              f"patients: {gain:+.4f}")
    else:
        print("\n[3/3] Skipped (--quick)")

    io.save_json(results, args.out)
    print(f"\nWritten {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
