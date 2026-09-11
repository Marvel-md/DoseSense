#!/usr/bin/env python3
"""Build the supervised panel, split by patient, train and calibrate the model.

Usage:
    python ml/train.py [--data data/synthetic] [--out artifacts]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                          # noqa: E402

from dosesense import config as C            # noqa: E402
from dosesense import io, model, panel as P   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=str, default=str(ROOT / "data" / "synthetic"))
    ap.add_argument("--out", type=str, default=str(ROOT / "artifacts"))
    ap.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading synthetic cohort")
    tables = io.load_tables(args.data)
    print(f"  {len(tables['patients']):,} patients, {len(tables['refills']):,} dispensing events")

    print("\nBuilding patient-snapshot panel (causal features only)")
    t0 = time.time()
    pan, _ = P.build_panel(tables)
    print(f"  {pan.shape[0]:,} snapshots x {pan.shape[1]} columns in {time.time() - t0:.1f}s")
    print(f"  concern rate {pan['label'].mean():.3f}")

    print("\nSplitting by patient, stratified by archetype")
    splits = P.split_patients(pan, seed=args.seed)
    for name in ("train", "calib", "test"):
        m = splits[name]
        print(f"  {name:6s} {int(m.sum()):5,d} snapshots  "
              f"{pan.loc[m, 'patient_id'].nunique():4d} patients  "
              f"concern rate {pan.loc[m, 'label'].mean():.3f}")

    overlap = set(pan.loc[splits["train"], "patient_id"]) & set(pan.loc[splits["test"], "patient_id"])
    assert not overlap, f"patient leakage across splits: {overlap}"
    print("  no patient appears in more than one split")

    print("\nTraining bagged ensemble")
    t0 = time.time()
    mdl = model.train_model(pan, splits, seed=args.seed)
    print(f"  {mdl.metadata['n_ensemble']} LightGBM models, "
          f"{mdl.metadata['n_features']} features, {time.time() - t0:.1f}s")
    print(f"  calibration: {mdl.metadata['calibration']}")

    pan.to_parquet(out / "panel.parquet") if _parquet_ok() else pan.to_csv(out / "panel.csv", index=False)
    np.save(out / "split_train.npy", splits["train"])
    np.save(out / "split_calib.npy", splits["calib"])
    np.save(out / "split_test.npy", splits["test"])
    mdl.save(out / "model.pkl")

    print("\nSignal-family importance (gain, normalised):")
    for fam, v in mdl.family_importance().items():
        bar = "#" * int(round(v * 50))
        print(f"  {fam:14s} {v:.3f}  {bar}")

    print("\nTop features:")
    for i, (f, v) in enumerate(list(mdl.gain_importance().items())[:10], 1):
        print(f"  {i:2d}. {f:32s} {v:.4f}")

    print(f"\nArtefacts written to {out}")
    return 0


def _parquet_ok() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
