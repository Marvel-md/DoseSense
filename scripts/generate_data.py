#!/usr/bin/env python3
"""Generate the synthetic cohort and write it to data/synthetic/.

Usage:
    python scripts/generate_data.py [--patients 600] [--seed 20260910]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dosesense import config as C          # noqa: E402
from dosesense import datagen, io           # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patients", type=int, default=600)
    ap.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    ap.add_argument("--out", type=str, default=str(ROOT / "data" / "synthetic"))
    args = ap.parse_args()

    print(f"Generating {args.patients} synthetic patients over "
          f"{C.OBSERVATION_DAYS} days (seed {args.seed})")
    tables = datagen.generate_cohort(n_patients=args.patients, seed=args.seed)
    written = io.save_tables(tables, args.out)

    print(f"\nWritten to {args.out}:")
    for name, path in written.items():
        print(f"  {name:22s} {len(tables[name]):8,d} rows")

    counts = tables["patients"]["archetype"].value_counts()
    print("\nBehavioural archetypes:")
    for name, n in counts.items():
        print(f"  {name:26s} {n:4d}")

    print("\nNote: ground_truth_states.csv holds the hidden adherence trajectory.")
    print("It is used only for evaluation and is never read by the serving path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
