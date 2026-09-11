"""Reading and writing the synthetic dataset and trained artefacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

TABLES = ("patients", "medications", "refills", "appointments", "symptoms",
          "labs", "activity", "events", "ground_truth_states")

# The hidden state file is named so that nobody can include it by accident.
GROUND_TRUTH_TABLE = "ground_truth_states"


def save_tables(tables: dict[str, pd.DataFrame], directory: str | Path) -> dict[str, str]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, df in tables.items():
        path = directory / f"{name}.csv"
        df.to_csv(path, index=False)
        written[name] = str(path)
    return written


def load_tables(directory: str | Path, include_ground_truth: bool = True) -> dict[str, pd.DataFrame]:
    directory = Path(directory)
    out: dict[str, pd.DataFrame] = {}
    for name in TABLES:
        if name == GROUND_TRUTH_TABLE and not include_ground_truth:
            out[name] = pd.DataFrame(columns=["patient_id", "start_day", "end_day", "state"])
            continue
        path = directory / f"{name}.csv"
        out[name] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return out


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=_default)


def load_json(path: str | Path):
    with open(path) as fh:
        return json.load(fh)


def _default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serialisable: {type(o)}")
