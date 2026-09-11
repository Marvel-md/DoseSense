"""
The clinical baseline: PDC, MPR and the refill-gap rule.

Proportion of Days Covered is the metric health systems and payers actually use
today, and an 80% threshold is the conventional cut-off for "adherent". If
DoseSense cannot beat this it has no reason to exist, so it is implemented
faithfully rather than as a straw man:

  PDC = (distinct days in the window covered by a dispensed supply) / (window days)
  MPR = (total days supplied in the window) / (window days)

PDC uses the union of coverage intervals and is capped at 1.0. MPR sums supply
and can exceed 1.0 when a patient stockpiles. For a multi-drug regimen both are
computed per medication and averaged, which is the standard approach.

The important limitation, and the reason the brief exists: a dispensing record
tells you a medicine was collected, not that it was taken, and it says nothing
at all until the next fill is due. PDC is silent on symptoms, laboratory drift
and missed follow-up.
"""

from __future__ import annotations

import numpy as np

PDC_ADHERENT_THRESHOLD = 0.80
REFILL_GAP_RULE_DAYS = 7


def coverage_days(fill_days, supply_days, window_start: int, window_end: int) -> int:
    """Number of distinct days in [window_start, window_end) covered by a supply."""
    if window_end <= window_start:
        return 0
    covered = np.zeros(window_end - window_start, dtype=bool)
    for start, supply in zip(fill_days, supply_days):
        lo = max(int(start), window_start)
        hi = min(int(start) + int(supply), window_end)
        if hi > lo:
            covered[lo - window_start:hi - window_start] = True
    return int(covered.sum())


def pdc(fill_days, supply_days, window_start: int, window_end: int) -> float:
    span = window_end - window_start
    if span <= 0:
        return float("nan")
    return coverage_days(fill_days, supply_days, window_start, window_end) / span


def mpr(fill_days, supply_days, window_start: int, window_end: int) -> float:
    span = window_end - window_start
    if span <= 0:
        return float("nan")
    total = 0
    for start, supply in zip(fill_days, supply_days):
        if window_start <= int(start) < window_end:
            total += int(supply)
    return total / span


def baseline_score(pdc_value: float, max_recent_delay: float | None) -> dict:
    """The comparator used in the evaluation.

    Two conventional rules, combined the way an audit report would combine them:
    a PDC below 0.80 in the trailing window, or any recent gap longer than a
    week. The continuous score used for ROC and PR curves is the PDC shortfall,
    which is the only continuous quantity this baseline has.
    """
    if pdc_value is None or not np.isfinite(pdc_value):
        return {"flag": 0, "score": 0.0, "reason": "No dispensing record in the window"}
    shortfall = float(np.clip(PDC_ADHERENT_THRESHOLD - pdc_value, 0.0, 1.0) / PDC_ADHERENT_THRESHOLD)
    gap_flag = bool(max_recent_delay is not None and max_recent_delay > REFILL_GAP_RULE_DAYS)
    flag = int(pdc_value < PDC_ADHERENT_THRESHOLD or gap_flag)
    if pdc_value < PDC_ADHERENT_THRESHOLD:
        reason = f"PDC {pdc_value:.2f} is below the 0.80 threshold"
    elif gap_flag:
        reason = f"A dispensing gap of {max_recent_delay:.0f} days exceeded the 7-day rule"
    else:
        reason = f"PDC {pdc_value:.2f} meets the 0.80 threshold"
    # Blend so that the gap rule contributes to ranking, not only to the flag.
    score = float(np.clip(0.75 * shortfall + (0.25 if gap_flag else 0.0), 0.0, 1.0))
    return {"flag": flag, "score": score, "reason": reason}
