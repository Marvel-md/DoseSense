"""
Personal-baseline change detection.

The central requirement of the brief is to separate a temporary irregularity
from a meaningful pattern. Population thresholds cannot do this: a 34-day gap is
unremarkable for a patient who has always run late and is a red flag for one who
has never been more than a day out. So every series is compared against the
patient's own history.

The method is exact binary segmentation with a Gaussian mean-shift cost and a
BIC-style penalty. It is deliberately simple: on the short series a pharmacy
record produces - typically twelve to eighteen refill intervals - a heavier
method such as PELT or a Bayesian online change-point detector gives no
measurable benefit and costs interpretability, which the alert copy depends on.

Once a change point is found the shift is classified:

  STABLE             no change point clears the penalty and effect-size floor
  TEMPORARY_ANOMALY  the level moved and then came back toward baseline
  PERSISTENT_SHIFT   the level moved and stayed moved

The distinction between the last two is what stops the system firing on the
patient who went on holiday once.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from . import config as C


@dataclass
class ChangeResult:
    status: str
    change_index: int | None
    change_position: float | None      # x-value (e.g. day) at the change point
    baseline_mean: float
    baseline_sd: float
    recent_mean: float
    effect_sd: float                   # shift size in baseline standard deviations
    effect_absolute: float             # shift size in the series' own units
    reverted: bool
    provisional: bool                  # too few points after the change to judge reversion
    after_n: int
    n_points: int
    score: float                       # cost reduction achieved by the split

    def to_dict(self) -> dict:
        return asdict(self)


def _empty(values: np.ndarray) -> ChangeResult:
    mean = float(np.mean(values)) if values.size else 0.0
    sd = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return ChangeResult(
        status=C.CHANGE_STABLE, change_index=None, change_position=None,
        baseline_mean=mean, baseline_sd=sd, recent_mean=mean, effect_sd=0.0,
        effect_absolute=0.0, reverted=False, provisional=False, after_n=0,
        n_points=int(values.size), score=0.0,
    )


def _cost(seg: np.ndarray, var_floor: float) -> float:
    """Gaussian negative log-likelihood cost of modelling a segment by its mean.

    The variance floor is not cosmetic. A one- or two-point segment has a sample
    variance of zero, log of which is unbounded below, so an unfloored cost makes
    the final point the most attractive split in every series. Flooring at a
    fraction of the whole series' variance keeps short tail segments comparable
    to long ones, which is what lets the search look at recent change points at
    all.
    """
    n = seg.size
    if n == 0:
        return 0.0
    return n * float(np.log(max(float(np.var(seg)), var_floor)))


def detect_change(values, positions=None, higher_is_worse: bool = True,
                  min_absolute_effect: float = 0.0) -> ChangeResult:
    """Find at most one mean-shift change point in a short numeric series.

    Args:
        values: the series, oldest first.
        positions: optional x-values (days) aligned with ``values``.
        higher_is_worse: whether an upward shift is the adverse direction. For
            INR or peak flow the adverse direction is downward, and the caller
            passes False so that ``effect_sd`` stays signed toward "worse".
        min_absolute_effect: a clinical floor in the series' own units. A shift
            from a mean of one day late to three days late can be highly
            significant statistically and still not worth a clinician's
            attention. Statistical detectability and clinical meaning are
            different tests and the caller supplies the second one.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = values.size
    if n < 2 * C.CP_MIN_SEGMENT:
        return _empty(values)

    if positions is None:
        positions = np.arange(n, dtype=float)
    positions = np.asarray(positions, dtype=float)[-n:]

    var_floor = max(float(np.var(values)) * 0.05, 1e-8)
    total = _cost(values, var_floor)
    penalty = C.CP_PENALTY_SCALE * np.log(max(n, 2))

    # The right-hand segment is allowed to be shorter than CP_MIN_SEGMENT so a
    # change that began last month can be found. Such a result is marked
    # provisional below rather than being asserted as a sustained shift.
    best_k, best_gain = None, 0.0
    for k in range(C.CP_MIN_SEGMENT, n):
        gain = total - (_cost(values[:k], var_floor) + _cost(values[k:], var_floor)) - penalty
        if gain > best_gain:
            best_k, best_gain = k, float(gain)

    if best_k is None:
        return _empty(values)

    before, after = values[:best_k], values[best_k:]
    base_mean = float(np.mean(before))
    base_sd = float(np.std(before, ddof=1)) if before.size > 1 else 0.0
    scale = max(base_sd, 1e-6, abs(base_mean) * 0.02)
    raw_shift = float(np.mean(after)) - base_mean
    signed = raw_shift if higher_is_worse else -raw_shift
    effect_sd = signed / scale

    if effect_sd < C.CP_MIN_EFFECT_SD or signed < min_absolute_effect:
        # A split that is statistically detectable but clinically trivial is
        # reported as stable rather than dressed up as a finding.
        res = _empty(values)
        res.score = float(best_gain)
        return res

    # Reversion test: has the tail come back toward the old level? With only one
    # or two observations after the change there is no honest answer yet, so the
    # result is marked provisional and the confidence layer downgrades it rather
    # than the detector guessing.
    provisional = bool(after.size < C.CP_MIN_SEGMENT)
    tail_n = max(1, min(after.size, C.CP_MIN_SEGMENT))
    tail_mean = float(np.mean(after[-tail_n:]))
    tail_shift = (tail_mean - base_mean) if higher_is_worse else -(tail_mean - base_mean)
    reverted = bool(not provisional and tail_shift < C.CP_REVERSION_TOLERANCE * signed)

    return ChangeResult(
        status=C.CHANGE_TEMPORARY if reverted else C.CHANGE_PERSISTENT,
        change_index=int(best_k),
        change_position=float(positions[best_k]) if best_k < positions.size else None,
        baseline_mean=base_mean,
        baseline_sd=base_sd,
        recent_mean=float(np.mean(after)),
        effect_sd=round(float(effect_sd), 3),
        effect_absolute=round(float(signed), 3),
        reverted=reverted,
        provisional=provisional,
        after_n=int(after.size),
        n_points=int(n),
        score=round(float(best_gain), 3),
    )
