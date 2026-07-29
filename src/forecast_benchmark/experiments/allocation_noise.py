"""High-precision allocation-noise classifier experiment.

This module detects jagged multiplicative reporting/allocation noise.  It does
not alter any forecast.  The intended decision is whether the signal is precise
enough to justify a future targeted experiment without damaging clean wells.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from forecast_benchmark.data import PHASES, WellSeries


@dataclass(frozen=True)
class AllocationNoiseAssessment:
    flagged: bool
    score: float
    residual_mad: float
    reversal_rate: float
    large_jump_rate: float
    second_difference_median: float
    phases_used: int
    evidence: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _phase_features(q: np.ndarray) -> tuple[float, float, float, float] | None:
    arr = np.asarray(q, dtype=float)
    idx = np.flatnonzero(np.isfinite(arr) & (arr > 0))[-36:]
    if idx.size < 10:
        return None
    y = np.log(arr[idx])
    x = idx.astype(float)
    X = np.column_stack([np.ones(len(x)), x - x[0]])
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    residual = y - X @ beta
    residual_mad = float(1.4826 * np.median(np.abs(residual - np.median(residual))))
    first = np.diff(y)
    second = np.diff(first)
    if first.size >= 2:
        reversals = (first[:-1] * first[1:] < 0) & (
            np.minimum(np.abs(first[:-1]), np.abs(first[1:])) > 0.08
        )
        reversal_rate = float(np.mean(reversals))
    else:
        reversal_rate = 0.0
    large_jump_rate = float(np.mean(np.abs(first) > 0.20)) if first.size else 0.0
    second_difference_median = float(np.median(np.abs(second))) if second.size else 0.0
    return residual_mad, reversal_rate, large_jump_rate, second_difference_median


def assess_allocation_noise(well: WellSeries) -> AllocationNoiseAssessment:
    """Return a fixed-threshold, high-precision allocation-noise assessment."""
    features = [_phase_features(getattr(well, phase)) for phase in PHASES]
    features = [f for f in features if f is not None]
    if not features:
        return AllocationNoiseAssessment(
            flagged=False,
            score=0.0,
            residual_mad=0.0,
            reversal_rate=0.0,
            large_jump_rate=0.0,
            second_difference_median=0.0,
            phases_used=0,
            evidence=("insufficient_history",),
        )
    aggregate = np.median(np.asarray(features, dtype=float), axis=0)
    residual_mad, reversal_rate, large_jump_rate, second_difference_median = map(float, aggregate)

    # The conjunction is intentional.  A terminal dip can create a high trend
    # residual, and downtime can create large jumps.  Allocation noise is more
    # specifically characterized by repeated jump/reversal behavior across the
    # visible history.
    checks = {
        "high_residual_mad": residual_mad >= 0.16,
        "frequent_reversals": reversal_rate >= 0.33,
        "frequent_large_jumps": large_jump_rate >= 0.35,
        "large_second_differences": second_difference_median >= 0.24,
    }
    flagged = sum(checks.values()) >= 3 and checks["frequent_reversals"] and checks["frequent_large_jumps"]
    scaled = np.asarray(
        [
            residual_mad / 0.16,
            reversal_rate / 0.33,
            large_jump_rate / 0.35,
            second_difference_median / 0.24,
        ],
        dtype=float,
    )
    score = float(np.clip(np.mean(np.minimum(scaled, 2.0)) / 2.0, 0.0, 1.0))
    evidence = tuple(name for name, passed in checks.items() if passed)
    return AllocationNoiseAssessment(
        flagged=flagged,
        score=score,
        residual_mad=residual_mad,
        reversal_rate=reversal_rate,
        large_jump_rate=large_jump_rate,
        second_difference_median=second_difference_median,
        phases_used=len(features),
        evidence=evidence,
    )
