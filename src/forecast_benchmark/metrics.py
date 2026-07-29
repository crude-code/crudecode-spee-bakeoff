"""Forecast error metrics with explicit nonpositive-value accounting.

The physical forecast and the numerical scoring policy are deliberately
separate.  A forecast is allowed to contain an exact zero when the model
represents a shut-in or abandonment state.  Log-error scoring must then choose
and report an explicit policy instead of silently discarding the observation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

ForecastZeroPolicy = Literal["drop", "epsilon", "error"]


@dataclass(frozen=True)
class SPEEDecomposition:
    """Auditable decomposition of the SPEE-style log-error score."""

    median_log_error: float
    stdev_log_error: float
    bias_component: float
    spread_component: float
    score: float
    bias_share: float
    spread_share: float
    n_log_errors: int
    n_actual_nan_dropped: int = 0
    n_nonpositive_actual_dropped: int = 0
    n_nonpositive_forecast_dropped: int = 0
    n_nonpositive_forecast_replaced: int = 0
    forecast_zero_policy: str = "drop"
    metric_epsilon: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _clean_pair(actual: np.ndarray, forecast: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Drop rows where actual is NaN. Return None when nothing remains."""
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    if a.shape != f.shape:
        raise ValueError(f"actual shape {a.shape} != forecast shape {f.shape}")
    mask = np.isfinite(a)
    a, f = a[mask], f[mask]
    if len(a) == 0:
        return None
    return a, f


def mape(actual: np.ndarray, forecast: np.ndarray) -> float | None:
    """Mean absolute percent error; undefined when every actual is zero."""
    pair = _clean_pair(actual, forecast)
    if pair is None:
        return None
    a, f = pair
    valid = (a != 0) & np.isfinite(f)
    if not np.any(valid):
        return None
    return float(np.mean(np.abs((f[valid] - a[valid]) / a[valid])) * 100.0)


def bias(actual: np.ndarray, forecast: np.ndarray) -> float | None:
    """Signed mean percent error. Positive means over-forecast."""
    pair = _clean_pair(actual, forecast)
    if pair is None:
        return None
    a, f = pair
    valid = (a != 0) & np.isfinite(f)
    if not np.any(valid):
        return None
    return float(np.mean((f[valid] - a[valid]) / a[valid]) * 100.0)


def _metric_epsilon(actual_positive: np.ndarray, epsilon_fraction: float) -> float:
    if epsilon_fraction <= 0:
        raise ValueError("epsilon_fraction must be positive")
    scale = float(np.median(actual_positive)) if actual_positive.size else 1.0
    return max(1e-12, epsilon_fraction * max(scale, 1e-12))


def log_error_with_audit(
    actual: np.ndarray,
    forecast: np.ndarray,
    *,
    forecast_zero_policy: ForecastZeroPolicy = "drop",
    epsilon_fraction: float = 1e-6,
) -> tuple[np.ndarray | None, dict[str, int | float | str | None]]:
    """Return log errors and complete row-accounting metadata.

    Actual values <= 0 are outside the logarithmic domain and are excluded.
    Positive actuals paired with forecasts <= 0 are handled explicitly:

    ``drop``
        Compatibility mode. Exclude them and report the count.
    ``epsilon``
        Replace only for metric arithmetic using a scale-relative epsilon.
        The underlying physical forecast is not modified.
    ``error``
        Fail closed. Useful when committee rules prohibit nonpositive values.
    """
    if forecast_zero_policy not in {"drop", "epsilon", "error"}:
        raise ValueError(f"unsupported forecast_zero_policy: {forecast_zero_policy}")
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    if a.shape != f.shape:
        raise ValueError(f"actual shape {a.shape} != forecast shape {f.shape}")

    finite_actual = np.isfinite(a)
    nan_dropped = int(np.size(a) - np.sum(finite_actual))
    a, f = a[finite_actual], f[finite_actual]
    positive_actual = a > 0
    nonpositive_actual_dropped = int(np.sum(~positive_actual))
    a, f = a[positive_actual], f[positive_actual]
    finite_forecast = np.isfinite(f)
    if not np.all(finite_forecast):
        raise ValueError("forecast contains NaN or infinity where actual is positive")

    nonpositive_forecast = f <= 0
    dropped = 0
    replaced = 0
    epsilon: float | None = None
    if np.any(nonpositive_forecast):
        if forecast_zero_policy == "error":
            raise ValueError(
                f"{int(np.sum(nonpositive_forecast))} positive-actual rows have nonpositive forecasts"
            )
        if forecast_zero_policy == "drop":
            dropped = int(np.sum(nonpositive_forecast))
            keep = ~nonpositive_forecast
            a, f = a[keep], f[keep]
        else:
            epsilon = _metric_epsilon(a, epsilon_fraction)
            f = f.copy()
            replaced = int(np.sum(nonpositive_forecast))
            f[nonpositive_forecast] = epsilon

    audit: dict[str, int | float | str | None] = {
        "n_actual_nan_dropped": nan_dropped,
        "n_nonpositive_actual_dropped": nonpositive_actual_dropped,
        "n_nonpositive_forecast_dropped": dropped,
        "n_nonpositive_forecast_replaced": replaced,
        "forecast_zero_policy": forecast_zero_policy,
        "metric_epsilon": epsilon,
    }
    if a.size == 0:
        return None, audit
    return np.log(f / a), audit


def log_error(
    actual: np.ndarray,
    forecast: np.ndarray,
    *,
    forecast_zero_policy: ForecastZeroPolicy = "drop",
    epsilon_fraction: float = 1e-6,
) -> np.ndarray | None:
    """Per-month ``ln(forecast / actual)`` with an explicit zero policy."""
    errors, _ = log_error_with_audit(
        actual,
        forecast,
        forecast_zero_policy=forecast_zero_policy,
        epsilon_fraction=epsilon_fraction,
    )
    return errors



def cumulative_log_error_with_audit(
    actual: np.ndarray,
    forecast: np.ndarray,
    *,
    forecast_zero_policy: ForecastZeroPolicy = "epsilon",
    epsilon_fraction: float = 1e-6,
) -> tuple[float | None, dict[str, int | float | str | None]]:
    """Return one log error for the cumulative volume of a well/phase.

    The SPEE comparison is cross-sectional: the score is computed from a
    distribution of forecast errors across wells, not from month-to-month
    errors inside one well.  For a finite holdout this function uses
    ``log(sum(forecast) / sum(actual))`` over reported actual months.
    """
    if forecast_zero_policy not in {"drop", "epsilon", "error"}:
        raise ValueError(f"unsupported forecast_zero_policy: {forecast_zero_policy}")
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    if a.shape != f.shape:
        raise ValueError(f"actual shape {a.shape} != forecast shape {f.shape}")
    finite_actual = np.isfinite(a)
    nan_dropped = int(np.size(a) - np.sum(finite_actual))
    a, f = a[finite_actual], f[finite_actual]
    if np.any(~np.isfinite(f)):
        raise ValueError("forecast contains NaN or infinity where actual is reported")
    actual_total = float(np.sum(np.maximum(a, 0.0)))
    forecast_total = float(np.sum(np.maximum(f, 0.0)))
    audit: dict[str, int | float | str | None] = {
        "n_actual_nan_dropped": nan_dropped,
        "n_nonpositive_actual_dropped": int(actual_total <= 0),
        "n_nonpositive_forecast_dropped": 0,
        "n_nonpositive_forecast_replaced": 0,
        "forecast_zero_policy": forecast_zero_policy,
        "metric_epsilon": None,
    }
    if actual_total <= 0:
        return None, audit
    if forecast_total <= 0:
        if forecast_zero_policy == "error":
            raise ValueError("positive cumulative actual has nonpositive cumulative forecast")
        if forecast_zero_policy == "drop":
            audit["n_nonpositive_forecast_dropped"] = 1
            return None, audit
        epsilon = _metric_epsilon(np.asarray([actual_total]), epsilon_fraction)
        forecast_total = epsilon
        audit["n_nonpositive_forecast_replaced"] = 1
        audit["metric_epsilon"] = epsilon
    return float(np.log(forecast_total / actual_total)), audit


def cumulative_log_error(
    actual: np.ndarray,
    forecast: np.ndarray,
    *,
    forecast_zero_policy: ForecastZeroPolicy = "epsilon",
    epsilon_fraction: float = 1e-6,
) -> float | None:
    """One cumulative-volume log error for a well/phase holdout."""
    value, _ = cumulative_log_error_with_audit(
        actual,
        forecast,
        forecast_zero_policy=forecast_zero_policy,
        epsilon_fraction=epsilon_fraction,
    )
    return value

def decompose_log_errors(
    errors: np.ndarray,
    *,
    audit: dict[str, int | float | str | None] | None = None,
) -> SPEEDecomposition | None:
    """Decompose an already-pooled log-error vector into score components."""
    le = np.asarray(errors, dtype=float)
    le = le[np.isfinite(le)]
    if le.size < 2:
        return None
    median_le = float(np.median(le))
    stdev = float(np.std(le))
    bias_component = (2.0 / 3.0) * abs(median_le)
    spread_component = (1.0 / 3.0) * stdev
    score = bias_component + spread_component
    audit = audit or {}
    if score > 0:
        bias_share = bias_component / score
        spread_share = spread_component / score
    else:
        bias_share = 0.0
        spread_share = 0.0
    return SPEEDecomposition(
        median_log_error=median_le,
        stdev_log_error=stdev,
        bias_component=bias_component,
        spread_component=spread_component,
        score=score,
        bias_share=bias_share,
        spread_share=spread_share,
        n_log_errors=int(le.size),
        n_actual_nan_dropped=int(audit.get("n_actual_nan_dropped", 0) or 0),
        n_nonpositive_actual_dropped=int(audit.get("n_nonpositive_actual_dropped", 0) or 0),
        n_nonpositive_forecast_dropped=int(audit.get("n_nonpositive_forecast_dropped", 0) or 0),
        n_nonpositive_forecast_replaced=int(audit.get("n_nonpositive_forecast_replaced", 0) or 0),
        forecast_zero_policy=str(audit.get("forecast_zero_policy", "drop")),
        metric_epsilon=(
            float(audit["metric_epsilon"])
            if audit.get("metric_epsilon") is not None
            else None
        ),
    )


def spee_decomposition(
    actual: np.ndarray,
    forecast: np.ndarray,
    *,
    forecast_zero_policy: ForecastZeroPolicy = "drop",
    epsilon_fraction: float = 1e-6,
) -> SPEEDecomposition | None:
    """Return a diagnostic decomposition for one aligned value vector.

    This helper computes per-element log errors. It is not, by itself, the
    competition portfolio score when rows are monthly observations. For the
    bake-off harness, first compute one cumulative log error per well/phase
    with :func:`cumulative_log_error`, then call :func:`decompose_log_errors`.
    """
    errors, audit = log_error_with_audit(
        actual,
        forecast,
        forecast_zero_policy=forecast_zero_policy,
        epsilon_fraction=epsilon_fraction,
    )
    if errors is None:
        return None
    return decompose_log_errors(errors, audit=audit)


def spee_score(
    actual: np.ndarray,
    forecast: np.ndarray,
    *,
    forecast_zero_policy: ForecastZeroPolicy = "drop",
    epsilon_fraction: float = 1e-6,
) -> float | None:
    """Return the score for one already-aligned value vector.

    Do not pass raw monthly rows from multiple wells to this convenience
    function. Competition validation must aggregate each well/phase holdout
    to one cumulative log error before cross-sectional scoring.
    """
    decomposition = spee_decomposition(
        actual,
        forecast,
        forecast_zero_policy=forecast_zero_policy,
        epsilon_fraction=epsilon_fraction,
    )
    return decomposition.score if decomposition is not None else None


ALL_METRICS = {
    "mape": mape,
    "bias": bias,
    "spee_score": spee_score,
}
