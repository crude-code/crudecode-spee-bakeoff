"""Original SciPy bounded-b Arps — THE REAL-DATA CHAMPION, frozen.

Ported verbatim from the historical benchmark repo so this package no longer
has to reach across repositories for it, and so the name says which
implementation it is.

WHY THIS FILE EXISTS
On the 299-well real development board this implementation scored 0.1209
major-phase SPEE. The fast linearized variant in `arps.py` scored 0.2098 with
more than twice the spread (0.5870 vs 0.3595) and, on the cumulative-major
metric, a spread of 0.8496 driven by a handful of extreme blow-ups.

SmartCast's "legacy safety anchor" was importing `arps_hyperbolic_bounded_b`
from `arps.py` — i.e. it was anchored to the WEAKER implementation — and its
default branch handed that anchor 100% of the weight. Because the default branch assigned the anchor 100% weight, a material share
of SmartCast forecasts could collapse to that weaker implementation while the
candidate, cohort, and ratio layers were bypassed.

This module is the fix's foundation: an explicitly-named, frozen champion that
SmartCast can anchor to. Do not "optimize" it. It is the control.

Its public name is `arps_bounded_scipy_v1`. The module-private helpers are
independent of `arps.py` on purpose — the two implementations must not share
state or drift into each other.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import curve_fit

ForecastFn = Callable[[int], np.ndarray]

# A month below this fraction of the fitted trend is downtime, not noise.
# Monthly allocation noise runs +/-10-15%; genuine downtime (the skill's
# example: one month down at half rate) sits far below it.
DOWNTIME_FRAC = 0.75

# Trailing reported months used to measure uptime — the skill's window.
UPTIME_WINDOW = 24

# Trailing post-peak months used for the curve fit. These wells move from
# transient linear flow early (apparent b well above 1) to boundary-dominated
# flow later, so a single-b curve forced through a well's whole life fits
# neither regime and lets year-1 flush production steer a year-6 forecast.
# 36 months is long enough to see real curvature and survive the downtime
# strike, recent enough to sit inside the current flow regime — and the
# scored horizon is 12 months, where local level and slope matter far more
# than asymptotic b. Fixed a priori, like DOWNTIME_FRAC and UPTIME_WINDOW;
# never tuned against the board. Wells with less history than this are
# unaffected — the window is simply everything they have.
FIT_WINDOW = 36


def arps_bounded_scipy_v1(
    q: np.ndarray, *, b_grid: tuple[float, ...] | None = None
) -> ForecastFn:
    """Best-fit b via grid search over a bounded range, with a downtime
    strike-and-refit and an uptime haircut, fit on a trailing window.

    Fits from the peak forward, because the pre-peak flowback ramp is not
    decline and dragging it into the fit flattens di — and only the last
    FIT_WINDOW months of that, because earlier months are a different flow
    regime (see the constant). Same shape as crudecode's fit_curve_best_b,
    reimplemented standalone so this repo has zero import dependency on
    crudecode.

    Two passes: the first fit (absolute SSE) locates the trend well enough
    to identify downtime months (below DOWNTIME_FRAC of trend, or reported
    zero); the refit without them is the capacity curve. The refit weights
    residuals by 1/reported — relative error — because even a 36-month
    window spans a several-fold rate decline, and under absolute SSE the
    high-rate early months dominate while the tail that anchors the
    forecast counts as rounding error. The first pass stays absolute: with
    relative weighting a deep downtime dip would drag the trend toward
    itself, which is exactly what the strike threshold needs it not to do.

    Uptime is mean(reported / capacity) over the trailing UPTIME_WINDOW
    months — struck months included, because they are exactly the downtime
    the factor exists to price in. NaN months are excluded everywhere: not
    reported is not downtime. The factor is capped at 1.0 — downtime only
    ever subtracts, so a ratio above 1 is fit bias, not an uptime boost.

    Falls back to a flat tail average when there's too little history to fit
    or the optimizer fails on every b — a stated fallback, never a crash and
    never a silent zero.
    """
    grid = b_grid or tuple(round(0.3 + 0.05 * i, 2) for i in range(21))  # 0.30..1.30
    mask = ~np.isnan(q)
    q_clean = q[mask]
    if len(q_clean) < 3:
        return _flat_tail(q)

    peak_idx = int(np.argmax(q_clean))
    q_fit = q_clean[peak_idx:][-FIT_WINDOW:]
    t_fit = np.arange(len(q_fit), dtype=float)

    first = _best_fit(t_fit, q_fit, grid)
    if first is None:
        return _flat_tail(q)

    params = first
    pred = _hyperbolic_q(t_fit, *first)
    keep = (q_fit >= DOWNTIME_FRAC * pred) & (q_fit > 0)
    if keep.sum() >= 3:
        refit = _best_fit(t_fit[keep], q_fit[keep], grid, sigma=q_fit[keep])
        if refit is not None:
            params = refit

    qi, di, b = params
    capacity = _hyperbolic_q(t_fit, qi, di, b)
    tail = slice(max(0, len(q_fit) - UPTIME_WINDOW), len(q_fit))
    valid = capacity[tail] > 0
    if np.any(valid):
        ratio = q_fit[tail][valid] / capacity[tail][valid]
        uptime = float(np.clip(np.mean(ratio), 0.0, 1.0))
    else:
        uptime = 1.0

    n_history = len(q_fit)

    def _forecast(n_future: int) -> np.ndarray:
        t_future = np.arange(n_history, n_history + n_future, dtype=float)
        return _hyperbolic_q(t_future, qi, di, b) * uptime

    return _forecast


def _best_fit(
    t: np.ndarray,
    q: np.ndarray,
    grid: tuple[float, ...],
    sigma: np.ndarray | None = None,
) -> tuple[float, float, float] | None:
    """Min-SSE (qi, di, b) over the b grid, or None if every b fails.

    With sigma, both the inner fit and the b selection minimize the same
    weighted objective — selecting b on a different objective than the one
    qi/di were fit under would quietly mix the two.
    """
    best_params, best_sse = None, float("inf")
    for b in grid:
        params = _fit_qi_di(t, q, b, sigma)
        if params is None:
            continue
        qi, di = params
        resid = _hyperbolic_q(t, qi, di, b) - q
        if sigma is not None:
            resid = resid / sigma
        sse = float(np.sum(resid ** 2))
        if sse < best_sse:
            best_sse, best_params = sse, (qi, di, b)
    return best_params


def _flat_tail(q: np.ndarray) -> ForecastFn:
    """Hold the average of the last 3 reported months flat. Only reachable
    as a degenerate fallback — it is not an arm on the board."""
    tail = q[~np.isnan(q)][-3:]
    level = float(np.mean(tail)) if len(tail) else 0.0

    def _forecast(n_future: int) -> np.ndarray:
        return np.full(n_future, level)

    return _forecast


def _hyperbolic_q(t: np.ndarray, qi: float, di: float, b: float) -> np.ndarray:
    if b < 1e-6:
        return qi * np.exp(-di * t)
    return qi / np.power(1.0 + b * di * t, 1.0 / b)


def _fit_qi_di(
    t: np.ndarray, q: np.ndarray, b: float, sigma: np.ndarray | None = None
) -> tuple[float, float] | None:
    if q[1:].sum() <= 0.0:
        return None

    def _model(t, qi, di):
        return _hyperbolic_q(t, qi, di, b)

    try:
        popt, _ = curve_fit(
            _model, t, q,
            p0=[float(q[0]), 0.05],
            sigma=sigma,
            bounds=([0.0, 0.0], [1e8, 1.0]),
            maxfev=5000,
        )
    except RuntimeError:
        return None
    return float(popt[0]), float(popt[1])
