"""The deterministic arm: bounded-b hyperbolic Arps decline fit on a
trailing window, with an uptime haircut.

One model, deliberately. This repo exists to answer whether an LLM
forecaster beats a competent deterministic curve fit, so the deterministic
side is a single named contender that gets improved in place rather than a
field of variants that makes "which one won" ambiguous.

The uptime treatment mirrors the LLM skill's v3 procedure (skill/SKILL.md,
"Capacity is not what gets reported") mechanically, because the LLM's
entire margin appeared when it got that procedure and fairness demands the
Arps side gets the same arithmetic: fit the well's *capacity* trend with
downtime months struck, then commit expected reported volumes = capacity
curve x measured historical uptime. A single-pass SSE fit through downtime
dips does neither half — the dips drag the level down and distort the
slope, and the forecast then assumes that depressed level persists.

A model is a plain function: (production_array) -> forecast_fn(n_future_months)
-> np.ndarray. No analogs, no cohort borrowing, no crudecode dependency.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

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

# Experimental variants below are intentionally not on the main board. They
# let maintainers answer Bill's concrete question — "is there improvement to
# be had in the Python function?" — on the larger well set without changing
# what the official 1v1 arm means.
DEFAULT_B_GRID = tuple(round(0.3 + 0.05 * i, 2) for i in range(21))  # 0.30..1.30
B_CAP_1P0_GRID = tuple(round(0.3 + 0.05 * i, 2) for i in range(15))  # 0.30..1.00

# Experimental routed arm: hold out the latest months inside the training
# history and let that internal hindcast pick the deterministic variant.
# This imports the commercial-tool idea of auto-selecting the current usable
# segment, but keeps it auditable: selection uses only pre-cutoff history.
INNER_BACKTEST_HORIZON = 6
INNER_BACKTEST_MIN_TRAIN = 18



def arps_hyperbolic_bounded_b(
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
    return _arps_with_options(q, b_grid=b_grid)


def arps_window_24(q: np.ndarray, *, b_grid: tuple[float, ...] | None = None) -> ForecastFn:
    """Experimental arm: same method as official Arps, but a 24-month fit window.

    Tests whether the official 36-month window is still too stale on wells
    whose current regime moved quickly. Not used by the default benchmark.
    """
    return _arps_with_options(q, b_grid=b_grid, fit_window=24)


def arps_window_48(q: np.ndarray, *, b_grid: tuple[float, ...] | None = None) -> ForecastFn:
    """Experimental arm: same method as official Arps, but a 48-month fit window.

    Tests whether the official 36-month window is too sensitive to late
    allocation noise. Not used by the default benchmark.
    """
    return _arps_with_options(q, b_grid=b_grid, fit_window=48)


def arps_all_post_peak(q: np.ndarray, *, b_grid: tuple[float, ...] | None = None) -> ForecastFn:
    """Experimental arm: full post-peak history, no trailing-window cutoff.

    This is mostly a regression check against the old whole-life tendency:
    if it wins on the expanded board, the 36-month assumption deserves a
    real challenge. Not used by the default benchmark.
    """
    return _arps_with_options(q, b_grid=b_grid, fit_window=None)


def arps_no_uptime_haircut(q: np.ndarray, *, b_grid: tuple[float, ...] | None = None) -> ForecastFn:
    """Experimental arm: fit capacity but do not multiply by measured uptime.

    This isolates whether the uptime haircut is helping or overpricing
    non-recurring downtime on the expanded board. Not used by the default
    benchmark.
    """
    return _arps_with_options(q, b_grid=b_grid, apply_uptime=False)


def arps_recent_low_guard(q: np.ndarray, *, b_grid: tuple[float, ...] | None = None) -> ForecastFn:
    """Experimental arm: preserve a terminal two-month crash as current status.

    The official arm strikes low months as downtime, which is right for
    interior downtime but can over-forecast a well that is actually shut in
    at the cutoff. This guard falls back to the last-three-month reported
    tail if the last two fit-window months are both below the first-pass
    trend threshold. It will lose when a frac-hit/downtime trough recovers;
    that tradeoff is exactly what the larger board should measure.
    """
    return _arps_with_options(q, b_grid=b_grid, recent_low_guard=True)


def arps_b_cap_1p0(q: np.ndarray, *, b_grid: tuple[float, ...] | None = None) -> ForecastFn:
    """Experimental arm: cap b at 1.0 instead of 1.3.

    This tests the research-backed consistency argument: if the trailing
    window is meant to sit inside the current/boundary-dominated regime,
    then b > 1 is the part of the grid most likely to buy unearned tail
    flatness from noise. Not used by the default benchmark.
    """
    return _arps_with_options(q, b_grid=b_grid or B_CAP_1P0_GRID)


def arps_lag_skip_2(q: np.ndarray, *, b_grid: tuple[float, ...] | None = None) -> ForecastFn:
    """Experimental arm: exclude the last 2 reported months from fit and uptime.

    Public/state production data for the freshest months may be revised
    upward later. This variant keeps the forecast origin at the actual
    cutoff but fits/measures uptime on the last reliable months before that
    lag window. Not used by the default benchmark.
    """
    return _arps_with_options(q, b_grid=b_grid, reporting_lag_skip=2)


def arps_lag_skip_3(q: np.ndarray, *, b_grid: tuple[float, ...] | None = None) -> ForecastFn:
    """Experimental arm: exclude the last 3 reported months from fit and uptime.

    Same question as arps_lag_skip_2, but with a wider reporting-lag
    lookback. Not used by the default benchmark.
    """
    return _arps_with_options(q, b_grid=b_grid, reporting_lag_skip=3)


def arps_lag_skip_2_b_cap_1p0(
    q: np.ndarray, *, b_grid: tuple[float, ...] | None = None
) -> ForecastFn:
    """Experimental arm: combine the two research-backed a-priori changes.

    Excludes the freshest 2 reported months from fit/uptime and caps b at
    1.0. This is the highest-signal candidate variant to test on the larger
    board, but it is not promoted without holdout evidence.
    """
    return _arps_with_options(
        q, b_grid=b_grid or B_CAP_1P0_GRID, reporting_lag_skip=2
    )



def arps_inner_backtest_routed(
    q: np.ndarray, *, b_grid: tuple[float, ...] | None = None
) -> ForecastFn:
    """Experimental arm: choose the deterministic variant by an inner hindcast.

    This is the most direct way to make the Python side smarter without
    letting it peek at the benchmark holdout. Commercial DCA tools often
    auto-select usable data subsets; university/operator hindcasts test a
    forecast on withheld history before trusting it. This arm combines those
    ideas: reserve the latest INNER_BACKTEST_HORIZON reported months inside
    the training history, score several a-priori deterministic variants on
    that inner holdout, then refit the winning variant on the full history.

    It is intentionally not the official board arm yet. It answers a narrow
    question on the larger board: does pre-cutoff self-validation pick a
    better deterministic setup than the fixed 36-month arm?
    """
    q_clean = q[~np.isnan(q)]
    horizon = min(INNER_BACKTEST_HORIZON, max(0, len(q_clean) // 5))
    if horizon < 3 or len(q_clean) - horizon < INNER_BACKTEST_MIN_TRAIN:
        return arps_hyperbolic_bounded_b(q, b_grid=b_grid)

    train = q_clean[:-horizon]
    actual = q_clean[-horizon:]

    best_name = "arps_bounded_b"
    best_fn: Callable[[np.ndarray], ForecastFn] = arps_hyperbolic_bounded_b
    best_score = float("inf")

    for name, model_fn in _inner_backtest_candidate_arms().items():
        try:
            forecast = model_fn(train)(horizon)
        except (RuntimeError, ValueError, FloatingPointError):
            continue
        score = _inner_validation_score(actual, forecast)
        if score is None:
            continue
        # Strict < preserves deterministic tie-break order: official arm first.
        if score < best_score:
            best_name, best_fn, best_score = name, model_fn, score

    _ = best_name  # kept named for debugger/trace readability without changing the interface
    return best_fn(q)


def python_arm_experiments() -> dict[str, Callable[[np.ndarray], ForecastFn]]:
    """Named deterministic variants for scripts/run_python_arm_experiments.py.

    The first entry is the official board arm. The rest are controlled
    experiments, not new default contenders.
    """
    return {
        "arps_bounded_b": arps_hyperbolic_bounded_b,
        "arps_inner_backtest_routed": arps_inner_backtest_routed,
        "arps_lag_skip_2": arps_lag_skip_2,
        "arps_lag_skip_3": arps_lag_skip_3,
        "arps_b_cap_1p0": arps_b_cap_1p0,
        "arps_lag_skip_2_b_cap_1p0": arps_lag_skip_2_b_cap_1p0,
        "arps_window_24": arps_window_24,
        "arps_window_48": arps_window_48,
        "arps_all_post_peak": arps_all_post_peak,
        "arps_no_uptime_haircut": arps_no_uptime_haircut,
        "arps_recent_low_guard": arps_recent_low_guard,
    }



def _inner_backtest_candidate_arms() -> dict[str, Callable[[np.ndarray], ForecastFn]]:
    """Candidate set for arps_inner_backtest_routed.

    Keep this small and a-priori. The router should choose among defensible
    deterministic assumptions, not brute-force every knob until in-sample
    history is overfit.
    """
    return {
        "arps_bounded_b": arps_hyperbolic_bounded_b,
        "arps_lag_skip_2": arps_lag_skip_2,
        "arps_lag_skip_3": arps_lag_skip_3,
        "arps_b_cap_1p0": arps_b_cap_1p0,
        "arps_lag_skip_2_b_cap_1p0": arps_lag_skip_2_b_cap_1p0,
        "arps_window_24": arps_window_24,
        "arps_window_48": arps_window_48,
        "arps_no_uptime_haircut": arps_no_uptime_haircut,
        "arps_recent_low_guard": arps_recent_low_guard,
    }


def _inner_validation_score(actual: np.ndarray, forecast: np.ndarray) -> float | None:
    """SPEE-style inner score, with sane fallbacks for zeros/sparse phases."""
    if len(actual) != len(forecast):
        return None
    mask = ~np.isnan(actual) & ~np.isnan(forecast)
    if not np.any(mask):
        return None
    a = actual[mask].astype(float)
    f = forecast[mask].astype(float)

    positive = (a > 0) & (f > 0)
    if positive.sum() >= 2:
        err = np.log(f[positive] / a[positive])
        return float((2.0 / 3.0) * abs(np.median(err)) + (1.0 / 3.0) * np.std(err))

    nonzero = a != 0
    if np.any(nonzero):
        return float(np.mean(np.abs((f[nonzero] - a[nonzero]) / a[nonzero])))

    denom = float(np.mean(np.abs(a)) + 1.0)
    return float(np.mean(np.abs(f - a)) / denom)


def _arps_with_options(
    q: np.ndarray,
    *,
    b_grid: tuple[float, ...] | None = None,
    fit_window: int | None = FIT_WINDOW,
    apply_uptime: bool = True,
    recent_low_guard: bool = False,
    reporting_lag_skip: int = 0,
) -> ForecastFn:
    grid = b_grid or DEFAULT_B_GRID
    mask = ~np.isnan(q)
    q_clean = q[mask]
    if len(q_clean) < 3:
        return _flat_tail(q)

    peak_idx = int(np.argmax(q_clean))
    q_window = q_clean[peak_idx:]
    if fit_window is not None:
        q_window = q_window[-fit_window:]

    skip = max(0, int(reporting_lag_skip))
    if skip and len(q_window) > skip + 2:
        q_fit = q_window[:-skip]
    else:
        q_fit = q_window
        skip = 0
    forecast_origin = len(q_window)
    t_fit = np.arange(len(q_fit), dtype=float)

    first = _best_fit(t_fit, q_fit, grid)
    if first is None:
        return _flat_tail(q)

    params = first
    pred = _hyperbolic_q(t_fit, *first)

    if recent_low_guard and _has_terminal_low_streak(q_fit, pred):
        return _flat_tail(q)

    keep = (q_fit >= DOWNTIME_FRAC * pred) & (q_fit > 0)
    if keep.sum() >= 3:
        refit = _best_fit(t_fit[keep], q_fit[keep], grid, sigma=q_fit[keep])
        if refit is not None:
            params = refit

    qi, di, b = params
    capacity = _hyperbolic_q(t_fit, qi, di, b)
    uptime = 1.0
    if apply_uptime:
        tail = slice(max(0, len(q_fit) - UPTIME_WINDOW), len(q_fit))
        valid = capacity[tail] > 0
        if np.any(valid):
            ratio = q_fit[tail][valid] / capacity[tail][valid]
            uptime = float(np.clip(np.mean(ratio), 0.0, 1.0))

    n_history = forecast_origin

    def _forecast(n_future: int) -> np.ndarray:
        t_future = np.arange(n_history, n_history + n_future, dtype=float)
        return _hyperbolic_q(t_future, qi, di, b) * uptime

    return _forecast


def trailing_capacity_ratios(
    q: np.ndarray, *, months_back: int = 6, b_grid: tuple[float, ...] | None = None
) -> list[dict[str, float | int]]:
    """Return reported/capacity ratios for the freshest months in the fit window.

    Used by scripts/diagnose_reporting_lag.py to test whether the newest
    reported months are systematically depressed versus fitted capacity. A
    broad cross-well dip in offsets 1-3 is evidence of reporting lag leaking
    into both the fit and uptime haircut, not proof of recurring downtime.
    """
    grid = b_grid or DEFAULT_B_GRID
    mask = ~np.isnan(q)
    q_clean = q[mask]
    if len(q_clean) < 3:
        return []

    peak_idx = int(np.argmax(q_clean))
    q_window = q_clean[peak_idx:][-FIT_WINDOW:]
    if len(q_window) < 3:
        return []
    t_fit = np.arange(len(q_window), dtype=float)

    first = _best_fit(t_fit, q_window, grid)
    if first is None:
        return []
    pred = _hyperbolic_q(t_fit, *first)
    keep = (q_window >= DOWNTIME_FRAC * pred) & (q_window > 0)
    params = first
    if keep.sum() >= 3:
        refit = _best_fit(t_fit[keep], q_window[keep], grid, sigma=q_window[keep])
        if refit is not None:
            params = refit

    capacity = _hyperbolic_q(t_fit, *params)
    out: list[dict[str, float | int]] = []
    for offset in range(1, min(months_back, len(q_window)) + 1):
        reported = float(q_window[-offset])
        cap = float(capacity[-offset])
        if cap <= 0 or not np.isfinite(reported) or not np.isfinite(cap):
            continue
        out.append({
            "months_before_cutoff": offset,
            "reported": reported,
            "capacity": cap,
            "ratio": reported / cap,
        })
    return out


def _has_terminal_low_streak(q_fit: np.ndarray, trend: np.ndarray) -> bool:
    """True when the last two reported months look like current-status lows.

    This is deliberately only used by the experimental recent-low variant.
    It is not safe enough to promote blindly: it helps true shut-ins and
    hurts temporary dips that recover.
    """
    if len(q_fit) < 2:
        return False
    recent = q_fit[-2:]
    recent_trend = trend[-2:]
    return bool(np.all((recent <= 0) | (recent < DOWNTIME_FRAC * recent_trend)))


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
    """Fast fixed-b hyperbolic fit using the linearized Arps relation.

    For fixed ``b``::

        q(t)^(-b) = qi^(-b) * (1 + b * di * t)

    so the transformed response is linear in time.  This avoids nonlinear
    optimizer stalls on zero-heavy or pathological histories and makes the
    1,000-well deadline predictable.  When ``sigma`` is supplied, the
    transformed regression uses relative-response weights, preserving the
    intent of the capacity refit.
    """
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    valid = np.isfinite(t) & np.isfinite(q) & (q > 0)
    if np.sum(valid) < 2:
        return None
    x = t[valid]
    y = q[valid]
    z = np.power(np.maximum(y, 1e-12), -b)
    X = np.column_stack([np.ones(len(x)), x])
    try:
        if sigma is not None:
            # Relative error in transformed response.  Cap weights to avoid a
            # single high-rate point dominating an otherwise stable tail.
            w = 1.0 / np.maximum(z, 1e-12)
            w = np.clip(w / np.median(w), 0.1, 10.0)
            beta, *_ = np.linalg.lstsq(X * w[:, None], z * w, rcond=None)
        else:
            beta, *_ = np.linalg.lstsq(X, z, rcond=None)
    except np.linalg.LinAlgError:
        return None
    a, c = float(beta[0]), float(beta[1])
    if not np.isfinite(a) or not np.isfinite(c) or a <= 0:
        return None
    c = max(c, 0.0)
    qi = a ** (-1.0 / b)
    di = c / (a * b)
    if not np.isfinite(qi) or not np.isfinite(di) or qi < 0 or di < 0 or di > 1.0:
        return None
    return float(qi), float(di)
