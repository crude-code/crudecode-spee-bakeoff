"""Competition-grade empirical rate-time forecasting for the SPEE bake-off.

SmartCast is deliberately deterministic and auditable.  It does not use the
hidden answer period, an LLM, or a black-box ML model.  It combines:

* fast empirical rate-time families (modified Arps, exponential, SEPD),
* rolling-origin selection using only visible history,
* explicit recovery/current-status hypotheses for cutoff disruptions,
* cohort-shape shrinkage for short histories, and
* ratio-consistent secondary-phase forecasts.

All long-horizon curves are constrained to a 6% effective annual terminal
decline, matching the common convention documented in the 2024 SPEE bake-off.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, log
from pathlib import Path
import csv
import json
from typing import Callable

import numpy as np

from forecast_benchmark.arps import arps_hyperbolic_bounded_b
from forecast_benchmark.data import PHASES, WellSeries
from forecast_benchmark.split import Split

MIN_POSITIVE = 1e-9

# Selection/ablation profiles share the same empirical fits. Caching only the
# fit-relevant settings avoids recomputing thousands of identical least-squares
# and SciPy anchor fits while keeping routing decisions fully independent.
_FIT_CACHE: dict[tuple[object, ...], "FittedCurve | None"] = {}
_ANCHOR_FORECAST_CACHE: dict[tuple[object, ...], np.ndarray] = {}
_CACHE_MAX_ITEMS = 50_000


class _NoAnchor(Exception):
    """Raised internally to route the no-anchor ablation through the existing
    except-branch, so the anchor-off path takes exactly the same code path as an
    anchor that failed to fit. Keeps one control flow rather than two."""


@dataclass(frozen=True)
class SmartCastConfig:
    fit_windows: tuple[int, ...] = (12, 18, 24, 36, 48)
    b_grid: tuple[float, ...] = (0.35, 0.50, 0.65, 0.80, 0.95, 1.10, 1.25)
    sepd_n_grid: tuple[float, ...] = (0.45, 0.60, 0.75, 0.90, 1.05)
    inner_horizons: tuple[int, ...] = (3, 6)
    max_backtest_origins: int = 2
    min_fit_points: int = 5
    downtime_fraction: float = 0.65
    uptime_window: int = 24
    terminal_decline_annual: float = 0.06
    cohort_thin_months: int = 18
    cohort_min_wells: int = 5
    max_ratio_monthly_log_slope: float = 0.025
    ratio_independent_blend: float = 0.93

    # ------------------------------------------------------------------ FIXES
    # anchor_impl — WHICH Arps the safety anchor uses.
    #   'scipy'      original SciPy bounded-b (arps_scipy_v1). Real-data champion:
    #                0.1209 major-phase SPEE on the 299-well dev board.
    #   'linearized' the fast variant in arps.py. Scored 0.2098 with spread
    #                0.5870 (vs 0.3595) and cum-major spread 0.8496. This is what
    #                v1.0.0/v1.1.0 silently used, which is the confirmed root cause
    #                of SmartCast's real-data regression.
    #   'none'       no anchor blending at all (ablation C2).
    # Default is 'scipy': anchoring to the weaker implementation was a defect,
    # and pointing at the champion is a fix, not a tuning choice.
    anchor_impl: str = "scipy"

    # default_smart_weight — weight given to SmartCast's own forecast on the
    # DEFAULT branch (no disruption, not thin-history, neither hindcast clearly
    # better). v1.1.0 hardcoded 0.0, i.e. the default was "return the anchor and
    # discard every candidate/cohort/ratio layer". Measured on real Delaware
    # well-phases, this branch can hand the anchor 100% of the forecast.
    # Left at 0.0 so behaviour is UNCHANGED unless deliberately ablated — this is
    # a model choice that must be earned on the dev board, not silently flipped.
    default_smart_weight: float = 0.0
    depressed_cutoff_smart_weight: float = 0.0

    # Expose every formerly-hardcoded routing weight so the real development
    # board can isolate each decision without editing production code. Defaults
    # reproduce v1.2 exactly; conservative profiles live in profiles.py.
    terminal_streak_smart_weight: float = 0.80
    anchor_better_smart_weight: float = 0.05
    candidate_better_smart_weight: float = 0.65
    thin_history_smart_weight: float = 0.35
    anchor_better_ratio: float = 0.92
    candidate_better_ratio: float = 0.85

    # Sparse cohorts must not silently fall back to a mixed-play global shape in
    # conservative profiles. v1.2 behavior remains the default for reproducible
    # ablation, while the competition profile disables this fallback.
    allow_global_cohort_fallback: bool = True
    cohort_full_support_wells: int = 20
    require_target_group_metadata: bool = False

    # Cohort risk controls. Defaults preserve v1.3.1 behavior; the v1.4 gated
    # profile opts into each control explicitly.
    cohort_max_weight: float = 0.65
    cohort_blend_space: str = "linear"  # "linear" or "log"
    cohort_support_horizon: int = 12
    cohort_min_forecast_age_support: int = 0
    cohort_max_log_mad: float | None = None
    cohort_similarity_window: int = 12
    cohort_max_history_log_error: float | None = None
    cohort_max_monthly_log_divergence: float | None = None
    cohort_max_cumulative_log_divergence: float | None = None
    cohort_endpoint_ratio_low: float | None = None
    cohort_endpoint_ratio_high: float | None = None
    cohort_endpoint_log_volatility_max: float | None = None

    # Optional evidence-backed routing restrictions.  These use only fields
    # available at forecast time: target primary phase and supplied play/basin
    # metadata.  ``None`` preserves the historical all-target behavior.
    cohort_allowed_primary_phases: tuple[str, ...] | None = None
    cohort_allowed_groups: tuple[str, ...] | None = None
    cohort_allowed_forecast_phases: tuple[str, ...] | None = None

    # Ablation switches for the C0..C6 matrix. All default True == current
    # behaviour, so enabling them changes nothing until a run turns one off.
    use_anchor: bool = True
    use_recovery: bool = True
    use_cohort: bool = True
    use_ratio_coupling: bool = True
    use_terminal_decline: bool = True


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    window: int
    skip_recent: int = 0
    apply_uptime: bool = True


@dataclass
class PhaseDiagnostic:
    well_id: str
    phase: str
    model_name: str
    history_months: int
    positive_months: int
    backtest_score: float | None
    anchor_backtest_score: float | None
    smart_weight: float
    routing_reason: str
    cohort_weight: float
    uptime: float
    review_score: int
    flags: list[str]
    cohort_support: int = 0
    cohort_log_mad: float | None = None
    cohort_similarity_error: float | None = None
    cohort_cumulative_divergence: float | None = None
    cohort_gate_reason: str = ""


@dataclass
class WellDiagnostic:
    well_id: str
    primary_phase: str
    phases: list[PhaseDiagnostic]


@dataclass(frozen=True)
class FittedCurve:
    name: str
    forecast: Callable[[int], np.ndarray]
    fitted_history: np.ndarray
    uptime: float


@dataclass(frozen=True)
class CohortProfile:
    values: np.ndarray
    support_n: int
    age_counts: np.ndarray
    age_log_mad: np.ndarray


def _anchor_impl(config: "SmartCastConfig"):
    """Resolve the safety-anchor implementation from config.

    Single source of truth so the anchor and the anchor's own hindcast score can
    never disagree about which model they mean — they did in v1.1.0.
    """
    if not config.use_anchor or config.anchor_impl == "none":
        raise _NoAnchor
    if config.anchor_impl == "scipy":
        from forecast_benchmark.arps_scipy_v1 import arps_bounded_scipy_v1
        return arps_bounded_scipy_v1
    if config.anchor_impl == "linearized":
        return arps_hyperbolic_bounded_b
    raise ValueError(f"unknown anchor_impl: {config.anchor_impl!r}")


def _terminal_monthly_log_decline(annual_effective: float) -> float:
    return -log(max(1e-12, 1.0 - annual_effective)) / 12.0


def _uses_anchor_base_without_candidates(config: "SmartCastConfig") -> bool:
    """Return True when no candidate/recovery branch can receive weight.

    Cohort-only profiles still need an anchor forecast, but they do not need the
    expensive candidate board or rolling-origin router.  This keeps accuracy
    identical while making the proven shrinkage path operationally viable.
    """
    return (
        config.use_anchor
        and config.anchor_impl != "none"
        and not config.use_recovery
        and config.default_smart_weight == 0.0
        and config.depressed_cutoff_smart_weight == 0.0
        and config.terminal_streak_smart_weight == 0.0
        and config.anchor_better_smart_weight == 0.0
        and config.candidate_better_smart_weight == 0.0
        and config.thin_history_smart_weight == 0.0
    )


def _is_anchor_only_config(config: "SmartCastConfig") -> bool:
    """Return True when every experimental layer is disabled."""
    return (
        _uses_anchor_base_without_candidates(config)
        and not config.use_cohort
        and not config.use_ratio_coupling
    )


def _anchor_only_phase(
    q: np.ndarray, n_future: int, config: "SmartCastConfig"
) -> tuple[np.ndarray, "PhaseDiagnostic"]:
    """Fast path for the verified control profile.

    Avoid fitting the experimental candidate board merely to assign it zero
    weight. This materially reduces bake-off runtime while preserving exactly
    the same frozen SciPy anchor and terminal-decline post-processing.
    """
    arr = np.asarray(q, dtype=float)
    forecast = _finalize_forecast(_anchor_forecast(arr, n_future, config), config)
    valid = np.flatnonzero(np.isfinite(arr) & (arr > 0))
    flags: list[str] = []
    if valid.size < config.cohort_thin_months:
        flags.append("thin_history")
    if valid.size >= 4:
        tail = arr[valid[-min(6, valid.size):]]
        if np.std(np.diff(np.log(np.maximum(tail, MIN_POSITIVE)))) > 0.35:
            flags.append("volatile_tail")
    review = min(100, (35 if "thin_history" in flags else 0) + (25 if "volatile_tail" in flags else 0))
    diagnostic = PhaseDiagnostic(
        well_id="",
        phase="",
        model_name=f"arps_{config.anchor_impl}_v1_anchor",
        history_months=len(arr),
        positive_months=int(valid.size),
        backtest_score=None,
        anchor_backtest_score=None,
        smart_weight=0.0,
        routing_reason="anchor_only_fast_path",
        cohort_weight=0.0,
        uptime=1.0,
        review_score=review,
        flags=flags,
    )
    return forecast, diagnostic


def enforce_terminal_decline(values: np.ndarray, annual_effective: float = 0.06) -> np.ndarray:
    """Vectorized floor on decline magnitude for any empirical curve family.

    Once the raw month-over-month log decline falls below the configured
    terminal decline, the remainder follows that terminal exponential.  The
    output is finite, non-negative and non-increasing.
    """
    y = np.asarray(values, dtype=float).copy()
    if y.size == 0:
        return y

    # BUG FIX (carried by v1.0.0 and v1.1.0): a single non-finite value used to
    # be set to 0.0, and the np.minimum.accumulate below then propagated that
    # zero forward forever. enforce_terminal_decline([1000,900,NaN,800,700,600])
    # returned [1000,900,0,0,0,0] — one bad month silently destroyed the rest of
    # the forecast. On a 1,000-well submission that is a catastrophic miss on any
    # well whose fit produced a non-finite value, and it fails quietly.
    # Interpolate across interior non-finite values instead; edge-extrapolate at
    # the ends; only fall back to zeros if nothing finite exists at all.
    finite = np.isfinite(y)
    if not finite.any():
        return np.zeros_like(y)
    if not finite.all():
        idx = np.arange(y.size)
        y = np.interp(idx, idx[finite], y[finite])

    y = np.maximum(y, 0.0)
    y = np.minimum.accumulate(y)
    if y.size < 2 or y[0] <= 0:
        return y
    dmin = _terminal_monthly_log_decline(annual_effective)
    prev = np.maximum(y[:-1], MIN_POSITIVE)
    nxt = np.maximum(y[1:], MIN_POSITIVE)
    declines = np.log(prev / nxt)
    idx = np.flatnonzero(declines < dmin)
    if idx.size:
        switch = int(idx[0])
        tail_n = y.size - switch - 1
        y[switch + 1 :] = y[switch] * np.exp(-dmin * np.arange(1, tail_n + 1))
    return y



def _sanitize_forecast(values: np.ndarray) -> np.ndarray:
    """Return a finite, non-negative forecast without imposing decline shape."""
    y = np.asarray(values, dtype=float).copy()
    if y.size == 0:
        return y
    finite = np.isfinite(y)
    if not finite.any():
        return np.zeros_like(y)
    if not finite.all():
        idx = np.arange(y.size)
        y = np.interp(idx, idx[finite], y[finite])
    return np.maximum(y, 0.0)


def _finalize_forecast(values: np.ndarray, config: "SmartCastConfig") -> np.ndarray:
    """Apply the production terminal-decline rule only when enabled.

    v1.2 exposed ``use_terminal_decline`` but ignored it at every call site,
    making the advertised ablation invalid. This helper is the single source of
    truth for all candidate, anchor, recovery, fallback, and cohort forecasts.
    """
    clean = _sanitize_forecast(values)
    if not config.use_terminal_decline:
        return clean
    return enforce_terminal_decline(clean, config.terminal_decline_annual)

def _calendar_positive(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(q, dtype=float)
    idx = np.flatnonzero(np.isfinite(arr) & (arr > 0))
    return idx.astype(float), arr[idx]


def _fit_linear(x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float] | None:
    if len(x) < 2 or len(y) != len(x):
        return None
    X = np.column_stack([np.ones(len(x)), x])
    try:
        if weights is not None:
            w = np.sqrt(np.asarray(weights, dtype=float))
            beta, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
        else:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(beta)):
        return None
    return float(beta[0]), float(beta[1])


def _select_window(q: np.ndarray, window: int, skip_recent: int) -> tuple[np.ndarray, int]:
    arr = np.asarray(q, dtype=float)
    end = max(0, len(arr) - max(0, skip_recent))
    if end == 0:
        return arr[:0].copy(), len(arr)
    visible = arr[:end]
    finite = np.flatnonzero(np.isfinite(visible))
    if finite.size:
        peak = int(finite[np.nanargmax(visible[finite])])
    else:
        peak = 0
    start = max(peak, end - window)
    return visible[start:end].copy(), len(arr) - start


def _fit_family_once_uncached(
    q: np.ndarray,
    spec: CandidateSpec,
    config: SmartCastConfig,
    *,
    strike_downtime: bool = True,
) -> FittedCurve | None:
    window, origin = _select_window(q, spec.window, spec.skip_recent)
    t, y = _calendar_positive(window)
    if len(y) < config.min_fit_points:
        return None

    def fit_on(tfit: np.ndarray, yfit: np.ndarray) -> tuple[Callable[[np.ndarray], np.ndarray], str] | None:
        if spec.family == "exponential":
            out = _fit_linear(tfit, np.log(np.maximum(yfit, MIN_POSITIVE)))
            if out is None:
                return None
            intercept, slope = out
            slope = min(slope, -1e-6)
            return (lambda tt: np.exp(intercept + slope * tt)), spec.name

        if spec.family == "hyperbolic":
            best: tuple[float, Callable[[np.ndarray], np.ndarray]] | None = None
            for b in config.b_grid:
                z = np.power(np.maximum(yfit, MIN_POSITIVE), -b)
                out = _fit_linear(tfit, z)
                if out is None:
                    continue
                a, c = out
                if a <= 0 or c < 0:
                    continue
                qi = a ** (-1.0 / b)
                di = c / (a * b) if a > 0 else -1.0
                if not (0 <= di <= 1.5) or not np.isfinite(qi):
                    continue
                pred = qi / np.power(1.0 + b * di * tfit, 1.0 / b)
                err = np.log(np.maximum(pred, MIN_POSITIVE) / yfit)
                score = float(np.median(np.abs(err)) + 0.25 * np.std(err))
                if best is None or score < best[0]:
                    best = (score, lambda tt, qi=qi, di=di, b=b: qi / np.power(1.0 + b * di * tt, 1.0 / b))
            if best is None:
                return None
            return best[1], spec.name

        if spec.family == "sepd":
            best = None
            for n in config.sepd_n_grid:
                x = np.power(np.maximum(tfit, 0.0), n)
                out = _fit_linear(x, np.log(np.maximum(yfit, MIN_POSITIVE)))
                if out is None:
                    continue
                intercept, slope = out
                if slope >= 0:
                    continue
                pred = np.exp(intercept + slope * x)
                err = np.log(np.maximum(pred, MIN_POSITIVE) / yfit)
                score = float(np.median(np.abs(err)) + 0.25 * np.std(err))
                if best is None or score < best[0]:
                    best = (score, lambda tt, a=intercept, s=slope, n=n: np.exp(a + s * np.power(np.maximum(tt, 0.0), n)))
            if best is None:
                return None
            return best[1], spec.name
        raise ValueError(f"unknown family: {spec.family}")

    fitted = fit_on(t, y)
    if fitted is None:
        return None
    model, name = fitted
    pred_positive = model(t)

    if strike_downtime:
        keep = y >= config.downtime_fraction * pred_positive
        if keep.sum() >= config.min_fit_points and keep.sum() < len(y):
            refit = fit_on(t[keep], y[keep])
            if refit is not None:
                model, name = refit

    all_t = np.arange(len(window), dtype=float)
    capacity = np.maximum(model(all_t), 0.0)
    uptime = 1.0
    if spec.apply_uptime:
        tail_start = max(0, len(window) - config.uptime_window)
        observed = window[tail_start:]
        cap_tail = capacity[tail_start:]
        valid = np.isfinite(observed) & (cap_tail > 0)
        if np.any(valid):
            uptime = float(np.clip(np.mean(np.clip(observed[valid] / cap_tail[valid], 0.0, 1.0)), 0.0, 1.0))

    def forecast(n_future: int) -> np.ndarray:
        t_future = np.arange(origin, origin + n_future, dtype=float)
        raw = np.maximum(model(t_future) * uptime, 0.0)
        return _finalize_forecast(raw, config)

    return FittedCurve(name=name, forecast=forecast, fitted_history=capacity * uptime, uptime=uptime)


def _fit_settings_key(config: SmartCastConfig) -> tuple[object, ...]:
    return (
        config.b_grid, config.sepd_n_grid, config.min_fit_points,
        config.downtime_fraction, config.uptime_window,
        config.terminal_decline_annual, config.use_terminal_decline,
    )


def _series_key(q: np.ndarray) -> tuple[int, bytes]:
    arr = np.ascontiguousarray(np.asarray(q, dtype=np.float64))
    return int(arr.size), arr.tobytes()


def clear_smartcast_caches() -> None:
    """Clear deterministic in-process fit caches (mainly useful in tests)."""
    _FIT_CACHE.clear()
    _ANCHOR_FORECAST_CACHE.clear()


def _fit_family_once(
    q: np.ndarray,
    spec: CandidateSpec,
    config: SmartCastConfig,
    *,
    strike_downtime: bool = True,
) -> FittedCurve | None:
    key = (*_series_key(q), spec, _fit_settings_key(config), bool(strike_downtime))
    cached = _FIT_CACHE.get(key)
    if cached is not None or key in _FIT_CACHE:
        return cached
    result = _fit_family_once_uncached(q, spec, config, strike_downtime=strike_downtime)
    if len(_FIT_CACHE) >= _CACHE_MAX_ITEMS:
        _FIT_CACHE.clear()
    _FIT_CACHE[key] = result
    return result


def _anchor_forecast(q: np.ndarray, n_future: int, config: SmartCastConfig) -> np.ndarray:
    key = (*_series_key(q), int(n_future), config.anchor_impl, bool(config.use_anchor))
    cached = _ANCHOR_FORECAST_CACHE.get(key)
    if cached is not None:
        return cached.copy()
    result = np.asarray(_anchor_impl(config)(np.asarray(q, dtype=float).copy())(n_future), dtype=float)
    if len(_ANCHOR_FORECAST_CACHE) >= _CACHE_MAX_ITEMS:
        _ANCHOR_FORECAST_CACHE.clear()
    _ANCHOR_FORECAST_CACHE[key] = result.copy()
    return result


def _fallback_forecast(q: np.ndarray, n_future: int, config: SmartCastConfig) -> np.ndarray:
    arr = np.asarray(q, dtype=float)
    positive = arr[np.isfinite(arr) & (arr > 0)]
    if positive.size == 0:
        return np.zeros(n_future)
    level = float(np.median(positive[-min(3, len(positive)) :]))
    if not config.use_terminal_decline:
        return np.full(n_future, level, dtype=float)
    d = _terminal_monthly_log_decline(config.terminal_decline_annual)
    return level * np.exp(-d * np.arange(1, n_future + 1))


def _score(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Robust visible-history routing loss with fail-closed zero treatment.

    Earlier versions excluded forecast zeros from the log component whenever the
    actual was positive. That made a candidate's worst months partially vanish
    from the evidence used to route away from the anchor. The real-board scorer
    uses an actual-driven mask and a metric-only positive floor, so the inner
    router now follows the same principle.
    """
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    if a.size != f.size or np.any(~np.isfinite(f)):
        return 9.0

    positive_actual = np.isfinite(a) & (a > 0)
    if int(positive_actual.sum()) < 2:
        return 9.0
    a_pos = a[positive_actual]
    scale = max(float(np.median(a_pos)), 1.0)
    eps = max(1e-9, 1e-4 * scale)
    f_pos = np.where(f[positive_actual] > 0, f[positive_actual], eps)
    le = np.log(f_pos / a_pos)
    log_component = (2.0 / 3.0) * abs(float(np.median(le))) + (1.0 / 3.0) * float(np.std(le))

    finite_actual = np.isfinite(a)
    relative_component = float(
        np.mean(np.abs(np.maximum(f[finite_actual], 0.0) - np.maximum(a[finite_actual], 0.0)))
        / scale
    )
    zero_component = float(
        np.mean(((a[finite_actual] <= 0) & (f[finite_actual] > 0.05 * scale)).astype(float))
    )
    return 0.70 * log_component + 0.20 * relative_component + 0.10 * zero_component


def _candidate_specs(config: SmartCastConfig) -> list[CandidateSpec]:
    # Deliberately compact.  Analytical fits make this fast enough for 1,000+
    # wells while retaining materially different hypotheses.
    return [
        CandidateSpec("mhyper_18", "hyperbolic", 18),
        CandidateSpec("mhyper_36", "hyperbolic", 36),
        CandidateSpec("mhyper_48", "hyperbolic", 48),
        CandidateSpec("exp_18", "exponential", 18),
        CandidateSpec("exp_36", "exponential", 36),
        CandidateSpec("sepd_24", "sepd", 24),
        CandidateSpec("recovery_skip3", "hyperbolic", 36, skip_recent=3),
    ]


def _rolling_backtest_score(q: np.ndarray, spec: CandidateSpec, config: SmartCastConfig) -> float | None:
    arr = np.asarray(q, dtype=float)
    scores: list[tuple[float, float]] = []
    for h in config.inner_horizons:
        latest = len(arr) - h
        origins = []
        cursor = latest
        while cursor >= max(config.min_fit_points + h, 8) and len(origins) < config.max_backtest_origins:
            origins.append(cursor)
            cursor -= h
        for rank, cutoff in enumerate(origins):
            actual = arr[cutoff : cutoff + h]
            if np.sum(np.isfinite(actual) & (actual > 0)) < max(2, h // 2):
                continue
            fitted = _fit_family_once(arr[:cutoff].copy(), spec, config)
            if fitted is None:
                continue
            pred = fitted.forecast(h)
            weight = 1.0 / (1.0 + rank)
            scores.append((_score(actual, pred), weight))
    if not scores:
        return None
    return float(sum(s * w for s, w in scores) / sum(w for _, w in scores))


def _legacy_backtest_score(q: np.ndarray, config: SmartCastConfig) -> float | None:
    arr = np.asarray(q, dtype=float)
    scores: list[tuple[float, float]] = []
    # Compare the safety anchor on the exact same rolling origins and weights
    # used by SmartCast candidates.  Different validation windows create a
    # biased router and can promote a complex family on a single lucky split.
    for h in config.inner_horizons:
        latest = len(arr) - h
        origins = []
        cursor = latest
        while cursor >= max(config.min_fit_points + h, 8) and len(origins) < config.max_backtest_origins:
            origins.append(cursor)
            cursor -= h
        for rank, cutoff in enumerate(origins):
            actual = arr[cutoff:cutoff + h]
            if np.sum(np.isfinite(actual) & (actual > 0)) < max(2, h // 2):
                continue
            try:
                # SECOND INSTANCE OF THE ANCHOR BUG: this hindcast decided the
                # `legacy_bt < 0.92*bt` branch, but it was scoring the LINEARIZED
                # Arps while the branch's purpose is to ask whether the ANCHOR
                # beats the candidate board. Scoring one model and then deferring
                # to a different one made the branch decision on the wrong
                # evidence. Use the same implementation the anchor uses.
                pred = _finalize_forecast(
                    _anchor_forecast(arr[:cutoff], h, config), config
                )
            except (_NoAnchor, RuntimeError, ValueError, FloatingPointError):
                continue
            weight = 1.0 / (1.0 + rank)
            scores.append((_score(actual, pred), weight))
    if not scores:
        return None
    return float(sum(v * w for v, w in scores) / sum(w for _, w in scores))


def _terminal_disruption(q: np.ndarray, fitted: FittedCurve | None, config: SmartCastConfig) -> tuple[int, bool]:
    arr = np.asarray(q, dtype=float)
    if len(arr) == 0:
        return 0, False
    streak = 0
    for value in arr[::-1]:
        if not np.isfinite(value):
            continue
        if value <= 0:
            streak += 1
        else:
            break
    if fitted is None or len(fitted.fitted_history) == 0:
        return streak, streak > 0
    recent = arr[-min(3, len(arr)) :]
    cap = fitted.fitted_history[-min(3, len(fitted.fitted_history)) :]
    valid = np.isfinite(recent) & (cap > 0)
    low_vs_fit = bool(np.any(valid) and np.median(recent[valid] / cap[valid]) < 0.40)

    # A fit that includes the cutoff disruption can chase the depressed points
    # and hide the event.  Add a model-independent level break check: compare
    # the newest one or two finite positive reports with the preceding six.
    finite_idx = np.flatnonzero(np.isfinite(arr))
    low_vs_prior = False
    if finite_idx.size >= 8:
        values = arr[finite_idx]
        tail = values[-2:]
        prior = values[-8:-2]
        tail_pos = tail[tail > 0]
        prior_pos = prior[prior > 0]
        if tail_pos.size and prior_pos.size >= 3:
            low_vs_prior = bool(np.median(tail_pos) < 0.40 * np.median(prior_pos))
    low = low_vs_fit or low_vs_prior
    return streak, low


def _past_recovery_evidence(q: np.ndarray) -> bool:
    arr = np.asarray(q, dtype=float)
    positive = arr[np.isfinite(arr) & (arr > 0)]
    if positive.size < 6:
        return False
    threshold = 0.35 * float(np.median(positive[-min(12, len(positive)) :]))
    for i in range(1, len(arr) - 2):
        if np.isfinite(arr[i]) and arr[i] <= threshold:
            future = arr[i + 1 : min(len(arr), i + 4)]
            if np.any(np.isfinite(future) & (future > 2.0 * max(arr[i], 1.0))):
                return True
    return False


def smartcast_phase(q: np.ndarray, n_future: int, config: SmartCastConfig | None = None) -> tuple[np.ndarray, PhaseDiagnostic]:
    """Forecast one phase and return the full selection/QC trace."""
    config = config or SmartCastConfig()
    arr = np.asarray(q, dtype=float).copy()  # candidate fits must never mutate shared input
    candidates = _candidate_specs(config)
    scored: list[tuple[float, CandidateSpec]] = []
    for spec in candidates:
        s = _rolling_backtest_score(arr, spec, config)
        if s is not None and np.isfinite(s):
            scored.append((s, spec))
    scored.sort(key=lambda x: (x[0], x[1].name))
    winner = scored[0][1] if scored else CandidateSpec("mhyper_36", "hyperbolic", 36)
    fitted = _fit_family_once(arr, winner, config)
    if fitted is None:
        forecast = _fallback_forecast(arr, n_future, config)
        winner_name = "terminal_fallback"
        uptime = 1.0
    else:
        forecast = fitted.forecast(n_future)
        winner_name = winner.name
        uptime = fitted.uptime

    flags: list[str] = []
    finite = arr[np.isfinite(arr)]
    positive_n = int(np.sum(np.isfinite(arr) & (arr > 0)))
    if positive_n < config.cohort_thin_months:
        flags.append("thin_history")
    if len(finite) >= 6:
        recent = finite[-6:]
        med = float(np.median(np.abs(recent)))
        if med > 0 and float(np.std(recent) / med) > 0.55:
            flags.append("volatile_tail")
    streak, low = _terminal_disruption(arr, fitted, config)
    if streak:
        flags.append(f"terminal_zero_streak_{streak}")
    if low:
        flags.append("terminal_low")

    # Current-status versus temporary-recovery hypotheses.  The blend is
    # explicit and deterministic; it avoids treating every cutoff zero as
    # either permanent shut-in or guaranteed recovery.
    if (low or streak) and config.use_recovery:
        recovery_spec = CandidateSpec("recovery_skip3", "hyperbolic", 36, skip_recent=min(3, max(1, streak)))
        recovery = _fit_family_once(arr, recovery_spec, config)
        if recovery is not None:
            recovery_fc = recovery.forecast(n_future)
            prior_recovery = _past_recovery_evidence(arr)
            if streak >= 3 and not prior_recovery:
                recovery_weight = 0.20
            elif streak >= 2:
                recovery_weight = 0.65 if prior_recovery else 0.45
            elif streak == 1:
                recovery_weight = 0.85 if prior_recovery else 0.72
            else:
                # A depressed but non-zero newest report is much more often a
                # partial-month/allocation/operational artifact than a permanent
                # abandonment signal.  Preserve the recovery hypothesis strongly.
                recovery_weight = 0.92 if prior_recovery else 0.86
            status_level = 0.0 if streak >= 2 else float(np.nanmedian(arr[-min(3, len(arr)) :]))
            status_fc = np.full(n_future, max(status_level, 0.0))
            forecast = recovery_weight * recovery_fc + (1.0 - recovery_weight) * status_fc
            forecast = _finalize_forecast(forecast, config)
            winner_name += "+recovery_status_blend"
            flags.append("recovery_status_hypotheses")

    bt = scored[0][0] if scored else None
    legacy_bt = _legacy_backtest_score(arr, config)

    # Safety ensemble selected by visible-history hindcasts.  This prevents
    # the richer model board from throwing away a strong bounded-Arps answer
    # on wells where the simpler model is demonstrably more stable.
    try:
        # ablation C2 (use_anchor=False / anchor_impl='none') raises _NoAnchor
        # here and falls through to the same except-branch as a failed fit.
        legacy_fc = _anchor_forecast(arr, n_future, config)
        routing_reason = "default_anchor"
        if streak >= 2:
            smart_weight = config.terminal_streak_smart_weight
            routing_reason = "terminal_streak"
        elif low:
            # A single depressed but non-zero cutoff report is usually a
            # partial-month/allocation/operational artifact.  Preserve the
            # bounded-Arps capacity path rather than allowing a transient dip
            # to depress the entire forecast.  This safety rule takes priority
            # over noisy short internal hindcasts.
            smart_weight = config.depressed_cutoff_smart_weight
            routing_reason = "depressed_cutoff"
        elif (legacy_bt is not None and bt is not None
              and legacy_bt < config.anchor_better_ratio * bt):
            smart_weight = config.anchor_better_smart_weight
            routing_reason = "anchor_hindcast_better"
        elif (legacy_bt is not None and bt is not None
              and bt < config.candidate_better_ratio * legacy_bt):
            smart_weight = config.candidate_better_smart_weight
            routing_reason = "candidate_hindcast_better"
        elif positive_n < config.cohort_thin_months:
            smart_weight = config.thin_history_smart_weight
            routing_reason = "thin_history"
        else:
            # Use the richer family only when visible-history hindcasts earn
            # the right to deviate.  Otherwise the bounded-Arps anchor remains
            # the deterministic default.
            smart_weight = config.default_smart_weight
            routing_reason = "default_anchor"
        forecast = smart_weight * forecast + (1.0 - smart_weight) * legacy_fc
        forecast = _finalize_forecast(forecast, config)
        winner_name += "+legacy_hindcast_ensemble"
    except (_NoAnchor, RuntimeError, ValueError, FloatingPointError):
        smart_weight = 1.0
        routing_reason = "no_anchor"

    review = 0
    review += 35 if "thin_history" in flags else 0
    review += 25 if "volatile_tail" in flags else 0
    review += 30 if low else 0
    review += min(30, streak * 10)
    if bt is not None:
        review += min(30, int(round(bt * 20)))
    diag = PhaseDiagnostic(
        well_id="",
        phase="",
        model_name=winner_name,
        history_months=len(arr),
        positive_months=positive_n,
        backtest_score=round(float(bt), 6) if bt is not None else None,
        anchor_backtest_score=round(float(legacy_bt), 6) if legacy_bt is not None else None,
        smart_weight=round(float(smart_weight), 6),
        routing_reason=routing_reason,
        cohort_weight=0.0,
        uptime=round(float(uptime), 6),
        review_score=min(100, review),
        flags=flags,
    )
    return np.maximum(forecast, 0.0), diag


class CohortLibrary:
    """Point-in-time cohort shapes and ratio-slope priors built from inputs.

    v1.4 retains the robust median type-well shape that produced the only
    replicated challenger signal, but also records age-specific support and
    robust log dispersion.  A large overall cohort is not evidence that enough
    peers support the specific ages being forecast.
    """

    def __init__(self, wells: list[WellSeries], metadata: dict[str, dict[str, str]], config: SmartCastConfig):
        self.config = config
        self.profiles: dict[tuple[str, str], CohortProfile] = {}
        self.ratio_slopes: dict[tuple[str, str], tuple[float, int]] = {}
        self._build(wells, metadata)

    @staticmethod
    def _group(well_id: str, metadata: dict[str, dict[str, str]]) -> str:
        raw = metadata.get(well_id, {}).get("basin", "").strip().lower()
        if not raw:
            return "__global__"
        label = raw.replace("-", "_").replace(" ", "_").replace("/", "_")
        while "__" in label:
            label = label.replace("__", "_")
        aliases = {
            "eagleford": "eagle_ford",
            "lower_eagle_ford": "eagle_ford",
            "upper_eagle_ford": "eagle_ford",
            "dj_basin": "dj",
            "denver_julesburg": "dj",
            "denver_julesburg_basin": "dj",
            "niobrara_a": "dj",
            "niobrara_b": "dj",
            "niobrara_c": "dj",
            "codell": "dj",
            "delaware_basin": "delaware",
            "permian_delaware": "delaware",
        }
        return aliases.get(label, label)

    def _build(self, wells: list[WellSeries], metadata: dict[str, dict[str, str]]) -> None:
        profile_values: dict[tuple[str, str, int], list[float]] = {}
        group_wells: dict[tuple[str, str], set[str]] = {}
        ratio_values: dict[tuple[str, str], list[float]] = {}
        for well in wells:
            group = self._group(well.well_id, metadata)
            for phase in PHASES:
                q = np.asarray(getattr(well, phase), dtype=float)
                valid = np.flatnonzero(np.isfinite(q) & (q > 0))
                if valid.size < 6:
                    continue
                peak = int(valid[np.argmax(q[valid])])
                peak_q = float(q[peak])
                for idx in valid:
                    age = int(idx - peak)
                    if age < 0 or age > 360:
                        continue
                    ratio = float(q[idx] / peak_q)
                    if 0 < ratio <= 2.0:
                        for g in (group, "__global__"):
                            profile_values.setdefault((g, phase, age), []).append(ratio)
                            group_wells.setdefault((g, phase), set()).add(well.well_id)
            primary = _primary_phase(well)
            if primary == "oil":
                pairs = (("gor", well.gas, well.oil), ("wor", well.water, well.oil))
            else:
                pairs = (("cgr", well.oil, well.gas), ("wgr", well.water, well.gas))
            for rel, num, den in pairs:
                slope = _ratio_slope(num, den, self.config.max_ratio_monthly_log_slope)
                if slope is not None:
                    for g in (group, "__global__"):
                        ratio_values.setdefault((g, rel), []).append(slope)

        for (group, phase), ids in group_wells.items():
            ages = sorted(age for (g, p, age) in profile_values if g == group and p == phase)
            if not ages:
                continue
            max_age = max(ages)
            profile = np.full(max_age + 1, np.nan)
            age_counts = np.zeros(max_age + 1, dtype=int)
            age_log_mad = np.full(max_age + 1, np.nan)
            for age in ages:
                values = np.asarray(profile_values[(group, phase, age)], dtype=float)
                values = values[np.isfinite(values) & (values > 0)]
                age_counts[age] = int(values.size)
                if values.size >= max(2, self.config.cohort_min_wells // 2):
                    # Median in log space is the same robust center as a median
                    # ratio, while making the multiplicative dispersion explicit.
                    lv = np.log(values)
                    center = float(np.median(lv))
                    profile[age] = float(np.exp(center))
                    age_log_mad[age] = float(1.4826 * np.median(np.abs(lv - center)))
            valid = np.flatnonzero(np.isfinite(profile))
            if valid.size >= 3:
                # Preserve v1.3.1 interpolation/shape behavior for the existing
                # cohort profile; only the new gate uses support/dispersion.
                profile = np.interp(np.arange(len(profile)), valid, profile[valid])
                profile = np.minimum.accumulate(np.maximum(profile, 0.0))
                self.profiles[(group, phase)] = CohortProfile(
                    values=profile,
                    support_n=len(ids),
                    age_counts=age_counts,
                    age_log_mad=age_log_mad,
                )
        for key, values in ratio_values.items():
            self.ratio_slopes[key] = (float(np.median(values)), len(values))

    def forecast(
        self,
        well: WellSeries,
        phase: str,
        n_future: int,
        metadata: dict[str, dict[str, str]],
    ) -> tuple[np.ndarray | None, float, dict[str, object]]:
        evidence: dict[str, object] = {
            "group": "",
            "support_n": 0,
            "min_age_support": 0,
            "log_mad": None,
            "similarity_error": None,
            "gate_reason": "",
        }
        q = np.asarray(getattr(well, phase), dtype=float)
        valid = np.flatnonzero(np.isfinite(q) & (q > 0))
        if valid.size < 2:
            evidence["gate_reason"] = "insufficient_target_history"
            return None, 0.0, evidence
        peak = int(valid[np.argmax(q[valid])])
        peak_q = float(q[peak])
        age = len(q) - 1 - peak
        group = self._group(well.well_id, metadata)
        evidence["group"] = group
        entry = self.profiles.get((group, phase))
        if (entry is None or entry.support_n < self.config.cohort_min_wells) and self.config.allow_global_cohort_fallback:
            entry = self.profiles.get(("__global__", phase))
            if entry is not None:
                evidence["group"] = "__global__"
        if entry is None or entry.support_n < self.config.cohort_min_wells:
            evidence["gate_reason"] = "insufficient_cohort_support"
            return None, 0.0, evidence

        profile = entry.values
        support_n = entry.support_n
        evidence["support_n"] = int(support_n)
        ages = age + np.arange(1, n_future + 1)
        raw = np.empty(n_future, dtype=float)
        in_range = ages < len(profile)
        raw[in_range] = profile[ages[in_range]]
        if np.any(~in_range):
            anchor = float(profile[-1])
            if self.config.use_terminal_decline:
                d = _terminal_monthly_log_decline(self.config.terminal_decline_annual)
                raw[~in_range] = anchor * np.exp(-d * (ages[~in_range] - (len(profile) - 1)))
            else:
                raw[~in_range] = anchor

        recent_idx = valid[-min(3, len(valid)) :]
        expected = np.interp(
            recent_idx - peak,
            np.arange(len(profile)),
            profile,
            left=profile[0],
            right=profile[-1],
        )
        scale = float(np.median(q[recent_idx] / np.maximum(expected, MIN_POSITIVE)))
        fc = _finalize_forecast(np.maximum(raw * scale, 0.0), self.config)

        visible_positive = int(valid.size)
        base_weight = float(np.clip(
            (self.config.cohort_thin_months - visible_positive) / self.config.cohort_thin_months,
            0.0,
            self.config.cohort_max_weight,
        ))
        support_span = max(1, self.config.cohort_full_support_wells - self.config.cohort_min_wells)
        support_factor = float(np.clip(
            (support_n - self.config.cohort_min_wells + 1) / support_span,
            0.0,
            1.0,
        ))
        weight = base_weight * support_factor

        check_n = min(n_future, max(1, int(self.config.cohort_support_horizon)))
        check_ages = ages[:check_n]
        counts = np.zeros(check_n, dtype=int)
        mads = np.full(check_n, np.nan)
        mask = check_ages < len(entry.age_counts)
        counts[mask] = entry.age_counts[check_ages[mask]]
        mads[mask] = entry.age_log_mad[check_ages[mask]]
        min_age_support = int(np.min(counts)) if counts.size else 0
        finite_mad = mads[np.isfinite(mads)]
        log_mad = float(np.median(finite_mad)) if finite_mad.size else None
        evidence["min_age_support"] = min_age_support
        evidence["log_mad"] = log_mad

        if self.config.cohort_min_forecast_age_support > 0:
            if min_age_support < self.config.cohort_min_forecast_age_support:
                evidence["gate_reason"] = "insufficient_age_specific_support"
                return fc, 0.0, evidence
            age_support_factor = float(np.clip(
                min_age_support / max(self.config.cohort_full_support_wells, 1),
                0.0,
                1.0,
            ))
            weight *= age_support_factor

        if self.config.cohort_max_log_mad is not None:
            if log_mad is None or log_mad > self.config.cohort_max_log_mad:
                evidence["gate_reason"] = "cohort_dispersion_too_high"
                return fc, 0.0, evidence
            # Smoothly reduce authority as peer dispersion approaches the cap.
            weight *= max(0.25, 1.0 - 0.5 * log_mad / max(self.config.cohort_max_log_mad, 1e-9))

        if valid.size >= 4:
            tail = q[valid[-min(6, valid.size) :]]
            prior_tail = q[valid[-min(4, valid.size) : -1]]
            if prior_tail.size:
                endpoint_ratio = float(tail[-1] / max(float(np.median(prior_tail)), MIN_POSITIVE))
                lo = self.config.cohort_endpoint_ratio_low
                hi = self.config.cohort_endpoint_ratio_high
                if (lo is not None and endpoint_ratio < lo) or (hi is not None and endpoint_ratio > hi):
                    evidence["gate_reason"] = "endpoint_level_anomaly"
                    return fc, 0.0, evidence
            if self.config.cohort_endpoint_log_volatility_max is not None and tail.size >= 4:
                volatility = float(np.std(np.diff(np.log(np.maximum(tail, MIN_POSITIVE)))))
                if volatility > self.config.cohort_endpoint_log_volatility_max:
                    evidence["gate_reason"] = "endpoint_volatility_anomaly"
                    return fc, 0.0, evidence

        sim_idx = valid[valid >= peak][-max(3, int(self.config.cohort_similarity_window)) :]
        sim_idx = sim_idx[(sim_idx - peak) < len(profile)]
        similarity_error = None
        if sim_idx.size >= 3:
            obs = np.log(np.maximum(q[sim_idx] / peak_q, MIN_POSITIVE))
            ref = np.log(np.maximum(profile[sim_idx - peak], MIN_POSITIVE))
            similarity_error = float(np.median(np.abs(obs - ref)))
        evidence["similarity_error"] = similarity_error
        if self.config.cohort_max_history_log_error is not None:
            if similarity_error is None or similarity_error > self.config.cohort_max_history_log_error:
                evidence["gate_reason"] = "target_not_cohort_like"
                return fc, 0.0, evidence
            weight *= max(0.25, 1.0 - 0.5 * similarity_error / max(self.config.cohort_max_history_log_error, 1e-9))

        evidence["gate_reason"] = "cohort_gate_pass" if weight > 0 else "zero_thin_history_weight"
        return fc, float(np.clip(weight, 0.0, self.config.cohort_max_weight)), evidence

    def ratio_prior(self, well_id: str, relation: str, metadata: dict[str, dict[str, str]]) -> float:
        group = self._group(well_id, metadata)
        value = self.ratio_slopes.get((group, relation))
        if (value is None or value[1] < self.config.cohort_min_wells) and self.config.allow_global_cohort_fallback:
            value = self.ratio_slopes.get(("__global__", relation))
        return float(value[0]) if value is not None else 0.0

def _primary_phase(well: WellSeries) -> str:
    oil = float(np.nansum(np.maximum(well.oil, 0.0)))
    gas_boe = float(np.nansum(np.maximum(well.gas, 0.0))) / 6.0
    return "oil" if oil >= gas_boe else "gas"


def _ratio_slope(num: np.ndarray, den: np.ndarray, cap: float) -> float | None:
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    valid = np.isfinite(num) & np.isfinite(den) & (num > 0) & (den > 0)
    idx = np.flatnonzero(valid)[-18:]
    if idx.size < 4:
        return None
    y = np.log(num[idx] / den[idx])
    x = idx.astype(float)
    fit = _fit_linear(x - x[0], y)
    if fit is None:
        return None
    return float(np.clip(fit[1], -cap, cap))


def _ratio_forecast(
    num: np.ndarray,
    den: np.ndarray,
    den_forecast: np.ndarray,
    prior_slope: float,
    config: SmartCastConfig,
) -> tuple[np.ndarray | None, list[str]]:
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    valid = np.isfinite(num) & np.isfinite(den) & (num > 0) & (den > 0)
    idx = np.flatnonzero(valid)[-18:]
    if idx.size < 3:
        return None, ["ratio_sparse"]
    ratios = num[idx] / den[idx]
    level = float(np.median(ratios[-min(6, len(ratios)) :]))
    local = _ratio_slope(num, den, config.max_ratio_monthly_log_slope)
    if local is None:
        local = prior_slope
    # Partial pooling: short/noisy histories get pulled harder to the cohort.
    shrink = float(np.clip(idx.size / 24.0, 0.20, 0.65))
    slope = shrink * local + (1.0 - shrink) * prior_slope
    slope = float(np.clip(slope, -config.max_ratio_monthly_log_slope, config.max_ratio_monthly_log_slope))
    future_ratio = level * np.exp(slope * np.arange(1, len(den_forecast) + 1))
    flags: list[str] = []
    if float(np.std(np.log(np.maximum(ratios, MIN_POSITIVE)))) > 0.30:
        flags.append("ratio_volatile")
    return np.maximum(den_forecast * future_ratio, 0.0), flags


def _cohort_route_allowed(
    well: WellSeries,
    phase: str,
    metadata: dict[str, dict[str, str]],
    config: SmartCastConfig,
) -> tuple[bool, str]:
    """Return whether the cohort layer may alter this target/phase.

    The selective route is intentionally simple and auditable.  It never uses
    provenance, producing-days, holdout information, or learned labels.
    """
    primary = _primary_phase(well)
    if config.cohort_allowed_primary_phases is not None:
        allowed = {str(v).strip().lower() for v in config.cohort_allowed_primary_phases}
        if primary.lower() not in allowed:
            return False, "selective_route_primary_phase"

    group = CohortLibrary._group(well.well_id, metadata)
    if config.cohort_allowed_groups is not None:
        allowed_groups = {str(v).strip().lower() for v in config.cohort_allowed_groups}
        if group.lower() not in allowed_groups:
            return False, "selective_route_play"

    if config.cohort_allowed_forecast_phases is not None:
        allowed_phases = {str(v).strip().lower() for v in config.cohort_allowed_forecast_phases}
        if phase.lower() not in allowed_phases:
            return False, "selective_route_forecast_phase"

    return True, "cohort_route_allowed"


class SmartCastProvider:
    """ForecastProvider-compatible, cross-well SmartCast implementation."""

    def __init__(
        self,
        wells: list[WellSeries],
        metadata: dict[str, dict[str, str]] | None = None,
        config: SmartCastConfig | None = None,
    ) -> None:
        self.wells = {w.well_id: w for w in wells}
        self.metadata = metadata or {}
        self.config = config or SmartCastConfig()
        cohort_wells = wells if (self.config.use_cohort or self.config.use_ratio_coupling) else []
        self.cohorts = CohortLibrary(cohort_wells, self.metadata, self.config)
        self._cache: dict[tuple[str, int], dict[str, np.ndarray]] = {}
        self.diagnostics: dict[tuple[str, int], WellDiagnostic] = {}

    def __call__(self, split: Split, phase: str) -> np.ndarray | None:
        key = (split.well_id, len(split.holdout_months))
        if key not in self._cache:
            self._cache[key] = self._forecast_well(split.train, len(split.holdout_months))
        return self._cache[key].get(phase)

    def _forecast_well(self, well: WellSeries, horizon: int) -> dict[str, np.ndarray]:
        if (self.config.require_target_group_metadata
                and (self.config.use_cohort or self.config.use_ratio_coupling)
                and not self.metadata.get(well.well_id, {}).get("basin", "").strip()):
            raise ValueError(
                f"missing play/basin metadata for target well {well.well_id!r}; "
                "refusing silent __global__ cohort fallback"
            )
        independent: dict[str, np.ndarray] = {}
        phase_diags: dict[str, PhaseDiagnostic] = {}
        for phase in PHASES:
            q = np.asarray(getattr(well, phase), dtype=float)
            if not np.any(np.isfinite(q)):
                continue
            cohort_allowed, cohort_route_reason = _cohort_route_allowed(
                well, phase, self.metadata, self.config
            )
            if _uses_anchor_base_without_candidates(self.config):
                fc, diag = _anchor_only_phase(q, horizon, self.config)
                if self.config.use_cohort and cohort_allowed:
                    cohort_fc, weight, cohort_evidence = self.cohorts.forecast(
                        well, phase, horizon, self.metadata
                    )
                    diag.routing_reason = "anchor_plus_cohort_fast_path"
                else:
                    cohort_fc, weight, cohort_evidence = None, 0.0, {
                        "support_n": 0,
                        "min_age_support": 0,
                        "log_mad": None,
                        "similarity_error": None,
                        "gate_reason": (cohort_route_reason if self.config.use_cohort else "anchor_only"),
                    }
            else:
                fc, diag = smartcast_phase(q, horizon, self.config)
                if self.config.use_cohort and cohort_allowed:
                    cohort_fc, weight, cohort_evidence = self.cohorts.forecast(
                        well, phase, horizon, self.metadata
                    )
                else:
                    cohort_fc, weight, cohort_evidence = None, 0.0, {
                        "support_n": 0,
                        "min_age_support": 0,
                        "log_mad": None,
                        "similarity_error": None,
                        "gate_reason": (cohort_route_reason if self.config.use_cohort else "cohort_disabled"),
                    }
            # A cohort prior must not overwrite a clear cutoff disruption.
            # Preserve the per-well capacity/status hypotheses until the newest
            # report is understood.
            if "terminal_low" in diag.flags or any(f.startswith("terminal_zero_streak_") for f in diag.flags):
                weight = 0.0
                cohort_evidence["gate_reason"] = "terminal_disruption"
                diag.flags.append("cohort_suppressed_terminal_disruption")

            cumulative_divergence = None
            if cohort_fc is not None and weight > 0 and self.config.use_cohort:
                anchor_safe = np.maximum(np.asarray(fc, dtype=float), MIN_POSITIVE)
                cohort_safe = np.maximum(np.asarray(cohort_fc, dtype=float), MIN_POSITIVE)
                check_n = min(
                    len(anchor_safe),
                    max(1, int(self.config.cohort_support_horizon)),
                )
                if self.config.cohort_max_monthly_log_divergence is not None:
                    cap = float(self.config.cohort_max_monthly_log_divergence)
                    log_ratio = np.log(cohort_safe[:check_n] / anchor_safe[:check_n])
                    if np.any(np.abs(log_ratio) > cap):
                        cohort_safe = np.clip(
                            cohort_safe,
                            anchor_safe * np.exp(-cap),
                            anchor_safe * np.exp(cap),
                        )
                        cohort_fc = cohort_safe
                        diag.flags.append("cohort_monthly_divergence_capped")

                sa = float(np.sum(anchor_safe[:check_n]))
                sc = float(np.sum(cohort_safe[:check_n]))
                if sa > 0 and sc > 0:
                    cumulative_divergence = float(abs(np.log(sc / sa)))
                if (
                    self.config.cohort_max_cumulative_log_divergence is not None
                    and cumulative_divergence is not None
                    and cumulative_divergence > self.config.cohort_max_cumulative_log_divergence
                ):
                    # Taper the adjustment rather than selecting a new model.
                    # The exact SciPy control remains the limiting forecast.
                    weight *= float(
                        self.config.cohort_max_cumulative_log_divergence
                        / max(cumulative_divergence, 1e-12)
                    )
                    diag.flags.append("cohort_cumulative_divergence_tapered")

                weight = float(np.clip(weight, 0.0, self.config.cohort_max_weight))
                if weight > 0:
                    if self.config.cohort_blend_space == "log":
                        both_zero = (np.asarray(fc) <= 0) & (np.asarray(cohort_fc) <= 0)
                        fc = np.exp(
                            (1.0 - weight) * np.log(anchor_safe)
                            + weight * np.log(np.maximum(cohort_fc, MIN_POSITIVE))
                        )
                        fc[both_zero] = 0.0
                    elif self.config.cohort_blend_space == "linear":
                        fc = (1.0 - weight) * fc + weight * cohort_fc
                    else:
                        raise ValueError(
                            f"unknown cohort_blend_space: {self.config.cohort_blend_space!r}"
                        )
                    diag.cohort_weight = round(weight, 6)
                    diag.flags.append("cohort_shrinkage")
                    diag.model_name += "+cohort"

            diag.cohort_support = int(cohort_evidence.get("min_age_support") or 0)
            log_mad = cohort_evidence.get("log_mad")
            diag.cohort_log_mad = round(float(log_mad), 6) if log_mad is not None else None
            sim = cohort_evidence.get("similarity_error")
            diag.cohort_similarity_error = round(float(sim), 6) if sim is not None else None
            diag.cohort_cumulative_divergence = (
                round(float(cumulative_divergence), 6)
                if cumulative_divergence is not None else None
            )
            diag.cohort_gate_reason = str(cohort_evidence.get("gate_reason") or "")
            if diag.cohort_gate_reason and diag.cohort_gate_reason != "cohort_gate_pass":
                diag.flags.append(f"cohort_gate:{diag.cohort_gate_reason}")
            diag.well_id = well.well_id
            diag.phase = phase
            independent[phase] = np.maximum(fc, 0.0)
            phase_diags[phase] = diag

        primary = _primary_phase(well)
        primary_fc = independent.get(primary)
        if primary_fc is not None and self.config.use_ratio_coupling:
            if primary == "oil":
                rels = (("gas", "gor", well.gas, well.oil), ("water", "wor", well.water, well.oil))
            else:
                rels = (("oil", "cgr", well.oil, well.gas), ("water", "wgr", well.water, well.gas))
            for phase, relation, num, den in rels:
                if phase not in independent:
                    continue
                # Do not force a phase-ratio relationship across an unresolved
                # cutoff disruption.  Independent phase curves are safer until
                # the newest month's allocation/operating status is known.
                primary_flags = phase_diags.get(primary).flags if primary in phase_diags else []
                secondary_flags = phase_diags[phase].flags
                if "terminal_low" in primary_flags or "terminal_low" in secondary_flags:
                    phase_diags[phase].flags.append("ratio_suppressed_terminal_disruption")
                    continue
                prior = self.cohorts.ratio_prior(well.well_id, relation, self.metadata)
                ratio_fc, flags = _ratio_forecast(num, den, primary_fc, prior, self.config)
                if ratio_fc is None:
                    phase_diags[phase].flags.extend(flags)
                    continue
                blend = 0.985 if "ratio_volatile" in flags else self.config.ratio_independent_blend
                independent[phase] = blend * independent[phase] + (1.0 - blend) * ratio_fc
                phase_diags[phase].model_name += f"+{relation}_pooled"
                phase_diags[phase].flags.extend(flags)
                if flags:
                    phase_diags[phase].review_score = min(100, phase_diags[phase].review_score + 20)

        self.diagnostics[(well.well_id, horizon)] = WellDiagnostic(
            well_id=well.well_id,
            primary_phase=primary,
            phases=[phase_diags[p] for p in PHASES if p in phase_diags],
        )
        return independent

    def write_diagnostics(self, json_path: str | Path, review_csv: str | Path | None = None) -> None:
        payload = []
        rows = []
        for key in sorted(self.diagnostics):
            wd = self.diagnostics[key]
            payload.append({
                "well_id": wd.well_id,
                "primary_phase": wd.primary_phase,
                "phases": [asdict(p) for p in wd.phases],
            })
            for p in wd.phases:
                rows.append({
                    **asdict(p),
                    "primary_phase": wd.primary_phase,
                    "flags": ";".join(p.flags),
                })
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        if review_csv is not None:
            review_csv = Path(review_csv)
            review_csv.parent.mkdir(parents=True, exist_ok=True)
            rows.sort(key=lambda r: (-int(r["review_score"]), str(r["well_id"]), str(r["phase"])))
            fields = [
                "well_id", "phase", "primary_phase", "review_score", "model_name",
                "history_months", "positive_months", "backtest_score", "anchor_backtest_score",
                "smart_weight", "routing_reason", "cohort_weight", "cohort_support",
                "cohort_log_mad", "cohort_similarity_error",
                "cohort_cumulative_divergence", "cohort_gate_reason", "uptime", "flags",
            ]
            with review_csv.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
