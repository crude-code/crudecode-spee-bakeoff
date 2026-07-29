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
    cohort_weight: float
    uptime: float
    review_score: int
    flags: list[str]


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


def _terminal_monthly_log_decline(annual_effective: float) -> float:
    return -log(max(1e-12, 1.0 - annual_effective)) / 12.0


def enforce_terminal_decline(values: np.ndarray, annual_effective: float = 0.06) -> np.ndarray:
    """Vectorized floor on decline magnitude for any empirical curve family.

    Once the raw month-over-month log decline falls below the configured
    terminal decline, the remainder follows that terminal exponential.  The
    output is finite, non-negative and non-increasing.
    """
    y = np.asarray(values, dtype=float).copy()
    if y.size == 0:
        return y
    y[~np.isfinite(y)] = 0.0
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


def _fit_family_once(
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
        return enforce_terminal_decline(raw, config.terminal_decline_annual)

    return FittedCurve(name=name, forecast=forecast, fitted_history=capacity * uptime, uptime=uptime)


def _fallback_forecast(q: np.ndarray, n_future: int, annual_decline: float) -> np.ndarray:
    arr = np.asarray(q, dtype=float)
    positive = arr[np.isfinite(arr) & (arr > 0)]
    if positive.size == 0:
        return np.zeros(n_future)
    level = float(np.median(positive[-min(3, len(positive)) :]))
    d = _terminal_monthly_log_decline(annual_decline)
    return level * np.exp(-d * np.arange(1, n_future + 1))


def _score(actual: np.ndarray, forecast: np.ndarray) -> float:
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    valid = np.isfinite(a) & np.isfinite(f)
    if not np.any(valid):
        return 9.0
    a, f = a[valid], np.maximum(f[valid], 0.0)
    positive = (a > 0) & (f > 0)
    if positive.sum() >= 2:
        le = np.log(f[positive] / a[positive])
        log_component = (2.0 / 3.0) * abs(float(np.median(le))) + (1.0 / 3.0) * float(np.std(le))
    else:
        log_component = 1.0
    scale = float(np.median(np.abs(a[a > 0]))) if np.any(a > 0) else 1.0
    relative_component = float(np.mean(np.abs(f - a)) / max(scale, 1.0))
    zero_component = float(np.mean(((a <= 0) & (f > 0.05 * max(scale, 1.0))).astype(float)))
    return 0.65 * log_component + 0.25 * relative_component + 0.10 * zero_component


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
            if np.sum(np.isfinite(actual)) < max(2, h // 2):
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
            if np.sum(np.isfinite(actual)) < max(2, h // 2):
                continue
            try:
                pred = arps_hyperbolic_bounded_b(arr[:cutoff].copy())(h)
            except (RuntimeError, ValueError, FloatingPointError):
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
        forecast = _fallback_forecast(arr, n_future, config.terminal_decline_annual)
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
    if low or streak:
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
            forecast = enforce_terminal_decline(forecast, config.terminal_decline_annual)
            winner_name += "+recovery_status_blend"
            flags.append("recovery_status_hypotheses")

    bt = scored[0][0] if scored else None
    legacy_bt = _legacy_backtest_score(arr, config)

    # Safety ensemble selected by visible-history hindcasts.  This prevents
    # the richer model board from throwing away a strong bounded-Arps answer
    # on wells where the simpler model is demonstrably more stable.
    try:
        legacy_fc = arps_hyperbolic_bounded_b(arr)(n_future)
        if streak >= 2:
            smart_weight = 0.80
        elif low:
            # A single depressed but non-zero cutoff report is usually a
            # partial-month/allocation/operational artifact.  Preserve the
            # bounded-Arps capacity path rather than allowing a transient dip
            # to depress the entire forecast.  This safety rule takes priority
            # over noisy short internal hindcasts.
            smart_weight = 0.0
        elif legacy_bt is not None and bt is not None and legacy_bt < 0.92 * bt:
            smart_weight = 0.05
        elif legacy_bt is not None and bt is not None and bt < 0.85 * legacy_bt:
            smart_weight = 0.65
        elif positive_n < config.cohort_thin_months:
            smart_weight = 0.35
        else:
            # Use the richer family only when visible-history hindcasts earn
            # the right to deviate.  Otherwise the bounded-Arps anchor remains
            # the deterministic default.
            smart_weight = 0.0
        forecast = smart_weight * forecast + (1.0 - smart_weight) * legacy_fc
        forecast = enforce_terminal_decline(forecast, config.terminal_decline_annual)
        winner_name += "+legacy_hindcast_ensemble"
    except (RuntimeError, ValueError, FloatingPointError):
        pass

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
        cohort_weight=0.0,
        uptime=round(float(uptime), 6),
        review_score=min(100, review),
        flags=flags,
    )
    return np.maximum(forecast, 0.0), diag


class CohortLibrary:
    """Point-in-time cohort shapes and ratio-slope priors built from inputs."""

    def __init__(self, wells: list[WellSeries], metadata: dict[str, dict[str, str]], config: SmartCastConfig):
        self.config = config
        self.profiles: dict[tuple[str, str], tuple[np.ndarray, int]] = {}
        self.ratio_slopes: dict[tuple[str, str], tuple[float, int]] = {}
        self._build(wells, metadata)

    @staticmethod
    def _group(well_id: str, metadata: dict[str, dict[str, str]]) -> str:
        basin = metadata.get(well_id, {}).get("basin", "").strip().lower()
        return basin or "__global__"

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
            for age in ages:
                values = profile_values[(group, phase, age)]
                if len(values) >= max(2, self.config.cohort_min_wells // 2):
                    profile[age] = float(np.median(values))
            # fill internal holes, then force non-increasing decline shape
            valid = np.flatnonzero(np.isfinite(profile))
            if valid.size >= 3:
                profile = np.interp(np.arange(len(profile)), valid, profile[valid])
                profile = np.minimum.accumulate(np.maximum(profile, 0.0))
                self.profiles[(group, phase)] = (profile, len(ids))
        for key, values in ratio_values.items():
            self.ratio_slopes[key] = (float(np.median(values)), len(values))

    def forecast(self, well: WellSeries, phase: str, n_future: int, metadata: dict[str, dict[str, str]]) -> tuple[np.ndarray | None, float]:
        q = np.asarray(getattr(well, phase), dtype=float)
        valid = np.flatnonzero(np.isfinite(q) & (q > 0))
        if valid.size < 2:
            return None, 0.0
        peak = int(valid[np.argmax(q[valid])])
        age = len(q) - 1 - peak
        group = self._group(well.well_id, metadata)
        entry = self.profiles.get((group, phase))
        if entry is None or entry[1] < self.config.cohort_min_wells:
            entry = self.profiles.get(("__global__", phase))
        if entry is None or entry[1] < self.config.cohort_min_wells:
            return None, 0.0
        profile, _ = entry
        ages = age + np.arange(1, n_future + 1)
        raw = np.empty(n_future, dtype=float)
        in_range = ages < len(profile)
        raw[in_range] = profile[ages[in_range]]
        if np.any(~in_range):
            anchor = float(profile[-1])
            d = _terminal_monthly_log_decline(self.config.terminal_decline_annual)
            raw[~in_range] = anchor * np.exp(-d * (ages[~in_range] - (len(profile) - 1)))
        recent_idx = valid[-min(3, len(valid)) :]
        expected = np.interp(recent_idx - peak, np.arange(len(profile)), profile, left=profile[0], right=profile[-1])
        scale = float(np.median(q[recent_idx] / np.maximum(expected, MIN_POSITIVE)))
        fc = enforce_terminal_decline(np.maximum(raw * scale, 0.0), self.config.terminal_decline_annual)
        # Shrink only genuinely short visible histories.  Using months since
        # the observed peak is unstable under allocation noise because one
        # late spike can make a mature well look artificially "thin".
        visible_positive = int(valid.size)
        weight = float(np.clip((self.config.cohort_thin_months - visible_positive) / self.config.cohort_thin_months, 0.0, 0.65))
        return fc, weight

    def ratio_prior(self, well_id: str, relation: str, metadata: dict[str, dict[str, str]]) -> float:
        group = self._group(well_id, metadata)
        value = self.ratio_slopes.get((group, relation))
        if value is None or value[1] < self.config.cohort_min_wells:
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
        self.cohorts = CohortLibrary(wells, self.metadata, self.config)
        self._cache: dict[tuple[str, int], dict[str, np.ndarray]] = {}
        self.diagnostics: dict[tuple[str, int], WellDiagnostic] = {}

    def __call__(self, split: Split, phase: str) -> np.ndarray | None:
        key = (split.well_id, len(split.holdout_months))
        if key not in self._cache:
            self._cache[key] = self._forecast_well(split.train, len(split.holdout_months))
        return self._cache[key].get(phase)

    def _forecast_well(self, well: WellSeries, horizon: int) -> dict[str, np.ndarray]:
        independent: dict[str, np.ndarray] = {}
        phase_diags: dict[str, PhaseDiagnostic] = {}
        for phase in PHASES:
            q = np.asarray(getattr(well, phase), dtype=float)
            if not np.any(np.isfinite(q)):
                continue
            fc, diag = smartcast_phase(q, horizon, self.config)
            cohort_fc, weight = self.cohorts.forecast(well, phase, horizon, self.metadata)
            # A cohort prior must not overwrite a clear cutoff disruption.
            # Preserve the per-well capacity/status hypotheses until the newest
            # report is understood.
            if "terminal_low" in diag.flags or any(f.startswith("terminal_zero_streak_") for f in diag.flags):
                weight = 0.0
                diag.flags.append("cohort_suppressed_terminal_disruption")
            if cohort_fc is not None and weight > 0:
                fc = (1.0 - weight) * fc + weight * cohort_fc
                diag.cohort_weight = round(weight, 6)
                diag.flags.append("cohort_shrinkage")
                diag.model_name += "+cohort"
            diag.well_id = well.well_id
            diag.phase = phase
            independent[phase] = np.maximum(fc, 0.0)
            phase_diags[phase] = diag

        primary = _primary_phase(well)
        primary_fc = independent.get(primary)
        if primary_fc is not None:
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
                "history_months", "positive_months", "backtest_score", "cohort_weight",
                "uptime", "flags",
            ]
            with review_csv.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
