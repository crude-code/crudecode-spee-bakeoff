"""Deterministic synthetic validation for cutoff-time DCA failure modes.

The stress board is a regression and experimental-design tool, not evidence
about the hidden committee data.  Every provider is evaluated on identical
well/phase units.  Reports include pooled SPEE-style score decomposition,
paired deltas, scenario results, seed win consistency, and bootstrap intervals.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np

from forecast_benchmark.arps import arps_hyperbolic_bounded_b
from forecast_benchmark.data import PHASES, WellSeries
from forecast_benchmark.metrics import cumulative_log_error_with_audit, decompose_log_errors
from forecast_benchmark.providers import ForecastProvider, from_model_fn
from forecast_benchmark.smartcast import SmartCastProvider
from forecast_benchmark.split import Split


@dataclass(frozen=True)
class StressCase:
    train: WellSeries
    actual: dict[str, np.ndarray]
    scenario: str
    metadata: dict[str, str]


@dataclass(frozen=True)
class _ScoredUnit:
    unit_id: str
    well_id: str
    phase: str
    scenario: str
    errors_by_arm: dict[str, np.ndarray]
    series_score_by_arm: dict[str, float]
    audit_by_arm: dict[str, dict]


def _months(n: int, start_idx: int = 2020 * 12) -> list[str]:
    return [f"{idx // 12:04d}-{idx % 12 + 1:02d}-01" for idx in range(start_idx, start_idx + n)]


def _base_curve(kind: str, n: int, qi: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n, dtype=float)
    if kind == "hyperbolic":
        b = rng.uniform(0.45, 1.2)
        di = rng.uniform(0.045, 0.16)
        return qi / np.power(1 + b * di * t, 1 / b)
    if kind == "exponential":
        return qi * np.exp(-rng.uniform(0.025, 0.09) * t)
    if kind == "sepd":
        tau = rng.uniform(10, 30)
        shape = rng.uniform(0.5, 0.95)
        return qi * np.exp(-np.power(t / tau, shape))
    early = qi / np.power(1 + 1.1 * 0.18 * t, 1 / 1.1)
    switch = min(18, n // 2)
    late_t = np.maximum(t - switch, 0)
    late = early[switch] * np.exp(-0.035 * late_t)
    return np.where(t <= switch, early, late)


def make_stress_cases(seed: int, n_wells: int = 180, horizon: int = 12) -> list[StressCase]:
    rng = np.random.default_rng(seed)
    scenarios = (
        "clean",
        "downtime",
        "temporary_cutoff_dip",
        "shut_in",
        "reactivation",
        "allocation_noise",
        "thin_history",
        "missing_months",
    )
    kinds = ("hyperbolic", "exponential", "sepd", "two_regime")
    cases: list[StressCase] = []
    for i in range(n_wells):
        scenario = scenarios[i % len(scenarios)]
        history = int(rng.integers(7, 49)) if scenario == "thin_history" else int(rng.integers(18, 49))
        total = history + horizon
        kind = kinds[i % len(kinds)]
        primary = "oil" if i % 3 else "gas"
        qi_primary = rng.uniform(2500, 12000)
        primary_rate = _base_curve(kind, total, qi_primary, rng)

        t = np.arange(total, dtype=float)
        if primary == "oil":
            gor0 = rng.uniform(0.8, 4.0)
            gor = gor0 * np.exp(rng.uniform(0.005, 0.035) * t)
            oil = primary_rate.copy()
            gas = oil * gor
            wor = rng.uniform(0.1, 1.0) * np.exp(rng.uniform(-0.015, 0.025) * t)
            water = oil * wor
        else:
            cgr0 = rng.uniform(0.005, 0.08)
            cgr = cgr0 * np.exp(rng.uniform(-0.030, 0.020) * t)
            gas = primary_rate.copy()
            oil = gas * cgr
            wgr = rng.uniform(0.003, 0.08) * np.exp(rng.uniform(-0.015, 0.025) * t)
            water = gas * wgr

        curves = {"oil": oil, "gas": gas, "water": water}
        common_noise = np.exp(rng.normal(0, 0.07, total))
        for phase in PHASES:
            curves[phase] = np.maximum(
                curves[phase] * common_noise * np.exp(rng.normal(0, 0.04, total)),
                0.0,
            )

        if scenario == "downtime":
            choices = np.arange(5, max(6, history - 2))
            if choices.size:
                for idx in rng.choice(choices, size=min(3, max(1, history // 12)), replace=False):
                    factor = rng.uniform(0.0, 0.35)
                    for phase in PHASES:
                        curves[phase][idx] *= factor
        elif scenario == "temporary_cutoff_dip":
            for phase in PHASES:
                curves[phase][history - 2 : history] *= rng.uniform(0.0, 0.25)
        elif scenario == "shut_in":
            for phase in PHASES:
                curves[phase][history - 3 :] = 0.0
        elif scenario == "reactivation":
            lo = max(4, history - 10)
            hi = max(lo + 1, history - 5)
            for phase in PHASES:
                curves[phase][lo:hi] = 0.0
                curves[phase][hi:] *= 0.88
        elif scenario == "allocation_noise":
            for phase in PHASES:
                curves[phase][:history] *= np.exp(rng.normal(0, 0.22, history))
        elif scenario == "missing_months":
            choices = np.arange(3, max(4, history - 2))
            if choices.size:
                for idx in rng.choice(choices, size=min(2, max(1, history // 15)), replace=False):
                    for phase in PHASES:
                        curves[phase][idx] = np.nan

        wid = f"seed{seed:04d}_w{i:04d}"
        train = WellSeries(
            well_id=wid,
            months=_months(history),
            oil=np.asarray(curves["oil"][:history], float),
            gas=np.asarray(curves["gas"][:history], float),
            water=np.asarray(curves["water"][:history], float),
        )
        actual = {p: np.asarray(curves[p][history : history + horizon], float) for p in PHASES}
        meta = {
            "well_id": wid,
            "pad_id": f"pad-{i // 6}",
            "basin": f"basin-{i % 4}",
            "lateral_length_ft": str(5000 + (i % 6) * 1000),
        }
        cases.append(StressCase(train=train, actual=actual, scenario=scenario, metadata=meta))
    return cases


def _merge_audits(audits: list[dict]) -> dict:
    numeric_keys = (
        "n_actual_nan_dropped",
        "n_nonpositive_actual_dropped",
        "n_nonpositive_forecast_dropped",
        "n_nonpositive_forecast_replaced",
    )
    merged = {key: int(sum(int(a.get(key, 0) or 0) for a in audits)) for key in numeric_keys}
    merged["forecast_zero_policy"] = audits[0].get("forecast_zero_policy", "epsilon") if audits else "epsilon"
    eps = [a.get("metric_epsilon") for a in audits if a.get("metric_epsilon") is not None]
    merged["metric_epsilon"] = float(np.median(eps)) if eps else None
    return merged


def _arm_summary(units: list[_ScoredUnit], arm: str) -> dict:
    eligible = [u for u in units if arm in u.errors_by_arm]
    if not eligible:
        return {"n_series": 0, "score": None}
    pooled = np.concatenate([u.errors_by_arm[arm] for u in eligible])
    decomposition = decompose_log_errors(
        pooled,
        audit=_merge_audits([u.audit_by_arm[arm] for u in eligible]),
    )
    series_scores = np.asarray([u.series_score_by_arm[arm] for u in eligible], dtype=float)
    result = decomposition.to_dict() if decomposition is not None else {"score": None}
    result.update(
        {
            "n_series": int(len(eligible)),
            "median_series_absolute_log_error": float(np.median(series_scores)),
            "mean_series_absolute_log_error": float(np.mean(series_scores)),
        }
    )
    return result


def _relative_delta(challenger: float, baseline: float) -> float | None:
    if not np.isfinite(challenger) or not np.isfinite(baseline) or baseline == 0:
        return None
    return float((challenger - baseline) / baseline)


def _pooled_score_for_sample(units: list[_ScoredUnit], arm: str, indices: np.ndarray) -> float:
    errors = np.concatenate([units[int(i)].errors_by_arm[arm] for i in indices])
    decomposition = decompose_log_errors(errors)
    return decomposition.score if decomposition is not None else float("nan")


def _paired_comparison(
    units: list[_ScoredUnit],
    challenger: str,
    baseline: str,
    *,
    bootstrap_reps: int,
    bootstrap_seed: int,
) -> dict:
    paired = [u for u in units if challenger in u.errors_by_arm and baseline in u.errors_by_arm]
    if not paired:
        return {"n_paired_series": 0}
    challenger_summary = _arm_summary(paired, challenger)
    baseline_summary = _arm_summary(paired, baseline)
    c_score = float(challenger_summary["score"])
    b_score = float(baseline_summary["score"])
    series_deltas = np.asarray(
        [u.series_score_by_arm[challenger] - u.series_score_by_arm[baseline] for u in paired],
        dtype=float,
    )

    ci: list[float] | None = None
    if bootstrap_reps > 0 and len(paired) >= 2:
        rng = np.random.default_rng(bootstrap_seed)
        deltas = np.empty(bootstrap_reps, dtype=float)
        for i in range(bootstrap_reps):
            sample = rng.integers(0, len(paired), size=len(paired))
            deltas[i] = _pooled_score_for_sample(paired, challenger, sample) - _pooled_score_for_sample(
                paired, baseline, sample
            )
        finite = deltas[np.isfinite(deltas)]
        if finite.size:
            ci = [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]

    return {
        "challenger": challenger,
        "baseline": baseline,
        "n_paired_series": len(paired),
        "pooled_score_delta": c_score - b_score,
        "pooled_relative_delta": _relative_delta(c_score, b_score),
        "paired_series_absolute_log_error_delta_mean": float(np.mean(series_deltas)),
        "paired_series_absolute_log_error_delta_median": float(np.median(series_deltas)),
        "paired_series_win_rate": float(np.mean(series_deltas < 0)),
        "paired_bootstrap_score_delta_ci95": ci,
        "bootstrap_unit": "well-phase series",
        "bootstrap_reps": bootstrap_reps,
    }


def _score_cases(
    cases: list[StressCase],
    providers: dict[str, ForecastProvider],
    *,
    horizon: int,
    forecast_zero_policy: str,
) -> list[_ScoredUnit]:
    units: list[_ScoredUnit] = []
    for case in cases:
        split = Split(
            well_id=case.train.well_id,
            train=case.train,
            holdout_months=_months(horizon, 2020 * 12 + len(case.train.months)),
            holdout_actuals=case.actual,
        )
        for phase in PHASES:
            actual = case.actual[phase]
            errors_by_arm: dict[str, np.ndarray] = {}
            series_score_by_arm: dict[str, float] = {}
            audit_by_arm: dict[str, dict] = {}
            for name, provider in providers.items():
                fc = provider(split, phase)
                if fc is None:
                    continue
                cumulative_error, audit = cumulative_log_error_with_audit(
                    actual,
                    fc,
                    forecast_zero_policy=forecast_zero_policy,
                )
                if cumulative_error is None or not np.isfinite(cumulative_error):
                    continue
                errors_by_arm[name] = np.asarray([cumulative_error], dtype=float)
                # One well/phase contributes one cross-sectional log error.
                # Absolute log error is the paired per-series diagnostic; the
                # SPEE-style score is computed only after pooling all units.
                series_score_by_arm[name] = abs(float(cumulative_error))
                audit_by_arm[name] = audit
            if errors_by_arm:
                units.append(
                    _ScoredUnit(
                        unit_id=f"{case.train.well_id}/{phase}",
                        well_id=case.train.well_id,
                        phase=phase,
                        scenario=case.scenario,
                        errors_by_arm=errors_by_arm,
                        series_score_by_arm=series_score_by_arm,
                        audit_by_arm=audit_by_arm,
                    )
                )
    return units


def run_stress_board(
    seed: int,
    n_wells: int = 180,
    horizon: int = 12,
    *,
    bootstrap_reps: int = 500,
    forecast_zero_policy: str = "epsilon",
    extra_provider_factories: dict[str, Callable[[list[WellSeries], dict[str, dict[str, str]]], ForecastProvider]] | None = None,
) -> dict:
    """Run one paired seed and return decision-grade decomposition evidence."""
    cases = make_stress_cases(seed, n_wells=n_wells, horizon=horizon)
    wells = [c.train for c in cases]
    metadata = {c.train.well_id: c.metadata for c in cases}
    providers: dict[str, ForecastProvider] = {
        "smartcast_v1": SmartCastProvider(wells, metadata),
        "legacy_arps": from_model_fn(arps_hyperbolic_bounded_b),
    }
    for name, factory in (extra_provider_factories or {}).items():
        if name in providers:
            raise ValueError(f"duplicate provider name: {name}")
        providers[name] = factory(wells, metadata)

    started = perf_counter()
    units = _score_cases(
        cases,
        providers,
        horizon=horizon,
        forecast_zero_policy=forecast_zero_policy,
    )
    elapsed = perf_counter() - started
    arm_summary = {name: _arm_summary(units, name) for name in providers}
    scenario_summary: dict[str, dict] = {}
    for scenario in sorted({u.scenario for u in units}):
        subset = [u for u in units if u.scenario == scenario]
        scenario_summary[scenario] = {
            "arms": {name: _arm_summary(subset, name) for name in providers},
            "smartcast_vs_legacy": _paired_comparison(
                subset,
                "smartcast_v1",
                "legacy_arps",
                bootstrap_reps=max(100, bootstrap_reps // 2),
                bootstrap_seed=seed * 1009 + sum(ord(c) for c in scenario),
            ),
        }

    comparison = _paired_comparison(
        units,
        "smartcast_v1",
        "legacy_arps",
        bootstrap_reps=bootstrap_reps,
        bootstrap_seed=seed * 7919 + 17,
    )
    smart_score = arm_summary["smartcast_v1"].get("score")
    legacy_score = arm_summary["legacy_arps"].get("score")
    return {
        "seed": seed,
        "n_wells": n_wells,
        "n_scored_series": len(units),
        "horizon": horizon,
        "elapsed_seconds": round(elapsed, 3),
        "forecast_zero_policy": forecast_zero_policy,
        "arms": arm_summary,
        "comparison": comparison,
        "scenario_summary": scenario_summary,
        "smartcast_beats_legacy": bool(
            smart_score is not None and legacy_score is not None and smart_score < legacy_score
        ),
        "disclaimer": "Synthetic regression evidence only; not evidence about the hidden committee dataset.",
    }


def _bootstrap_seed_mean_ci(values: np.ndarray, reps: int, seed: int) -> list[float] | None:
    finite = values[np.isfinite(values)]
    if finite.size < 2 or reps <= 0:
        return None
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=float)
    for i in range(reps):
        means[i] = float(np.mean(rng.choice(finite, size=len(finite), replace=True)))
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def aggregate_stress_runs(
    results: list[dict],
    *,
    min_seed_win_rate: float = 0.70,
    max_clean_relative_regression: float = 0.05,
    max_worst_scenario_relative_regression: float = 0.25,
    bootstrap_reps: int = 5000,
) -> dict:
    """Aggregate paired seeds and apply explicit release gates."""
    if not results:
        raise ValueError("at least one stress result is required")
    deltas = np.asarray([r["comparison"]["pooled_score_delta"] for r in results], dtype=float)
    relatives = np.asarray([r["comparison"]["pooled_relative_delta"] for r in results], dtype=float)
    seed_win_rate = float(np.mean(deltas < 0))

    scenario_names = sorted({s for r in results for s in r["scenario_summary"]})
    scenarios: dict[str, dict] = {}
    for scenario in scenario_names:
        rows = [r["scenario_summary"].get(scenario, {}).get("smartcast_vs_legacy", {}) for r in results]
        sd = np.asarray([row.get("pooled_score_delta", np.nan) for row in rows], dtype=float)
        sr = np.asarray([row.get("pooled_relative_delta", np.nan) for row in rows], dtype=float)
        finite_d = sd[np.isfinite(sd)]
        finite_r = sr[np.isfinite(sr)]
        scenarios[scenario] = {
            "n_seeds": int(len(finite_d)),
            "mean_score_delta": float(np.mean(finite_d)) if finite_d.size else None,
            "median_score_delta": float(np.median(finite_d)) if finite_d.size else None,
            "mean_relative_delta": float(np.mean(finite_r)) if finite_r.size else None,
            "stdev_relative_delta": float(np.std(finite_r)) if finite_r.size else None,
            "seeds_won": int(np.sum(finite_d < 0)),
            "seed_win_rate": float(np.mean(finite_d < 0)) if finite_d.size else None,
            "mean_score_delta_ci95": _bootstrap_seed_mean_ci(
                finite_d,
                bootstrap_reps,
                100003 + sum(ord(c) for c in scenario),
            ),
        }

    clean_relative = scenarios.get("clean", {}).get("mean_relative_delta")
    finite_scenario_relatives = [
        v["mean_relative_delta"] for v in scenarios.values() if v.get("mean_relative_delta") is not None
    ]
    worst_relative = max(finite_scenario_relatives) if finite_scenario_relatives else None
    gates = {
        "overall_mean_score_delta_negative": bool(float(np.mean(deltas)) < 0),
        "seed_win_rate": bool(seed_win_rate >= min_seed_win_rate),
        "clean_non_regression": bool(
            clean_relative is not None and clean_relative <= max_clean_relative_regression
        ),
        "worst_scenario_regression": bool(
            worst_relative is not None and worst_relative <= max_worst_scenario_relative_regression
        ),
    }
    return {
        "n_seeds": len(results),
        "n_wells_per_seed": sorted({int(r["n_wells"]) for r in results}),
        "horizons": sorted({int(r["horizon"]) for r in results}),
        "mean_score_delta": float(np.mean(deltas)),
        "median_score_delta": float(np.median(deltas)),
        "mean_relative_delta": float(np.mean(relatives)),
        "stdev_relative_delta": float(np.std(relatives)),
        "seeds_won": int(np.sum(deltas < 0)),
        "seeds_lost": int(np.sum(deltas > 0)),
        "seed_win_rate": seed_win_rate,
        "mean_score_delta_ci95": _bootstrap_seed_mean_ci(deltas, bootstrap_reps, 20260728),
        "scenarios": scenarios,
        "thresholds": {
            "min_seed_win_rate": min_seed_win_rate,
            "max_clean_relative_regression": max_clean_relative_regression,
            "max_worst_scenario_relative_regression": max_worst_scenario_relative_regression,
        },
        "gates": gates,
        "champion_gate_pass": gates["overall_mean_score_delta_negative"] and gates["seed_win_rate"],
        "strict_nonregression_gate_pass": all(gates.values()),
        "disclaimer": "Synthetic multi-seed evidence only; real-well validation remains required.",
    }
