"""Retrospective real-well holdout evaluation with point-in-time discipline.

This evaluates stabilized historical production by truncating every eligible
well at one common calendar cutoff.  It is not a historical reporting-snapshot
simulation unless the supplied source is itself an archived point-in-time
snapshot.  Cohorts are always built from truncated training histories only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forecast_benchmark.arps import arps_hyperbolic_bounded_b
from forecast_benchmark.data import PHASES, WellSeries
from forecast_benchmark.metrics import cumulative_log_error_with_audit, decompose_log_errors
from forecast_benchmark.providers import from_model_fn
from forecast_benchmark.smartcast import SmartCastProvider
from forecast_benchmark.spee import add_month
from forecast_benchmark.split import Split, make_splits_at_date


@dataclass(frozen=True)
class HoldoutUnit:
    well_id: str
    phase: str
    smart_errors: np.ndarray
    legacy_errors: np.ndarray
    smart_score: float
    legacy_score: float
    smart_audit: dict
    legacy_audit: dict


def choose_retrospective_cutoff(
    wells: list[WellSeries],
    *,
    horizon: int,
    stabilization_lag: int,
) -> str:
    """Choose the latest cutoff leaving holdout plus a stabilization buffer."""
    if not wells:
        raise ValueError("no wells supplied")
    if horizon <= 0 or stabilization_lag < 0:
        raise ValueError("horizon must be positive and stabilization_lag nonnegative")
    latest = max(month for well in wells for month in well.months)
    return add_month(latest, -(horizon + stabilization_lag))


def _sum_audit(units: list[HoldoutUnit], attr: str) -> dict:
    rows = [getattr(u, attr) for u in units]
    keys = (
        "n_actual_nan_dropped",
        "n_nonpositive_actual_dropped",
        "n_nonpositive_forecast_dropped",
        "n_nonpositive_forecast_replaced",
    )
    out = {key: int(sum(int(r.get(key, 0) or 0) for r in rows)) for key in keys}
    out["forecast_zero_policy"] = rows[0].get("forecast_zero_policy", "epsilon") if rows else "epsilon"
    eps = [r.get("metric_epsilon") for r in rows if r.get("metric_epsilon") is not None]
    out["metric_epsilon"] = float(np.median(eps)) if eps else None
    return out


def _summary(units: list[HoldoutUnit], arm: str) -> dict:
    if not units:
        return {"n_series": 0, "score": None}
    error_attr = "smart_errors" if arm == "smartcast_v1" else "legacy_errors"
    score_attr = "smart_score" if arm == "smartcast_v1" else "legacy_score"
    audit_attr = "smart_audit" if arm == "smartcast_v1" else "legacy_audit"
    errors = np.concatenate([getattr(u, error_attr) for u in units])
    decomp = decompose_log_errors(errors, audit=_sum_audit(units, audit_attr))
    scores = np.asarray([getattr(u, score_attr) for u in units], dtype=float)
    result = decomp.to_dict() if decomp is not None else {"score": None}
    result.update(
        {
            "n_series": len(units),
            "median_series_absolute_log_error": float(np.median(scores)),
            "mean_series_absolute_log_error": float(np.mean(scores)),
        }
    )
    return result


def _bootstrap_delta(units: list[HoldoutUnit], reps: int, seed: int) -> list[float] | None:
    if len(units) < 2 or reps <= 0:
        return None
    rng = np.random.default_rng(seed)
    deltas = np.empty(reps, dtype=float)
    for i in range(reps):
        sample = rng.integers(0, len(units), size=len(units))
        chosen = [units[int(j)] for j in sample]
        smart = _summary(chosen, "smartcast_v1")["score"]
        legacy = _summary(chosen, "legacy_arps")["score"]
        deltas[i] = float(smart - legacy)
    return [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))]


def run_retrospective_holdout(
    wells: list[WellSeries],
    metadata: dict[str, dict[str, str]],
    *,
    cutoff_date: str,
    horizon: int,
    min_history: int = 12,
    forecast_zero_policy: str = "epsilon",
    bootstrap_reps: int = 2000,
) -> dict:
    """Score SmartCast and legacy Arps on identical real-well holdouts."""
    splits, skipped = make_splits_at_date(wells, cutoff_date=cutoff_date, horizon=horizon)
    splits = [s for s in splits if len(s.train.months) >= min_history]
    eligible_ids = {s.well_id for s in splits}
    training_wells = [s.train for s in splits]
    training_metadata = {wid: metadata.get(wid, {}) for wid in eligible_ids}
    smart = SmartCastProvider(training_wells, training_metadata)
    legacy = from_model_fn(arps_hyperbolic_bounded_b)

    units: list[HoldoutUnit] = []
    failures: list[dict[str, str]] = []
    for split in splits:
        for phase in PHASES:
            if not split.train.phase_available(phase):
                continue
            actual = split.holdout_actuals[phase]
            try:
                smart_fc = smart(split, phase)
                legacy_fc = legacy(split, phase)
                if smart_fc is None or legacy_fc is None:
                    raise ValueError("provider returned no forecast")
                smart_error, smart_audit = cumulative_log_error_with_audit(
                    actual, smart_fc, forecast_zero_policy=forecast_zero_policy
                )
                legacy_error, legacy_audit = cumulative_log_error_with_audit(
                    actual, legacy_fc, forecast_zero_policy=forecast_zero_policy
                )
                if smart_error is None or legacy_error is None:
                    continue
                units.append(
                    HoldoutUnit(
                        well_id=split.well_id,
                        phase=phase,
                        smart_errors=np.asarray([smart_error], dtype=float),
                        legacy_errors=np.asarray([legacy_error], dtype=float),
                        smart_score=abs(float(smart_error)),
                        legacy_score=abs(float(legacy_error)),
                        smart_audit=smart_audit,
                        legacy_audit=legacy_audit,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - report every failed unit
                failures.append({"well_id": split.well_id, "phase": phase, "reason": str(exc)})

    smart_summary = _summary(units, "smartcast_v1")
    legacy_summary = _summary(units, "legacy_arps")
    series_deltas = np.asarray([u.smart_score - u.legacy_score for u in units], dtype=float)
    score_delta = (
        float(smart_summary["score"] - legacy_summary["score"])
        if smart_summary.get("score") is not None and legacy_summary.get("score") is not None
        else None
    )
    return {
        "evaluation_mode": "retrospective_stabilized_history",
        "cutoff_date": cutoff_date,
        "horizon": horizon,
        "min_history": min_history,
        "forecast_zero_policy": forecast_zero_policy,
        "n_input_wells": len(wells),
        "n_eligible_wells": len(splits),
        "n_skipped_before_min_history": len(skipped),
        "n_scored_series": len(units),
        "failure_count": len(failures),
        "failures": failures[:100],
        "arms": {"smartcast_v1": smart_summary, "legacy_arps": legacy_summary},
        "comparison": {
            "pooled_score_delta": score_delta,
            "pooled_relative_delta": (
                score_delta / legacy_summary["score"]
                if score_delta is not None and legacy_summary.get("score")
                else None
            ),
            "paired_series_absolute_log_error_delta_mean": float(np.mean(series_deltas)) if series_deltas.size else None,
            "paired_series_absolute_log_error_delta_median": float(np.median(series_deltas)) if series_deltas.size else None,
            "paired_series_win_rate": float(np.mean(series_deltas < 0)) if series_deltas.size else None,
            "paired_bootstrap_score_delta_ci95": _bootstrap_delta(units, bootstrap_reps, 20260728),
            "bootstrap_unit": "well-phase series",
            "bootstrap_reps": bootstrap_reps,
        },
        "leakage_control": "SmartCast cohorts were built only from each eligible well's truncated training history.",
        "limitation": (
            "A revised current download is not a historical reporting snapshot. This test measures retrospective "
            "forecasting on stabilized data, not reporting-lag behavior, unless the input is an archived point-in-time snapshot."
        ),
    }
