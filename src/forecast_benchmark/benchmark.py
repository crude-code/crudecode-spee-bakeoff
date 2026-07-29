"""Ties data -> split -> forecast -> metrics together into one benchmark run.

Every arm reaches the scorer through the provider seam (see providers.py),
so the deterministic model and the LLM are scored by literally the same
code path: same calendar cutoff, same holdout window, same metrics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forecast_benchmark.data import PHASES, WellSeries
from forecast_benchmark.metrics import ALL_METRICS
from forecast_benchmark.split import make_splits_at_date


@dataclass
class WellResult:
    well_id: str
    phase: str
    phase_available: bool
    scores: dict[str, float | None]
    forecast_missing: bool = False  # phase had data but the provider had no forecast


@dataclass
class BenchmarkResult:
    model_name: str
    cutoff_month: str  # ISO cutoff date, e.g. "2025-04-01"
    horizon: int
    n_wells_scored: int
    n_wells_skipped: int
    skipped_well_ids: list[str]
    per_well: list[WellResult]
    summary: dict[str, dict[str, float | None]]  # phase -> metric -> aggregate


def run_provider_benchmark(
    wells: list[WellSeries],
    provider,  # ForecastProvider — see providers.py
    *,
    model_name: str,
    cutoff_date: str,
    horizon: int,
) -> BenchmarkResult:
    """Calendar-cutoff benchmark over the provider seam. This is the runner
    both arms of a comparison share: in-repo models come through
    providers.from_model_fn, external forecasts (LLM, vendor, human)
    through providers.precomputed — identical splits, identical metrics.

    A provider returning None for an available phase is recorded as
    forecast_missing, counted in the summary, and never back-filled.
    """
    splits, skipped = make_splits_at_date(wells, cutoff_date=cutoff_date, horizon=horizon)

    per_well: list[WellResult] = []
    for split in splits:
        for phase in PHASES:
            available = split.train.phase_available(phase)
            if not available:
                per_well.append(WellResult(
                    well_id=split.well_id, phase=phase,
                    phase_available=False, scores={},
                ))
                continue

            forecast = provider(split, phase)
            if forecast is None:
                per_well.append(WellResult(
                    well_id=split.well_id, phase=phase,
                    phase_available=True, scores={}, forecast_missing=True,
                ))
                continue

            actual = split.holdout_actuals[phase]
            scores = {name: fn(actual, forecast) for name, fn in ALL_METRICS.items()}
            per_well.append(WellResult(
                well_id=split.well_id, phase=phase,
                phase_available=True, scores=scores,
            ))

    return BenchmarkResult(
        model_name=model_name,
        cutoff_month=cutoff_date,
        horizon=horizon,
        n_wells_scored=len(splits),
        n_wells_skipped=len(skipped),
        skipped_well_ids=skipped,
        per_well=per_well,
        summary=_summarize(per_well),
    )


def _summarize(per_well: list[WellResult]) -> dict[str, dict[str, float | None]]:
    summary: dict[str, dict[str, float | None]] = {}
    for phase in PHASES:
        phase_results = [
            r for r in per_well
            if r.phase == phase and r.phase_available and not r.forecast_missing
        ]
        phase_summary: dict[str, float | None] = {
            "n_scored": len(phase_results),
            "n_phase_unavailable": sum(
                1 for r in per_well if r.phase == phase and not r.phase_available
            ),
            "n_forecast_missing": sum(
                1 for r in per_well if r.phase == phase and r.forecast_missing
            ),
        }
        for metric_name in ALL_METRICS:
            values = [
                r.scores[metric_name] for r in phase_results
                if r.scores.get(metric_name) is not None
            ]
            if values:
                phase_summary[f"{metric_name}_median"] = float(np.median(values))
                phase_summary[f"{metric_name}_mean"] = float(np.mean(values))
            else:
                phase_summary[f"{metric_name}_median"] = None
                phase_summary[f"{metric_name}_mean"] = None
        summary[phase] = phase_summary
    return summary
