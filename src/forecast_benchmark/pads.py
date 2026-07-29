"""Pad-level grouping and scoring.

A pad is the unit a deal actually transacts on: co-developed wells sharing
a surface location, spacing, and operational history. Forecasts are still
made per well (by whatever provider), but the money question is the summed
stream — so this module scores both the parts and the sum.

Pad-level scoring is deliberately strict: if any well on the pad can't be
split at the cutoff, the pad is skipped loudly; if any well is missing a
forecast for an available phase, the pad's phase is scored as
forecast_missing. A pad sum built from a subset of its wells would look
like a forecast miss when it's actually a bookkeeping hole.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import json

import numpy as np

from forecast_benchmark.benchmark import BenchmarkResult, WellResult, _summarize
from forecast_benchmark.data import PHASES, WellSeries, load_csv
from forecast_benchmark.metrics import ALL_METRICS
from forecast_benchmark.split import make_split_at_date


@dataclass(frozen=True)
class Pad:
    pad_id: str
    wells: list[WellSeries]
    meta: dict = field(default_factory=dict)


def concat_series(train: WellSeries, holdout: WellSeries) -> WellSeries:
    """Stitch a train file and a holdout file back into one full series.

    The snapshot on disk keeps them in separate directories so the train
    side can be handed out with the holdout physically absent; the scorer
    is the only consumer that reassembles them. Raises unless the holdout
    starts exactly one month after the train ends.
    """
    if train.well_id != holdout.well_id:
        raise ValueError(f"cannot concat {train.well_id} with {holdout.well_id}")
    last, first = train.months[-1], holdout.months[0]
    y, m = int(last[:4]), int(last[5:7])
    expected = f"{y + 1}-01-01" if m == 12 else f"{y}-{m + 1:02d}-01"
    if first != expected:
        raise ValueError(
            f"{train.well_id}: holdout starts {first}, expected {expected} after train end {last}"
        )
    return WellSeries(
        well_id=train.well_id,
        months=train.months + holdout.months,
        oil=np.concatenate([train.oil, holdout.oil]),
        gas=np.concatenate([train.gas, holdout.gas]),
        water=np.concatenate([train.water, holdout.water]),
    )


def load_pads(data_dir: str | Path) -> list[Pad]:
    """Load a snapshot directory: pads.json + train/<pad>.csv + holdout/<pad>.csv.

    pads.json is a list of {"pad_id": ..., "wells": [...], ...meta}. Every
    well in the train file must have a matching holdout file entry — this
    loader reassembles full series for scoring and refuses to guess around
    a well that's present on one side only.
    """
    data_dir = Path(data_dir)
    pads_meta = json.loads((data_dir / "pads.json").read_text())
    pads: list[Pad] = []
    for meta in pads_meta:
        pad_id = meta["pad_id"]
        train_wells = {w.well_id: w for w in load_csv(str(data_dir / "train" / f"{pad_id}.csv"))}
        holdout_wells = {w.well_id: w for w in load_csv(str(data_dir / "holdout" / f"{pad_id}.csv"))}
        if set(train_wells) != set(holdout_wells):
            only = set(train_wells) ^ set(holdout_wells)
            raise ValueError(f"{pad_id}: wells present on one side of the split only: {sorted(only)}")
        wells = [concat_series(train_wells[wid], holdout_wells[wid]) for wid in sorted(train_wells)]
        pads.append(Pad(pad_id=pad_id, wells=wells, meta=meta))
    return pads


def _sum_streams(arrays: list[np.ndarray]) -> np.ndarray:
    """Month-wise sum where NaN means 'not reported': a month is NaN only
    if every component is NaN; otherwise reported values sum and missing
    ones contribute nothing. Zero and missing stay distinguishable at the
    component level, which is where that distinction is scored."""
    stacked = np.vstack(arrays)
    all_nan = np.all(np.isnan(stacked), axis=0)
    out = np.nansum(stacked, axis=0)
    out[all_nan] = np.nan
    return out


def run_pad_benchmark(
    pads: list[Pad],
    provider,  # ForecastProvider — see providers.py
    *,
    model_name: str,
    cutoff_date: str,
    horizon: int,
) -> BenchmarkResult:
    """Score summed pad streams under the same metrics as per-well runs.

    Rows in per_well are pads (well_id = pad_id). The calendar cutoff is
    what makes the sum meaningful: every well is blind to the same months,
    so summed forecasts and summed actuals align month-for-month.
    """
    per_pad: list[WellResult] = []
    skipped: list[str] = []
    n_scored = 0

    for pad in pads:
        splits = [make_split_at_date(w, cutoff_date=cutoff_date, horizon=horizon) for w in pad.wells]
        if any(s is None for s in splits):
            skipped.append(pad.pad_id)
            continue
        n_scored += 1

        for phase in PHASES:
            available = [s for s in splits if s.train.phase_available(phase)]
            if not available:
                per_pad.append(WellResult(
                    well_id=pad.pad_id, phase=phase, phase_available=False, scores={},
                ))
                continue

            forecasts = [provider(s, phase) for s in available]
            if any(f is None for f in forecasts):
                per_pad.append(WellResult(
                    well_id=pad.pad_id, phase=phase,
                    phase_available=True, scores={}, forecast_missing=True,
                ))
                continue

            forecast_sum = _sum_streams(forecasts)
            actual_sum = _sum_streams([s.holdout_actuals[phase] for s in available])
            scores = {name: fn(actual_sum, forecast_sum) for name, fn in ALL_METRICS.items()}
            per_pad.append(WellResult(
                well_id=pad.pad_id, phase=phase, phase_available=True, scores=scores,
            ))

    return BenchmarkResult(
        model_name=model_name,
        cutoff_month=cutoff_date,
        horizon=horizon,
        n_wells_scored=n_scored,
        n_wells_skipped=len(skipped),
        skipped_well_ids=skipped,
        per_well=per_pad,
        summary=_summarize(per_pad),
    )
