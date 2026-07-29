"""Train/holdout split for hindcasting. Point-in-time discipline: a model
under test only ever sees months up to the cutoff — never the holdout.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forecast_benchmark.data import PHASES, WellSeries


@dataclass(frozen=True)
class Split:
    well_id: str
    train: WellSeries          # months[:cutoff]
    holdout_months: list[str]  # months[cutoff:cutoff+horizon]
    holdout_actuals: dict[str, np.ndarray]  # phase -> actual values over holdout_months


def make_split_at_date(well: WellSeries, *, cutoff_date: str, horizon: int) -> Split | None:
    """Split one well at a calendar month, scoring the next `horizon` months.

    cutoff_date is an ISO "YYYY-MM-01" string; train is every month up to
    and including it. The cutoff is calendar-based rather than an index
    because wells on the same pad start producing on different dates — an
    index cutoff would blind each well to a different calendar period and
    make the pad sum meaningless.

    Returns None (not an exception) if the well starts after the cutoff or
    doesn't extend through the full holdout window — a benchmark should
    skip ineligible wells loudly in aggregate stats, not crash and not
    silently truncate the holdout shorter than requested.
    """
    n_train = sum(1 for m in well.months if m <= cutoff_date)
    if n_train < 1 or n_train + horizon > len(well.months):
        return None

    return Split(
        well_id=well.well_id,
        train=well.truncate(n_train),
        holdout_months=well.months[n_train:n_train + horizon],
        holdout_actuals={
            phase: getattr(well, phase)[n_train:n_train + horizon]
            for phase in PHASES
        },
    )


def make_splits_at_date(
    wells: list[WellSeries], *, cutoff_date: str, horizon: int
) -> tuple[list[Split], list[str]]:
    """Split every well at the same calendar cutoff; returns
    (splits, skipped_well_ids). Skips are a first-class output, not a silent
    filter — a benchmark run should always report how many wells it could
    and couldn't score."""
    splits, skipped = [], []
    for well in wells:
        s = make_split_at_date(well, cutoff_date=cutoff_date, horizon=horizon)
        if s is None:
            skipped.append(well.well_id)
        else:
            splits.append(s)
    return splits, skipped
