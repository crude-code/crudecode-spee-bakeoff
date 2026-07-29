import numpy as np

from forecast_benchmark.data import WellSeries
from forecast_benchmark.split import make_split_at_date


def _well(n_months: int, well_id: str = "W1") -> WellSeries:
    """A well starting 2021-01 with n_months of contiguous history."""
    months, y, m = [], 2021, 1
    for _ in range(n_months):
        months.append(f"{y}-{m:02d}-01")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    oil = np.linspace(1000, 500, n_months)
    return WellSeries(
        well_id=well_id,
        months=months,
        oil=oil,
        gas=oil * 2,
        water=np.full(n_months, np.nan),
    )


def test_splits_at_the_named_month():
    split = make_split_at_date(_well(24), cutoff_date="2021-12-01", horizon=6)
    assert split is not None
    assert split.train.months[-1] == "2021-12-01"
    assert len(split.train.months) == 12
    assert split.holdout_months[0] == "2022-01-01"
    assert len(split.holdout_months) == 6
    assert split.holdout_actuals["oil"].shape == (6,)


def test_returns_none_when_history_cannot_cover_the_holdout():
    # 15 months of history: cutoff at month 12 leaves only 3 for a 6mo window
    assert make_split_at_date(_well(15), cutoff_date="2021-12-01", horizon=6) is None


def test_returns_none_when_well_starts_after_the_cutoff():
    assert make_split_at_date(_well(24), cutoff_date="2020-06-01", horizon=6) is None


def test_exact_boundary_is_allowed():
    # 18 months: 12 train + exactly 6 holdout
    split = make_split_at_date(_well(18), cutoff_date="2021-12-01", horizon=6)
    assert split is not None
    assert len(split.holdout_months) == 6


def test_train_never_contains_holdout_months():
    split = make_split_at_date(_well(24), cutoff_date="2021-12-01", horizon=6)
    assert not set(split.train.months) & set(split.holdout_months)
