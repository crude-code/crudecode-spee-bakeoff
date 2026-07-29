"""Tests for calendar-date splits, the provider seam, and pad-level scoring."""
import json

import numpy as np
import pytest

from forecast_benchmark.arps import arps_hyperbolic_bounded_b
from forecast_benchmark.benchmark import run_provider_benchmark
from forecast_benchmark.data import WellSeries
from forecast_benchmark.pads import Pad, _sum_streams, concat_series, load_pads, run_pad_benchmark
from forecast_benchmark.providers import from_model_fn, load_precomputed_dir, precomputed
from forecast_benchmark.split import make_split_at_date, make_splits_at_date


def _months(start_year, start_month, n):
    out, y, m = [], start_year, start_month
    for _ in range(n):
        out.append(f"{y}-{m:02d}-01")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _well(well_id, start_year, start_month, oil, gas=None, water=None):
    n = len(oil)
    return WellSeries(
        well_id=well_id,
        months=_months(start_year, start_month, n),
        oil=np.array(oil, dtype=float),
        gas=np.array(gas, dtype=float) if gas is not None else np.full(n, np.nan),
        water=np.array(water, dtype=float) if water is not None else np.full(n, np.nan),
    )


# --- calendar-date splits ---

def test_date_split_hides_same_months_for_staggered_wells():
    early = _well("early", 2023, 1, list(range(30, 0, -1)))   # 2023-01 .. 2025-06
    late = _well("late", 2023, 7, list(range(24, 0, -1)))     # 2023-07 .. 2025-06
    for well in (early, late):
        s = make_split_at_date(well, cutoff_date="2024-12-01", horizon=6)
        assert s.train.months[-1] == "2024-12-01"
        assert s.holdout_months == _months(2025, 1, 6)


def test_date_split_skips_wells_that_cannot_cover_the_window():
    starts_after_cutoff = _well("young", 2025, 3, [10.0] * 12)
    ends_before_holdout = _well("dead", 2023, 1, [10.0] * 12)  # ends 2023-12
    splits, skipped = make_splits_at_date(
        [starts_after_cutoff, ends_before_holdout],
        cutoff_date="2024-12-01", horizon=6,
    )
    assert splits == []
    assert set(skipped) == {"young", "dead"}


# --- provider seam ---

def test_model_fn_provider_matches_direct_call():
    well = _well("w", 2023, 1, [100, 90, 80, 70, 60, 50, 40, 30])
    s = make_split_at_date(well, cutoff_date="2023-06-01", horizon=2)
    provider = from_model_fn(arps_hyperbolic_bounded_b)
    expected = arps_hyperbolic_bounded_b(s.train.oil)(2)
    np.testing.assert_allclose(provider(s, "oil"), expected)


def test_precomputed_provider_returns_none_for_unknown_well():
    well = _well("w", 2023, 1, [100, 90, 80, 70, 60, 50])
    s = make_split_at_date(well, cutoff_date="2023-04-01", horizon=2)
    provider = precomputed({"other_well": {"oil": [1.0, 2.0]}})
    assert provider(s, "oil") is None


def test_precomputed_provider_rejects_wrong_horizon():
    well = _well("w", 2023, 1, [100, 90, 80, 70, 60, 50])
    s = make_split_at_date(well, cutoff_date="2023-04-01", horizon=2)
    provider = precomputed({"w": {"oil": [1.0, 2.0, 3.0]}})
    with pytest.raises(ValueError, match="forecast has 3 months"):
        provider(s, "oil")


def test_provider_benchmark_counts_missing_forecasts_loudly():
    wells = [
        _well("has_forecast", 2023, 1, [100, 90, 80, 70, 60, 50]),
        _well("no_forecast", 2023, 1, [100, 90, 80, 70, 60, 50]),
    ]
    provider = precomputed({"has_forecast": {"oil": [65.0, 55.0]}})
    result = run_provider_benchmark(
        wells, provider, model_name="ext", cutoff_date="2023-04-01", horizon=2,
    )
    oil = result.summary["oil"]
    assert oil["n_scored"] == 1
    assert oil["n_forecast_missing"] == 1


def test_load_precomputed_dir_merges_and_rejects_duplicates(tmp_path):
    (tmp_path / "pad_a.json").write_text(json.dumps(
        {"forecasts": {"w1": {"oil": [1, 2]}}, "rationale": "ignored"}
    ))
    (tmp_path / "pad_b.json").write_text(json.dumps({"w2": {"gas": [3, 4]}}))
    merged = load_precomputed_dir(tmp_path)
    assert merged == {"w1": {"oil": [1.0, 2.0]}, "w2": {"gas": [3.0, 4.0]}}

    (tmp_path / "pad_c.json").write_text(json.dumps({"w1": {"oil": [9, 9]}}))
    with pytest.raises(ValueError, match="already provided"):
        load_precomputed_dir(tmp_path)


# --- pad summing and scoring ---

def test_sum_streams_nan_only_when_all_nan():
    a = np.array([10.0, np.nan, np.nan])
    b = np.array([5.0, 7.0, np.nan])
    out = _sum_streams([a, b])
    np.testing.assert_allclose(out[:2], [15.0, 7.0])
    assert np.isnan(out[2])


def test_concat_series_rejects_gap():
    train = _well("w", 2023, 1, [10.0, 9.0])          # ends 2023-02
    holdout = _well("w", 2023, 4, [7.0])               # starts 2023-04, gap
    with pytest.raises(ValueError, match="expected 2023-03-01"):
        concat_series(train, holdout)


def test_pad_benchmark_scores_summed_stream():
    # Two wells whose oil sums to a clean stream; a perfect summed forecast
    # must score zero error even though each per-well forecast is imperfect.
    w1 = _well("w1", 2023, 1, [100, 90, 80, 70, 60, 50])
    w2 = _well("w2", 2023, 3, [40, 30, 20, 10])
    pad = Pad(pad_id="pad1", wells=[w1, w2])
    # holdout months 2023-05, 2023-06: actual sums are 60+20=80, 50+10=60
    provider = precomputed({
        "w1": {"oil": [70.0, 50.0]},   # off by +10, 0
        "w2": {"oil": [10.0, 10.0]},   # off by -10, 0
    })
    result = run_pad_benchmark(
        [pad], provider, model_name="ext", cutoff_date="2023-04-01", horizon=2,
    )
    (oil_row,) = [r for r in result.per_well if r.phase == "oil"]
    assert oil_row.scores["mape"] == 0.0
    assert oil_row.scores["bias"] == 0.0


def test_pad_benchmark_marks_pad_missing_if_any_well_missing():
    w1 = _well("w1", 2023, 1, [100, 90, 80, 70, 60, 50])
    w2 = _well("w2", 2023, 1, [40, 35, 30, 25, 20, 15])
    pad = Pad(pad_id="pad1", wells=[w1, w2])
    provider = precomputed({"w1": {"oil": [65.0, 55.0]}})  # w2 missing
    result = run_pad_benchmark(
        [pad], provider, model_name="ext", cutoff_date="2023-04-01", horizon=2,
    )
    (oil_row,) = [r for r in result.per_well if r.phase == "oil"]
    assert oil_row.forecast_missing
    assert result.summary["oil"]["n_forecast_missing"] == 1


def test_pad_benchmark_skips_pad_if_any_well_cannot_split():
    ok = _well("ok", 2023, 1, [100, 90, 80, 70, 60, 50])
    young = _well("young", 2023, 5, [30, 20])  # starts after cutoff
    pad = Pad(pad_id="pad1", wells=[ok, young])
    provider = from_model_fn(arps_hyperbolic_bounded_b)
    result = run_pad_benchmark(
        [pad], provider, model_name="arps", cutoff_date="2023-04-01", horizon=2,
    )
    assert result.n_wells_scored == 0
    assert result.skipped_well_ids == ["pad1"]


def test_load_pads_roundtrip(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "holdout").mkdir()
    (tmp_path / "pads.json").write_text(json.dumps(
        [{"pad_id": "padx", "basin": "TESTBASIN"}]
    ))
    (tmp_path / "train" / "padx.csv").write_text(
        "well_id,month,oil,gas,water\n"
        "w1,2023-01-01,100,,\n"
        "w1,2023-02-01,90,,\n"
    )
    (tmp_path / "holdout" / "padx.csv").write_text(
        "well_id,month,oil,gas,water\n"
        "w1,2023-03-01,80,,\n"
    )
    (pad,) = load_pads(tmp_path)
    assert pad.pad_id == "padx"
    assert pad.meta["basin"] == "TESTBASIN"
    (w1,) = pad.wells
    assert w1.months == ["2023-01-01", "2023-02-01", "2023-03-01"]
    np.testing.assert_allclose(w1.oil, [100.0, 90.0, 80.0])
    assert w1.phase_available("gas") is False
