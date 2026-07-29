import numpy as np

from forecast_benchmark.arps import (
    _flat_tail,
    arps_b_cap_1p0,
    arps_hyperbolic_bounded_b,
    arps_inner_backtest_routed,
    arps_lag_skip_2,
    arps_lag_skip_2_b_cap_1p0,
    trailing_capacity_ratios,
)


def _synthetic_hyperbolic(qi=5000, di=0.08, b=0.8, n=36):
    t = np.arange(n, dtype=float)
    q = qi / np.power(1.0 + b * di * t, 1.0 / b)
    return q


def test_fits_reasonably_on_synthetic_hyperbolic():
    q = _synthetic_hyperbolic(qi=5000, di=0.08, b=0.8, n=36)
    fn = arps_hyperbolic_bounded_b(q)
    forecast = fn(12)
    # forecast should continue the decline, staying in a sane range
    assert forecast[0] < q[-1] * 1.2
    assert forecast[0] > q[-1] * 0.5


def test_forecast_declines_monotonically():
    q = _synthetic_hyperbolic(n=30)
    forecast = arps_hyperbolic_bounded_b(q)(12)
    assert len(forecast) == 12
    assert np.all(np.diff(forecast) <= 0)


def test_fits_from_peak_not_from_first_month():
    """A flowback ramp before the peak is not decline. Including it would
    flatten the fit, so the forecast off a ramped history should match the
    forecast off the same history with the ramp removed."""
    decline = _synthetic_hyperbolic(qi=5000, di=0.08, b=0.8, n=30)
    ramped = np.concatenate([[800.0, 2600.0], decline])
    assert np.allclose(
        arps_hyperbolic_bounded_b(ramped)(12),
        arps_hyperbolic_bounded_b(decline)(12),
    )


def test_clean_well_gets_no_haircut():
    """A well with no downtime should forecast the bare curve: uptime
    measures ~1 and is capped there, so the haircut must be a no-op."""
    q = _synthetic_hyperbolic(qi=5000, di=0.08, b=0.8, n=36)
    t_future = np.arange(36, 48, dtype=float)
    true_continuation = 5000 / np.power(1.0 + 0.8 * 0.08 * t_future, 1.0 / 0.8)
    forecast = arps_hyperbolic_bounded_b(q)(12)
    assert np.allclose(forecast, true_continuation, rtol=0.02)


def test_downtime_costs_measured_uptime_not_a_dragged_fit():
    """Two half-rate months must not drag the fitted level or slope down;
    they should cost the forecast exactly their measured uptime share.
    Expected reported volumes = capacity curve x uptime — the same
    arithmetic the LLM skill commits to."""
    clean = _synthetic_hyperbolic(qi=5000, di=0.08, b=0.8, n=36)
    dipped = clean.copy()
    dipped[20] *= 0.5
    dipped[28] *= 0.5
    f_clean = arps_hyperbolic_bounded_b(clean)(12)
    f_dipped = arps_hyperbolic_bounded_b(dipped)(12)
    # trailing 24-month window sees 22 clean months and 2 at half rate
    expected_uptime = (22 + 2 * 0.5) / 24
    assert np.allclose(f_dipped / f_clean, expected_uptime, atol=0.02)


def test_haircut_never_boosts():
    """Uptime is capped at 1.0: downtime only ever subtracts, so a fit that
    happens to sit under the data must not become an upward adjustment."""
    clean = _synthetic_hyperbolic(n=36)
    dipped = clean.copy()
    dipped[30] *= 0.4
    f_clean = arps_hyperbolic_bounded_b(clean)(12)
    f_dipped = arps_hyperbolic_bounded_b(dipped)(12)
    assert np.all(f_dipped <= f_clean * 1.001)


def test_reported_zero_month_is_downtime_not_curve_data():
    """A full month down (reported zero) is struck from the fit but priced
    into the haircut — the curve keeps its shape, the level pays ~1/24."""
    clean = _synthetic_hyperbolic(qi=5000, di=0.08, b=0.8, n=36)
    down = clean.copy()
    down[27] = 0.0
    f_clean = arps_hyperbolic_bounded_b(clean)(12)
    f_down = arps_hyperbolic_bounded_b(down)(12)
    assert np.allclose(f_down / f_clean, 23 / 24, atol=0.02)


def test_history_beyond_fit_window_is_ignored():
    """A five-year well and the same well truncated to its last 36 months
    must forecast identically: month 3 has no business steering year 6."""
    old = _synthetic_hyperbolic(qi=8000, di=0.10, b=1.1, n=60)
    assert np.allclose(
        arps_hyperbolic_bounded_b(old)(12),
        arps_hyperbolic_bounded_b(old[-36:])(12),
    )


def test_early_flow_regime_does_not_contaminate_forecast():
    """A well whose early transient decline is steeper than its current
    boundary-dominated regime must be forecast off the current regime.
    History: 24 steep months, then 36 months of a shallower curve — the
    window covers exactly the late regime, so the forecast should match
    that curve's true continuation."""
    early = _synthetic_hyperbolic(qi=20000, di=0.30, b=1.3, n=24)
    late_qi, late_di, late_b = early[-1] * 0.98, 0.04, 0.5
    t_late = np.arange(36, dtype=float)
    late = late_qi / np.power(1.0 + late_b * late_di * t_late, 1.0 / late_b)
    q = np.concatenate([early, late])

    t_future = np.arange(36, 48, dtype=float)
    true_continuation = late_qi / np.power(
        1.0 + late_b * late_di * t_future, 1.0 / late_b
    )
    forecast = arps_hyperbolic_bounded_b(q)(12)
    assert np.allclose(forecast, true_continuation, rtol=0.03)


def test_short_history_falls_back_to_flat_tail():
    q = np.array([100.0, 90.0])  # too short to fit
    forecast = arps_hyperbolic_bounded_b(q)(3)
    assert len(forecast) == 3
    assert np.allclose(forecast, np.mean([100.0, 90.0]))


def test_all_nan_history_falls_back_to_zero():
    forecast = arps_hyperbolic_bounded_b(np.full(10, np.nan))(3)
    assert np.allclose(forecast, 0.0)


def test_flat_tail_averages_last_three_reported_months():
    q = np.array([100.0, 90.0, 80.0, 70.0, 60.0])
    forecast = _flat_tail(q)(4)
    assert np.allclose(forecast, np.mean([80.0, 70.0, 60.0]))


def test_flat_tail_skips_nan_months():
    forecast = _flat_tail(np.array([100.0, 90.0, np.nan]))(2)
    assert np.allclose(forecast, np.mean([100.0, 90.0]))

from forecast_benchmark.arps import (
    arps_no_uptime_haircut,
    arps_recent_low_guard,
    arps_window_24,
    arps_window_48,
    arps_all_post_peak,
    python_arm_experiments,
)


def test_python_arm_experiments_are_named_and_callable():
    arms = python_arm_experiments()
    assert list(arms) == [
        "arps_bounded_b",
        "arps_inner_backtest_routed",
        "arps_lag_skip_2",
        "arps_lag_skip_3",
        "arps_b_cap_1p0",
        "arps_lag_skip_2_b_cap_1p0",
        "arps_window_24",
        "arps_window_48",
        "arps_all_post_peak",
        "arps_no_uptime_haircut",
        "arps_recent_low_guard",
    ]
    q = _synthetic_hyperbolic(qi=3000, di=0.06, b=0.7, n=48)
    for model_fn in arms.values():
        forecast = model_fn(q)(12)
        assert forecast.shape == (12,)
        assert np.all(np.isfinite(forecast))


def test_window_variants_actually_change_the_question():
    early = _synthetic_hyperbolic(qi=20000, di=0.30, b=1.3, n=24)
    late_qi, late_di, late_b = early[-1] * 0.98, 0.04, 0.5
    late = late_qi / np.power(
        1.0 + late_b * late_di * np.arange(60, dtype=float),
        1.0 / late_b,
    )
    q = np.concatenate([early, late[:48]])
    q[-48:-36] *= 1.35  # older late-window allocation/regime contamination
    f24 = arps_window_24(q)(12)
    f48 = arps_window_48(q)(12)
    fall = arps_all_post_peak(q)(12)
    # The variants should be true experiments, not aliases for each other.
    assert not np.allclose(f24, f48)
    assert not np.allclose(f24, fall)


def test_no_uptime_haircut_is_capacity_curve_only():
    clean = _synthetic_hyperbolic(qi=5000, di=0.08, b=0.8, n=36)
    dipped = clean.copy()
    dipped[20] *= 0.5
    dipped[28] *= 0.5
    with_haircut = arps_hyperbolic_bounded_b(dipped)(12)
    capacity_only = arps_no_uptime_haircut(dipped)(12)
    assert np.all(capacity_only >= with_haircut)
    assert np.mean(capacity_only / with_haircut) > 1.04


def test_recent_low_guard_preserves_terminal_crash_as_status():
    clean = _synthetic_hyperbolic(qi=5000, di=0.08, b=0.8, n=36)
    shut_in = clean.copy()
    shut_in[-2:] = 0.0
    official = arps_hyperbolic_bounded_b(shut_in)(12)
    guarded = arps_recent_low_guard(shut_in)(12)
    # Official arm reads terminal zeros as downtime and preserves capacity;
    # the experimental guard reads them as possible current status.
    assert np.mean(guarded) < np.mean(official) * 0.5


def test_lag_skip_ignores_depressed_fresh_months_but_keeps_cutoff_origin():
    clean = _synthetic_hyperbolic(qi=5000, di=0.08, b=0.8, n=36)
    lagged = clean.copy()
    lagged[-2:] *= 0.45
    official = arps_hyperbolic_bounded_b(lagged)(12)
    lag_skip = arps_lag_skip_2(lagged)(12)
    clean_forecast = arps_hyperbolic_bounded_b(clean)(12)

    # The lag-skip arm should be much closer to the clean forecast because
    # the final two low reports are not allowed to anchor fit or uptime.
    assert np.mean(np.abs(lag_skip - clean_forecast)) < np.mean(np.abs(official - clean_forecast))
    # It still forecasts from the true cutoff, not two months early.
    assert lag_skip[0] < clean[-1] * 1.05


def test_b_cap_1p0_variant_blocks_high_b_tail_flatness():
    t = np.arange(36, dtype=float)
    # A clean b=1.3 history gives the default grid permission to pick a
    # high-b flatter tail. The capped variant should forecast lower.
    q = 7000 / np.power(1.0 + 1.3 * 0.055 * t, 1.0 / 1.3)
    default = arps_hyperbolic_bounded_b(q)(12)
    capped = arps_b_cap_1p0(q)(12)
    assert capped[-1] < default[-1]


def test_combined_lag_and_b_cap_variant_is_callable():
    q = _synthetic_hyperbolic(qi=4000, di=0.07, b=0.9, n=42)
    q[-2:] *= 0.5
    forecast = arps_lag_skip_2_b_cap_1p0(q)(12)
    assert forecast.shape == (12,)
    assert np.all(np.isfinite(forecast))


def test_trailing_capacity_ratios_returns_recent_offsets():
    q = _synthetic_hyperbolic(qi=5000, di=0.08, b=0.8, n=36)
    q[-1] *= 0.5
    ratios = trailing_capacity_ratios(q, months_back=3)
    assert [r["months_before_cutoff"] for r in ratios] == [1, 2, 3]
    assert ratios[0]["ratio"] < ratios[1]["ratio"]


def test_inner_backtest_routed_is_finite_and_preserves_short_history_fallback():
    q = _synthetic_hyperbolic(qi=4500, di=0.07, b=0.8, n=54)
    forecast = arps_inner_backtest_routed(q)(12)
    assert forecast.shape == (12,)
    assert np.all(np.isfinite(forecast))

    short = np.array([100.0, 90.0, 80.0, 70.0])
    assert np.allclose(
        arps_inner_backtest_routed(short)(3),
        arps_hyperbolic_bounded_b(short)(3),
    )
