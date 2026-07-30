import json

import numpy as np
import pytest

from forecast_benchmark.data import WellSeries, load_csv
from forecast_benchmark.overrides import load_overrides
from forecast_benchmark.smartcast import SmartCastProvider, enforce_terminal_decline, smartcast_phase
from forecast_benchmark.split import Split


def test_terminal_decline_applies_to_flat_family():
    raw = np.full(360, 100.0)
    out = enforce_terminal_decline(raw, 0.06)
    assert np.all(np.diff(out) <= 0)
    # One year after the starting point is approximately 6% lower.
    assert out[12] == pytest.approx(out[0] * 0.94, rel=0.01)


def test_smartcast_is_finite_and_does_not_mutate_input():
    t = np.arange(30, dtype=float)
    q = 5000 / np.power(1 + 0.8 * 0.08 * t, 1 / 0.8)
    original = q.copy()
    fc, diag = smartcast_phase(q, 24)
    assert np.array_equal(q, original)
    assert fc.shape == (24,)
    assert np.all(np.isfinite(fc))
    assert np.all(fc >= 0)
    assert diag.model_name


def test_provider_ratio_forecasts_all_phases():
    t = np.arange(24, dtype=float)
    oil = 4000 * np.exp(-0.05 * t)
    gas = oil * (2.0 * np.exp(0.01 * t))
    water = oil * 0.4
    w = WellSeries("w", [f"2024-{i+1:02d}-01" if i < 12 else f"2025-{i-11:02d}-01" for i in range(24)], oil, gas, water)
    provider = SmartCastProvider([w], {"w": {"basin": "x"}})
    split = Split("w", w, ["x"] * 12, {"oil": np.zeros(12), "gas": np.zeros(12), "water": np.zeros(12)})
    for phase in ("oil", "gas", "water"):
        fc = provider(split, phase)
        assert fc is not None and len(fc) == 12 and np.all(fc >= 0)


def test_load_csv_explicit_nan_gap_policy(tmp_path):
    path = tmp_path / "history.csv"
    path.write_text("well_id,month,oil,gas,water\nw,2024-01-01,10,20,1\nw,2024-03-01,8,18,1\n")
    with pytest.raises(ValueError):
        load_csv(str(path))
    w = load_csv(str(path), gap_policy="nan")[0]
    assert w.months == ["2024-01-01", "2024-02-01", "2024-03-01"]
    assert np.isnan(w.oil[1])


def test_override_package_requires_approval_and_full_horizon(tmp_path):
    path = tmp_path / "overrides.json"
    path.write_text(json.dumps({
        "approver": "Engineer A",
        "approved_at": "2026-07-28T12:00:00Z",
        "overrides": [{"well_id": "w", "phase": "oil", "reason": "reviewed cutoff upset", "values": [1, 2, 3]}],
    }))
    overrides, audit = load_overrides(path, horizon=3)
    assert audit.count == 1
    assert np.array_equal(overrides[("w", "oil")], np.array([1.0, 2.0, 3.0]))


def test_terminal_cutoff_dip_preserves_scipy_capacity_anchor():
    from forecast_benchmark.arps_scipy_v1 import arps_bounded_scipy_v1

    t = np.arange(36, dtype=float)
    q = 6000 / np.power(1 + 0.75 * 0.08 * t, 1 / 0.75)
    q[-2:] *= 0.12
    fc, diag = smartcast_phase(q, 12)
    legacy = enforce_terminal_decline(arps_bounded_scipy_v1(q)(12), 0.06)
    assert "terminal_low" in diag.flags
    assert np.allclose(fc, legacy, rtol=1e-10, atol=1e-10)


def test_ratio_and_cohort_are_suppressed_during_cutoff_disruption():
    t = np.arange(30, dtype=float)
    oil = 5000 / np.power(1 + 0.8 * 0.09 * t, 1 / 0.8)
    gas = oil * np.exp(0.02 * t)
    water = oil * 0.3
    oil[-2:] *= 0.1
    gas[-2:] *= 0.1
    water[-2:] *= 0.1
    months = [f"{2023 + i // 12:04d}-{i % 12 + 1:02d}-01" for i in range(30)]
    wells = []
    metadata = {}
    for i in range(6):
        wid = f"w{i}"
        wells.append(WellSeries(wid, months, oil.copy(), gas.copy(), water.copy()))
        metadata[wid] = {"basin": "x"}
    provider = SmartCastProvider(wells, metadata)
    split = Split("w0", wells[0], ["future"] * 12, {p: np.zeros(12) for p in ("oil", "gas", "water")})
    provider(split, "oil")
    diagnostics = provider.diagnostics[("w0", 12)]
    for phase_diag in diagnostics.phases:
        assert "cohort_suppressed_terminal_disruption" in phase_diag.flags
    secondary = [d for d in diagnostics.phases if d.phase != diagnostics.primary_phase]
    assert all("ratio_suppressed_terminal_disruption" in d.flags for d in secondary)


def test_terminal_decline_ablation_switch_is_live():
    from forecast_benchmark.smartcast import SmartCastConfig, _finalize_forecast

    raw = np.full(24, 100.0)
    off = _finalize_forecast(raw, SmartCastConfig(use_terminal_decline=False))
    on = _finalize_forecast(raw, SmartCastConfig(use_terminal_decline=True))
    assert np.array_equal(off, raw)
    assert on[-1] < on[0]


def test_conservative_cohort_requires_target_play_metadata():
    from forecast_benchmark.smartcast import SmartCastConfig

    t = np.arange(24, dtype=float)
    q = 5000 * np.exp(-0.05 * t)
    months = [f"{2023 + i // 12:04d}-{i % 12 + 1:02d}-01" for i in range(24)]
    cohort = [WellSeries(f"c{i}", months, q, q * 2, q * 0.2) for i in range(8)]
    metadata = {w.well_id: {"basin": "delaware"} for w in cohort}
    cfg = SmartCastConfig(
        use_cohort=True,
        use_ratio_coupling=False,
        require_target_group_metadata=True,
        allow_global_cohort_fallback=False,
    )
    provider = SmartCastProvider(cohort, metadata, cfg)
    target = WellSeries("target", months, q, q * 2, q * 0.2)
    split = Split("target", target, ["future"] * 12, {p: np.zeros(12) for p in ("oil", "gas", "water")})
    with pytest.raises(ValueError, match="missing play/basin metadata"):
        provider(split, "oil")


def test_fit_cache_reused_across_routing_profiles():
    import forecast_benchmark.smartcast as sc

    t = np.arange(36, dtype=float)
    q = 5000 / np.power(1 + 0.8 * 0.08 * t, 1 / 0.8)
    sc.clear_smartcast_caches()
    sc.smartcast_phase(q, 12, sc.SmartCastConfig(candidate_better_smart_weight=0.4))
    first = len(sc._FIT_CACHE)
    sc.smartcast_phase(q, 12, sc.SmartCastConfig(candidate_better_smart_weight=0.7))
    second = len(sc._FIT_CACHE)
    assert first > 0
    assert second == first


def test_anchor_only_profile_matches_frozen_scipy_and_uses_fast_path():
    from forecast_benchmark.arps_scipy_v1 import arps_bounded_scipy_v1
    from forecast_benchmark.profiles import get_profile

    # Two reported points trigger the frozen control's flat-tail fallback, so
    # terminal post-processing must visibly convert that flat path to 6% decline.
    q = np.array([4500.0, 4400.0])
    months = ["2024-01-01", "2024-02-01"]
    well = WellSeries("w", months, q, q * 2.0, q * 0.2)
    provider = SmartCastProvider([well], {"w": {"basin": "delaware"}}, get_profile("anchor_only"))
    split = Split("w", well, ["future"] * 12, {p: np.zeros(12) for p in ("oil", "gas", "water")})

    actual = provider(split, "oil")
    expected = enforce_terminal_decline(arps_bounded_scipy_v1(q)(12), 0.06)
    assert np.allclose(actual, expected, rtol=1e-10, atol=1e-10)
    diag = provider.diagnostics[("w", 12)].phases[0]
    assert diag.routing_reason == "anchor_only_fast_path"
    assert diag.smart_weight == 0.0


def test_scipy_control_profile_is_exact_unmodified_control():
    from forecast_benchmark.arps_scipy_v1 import arps_bounded_scipy_v1
    from forecast_benchmark.profiles import get_profile

    t = np.arange(30, dtype=float)
    q = 3200 / np.power(1 + 0.9 * 0.08 * t, 1 / 0.9)
    months = [f"{2023 + i // 12:04d}-{i % 12 + 1:02d}-01" for i in range(30)]
    well = WellSeries("w-control", months, q, q * 3.0, q * 0.15)
    provider = SmartCastProvider(
        [well], {"w-control": {"basin": "midland"}}, get_profile("scipy_control")
    )
    split = Split(
        "w-control", well, ["future"] * 12,
        {p: np.zeros(12) for p in ("oil", "gas", "water")},
    )
    assert np.allclose(
        provider(split, "oil"), arps_bounded_scipy_v1(q)(12),
        rtol=1e-10, atol=1e-10,
    )


def test_inner_router_penalizes_zero_forecast_on_positive_actual():
    from forecast_benchmark.smartcast import _score

    actual = np.array([100.0, 90.0, 80.0, 70.0])
    accurate = np.array([98.0, 91.0, 79.0, 72.0])
    hidden_zero = np.array([98.0, 0.0, 79.0, 72.0])
    assert _score(actual, hidden_zero) > _score(actual, accurate)
    assert _score(actual, hidden_zero) > 0.9
