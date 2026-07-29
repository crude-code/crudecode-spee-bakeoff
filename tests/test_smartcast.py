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


def test_terminal_cutoff_dip_preserves_legacy_capacity_anchor():
    from forecast_benchmark.arps import arps_hyperbolic_bounded_b

    t = np.arange(36, dtype=float)
    q = 6000 / np.power(1 + 0.75 * 0.08 * t, 1 / 0.75)
    q[-2:] *= 0.12
    fc, diag = smartcast_phase(q, 12)
    legacy = enforce_terminal_decline(arps_hyperbolic_bounded_b(q)(12), 0.06)
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
