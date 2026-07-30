import numpy as np

from forecast_benchmark.arps_scipy_v1 import arps_bounded_scipy_v1
from forecast_benchmark.data import WellSeries
from forecast_benchmark.profiles import get_profile, profile_names
from forecast_benchmark.smartcast import SmartCastProvider
from forecast_benchmark.split import Split


def _months(n: int) -> list[str]:
    return [f"{2020 + i // 12:04d}-{i % 12 + 1:02d}-01" for i in range(n)]


def _well(well_id: str, n: int, *, oil_scale: float = 1.0, gas_multiple: float = 2.0) -> WellSeries:
    t = np.arange(n, dtype=float)
    oil = oil_scale * 5000.0 / np.power(1.0 + 0.75 * 0.08 * t, 1.0 / 0.75)
    gas = oil * gas_multiple
    return WellSeries(well_id, _months(n), oil, gas, oil * 0.2)


def _split(well: WellSeries, horizon: int = 12) -> Split:
    return Split(
        well.well_id,
        well,
        ["future"] * horizon,
        {p: np.zeros(horizon) for p in ("oil", "gas", "water")},
    )


def _provider(target: WellSeries, play: str) -> SmartCastProvider:
    cohort = [_well(f"c{i}", 36, oil_scale=0.82 + 0.012 * i) for i in range(40)]
    metadata = {w.well_id: {"basin": play} for w in cohort + [target]}
    return SmartCastProvider(cohort, metadata, get_profile("selective_cohort_v1"))


def test_selective_profile_is_production_visible_and_frozen():
    assert "selective_cohort_v1" in profile_names(production_only=True)
    cfg = get_profile("selective_cohort_v1")
    assert cfg.cohort_allowed_primary_phases == ("oil",)
    assert set(cfg.cohort_allowed_groups or ()) == {"dj", "eagle_ford", "delaware"}
    assert cfg.cohort_allowed_forecast_phases == ("oil",)
    assert cfg.cohort_min_wells == 8
    assert cfg.cohort_full_support_wells == 30


def test_selective_profile_uses_cohort_for_allowed_oil_target():
    target = _well("target", 7, gas_multiple=2.0)  # oil-primary on BOE basis
    provider = _provider(target, "EAGLE_FORD")
    forecast = provider(_split(target), "oil")
    assert forecast is not None
    diag = next(d for d in provider.diagnostics[(target.well_id, 12)].phases if d.phase == "oil")
    assert diag.cohort_weight > 0.0
    assert "cohort_shrinkage" in diag.flags


def test_selective_profile_falls_back_to_exact_scipy_for_nonallowed_play():
    target = _well("target", 7, gas_multiple=2.0)
    provider = _provider(target, "MIDLAND")
    actual = provider(_split(target), "oil")
    expected = arps_bounded_scipy_v1(target.oil)(12)
    assert np.allclose(actual, expected, rtol=1e-10, atol=1e-10)
    diag = next(d for d in provider.diagnostics[(target.well_id, 12)].phases if d.phase == "oil")
    assert diag.cohort_weight == 0.0
    assert diag.cohort_gate_reason == "selective_route_play"


def test_selective_profile_falls_back_for_gas_primary_target():
    target = _well("target", 7, gas_multiple=12.0)  # gas-primary on BOE basis
    provider = _provider(target, "DJ")
    actual = provider(_split(target), "gas")
    expected = arps_bounded_scipy_v1(target.gas)(12)
    assert np.allclose(actual, expected, rtol=1e-10, atol=1e-10)
    diag = next(d for d in provider.diagnostics[(target.well_id, 12)].phases if d.phase == "gas")
    assert diag.cohort_weight == 0.0
    assert diag.cohort_gate_reason == "selective_route_primary_phase"


def test_selective_profile_keeps_secondary_phases_on_scipy():
    target = _well("target", 7, gas_multiple=2.0)
    provider = _provider(target, "DELAWARE")
    actual = provider(_split(target), "gas")
    expected = arps_bounded_scipy_v1(target.gas)(12)
    assert np.allclose(actual, expected, rtol=1e-10, atol=1e-10)
    diag = next(d for d in provider.diagnostics[(target.well_id, 12)].phases if d.phase == "gas")
    assert diag.cohort_weight == 0.0
    assert diag.cohort_gate_reason == "selective_route_forecast_phase"


def test_selective_profile_missing_play_fails_safe_to_scipy():
    target = _well("target-missing", 7, gas_multiple=2.0)
    cohort = [_well(f"c{i}", 36, oil_scale=0.82 + 0.012 * i) for i in range(40)]
    metadata = {w.well_id: {"basin": "DJ"} for w in cohort}
    provider = SmartCastProvider(cohort, metadata, get_profile("selective_cohort_v1"))
    actual = provider(_split(target), "oil")
    expected = arps_bounded_scipy_v1(target.oil)(12)
    assert np.allclose(actual, expected, rtol=1e-10, atol=1e-10)
    diag = next(d for d in provider.diagnostics[(target.well_id, 12)].phases if d.phase == "oil")
    assert diag.cohort_weight == 0.0
    assert diag.cohort_gate_reason == "selective_route_play"


def test_selective_profile_normalizes_supported_play_alias():
    target = _well("target-alias", 7, gas_multiple=2.0)
    cohort = [_well(f"c{i}", 36, oil_scale=0.82 + 0.012 * i) for i in range(40)]
    metadata = {w.well_id: {"basin": "Lower Eagle Ford"} for w in cohort + [target]}
    provider = SmartCastProvider(cohort, metadata, get_profile("selective_cohort_v1"))
    provider(_split(target), "oil")
    diag = next(d for d in provider.diagnostics[(target.well_id, 12)].phases if d.phase == "oil")
    assert diag.cohort_weight > 0.0
