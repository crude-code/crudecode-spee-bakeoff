import numpy as np

from forecast_benchmark.data import WellSeries
from forecast_benchmark.profiles import get_profile
from forecast_benchmark.smartcast import SmartCastProvider
from forecast_benchmark.split import Split


def _months(n: int) -> list[str]:
    return [f"{2020 + i // 12:04d}-{i % 12 + 1:02d}-01" for i in range(n)]


def _well(well_id: str, n: int, *, scale: float = 1.0, endpoint_multiplier: float = 1.0) -> WellSeries:
    t = np.arange(n, dtype=float)
    oil = scale * 5000.0 / np.power(1.0 + 0.75 * 0.08 * t, 1.0 / 0.75)
    oil = oil.copy()
    oil[-1] *= endpoint_multiplier
    return WellSeries(
        well_id,
        _months(n),
        oil,
        oil * 2.0,
        oil * 0.2,
    )


def _split(well: WellSeries, horizon: int = 12) -> Split:
    return Split(
        well.well_id,
        well,
        ["future"] * horizon,
        {p: np.zeros(horizon) for p in ("oil", "gas", "water")},
    )


def test_gated_cohort_caps_authority_and_records_evidence():
    cohort = [_well(f"c{i}", 36, scale=0.85 + 0.01 * i) for i in range(40)]
    target = _well("target", 7)
    metadata = {w.well_id: {"basin": "delaware"} for w in cohort + [target]}
    provider = SmartCastProvider(cohort, metadata, get_profile("gated_cohort_v1"))

    fc = provider(_split(target), "oil")
    assert fc is not None and np.all(np.isfinite(fc)) and np.all(fc >= 0)
    diag = next(
        d for d in provider.diagnostics[(target.well_id, 12)].phases if d.phase == "oil"
    )
    assert 0.0 < diag.cohort_weight <= 0.25
    assert diag.cohort_support >= 8
    assert diag.cohort_gate_reason == "cohort_gate_pass"
    assert diag.cohort_log_mad is not None
    assert diag.cohort_similarity_error is not None


def test_gated_cohort_rejects_endpoint_spike():
    cohort = [_well(f"c{i}", 36, scale=0.85 + 0.01 * i) for i in range(40)]
    target = _well("target-spike", 7, endpoint_multiplier=3.0)
    metadata = {w.well_id: {"basin": "delaware"} for w in cohort + [target]}
    provider = SmartCastProvider(cohort, metadata, get_profile("gated_cohort_v1"))

    provider(_split(target), "oil")
    diag = next(
        d for d in provider.diagnostics[(target.well_id, 12)].phases if d.phase == "oil"
    )
    assert diag.cohort_weight == 0.0
    assert diag.cohort_gate_reason == "endpoint_level_anomaly"


def test_age_specific_support_prevents_false_confidence():
    # Overall peer count is 40, but only six peers support the forecast ages.
    short = [_well(f"short{i}", 8, scale=0.9 + 0.005 * i) for i in range(34)]
    long = [_well(f"long{i}", 36, scale=0.9 + 0.01 * i) for i in range(6)]
    cohort = short + long
    target = _well("target-support", 7)
    metadata = {w.well_id: {"basin": "delaware"} for w in cohort + [target]}
    provider = SmartCastProvider(cohort, metadata, get_profile("gated_cohort_v1"))

    provider(_split(target), "oil")
    diag = next(
        d for d in provider.diagnostics[(target.well_id, 12)].phases if d.phase == "oil"
    )
    assert diag.cohort_weight == 0.0
    assert diag.cohort_support < 8
    assert diag.cohort_gate_reason == "insufficient_age_specific_support"
