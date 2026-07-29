import importlib

import numpy as np

from forecast_benchmark.data import WellSeries
from forecast_benchmark.experiments.allocation_noise import assess_allocation_noise
from forecast_benchmark.real_holdout import choose_retrospective_cutoff, run_retrospective_holdout
from forecast_benchmark.stressboard import aggregate_stress_runs, make_stress_cases, run_stress_board


def test_multiseed_aggregation_reports_required_gates():
    result = run_stress_board(1, n_wells=16, horizon=6, bootstrap_reps=10)
    aggregate = aggregate_stress_runs([result, result], bootstrap_reps=20)
    assert aggregate["n_seeds"] == 2
    assert "clean_non_regression" in aggregate["gates"]
    assert "worst_scenario_regression" in aggregate["gates"]
    assert "mean_score_delta_ci95" in aggregate
    assert "bias_component" in result["arms"]["smartcast_v1"]
    assert "spread_component" in result["arms"]["smartcast_v1"]


def test_allocation_noise_classifier_is_high_precision_on_fixed_fixture():
    cases = make_stress_cases(3, n_wells=80, horizon=12)
    by_scenario = {case.scenario: assess_allocation_noise(case.train).flagged for case in cases}
    # At least one allocation-noise case is identified while the clean fixture
    # does not trigger. The full multi-seed script measures precision/recall.
    allocation_flags = [assess_allocation_noise(c.train).flagged for c in cases if c.scenario == "allocation_noise"]
    clean_flags = [assess_allocation_noise(c.train).flagged for c in cases if c.scenario == "clean"]
    assert any(allocation_flags)
    assert not any(clean_flags)
    assert set(by_scenario)


def test_real_holdout_uses_common_cutoff_and_scores_without_leakage():
    cases = make_stress_cases(7, n_wells=16, horizon=12)
    wells = []
    metadata = {}
    for case in cases:
        full = WellSeries(
            well_id=case.train.well_id,
            months=case.train.months + [f"future-{i}" for i in range(12)],
            oil=np.concatenate([case.train.oil, case.actual["oil"]]),
            gas=np.concatenate([case.train.gas, case.actual["gas"]]),
            water=np.concatenate([case.train.water, case.actual["water"]]),
        )
        # Replace synthetic labels with valid sortable calendar months.
        start = 2020 * 12
        full = WellSeries(
            full.well_id,
            [f"{idx // 12:04d}-{idx % 12 + 1:02d}-01" for idx in range(start, start + len(full.months))],
            full.oil,
            full.gas,
            full.water,
        )
        wells.append(full)
        metadata[full.well_id] = case.metadata
    cutoff = choose_retrospective_cutoff(wells, horizon=6, stabilization_lag=2)
    result = run_retrospective_holdout(
        wells,
        metadata,
        cutoff_date=cutoff,
        horizon=6,
        min_history=8,
        bootstrap_reps=20,
    )
    assert result["n_scored_series"] > 0
    assert result["failure_count"] == 0
    assert "truncated training history" in result["leakage_control"]


def test_production_modules_do_not_import_experiments_package():
    for module_name in (
        "forecast_benchmark.smartcast",
        "forecast_benchmark.spee",
        "forecast_benchmark.overrides",
    ):
        module = importlib.import_module(module_name)
        source = open(module.__file__, encoding="utf-8").read()
        assert "forecast_benchmark.experiments" not in source
