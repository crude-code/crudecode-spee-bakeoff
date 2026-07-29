import json

import numpy as np

from scripts.diagnose_reporting_lag import diagnose_reporting_lag, markdown_summary as lag_markdown_summary
from scripts.run_python_arm_experiments import markdown_summary, run_experiments


def _rows(well_id, start_year, start_month, values):
    y, m = start_year, start_month
    out = []
    for value in values:
        out.append(f"{well_id},{y}-{m:02d}-01,{value},,\n")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def test_python_arm_experiment_runner_scores_all_variants(tmp_path):
    data_dir = tmp_path / "snapshot"
    (data_dir / "train").mkdir(parents=True)
    (data_dir / "holdout").mkdir()
    (data_dir / "snapshot.json").write_text(json.dumps({"cutoff": "2024-12-01", "horizon": 3}))
    (data_dir / "pads.json").write_text(json.dumps([{"pad_id": "pad-a", "wells": ["w1"]}]))

    header = "well_id,month,oil,gas,water\n"
    train = np.linspace(120.0, 70.0, 24)
    holdout = [68.0, 66.0, 64.0]
    (data_dir / "train" / "pad-a.csv").write_text(header + "".join(_rows("w1", 2023, 1, train)))
    (data_dir / "holdout" / "pad-a.csv").write_text(header + "".join(_rows("w1", 2025, 1, holdout)))

    results = run_experiments(data_dir)
    assert results["n_pads"] == 1
    assert results["n_wells"] == 1
    assert [arm["model_name"] for arm in results["arms"]] == [
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
    text = markdown_summary(results)
    assert "Python arm experiment board" in text
    assert "Pad-level summed streams" in text
    assert "`arps_bounded_b`" in text


def test_reporting_lag_diagnostic_summarizes_recent_capacity_ratios(tmp_path):
    data_dir = tmp_path / "snapshot"
    (data_dir / "train").mkdir(parents=True)
    (data_dir / "holdout").mkdir()
    (data_dir / "snapshot.json").write_text(json.dumps({"cutoff": "2024-12-01", "horizon": 3}))
    (data_dir / "pads.json").write_text(json.dumps([{"pad_id": "pad-a", "wells": ["w1"]}]))

    header = "well_id,month,oil,gas,water\n"
    train = np.linspace(120.0, 70.0, 24)
    train[-1] *= 0.5
    holdout = [68.0, 66.0, 64.0]
    (data_dir / "train" / "pad-a.csv").write_text(header + "".join(_rows("w1", 2023, 1, train)))
    (data_dir / "holdout" / "pad-a.csv").write_text(header + "".join(_rows("w1", 2025, 1, holdout)))

    results = diagnose_reporting_lag(data_dir, months_back=3)
    assert results["n_pads"] == 1
    assert results["n_rows"] >= 3
    text = lag_markdown_summary(results)
    assert "Reporting-lag diagnostic" in text
    assert "months before cutoff" in text
