import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from forecast_benchmark.arps import arps_hyperbolic_bounded_b
from forecast_benchmark.providers import from_model_fn
from forecast_benchmark.spee import generate_submission, load_submission_wells
from scripts.prepare_spee_input import normalize
from scripts.validate_spee_input import validate_submission


def _write_raw_package(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True)
    monthly = root / "monthly.csv"
    headers = root / "headers.csv"
    headers.write_text("api,pad,operator,basin\nw1,pad-a,op,basin\nw2,pad-a,op,basin\n")
    lines = ["api,prod_month,oil_bbl,gas_mcf,water_bbl\n"]
    for well in ("w1", "w2"):
        for month, oil in enumerate([100, 95, 90, 86, 82, 78], start=1):
            lines.append(f"{well},2024-{month:02d},{oil},{oil*10},{oil*0.5}\n")
    monthly.write_text("".join(lines))
    return monthly, headers


def test_prepare_validate_and_generate_strict_auto_submission(tmp_path):
    monthly, headers = _write_raw_package(tmp_path / "raw")
    out_dir = tmp_path / "spee_data"
    manifest = normalize(monthly, headers, out_dir)
    assert manifest["n_wells"] == 2
    assert (out_dir / "history.csv").is_file()
    assert (out_dir / "wells.csv").is_file()

    wells, metadata = load_submission_wells(out_dir)
    result = generate_submission(
        wells,
        from_model_fn(arps_hyperbolic_bounded_b),
        submission_type="strict_auto",
        model_name="arps_bounded_b",
        horizon=3,
        metadata=metadata,
    )
    assert result.n_wells == 2
    assert len(result.rows) == 2 * 3 * 3  # two wells, three phases, three months
    assert result.failures == []


def test_strict_auto_cli_writes_submission_and_run_log(tmp_path):
    monthly, headers = _write_raw_package(tmp_path / "raw")
    data_dir = tmp_path / "spee_data"
    normalize(monthly, headers, data_dir)

    repo = Path(__file__).resolve().parents[1]
    out = tmp_path / "submissions" / "strict.csv"
    log = tmp_path / "results" / "run.json"
    result = subprocess.run(
        [
            sys.executable, str(repo / "scripts" / "run_spee_strict_auto.py"), str(data_dir),
            "--horizon", "2", "--out", str(out), "--run-log", str(log),
            "--failures", str(tmp_path / "results" / "strict_failures.csv"),
            "--failure-report", str(tmp_path / "results" / "strict_failures.md"),
            "--diagnostics", str(tmp_path / "results" / "strict_diagnostics.json"),
            "--review-queue", str(tmp_path / "results" / "strict_review.csv"),
        ],
        cwd=repo, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.is_file()
    payload = json.loads(log.read_text())
    assert payload["submission_type"] == "strict_auto"
    assert payload["forecast_rows"] == 2 * 3 * 2
    assert payload["profile"] == "scipy_control"
    assert payload["algorithm_version"] == "smartcast_v1.5"
    assert payload["model_name"] == "smartcast_v1.5:scipy_control"
    assert validate_submission(out, horizon=2)["n_rows"] == 12


def test_forecast_start_common_window_skips_gap_before_submission(tmp_path):
    monthly, headers = _write_raw_package(tmp_path / "raw")
    data_dir = tmp_path / "spee_data"
    normalize(monthly, headers, data_dir)
    wells, metadata = load_submission_wells(data_dir)
    result = generate_submission(
        wells,
        from_model_fn(arps_hyperbolic_bounded_b),
        submission_type="strict_auto",
        model_name="arps_bounded_b",
        horizon=2,
        metadata=metadata,
        forecast_start="2024-09-01",
    )
    months = {row.forecast_month for row in result.rows}
    assert months == {"2024-09-01", "2024-10-01"}


def test_vendor_best_cli_uses_precomputed_llm_forecasts(tmp_path):
    monthly, headers = _write_raw_package(tmp_path / "raw")
    data_dir = tmp_path / "spee_data"
    normalize(monthly, headers, data_dir)
    llm_dir = tmp_path / "llm"
    llm_dir.mkdir()
    (llm_dir / "pad-a.json").write_text(json.dumps({
        "forecasts": {
            "w1": {"oil": [1, 1], "gas": [2, 2], "water": [3, 3]},
            "w2": {"oil": [1, 1], "gas": [2, 2], "water": [3, 3]},
        }
    }))

    repo = Path(__file__).resolve().parents[1]
    out = tmp_path / "submissions" / "vendor.csv"
    result = subprocess.run(
        [
            sys.executable, str(repo / "scripts" / "run_spee_vendor_best.py"), str(data_dir),
            "--llm-dir", str(llm_dir), "--horizon", "2", "--out", str(out),
            "--run-log", str(tmp_path / "results" / "vendor_run.json"),
            "--failures", str(tmp_path / "results" / "vendor_failures.csv"),
            "--failure-report", str(tmp_path / "results" / "vendor_failures.md"),
            "--diagnostics", str(tmp_path / "results" / "vendor_diagnostics.json"),
            "--review-queue", str(tmp_path / "results" / "vendor_review.csv"),
        ],
        cwd=repo, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12
    assert {row["submission_type"] for row in rows} == {"vendor_best"}
    assert {row["model_name"] for row in rows} == {"llm_precomputed"}
