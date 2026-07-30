#!/usr/bin/env python3
"""Generate the no-touch Strict Auto SPEE submission.

The safe default is the verified ``anchor_only`` v1.3 profile: frozen original
SciPy bounded-Arps plus the common terminal-decline rule. Experimental SmartCast
profiles must be selected explicitly after they pass the real-board gate.
"""
from __future__ import annotations

import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_SRC = _BootstrapPath(__file__).resolve().parents[1] / "src"
if str(_BOOTSTRAP_SRC) not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, str(_BOOTSTRAP_SRC))

import argparse
from dataclasses import asdict
import sys
from pathlib import Path

from forecast_benchmark.arps import arps_hyperbolic_bounded_b, arps_inner_backtest_routed, arps_lag_skip_2_b_cap_1p0
from forecast_benchmark.profiles import get_profile, profile_names
from forecast_benchmark.providers import from_model_fn
from forecast_benchmark.smartcast import SmartCastProvider
from forecast_benchmark.spee import Timer, generate_submission, load_submission_wells, write_failure_markdown, write_failures_csv, write_run_log, write_submission_csv

LEGACY_MODELS = {
    "arps_bounded_b": arps_hyperbolic_bounded_b,
    "arps_inner_backtest_routed": arps_inner_backtest_routed,
    "arps_lag_skip_2_b_cap_1p0": arps_lag_skip_2_b_cap_1p0,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_dir", nargs="?", default="spee_data", type=Path)
    p.add_argument(
        "--model",
        choices=("smartcast_v15", "smartcast_v13", "smartcast_v1", *sorted(LEGACY_MODELS)),
        default="smartcast_v15",
        help="smartcast_v13 and smartcast_v1 are compatibility aliases for smartcast_v15",
    )
    p.add_argument(
        "--profile",
        choices=profile_names(production_only=True),
        default="scipy_control",
        help="pre-registered profile; do not change after Strict Auto starts",
    )
    p.add_argument("--horizon", type=int, default=360)
    p.add_argument("--forecast-start")
    p.add_argument("--gap-policy", choices=("nan", "zero", "error"), default=None)
    p.add_argument("--out", default="submissions/strict_auto_forecast.csv", type=Path)
    p.add_argument("--failures", default="results/strict_auto_failures.csv", type=Path)
    p.add_argument("--failure-report", default="results/strict_auto_failures.md", type=Path)
    p.add_argument("--run-log", default="results/strict_auto_run.json", type=Path)
    p.add_argument("--diagnostics", default="results/strict_auto_diagnostics.json", type=Path)
    p.add_argument("--review-queue", default="results/strict_auto_review_queue.csv", type=Path)
    args = p.parse_args()
    if args.horizon <= 0:
        sys.exit("run_spee_strict_auto failed: horizon must be positive")

    timer = Timer.start()
    try:
        wells, metadata = load_submission_wells(args.input_dir, gap_policy=args.gap_policy)
        if args.model in {"smartcast_v15", "smartcast_v13", "smartcast_v1"}:
            config = get_profile(args.profile)
            smart = SmartCastProvider(wells, metadata, config)
            provider = smart
            model_name = f"smartcast_v1.5:{args.profile}"
        else:
            smart = None
            config = None
            provider = from_model_fn(LEGACY_MODELS[args.model])
            model_name = args.model
        result = generate_submission(
            wells, provider, submission_type="strict_auto", model_name=model_name,
            horizon=args.horizon, metadata=metadata, forecast_start=args.forecast_start,
        )
        write_submission_csv(result, args.out)
        write_failures_csv(result, args.failures)
        write_failure_markdown(result, args.failure_report)
        if smart is not None:
            smart.write_diagnostics(args.diagnostics, args.review_queue)
        write_run_log(
            result, args.run_log, elapsed_seconds=timer.elapsed_seconds(),
            command="run_spee_strict_auto", model_name=model_name,
            input_dir=args.input_dir, output_csv=args.out,
            extra={
                "submission_type": "strict_auto",
                "algorithm_version": "smartcast_v1.5" if smart is not None else "legacy",
                "profile": args.profile if smart is not None else None,
                "profile_config": asdict(config) if config is not None else None,
                "diagnostics": str(args.diagnostics) if smart is not None else None,
                "review_queue": str(args.review_queue) if smart is not None else None,
                "human_intervention_after_start": False,
            },
        )
    except Exception as exc:
        sys.exit(f"run_spee_strict_auto failed: {exc}")
    print(f"wrote {args.out} ({len(result.rows)} rows, {len(result.failures)} failures, {timer.elapsed_seconds()}s)")


if __name__ == "__main__":
    main()
