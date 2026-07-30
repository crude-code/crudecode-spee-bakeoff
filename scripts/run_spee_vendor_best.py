#!/usr/bin/env python3
"""Generate the Vendor Best SPEE submission.

The default is SmartCast plus an automatically ranked review queue.  Optional
approved overrides replace complete well/phase trajectories.  The historical
precomputed-LLM workflow remains supported for regression, but is not the
recommended competition path.
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

from forecast_benchmark.arps_scipy_v1 import arps_bounded_scipy_v1
from forecast_benchmark.overrides import load_overrides, with_overrides
from forecast_benchmark.profiles import get_profile, profile_names
from forecast_benchmark.providers import from_model_fn, load_precomputed_dir, precomputed
from forecast_benchmark.smartcast import SmartCastProvider
from forecast_benchmark.spee import Timer, generate_submission, load_submission_wells, write_failure_markdown, write_failures_csv, write_run_log, write_submission_csv


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_dir", nargs="?", default="spee_data", type=Path)
    p.add_argument("--llm-dir", type=Path, help="legacy precomputed forecast directory")
    p.add_argument(
        "--profile",
        choices=profile_names(production_only=True),
        default="scipy_control",
        help="pre-registered profile; use only a profile that passed the real-board gate",
    )
    p.add_argument("--fallback-strict-auto", action="store_true")
    p.add_argument("--overrides", type=Path, help="approved override JSON package")
    p.add_argument("--horizon", type=int, default=360)
    p.add_argument("--forecast-start")
    p.add_argument("--gap-policy", choices=("nan", "zero", "error"), default=None)
    p.add_argument("--out", default="submissions/vendor_best_forecast.csv", type=Path)
    p.add_argument("--failures", default="results/vendor_best_failures.csv", type=Path)
    p.add_argument("--failure-report", default="results/vendor_best_failures.md", type=Path)
    p.add_argument("--run-log", default="results/vendor_best_run.json", type=Path)
    p.add_argument("--diagnostics", default="results/vendor_best_diagnostics.json", type=Path)
    p.add_argument("--review-queue", default="results/vendor_best_review_queue.csv", type=Path)
    args = p.parse_args()
    if args.horizon <= 0:
        sys.exit("run_spee_vendor_best failed: horizon must be positive")

    timer = Timer.start()
    try:
        wells, metadata = load_submission_wells(args.input_dir, gap_policy=args.gap_policy)
        smart = None
        config = None
        audit = None
        if args.llm_dir is not None:
            if not args.llm_dir.is_dir():
                raise FileNotFoundError(f"forecast dir not found: {args.llm_dir}")
            external = precomputed(load_precomputed_dir(args.llm_dir))
            if args.fallback_strict_auto:
                strict = from_model_fn(arps_bounded_scipy_v1)
                def base_provider(split, phase):
                    out = external(split, phase)
                    return out if out is not None else strict(split, phase)
                model_name = "llm_with_strict_auto_fallback"
            else:
                base_provider = external
                model_name = "llm_precomputed"
        else:
            config = get_profile(args.profile)
            smart = SmartCastProvider(wells, metadata, config)
            base_provider = smart
            model_name = f"smartcast_v1.5:{args.profile}:vendor_best"

        provider = base_provider
        if args.overrides is not None:
            overrides, audit = load_overrides(args.overrides, horizon=args.horizon)
            unknown = sorted({wid for wid, _ in overrides} - {w.well_id for w in wells})
            if unknown:
                raise ValueError(f"overrides reference unknown wells: {unknown[:10]}")
            provider = with_overrides(base_provider, overrides)
            model_name += "+approved_overrides"

        result = generate_submission(
            wells, provider, submission_type="vendor_best", model_name=model_name,
            horizon=args.horizon, metadata=metadata, forecast_start=args.forecast_start,
        )
        write_submission_csv(result, args.out)
        write_failures_csv(result, args.failures)
        write_failure_markdown(result, args.failure_report)
        if smart is not None:
            smart.write_diagnostics(args.diagnostics, args.review_queue)
        extra = {
            "submission_type": "vendor_best",
            "algorithm_version": "smartcast_v1.5" if smart is not None else "legacy",
            "profile": args.profile if smart is not None else None,
            "profile_config": asdict(config) if smart is not None else None,
            "llm_dir": str(args.llm_dir) if args.llm_dir else None,
            "fallback_strict_auto": args.fallback_strict_auto,
            "review_queue": str(args.review_queue) if smart is not None else None,
            "override_audit": audit.__dict__ if audit is not None else None,
        }
        write_run_log(
            result, args.run_log, elapsed_seconds=timer.elapsed_seconds(),
            command="run_spee_vendor_best", model_name=model_name,
            input_dir=args.input_dir, output_csv=args.out, extra=extra,
        )
    except Exception as exc:
        sys.exit(f"run_spee_vendor_best failed: {exc}")
    print(f"wrote {args.out} ({len(result.rows)} rows, {len(result.failures)} failures, {timer.elapsed_seconds()}s)")


if __name__ == "__main__":
    main()
