#!/usr/bin/env python3
"""Run paired multi-seed SmartCast validation with confidence intervals."""
from __future__ import annotations

import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_SRC = _BootstrapPath(__file__).resolve().parents[1] / "src"
if str(_BOOTSTRAP_SRC) not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, str(_BOOTSTRAP_SRC))

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path

from forecast_benchmark.stressboard import aggregate_stress_runs, run_stress_board


def _parse_seeds(raw: str) -> list[int]:
    seeds: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            a, b = int(start), int(end)
            if b < a:
                raise ValueError(f"invalid seed range: {token}")
            seeds.extend(range(a, b + 1))
        else:
            seeds.append(int(token))
    unique = sorted(set(seeds))
    if not unique:
        raise ValueError("at least one seed is required")
    return unique


def _run_one(args: tuple[int, int, int, int, str]) -> dict:
    seed, wells, horizon, bootstrap_reps, zero_policy = args
    return run_stress_board(
        seed,
        n_wells=wells,
        horizon=horizon,
        bootstrap_reps=bootstrap_reps,
        forecast_zero_policy=zero_policy,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default="1-30", help="comma list and/or inclusive ranges, e.g. 1-30,101")
    p.add_argument("--wells", type=int, default=180)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    p.add_argument("--bootstrap-reps", type=int, default=500)
    p.add_argument("--aggregate-bootstrap-reps", type=int, default=5000)
    p.add_argument("--forecast-zero-policy", choices=("drop", "epsilon", "error"), default="epsilon")
    p.add_argument("--min-seed-win-rate", type=float, default=0.70)
    p.add_argument("--max-clean-relative-regression", type=float, default=0.05)
    p.add_argument("--max-worst-scenario-relative-regression", type=float, default=0.25)
    p.add_argument("--out", type=Path, default=Path("results/stress_board_multiseed.json"))
    exit_group = p.add_mutually_exclusive_group()
    exit_group.add_argument(
        "--report-only",
        action="store_true",
        help="write and print the report but exit zero regardless of promotion-gate outcome",
    )
    exit_group.add_argument(
        "--strict-nonregression-exit",
        action="store_true",
        help="fail unless clean and worst-scenario gates also pass; intended for candidate promotion",
    )
    args = p.parse_args()
    seeds = _parse_seeds(args.seeds)
    jobs = [(s, args.wells, args.horizon, args.bootstrap_reps, args.forecast_zero_policy) for s in seeds]

    results: list[dict] = []
    if args.workers <= 1:
        for job in jobs:
            result = _run_one(job)
            results.append(result)
            print(
                f"seed={result['seed']} score_delta={result['comparison']['pooled_score_delta']:.6f} "
                f"win={result['smartcast_beats_legacy']} elapsed={result['elapsed_seconds']:.1f}s",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            future_map = {pool.submit(_run_one, job): job[0] for job in jobs}
            for future in as_completed(future_map):
                result = future.result()
                results.append(result)
                print(
                    f"seed={result['seed']} score_delta={result['comparison']['pooled_score_delta']:.6f} "
                    f"win={result['smartcast_beats_legacy']} elapsed={result['elapsed_seconds']:.1f}s",
                    flush=True,
                )
    results.sort(key=lambda r: r["seed"])
    aggregate = aggregate_stress_runs(
        results,
        min_seed_win_rate=args.min_seed_win_rate,
        max_clean_relative_regression=args.max_clean_relative_regression,
        max_worst_scenario_relative_regression=args.max_worst_scenario_relative_regression,
        bootstrap_reps=args.aggregate_bootstrap_reps,
    )
    payload = {
        "gate": "smartcast_v1_vs_legacy_arps_paired_multiseed",
        "configuration": {
            "seeds": seeds,
            "wells_per_seed": args.wells,
            "horizon": args.horizon,
            "workers": args.workers,
            "forecast_zero_policy": args.forecast_zero_policy,
            "per_seed_bootstrap_reps": args.bootstrap_reps,
            "aggregate_bootstrap_reps": args.aggregate_bootstrap_reps,
        },
        "aggregate": aggregate,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    if args.report_only:
        raise SystemExit(0)
    passed = (
        aggregate["strict_nonregression_gate_pass"]
        if args.strict_nonregression_exit
        else aggregate["champion_gate_pass"]
    )
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
