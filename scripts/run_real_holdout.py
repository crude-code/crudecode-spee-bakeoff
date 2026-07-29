#!/usr/bin/env python3
"""Run a leakage-controlled retrospective holdout on canonical real-well data."""
from __future__ import annotations

import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_SRC = _BootstrapPath(__file__).resolve().parents[1] / "src"
if str(_BOOTSTRAP_SRC) not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, str(_BOOTSTRAP_SRC))

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from forecast_benchmark.real_holdout import choose_retrospective_cutoff, run_retrospective_holdout
from forecast_benchmark.spee import load_submission_wells


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_dir", type=Path, help="canonical directory containing history.csv and optional wells.csv")
    p.add_argument("--cutoff-date", help="common YYYY-MM-01 cutoff; auto-selected when omitted")
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--stabilization-lag", type=int, default=6)
    p.add_argument("--min-history", type=int, default=12)
    p.add_argument("--bootstrap-reps", type=int, default=2000)
    p.add_argument("--forecast-zero-policy", choices=("drop", "epsilon", "error"), default="epsilon")
    p.add_argument("--source-name", required=True)
    p.add_argument("--source-url", default="")
    p.add_argument("--snapshot-date", required=True, help="date the source files were downloaded, YYYY-MM-DD")
    p.add_argument("--data-as-of", default="", help="latest production month represented, when known")
    p.add_argument("--out", type=Path, default=Path("results/real_holdout.json"))
    args = p.parse_args()

    wells, metadata = load_submission_wells(args.input_dir)
    cutoff = args.cutoff_date or choose_retrospective_cutoff(
        wells,
        horizon=args.horizon,
        stabilization_lag=args.stabilization_lag,
    )
    evaluation = run_retrospective_holdout(
        wells,
        metadata,
        cutoff_date=cutoff,
        horizon=args.horizon,
        min_history=args.min_history,
        forecast_zero_policy=args.forecast_zero_policy,
        bootstrap_reps=args.bootstrap_reps,
    )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_attestation": {
            "source_name": args.source_name,
            "source_url": args.source_url,
            "snapshot_date": args.snapshot_date,
            "data_as_of": args.data_as_of,
        },
        "stabilization_lag_months": args.stabilization_lag,
        "evaluation": evaluation,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    passed = evaluation["n_scored_series"] > 0 and evaluation["failure_count"] == 0
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
