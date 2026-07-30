#!/usr/bin/env python3
"""Measure SmartCast throughput on a deterministic synthetic well population."""
from __future__ import annotations

import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_SRC = _BootstrapPath(__file__).resolve().parents[1] / "src"
if str(_BOOTSTRAP_SRC) not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, str(_BOOTSTRAP_SRC))

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from forecast_benchmark.data import PHASES
from forecast_benchmark.profiles import get_profile, profile_names
from forecast_benchmark.smartcast import SmartCastProvider
from forecast_benchmark.split import Split
from forecast_benchmark.stressboard import make_stress_cases


def run(n_wells: int, horizon: int, seed: int, profile: str = "scipy_control") -> dict:
    cases = make_stress_cases(seed, n_wells=n_wells, horizon=max(12, min(horizon, 24)))
    wells = [c.train for c in cases]
    metadata = {c.train.well_id: c.metadata for c in cases}
    provider = SmartCastProvider(wells, metadata, get_profile(profile))
    started = perf_counter()
    values = 0
    failures = []
    for case in cases:
        split = Split(case.train.well_id, case.train, ["future"] * horizon, {p: np.zeros(horizon) for p in PHASES})
        for phase in PHASES:
            try:
                fc = provider(split, phase)
            except Exception as exc:  # noqa: BLE001
                failures.append({"well_id": case.train.well_id, "phase": phase, "reason": str(exc)})
                continue
            if fc is None or len(fc) != horizon or np.any(~np.isfinite(fc)) or np.any(fc < 0):
                failures.append({"well_id": case.train.well_id, "phase": phase, "reason": "invalid forecast"})
            else:
                values += len(fc)
    elapsed = perf_counter() - started
    return {
        "profile": profile,
        "n_wells": n_wells,
        "horizon": horizon,
        "phases": len(PHASES),
        "forecast_values": values,
        "elapsed_seconds": round(elapsed, 3),
        "wells_per_second": round(n_wells / elapsed, 3) if elapsed else None,
        "failure_count": len(failures),
        "failures": failures[:20],
        "passed": not failures,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wells", type=int, default=1000)
    p.add_argument("--horizon", type=int, default=360)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--profile", choices=profile_names(production_only=True), default="scipy_control")
    p.add_argument("--out", type=Path, default=Path("results/throughput_test.json"))
    args = p.parse_args()
    result = run(args.wells, args.horizon, args.seed, args.profile)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 2)


if __name__ == "__main__":
    main()
