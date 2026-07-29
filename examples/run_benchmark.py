"""Run the 1v1: deterministic Arps vs the LLM, on a pad snapshot.

Usage:
    python examples/run_benchmark.py [data_dir] [--llm-dir NAME]

The deterministic arm fits live from each well's training history. The LLM
arm is read from <data_dir>/<llm-dir>/*.json, one file per pad, shaped
{"forecasts": {well_id: {"oil": [...12], "gas": [...12]}}}. Both go through
the identical provider seam — same splits, same metrics, same code path.

Writes the scored runs to stdout as JSON. There is no formatting layer —
consumers decide how to present it.
"""
from __future__ import annotations

# Allow direct execution from a fresh checkout without requiring an editable install.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath

_BOOTSTRAP_SRC = _BootstrapPath(__file__).resolve().parents[1] / "src"
if str(_BOOTSTRAP_SRC) not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, str(_BOOTSTRAP_SRC))

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from forecast_benchmark.arps import arps_hyperbolic_bounded_b
from forecast_benchmark.benchmark import run_provider_benchmark
from forecast_benchmark.pads import load_pads, run_pad_benchmark
from forecast_benchmark.providers import from_model_fn, load_precomputed_dir, precomputed

DETERMINISTIC_ARM = "arps_bounded_b"
LLM_ARM = "llm"


def build_arms(data_dir: Path, llm_dir: str) -> list[tuple[str, object]]:
    """The board: exactly one deterministic arm, plus the LLM if its
    forecasts are on disk. Deliberately not a glob over every
    llm_forecasts* directory — a 1v1 with five silent contenders is not a
    1v1, and which run is being scored should be an explicit argument."""
    arms: list[tuple[str, object]] = [
        (DETERMINISTIC_ARM, from_model_fn(arps_hyperbolic_bounded_b)),
    ]
    llm_path = data_dir / llm_dir
    if llm_path.is_dir() and any(llm_path.glob("*.json")):
        arms.append((LLM_ARM, precomputed(load_precomputed_dir(llm_path))))
    else:
        print(f"note: no LLM forecasts at {llm_path} — scoring deterministic arm only",
              file=sys.stderr)
    return arms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", nargs="?", default="benchmark_data",
                        help="snapshot directory (default: benchmark_data)")
    parser.add_argument("--llm-dir", default="llm_forecasts",
                        help="subdirectory of data_dir holding LLM forecast JSON")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        sys.exit(f"no snapshot at {data_dir} — build one with scripts/extract_pads.py")

    snapshot = json.loads((data_dir / "snapshot.json").read_text())
    cutoff, horizon = snapshot["cutoff"], snapshot["horizon"]
    pads = load_pads(data_dir)
    wells = [w for p in pads for w in p.wells]
    arms = build_arms(data_dir, args.llm_dir)

    results = {
        "per_well": [
            asdict(run_provider_benchmark(wells, provider, model_name=name,
                                          cutoff_date=cutoff, horizon=horizon))
            for name, provider in arms
        ],
        "pad_level": [
            asdict(run_pad_benchmark(pads, provider, model_name=name,
                                     cutoff_date=cutoff, horizon=horizon))
            for name, provider in arms
        ],
    }
    json.dump(results, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
