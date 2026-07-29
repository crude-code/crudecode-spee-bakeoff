#!/usr/bin/env python3
"""Evaluate the experiment-only allocation-noise classifier across seeds."""
from __future__ import annotations

import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_SRC = _BootstrapPath(__file__).resolve().parents[1] / "src"
if str(_BOOTSTRAP_SRC) not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, str(_BOOTSTRAP_SRC))

import argparse
import json
from pathlib import Path

import numpy as np

from forecast_benchmark.experiments.allocation_noise import assess_allocation_noise
from forecast_benchmark.stressboard import make_stress_cases


def _parse_seeds(raw: str) -> list[int]:
    out: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if "-" in token:
            a, b = map(int, token.split("-", 1))
            out.extend(range(a, b + 1))
        elif token:
            out.append(int(token))
    return sorted(set(out))


def _safe_div(a: int, b: int) -> float | None:
    return float(a / b) if b else None


def run_seed(seed: int, wells: int) -> dict:
    cases = make_stress_cases(seed, n_wells=wells, horizon=12)
    tp = fp = tn = fn = 0
    scenario_counts: dict[str, dict[str, int]] = {}
    for case in cases:
        assessment = assess_allocation_noise(case.train)
        actual = case.scenario == "allocation_noise"
        predicted = assessment.flagged
        if actual and predicted:
            tp += 1
        elif actual and not predicted:
            fn += 1
        elif not actual and predicted:
            fp += 1
        else:
            tn += 1
        row = scenario_counts.setdefault(case.scenario, {"n": 0, "flagged": 0})
        row["n"] += 1
        row["flagged"] += int(predicted)
    return {
        "seed": seed,
        "n_wells": wells,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),
        "false_positive_rate": _safe_div(fp, fp + tn),
        "scenario_flag_rates": {
            name: count["flagged"] / count["n"] for name, count in sorted(scenario_counts.items())
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default="1-50")
    p.add_argument("--wells", type=int, default=180)
    p.add_argument("--min-precision", type=float, default=0.90)
    p.add_argument("--max-clean-fpr", type=float, default=0.05)
    p.add_argument("--out", type=Path, default=Path("results/allocation_noise_classifier.json"))
    args = p.parse_args()
    seeds = _parse_seeds(args.seeds)
    results = [run_seed(seed, args.wells) for seed in seeds]
    tp = sum(r["tp"] for r in results)
    fp = sum(r["fp"] for r in results)
    tn = sum(r["tn"] for r in results)
    fn = sum(r["fn"] for r in results)
    scenario_names = sorted({name for r in results for name in r["scenario_flag_rates"]})
    scenario_summary = {
        name: {
            "mean_flag_rate": float(np.mean([r["scenario_flag_rates"][name] for r in results])),
            "stdev_flag_rate": float(np.std([r["scenario_flag_rates"][name] for r in results])),
        }
        for name in scenario_names
    }
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    fpr = _safe_div(fp, fp + tn)
    clean_fpr = scenario_summary.get("clean", {}).get("mean_flag_rate")
    gates = {
        "precision": precision is not None and precision >= args.min_precision,
        "clean_false_positive_rate": clean_fpr is not None and clean_fpr <= args.max_clean_fpr,
    }
    payload = {
        "experiment": "allocation_noise_classifier_only_no_forecast_changes",
        "configuration": {"seeds": seeds, "wells_per_seed": args.wells},
        "aggregate": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "false_positive_rate": fpr,
            "scenario_summary": scenario_summary,
            "gates": gates,
            "classifier_gate_pass": all(gates.values()),
        },
        "results": results,
        "decision": "Classifier evidence only. No re-anchoring or forecast path is promoted by this script.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))
    raise SystemExit(0 if all(gates.values()) else 2)


if __name__ == "__main__":
    main()
