"""Compare deterministic Arps function variants on the same snapshot.

This is not the public 1v1 board. It is a zero-LLM-cost workbench for the
Python arm: same train/holdout split, same pad sums, same metrics, multiple
controlled Arps variants. Use it before deciding whether to replace the
official deterministic function.

Usage:
    python scripts/run_python_arm_experiments.py benchmark_data
    python scripts/run_python_arm_experiments.py benchmark_data --json
    python scripts/run_python_arm_experiments.py benchmark_data --out results/arps.json
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
from typing import Any

from forecast_benchmark.arps import python_arm_experiments
from forecast_benchmark.benchmark import run_provider_benchmark
from forecast_benchmark.data import PHASES
from forecast_benchmark.pads import load_pads, run_pad_benchmark
from forecast_benchmark.providers import from_model_fn

PRIMARY_METRICS = ("mape_median", "bias_median", "spee_score_median")


def run_experiments(data_dir: str | Path) -> dict[str, Any]:
    data_dir = Path(data_dir)
    snapshot_path = data_dir / "snapshot.json"
    if not snapshot_path.exists():
        raise FileNotFoundError(
            f"no snapshot.json at {snapshot_path}; build a snapshot with scripts/extract_pads.py"
        )

    snapshot = json.loads(snapshot_path.read_text())
    cutoff, horizon = snapshot["cutoff"], int(snapshot["horizon"])
    pads = load_pads(data_dir)
    wells = [w for p in pads for w in p.wells]

    arms = []
    for name, model_fn in python_arm_experiments().items():
        provider = from_model_fn(model_fn)
        per_well = run_provider_benchmark(
            wells, provider, model_name=name, cutoff_date=cutoff, horizon=horizon
        )
        pad_level = run_pad_benchmark(
            pads, provider, model_name=name, cutoff_date=cutoff, horizon=horizon
        )
        arms.append({
            "model_name": name,
            "per_well": asdict(per_well),
            "pad_level": asdict(pad_level),
        })

    return {
        "data_dir": str(data_dir),
        "cutoff": cutoff,
        "horizon": horizon,
        "n_pads": len(pads),
        "n_wells": len(wells),
        "arms": arms,
    }


def markdown_summary(results: dict[str, Any]) -> str:
    lines = [
        "# Python arm experiment board",
        "",
        f"Snapshot: `{results['data_dir']}`",
        f"Cutoff: `{results['cutoff']}`; horizon: `{results['horizon']}` months",
        f"Pads: **{results['n_pads']}**; wells: **{results['n_wells']}**",
        "",
        "Lower is better for MAPE and SPEE. Bias is signed: positive means over-forecast.",
        "",
    ]

    for level in ("pad_level", "per_well"):
        title = "Pad-level summed streams" if level == "pad_level" else "Per-well streams"
        lines.extend([f"## {title}", ""])
        for phase in PHASES:
            rows = []
            for arm in results["arms"]:
                summary = arm[level]["summary"][phase]
                if summary.get("n_scored", 0) == 0:
                    continue
                rows.append((arm["model_name"], summary))
            if not rows:
                continue
            rows.sort(key=lambda item: (
                float("inf") if item[1].get("spee_score_median") is None else item[1]["spee_score_median"],
                float("inf") if item[1].get("mape_median") is None else item[1]["mape_median"],
            ))
            lines.extend([
                f"### {phase}",
                "",
                "| rank | arm | n | bias median | MAPE median | SPEE median |",
                "|---:|---|---:|---:|---:|---:|",
            ])
            for rank, (name, summary) in enumerate(rows, start=1):
                lines.append(
                    "| {rank} | `{name}` | {n} | {bias} | {mape} | {spee} |".format(
                        rank=rank,
                        name=name,
                        n=int(summary.get("n_scored", 0)),
                        bias=_fmt(summary.get("bias_median")),
                        mape=_fmt(summary.get("mape_median")),
                        spee=_fmt(summary.get("spee_score_median")),
                    )
                )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _fmt(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{float(value):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", nargs="?", default="benchmark_data")
    parser.add_argument("--json", action="store_true", help="write full JSON instead of a Markdown summary")
    parser.add_argument("--out", type=Path, help="optional output file")
    args = parser.parse_args()

    try:
        results = run_experiments(args.data_dir)
    except FileNotFoundError as exc:
        sys.exit(str(exc))

    text = json.dumps(results, indent=1) if args.json else markdown_summary(results)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
