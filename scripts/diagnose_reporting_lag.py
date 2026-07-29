"""Diagnose possible recent-month reporting lag in a benchmark snapshot.

This does not prove lag. It measures whether the freshest reported months sit
systematically below the fitted capacity curve across wells/phases. A broad dip
in months 1-3 before cutoff is exactly the failure mode that can bias the Arps
fit and the uptime haircut downward.

Usage:
    python scripts/diagnose_reporting_lag.py benchmark_data
    python scripts/diagnose_reporting_lag.py benchmark_data --months-back 9
    python scripts/diagnose_reporting_lag.py benchmark_data --json
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
from pathlib import Path
from typing import Any

import numpy as np

from forecast_benchmark.arps import trailing_capacity_ratios
from forecast_benchmark.data import PHASES
from forecast_benchmark.pads import load_pads
from forecast_benchmark.split import make_split_at_date


def diagnose_reporting_lag(data_dir: str | Path, *, months_back: int = 6) -> dict[str, Any]:
    data_dir = Path(data_dir)
    snapshot_path = data_dir / "snapshot.json"
    if not snapshot_path.exists():
        raise FileNotFoundError(
            f"no snapshot.json at {snapshot_path}; build a snapshot with scripts/extract_pads.py"
        )

    snapshot = json.loads(snapshot_path.read_text())
    cutoff, horizon = snapshot["cutoff"], int(snapshot["horizon"])
    pads = load_pads(data_dir)

    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for pad in pads:
        for well in pad.wells:
            split = make_split_at_date(well, cutoff_date=cutoff, horizon=horizon)
            if split is None:
                skipped.append(well.well_id)
                continue
            for phase in PHASES:
                arr = getattr(split.train, phase)
                if not split.train.phase_available(phase):
                    continue
                for item in trailing_capacity_ratios(arr, months_back=months_back):
                    rows.append({
                        "pad_id": pad.pad_id,
                        "well_id": well.well_id,
                        "phase": phase,
                        **item,
                    })

    return {
        "data_dir": str(data_dir),
        "cutoff": cutoff,
        "horizon": horizon,
        "months_back": months_back,
        "n_pads": len(pads),
        "n_rows": len(rows),
        "n_skipped_wells": len(set(skipped)),
        "skipped_well_ids": sorted(set(skipped)),
        "summary": _summarize(rows),
        "rows": rows,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float | int | None]]]:
    summary: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for phase in PHASES:
        summary[phase] = {}
        phase_rows = [r for r in rows if r["phase"] == phase]
        offsets = sorted({int(r["months_before_cutoff"]) for r in phase_rows})
        for offset in offsets:
            ratios = [float(r["ratio"]) for r in phase_rows if int(r["months_before_cutoff"]) == offset]
            if ratios:
                summary[phase][str(offset)] = {
                    "n": len(ratios),
                    "mean_ratio": float(np.mean(ratios)),
                    "median_ratio": float(np.median(ratios)),
                    "p25_ratio": float(np.percentile(ratios, 25)),
                    "p75_ratio": float(np.percentile(ratios, 75)),
                }
            else:
                summary[phase][str(offset)] = {
                    "n": 0,
                    "mean_ratio": None,
                    "median_ratio": None,
                    "p25_ratio": None,
                    "p75_ratio": None,
                }
    return summary


def markdown_summary(results: dict[str, Any]) -> str:
    lines = [
        "# Reporting-lag diagnostic",
        "",
        f"Snapshot: `{results['data_dir']}`",
        f"Cutoff: `{results['cutoff']}`; horizon: `{results['horizon']}` months",
        f"Pads: **{results['n_pads']}**; ratio rows: **{results['n_rows']}**",
        "",
        "Ratio = reported volume / fitted capacity curve for the training months nearest the cutoff.",
        "A systematic dip in offsets 1-3 is evidence to test lag-skip Arps variants; it is not proof by itself.",
        "",
    ]
    for phase in PHASES:
        rows = results["summary"].get(phase, {})
        if not rows:
            continue
        lines.extend([
            f"## {phase}",
            "",
            "| months before cutoff | n | mean ratio | median ratio | p25 | p75 |",
            "|---:|---:|---:|---:|---:|---:|",
        ])
        for offset in sorted(rows, key=lambda x: int(x)):
            item = rows[offset]
            lines.append(
                "| {offset} | {n} | {mean} | {median} | {p25} | {p75} |".format(
                    offset=offset,
                    n=item["n"],
                    mean=_fmt(item["mean_ratio"]),
                    median=_fmt(item["median_ratio"]),
                    p25=_fmt(item["p25_ratio"]),
                    p75=_fmt(item["p75_ratio"]),
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
    parser.add_argument("--months-back", type=int, default=6)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    try:
        results = diagnose_reporting_lag(args.data_dir, months_back=args.months_back)
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
