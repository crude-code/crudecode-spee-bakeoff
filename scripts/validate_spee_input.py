#!/usr/bin/env python3
"""Validate SPEE canonical input and, optionally, a generated submission CSV."""
from __future__ import annotations

# Allow direct execution from a fresh checkout without requiring an editable install.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath

_BOOTSTRAP_SRC = _BootstrapPath(__file__).resolve().parents[1] / "src"
if str(_BOOTSTRAP_SRC) not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, str(_BOOTSTRAP_SRC))

import argparse
import csv
import json
import sys
from pathlib import Path

from forecast_benchmark.data import PHASES
from forecast_benchmark.spee import load_submission_wells

REQUIRED_SUBMISSION_COLUMNS = {
    "submission_type", "model_name", "well_id", "pad_id", "phase", "forecast_month", "forecast_volume"
}


def validate_input(input_dir: Path) -> dict:
    wells, meta = load_submission_wells(input_dir)
    if not wells:
        raise ValueError("no wells loaded")
    duplicate_ids = len({w.well_id for w in wells}) != len(wells)
    if duplicate_ids:
        raise ValueError("duplicate well_id values loaded")
    phase_counts = {
        phase: sum(1 for well in wells if well.phase_available(phase))
        for phase in PHASES
    }
    return {
        "input_dir": str(input_dir),
        "n_wells": len(wells),
        "n_metadata_rows": len(meta),
        "phase_available": phase_counts,
    }


def validate_submission(path: Path, *, horizon: int | None = None) -> dict:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("submission has no header row")
        missing = REQUIRED_SUBMISSION_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"submission missing required columns: {sorted(missing)}")
        rows = list(reader)
    bad = []
    well_phase_counts: dict[tuple[str, str], int] = {}
    for i, row in enumerate(rows, start=2):
        phase = row["phase"]
        if phase not in PHASES:
            bad.append(f"line {i}: invalid phase {phase!r}")
        try:
            volume = float(row["forecast_volume"])
        except ValueError:
            bad.append(f"line {i}: forecast_volume is not numeric")
            continue
        if volume < 0:
            bad.append(f"line {i}: negative forecast_volume")
        well_phase_counts[(row["well_id"], phase)] = well_phase_counts.get((row["well_id"], phase), 0) + 1
    if horizon is not None:
        wrong = {key: n for key, n in well_phase_counts.items() if n != horizon}
        if wrong:
            first = list(wrong.items())[:5]
            bad.append(f"{len(wrong)} well/phase groups do not have horizon={horizon}; examples={first}")
    if bad:
        raise ValueError("; ".join(bad[:10]))
    return {
        "submission": str(path),
        "n_rows": len(rows),
        "n_well_phase_groups": len(well_phase_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", nargs="?", default="spee_data", type=Path)
    parser.add_argument("--submission", type=Path, help="optional generated submission CSV to validate")
    parser.add_argument("--horizon", type=int, default=None, help="expected rows per well/phase in submission")
    args = parser.parse_args()
    try:
        result = {"input": validate_input(args.input_dir)}
        if args.submission:
            result["submission"] = validate_submission(args.submission, horizon=args.horizon)
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"validate_spee_input failed: {exc}")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
