"""SPEE bake-off submission utilities.

This module is intentionally format-light. The committee's final schema is
not known yet, so the repo keeps one public-safe internal schema:

- history.csv: well_id, month, oil, gas, water
- wells.csv: optional well metadata; at minimum well_id, optionally pad_id,
  basin, operator, api/uwi, lateral_length_ft, etc.

Submission writers are deterministic and auditable. They do not score; they
only create forecast rows and failure reports for Strict Auto / Vendor Best
runs. Scoring stays in benchmark.py/pads.py for holdout snapshots.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable

import numpy as np

from forecast_benchmark.data import PHASES, WellSeries, load_csv
from forecast_benchmark.providers import ForecastProvider
from forecast_benchmark.split import Split


@dataclass(frozen=True)
class ForecastRow:
    submission_type: str
    model_name: str
    well_id: str
    pad_id: str
    phase: str
    forecast_month: str
    forecast_volume: float


@dataclass(frozen=True)
class FailureRow:
    well_id: str
    pad_id: str
    phase: str
    reason: str


@dataclass(frozen=True)
class SubmissionResult:
    rows: list[ForecastRow]
    failures: list[FailureRow]
    n_wells: int
    horizon: int
    forecast_start: str | None


@dataclass(frozen=True)
class Timer:
    started: float

    @classmethod
    def start(cls) -> "Timer":
        return cls(started=perf_counter())

    def elapsed_seconds(self) -> float:
        return round(perf_counter() - self.started, 3)


def add_month(month: str, n: int = 1) -> str:
    """Add n calendar months to an ISO YYYY-MM-01 month string."""
    year, mon = int(month[:4]), int(month[5:7])
    idx = year * 12 + (mon - 1) + n
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}-01"


def months_between(start: str, end: str) -> int:
    """Number of calendar months from start to end. end must be >= start."""
    sy, sm = int(start[:4]), int(start[5:7])
    ey, em = int(end[:4]), int(end[5:7])
    return (ey - sy) * 12 + (em - sm)


def forecast_months_after(last_month: str, horizon: int) -> list[str]:
    return [add_month(last_month, i) for i in range(1, horizon + 1)]


def forecast_months_from(start_month: str, horizon: int) -> list[str]:
    return [add_month(start_month, i) for i in range(horizon)]


def load_well_metadata(input_dir: str | Path) -> dict[str, dict[str, str]]:
    """Load optional wells.csv metadata. Missing file is fine."""
    input_dir = Path(input_dir)
    path = input_dir / "wells.csv"
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "well_id" not in reader.fieldnames:
            raise ValueError("wells.csv must include well_id")
        return {row["well_id"]: row for row in reader}


def load_submission_wells(
    input_dir: str | Path, *, gap_policy: str | None = None
) -> tuple[list[WellSeries], dict[str, dict[str, str]]]:
    """Load wells for a forecast-only SPEE run.

    Preferred input is canonical SPEE mode: history.csv + optional wells.csv.
    For convenience, a benchmark snapshot with train/*.csv is also accepted;
    only train files are loaded, never holdout files.
    """
    input_dir = Path(input_dir)
    history = input_dir / "history.csv"
    if history.exists():
        # Submission data must not fail merely because a source omitted an
        # empty calendar row.  The policy is explicit in manifest.json when
        # prepared by this repo; otherwise NaN is the conservative default
        # (missing report is not a reported zero).
        policy = gap_policy
        manifest_path = input_dir / "manifest.json"
        if policy is None and manifest_path.exists():
            try:
                policy = json.loads(manifest_path.read_text()).get("gap_policy")
            except (OSError, json.JSONDecodeError):
                policy = None
        return load_csv(str(history), gap_policy=policy or "nan"), load_well_metadata(input_dir)

    train_dir = input_dir / "train"
    pads_file = input_dir / "pads.json"
    if train_dir.is_dir() and pads_file.exists():
        wells: list[WellSeries] = []
        metadata: dict[str, dict[str, str]] = {}
        pads = json.loads(pads_file.read_text())
        for pad in pads:
            pad_id = str(pad["pad_id"])
            for well in load_csv(str(train_dir / f"{pad_id}.csv")):
                wells.append(well)
                metadata[well.well_id] = {"well_id": well.well_id, "pad_id": pad_id}
        return wells, metadata

    raise FileNotFoundError(
        f"{input_dir} is not a SPEE input directory. Expected history.csv, "
        "or train/*.csv plus pads.json."
    )


def _fake_split(well: WellSeries, target_months: list[str]) -> Split:
    return Split(
        well_id=well.well_id,
        train=well,
        holdout_months=target_months,
        holdout_actuals={phase: np.full(len(target_months), np.nan) for phase in PHASES},
    )


def _target_months_and_offset(well: WellSeries, horizon: int, forecast_start: str | None) -> tuple[list[str], int]:
    natural_start = add_month(well.months[-1], 1)
    if forecast_start is None:
        return forecast_months_after(well.months[-1], horizon), 0
    gap = months_between(natural_start, forecast_start)
    if gap < 0:
        raise ValueError(
            f"forecast_start {forecast_start} is not after {well.well_id}'s last history month {well.months[-1]}"
        )
    return forecast_months_from(forecast_start, horizon), gap


def generate_submission(
    wells: list[WellSeries],
    provider: ForecastProvider,
    *,
    submission_type: str,
    model_name: str,
    horizon: int,
    metadata: dict[str, dict[str, str]] | None = None,
    forecast_start: str | None = None,
) -> SubmissionResult:
    """Generate forecast rows for a SPEE-style forecast-only submission.

    If forecast_start is provided, every well is forecast to the same target
    calendar window. Wells whose last history month is older than the common
    start are forecast through the gap and sliced to the target window. This
    keeps the output schema compatible with a standardized bake-off horizon
    while still respecting each well's actual visible history.
    """
    metadata = metadata or {}
    rows: list[ForecastRow] = []
    failures: list[FailureRow] = []

    for well in wells:
        pad_id = metadata.get(well.well_id, {}).get("pad_id", well.well_id)
        try:
            target_months, gap = _target_months_and_offset(well, horizon, forecast_start)
        except Exception as exc:  # noqa: BLE001 - turned into a failure row
            for phase in PHASES:
                failures.append(FailureRow(well.well_id, pad_id, phase, f"date_error: {exc}"))
            continue

        provider_months = target_months if gap == 0 else forecast_months_after(well.months[-1], horizon + gap)
        split = _fake_split(well, provider_months)

        for phase in PHASES:
            if not well.phase_available(phase):
                failures.append(FailureRow(well.well_id, pad_id, phase, "phase_unavailable"))
                continue
            try:
                forecast = provider(split, phase)
            except Exception as exc:  # noqa: BLE001 - forecast failure is reportable, not fatal
                failures.append(FailureRow(well.well_id, pad_id, phase, f"provider_error: {exc}"))
                continue
            if forecast is None:
                failures.append(FailureRow(well.well_id, pad_id, phase, "forecast_missing"))
                continue
            arr = np.asarray(forecast, dtype=float)
            if gap:
                arr = arr[gap: gap + horizon]
            if len(arr) != horizon:
                failures.append(FailureRow(well.well_id, pad_id, phase, f"length_mismatch:{len(arr)}"))
                continue
            if np.any(~np.isfinite(arr)) or np.any(arr < 0):
                failures.append(FailureRow(well.well_id, pad_id, phase, "nonfinite_or_negative_forecast"))
                continue
            rows.extend(
                ForecastRow(
                    submission_type=submission_type,
                    model_name=model_name,
                    well_id=well.well_id,
                    pad_id=pad_id,
                    phase=phase,
                    forecast_month=month,
                    forecast_volume=round(float(value), 6),
                )
                for month, value in zip(target_months, arr)
            )

    return SubmissionResult(
        rows=rows,
        failures=failures,
        n_wells=len(wells),
        horizon=horizon,
        forecast_start=forecast_start,
    )


def write_submission_csv(result: SubmissionResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ForecastRow.__dataclass_fields__)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in result.rows:
            writer.writerow(asdict(row))


def write_failures_csv(result: SubmissionResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(FailureRow.__dataclass_fields__)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in result.failures:
            writer.writerow(asdict(row))


def write_failure_markdown(result: SubmissionResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    by_reason: dict[str, int] = {}
    for failure in result.failures:
        by_reason[failure.reason] = by_reason.get(failure.reason, 0) + 1
    lines = [
        "# SPEE forecast failure report",
        "",
        f"Wells loaded: {result.n_wells}",
        f"Forecast rows written: {len(result.rows)}",
        f"Failure rows: {len(result.failures)}",
        "",
        "## Failure reasons",
        "",
        "| reason | count |",
        "|---|---:|",
    ]
    for reason, count in sorted(by_reason.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"| `{reason}` | {count} |")
    if not by_reason:
        lines.append("| none | 0 |")
    path.write_text("\n".join(lines) + "\n")


def write_run_log(
    result: SubmissionResult,
    path: str | Path,
    *,
    elapsed_seconds: float,
    command: str,
    model_name: str,
    input_dir: str | Path,
    output_csv: str | Path,
    extra: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "command": command,
        "model_name": model_name,
        "input_dir": str(input_dir),
        "output_csv": str(output_csv),
        "n_wells": result.n_wells,
        "horizon": result.horizon,
        "forecast_start": result.forecast_start,
        "forecast_rows": len(result.rows),
        "failure_rows": len(result.failures),
        "elapsed_seconds": elapsed_seconds,
        "schema": "crudecode_spee_submission_v0",
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
