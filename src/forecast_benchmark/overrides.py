"""Audited Vendor Best override handling.

Overrides are intentionally narrow: a named approver may replace one
well/phase trajectory with a complete, non-negative vector and a reason.
There are no silent multipliers or partial arrays.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Callable

import numpy as np

from forecast_benchmark.data import PHASES
from forecast_benchmark.split import Split


@dataclass(frozen=True)
class OverrideAudit:
    approver: str
    approved_at: str
    source: str
    count: int


def load_overrides(path: str | Path, *, horizon: int) -> tuple[dict[tuple[str, str], np.ndarray], OverrideAudit]:
    path = Path(path)
    payload = json.loads(path.read_text())
    approver = str(payload.get("approver", "")).strip()
    approved_at = str(payload.get("approved_at", "")).strip()
    if not approver:
        raise ValueError("override package requires non-empty approver")
    if not approved_at:
        raise ValueError("override package requires approved_at")
    try:
        datetime.fromisoformat(approved_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("approved_at must be ISO-8601") from exc

    out: dict[tuple[str, str], np.ndarray] = {}
    rows = payload.get("overrides")
    if not isinstance(rows, list):
        raise ValueError("overrides must be a list")
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"override {idx} must be an object")
        well_id = str(row.get("well_id", "")).strip()
        phase = str(row.get("phase", "")).strip().lower()
        reason = str(row.get("reason", "")).strip()
        values = row.get("values")
        if not well_id or phase not in PHASES or not reason:
            raise ValueError(f"override {idx}: well_id, valid phase, and reason are required")
        key = (well_id, phase)
        if key in out:
            raise ValueError(f"duplicate override for {well_id}/{phase}")
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 1 or len(arr) != horizon:
            raise ValueError(f"override {well_id}/{phase}: expected {horizon} values")
        if np.any(~np.isfinite(arr)) or np.any(arr < 0):
            raise ValueError(f"override {well_id}/{phase}: values must be finite and non-negative")
        out[key] = arr
    return out, OverrideAudit(approver=approver, approved_at=approved_at, source=str(path), count=len(out))


def with_overrides(base_provider: Callable, overrides: dict[tuple[str, str], np.ndarray]) -> Callable:
    def provider(split: Split, phase: str) -> np.ndarray | None:
        key = (split.well_id, phase)
        if key in overrides:
            arr = overrides[key]
            if len(arr) != len(split.holdout_months):
                raise ValueError(f"override {split.well_id}/{phase} does not match requested horizon")
            return arr.copy()
        return base_provider(split, phase)
    return provider
