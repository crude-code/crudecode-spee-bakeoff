"""Forecast providers — the seam that lets externally-generated forecasts
(an LLM, a vendor tool, a human engineer) compete against the in-repo model
functions under identical scoring.

A provider is any callable: (split, phase) -> np.ndarray of length
`horizon`, or None when it has no forecast for that well/phase. None is a
loud skip — the benchmark reports it, never silently fills it.

Two adapters cover the current cases:
- from_model_fn wraps the existing (train_array -> forecast_fn) models.
- precomputed wraps a {well_id: {phase: [monthly volumes]}} mapping, which
  is how offline forecasts arrive (e.g. one JSON file per pad from the LLM
  arm). Precomputed volumes are trusted as-is; the provider only checks
  shape, because the whole point is that the scorer never re-derives
  anyone's curve math.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from forecast_benchmark.data import PHASES
from forecast_benchmark.split import Split

ForecastProvider = Callable[[Split, str], "np.ndarray | None"]


def from_model_fn(model_fn) -> ForecastProvider:
    """Adapt a (train_array -> forecast_fn) model to the provider seam."""
    def _provider(split: Split, phase: str) -> np.ndarray | None:
        train_arr = getattr(split.train, phase)
        forecast_fn = model_fn(train_arr)
        return forecast_fn(len(split.holdout_months))
    return _provider


def precomputed(forecasts: dict[str, dict[str, list[float]]]) -> ForecastProvider:
    """Adapt {well_id: {phase: [monthly volumes]}} to the provider seam.

    Raises on a length mismatch instead of truncating or padding: a
    forecast that doesn't cover the scored window is a broken submission,
    not a shorter one.
    """
    def _provider(split: Split, phase: str) -> np.ndarray | None:
        well_forecasts = forecasts.get(split.well_id)
        if well_forecasts is None or phase not in well_forecasts:
            return None
        arr = np.asarray(well_forecasts[phase], dtype=float)
        if len(arr) != len(split.holdout_months):
            raise ValueError(
                f"{split.well_id}/{phase}: forecast has {len(arr)} months, "
                f"scored window has {len(split.holdout_months)}"
            )
        return arr
    return _provider


def load_precomputed_dir(path: str | Path) -> dict[str, dict[str, list[float]]]:
    """Merge every *.json in a directory into one well_id -> phase -> volumes
    mapping. Files are typically one-per-pad: {"forecasts": {well_id:
    {"oil": [...], "gas": [...]}, ...}} — anything outside "forecasts"
    (params, rationale) is audit trail and ignored here. A bare mapping
    without the "forecasts" wrapper is accepted too. Duplicate well_ids
    across files raise: two files claiming the same well is a submission
    bug worth stopping on.
    """
    merged: dict[str, dict[str, list[float]]] = {}
    for f in sorted(Path(path).glob("*.json")):
        payload = json.loads(f.read_text())
        wells = payload.get("forecasts", payload)
        for well_id, phase_map in wells.items():
            if well_id in merged:
                raise ValueError(f"{f.name}: well {well_id} already provided by another file")
            merged[well_id] = {
                phase: list(map(float, vols))
                for phase, vols in phase_map.items()
                if phase in PHASES
            }
    return merged
