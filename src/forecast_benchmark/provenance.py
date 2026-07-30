"""Production-provenance helpers for private real-well validation boards.

These helpers classify warehouse metadata.  They are intentionally isolated
from the forecasting package: production provenance and producing-day fields
must never become model inputs for the SPEE submission unless the committee
provides equivalent columns.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

KNOWN_METHODS = frozenset({"REPORTED", "DCA"})
PROVENANCE_MODES = frozenset({
    "mixed",
    "reported_holdout",
    "reported_full",
    "reported_threshold",
})


def normalize_method(value: object) -> str:
    """Normalize a warehouse reporting-method value without inventing meaning."""
    if value is None:
        return "UNKNOWN"
    text = str(value).strip().upper()
    return text or "UNKNOWN"


def classify_month_methods(values: Iterable[object]) -> str:
    """Classify all source rows contributing to one well-calendar-month.

    A month with both REPORTED and DCA source rows is MIXED.  Unknown labels are
    retained as UNKNOWN unless a known label is also present, in which case the
    month is MIXED because provenance is not homogeneous.
    """
    methods = {normalize_method(value) for value in values}
    methods.discard("")
    if not methods:
        return "UNKNOWN"
    if methods == {"REPORTED"}:
        return "REPORTED"
    if methods == {"DCA"}:
        return "DCA"
    if len(methods) == 1:
        return next(iter(methods))
    return "MIXED"


def provenance_counts(labels: Iterable[str]) -> dict[str, int]:
    counts = Counter(str(label).upper() for label in labels)
    return {
        "REPORTED": int(counts.get("REPORTED", 0)),
        "DCA": int(counts.get("DCA", 0)),
        "MIXED": int(counts.get("MIXED", 0)),
        "UNKNOWN": int(counts.get("UNKNOWN", 0)),
        "MISSING": int(counts.get("MISSING", 0)),
        "OTHER": int(sum(v for k, v in counts.items() if k not in {
            "REPORTED", "DCA", "MIXED", "UNKNOWN", "MISSING"
        })),
    }


def fraction(counts: dict[str, int], label: str) -> float:
    total = sum(int(v) for v in counts.values())
    return float(counts.get(label, 0) / total) if total else 0.0


@dataclass(frozen=True)
class ProvenanceGate:
    mode: str = "mixed"
    min_train_reported_fraction: float = 0.80
    min_holdout_reported_fraction: float = 1.00

    def __post_init__(self) -> None:
        if self.mode not in PROVENANCE_MODES:
            raise ValueError(f"unknown provenance mode {self.mode!r}")
        for name, value in (
            ("min_train_reported_fraction", self.min_train_reported_fraction),
            ("min_holdout_reported_fraction", self.min_holdout_reported_fraction),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")

    def accepts(self, train_labels: Iterable[str], holdout_labels: Iterable[str]) -> bool:
        train = provenance_counts(train_labels)
        hold = provenance_counts(holdout_labels)
        tr = fraction(train, "REPORTED")
        hd = fraction(hold, "REPORTED")
        if self.mode == "mixed":
            return True
        if self.mode == "reported_holdout":
            return hd >= 1.0
        if self.mode == "reported_full":
            return tr >= 1.0 and hd >= 1.0
        return tr >= self.min_train_reported_fraction and hd >= self.min_holdout_reported_fraction
