"""Well production data types and explicit calendar-gap handling."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PHASES = ("oil", "gas", "water")


@dataclass(frozen=True)
class WellSeries:
    """One well's monthly production history in contiguous calendar order.

    Arrays align to ``months``.  NaN means not reported; zero means reported
    zero.  Ingestion decides how source gaps are represented and records that
    policy in the input manifest.
    """

    well_id: str
    months: list[str]
    oil: np.ndarray
    gas: np.ndarray
    water: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.months)
        for name in PHASES:
            arr = getattr(self, name)
            if len(arr) != n:
                raise ValueError(f"{self.well_id}: {name} length {len(arr)} != months length {n}")

    def phase_available(self, phase: str) -> bool:
        arr = getattr(self, phase)
        return bool(np.any(~np.isnan(arr)))

    def truncate(self, n_months: int) -> "WellSeries":
        if n_months > len(self.months):
            raise ValueError(f"{self.well_id}: cannot truncate to {n_months}, only {len(self.months)} available")
        return WellSeries(
            well_id=self.well_id,
            months=self.months[:n_months],
            oil=self.oil[:n_months],
            gas=self.gas[:n_months],
            water=self.water[:n_months],
        )


def load_csv(path: str, *, gap_policy: str = "error", duplicate_policy: str = "error") -> list[WellSeries]:
    """Load long-format ``well_id,month,oil,gas,water`` production data.

    ``gap_policy`` is one of:

    * ``error``: reject missing calendar rows (benchmark/research default),
    * ``nan``: insert missing rows with NaN values (submission default),
    * ``zero``: insert missing rows with reported zeros.

    Duplicate well-month rows are rejected by default.  ``duplicate_policy``
    may be ``sum`` only when the source is known to contain additive records.
    No policy is silent.
    """
    import csv as csv_mod
    from collections import defaultdict

    if gap_policy not in {"error", "nan", "zero"}:
        raise ValueError(f"unsupported gap_policy: {gap_policy}")
    if duplicate_policy not in {"error", "sum"}:
        raise ValueError(f"unsupported duplicate_policy: {duplicate_policy}")

    rows_by_well: dict[str, list[dict]] = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv_mod.DictReader(f)
        if not reader.fieldnames or "well_id" not in reader.fieldnames or "month" not in reader.fieldnames:
            raise ValueError("history CSV must include well_id and month")
        for row in reader:
            rows_by_well[row["well_id"]].append(row)

    wells = []
    for well_id, rows in rows_by_well.items():
        rows.sort(key=lambda r: r["month"])
        collapsed: list[dict] = []
        for row in rows:
            if collapsed and row["month"] == collapsed[-1]["month"]:
                if duplicate_policy == "error":
                    raise ValueError(f"{well_id}: duplicate month {row['month']}")
                for phase in PHASES:
                    a = _to_float(collapsed[-1].get(phase))
                    b = _to_float(row.get(phase))
                    if np.isnan(a) and np.isnan(b):
                        collapsed[-1][phase] = ""
                    else:
                        collapsed[-1][phase] = str(float(np.nansum([a, b])))
            else:
                collapsed.append(dict(row))
        expanded = _expand_gaps(well_id, collapsed, gap_policy)
        months = [r["month"] for r in expanded]
        oil = np.array([_to_float(r.get("oil")) for r in expanded])
        gas = np.array([_to_float(r.get("gas")) for r in expanded])
        water = np.array([_to_float(r.get("water")) for r in expanded])
        wells.append(WellSeries(well_id=well_id, months=months, oil=oil, gas=gas, water=water))
    return wells


def _to_float(v: str | None) -> float:
    if v is None or v == "":
        return float("nan")
    return float(v)


def _month_index(month: str) -> int:
    return int(month[:4]) * 12 + int(month[5:7]) - 1


def _month_from_index(idx: int) -> str:
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}-01"


def _expand_gaps(well_id: str, rows: list[dict], gap_policy: str) -> list[dict]:
    if not rows:
        return []
    out = [rows[0]]
    fill = "" if gap_policy == "nan" else "0"
    for row in rows[1:]:
        prev_idx = _month_index(out[-1]["month"])
        idx = _month_index(row["month"])
        if idx <= prev_idx:
            raise ValueError(f"{well_id}: disorder between {out[-1]['month']} and {row['month']}")
        if idx > prev_idx + 1:
            if gap_policy == "error":
                raise ValueError(
                    f"{well_id}: gap between {out[-1]['month']} and {row['month']} — choose an explicit gap policy"
                )
            for missing in range(prev_idx + 1, idx):
                out.append({"well_id": well_id, "month": _month_from_index(missing), "oil": fill, "gas": fill, "water": fill})
        out.append(row)
    return out


def _assert_contiguous(well_id: str, months: list[str]) -> None:
    """Backward-compatible helper retained for tests and external callers."""
    for a, b in zip(months, months[1:]):
        if _month_index(b) != _month_index(a) + 1:
            raise ValueError(f"{well_id}: gap or disorder between {a} and {b} — this loader does not fill gaps")
