#!/usr/bin/env python3
"""Audit warehouse production provenance for an existing private board.

This script does not modify forecasts or series.csv.  It queries only the wells
already present in a private board and writes:

  provenance_by_well.csv       private well-level diagnostics
  provenance_audit.json        aggregate, shareable only after identifier review

The reporting-method label is described literally.  In particular, this tool
never equates ``DCA`` with a future forecast without the licensed vendor data
dictionary; in Texas it may denote lease-to-well allocation/estimation.
"""
from __future__ import annotations

import argparse
import calendar
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forecast_benchmark.provenance import (
    classify_month_methods,
    fraction,
    provenance_counts,
)

_TABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$")


def table_name(value: str) -> str:
    if not _TABLE_RE.fullmatch(value):
        raise ValueError(f"production table must be schema.table, got {value!r}")
    return value


def load_db_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    for parent in (Path.cwd(), *Path(__file__).resolve().parents[:3]):
        env = parent / ".env"
        if not env.exists():
            continue
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("DATABASE_URL="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("Set DATABASE_URL or provide .env with DATABASE_URL=...")


def month_start(value: date) -> date:
    return date(value.year, value.month, 1)


def add_month(value: date) -> date:
    return date(value.year + (value.month == 12), 1 if value.month == 12 else value.month + 1, 1)


def months_between(start: date, end: date) -> list[date]:
    out: list[date] = []
    cur = month_start(start)
    while cur <= month_start(end):
        out.append(cur)
        cur = add_month(cur)
    return out


def finite(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "mean_train_reported_fraction": sum(r["train_reported_fraction"] for r in rows) / len(rows),
        "mean_holdout_reported_fraction": sum(r["holdout_reported_fraction"] for r in rows) / len(rows),
        "all_holdout_reported_wells": sum(r["holdout_reported_fraction"] >= 1.0 for r in rows),
        "all_full_window_reported_wells": sum(
            r["train_reported_fraction"] >= 1.0 and r["holdout_reported_fraction"] >= 1.0
            for r in rows
        ),
        "mean_partial_producingdays_fraction": (
            sum(r["partial_producingdays_fraction"] for r in rows if math.isfinite(r["partial_producingdays_fraction"]))
            / max(1, sum(math.isfinite(r["partial_producingdays_fraction"]) for r in rows))
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="board/private-v131")
    ap.add_argument("--production-table", default="dde_remote.production")
    ap.add_argument("--chunk-size", type=int, default=25)
    ap.add_argument("--statement-timeout-ms", type=int, default=300_000)
    ap.add_argument("--out", default="board/results/provenance-audit")
    args = ap.parse_args()
    if args.chunk_size < 1:
        ap.error("--chunk-size must be positive")
    production_table = table_name(args.production_table)

    try:
        import psycopg2
    except ImportError:
        raise SystemExit("psycopg2 required: pip install psycopg2-binary")

    board = Path(args.board)
    manifest = json.loads((board / "manifest.json").read_text(encoding="utf-8"))
    cutoff = date.fromisoformat(manifest["cutoff"])
    holdout_end = date.fromisoformat(manifest["holdout_end"])
    wells: list[dict[str, str]] = []
    with (board / "wells.csv").open(encoding="utf-8") as fh:
        wells = list(csv.DictReader(fh))
    by_id = {int(row["wellid"]): row for row in wells}
    ids = sorted(by_id)

    conn = psycopg2.connect(load_db_url())
    conn.set_session(readonly=True, autocommit=False)
    cur = conn.cursor()
    cur.execute("SET LOCAL statement_timeout = %s", (args.statement_timeout_ms,))

    monthly: dict[int, dict[date, dict[str, list[Any]]]] = defaultdict(
        lambda: defaultdict(lambda: {"methods": [], "days": []})
    )
    row_methods: Counter[str] = Counter()
    row_methods_by_play: dict[str, Counter[str]] = defaultdict(Counter)
    row_days_by_method: dict[str, list[float]] = defaultdict(list)

    for start in range(0, len(ids), args.chunk_size):
        chunk = ids[start:start + args.chunk_size]
        id_sql = ",".join(str(x) for x in chunk)
        print(f"querying {start + 1}-{start + len(chunk)} / {len(ids)}", flush=True)
        cur.execute(
            f"""
            SELECT wellid, producingmonth, productionreportedmethod, producingdays
            FROM {production_table}
            WHERE wellid IN ({id_sql})
              AND producingmonth <= %s
            ORDER BY wellid, producingmonth
            """,
            (holdout_end,),
        )
        while True:
            batch = cur.fetchmany(10_000)
            if not batch:
                break
            for wid, month, method, days in batch:
                wid = int(wid)
                if wid not in by_id:
                    continue
                label = str(method).strip().upper() if method is not None else "UNKNOWN"
                label = label or "UNKNOWN"
                play = by_id[wid]["play"]
                row_methods[label] += 1
                row_methods_by_play[play][label] += 1
                d = finite(days)
                if d is not None:
                    row_days_by_method[label].append(d)
                m = month_start(month)
                monthly[wid][m]["methods"].append(method)
                monthly[wid][m]["days"].append(days)

    results: list[dict[str, Any]] = []
    month_methods: Counter[str] = Counter()
    for wid, well in by_id.items():
        first_prod = date.fromisoformat(well["first_prod"])
        window = months_between(first_prod, holdout_end)
        labels: dict[date, str] = {}
        partial = 0
        days_observed = 0
        for month in window:
            rec = monthly.get(wid, {}).get(month)
            label = classify_month_methods(rec["methods"]) if rec else "MISSING"
            labels[month] = label
            month_methods[label] += 1
            if rec:
                valid_days = [finite(v) for v in rec["days"]]
                valid_days = [v for v in valid_days if v is not None]
                if valid_days:
                    days_observed += 1
                    days = max(valid_days)
                    if 0 < days < calendar.monthrange(month.year, month.month)[1]:
                        partial += 1
        tr_counts = provenance_counts(labels[m] for m in window if m <= cutoff)
        hd_counts = provenance_counts(labels[m] for m in window if cutoff < m <= holdout_end)
        results.append({
            "well_key": well["well_key"],
            "wellid": wid,
            "role": well["role"],
            "play": well["play"],
            "bucket": well["bucket"],
            "major_phase": well.get("major_phase", ""),
            "train_reported_fraction": fraction(tr_counts, "REPORTED"),
            "holdout_reported_fraction": fraction(hd_counts, "REPORTED"),
            "train_dca_fraction": fraction(tr_counts, "DCA"),
            "holdout_dca_fraction": fraction(hd_counts, "DCA"),
            "partial_producingdays_fraction": partial / days_observed if days_observed else float("nan"),
            "producingdays_observed_months": days_observed,
        })

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_well = out / "provenance_by_well.csv"
    with per_well.open("w", newline="", encoding="utf-8") as fh:
        fields = list(results[0]) if results else []
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    def grouped(field: str) -> dict[str, Any]:
        values = sorted({str(row[field]) for row in results})
        return {value: summarize_group([r for r in results if str(r[field]) == value]) for value in values}

    row_total = sum(row_methods.values())
    audit = {
        "board_id": manifest["board_id"],
        "production_table": production_table,
        "wells": len(results),
        "interpretation_guardrail": (
            "DCA is a warehouse reporting-method label. Confirm its licensed data-dictionary definition. "
            "Do not call it a future forecast automatically; for Texas oil it may identify allocated/estimated well-level production."
        ),
        "contest_compatibility": (
            "productionreportedmethod and producingdays are validation-only diagnostics and must not be model features"
        ),
        "source_row_method_counts": dict(sorted(row_methods.items())),
        "source_row_method_fractions": {
            key: value / row_total for key, value in sorted(row_methods.items())
        } if row_total else {},
        "well_month_method_counts": dict(sorted(month_methods.items())),
        "by_role": grouped("role"),
        "by_play": grouped("play"),
        "by_major_phase": grouped("major_phase"),
        "row_method_counts_by_play": {
            play: dict(sorted(counts.items())) for play, counts in sorted(row_methods_by_play.items())
        },
        "producingdays_by_method": {
            method: {
                "n": len(values),
                "zero": sum(v == 0 for v in values),
                "partial_1_to_27": sum(1 <= v <= 27 for v in values),
                "at_least_28": sum(v >= 28 for v in values),
            }
            for method, values in sorted(row_days_by_method.items())
        },
        "files": {"private_per_well": str(per_well)},
    }
    output = out / "provenance_audit.json"
    output.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    cur.close()
    conn.rollback()
    conn.close()
    print(f"Wrote {output} and {per_well}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
