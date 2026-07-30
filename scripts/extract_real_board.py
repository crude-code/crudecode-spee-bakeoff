#!/usr/bin/env python3
"""Extract a leakage-controlled real-well board from the licensed production warehouse.

This version avoids a full aggregation over ``dde_remote.production``.  It:

1. fetches only filtered well headers from ``dde_remote.wells``;
2. creates a deterministic, play/history-balanced candidate order locally;
3. fetches production only for a bounded candidate buffer, in small ID chunks;
4. aggregates duplicate completion rows and validates eligibility locally;
5. expands the buffer only when deduplication/eligibility cannot fill quotas.

Private outputs (never commit):
  board/private/wells.csv
  board/private/series.csv
  board/private/dedup.json
  board/private/manifest.json
"""
from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forecast_benchmark.provenance import (
    PROVENANCE_MODES,
    ProvenanceGate,
    classify_month_methods,
    fraction as provenance_fraction,
    provenance_counts,
)

SEED_SALT = "seed20260728-v2.2"
CUTOFF = date(2024, 6, 1)
HOLDOUT_END = date(2025, 6, 1)
MIN_TRAIN, MAX_TRAIN = 7, 48
MIN_HOLDOUT_REPORTED = 11
EARLIEST_PROD_QUERY = date(2020, 7, 1)

UNAVAILABLE_PLAYS = {
    "APPALACHIAN_MARCELLUS": (
        "absent from this warehouse population; do not substitute Point Pleasant/Utica"
    ),
    "FORT_WORTH_BARNETT": "too thin under the frozen eligibility filters",
}

_TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$")


def validated_table_name(value: str, label: str) -> str:
    if not _TABLE_NAME_RE.fullmatch(value):
        raise ValueError(f"{label} must be schema.table, got {value!r}")
    return value


def play_case(alias: str = "w") -> str:
    return f"""
      CASE
        WHEN {alias}.envbasin='DELAWARE' THEN 'DELAWARE'
        WHEN {alias}.envbasin='MIDLAND' THEN 'MIDLAND'
        WHEN {alias}.envinterval IN ('LOWER EAGLE FORD','UPPER EAGLE FORD') THEN 'EAGLE_FORD'
        WHEN {alias}.envbasin='WILLISTON' THEN 'WILLISTON'
        WHEN {alias}.envinterval IN ('NIOBRARA A','NIOBRARA B','NIOBRARA C','CODELL') THEN 'DJ'
        WHEN {alias}.envinterval='HAYNESVILLE' THEN 'HAYNESVILLE'
      END
    """


def build_header_query(wells_table: str) -> str:
    wells_table = validated_table_name(wells_table, "wells table")
    return f"""
      SELECT
        w.wellid,
        w.api_uwi,
        w.envbasin,
        w.envinterval,
        w.laterallength_ft,
        w.firstproddate,
        w.envoperator,
        {play_case('w')} AS play
      FROM {wells_table} w
      WHERE w.wellid IS NOT NULL
        AND w.laterallength_ft >= 3000
        AND w.firstproddate BETWEEN %(first_prod_min)s AND %(first_prod_max)s
        AND (
             w.envbasin IN ('DELAWARE','MIDLAND','WILLISTON')
          OR w.envinterval IN (
               'LOWER EAGLE FORD','UPPER EAGLE FORD',
               'NIOBRARA A','NIOBRARA B','NIOBRARA C','CODELL',
               'HAYNESVILLE'
             )
        )
    """


def build_production_query(production_table: str, ids: list[int]) -> str:
    production_table = validated_table_name(production_table, "production table")
    if not ids:
        raise ValueError("production query requires at least one well ID")
    # IDs are converted to Python ints before interpolation, so this is safe and
    # gives postgres_fdw a literal IN-list that is easier to push down remotely.
    id_sql = ",".join(str(int(x)) for x in ids)
    return f"""
      SELECT
        p.wellid,
        p.producingmonth,
        p.liquidsprod_bbl,
        p.gasprod_mcf,
        p.waterprod_bbl,
        p.productionreportedmethod,
        p.producingdays
      FROM {production_table} p
      WHERE p.wellid IN ({id_sql})
        AND p.producingmonth >= %(prod_min)s
        AND p.producingmonth <= %(hold_end)s
      ORDER BY p.wellid, p.producingmonth
    """


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
    sys.exit("Set DATABASE_URL or provide a .env containing DATABASE_URL=...")


def month_start(value: date) -> date:
    return date(value.year, value.month, 1)


def add_month(value: date) -> date:
    return date(value.year + (value.month == 12), 1 if value.month == 12 else value.month + 1, 1)


def month_range(start: date, end: date) -> list[date]:
    out: list[date] = []
    cur = month_start(start)
    end = month_start(end)
    while cur <= end:
        out.append(cur)
        cur = add_month(cur)
    return out


def month_span(start: date, end: date) -> int:
    return (end.year - start.year) * 12 + (end.month - start.month) + 1


def bucket_for_span(span: int) -> str:
    if 7 <= span <= 12:
        return "b1"
    if 13 <= span <= 24:
        return "b2"
    if 25 <= span <= 36:
        return "b3"
    return "b4"


def normalized_identifier(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def stable_hash(value: str) -> str:
    return hashlib.md5(value.encode("utf-8"), usedforsecurity=False).hexdigest()


def hash_rows(rows: list[tuple[str, float, float, float]]) -> str:
    h = hashlib.sha256()
    for month, oil, gas, water in rows:
        h.update(f"{month}|{oil:.6g}|{gas:.6g}|{water:.6g};".encode())
    return h.hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def sum_nullable(values: Iterable[float | None]) -> float:
    nums = [x for x in values if x is not None and math.isfinite(x)]
    return float(sum(nums)) if nums else float("nan")


class UnionFind:
    def __init__(self, items: list[int]):
        self.parent = {x: x for x in items}

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def collapse_exact_duplicates(wells: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ids = [int(w["wellid"]) for w in wells]
    uf = UnionFind(ids)
    by_api: dict[str, list[int]] = defaultdict(list)
    by_hash: dict[str, list[int]] = defaultdict(list)
    for w in wells:
        api = normalized_identifier(w.get("api_uwi"))
        if api:
            by_api[api].append(int(w["wellid"]))
        by_hash[w["train_series_hash"]].append(int(w["wellid"]))
    for mapping in (by_api, by_hash):
        for members in mapping.values():
            if len(members) > 1:
                for other in members[1:]:
                    uf.union(members[0], other)
    clusters: dict[int, list[int]] = defaultdict(list)
    for wid in ids:
        clusters[uf.find(wid)].append(wid)
    by_id = {int(w["wellid"]): w for w in wells}
    kept: list[dict[str, Any]] = []
    audit_clusters: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for members in clusters.values():
        ordered = sorted(members, key=lambda wid: by_id[wid]["ord_hash"])
        keep = ordered[0]
        kept.append(by_id[keep])
        if len(ordered) > 1:
            audit_clusters.append({"kept": keep, "excluded": ordered[1:]})
            for wid in ordered[1:]:
                excluded.append({
                    "wellid": wid,
                    "kept_wellid": keep,
                    "same_identifier": normalized_identifier(by_id[wid].get("api_uwi"))
                    == normalized_identifier(by_id[keep].get("api_uwi")),
                    "same_train_series": by_id[wid]["train_series_hash"]
                    == by_id[keep]["train_series_hash"],
                })
    return kept, {
        "exact_identifier_clusters": sum(len(v) > 1 for v in by_api.values()),
        "exact_pre_cutoff_series_clusters": sum(len(v) > 1 for v in by_hash.values()),
        "combined_duplicate_clusters": len(audit_clusters),
        "wells_excluded": len(excluded),
        "clusters": audit_clusters,
        "excluded": excluded,
        "selection_rule": "smallest frozen ord_hash retained; holdout values not used",
        "near_duplicate_detection": "not implemented",
    }


def balanced_order(wells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for w in wells:
        cells[(w["play"], w["bucket_hint"])].append(w)
    for values in cells.values():
        values.sort(key=lambda r: r["ord_hash"])
    keys = sorted(cells)
    cursor = {k: 0 for k in keys}
    out: list[dict[str, Any]] = []
    while True:
        progress = False
        for key in keys:
            values = cells[key]
            if cursor[key] >= len(values):
                continue
            out.append(values[cursor[key]])
            cursor[key] += 1
            progress = True
        if not progress:
            return out


def assign_roles(
    wells: list[dict[str, Any]], n_eval: int, n_cohort: int, n_dev: int,
    n_confirm: int, n_confirm2: int = 0, n_confirm3: int = 0,
) -> None:
    for w in wells:
        w.pop("role", None)
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for w in wells:
        cells[(w["play"], w["bucket"])].append(w)
    for values in cells.values():
        values.sort(key=lambda r: r["ord_hash"])
    keys = sorted(cells)
    cursor = {k: 0 for k in keys}
    # Keep all confirmation roles after the frozen eval/cohort/dev prefix.
    # Each new confirmation role is appended last so no prior role is relabeled.
    for role, quota in (
        ("eval", n_eval), ("cohort", n_cohort), ("dev", n_dev),
        ("confirm", n_confirm), ("confirm2", n_confirm2),
        ("confirm3", n_confirm3),
    ):
        placed = 0
        while placed < quota:
            progress = False
            for key in keys:
                if placed >= quota:
                    break
                values = cells[key]
                if cursor[key] >= len(values):
                    continue
                values[cursor[key]]["role"] = role
                cursor[key] += 1
                placed += 1
                progress = True
            if not progress:
                raise RuntimeError(
                    f"eligible deduplicated population cannot fill {role} quota {quota}; filled {placed}"
                )
    for w in wells:
        w.setdefault("role", "unused")


def warn_gitignore(out: Path) -> None:
    gi = Path.cwd() / ".gitignore"
    text = gi.read_text(encoding="utf-8", errors="ignore") if gi.exists() else ""
    if "board/private/" not in text:
        print("WARNING: .gitignore does not contain board/private/", file=sys.stderr)
    if "private" not in out.parts:
        print(f"WARNING: output path {out} does not contain a private directory", file=sys.stderr)


def fetch_headers(cur: Any, wells_table: str) -> list[dict[str, Any]]:
    cur.execute(
        build_header_query(wells_table),
        {
            "first_prod_min": date(2020, 7, 1),
            "first_prod_max": date(2023, 12, 31),
        },
    )
    cols = [d[0] for d in cur.description]
    headers: list[dict[str, Any]] = []
    for row in cur.fetchall():
        w = dict(zip(cols, row))
        if not w.get("play") or not w.get("firstproddate"):
            continue
        w["wellid"] = int(w["wellid"])
        w["ord_hash"] = stable_hash(f"{w['wellid']}|{SEED_SALT}")
        hint_span = month_span(month_start(w["firstproddate"]), CUTOFF)
        if not MIN_TRAIN <= hint_span <= MAX_TRAIN:
            continue
        w["bucket_hint"] = bucket_for_span(hint_span)
        headers.append(w)
    return headers


def fetch_production(
    cur: Any,
    production_table: str,
    headers: list[dict[str, Any]],
    chunk_size: int,
) -> dict[int, dict[date, dict[str, Any]]]:
    """Fetch volumes plus provenance metadata without exposing it to models.

    Multiple completion rows can contribute to a well-calendar-month.  Volumes
    are summed exactly as before.  Provenance is classified at the month level:
    homogeneous REPORTED, homogeneous DCA, MIXED, or UNKNOWN.  Producing days
    are retained only for private validation diagnostics and are never written
    to series.csv.
    """
    agg: dict[int, dict[date, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(lambda: {
            "phases": [[], [], []],
            "methods": [],
            "producingdays": [],
        })
    )
    for start in range(0, len(headers), chunk_size):
        chunk = headers[start : start + chunk_size]
        ids = [int(w["wellid"]) for w in chunk]
        print(f"    production chunk {start + 1}-{start + len(chunk)} / {len(headers)}", flush=True)
        cur.execute(
            build_production_query(production_table, ids),
            {"prod_min": EARLIEST_PROD_QUERY, "hold_end": HOLDOUT_END},
        )
        while True:
            rows = cur.fetchmany(10_000)
            if not rows:
                break
            for wid, month, oil, gas, water, method, producingdays in rows:
                m = month_start(month)
                record = agg[int(wid)][m]
                vals = (finite_number(oil), finite_number(gas), finite_number(water))
                for i, value in enumerate(vals):
                    if value is not None:
                        record["phases"][i].append(value)
                record["methods"].append(method)
                days = finite_number(producingdays)
                if days is not None:
                    record["producingdays"].append(days)

    out: dict[int, dict[date, dict[str, Any]]] = defaultdict(dict)
    for wid, months in agg.items():
        for month, record in months.items():
            phase_values = record["phases"]
            days_values = record["producingdays"]
            out[wid][month] = {
                "volumes": tuple(sum_nullable(values) for values in phase_values),
                "method": classify_month_methods(record["methods"]),
                # max avoids double-counting producing days across completion rows
                "producingdays": max(days_values) if days_values else float("nan"),
                "source_rows": max(len(record["methods"]), 1),
            }
    return out


def build_eligible(
    headers: list[dict[str, Any]],
    raw: dict[int, dict[date, dict[str, Any]]],
    provenance_gate: ProvenanceGate,
) -> tuple[list[dict[str, Any]], dict[int, list[tuple[str, float, float, float]]]]:
    eligible: list[dict[str, Any]] = []
    calendar_rows: dict[int, list[tuple[str, float, float, float]]] = {}
    for header in headers:
        wid = int(header["wellid"])
        monthly = raw.get(wid, {})
        positive_train = [
            month
            for month, rec in monthly.items()
            if month <= CUTOFF and any(
                math.isfinite(v) and v > 0 for v in rec["volumes"]
            )
        ]
        if not positive_train:
            continue
        first_prod = min(positive_train)
        span = month_span(first_prod, CUTOFF)
        if not MIN_TRAIN <= span <= MAX_TRAIN:
            continue
        train_reported = sum(month <= CUTOFF for month in monthly)
        hold_reported = sum(CUTOFF < month <= HOLDOUT_END for month in monthly)
        if hold_reported < MIN_HOLDOUT_REPORTED:
            continue
        train_vals = [rec["volumes"] for month, rec in monthly.items() if month <= CUTOFF]
        oil_tr = sum(v[0] for v in train_vals if math.isfinite(v[0]))
        gas_tr = sum(v[1] for v in train_vals if math.isfinite(v[1]))
        water_tr = sum(v[2] for v in train_vals if math.isfinite(v[2]))
        if oil_tr <= 0 and gas_tr <= 0:
            continue

        months = month_range(first_prod, HOLDOUT_END)
        labels: dict[date, str] = {
            month: str(monthly.get(month, {}).get("method", "MISSING"))
            for month in months
        }
        train_labels = [labels[m] for m in months if m <= CUTOFF]
        hold_labels = [labels[m] for m in months if CUTOFF < m <= HOLDOUT_END]
        if not provenance_gate.accepts(train_labels, hold_labels):
            continue

        train_counts = provenance_counts(train_labels)
        hold_counts = provenance_counts(hold_labels)
        days_values: list[tuple[date, float]] = []
        for month in months:
            days = finite_number(monthly.get(month, {}).get("producingdays"))
            if days is not None:
                days_values.append((month, days))
        partial_days = sum(
            0 < days < calendar.monthrange(month.year, month.month)[1]
            for month, days in days_values
        )
        zero_days = sum(days == 0 for _, days in days_values)

        w = dict(header)
        w.update({
            "first_prod_month": first_prod,
            "train_span_mo": span,
            "train_reported_mo": train_reported,
            "hold_reported_mo": hold_reported,
            "oil_tr": oil_tr,
            "gas_tr": gas_tr,
            "water_tr": water_tr,
            "bucket": bucket_for_span(span),
            "train_provenance_counts": train_counts,
            "holdout_provenance_counts": hold_counts,
            "train_reported_fraction": provenance_fraction(train_counts, "REPORTED"),
            "holdout_reported_fraction": provenance_fraction(hold_counts, "REPORTED"),
            "train_dca_fraction": provenance_fraction(train_counts, "DCA"),
            "holdout_dca_fraction": provenance_fraction(hold_counts, "DCA"),
            "producingdays_observed_months": len(days_values),
            "producingdays_partial_fraction": (
                partial_days / len(days_values) if days_values else float("nan")
            ),
            "producingdays_zero_fraction": (
                zero_days / len(days_values) if days_values else float("nan")
            ),
        })
        rows: list[tuple[str, float, float, float]] = []
        for month in months:
            rec = monthly.get(month)
            oil, gas, water = rec["volumes"] if rec else (float("nan"),) * 3
            rows.append((month.isoformat(), oil, gas, water))
        train_rows = [row for row in rows if row[0] <= CUTOFF.isoformat()]
        w["train_series_hash"] = hash_rows(train_rows)
        w["full_series_hash"] = hash_rows(rows)
        eligible.append(w)
        calendar_rows[wid] = rows
    return eligible, calendar_rows

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=int, default=1000)
    ap.add_argument("--cohort", type=int, default=400)
    ap.add_argument("--dev", type=int, default=300)
    ap.add_argument(
        "--confirm", type=int, default=0,
        help="first untouched confirmation pool assigned after dev",
    )
    ap.add_argument(
        "--confirm2", type=int, default=0,
        help="second untouched fixed-profile confirmation pool assigned after confirm",
    )
    ap.add_argument(
        "--confirm3", type=int, default=0,
        help="third untouched pool reserved for the pre-registered v1.4 challenger",
    )
    ap.add_argument("--out", default="board/private")
    ap.add_argument("--wells-table", default="dde_remote.wells")
    ap.add_argument("--production-table", default="dde_remote.production")
    ap.add_argument(
        "--provenance-mode", choices=sorted(PROVENANCE_MODES), default="mixed",
        help=(
            "well-level provenance gate; volumes are never row-filtered. "
            "reported_holdout requires all 12 holdout months REPORTED; "
            "reported_full requires all train+holdout months REPORTED"
        ),
    )
    ap.add_argument("--min-train-reported-fraction", type=float, default=0.80)
    ap.add_argument("--min-holdout-reported-fraction", type=float, default=1.00)
    ap.add_argument("--candidate-buffer-factor", type=float, default=4.0)
    ap.add_argument("--production-chunk-size", type=int, default=50)
    ap.add_argument("--statement-timeout-ms", type=int, default=300_000)
    ap.add_argument(
        "--assert-role-prefix-from", type=Path,
        help="prior wells.csv whose existing role assignments must remain unchanged",
    )
    args = ap.parse_args()
    if min(args.eval, args.cohort, args.dev, args.confirm, args.confirm2, args.confirm3) < 0:
        ap.error("role counts must be nonnegative")
    if args.candidate_buffer_factor < 1:
        ap.error("--candidate-buffer-factor must be >= 1")
    if args.production_chunk_size < 1:
        ap.error("--production-chunk-size must be >= 1")
    try:
        provenance_gate = ProvenanceGate(
            mode=args.provenance_mode,
            min_train_reported_fraction=args.min_train_reported_fraction,
            min_holdout_reported_fraction=args.min_holdout_reported_fraction,
        )
    except ValueError as exc:
        ap.error(str(exc))

    try:
        import psycopg2
    except ImportError:
        sys.exit("psycopg2 required: pip install psycopg2-binary")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    warn_gitignore(out)
    required = (
        args.eval + args.cohort + args.dev + args.confirm + args.confirm2 + args.confirm3
    )
    if required == 0:
        ap.error("at least one role count must be positive")

    conn = psycopg2.connect(load_db_url())
    conn.set_session(readonly=True, autocommit=False)
    cur = conn.cursor()
    cur.execute("SET LOCAL statement_timeout = %s", (args.statement_timeout_ms,))

    print(f"Fetching filtered headers from {args.wells_table}...", flush=True)
    print(f"  provenance mode: {args.provenance_mode}", flush=True)
    headers = fetch_headers(cur, args.wells_table)
    print(f"  filtered headers: {len(headers)}", flush=True)
    if len(headers) < required:
        raise RuntimeError(f"only {len(headers)} header candidates for {required} requested wells")
    ordered = balanced_order(headers)

    initial_target = min(
        len(ordered),
        max(required + 96, int(math.ceil(required * args.candidate_buffer_factor))),
    )
    target = initial_target
    raw_all: dict[int, dict[date, tuple[float, float, float]]] = defaultdict(dict)
    fetched = 0
    final_eligible: list[dict[str, Any]] = []
    final_calendar: dict[int, list[tuple[str, float, float, float]]] = {}
    final_dedup: dict[str, Any] = {}

    while True:
        new_headers = ordered[fetched:target]
        if new_headers:
            print(
                f"Fetching production for candidate buffer {fetched + 1}-{target} "
                f"from {args.production_table}...",
                flush=True,
            )
            new_raw = fetch_production(
                cur,
                args.production_table,
                new_headers,
                args.production_chunk_size,
            )
            for wid, months in new_raw.items():
                raw_all[wid].update(months)
            fetched = target

        considered = ordered[:fetched]
        eligible, calendar_rows = build_eligible(considered, raw_all, provenance_gate)
        deduped, dedup = collapse_exact_duplicates(eligible)
        print(
            f"  considered={len(considered)} eligible={len(eligible)} "
            f"deduplicated={len(deduped)} required={required}",
            flush=True,
        )
        try:
            assign_roles(
                deduped, args.eval, args.cohort, args.dev,
                args.confirm, args.confirm2, args.confirm3,
            )
            final_eligible = deduped
            final_calendar = calendar_rows
            final_dedup = dedup
            break
        except RuntimeError as exc:
            if fetched >= len(ordered):
                raise RuntimeError(f"cannot fill requested quotas after all candidates: {exc}") from exc
            next_target = min(len(ordered), max(fetched + required, int(math.ceil(fetched * 1.75))))
            print(f"  expanding candidate buffer to {next_target}: {exc}", flush=True)
            target = next_target

    roles = ("eval", "cohort", "dev", "confirm", "confirm2", "confirm3")
    chosen = [w for w in final_eligible if w.get("role") in set(roles)]
    counts = {role: sum(w.get("role") == role for w in chosen) for role in roles}
    expected = {
        "eval": args.eval, "cohort": args.cohort, "dev": args.dev,
        "confirm": args.confirm, "confirm2": args.confirm2,
        "confirm3": args.confirm3,
    }
    if counts != expected:
        raise RuntimeError(f"role count mismatch: expected={expected}, actual={counts}")
    for w in chosen:
        w["well_key"] = f"{w['play']}|{w['bucket']}|{w['ord_hash'][:10]}"

    if args.assert_role_prefix_from is not None:
        prior: dict[str, str] = {}
        with args.assert_role_prefix_from.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("role") in {
                    "eval", "cohort", "dev", "confirm", "confirm2", "confirm3"
                }:
                    prior[row["well_key"]] = row["role"]
        current = {w["well_key"]: w["role"] for w in chosen}
        missing = sorted(set(prior) - set(current))
        changed = sorted(
            key for key, role in prior.items()
            if key in current and current[key] != role
        )
        if missing or changed:
            raise RuntimeError(
                "role-prefix preservation failed: "
                f"missing={len(missing)} changed={len(changed)}; "
                "do not replace the frozen board"
            )
        print(f"Role-prefix preservation passed for {len(prior)} prior wells")

    print(f"Assigned roles: {counts}")

    wells_path = out / "wells.csv"
    series_path = out / "series.csv"
    dedup_path = out / "dedup.json"
    provenance_path = out / "provenance_summary.json"
    manifest_path = out / "manifest.json"

    with wells_path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow([
            "well_key", "role", "play", "bucket", "major_phase", "wellid", "api_uwi",
            "envbasin", "envinterval", "lateral_ft", "first_prod", "operator",
            "train_span_mo", "train_reported_mo", "hold_reported_mo",
            "train_reported_fraction", "holdout_reported_fraction",
            "train_dca_fraction", "holdout_dca_fraction",
            "producingdays_observed_months", "producingdays_partial_fraction",
            "producingdays_zero_fraction", "train_series_hash", "full_series_hash",
        ])
        for w in chosen:
            major = "oil" if float(w["oil_tr"]) >= float(w["gas_tr"]) / 6.0 else "gas"
            key = f"{w['play']}|{w['bucket']}|{w['ord_hash'][:10]}"
            w["well_key"] = key
            wr.writerow([
                key, w["role"], w["play"], w["bucket"], major, w["wellid"], w.get("api_uwi"),
                w["envbasin"], w["envinterval"], w["laterallength_ft"], w["first_prod_month"],
                w.get("envoperator"), w["train_span_mo"], w["train_reported_mo"],
                w["hold_reported_mo"], w["train_reported_fraction"],
                w["holdout_reported_fraction"], w["train_dca_fraction"],
                w["holdout_dca_fraction"], w["producingdays_observed_months"],
                w["producingdays_partial_fraction"], w["producingdays_zero_fraction"],
                w["train_series_hash"], w["full_series_hash"],
            ])

    chosen_by_id = {int(w["wellid"]): w for w in chosen}
    with series_path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["well_key", "month", "oil", "gas", "water", "split"])
        for wid, w in chosen_by_id.items():
            for month, oil, gas, water in final_calendar[wid]:
                split = "train" if month <= CUTOFF.isoformat() else "holdout"
                wr.writerow([w["well_key"], month, oil, gas, water, split])

    dedup_path.write_text(json.dumps(final_dedup, indent=2) + "\n", encoding="utf-8")

    def summarize_provenance(rows: list[dict[str, Any]]) -> dict[str, Any]:
        by_role: dict[str, Any] = {}
        by_play: dict[str, Any] = {}
        for label, groups in (("role", by_role), ("play", by_play)):
            values = sorted({str(w[label]) for w in rows})
            for value in values:
                subset = [w for w in rows if str(w[label]) == value]
                groups[value] = {
                    "n": len(subset),
                    "mean_train_reported_fraction": float(sum(w["train_reported_fraction"] for w in subset) / len(subset)),
                    "mean_holdout_reported_fraction": float(sum(w["holdout_reported_fraction"] for w in subset) / len(subset)),
                    "all_holdout_reported_wells": int(sum(w["holdout_reported_fraction"] >= 1.0 for w in subset)),
                    "all_full_window_reported_wells": int(sum(
                        w["train_reported_fraction"] >= 1.0 and w["holdout_reported_fraction"] >= 1.0
                        for w in subset
                    )),
                }
        return {
            "interpretation": (
                "productionreportedmethod is warehouse provenance metadata. "
                "DCA must not be described as a future forecast without the licensed data dictionary; "
                "in Texas it may identify lease-to-well allocation/estimation."
            ),
            "model_input_policy": (
                "reporting method and producing days are validation-only and are not written to series.csv"
            ),
            "gate": {
                "mode": provenance_gate.mode,
                "min_train_reported_fraction": provenance_gate.min_train_reported_fraction,
                "min_holdout_reported_fraction": provenance_gate.min_holdout_reported_fraction,
            },
            "by_role": by_role,
            "by_play": by_play,
        }

    provenance_path.write_text(
        json.dumps(summarize_provenance(chosen), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    board_basis = {
        "seed_salt": SEED_SALT,
        "cutoff": CUTOFF.isoformat(),
        "holdout_end": HOLDOUT_END.isoformat(),
        "requested": expected,
        "provenance_gate": {
            "mode": provenance_gate.mode,
            "min_train_reported_fraction": provenance_gate.min_train_reported_fraction,
            "min_holdout_reported_fraction": provenance_gate.min_holdout_reported_fraction,
        },
        "selected": sorted((w["well_key"], w["train_series_hash"]) for w in chosen),
    }
    board_id = hashlib.sha256(json.dumps(board_basis, sort_keys=True).encode()).hexdigest()
    manifest = {
        "schema_version": "2.6",
        "board_id": board_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed_salt": SEED_SALT,
        "cutoff": CUTOFF.isoformat(),
        "holdout_end": HOLDOUT_END.isoformat(),
        "expected_holdout_calendar_months": 12,
        "requested": expected,
        "written": counts,
        "warehouse_tables": {"wells": args.wells_table, "production": args.production_table},
        "provenance_gate": {
            "mode": provenance_gate.mode,
            "min_train_reported_fraction": provenance_gate.min_train_reported_fraction,
            "min_holdout_reported_fraction": provenance_gate.min_holdout_reported_fraction,
            "row_filtering": False,
        },
        "header_candidates": len(headers),
        "candidate_rows_materialized": fetched,
        "eligible_before_dedup": len(final_eligible) + final_dedup["wells_excluded"],
        "eligible_after_dedup": len(final_eligible),
        "production_chunk_size": args.production_chunk_size,
        "plays_unavailable": UNAVAILABLE_PLAYS,
        "point_in_time": "current snapshot; not valid for reporting-lag estimation",
        "licensing": "Licensed-derived. Never commit board/private/.",
        "dedup_selection_uses_holdout": False,
        "files": {
            "wells.csv": file_sha256(wells_path),
            "series.csv": file_sha256(series_path),
            "dedup.json": file_sha256(dedup_path),
            "provenance_summary.json": file_sha256(provenance_path),
            "extractor": file_sha256(Path(__file__)),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    cur.close()
    conn.rollback()
    conn.close()
    print(f"\nWrote {wells_path}, {series_path}, {dedup_path}, {provenance_path}, {manifest_path}")
    print(f"Board ID: {board_id}")
    print("REMINDER: board/private/ must remain gitignored and private.")


if __name__ == "__main__":
    main()
