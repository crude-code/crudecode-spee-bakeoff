#!/usr/bin/env python3
"""Extract a leakage-controlled, calendar-correct real-well validation board.

Private outputs (Enverus-derived; never commit):
  board/private/wells.csv
  board/private/series.csv
  board/private/dedup.json
  board/private/manifest.json

Key controls:
- one aggregated row per well/calendar month;
- explicit NaN rows for missing calendar months;
- 7-48 *calendar* months of visible history;
- exact duplicate clustering uses identifiers and PRE-CUTOFF series only;
- duplicates are collapsed before role assignment, so quotas are backfilled;
- eval, cohort and dev pools are disjoint and deterministically sampled;
- the database session is read-only.

This current-snapshot board cannot answer point-in-time reporting-lag questions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

SEED_SALT = "seed20260728-v2"
CUTOFF = date(2024, 6, 1)          # last visible production month
HOLDOUT_END = date(2025, 6, 1)     # July 2024 through June 2025 = 12 calendar months
MIN_TRAIN, MAX_TRAIN = 7, 48
MIN_HOLDOUT_REPORTED = 11

PLAY_CASE = """
    CASE WHEN w.envbasin='DELAWARE' THEN 'DELAWARE'
         WHEN w.envbasin='MIDLAND'  THEN 'MIDLAND'
         WHEN w.envinterval IN ('LOWER EAGLE FORD','UPPER EAGLE FORD') THEN 'EAGLE_FORD'
         WHEN w.envbasin='WILLISTON' THEN 'WILLISTON'
         WHEN w.envinterval IN ('NIOBRARA A','NIOBRARA B','NIOBRARA C','CODELL') THEN 'DJ'
         WHEN w.envinterval='HAYNESVILLE' THEN 'HAYNESVILLE'
    END
"""

UNAVAILABLE_PLAYS = {
    "APPALACHIAN_MARCELLUS": "absent from this warehouse population; do not substitute Point Pleasant/Utica",
    "FORT_WORTH_BARNETT": "too thin under the frozen eligibility filters",
}

MONTHLY_CTE = """
monthly AS (
  SELECT p.wellid,
         date_trunc('month', p.producingmonth)::date AS producingmonth,
         sum(p.liquidsprod_bbl)::float8 AS oil,
         sum(p.gasprod_mcf)::float8 AS gas,
         sum(p.waterprod_bbl)::float8 AS water
  FROM data.production p
  WHERE p.producingmonth <= %(hold_end)s
  GROUP BY p.wellid, date_trunc('month', p.producingmonth)::date
)
"""

SAMPLER = f"""
WITH {MONTHLY_CTE}, play AS (
  SELECT w.wellid, w.api_uwi, w.envbasin, w.envinterval,
         w.laterallength_ft, w.firstproddate, w.envoperator,
         {PLAY_CASE} AS play
  FROM data.wells w
  WHERE w.laterallength_ft >= 3000
    AND w.firstproddate BETWEEN '2020-07-01' AND '2023-12-31'
), agg AS (
  SELECT wellid,
         min(producingmonth) FILTER (
           WHERE producingmonth <= %(cutoff)s
             AND (coalesce(oil,0)>0 OR coalesce(gas,0)>0 OR coalesce(water,0)>0)
         ) AS first_prod_month,
         count(*) FILTER (WHERE producingmonth <= %(cutoff)s) AS train_reported_mo,
         count(*) FILTER (WHERE producingmonth > %(cutoff)s AND producingmonth <= %(hold_end)s) AS hold_reported_mo,
         sum(coalesce(oil,0)) FILTER (WHERE producingmonth <= %(cutoff)s) AS oil_tr,
         sum(coalesce(gas,0)) FILTER (WHERE producingmonth <= %(cutoff)s) AS gas_tr,
         sum(coalesce(water,0)) FILTER (WHERE producingmonth <= %(cutoff)s) AS water_tr
  FROM monthly
  GROUP BY wellid
), eligible AS (
  SELECT p.*, a.first_prod_month, a.train_reported_mo, a.hold_reported_mo,
         a.oil_tr, a.gas_tr, a.water_tr,
         (
           extract(year from age(%(cutoff)s::date, a.first_prod_month))::int * 12
           + extract(month from age(%(cutoff)s::date, a.first_prod_month))::int + 1
         ) AS train_span_mo
  FROM play p JOIN agg a ON a.wellid=p.wellid
  WHERE p.play IS NOT NULL
    AND a.first_prod_month IS NOT NULL
    AND a.hold_reported_mo >= %(min_holdout)s
    AND (coalesce(a.oil_tr,0)>0 OR coalesce(a.gas_tr,0)>0)
)
SELECT e.*,
       CASE WHEN e.train_span_mo BETWEEN 7 AND 12 THEN 'b1'
            WHEN e.train_span_mo BETWEEN 13 AND 24 THEN 'b2'
            WHEN e.train_span_mo BETWEEN 25 AND 36 THEN 'b3'
            ELSE 'b4' END AS bucket,
       md5(e.wellid::text || %(salt)s) AS ord_hash
FROM eligible e
WHERE e.train_span_mo BETWEEN %(min_train)s AND %(max_train)s
ORDER BY e.play, bucket, ord_hash
"""

SERIES_SQL = f"""
WITH {MONTHLY_CTE}
SELECT wellid, producingmonth, oil, gas, water
FROM monthly
WHERE wellid = ANY(%(ids)s)
ORDER BY wellid, producingmonth
"""


def load_db_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    candidates = [Path.cwd(), *Path(__file__).resolve().parents[:3]]
    for parent in candidates:
        env = parent / ".env"
        if not env.exists():
            continue
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("DATABASE_URL="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("Set DATABASE_URL or provide a .env containing DATABASE_URL=...")


def add_month(d: date) -> date:
    return date(d.year + (1 if d.month == 12 else 0), 1 if d.month == 12 else d.month + 1, 1)


def month_range(start: date, end: date) -> list[date]:
    out: list[date] = []
    cursor = date(start.year, start.month, 1)
    end = date(end.year, end.month, 1)
    while cursor <= end:
        out.append(cursor)
        cursor = add_month(cursor)
    return out


def normalized_identifier(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def hash_rows(rows: list[tuple[str, float, float, float]]) -> str:
    h = hashlib.sha256()
    for m, o, g, w in rows:
        h.update(f"{m}|{o:.6g}|{g:.6g}|{w:.6g};".encode())
    return h.hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
    """Collapse exact identifier or exact PRE-CUTOFF-series duplicates.

    Holdout values are never used to decide membership or the retained record.
    Near-duplicate similarity is intentionally not claimed or implemented here.
    """
    ids = [int(w["wellid"]) for w in wells]
    uf = UnionFind(ids)

    by_api: dict[str, list[int]] = defaultdict(list)
    by_train_hash: dict[str, list[int]] = defaultdict(list)
    for w in wells:
        api = normalized_identifier(w.get("api_uwi"))
        if api:
            by_api[api].append(int(w["wellid"]))
        by_train_hash[w["train_series_hash"]].append(int(w["wellid"]))

    for groups in (by_api, by_train_hash):
        for members in groups.values():
            if len(members) > 1:
                first = members[0]
                for other in members[1:]:
                    uf.union(first, other)

    clusters: dict[int, list[int]] = defaultdict(list)
    for wid in ids:
        clusters[uf.find(wid)].append(wid)

    by_id = {int(w["wellid"]): w for w in wells}
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    duplicate_clusters = []
    for members in clusters.values():
        ordered = sorted(members, key=lambda wid: by_id[wid]["ord_hash"])
        keep = ordered[0]
        kept.append(by_id[keep])
        if len(ordered) > 1:
            duplicate_clusters.append({"kept": keep, "excluded": ordered[1:]})
            for wid in ordered[1:]:
                excluded.append({
                    "wellid": wid,
                    "kept_wellid": keep,
                    "same_identifier": normalized_identifier(by_id[wid].get("api_uwi")) == normalized_identifier(by_id[keep].get("api_uwi")),
                    "same_train_series": by_id[wid]["train_series_hash"] == by_id[keep]["train_series_hash"],
                })

    audit = {
        "exact_identifier_clusters": sum(len(v) > 1 for v in by_api.values()),
        "exact_pre_cutoff_series_clusters": sum(len(v) > 1 for v in by_train_hash.values()),
        "combined_duplicate_clusters": len(duplicate_clusters),
        "wells_excluded": len(excluded),
        "clusters": duplicate_clusters,
        "excluded": excluded,
        "near_duplicate_detection": "not implemented; no near-duplicate claim is made",
        "selection_rule": "smallest frozen ord_hash retained; holdout values not used",
    }
    return kept, audit


def assign_roles(wells: list[dict[str, Any]], n_eval: int, n_cohort: int, n_dev: int) -> None:
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for w in wells:
        cells[(w["play"], w["bucket"])].append(w)
    for values in cells.values():
        values.sort(key=lambda r: r["ord_hash"])

    keys = sorted(cells)
    cursor = {k: 0 for k in keys}
    for role, quota in (("eval", n_eval), ("cohort", n_cohort), ("dev", n_dev)):
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
                raise RuntimeError(f"Eligible deduplicated population cannot fill requested {role} quota {quota}; filled {placed}")
    for w in wells:
        w.setdefault("role", "unused")


def warn_gitignore(out: Path) -> None:
    if "private" not in out.parts:
        print(f"WARNING: output path {out} does not include a private directory name", file=sys.stderr)
    gi = Path.cwd() / ".gitignore"
    if gi.exists() and "board/private/" not in gi.read_text(encoding="utf-8", errors="ignore"):
        print("WARNING: .gitignore does not contain board/private/", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=int, default=1000)
    ap.add_argument("--cohort", type=int, default=400)
    ap.add_argument("--dev", type=int, default=300)
    ap.add_argument("--out", default="board/private")
    ap.add_argument("--statement-timeout-ms", type=int, default=600_000)
    args = ap.parse_args()
    if min(args.eval, args.cohort, args.dev) < 0:
        ap.error("role counts must be nonnegative")

    try:
        import psycopg2
    except ImportError:
        sys.exit("psycopg2 required: pip install psycopg2-binary")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    warn_gitignore(out)

    conn = psycopg2.connect(load_db_url())
    conn.set_session(readonly=True, autocommit=False)
    cur = conn.cursor()
    cur.execute("SET LOCAL statement_timeout = %s", (args.statement_timeout_ms,))

    params = {
        "cutoff": CUTOFF.isoformat(), "hold_end": HOLDOUT_END.isoformat(),
        "salt": SEED_SALT, "min_train": MIN_TRAIN, "max_train": MAX_TRAIN,
        "min_holdout": MIN_HOLDOUT_REPORTED,
    }
    print("Sampling eligible wells...")
    cur.execute(SAMPLER, params)
    cols = [d[0] for d in cur.description]
    wells = [dict(zip(cols, row)) for row in cur.fetchall()]
    print(f"  eligible before dedup: {len(wells)}")

    ids = [int(w["wellid"]) for w in wells]
    print("Pulling monthly production...")
    cur.execute(SERIES_SQL, {"ids": ids, "hold_end": HOLDOUT_END.isoformat()})
    raw: dict[int, dict[date, tuple[float, float, float]]] = defaultdict(dict)
    for wid, month, oil, gas, water in cur.fetchall():
        raw[int(wid)][month] = (
            float(oil) if oil is not None else float("nan"),
            float(gas) if gas is not None else float("nan"),
            float(water) if water is not None else float("nan"),
        )

    # Calendar-complete rows and PRE-CUTOFF hashes.
    calendar_rows: dict[int, list[tuple[str, float, float, float]]] = {}
    for w in wells:
        wid = int(w["wellid"])
        start = w["first_prod_month"]
        rows = []
        for month in month_range(start, HOLDOUT_END):
            oil, gas, water = raw.get(wid, {}).get(month, (float("nan"),) * 3)
            rows.append((month.isoformat(), oil, gas, water))
        calendar_rows[wid] = rows
        train_rows = [r for r in rows if r[0] <= CUTOFF.isoformat()]
        w["train_series_hash"] = hash_rows(train_rows)
        w["full_series_hash"] = hash_rows(rows)  # audit only; never used for selection

    wells, dedup = collapse_exact_duplicates(wells)
    print(f"  eligible after exact dedup: {len(wells)} ({dedup['wells_excluded']} excluded)")

    assign_roles(wells, args.eval, args.cohort, args.dev)
    chosen = [w for w in wells if w["role"] in {"eval", "cohort", "dev"}]
    counts = {role: sum(w["role"] == role for w in chosen) for role in ("eval", "cohort", "dev")}
    requested = {"eval": args.eval, "cohort": args.cohort, "dev": args.dev}
    if counts != requested:
        raise RuntimeError(f"Role counts do not match request: requested={requested}, actual={counts}")
    print(f"  assigned: {counts}")

    wells_path = out / "wells.csv"
    series_path = out / "series.csv"
    dedup_path = out / "dedup.json"
    manifest_path = out / "manifest.json"

    with wells_path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow([
            "well_key", "role", "play", "bucket", "major_phase", "wellid", "api_uwi",
            "envbasin", "envinterval", "lateral_ft", "first_prod", "operator",
            "train_span_mo", "train_reported_mo", "hold_reported_mo",
            "train_series_hash", "full_series_hash",
        ])
        for w in chosen:
            oil = float(w.get("oil_tr") or 0.0)
            gas_boe = float(w.get("gas_tr") or 0.0) / 6.0
            major = "oil" if oil >= gas_boe else "gas"
            key = f"{w['play']}|{w['bucket']}|{w['ord_hash'][:10]}"
            w["well_key"] = key
            wr.writerow([
                key, w["role"], w["play"], w["bucket"], major, w["wellid"], w.get("api_uwi"),
                w["envbasin"], w["envinterval"], w["laterallength_ft"], w["first_prod_month"],
                w.get("envoperator"), w["train_span_mo"], w["train_reported_mo"],
                w["hold_reported_mo"], w["train_series_hash"], w["full_series_hash"],
            ])

    chosen_by_id = {int(w["wellid"]): w for w in chosen}
    with series_path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["well_key", "month", "oil", "gas", "water", "split"])
        for wid, w in chosen_by_id.items():
            for month, oil, gas, water in calendar_rows[wid]:
                split = "train" if month <= CUTOFF.isoformat() else "holdout"
                wr.writerow([w["well_key"], month, oil, gas, water, split])

    dedup_path.write_text(json.dumps(dedup, indent=2) + "\n", encoding="utf-8")

    board_basis = {
        "seed_salt": SEED_SALT,
        "cutoff": CUTOFF.isoformat(),
        "holdout_end": HOLDOUT_END.isoformat(),
        "requested": requested,
        "selected": sorted((w["well_key"], w["train_series_hash"]) for w in chosen),
    }
    board_id = hashlib.sha256(json.dumps(board_basis, sort_keys=True).encode()).hexdigest()
    manifest = {
        "schema_version": 2,
        "board_id": board_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed_salt": SEED_SALT,
        "cutoff": CUTOFF.isoformat(),
        "holdout_end": HOLDOUT_END.isoformat(),
        "expected_holdout_calendar_months": 12,
        "min_train_calendar_months": MIN_TRAIN,
        "max_train_calendar_months": MAX_TRAIN,
        "min_holdout_reported_rows": MIN_HOLDOUT_REPORTED,
        "requested": requested,
        "written": counts,
        "eligible_before_dedup": len(wells) + dedup["wells_excluded"],
        "eligible_after_dedup": len(wells),
        "plays_unavailable": UNAVAILABLE_PLAYS,
        "point_in_time": "current snapshot; not valid for reporting-lag estimation",
        "licensing": "Enverus-derived. Never commit board/private/.",
        "dedup_selection_uses_holdout": False,
        "files": {
            "wells.csv": file_sha256(wells_path),
            "series.csv": file_sha256(series_path),
            "dedup.json": file_sha256(dedup_path),
            "extractor": file_sha256(Path(__file__)),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    cur.close()
    conn.rollback()
    conn.close()
    print(f"\nWrote {wells_path}, {series_path}, {dedup_path}, {manifest_path}")
    print(f"Board ID: {board_id}")
    print("REMINDER: board/private/ must remain gitignored and private.")


if __name__ == "__main__":
    main()
