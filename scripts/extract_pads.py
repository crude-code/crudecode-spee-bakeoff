"""Extract the frozen 20-pad benchmark snapshot into benchmark_data/.

Writes:
    benchmark_data/pads.json          pad + well metadata
    benchmark_data/train/<pad>.csv    well_id,month,oil,gas,water — months <= CUTOFF
    benchmark_data/holdout/<pad>.csv  the HORIZON months after CUTOFF

The train/ directory is the only thing a forecasting arm may see. holdout/
exists so the scorer can reassemble full series; nothing that produces a
forecast gets it.

Pad selection is frozen here, not queried: the 20 pads were drawn once
(2026-07-23) by md5-hash order within basin/operator strata — 2-8 co-developed
horizontal wells each, every well producing through >= 2026-04 with >= 36
months of history — and the benchmark is only honest if this list never
quietly changes. Reruns refresh volumes for the same pads.

Needs BENCHMARK_DB_URL in the environment or in .env next to the repo root.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import psycopg2

CUTOFF = date(2025, 4, 1)     # last training month
HORIZON = 12                  # holdout months: 2025-05 .. 2026-04
MIN_TRAIN_MONTHS = 24

PAD_IDS = [
    "05-123-38699", "05-123-42994",  # DJ: Fundare, Occidental
    "17-031-26868", "17-081-21286",  # Ark-La-Tx: Expand, BP
    "30-015-44838", "42-371-40373", "42-389-35910",  # Delaware: Exxon, Continental, Chevron
    "33-023-00833", "33-105-05667",  # Williston: Formentera, Kraken
    "37-105-21870", "37-129-28821",  # Appalachian: JKLM, CNX
    "42-033-32605",                  # Permian Other: HighPeak
    "42-123-34653", "42-479-41450",  # Western Gulf: ConocoPhillips, TBM Catarina
    "42-227-40903", "42-329-44705",  # Midland: Ovintiv, Discovery
    "43-013-54007", "43-047-55886",  # Uinta: Crescent, FourPoint
    "49-005-77619", "49-009-48699",  # Powder River: EOG, Devon
]

ROSTER_SQL = """
select distinct on (w.wellid)
    w.wellpadid, w.wellid, w.api_uwi, w.wellname, w.envoperator, w.envbasin,
    w.envplay, w.envinterval, round(w.laterallength_ft) as lat_ft,
    to_char(w.firstproddate, 'YYYY-MM') as first_prod, w.county, w.stateprovince
from data.wells w
where w.trajectory = 'HORIZONTAL' and w.firstproddate is not null
  and w.wellpadid = any(%s)
order by w.wellid, w.laterallength_ft desc nulls last
"""

PRODUCTION_SQL = """
select w.api_uwi, p.producingmonth,
       sum(p.liquidsprod_bbl) as oil, sum(p.gasprod_mcf) as gas, sum(p.waterprod_bbl) as water
from data.production p
join (
    select distinct wellid, api_uwi from data.wells
    where trajectory = 'HORIZONTAL' and firstproddate is not null and wellpadid = any(%s)
) w on w.wellid = p.wellid
where p.producingmonth <= %s
group by w.api_uwi, p.producingmonth
order by w.api_uwi, p.producingmonth
"""


def month_add(d: date, n: int) -> date:
    y, m = d.year, d.month + n
    return date(y + (m - 1) // 12, (m - 1) % 12 + 1, 1)


def month_range(start: date, end: date) -> list[date]:
    out, d = [], start
    while d <= end:
        out.append(d)
        d = month_add(d, 1)
    return out


def load_db_url() -> str:
    if os.environ.get("BENCHMARK_DB_URL"):
        return os.environ["BENCHMARK_DB_URL"]
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("BENCHMARK_DB_URL="):
                value = line.split("=", 1)[1].strip()
                if value:
                    return value
    raise SystemExit("BENCHMARK_DB_URL not set (env or .env)")


def main() -> None:
    out_root = Path(__file__).resolve().parents[1] / "benchmark_data"
    (out_root / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "holdout").mkdir(parents=True, exist_ok=True)

    holdout_months = [month_add(CUTOFF, i + 1) for i in range(HORIZON)]
    conn = psycopg2.connect(load_db_url())
    cur = conn.cursor()

    cur.execute(ROSTER_SQL, (PAD_IDS,))
    pads: dict[str, dict] = {p: {"pad_id": p, "wells": []} for p in PAD_IDS}
    for (pad_id, _wellid, api, name, op, basin, play, interval,
         lat_ft, first_prod, county, state) in cur.fetchall():
        meta = pads[pad_id]
        meta.update(operator=op, basin=basin, play=play, county=county, state=state)
        meta["wells"].append({
            "well_id": api, "name": name, "interval": interval,
            "lateral_ft": float(lat_ft) if lat_ft is not None else None,
            "first_prod": first_prod,
        })

    cur.execute(PRODUCTION_SQL, (PAD_IDS, holdout_months[-1]))
    prod: dict[str, dict[date, tuple]] = defaultdict(dict)
    for api, month, oil, gas, water in cur.fetchall():
        prod[api][month] = (oil, gas, water)
    conn.close()

    api_to_pad = {w["well_id"]: p["pad_id"] for p in pads.values() for w in p["wells"]}
    problems: list[str] = []
    n_wells = 0

    for pad_id in PAD_IDS:
        rows_train, rows_holdout = [], []
        for well in sorted(pads[pad_id]["wells"], key=lambda w: w["well_id"]):
            api = well["well_id"]
            months = prod.get(api, {})
            if not months:
                problems.append(f"{pad_id}/{api}: no production rows")
                continue
            first = min(months)
            n_train = len(month_range(first, CUTOFF))
            if n_train < MIN_TRAIN_MONTHS:
                problems.append(f"{pad_id}/{api}: only {n_train} train months")
            missing_holdout = [m for m in holdout_months
                               if all(v is None for v in months.get(m, (None,) * 3))]
            if len(missing_holdout) > 2:
                problems.append(f"{pad_id}/{api}: {len(missing_holdout)} unreported holdout months")
            # contiguous calendar; reporting gaps become blank cells (NaN), never 0
            for m in month_range(first, holdout_months[-1]):
                oil, gas, water = months.get(m, (None, None, None))
                row = (api, m.isoformat(),
                       "" if oil is None else round(oil),
                       "" if gas is None else round(gas),
                       "" if water is None else round(water))
                (rows_train if m <= CUTOFF else rows_holdout).append(row)
            n_wells += 1

        for sub, rows in (("train", rows_train), ("holdout", rows_holdout)):
            with open(out_root / sub / f"{pad_id}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["well_id", "month", "oil", "gas", "water"])
                w.writerows(rows)

    (out_root / "pads.json").write_text(json.dumps([pads[p] for p in PAD_IDS], indent=1))
    (out_root / "snapshot.json").write_text(json.dumps({
        "cutoff": CUTOFF.isoformat(), "horizon": HORIZON,
        "holdout_months": [m.isoformat() for m in holdout_months],
        "extracted": date.today().isoformat(),
        "n_pads": len(PAD_IDS), "n_wells": n_wells,
    }, indent=1))

    print(f"wrote {len(PAD_IDS)} pads, {n_wells} wells -> {out_root}")
    for p in problems:
        print(f"WARN {p}", file=sys.stderr)
    if len(api_to_pad) != n_wells:
        print(f"WARN roster has {len(api_to_pad)} wells, extracted {n_wells}", file=sys.stderr)


if __name__ == "__main__":
    main()
