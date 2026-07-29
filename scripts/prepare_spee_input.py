#!/usr/bin/env python3
"""Normalize committee files into a canonical, auditable SPEE input package.

Outputs:
  history.csv  well_id,month,oil,gas,water
  wells.csv    well_id,pad_id plus available metadata
  manifest.json ingestion policy, aliases, counts and data-quality findings
"""
from __future__ import annotations

# Fresh-checkout bootstrap.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_SRC = _BootstrapPath(__file__).resolve().parents[1] / "src"
if str(_BOOTSTRAP_SRC) not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, str(_BOOTSTRAP_SRC))

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ALIASES = {
    "well_id": ("well_id", "api", "api10", "api12", "api14", "uwi", "well", "wellid", "entity_id", "well_number"),
    "month": ("month", "prod_month", "production_month", "date", "prod_date", "report_month", "production_date"),
    "oil": ("oil", "oil_bbl", "oil_volume", "oil_volume_bbl", "oil_prod", "oil_production", "monthly_oil"),
    "gas": ("gas", "gas_mcf", "gas_volume", "gas_volume_mcf", "gas_prod", "gas_production", "monthly_gas"),
    "water": ("water", "water_bbl", "water_volume", "water_volume_bbl", "water_prod", "water_production", "monthly_water"),
    "pad_id": ("pad_id", "pad", "pad_name", "lease", "lease_id", "facility_id"),
}
HEADER_META_ALIASES = {
    "operator": ("operator", "operator_name", "current_operator"),
    "basin": ("basin", "play", "formation", "reservoir", "sub_basin"),
    "well_name": ("well_name", "name"),
    "lateral_length_ft": ("lateral_length_ft", "lateral_ft", "lateral_length", "perf_length_ft", "completed_length_ft"),
    "latitude": ("latitude", "lat", "surface_latitude", "bottom_latitude"),
    "longitude": ("longitude", "lon", "lng", "surface_longitude", "bottom_longitude"),
    "first_prod_date": ("first_prod_date", "first_production_date", "first_prod_month"),
}


def _norm(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def _alias_map(fieldnames: list[str]) -> dict[str, str]:
    by_norm = {_norm(name): name for name in fieldnames}
    out: dict[str, str] = {}
    for canonical, aliases in {**ALIASES, **HEADER_META_ALIASES}.items():
        for alias in aliases:
            if alias in by_norm:
                out[canonical] = by_norm[alias]
                break
    return out


def _month(value: str) -> str:
    value = value.strip()
    if len(value) >= 7 and value[4] == "-":
        return value[:7] + "-01"
    if len(value) >= 7 and value[2] == "/":
        mm, yyyy = value[:2], value[3:7]
        return f"{yyyy}-{mm}-01"
    if len(value) >= 7 and value[4] == "/":
        yyyy, mm = value[:4], value[5:7]
        return f"{yyyy}-{mm}-01"
    if len(value) == 6 and value.isdigit():
        return f"{value[:4]}-{value[4:]}-01"
    raise ValueError(f"cannot normalize month value {value!r}")


def _num(value: str | None) -> str:
    if value is None or value.strip() == "":
        return ""
    v = float(value.replace(",", ""))
    if v < 0:
        raise ValueError(f"negative production volume {value!r}")
    return str(v)


def _month_index(month: str) -> int:
    return int(month[:4]) * 12 + int(month[5:7]) - 1


def _month_from_index(idx: int) -> str:
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}-01"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_headers(path: Path | None) -> tuple[list[dict[str, str]], dict[str, dict[str, str]], dict[str, str]]:
    if path is None:
        return [], {}, {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("header file has no header row")
        amap = _alias_map(reader.fieldnames)
        if "well_id" not in amap:
            raise ValueError("header file must include a well id column")
        rows, meta = [], {}
        for row in reader:
            well_id = row[amap["well_id"]].strip()
            if not well_id:
                continue
            out = {"well_id": well_id, "pad_id": row.get(amap.get("pad_id", ""), "").strip() or well_id}
            for key in HEADER_META_ALIASES:
                if key in amap:
                    out[key] = row.get(amap[key], "").strip()
            rows.append(out)
            meta[well_id] = out
    return rows, meta, amap


def normalize(
    monthly: Path,
    headers: Path | None,
    out_dir: Path,
    *,
    gap_policy: str = "nan",
    duplicate_policy: str = "error",
) -> dict:
    if gap_policy not in {"error", "nan", "zero"}:
        raise ValueError("gap_policy must be error, nan, or zero")
    if duplicate_policy not in {"error", "sum"}:
        raise ValueError("duplicate_policy must be error or sum")
    header_rows, _, header_aliases = _read_headers(headers)
    out_dir.mkdir(parents=True, exist_ok=True)

    with monthly.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("monthly file has no header row")
        amap = _alias_map(reader.fieldnames)
        missing = [key for key in ("well_id", "month") if key not in amap]
        if missing:
            raise ValueError(f"monthly file missing required columns: {missing}")
        if not any(p in amap for p in ("oil", "gas", "water")):
            raise ValueError("monthly file must include at least one production phase")

        raw_rows: list[dict[str, str]] = []
        for line_no, row in enumerate(reader, start=2):
            well_id = row[amap["well_id"]].strip()
            if not well_id:
                continue
            try:
                raw_rows.append({
                    "well_id": well_id,
                    "month": _month(row[amap["month"]]),
                    "oil": _num(row.get(amap.get("oil", ""))),
                    "gas": _num(row.get(amap.get("gas", ""))),
                    "water": _num(row.get(amap.get("water", ""))),
                })
            except Exception as exc:
                raise ValueError(f"monthly row {line_no}: {exc}") from exc

    raw_rows.sort(key=lambda r: (r["well_id"], r["month"]))
    deduped: list[dict[str, str]] = []
    duplicate_rows = 0
    for row in raw_rows:
        if deduped and row["well_id"] == deduped[-1]["well_id"] and row["month"] == deduped[-1]["month"]:
            duplicate_rows += 1
            if duplicate_policy == "error":
                raise ValueError(f"duplicate well-month: {row['well_id']} {row['month']}")
            for phase in ("oil", "gas", "water"):
                vals = [v for v in (deduped[-1][phase], row[phase]) if v != ""]
                deduped[-1][phase] = str(sum(float(v) for v in vals)) if vals else ""
        else:
            deduped.append(row.copy())

    by_well: dict[str, list[dict[str, str]]] = {}
    for row in deduped:
        by_well.setdefault(row["well_id"], []).append(row)
    rows: list[dict[str, str]] = []
    inserted_gaps = 0
    wells_with_gaps = 0
    fill = "" if gap_policy == "nan" else "0.0"
    for well_id, wr in sorted(by_well.items()):
        expanded = [wr[0]]
        had_gap = False
        for row in wr[1:]:
            prev, cur = _month_index(expanded[-1]["month"]), _month_index(row["month"])
            if cur <= prev:
                raise ValueError(f"calendar disorder for {well_id}")
            if cur > prev + 1:
                had_gap = True
                if gap_policy == "error":
                    raise ValueError(f"calendar gap for {well_id}: {expanded[-1]['month']} -> {row['month']}")
                for idx in range(prev + 1, cur):
                    expanded.append({"well_id": well_id, "month": _month_from_index(idx), "oil": fill, "gas": fill, "water": fill})
                    inserted_gaps += 1
            expanded.append(row)
        wells_with_gaps += int(had_gap)
        rows.extend(expanded)

    with (out_dir / "history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["well_id", "month", "oil", "gas", "water"])
        writer.writeheader(); writer.writerows(rows)

    seen_wells = set(by_well)
    if not header_rows:
        header_rows = [{"well_id": wid, "pad_id": wid} for wid in sorted(seen_wells)]
    else:
        existing = {r["well_id"] for r in header_rows}
        header_rows.extend({"well_id": wid, "pad_id": wid} for wid in sorted(seen_wells - existing))
    fields = sorted({k for row in header_rows for k in row})
    core = ["well_id", "pad_id", "operator", "basin", "well_name", "lateral_length_ft", "latitude", "longitude", "first_prod_date"]
    fields = [x for x in core if x in fields] + [x for x in fields if x not in core]
    with (out_dir / "wells.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(header_rows)

    manifest = {
        "schema": "crudecode_spee_input_v1",
        "monthly_source": str(monthly),
        "monthly_sha256": _sha256(monthly),
        "headers_source": str(headers) if headers else None,
        "headers_sha256": _sha256(headers) if headers else None,
        "monthly_aliases": amap,
        "header_aliases": header_aliases,
        "gap_policy": gap_policy,
        "duplicate_policy": duplicate_policy,
        "source_rows": len(raw_rows),
        "canonical_rows": len(rows),
        "n_wells": len(seen_wells),
        "duplicate_rows_combined": duplicate_rows,
        "inserted_gap_rows": inserted_gaps,
        "wells_with_gaps": wells_with_gaps,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monthly", required=True, type=Path)
    parser.add_argument("--headers", type=Path)
    parser.add_argument("--out", default="spee_data", type=Path)
    parser.add_argument("--gap-policy", choices=("error", "nan", "zero"), default="nan")
    parser.add_argument("--duplicate-policy", choices=("error", "sum"), default="error")
    args = parser.parse_args()
    try:
        manifest = normalize(args.monthly, args.headers, args.out, gap_policy=args.gap_policy, duplicate_policy=args.duplicate_policy)
    except Exception as exc:
        sys.exit(f"prepare_spee_input failed: {exc}")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
