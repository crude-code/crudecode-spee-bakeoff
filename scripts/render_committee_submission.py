#!/usr/bin/env python3
"""Render canonical forecast rows into a committee-specified long-form CSV.

The mapping JSON contains an ordered ``columns`` list.  Each output column
must define either ``source`` (a canonical column name) or ``constant``.
This adapter intentionally does not guess a wide/pivoted format; implement a
separate tested adapter if the committee sample requires one.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def render(input_csv: Path, mapping_path: Path, out: Path, template: Path | None = None) -> dict:
    mapping = json.loads(mapping_path.read_text())
    columns = mapping.get("columns")
    if not isinstance(columns, list) or not columns:
        raise ValueError("mapping requires a non-empty columns list")
    names = [str(c.get("name", "")).strip() for c in columns]
    if any(not n for n in names) or len(set(names)) != len(names):
        raise ValueError("output column names must be non-empty and unique")
    for idx, col in enumerate(columns):
        has_source = "source" in col
        has_constant = "constant" in col
        if has_source == has_constant:
            raise ValueError(f"column {idx} must define exactly one of source or constant")

    with input_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("canonical submission has no header")
        source_fields = set(reader.fieldnames)
        missing = sorted({str(c["source"]) for c in columns if "source" in c} - source_fields)
        if missing:
            raise ValueError(f"mapping references missing canonical columns: {missing}")
        rows = list(reader)

    if template is not None:
        with template.open(newline="") as f:
            header = next(csv.reader(f), None)
        if header != names:
            raise ValueError(f"mapping output header {names} does not match template header {header}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        for row in rows:
            rendered = {}
            for col in columns:
                rendered[col["name"]] = row[col["source"]] if "source" in col else col["constant"]
            writer.writerow(rendered)
    return {"input_rows": len(rows), "output_rows": len(rows), "output_columns": names, "output": str(out)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_csv", type=Path)
    p.add_argument("--mapping", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--template", type=Path, help="optional committee sample CSV for exact header validation")
    args = p.parse_args()
    try:
        result = render(args.input_csv, args.mapping, args.out, args.template)
    except Exception as exc:
        sys.exit(f"render_committee_submission failed: {exc}")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
