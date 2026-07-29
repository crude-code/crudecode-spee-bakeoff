"""Validate raw LLM responses and promote them into llm_forecasts/.

Checks each benchmark_data/llm_raw/<pad>.json:
- parses as JSON (tolerating accidental markdown fences),
- covers every well on the pad's roster,
- every phase array is exactly HORIZON months of finite, non-negative numbers.

A pad that fails any check is reported and NOT promoted — the scorer then
counts its wells as forecast_missing rather than silently scoring a partial
or malformed submission.

Usage: python scripts/collect_llm_forecasts.py [data_dir]

Set PAD_IDS=pad-a,pad-b to validate/promote a subset run. This is used by
scripts/run_llm.py --pads so smoke tests do not fail the missing pads that
were intentionally not called.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else REPO / "benchmark_data")
RAW_SUBDIR = os.environ.get("RAW_SUBDIR", "llm_raw")
OUT_SUBDIR = os.environ.get("OUT_SUBDIR", "llm_forecasts")


def parse_response(text: str) -> dict:
    """Extract the forecast object from a response that may carry prose or
    fences around it. Anchors on the "pad_id" key and decodes the balanced
    JSON object containing it, so braces in surrounding prose don't matter."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        return json.loads(fenced.group(1), strict=False)
    # strict=False tolerates literal newlines inside strings (long rationales)
    decoder = json.JSONDecoder(strict=False)
    anchor = text.find('"pad_id"')
    start = text.rfind("{", 0, anchor + 1) if anchor != -1 else text.find("{")
    first_start = start
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(text[start:])
            if isinstance(obj, dict) and "forecasts" in obj:
                return obj
        except json.JSONDecodeError:
            pass
        start = text.find("{", start + 1)
    # Truncated-at-EOF repair: a response that is valid except for missing
    # closing braces at the very end gets them appended. Structural only —
    # no values are invented; anything else still fails loudly.
    if first_start != -1:
        for k in range(1, 4):
            try:
                obj, _ = decoder.raw_decode(text[first_start:] + "}" * k)
                if isinstance(obj, dict) and "forecasts" in obj:
                    print(f"NOTE: repaired truncated JSON by appending {k} closing brace(s)",
                          file=sys.stderr)
                    return obj
            except json.JSONDecodeError:
                continue
    raise ValueError("no parseable forecast object in response")


def main() -> None:
    horizon = json.loads((DATA_DIR / "snapshot.json").read_text())["horizon"]
    roster = {p["pad_id"]: {w["well_id"] for w in p["wells"]}
              for p in json.loads((DATA_DIR / "pads.json").read_text())}
    if os.environ.get("PAD_IDS"):
        wanted = [p for p in os.environ["PAD_IDS"].split(",") if p]
        unknown = sorted(set(wanted) - set(roster))
        if unknown:
            raise SystemExit(f"unknown PAD_IDS: {unknown}")
        roster = {p: roster[p] for p in wanted}
    out_dir = DATA_DIR / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)  # OUT_SUBDIR may be runs/<id>/forecasts

    ok, failed = 0, []
    for pad_id, expected_wells in sorted(roster.items()):
        raw_path = DATA_DIR / RAW_SUBDIR / f"{pad_id}.json"
        try:
            payload = parse_response(raw_path.read_text())
            forecasts = payload["forecasts"]
            missing = expected_wells - set(forecasts)
            if missing:
                raise ValueError(f"missing wells {sorted(missing)}")
            for well_id, phases in forecasts.items():
                if well_id not in expected_wells:
                    raise ValueError(f"unknown well {well_id}")
                for phase, arr in phases.items():
                    if len(arr) != horizon:
                        raise ValueError(f"{well_id}/{phase}: {len(arr)} months, want {horizon}")
                    if not all(isinstance(v, (int, float)) and v >= 0 for v in arr):
                        raise ValueError(f"{well_id}/{phase}: non-numeric or negative value")
        except Exception as e:  # noqa: BLE001 — every failure mode gets the same loud treatment
            failed.append(f"{pad_id}: {e}")
            continue
        (out_dir / f"{pad_id}.json").write_text(json.dumps(payload, indent=1))
        ok += 1

    print(f"promoted {ok}/{len(roster)} pads -> {out_dir}")
    for f in failed:
        print(f"FAILED {f}", file=sys.stderr)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
