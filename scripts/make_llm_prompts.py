"""Build one self-contained forecasting prompt per pad from the frozen
snapshot's TRAIN side only.

Each prompt = the well-forecasting skill + pad metadata + per-well monthly
history through the cutoff + a strict JSON output contract. The holdout
never enters a prompt; the process that answers these prompts is run with
no tools, so it cannot go find it either.

Usage: python scripts/make_llm_prompts.py [data_dir]
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else REPO / "benchmark_data")
SKILL_PATH = Path(os.environ.get("SKILL_PATH", REPO / "skill" / "SKILL.md"))
PROMPT_SUBDIR = os.environ.get("PROMPT_SUBDIR", "prompts")

NO_TOOLS_EVIDENCE = """There is no database access; the population evidence available to you is
the other wells on this pad plus your general knowledge of the basin and formation."""


CONTRACT = """
## Your task

Forecast each well's monthly production for the 12 months {first_holdout} through {last_holdout}
(the months immediately after the history above ends). Forecast every phase the well
actually reports (a phase that is entirely blank in the history should be omitted, not zeroed).

Follow the six questions in the skill for every well. The calculator/echo tooling the
skill mentions is not available here — do the arithmetic yourself, and pressure-test
your own consequences (implied next-12 vs trailing-12, year-1 effective decline) before
committing. {evidence}

## Output

Return ONLY a JSON object, no markdown fences, no prose outside it:

{{
  "pad_id": "{pad_id}",
  "forecasts": {{
    "<well_id>": {{"oil": [12 monthly bbl], "gas": [12 monthly mcf], "water": [12 monthly bbl]}}
  }},
  "params": {{
    "<well_id>": {{"oil": {{"qi_per_month": n, "di_nominal_monthly": n, "b": n, "anchor": "YYYY-MM"}}}}
  }},
  "rationale": "per well: months struck and why, trust judgment, qi/anchor source, Di source, b source, consequence check"
}}

Arrays are calendar volumes for {first_holdout}..{last_holdout} in order. Include every
well listed above under "forecasts". "params" documents your committed curve per phase
where you used one (gas may ride GOR off oil — then document that instead). The
rationale is the audit trail; keep it tight but complete.
"""


def main() -> None:
    pads = json.loads((DATA_DIR / "pads.json").read_text())
    skill = SKILL_PATH.read_text()
    out_dir = DATA_DIR / PROMPT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)  # PROMPT_SUBDIR may be runs/<id>/prompts
    snapshot = json.loads((DATA_DIR / "snapshot.json").read_text())
    first_h, last_h = snapshot["holdout_months"][0][:7], snapshot["holdout_months"][-1][:7]

    for pad in pads:
        pad_id = pad["pad_id"]
        by_well: dict[str, list[dict]] = defaultdict(list)
        with open(DATA_DIR / "train" / f"{pad_id}.csv") as f:
            for row in csv.DictReader(f):
                by_well[row["well_id"]].append(row)

        lines = [skill, "\n---\n", "# The pad\n"]
        lines.append(
            f"Pad {pad_id} — operator {pad['operator']}, {pad['basin']} basin, "
            f"{pad['play']} play, {pad['county']} Co., {pad['state']}. "
            f"{len(pad['wells'])} co-developed horizontal wells.\n"
        )
        for w in pad["wells"]:
            lines.append(
                f"- **{w['well_id']}** {w['name']} — {w['interval']}, "
                f"{w['lateral_ft'] and int(w['lateral_ft'])} ft lateral, first prod {w['first_prod']}"
            )
        lines.append("\n# Production history (monthly volumes; blank = not reported)\n")
        for well_id in sorted(by_well):
            lines.append(f"\n## {well_id}\n")
            lines.append("month | oil bbl | gas mcf | water bbl")
            lines.append("--- | --- | --- | ---")
            for row in by_well[well_id]:
                lines.append(f"{row['month'][:7]} | {row['oil']} | {row['gas']} | {row['water']}")

        lines.append(CONTRACT.format(pad_id=pad_id, first_holdout=first_h,
                                     last_holdout=last_h, evidence=NO_TOOLS_EVIDENCE))
        (out_dir / f"{pad_id}.md").write_text("\n".join(lines))

    print(f"wrote {len(pads)} prompts -> {out_dir}")


if __name__ == "__main__":
    main()
