#!/usr/bin/env python3
"""Write or verify RELEASE_MANIFEST.json from a source checkout or unpacked release."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forecast_benchmark.release_integrity import verify_manifest, write_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--verify", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--built-at", default="2026-07-28")
    parser.add_argument("--allow-unexpected", action="store_true")
    args = parser.parse_args()
    if args.write:
        path = write_manifest(args.root, built_at=args.built_at)
        payload = json.loads(path.read_text(encoding="utf-8"))
        print(json.dumps({"written": str(path), "file_count": payload["file_count"]}, indent=2))
        return 0
    result = verify_manifest(args.root, reject_unexpected=not args.allow_unexpected)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
