"""Audit the public repo boundary before pushing forecast-benchmark.

This repo is public, while benchmark_data/ is intentionally private.  The
normal hygiene test protects the current tree; this script adds a maintainer
pre-push check that also looks through available Git history for accidentally
tracked benchmark snapshots or .env files.

Usage:
    python scripts/audit_public_history.py
    python scripts/audit_public_history.py --current-only

Run from anywhere inside the repo.  The history check only sees history present
in the local clone.  For a complete audit, run `git fetch --all --tags` first.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def repo_root() -> Path:
    try:
        out = _git(Path.cwd(), "rev-parse", "--show-toplevel").stdout.strip()
        return Path(out)
    except (OSError, subprocess.CalledProcessError):
        # Release archives intentionally do not contain .git.  Current-tree
        # auditing must still work from a fresh unzip.
        return Path(__file__).resolve().parents[1]


# Build terms indirectly so the source file does not trip simple string greps.
BLOCKLIST = [
    "D" + "DE",
    "Env" + "erus",
    "Land" + "trac",
    "AWS_SECRET" + "_ACCESS_KEY",
    "SUPABASE_SERVICE" + "_ROLE_KEY",
    "postgres" + "ql://",
    "/home" + "/ubuntu",
]

# Text-ish suffixes worth scanning for repo-boundary strings. Path-based checks
# below cover benchmark_data/ and .env regardless of suffix or binary/text type.
TEXT_SUFFIXES = {
    "", ".cfg", ".css", ".csv", ".env", ".html", ".ini", ".json",
    ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml",
}


def tracked_files(root: Path) -> list[Path]:
    proc = _git(root, "ls-files", "-z", check=False)
    if proc.returncode == 0:
        return [root / p for p in proc.stdout.split("\0") if p]
    excluded = {".git", ".pytest_cache", "__pycache__", ".venv", "venv"}
    return [p for p in root.rglob("*") if p.is_file() and not any(part in excluded for part in p.relative_to(root).parts)]


def current_tree_offenders(root: Path) -> list[str]:
    offenders: list[str] = []
    for path in tracked_files(root):
        rel = path.relative_to(root)
        parts = rel.parts
        if parts and parts[0] == "benchmark_data":
            offenders.append(f"tracked private snapshot path: {rel}")
        if path.name == ".env":
            offenders.append(f"tracked env file: {rel}")
        if path.is_file() and path.suffix in TEXT_SUFFIXES:
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            for term in BLOCKLIST:
                if term in text:
                    offenders.append(f"tracked boundary term {term!r} in {rel}")
    return offenders


def history_path_offenders(root: Path) -> list[str]:
    """Return prior commits that touched paths that must never be public.

    This intentionally focuses on path leaks, not arbitrary string history.
    A historical string search easily flags the audit code itself; path leaks
    are the high-risk event for this repo because benchmark_data/ is where
    private well snapshots are written.
    """
    # If the repo has no commits, `git log` exits 128.  Treat as no history.
    proc = _git(
        root,
        "log", "--all", "--name-only", "--pretty=format:%H", "--",
        "benchmark_data", ".env",
        check=False,
    )
    if proc.returncode != 0:
        return []

    offenders: list[str] = []
    current_commit: str | None = None
    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if len(line) == 40 and all(c in "0123456789abcdef" for c in line.lower()):
            current_commit = line
            continue
        offenders.append(f"{current_commit or '<unknown>'}: {line}")
    return offenders


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--current-only",
        action="store_true",
        help="skip Git-history scan and check only the current tracked tree",
    )
    args = parser.parse_args()

    root = repo_root()
    offenders = current_tree_offenders(root)
    if not args.current_only:
        offenders.extend(f"historical private path: {item}" for item in history_path_offenders(root))

    if offenders:
        print("PUBLIC BOUNDARY AUDIT FAILED", file=sys.stderr)
        for item in offenders:
            print(f" - {item}", file=sys.stderr)
        print("\nDo not push until these are removed from tracked files/history.", file=sys.stderr)
        return 1

    mode = "current tree" if args.current_only else "current tree + available history"
    print(f"public boundary audit passed ({mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
