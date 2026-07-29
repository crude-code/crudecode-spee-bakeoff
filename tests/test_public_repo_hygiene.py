"""Guards the repo's public boundary: nothing private in a tracked file.

Scoped to files git actually tracks, not everything on disk. The snapshot in
benchmark_data/ is real production data and is *supposed* to contain terms
on this blocklist — it is gitignored and never committed. Scanning it would
fail the build for data that was never at risk, and would train everyone to
ignore this test.
"""
import subprocess
from pathlib import Path

import pytest

# Build these strings indirectly so this test does not trip on its own source.
BLOCKLIST = [
    "D" + "DE",
    "Env" + "erus",
    "Land" + "trac",
    "AWS_SECRET" + "_ACCESS_KEY",
    "SUPABASE_SERVICE" + "_ROLE_KEY",
    "postgres" + "ql://",
    "/home" + "/ubuntu",
]

TEXT_EXTENSIONS = {".md", ".py", ".toml", ".yml", ".yaml", ".txt"}


def _tracked_files(root: Path) -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=root, capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout — nothing tracked to scan")
    return [root / p for p in out.split("\0") if p]


def _tracked_text_files(root: Path) -> list[Path]:
    return [p for p in _tracked_files(root) if p.suffix in TEXT_EXTENSIONS and p.is_file()]


def test_tracked_files_contain_no_private_boundary_terms():
    root = Path(__file__).resolve().parents[1]
    files = _tracked_text_files(root)
    assert files, "expected some tracked text files to scan"

    offenders = [
        f"{path.relative_to(root)} contains blocked boundary term {term!r}"
        for path in files
        for term in BLOCKLIST
        if term in path.read_text(errors="ignore")
    ]
    assert offenders == []


def test_snapshot_and_env_are_not_tracked():
    """The two things that must never be committed, checked directly rather
    than trusting .gitignore — `git add -f` overrides ignore rules.

    This checks all tracked paths, not just tracked text files. A forced-added
    CSV snapshot or binary artifact under benchmark_data/ is just as unsafe as
    a text file.
    """
    root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in _tracked_files(root):
        rel = path.relative_to(root)
        parts = rel.parts
        if parts and parts[0] == "benchmark_data":
            offenders.append(str(rel))
        if path.name == ".env":
            offenders.append(str(rel))
    assert offenders == []
