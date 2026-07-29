"""Deterministic release-manifest creation and verification."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

DEFAULT_EXCLUDED_PARTS = {".git", ".pytest_cache", "__pycache__"}
DEFAULT_EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".zip"}


@dataclass(frozen=True)
class ManifestVerification:
    ok: bool
    checked_files: int
    missing: tuple[str, ...]
    mismatched: tuple[str, ...]
    unexpected: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "checked_files": self.checked_files,
            "missing": list(self.missing),
            "mismatched": list(self.mismatched),
            "unexpected": list(self.unexpected),
        }


def _included(path: Path, root: Path, manifest_name: str) -> bool:
    rel = path.relative_to(root)
    if rel.as_posix() == manifest_name:
        return False
    if any(part in DEFAULT_EXCLUDED_PARTS for part in rel.parts):
        return False
    if path.suffix.lower() in DEFAULT_EXCLUDED_SUFFIXES:
        return False
    return path.is_file()


def iter_release_files(root: Path, manifest_name: str = "RELEASE_MANIFEST.json") -> Iterable[Path]:
    root = root.resolve()
    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        if _included(path, root, manifest_name):
            yield path


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(root: Path, *, built_at: str, manifest_name: str = "RELEASE_MANIFEST.json") -> dict:
    root = root.resolve()
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in iter_release_files(root, manifest_name)
    ]
    return {
        "schema_version": 1,
        "built_at": built_at,
        "file_count": len(files),
        "files": files,
    }


def write_manifest(root: Path, *, built_at: str, manifest_name: str = "RELEASE_MANIFEST.json") -> Path:
    root = root.resolve()
    output = root / manifest_name
    payload = build_manifest(root, built_at=built_at, manifest_name=manifest_name)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def verify_manifest(
    root: Path,
    *,
    manifest_name: str = "RELEASE_MANIFEST.json",
    reject_unexpected: bool = True,
) -> ManifestVerification:
    root = root.resolve()
    manifest_path = root / manifest_name
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {entry["path"]: entry for entry in payload["files"]}
    current = {
        path.relative_to(root).as_posix(): path
        for path in iter_release_files(root, manifest_name)
    }
    missing = tuple(sorted(set(expected) - set(current)))
    unexpected = tuple(sorted(set(current) - set(expected))) if reject_unexpected else ()
    mismatched: list[str] = []
    for rel in sorted(set(expected) & set(current)):
        entry = expected[rel]
        path = current[rel]
        if path.stat().st_size != int(entry["bytes"]) or file_sha256(path) != entry["sha256"]:
            mismatched.append(rel)
    ok = not missing and not mismatched and not unexpected
    return ManifestVerification(
        ok=ok,
        checked_files=len(expected),
        missing=missing,
        mismatched=tuple(mismatched),
        unexpected=unexpected,
    )
