from pathlib import Path

from forecast_benchmark.release_integrity import verify_manifest, write_manifest


def test_manifest_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.txt").write_text("beta", encoding="utf-8")
    write_manifest(tmp_path, built_at="2026-07-28")
    clean = verify_manifest(tmp_path)
    assert clean.ok
    assert clean.checked_files == 2

    (tmp_path / "a.txt").write_text("changed", encoding="utf-8")
    tampered = verify_manifest(tmp_path)
    assert not tampered.ok
    assert tampered.mismatched == ("a.txt",)


def test_manifest_rejects_unexpected_files(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    write_manifest(tmp_path, built_at="2026-07-28")
    (tmp_path / "extra.txt").write_text("extra", encoding="utf-8")
    result = verify_manifest(tmp_path)
    assert not result.ok
    assert result.unexpected == ("extra.txt",)


def test_runtime_dependencies_include_scipy_for_default_anchor():
    import tomllib

    root = Path(__file__).resolve().parents[1]
    payload = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    deps = payload["project"]["dependencies"]
    assert any(str(dep).lower().startswith("scipy") for dep in deps)
