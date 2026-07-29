import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "audit_public_history.py"


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)


def _init_repo(root: Path) -> None:
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test User")


def test_history_audit_passes_clean_repo(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "README.md").write_text("clean public repo\n")
    _git(tmp_path, "add", "README.md")
    _git(tmp_path, "commit", "-q", "-m", "clean")

    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "public boundary audit passed" in result.stdout


def test_history_audit_catches_force_added_snapshot_path(tmp_path):
    _init_repo(tmp_path)
    private = tmp_path / "benchmark_data" / "foo.csv"
    private.parent.mkdir()
    private.write_text("api,oil\n42-001-00001,100\n")
    _git(tmp_path, "add", "-f", "benchmark_data/foo.csv")
    _git(tmp_path, "commit", "-q", "-m", "bad snapshot")

    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    assert "benchmark_data/foo.csv" in result.stderr


def test_history_audit_catches_tracked_env_current_tree(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=dummy\n")
    _git(tmp_path, "add", "-f", ".env")
    _git(tmp_path, "commit", "-q", "-m", "bad env")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--current-only"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    assert ".env" in result.stderr
