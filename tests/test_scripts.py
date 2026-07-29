import json
import os
import subprocess
import sys
from pathlib import Path


def _write_snapshot(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "snapshot.json").write_text(json.dumps({"horizon": 2}))
    (root / "pads.json").write_text(json.dumps([
        {"pad_id": "pad-a", "wells": [{"well_id": "well-a"}]},
        {"pad_id": "pad-b", "wells": [{"well_id": "well-b"}]},
    ]))


def test_collect_llm_forecasts_supports_pad_subset(tmp_path):
    data = tmp_path / "benchmark_data"
    _write_snapshot(data)
    raw = data / "runs" / "smoke" / "raw"
    raw.mkdir(parents=True)
    (raw / "pad-a.json").write_text(json.dumps({
        "pad_id": "pad-a",
        "forecasts": {"well-a": {"oil": [1, 1], "gas": [2, 2], "water": [3, 3]}},
    }))

    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, "RAW_SUBDIR": "runs/smoke/raw", "OUT_SUBDIR": "runs/smoke/forecasts", "PAD_IDS": "pad-a"}
    result = subprocess.run(
        [sys.executable, str(repo / "scripts" / "collect_llm_forecasts.py"), str(data)],
        env=env, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert (data / "runs" / "smoke" / "forecasts" / "pad-a.json").is_file()


def test_run_llm_imports_without_optional_anthropic_dependency():
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", "import scripts.run_llm as r; print(r.load_config)"],
        cwd=repo, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
