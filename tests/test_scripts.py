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


def test_adding_confirmation_role_preserves_existing_role_prefix():
    from copy import deepcopy
    from scripts.extract_real_board import assign_roles

    base = []
    for play in ("A", "B"):
        for bucket in ("b1", "b2"):
            for i in range(20):
                base.append({
                    "play": play,
                    "bucket": bucket,
                    "ord_hash": f"{play}-{bucket}-{i:03d}",
                })

    old = deepcopy(base)
    assign_roles(old, n_eval=12, n_cohort=8, n_dev=8, n_confirm=0)
    old_roles = {
        w["ord_hash"]: w["role"]
        for w in old if w["role"] in {"eval", "cohort", "dev"}
    }

    extended = deepcopy(base)
    assign_roles(extended, n_eval=12, n_cohort=8, n_dev=8, n_confirm=10)
    new_roles = {w["ord_hash"]: w["role"] for w in extended}
    assert all(new_roles[key] == role for key, role in old_roles.items())
    assert sum(w["role"] == "confirm" for w in extended) == 10


def test_adding_second_confirmation_preserves_first_confirmation_pool():
    from copy import deepcopy
    from scripts.extract_real_board import assign_roles

    base = []
    for play in ("A", "B"):
        for bucket in ("b1", "b2"):
            for i in range(40):
                base.append({
                    "play": play,
                    "bucket": bucket,
                    "ord_hash": f"{play}-{bucket}-{i:03d}",
                })

    first = deepcopy(base)
    assign_roles(first, n_eval=12, n_cohort=8, n_dev=8, n_confirm=10)
    frozen = {
        w["ord_hash"]: w["role"]
        for w in first if w["role"] in {"eval", "cohort", "dev", "confirm"}
    }

    extended = deepcopy(base)
    assign_roles(
        extended,
        n_eval=12,
        n_cohort=8,
        n_dev=8,
        n_confirm=10,
        n_confirm2=16,
    )
    current = {w["ord_hash"]: w["role"] for w in extended}
    assert all(current[key] == role for key, role in frozen.items())
    assert sum(w["role"] == "confirm2" for w in extended) == 16
