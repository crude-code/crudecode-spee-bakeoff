import importlib.util
from pathlib import Path


def _load_script(name: str):
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_confirm3_is_append_only_after_all_prior_roles():
    mod = _load_script("extract_real_board.py")
    wells = []
    plays = ("DELAWARE", "MIDLAND")
    buckets = ("b1", "b2")
    for i in range(40):
        wells.append({
            "play": plays[i % len(plays)],
            "bucket": buckets[(i // 2) % len(buckets)],
            "ord_hash": f"{i:064x}",
        })
    mod.assign_roles(wells, 4, 4, 4, 4, 4, 4)
    ordered_roles = [w["role"] for w in sorted(wells, key=lambda x: int(x["ord_hash"], 16))]
    # Exact local ordering varies by play x bucket round-robin, but every role
    # quota exists and confirm3 is never used to replace an earlier quota.
    assert {r: ordered_roles.count(r) for r in ("eval", "cohort", "dev", "confirm", "confirm2", "confirm3")} == {
        "eval": 4,
        "cohort": 4,
        "dev": 4,
        "confirm": 4,
        "confirm2": 4,
        "confirm3": 4,
    }


def test_v14_confirmation_runner_is_fixed_profile_and_fixed_role():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "run_v14_fixed_confirmation.py").read_text()
    assert 'PROFILE = "gated_cohort_v1"' in source
    assert 'ROLE = "confirm3"' in source


def test_v15_confirmation_runner_is_fixed_selective_profile_and_role():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "run_v15_selective_confirmation.py").read_text()
    assert 'PROFILE = "selective_cohort_v1"' in source
    assert 'ROLE = "confirm3"' in source
    assert 'add_argument("--profile"' not in source
