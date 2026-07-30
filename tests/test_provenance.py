from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import pytest

from forecast_benchmark.provenance import (
    ProvenanceGate,
    classify_month_methods,
    fraction,
    provenance_counts,
)


def _load_extractor():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "extract_real_board.py"
    spec = importlib.util.spec_from_file_location("extract_real_board_provenance", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_month_method_classification_is_fail_closed():
    assert classify_month_methods(["REPORTED", "reported"]) == "REPORTED"
    assert classify_month_methods(["DCA", "dca"]) == "DCA"
    assert classify_month_methods(["DCA", "REPORTED"]) == "MIXED"
    assert classify_month_methods([None]) == "UNKNOWN"
    assert classify_month_methods([]) == "UNKNOWN"


def test_provenance_gate_never_filters_rows_only_wells():
    train = ["REPORTED"] * 8 + ["DCA"] * 2
    hold = ["REPORTED"] * 12
    assert ProvenanceGate("mixed").accepts(train, hold)
    assert ProvenanceGate("reported_holdout").accepts(train, hold)
    assert not ProvenanceGate("reported_full").accepts(train, hold)
    assert ProvenanceGate(
        "reported_threshold",
        min_train_reported_fraction=0.8,
        min_holdout_reported_fraction=1.0,
    ).accepts(train, hold)


def test_provenance_counts_include_missing_calendar_months():
    counts = provenance_counts(["REPORTED", "DCA", "MISSING", "MIXED"])
    assert counts["REPORTED"] == 1
    assert counts["DCA"] == 1
    assert counts["MISSING"] == 1
    assert fraction(counts, "REPORTED") == pytest.approx(0.25)


def test_extractor_query_fetches_provenance_but_series_contract_stays_volume_only():
    mod = _load_extractor()
    query = mod.build_production_query("dde_remote.production", [1, 2])
    assert "productionreportedmethod" in query
    assert "producingdays" in query
    source = (Path(mod.__file__).read_text(encoding="utf-8"))
    assert 'wr.writerow(["well_key", "month", "oil", "gas", "water", "split"])' in source


def test_reported_holdout_eligibility_is_well_level_not_row_deletion():
    mod = _load_extractor()
    header = {
        "wellid": 1,
        "api_uwi": "x",
        "play": "MIDLAND",
        "bucket_hint": "b1",
        "ord_hash": "a" * 32,
        "envbasin": "MIDLAND",
        "envinterval": "SPRABERRY",
        "laterallength_ft": 10000,
        "firstproddate": date(2023, 12, 1),
        "envoperator": "op",
    }
    raw = {1: {}}
    # Seven train months. One DCA training month is retained in the series.
    month = date(2023, 12, 1)
    for i in range(19):
        if i:
            month = mod.add_month(month)
        method = "DCA" if i == 0 else "REPORTED"
        raw[1][month] = {
            "volumes": (100.0 - i, 600.0 - i, 10.0),
            "method": method,
            "producingdays": 30.0,
            "source_rows": 1,
        }
    eligible, rows = mod.build_eligible(
        [header], raw, ProvenanceGate("reported_holdout")
    )
    assert len(eligible) == 1
    assert eligible[0]["train_dca_fraction"] > 0
    # The DCA month was not deleted or converted into a gap.
    assert rows[1][0][1] == 100.0
    strict, _ = mod.build_eligible([header], raw, ProvenanceGate("reported_full"))
    assert strict == []


def test_reported_holdout_confirmation_is_fixed_and_manifest_gated():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "run_reported_holdout_confirmation.py").read_text()
    assert 'PROFILE = "gated_cohort_v1"' in source
    assert 'ROLE = "confirm"' in source
    assert 'reported_holdout' in source
    assert 'min_holdout_reported_fraction' in source
