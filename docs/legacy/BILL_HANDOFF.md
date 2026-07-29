# Bill handoff — SPEE bake-off version

This package specializes the forecast benchmark for the bake-off process.

## What changed

- Added a canonical SPEE input path: `history.csv` + optional `wells.csv`.
- Added raw-package normalization: `scripts/prepare_spee_input.py`.
- Added input/submission validation: `scripts/validate_spee_input.py`.
- Added Strict Auto forecast generation: `scripts/run_spee_strict_auto.py`.
- Added Vendor Best forecast generation from precomputed LLM outputs: `scripts/run_spee_vendor_best.py`.
- Added run logs, failure CSVs, and failure markdown reports for both submission types.
- Kept the deterministic research workbench and reporting-lag diagnostics from the prior version.
- Kept probabilistic work out of this branch on purpose.

## Strategic shape

The repo now maps directly to the bake-off categories:

| Bake-off category | Repo path |
|---|---|
| Strict Auto Forecast | `run_spee_strict_auto.py` with `arps_bounded_b` |
| Vendor Best Forecast | precomputed LLM forecasts through `run_spee_vendor_best.py` |
| Test-data process | `prepare_spee_input.py` + `validate_spee_input.py` |
| Scale/runtime tracking | run-log JSON for every submission command |
| Failure handling | failure CSV + markdown report, no silent fills |

## Commands

```bash
python -m compileall -q .
python -m pytest -q
python scripts/audit_public_history.py --current-only
```

Normalize a test package:

```bash
python scripts/prepare_spee_input.py --monthly raw/monthly.csv --headers raw/headers.csv --out spee_data
python scripts/validate_spee_input.py spee_data
```

Strict Auto:

```bash
python scripts/run_spee_strict_auto.py spee_data --horizon 12
python scripts/validate_spee_input.py spee_data --submission submissions/strict_auto_forecast.csv --horizon 12
```

Vendor Best, after a completed LLM run:

```bash
python scripts/run_spee_vendor_best.py spee_data --llm-dir benchmark_data/runs/<run_id>/forecasts --horizon 12
```

## Note on the Python arm

The official Strict Auto model remains `arps_bounded_b`. The experimental Python variants and `arps_inner_backtest_routed` remain available for deciding whether to promote a better strict-auto default after the larger board runs.

Do not promote a variant until the expanded test set picks it by a declared metric.
