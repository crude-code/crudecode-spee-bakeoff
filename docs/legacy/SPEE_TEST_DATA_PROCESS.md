# SPEE Test Data Process Checklist

Use this checklist when the committee sends the test data package.

## Intake

- Save the raw package outside the repo if it contains real well data.
- Do not commit raw data, normalized `spee_data/`, `benchmark_data/`, `.env`, or generated submissions if they contain real well identifiers.
- Record the package received date and any committee instructions in a local-only note.

## Normalize

```bash
python scripts/prepare_spee_input.py \
  --monthly <monthly csv> \
  --headers <header csv> \
  --out spee_data
```

Then:

```bash
python scripts/validate_spee_input.py spee_data
```

## Strict Auto smoke run

```bash
python scripts/run_spee_strict_auto.py spee_data --horizon 12
python scripts/validate_spee_input.py spee_data \
  --submission submissions/strict_auto_forecast.csv \
  --horizon 12
```

Check:

- `results/strict_auto_run.json` has reasonable runtime.
- `results/strict_auto_failures.md` does not show unexpected provider failures.
- Any `phase_unavailable` rows reflect true missing phase data, not parser mistakes.

## Vendor Best dry path

Only after an LLM run exists:

```bash
python scripts/run_spee_vendor_best.py spee_data \
  --llm-dir benchmark_data/runs/<run_id>/forecasts
```

Use `--fallback-strict-auto` only if that policy is approved and documented.

## Before sending anything back

- Reconcile column names against the official committee output schema.
- Run `python scripts/validate_spee_input.py` against the exact CSV to be sent.
- Keep the run log and failure report with the submission package.
- Do not include holdout/scoring-only files unless requested.
