# Bake-Off Day Runbook

## Before August 13

- Pass the committee test-data process.
- Freeze the exact raw-to-canonical aliases and output mapping.
- Record the final horizon, common forecast start, economic cutoff, terminal-decline instruction, and submission deadlines.
- Run `make check`, `make stress`, and `make throughput` on the competition laptop.
- Keep two clean copies of the repository and one offline Python wheel/cache.
- Prepare an empty `raw/`, `spee_data/`, `submissions/`, and `results/` directory.

## T+00:00 — Receive data

1. Save the untouched files under `raw/original/`.
2. Record filenames, sizes, and SHA-256 hashes.
3. Copy files to `raw/working/`; never edit the originals.
4. Read the committee README and sample output before running forecasts.

## T+00:15 — Normalize and validate

```bash
python scripts/prepare_spee_input.py --monthly <monthly.csv> --headers <headers.csv> --out spee_data --gap-policy nan --duplicate-policy error
python scripts/validate_spee_input.py spee_data
```

Stop on duplicate rows, negative volumes, invalid dates, unknown required columns, or unexpected well counts. Resolve format issues explicitly and preserve the manifest from every attempt.

## T+00:35 — Strict Auto

```bash
python scripts/run_spee_strict_auto.py spee_data --horizon <H> --forecast-start <YYYY-MM-01>
python scripts/validate_spee_input.py spee_data --submission submissions/strict_auto_forecast.csv --horizon <H>
python scripts/render_committee_submission.py submissions/strict_auto_forecast.csv --mapping <frozen-mapping.json> --out submissions/strict_auto_committee.csv
```

Archive the canonical CSV, committee CSV, run log, diagnostics, failures, manifest, mapping, and terminal console output. Do not rerun with changed settings and still call it the same Strict Auto entry.

## T+01:30 — Vendor Best triage

1. Open `results/strict_auto_review_queue.csv`.
2. Prioritize high review score, primary phase, material entrance rate, terminal disruptions, ratio volatility, and thin histories.
3. Review a targeted subset; do not manually reforecast every well.
4. Put approved complete trajectories in one override JSON package.
5. Run Vendor Best and validate:

```bash
python scripts/run_spee_vendor_best.py spee_data --horizon <H> --forecast-start <YYYY-MM-01> --overrides approved_overrides.json
python scripts/validate_spee_input.py spee_data --submission submissions/vendor_best_forecast.csv --horizon <H>
python scripts/render_committee_submission.py submissions/vendor_best_forecast.csv --mapping <frozen-mapping.json> --out submissions/vendor_best_committee.csv
```

## Final pre-submit checks

- exact header and column order;
- exact well count and phase coverage;
- exact forecast month count per well/phase;
- no duplicate keys;
- no negative, NaN, or infinite values;
- common forecast start is correct;
- run log model and submission type are correct;
- file opens in a separate environment;
- SHA-256 recorded after final write;
- submit with time buffer, then preserve receipt/confirmation.

## Failure policy

Never silently fill a forecast failure with zero. The failure report must be empty or every exception must be resolved before submission. If SmartCast fails on one phase, use the declared legacy fallback or an approved Vendor Best trajectory and record it.
