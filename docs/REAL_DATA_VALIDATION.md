# Real-Well Holdout Validation

## What this runner measures

`run_real_holdout.py` truncates all eligible wells at one common calendar cutoff, builds SmartCast cohorts from truncated histories only, and scores the following holdout months against later reported production.

This is a **retrospective stabilized-history** test when run on a current revised download. It is not a reporting-lag study unless the source is an archived point-in-time snapshot.

## Recommended sources

Prefer official regulators that report monthly production by well. Colorado ECMC states that monthly oil, gas, and water production has been submitted by well since 1999 and provides downloadable production data. Avoid treating Texas lease-level production as measured well-level truth unless an explicit allocation method is part of the experiment.

Official starting points:

- Colorado ECMC downloadable data: https://ecmc.colorado.gov/data-maps/downloadable-data-documents
- Colorado production inquiry: https://ecmc.state.co.us/cogisdb/Production/ProdSearch
- Pennsylvania well production extracts: https://greenport.pa.gov/ReportExtracts/OG/OilGasWellProdReport
- Texas RRC data caveat and downloads: https://www.rrc.state.tx.us/resource-center/research/data-sets-available-for-download/

## Canonical input

```text
input_dir/
  history.csv  # well_id,month,oil,gas,water
  wells.csv    # optional well_id, basin, formation, lateral length, etc.
```

## Run

```bash
python scripts/run_real_holdout.py real_data/colorado \
  --source-name "Colorado ECMC monthly well production" \
  --source-url "https://ecmc.colorado.gov/data-maps/downloadable-data-documents" \
  --snapshot-date 2026-07-28 \
  --horizon 12 \
  --stabilization-lag 6
```

The stabilization lag keeps the scored holdout away from the newest, most revision-prone months. It does not reconstruct what was known at the historical cutoff.

## Evidence standard

Record the source, download date, data-as-of month, transformations, exclusions, cutoff, horizon, eligible/skipped wells, failures, zero policy, score decomposition, and paired confidence interval. Never tune a candidate on the same real holdout used for final acceptance.
