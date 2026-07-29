# Day 1 Real-Well Board Audit

## Verdict

Do not run the locked evaluation set with the original `extract_board.py` and
`score_board.py`. The overall design is correct, but the implementation has
several issues that can change the score or invalidate the locked-test claim.

## Release blockers in the submitted scripts

1. **The scorer cannot score the dev pool.** It hardcodes `role == "eval"`, so
   following the runbook would expose the locked set during iteration.
2. **Calendar gaps are compressed out of time.** The extractor emits only rows
   present in the warehouse, and the scorer discards month labels and replaces
   them with synthetic consecutive months. A missing month becomes a shorter
   decline history rather than an explicit missing calendar month.
3. **The history buckets use database row counts, not elapsed calendar months.**
   Missing or duplicate rows can move wells into the wrong 7-12/13-24/etc.
   bucket.
4. **Zero forecasts can disappear from the metric.** The monthly log-error mask
   requires both forecast and actual to be positive. An arm that predicts zero
   against positive production is silently omitted for that month.
5. **Forecast failures can improve an arm's score.** Point estimates are
   computed on each arm's available values rather than a common paired set, and
   exceptions are skipped.
6. **The claimed cumulative metric is not reported.** Cumulative phase errors
   are calculated and stored but never aggregated into a headline score.
7. **Duplicate selection uses the holdout stream.** Full-series hashes through
   June 2025 determine which records are excluded. Sample membership should not
   depend on answer-period values.
8. **Exact identifier duplicates are reported but not quarantined.** Only exact
   production-series duplicates are removed.
9. **The documentation claims near-duplicate detection, but none is
   implemented.**
10. **Quotas are assigned before duplicate removal.** Deduplication can reduce
    eval/cohort/dev below the requested counts without backfilling.
11. **The scorer uses hardcoded Claude/Linux source paths.** It will not run in
    Jonathan's Windows/VS Code checkout without manual path edits.
12. **The scorer replaces real month labels with dates starting in 2015.** This
    can break any date- or gap-aware model behavior.
13. **Eligibility requires both oil and gas to be positive.** That selects
    against dry-gas and very-low-secondary-phase wells and can distort the
    intended play population.
14. **Monthly production is not aggregated before counting.** Multiple source
    rows for the same well/month can distort train and holdout lengths.

## Corrections in v2

- Default scoring role is `dev`.
- Locked eval requires `--role eval --confirm-locked-eval`.
- Monthly production is aggregated to one row per well/calendar month.
- Missing calendar months are emitted explicitly as NaN rows.
- Visible-history buckets use elapsed calendar months.
- Exact duplicate clustering uses canonical identifiers and pre-cutoff series
  only; holdout values do not affect inclusion.
- Deduplication occurs before role assignment, and quotas are validated.
- Scoring uses actual-driven masks, with explicit metric-only treatment of
  forecast zeros.
- Runs fail closed on missing major-phase forecasts unless a debugging override
  is explicitly supplied.
- Headline scores use common paired wells across requested arms.
- Major, all-phase, cumulative-major and cumulative-all-phase metrics are all
  reported and labelled separately.
- Clustered bootstrap, bootstrap win probability and leave-one-well-out
  influence are reported.
- Actual calendar labels are passed to SmartCast.
- Source paths are explicit CLI/environment inputs rather than Claude paths.
- Board and code hashes are written into the comparison report.

## Safe execution order

```powershell
# Private-data protection first
Add-Content .gitignore "board/private/"
Add-Content .gitignore "board/results/"

# Extract once
python extract_board_v2.py --eval 1000 --cohort 400 --dev 300

# Iterate only on dev
python score_board_v2.py `
  --role dev `
  --legacy-src "C:\path\to\forecast-benchmark\src" `
  --smartcast-src ".\src" `
  --boot 10000

# Freeze code and sampler, commit/tag privately, then run eval once
python score_board_v2.py `
  --role eval `
  --confirm-locked-eval `
  --legacy-src "C:\path\to\forecast-benchmark\src" `
  --smartcast-src ".\src" `
  --boot 20000
```

## Validation performed here

- Both v2 scripts compile.
- The scorer was executed end-to-end against the actual original Arps and
  SmartCast packages using a synthetic calendar-complete board.
- Dev-role scoring, four metric families, paired clustered bootstrap, segment
  output and provenance reports completed successfully.
- Eval execution correctly refuses to run without the explicit locked-eval
  confirmation flag.

Database extraction itself could not be executed in this environment because
it has no access to the private Datum/Postgres warehouse. The SQL therefore
still needs one dev-size warehouse smoke run before the 1,000-well extraction.
