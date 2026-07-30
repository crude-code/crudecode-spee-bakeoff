# SmartCast v1.4.1 — Production-Provenance Hardening

## Why this release exists

The licensed warehouse exposes `productionreportedmethod` values including
`REPORTED` and `DCA`, plus `producingdays`.  Those fields were not previously
captured by the real-board extractor.

This is material, but the label must be interpreted carefully:

- Texas oil production is reported by lease, not by individual well.
- Commercial data products therefore allocate or estimate lease production to
  wells.
- A `DCA` method label can indicate a decline-curve-based allocation/estimation
  process. It must not automatically be described as a free-standing future
  forecast without the licensed data dictionary.

Either way, `DCA` months are not direct individual-well measurements. They can
make a well-level validation board reward agreement with the data vendor's
allocation method, particularly around downtime, restarts and new wells.

## The critical design choice: do not delete DCA rows

Dropping individual `DCA` rows would:

1. create artificial calendar gaps;
2. change well age and history buckets;
3. bias the sample toward states, phases and operators with well-level reports;
4. make a monthly sequence unlike either the warehouse or the competition
   package.

v1.4.1 therefore keeps volume sequences intact and gates **whole wells** by
provenance.

## New extraction modes

`extract_real_board.py` now accepts:

- `mixed` — existing behavior; no provenance restriction.
- `reported_holdout` — every calendar month in the scored 12-month holdout must
  classify as `REPORTED`; training may contain DCA/allocation months.
- `reported_full` — every calendar month from first production through the end
  of the holdout must classify as `REPORTED`.
- `reported_threshold` — configurable minimum reported fractions in training
  and holdout.

A well-calendar-month is classified as:

- `REPORTED` if every contributing warehouse row is reported;
- `DCA` if every contributing row is DCA;
- `MIXED` if methods differ;
- `UNKNOWN` if source rows have no usable method;
- `MISSING` if no warehouse row exists for that calendar month.

## Competition compatibility boundary

`productionreportedmethod` and `producingdays` are validation diagnostics only.
They are not written to `series.csv` and never enter SmartCast, SciPy Arps or
any submission forecast.

The competition model still sees only:

- monthly dates;
- oil;
- gas;
- water;
- committee-provided headers.

## Step 1 — audit the existing mixed board

```powershell
python scripts/audit_production_provenance.py `
  --board "C:\PATH\TO\board\private-v131" `
  --production-table "dde_remote.production" `
  --chunk-size 25 `
  --out "board/results/provenance-audit"
```

Review:

```powershell
Get-Content "board/results/provenance-audit/provenance_audit.json"
```

The audit reports row-level and well-month method shares by play, role and
major phase, plus the distribution of producing days by method. The private
`provenance_by_well.csv` must never be committed.

## Step 2 — smoke-test a reported-holdout board

```powershell
python scripts/extract_real_board.py `
  --eval 0 `
  --cohort 40 `
  --dev 40 `
  --confirm 80 `
  --provenance-mode reported_holdout `
  --out "board/private-reported-smoke" `
  --wells-table "dde_remote.wells" `
  --production-table "dde_remote.production" `
  --production-chunk-size 25 `
  --statement-timeout-ms 300000
```

If a play × history-bucket cell cannot fill, do not silently substitute another
play or relax the gate. The manifest must disclose the unavailable cell.

## Step 3 — build the fixed reported-holdout validation board

The v1.4 `gated_cohort_v1` profile is already frozen. This new board changes the
validation estimand; it is not another look at the earlier mixed-data
confirmation pools.

```powershell
python scripts/extract_real_board.py `
  --eval 0 `
  --cohort 400 `
  --dev 300 `
  --confirm 500 `
  --provenance-mode reported_holdout `
  --out "board/private-reported-holdout" `
  --wells-table "dde_remote.wells" `
  --production-table "dde_remote.production" `
  --production-chunk-size 25 `
  --statement-timeout-ms 300000
```

The board manifest must contain:

```json
"provenance_gate": {
  "mode": "reported_holdout",
  "min_holdout_reported_fraction": 1.0,
  "row_filtering": false
}
```

## Step 4 — spent-pool diagnostics on the new board

Use `role=dev` only for diagnostics; do not alter the already-frozen
`gated_cohort_v1` thresholds.

```powershell
python scripts/run_v14_tail_diagnostics.py `
  --board "board/private-reported-holdout" `
  --smartcast-src ".\src" `
  --roles "dev" `
  --boot 10000 `
  --out "board/results/reported-holdout-dev"
```

## Step 5 — one fixed confirmation

```powershell
python scripts/run_reported_holdout_confirmation.py `
  --board "board/private-reported-holdout" `
  --legacy-src "C:\Users\jonat\Downloads\forecast-benchmark-main\forecast-benchmark-main\src" `
  --smartcast-src ".\src" `
  --boot 30000 `
  --out "board/results/reported-holdout-confirm" `
  --report-only
```

The runner refuses a mixed board and tests only:

- exact original SciPy control;
- frozen `gated_cohort_v1`;
- untouched `role=confirm` wells.

## Interpretation matrix

| Mixed board | Reported holdout | Interpretation |
|---|---|---|
| gain | gain | cohort signal survives provenance correction |
| gain | no gain | earlier benefit likely tracks vendor allocation conventions |
| no gain | gain | cohort helps real operational noise but not allocated smooth data |
| no gain | no gain | keep exact SciPy; stop cohort promotion work |

## What must still be confirmed externally

Ask the data vendor for the formal definition of
`productionreportedmethod='DCA'`, including whether it means:

- lease-to-well allocation;
- interpolation between tests;
- decline-curve projection;
- another source-specific imputation method.

Ask the committee whether its package contains operator-supplied well volumes,
state-reported/allocated volumes, commercial-aggregator volumes, or a mixture,
and whether estimated months are flagged.
