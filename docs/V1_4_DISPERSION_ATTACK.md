# SmartCast v1.4 — Dispersion Attack

## Decision premise

The exact SciPy bounded-Arps model remains the production control. The only
experimental layer with replicated positive real-well evidence is formation/play
cohort shrinkage:

| Pool | SciPy SPEE | Cohort SPEE | Relative improvement |
|---|---:|---:|---:|
| confirm, 150 wells | 0.1771 | 0.1528 | 13.7% |
| confirm2, 300 wells | 0.2089 | 0.1971 | 5.6% |

Neither predeclared confirmation look crossed its original strict boundary, so
v1.4 does not promote the cohort layer by assertion. It creates one new,
mechanism-driven challenger and one fresh confirmation protocol.

The pool audit showed that 91–99% of the score came from the dispersion term,
not median bias. The next model therefore does not attempt a global bias
correction and does not add another decline-curve family. It limits the ability
of cohort shrinkage to create extreme well errors.

## Confirmed weaknesses in v1.3.1 cohort shrinkage

The old `cohort_conservative` label understated its authority:

- a seven-positive-month well could receive about 61% cohort weight;
- overall play/phase peer count was used as confidence even when only a few
  peers supported the specific future ages;
- no robust peer-dispersion gate existed;
- no target-to-cohort shape-similarity gate existed;
- arithmetic blending was used even though the competition error is measured
  in log space;
- cohort-to-SciPy divergence was not bounded;
- endpoint spikes could bypass the existing low/zero disruption flags.

## `gated_cohort_v1`

The new profile is deliberately narrow:

```text
exact frozen SciPy forecast
        +
formation/play cohort prior
        +
visible-history risk gates
```

It contains no candidate router, no recovery model, no ratio coupling, no
mixed-play fallback and no learned black-box selector.

### Gate and shrinkage rules

- play/formation metadata is mandatory;
- no global cohort fallback;
- at least 12 peer wells in the play/phase cohort;
- at least 8 peers at every forecast age in the first 12 months;
- cohort authority capped at 25%;
- peer normalized-rate log-MAD must be at most 0.40;
- target-to-cohort median absolute log-shape error must be at most 0.35;
- latest-month level must remain between 0.60× and 1.60× the preceding tail;
- endpoint log-return volatility must be at most 0.45;
- cohort forecast is capped to ±35% of SciPy month by month;
- first-year cumulative divergence is tapered beyond 18%;
- accepted shrinkage is blended geometrically in log-rate space.

Every rejected gate returns the exact SciPy forecast. The control is therefore
both the fallback and the limiting forecast.

## Accuracy and runtime design

Cohort-only profiles now use an anchor-plus-cohort fast path. They do not fit the
unused candidate curve board or run rolling-origin candidate selection. This is
an exact computational optimization, not a model approximation.

A deterministic 100-well × 3-phase × 360-month smoke produced 108,000 finite,
nonnegative forecast values with zero failures. The release does not claim a
1,000-well timing because the 1,000-well synthetic stress population included
pathological SciPy optimizer cases and exceeded the local validation timeout.
The 12-hour committee window still leaves substantial operational margin, but a
full package rehearsal on the actual contest machine remains mandatory.

## Diagnostic use of spent pools

Run:

```powershell
python scripts/run_v14_tail_diagnostics.py `
  --board "board/private-v131" `
  --smartcast-src ".\src" `
  --roles "dev,confirm,confirm2" `
  --boot 10000 `
  --out "board/results/v14-tail"
```

The runner compares:

- `scipy_control`;
- `cohort_conservative`;
- `gated_cohort_v1`.

It reports p90/p95/p99/max absolute log error, catastrophic rate, leave-one-out
SPEE influence, endpoint bands, segments, gate reasons and private per-well
records. Trimmed results are diagnostics only and never replace the official
score.

## One fresh confirmation

The earlier confirmation pools are spent. v1.4 adds `role=confirm3` after every
existing role and fixes a single profile in code. The extractor now preserves
`confirm2` as well; v1.3.1's prefix check did not include it.

Create one 500-well confirmation pool:

```powershell
python scripts/extract_real_board.py `
  --eval 1000 `
  --cohort 400 `
  --dev 300 `
  --confirm 150 `
  --confirm2 300 `
  --confirm3 500 `
  --out "board/private-v14" `
  --assert-role-prefix-from "board/private-v131/wells.csv" `
  --wells-table "dde_remote.wells" `
  --production-table "dde_remote.production" `
  --production-chunk-size 25 `
  --statement-timeout-ms 300000
```

Then run exactly one look:

```powershell
python scripts/run_v14_fixed_confirmation.py `
  --board "board/private-v14" `
  --legacy-src "C:\Users\jonat\Downloads\forecast-benchmark-main\forecast-benchmark-main\src" `
  --smartcast-src ".\src" `
  --boot 30000 `
  --out "board/results/v14-confirm3" `
  --report-only
```

The profile and role are hardcoded:

```text
profile = gated_cohort_v1
role    = confirm3
```

### Predeclared promotion rule

All gates must pass:

- at least 1% relative SPEE improvement;
- bootstrap probability of improvement at least 95%;
- no profile forecast failures;
- exact packaged-control parity;
- catastrophic-rate regression no greater than 0.5 percentage points;
- p95 absolute-log-error regression no greater than 0.02;
- cumulative-major regression no greater than 0.01;
- no supported play/history-bucket regression greater than 0.03;
- runtime no more than 2.5× the exact control.

This is a prospective one-sided decision rule matched to the competition loss,
not a retroactive weakening of the previous gate. There is no `confirm4` plan
for an unchanged profile.

## Deployment posture

Until `confirm3` passes:

```text
Strict Auto       scipy_control
Vendor Best base  scipy_control
Shadow challenger gated_cohort_v1
Locked eval       untouched
```

If `confirm3` passes, freeze the code and board identities before the single
locked-eval run. If it fails, keep SciPy and use the gated cohort only as a
Vendor Best disagreement signal.

## Research basis

The design follows the evidence rather than adding model families:

- SPEE's 2024 bake-off reported 15 vendors, 1,000+ wells and a 12-hour window,
  with operational upsets and short histories among the central automated-DCA
  challenges.
- Robust outlier handling has been shown to materially affect empirical DCA on
  shale production data.
- Hierarchical shale-well models support borrowing information across related
  wells while retaining well-level forecasts.
- Forecast-combination research warns that estimated combinations can fail
  under heavy-tailed errors; bounded conservative combinations are safer than
  unrestricted weights.

References:

- SPEE, *2024 Software Symposium Roadshow*:
  https://www.spee.org/pdf/2024-Software-Symposium-Roadshow.pdf
- Yehia et al., *Removing the Outlier from the Production Data for the Decline
  Curve Analysis of Shale Gas Reservoirs*, ACS Omega, 2022:
  https://doi.org/10.1021/acsomega.2c03238
- Lee and Mallick, *Bayesian Hierarchical Modeling: Application Towards
  Production Results in the Eagle Ford Shale of South Texas*.
- Wang et al., *Forecast combinations: an over 50-year review*, 2022:
  https://arxiv.org/abs/2205.04216
- Cheng, Wang and Yang, *Forecast Combination Under Heavy-Tailed Errors*, 2015:
  https://arxiv.org/abs/1508.06359
