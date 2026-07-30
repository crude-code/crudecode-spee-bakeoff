# SmartCast v1.5.0 — Selective Cohort Confirmation Candidate

## Evidence-backed model change

- Added `selective_cohort_v1`, the fixed policy selected on 749 spent wells.
- Exact SciPy remains the forecast for every gas-primary target, every non-DJ/
  Eagle Ford/Delaware target, and every secondary phase.
- The unchanged `cohort_conservative` oil forecast is used only for oil-primary
  targets in DJ, Eagle Ford, and Delaware.
- No provenance, producing-days, holdout result, candidate router, recovery
  model, ratio coupling, or black-box classifier enters routing.
- Added `run_v15_selective_confirmation.py`, fixed to one profile and the
  untouched `confirm3` role.

## Development evidence

On the 749-well spent policy board, the selected route improved major-phase
SPEE from 0.1719 to 0.1592, lowered catastrophic rate from 7.076% to 6.676%,
and lowered p95 absolute log error from 0.8805 to 0.8595. These are development
results only; production remains `scipy_control` until confirm3 passes.

# SmartCast v1.4.1 — Production-Provenance Hardening

## Validation-data changes

- Added literal classification of `productionreportedmethod` at the
  well-calendar-month level: REPORTED, DCA, MIXED, UNKNOWN or MISSING.
- Added whole-well provenance gates; individual DCA rows are never deleted.
- Added `reported_holdout`, `reported_full` and threshold extraction modes.
- Added private producing-day diagnostics without making them model features.
- Added `audit_production_provenance.py`.
- Added a fixed `gated_cohort_v1` confirmation runner that refuses mixed boards.
- Added provenance criteria and hashes to board manifests and board IDs.

## Interpretation guardrail

The release does not automatically call DCA-labelled warehouse rows future
forecasts. In Texas, the label may identify lease-to-well allocation or another
well-level estimation method. The licensed data dictionary controls the exact
meaning. Either way, provenance is now explicit and testable.

## Production posture

`scipy_control` remains the Strict Auto default. `gated_cohort_v1` can be
promoted only after a fixed confirmation on a provenance-gated board.

# SmartCast v1.4.0 — Dispersion Attack

## Scope

This release does not add another forecasting stack. It extends the only
replicated positive challenger—play-specific cohort shrinkage—and limits its
ability to create tail errors.

## Accuracy changes

- Added `gated_cohort_v1`: exact SciPy plus bounded cohort shrinkage only.
- Capped cohort weight at 25%.
- Added age-specific peer-support and robust log-MAD gates.
- Added target/cohort visible-history similarity gating.
- Added endpoint level and volatility suppression.
- Added monthly and first-year cumulative divergence controls.
- Added geometric log-rate blending aligned with the log-error objective.
- Added an anchor-plus-cohort fast path that skips unused candidate fits.

## Validation changes

- Added a tail/influence diagnostic runner for spent dev/confirm/confirm2 pools.
- Added append-only `confirm3` extraction.
- Fixed prefix preservation so prior `confirm2` assignments are checked.
- Added a fixed-profile, fixed-role, one-look v1.4 confirmation runner.
- Predeclared a directional improvement gate with p95, catastrophic, cumulative,
  segment, parity, failure and runtime guardrails.

## Production posture

`scipy_control` remains the default until `gated_cohort_v1` passes the untouched
`confirm3` run. See `docs/V1_4_DISPERSION_ATTACK.md`.

# SmartCast v1.3.1 — Independent Confirmation Hardening

## Why this patch exists

The pre-registered `cohort_conservative` profile won the 300-well tune board and improved the first untouched 150-well confirmation score from 0.1771 to 0.1528. The clustered 95% interval was `[-0.0525, +0.0006]`, so the strict promotion gate correctly failed by a narrow margin.

The first confirmation pool is now spent. Re-running profiles or tuning against it would invalidate its confirmatory role. v1.3.1 therefore adds a second append-only role and a fixed-profile runner that cannot reselect among profiles.

## Changes

- Added append-only `role=confirm2` support to `extract_real_board.py`.
- Role-prefix verification now preserves the first `confirm` pool as well as eval/cohort/dev.
- Added `run_fixed_profile_confirmation.py`.
- The fixed runner scores exactly one pre-declared profile against the exact SciPy control.
- Promotion uses confirm2 alone; prior confirmation results are not pooled into the gate.
- Added cumulative-major non-regression to the confirmation gate.
- Retained paired scoring, fail-closed forecast handling, clustered bootstrap, catastrophic-error gating, segment gating, and exact control parity.

## Production posture

Deployment remains `scipy_control` until a frozen profile passes every independent confirm2 gate. The pre-declared candidate for the next run is `cohort_conservative`.

# SmartCast v1.3.0 — Accuracy-Hardened Release

## Release posture

The original SciPy bounded-Arps implementation is the production control. The richer SmartCast stack is experimental until it passes the real development confirmation gates.

## Kept from Claude v1.2

- Correct diagnosis that the safety anchor pointed at the fast linearized implementation instead of the original SciPy control.
- Frozen `arps_scipy_v1.py` control implementation.
- Repair of non-finite values before terminal-decline enforcement.

## Additional defects fixed

- Added the missing SciPy runtime dependency.
- Fixed stale package version and release provenance.
- Fixed target-well metadata omission that silently routed development/evaluation wells into the mixed global cohort.
- Made `use_terminal_decline` effective at every forecast-finalization path.
- Added fail-closed target metadata requirements for cohort-enabled production profiles.
- Added support-aware cohort weighting and an explicit global-fallback switch.
- Added shared fit/anchor caches for profile-board runtime.
- Added a direct control fast path.
- Prevented CLI tests from mutating tracked release evidence.
- Added profile/config provenance to Strict Auto and Vendor Best run logs.

## Validation protocol

`run_v13_selection.py` performs deterministic stratified development tuning and untouched confirmation. It requires:

- exact packaged-control parity with the original repository;
- no selected-profile model failures;
- a better confirmation point estimate;
- a clustered-bootstrap interval entirely below zero;
- at least 2% relative improvement by default;
- catastrophic-error non-regression;
- no material supported play/history-bucket regression.

When any gate fails, deployment remains `scipy_control`.

## No unsupported performance claim

The code, test suite, profile board, and selection protocol are validated locally. A new real-board score cannot be produced inside this release environment because the licensed board exists only in the operator's private database. The release therefore does not claim that any experimental v1.3 profile beats the 0.1209 control.
