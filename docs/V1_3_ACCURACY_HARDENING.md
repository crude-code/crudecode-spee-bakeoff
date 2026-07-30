# SmartCast v1.3 accuracy hardening audit

## Decision

Claude's v1.2 proposal was directionally correct but incomplete. Two changes were retained:

1. restore the original SciPy bounded-Arps control as the safety anchor;
2. repair non-finite forecast values before terminal-decline enforcement.

The package was not accepted wholesale because additional defects materially affected both accuracy and the credibility of the ablation board.

## Defects found after v1.2

### Target wells silently used the global cohort

The real-board scripts constructed metadata only for cohort-pool wells. Target development/evaluation IDs were absent. `CohortLibrary._group()` therefore returned `__global__` for every target, despite the board's formation-level play mapping.

Effect: Delaware, Midland, Haynesville, Eagle Ford, DJ, and Williston target wells could receive a mixed-play cohort shape. v1.3 supplies metadata for all board wells, disables global fallback in conservative profiles, and can fail closed when target metadata is missing.

### Terminal ablation did not work

`use_terminal_decline` existed but forecast call sites still enforced the terminal rule. v1.3 centralizes finalization so the switch controls candidate, anchor, recovery, fallback, cohort, and blended forecasts.

### The validation harness was too permissive

The earlier ablation harness reused one development set for broad comparison and could let an experimental-arm failure alter the common paired subset. v1.3 penalizes failures and uses a deterministic tune/confirmation split within play × history bucket.

### The default anchor was expensive

A zero-weight candidate path still fitted all candidate families. v1.3 detects control-only profiles and runs the frozen anchor directly. Shared caches avoid repeating identical fits across profiles.

### Release engineering drift

SciPy was not declared despite becoming a default dependency; package version and release manifest were stale; CLI tests wrote into tracked `results/`. All were corrected.

## Why no new model families were added

The real result did not show that the candidate library was too small. It showed that routing, fallback identity, cohort grouping, and validation were wrong. Adding more DCA equations before fixing those defects would increase selection variance and runtime without establishing accuracy.

The v1.3 profile board therefore stays small:

- exact SciPy control;
- terminal-tail variant;
- conservative candidate gate;
- conservative recovery branch;
- play-specific cohort branch;
- combined major-phase profile;
- a separate all-phase ratio experiment.

## Promotion protocol

Because aggregate results from the original 300 development wells have already been inspected, the preferred protocol extracts a new role=`confirm` pool after the frozen eval/cohort/dev role prefix. The extractor verifies that all prior assignments remain unchanged. The original 300 dev wells become tune data; the new confirm wells are used once.

When an external confirm pool is unavailable, the runner can use a deterministic play × history-bucket split, but it labels that path as a fallback rather than claiming it is fully pristine.

A profile is selected only on tune. It replaces the control only when confirmation shows:

- lower score;
- clustered-bootstrap 95% interval entirely below zero;
- minimum relative improvement;
- no catastrophic-rate regression;
- no material supported segment regression;
- no model failures.

The 1,000 evaluation wells remain locked until code and profile are frozen.

## Control identity

The packaged `arps_bounded_scipy_v1` was compared against the historical benchmark source on 120 randomized histories containing decline noise, downtime, and missing values. All 120 forecasts matched at tight numerical tolerance. The real-board selection script repeats this parity check on every scored development well and aborts if any mismatch appears.

## Operational recommendation

Until a profile passes the confirmation gates:

```text
Strict Auto: scipy_control
Vendor Best base: scipy_control
Experimental profiles: research only
```

This is not surrendering SmartCast. It prevents a weaker stack from replacing the known control while the useful layers are isolated and retested.
