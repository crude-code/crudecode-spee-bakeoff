# Python arm experiments

This repo's public board intentionally has one deterministic arm:

```text
arps_bounded_b
```

Bill asked whether there is improvement to be had in the Python function. The answer should come from the same benchmark discipline as the LLM comparison, not from eyeballing curves or adding a model zoo. This document records the deterministic variants added for testing and how to run them on the expanded snapshot.

## Status

The official board arm is still `arps_bounded_b`. It was not replaced.

What changed is that the Python arm now has a **zero-LLM-cost experiment board**:

```bash
python scripts/run_python_arm_experiments.py benchmark_data
```

or:

```bash
make arps-experiments
```

The runner scores every deterministic variant through the same provider seam, calendar cutoff, holdout, metrics, and pad summing as the public 1v1 benchmark. It writes either Markdown or full JSON:

```bash
python scripts/run_python_arm_experiments.py benchmark_data --out results/python-arm.md
python scripts/run_python_arm_experiments.py benchmark_data --json --out results/python-arm.json
```

## Variants

| Arm | What it tests | Why it matters |
|---|---|---|
| `arps_bounded_b` | Current official v3 function: trailing 36-month post-peak fit, downtime strike, relative-error refit, uptime haircut | Baseline deterministic contender |
| `arps_inner_backtest_routed` | Uses only pre-cutoff history to run an inner hindcast and choose among the fixed deterministic variants | Tests commercial-style auto-selection/current-segment logic without peeking at the benchmark holdout |
| `arps_lag_skip_2` | Excludes the freshest 2 reported months from both fit and uptime measurement | Tests the reporting-lag hypothesis without changing the cutoff/horizon |
| `arps_lag_skip_3` | Same, but excludes 3 months | Tests whether the lag window is wider than two months |
| `arps_b_cap_1p0` | Same method, but caps the b grid at 1.0 | Tests whether the 36-month current-regime fit should prohibit transient-style b > 1 tails |
| `arps_lag_skip_2_b_cap_1p0` | Combines the two highest-signal research-backed variants | Candidate challenger if the expanded board confirms both effects |
| `arps_window_24` | Same method, 24-month fit window | Tests whether 36 months is too stale for current-regime forecasting |
| `arps_window_48` | Same method, 48-month fit window | Tests whether 36 months is too sensitive to noisy late allocation |
| `arps_all_post_peak` | Same method, full post-peak history | Regression check against the older whole-life style |
| `arps_no_uptime_haircut` | Capacity curve without multiplying by measured uptime | Tests whether the uptime haircut is helping or overpricing non-recurring downtime |
| `arps_recent_low_guard` | Falls back to recent reported tail if the final two months are both below the downtime threshold | Tests the known shut-in-at-cutoff failure mode |

## Own-research addition: inner-backtest routing

A second research pass looked at commercial DCA behavior and automated-DCA patterns. The useful idea was not adding ML; it was **auto-selecting the usable current segment and validating by hindcast**. Harmony-style commercial tools select subsets/outliers automatically, and automated-DCA work points toward forecasting from the latest production segment.

Implemented arm:

```text
arps_inner_backtest_routed
```

It withholds the freshest months inside the training history, scores the fixed deterministic variants on that inner holdout, then refits the best pre-cutoff variant on the full history. The benchmark holdout remains unseen. This is the safest way to test whether the Python arm can improve beyond a fixed 36-month window without hand-tuning to the real board.

See `docs/OWN_RESEARCH_IMPLEMENTATION.md`.

## Reporting-lag diagnostic

The research's strongest operational claim is that the freshest public/state production months can be incomplete and later revised upward. That can hit the current function twice: the last months anchor fit level/slope, and the same depressed months enter the reported/capacity uptime haircut.

This pass adds a diagnostic script:

```bash
python scripts/diagnose_reporting_lag.py benchmark_data
python scripts/diagnose_reporting_lag.py benchmark_data --months-back 9 --out results/reporting-lag.md
python scripts/diagnose_reporting_lag.py benchmark_data --json --out results/reporting-lag.json
```

or:

```bash
make lag-diagnostic
```

It computes `reported / fitted_capacity` for the training months nearest the cutoff, grouped by phase and months-before-cutoff. A broad dip in offsets 1-3 is evidence that reporting lag may be contaminating the fit and uptime haircut. It is not proof by itself; the proof is whether `arps_lag_skip_2` / `arps_lag_skip_3` improve the declared expanded-board metric.

## What the research changed

Useful and implemented:

1. **Recent reporting lag.** Added diagnostic plus `arps_lag_skip_2` and `arps_lag_skip_3` variants. These keep the forecast origin at the real cutoff, but fit and measure uptime on the last reliable months before the lag window.
2. **b cap at 1.0.** Added `arps_b_cap_1p0`. The official grid still runs 0.30-1.30, but the capped variant is now one scored experiment.
3. **Combined candidate.** Added `arps_lag_skip_2_b_cap_1p0` to test the most plausible a-priori deterministic challenger without hand-editing the official arm.
4. **Terminal decline note.** Documented as a future blocker for EUR/long-horizon work, not relevant enough to alter the 12-month board now.

Intentionally not changed:

- Relative-error refit stays. The research supports it for multiplicative production/allocation noise.
- Official `arps_bounded_b` stays stable until the expanded board selects a replacement by a declared metric.
- No probabilistic scoring is reintroduced in this phase.

## What I learned from synthetic stress testing

The current v3 function is strong on the exact cases it was designed for:

- clean hyperbolic decline
- early transient / late shallow regime shift
- isolated downtime months inside the fit window
- reported zero months that should be struck from the capacity fit but priced through uptime

The two highest-value experiments are now:

1. **Reporting-lag skip.** If the newest 2-3 training months are depressed by reporting lag, the official arm can under-forecast because both fit and uptime are pulled down.
2. **b cap 1.0.** If the trailing window is meant to represent the current/boundary-dominated regime, b > 1 may be buying tail flatness from late noise.

The previous `arps_recent_low_guard` remains useful, but it answers a different question: current status/shut-in recognition at the cutoff. It can help true shut-ins and hurt temporary dips that recover.

## Recommendation before the 70/1000-well board

Do not tune the official `arps_bounded_b` function again until the expanded board is run.

Run the Python experiment board and reporting-lag diagnostic first. The larger test set should answer these questions:

1. Is there a systematic reported/capacity dip in the last 2-3 months before cutoff?
2. Does `arps_lag_skip_2` or `arps_lag_skip_3` improve the selected pad-level metric?
3. Does `arps_b_cap_1p0` reduce tail optimism without damaging legitimate shale tails?
4. Does `arps_lag_skip_2_b_cap_1p0` beat either change alone?
5. Does `arps_recent_low_guard` fix more true shut-ins than it damages recovering frac-hit/downtime wells?
6. Which score should choose the Python arm: pad-level oil SPEE, all-phase pad SPEE, or a declared composite?

The final Python-vs-LLM comparison should not silently pick the Python variant that won the old 20-pad board. Use the expanded set, declare the selection metric, and then freeze the Python arm before comparing to fresh LLM runs.

## Bottom line

This adds the missing Python-function work without blowing up the repo:

- no LLM/API cost
- no probabilistic work yet
- no new production dependency
- no hidden CrudeCode import
- same scorer as the real benchmark
- research-backed variants are explicit and auditable

The official contender remains stable, but Bill now has an actual deterministic workbench to run against 70 wells and then 1000 wells.
