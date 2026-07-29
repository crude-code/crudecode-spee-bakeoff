# Own research implementation pass

This pass looked beyond the prior Claude/Fable notes and asked what commercial
DCA tools, public methodology notes, academic work, and practitioner forums
suggest for improving the deterministic Python arm without bloating the public
benchmark.

## Sources reviewed

High-signal sources:

- IHS Harmony Enterprise docs: automatic decline best-fit uses a selected data
  subset and outlier exclusion before initial fitting. Takeaway: auto-selection
  of usable history is normal commercial-tool behavior, not cheating.
- EIA state production methodology / DPR notes: recent reported production can
  be noisy, smoothed, lagged, or revised. Takeaway: freshness and reporting-lag
  diagnostics belong in the benchmark, especially near cutoff.
- WellDatabase decline-curve notes: the last few months should often be avoided
  at well level. Takeaway: last-2/3-month lag-skip variants are worth testing.
- RFour practical DCA guidance: shut-ins/cycling can produce garbage fits;
  terminal decline is required for long-horizon EUR. Takeaway: current-status
  lows and long-tail optimism should stay explicit failure modes.
- SPE/JPT unconventional DCA summary and probabilistic DCA review literature:
  simplified time-rate models do not capture all unconventional behavior, and
  b > 1 can create optimistic EUR tails. Takeaway: b-cap and current-regime
  testing are reasonable experiments.
- Recent automated-DCA patent language: segment production and forecast from the
  last production segment. Takeaway: a fixed 36-month window is a useful proxy,
  but a validation-selected current segment is the natural next deterministic
  experiment.
- Practitioner forums: working engineers repeatedly treat DCA as empirical,
  current-regime forecasting rather than proof of reservoir physics. Takeaway:
  avoid fancy ML until the baseline is honest and well-scored.

## Implemented from this pass

### 1. Inner-backtest routed Arps arm

New experimental arm:

```text
arps_inner_backtest_routed
```

The arm does not use the benchmark holdout. It splits the already-visible
training history again, withholds the latest training months as an internal
hindcast, scores the deterministic variants on that inner holdout, then refits
the winning variant on the full training history.

This is the programmatic version of the research pattern:

```text
auto-select usable history + verify by hindcast + forecast from the current segment
```

Candidate variants are intentionally small and a-priori:

```text
arps_bounded_b
arps_lag_skip_2
arps_lag_skip_3
arps_b_cap_1p0
arps_lag_skip_2_b_cap_1p0
arps_window_24
arps_window_48
arps_no_uptime_haircut
arps_recent_low_guard
```

The router uses a SPEE-style log-error score when possible and falls back to
relative/normalized error for sparse zero-heavy histories.

### 2. No promotion yet

The official deterministic board arm remains:

```text
arps_bounded_b
```

The routed arm is a scored experiment, not the new default. It should earn
promotion only on the expanded board by a declared metric.

## Why this is better than another hand-tuned knob

A fixed 36-month window is defensible, but not universally correct. Some wells
need a shorter current-regime segment; some need a longer/noise-resistant
segment; some should skip fresh reporting-lag months; some should cap b; some
should preserve current-status lows. Hard-coding one answer is weaker than
letting the visible history run a small internal hindcast.

The key discipline is that the selection is point-in-time safe: the benchmark
holdout remains unseen.

## How to run it

```bash
python scripts/run_python_arm_experiments.py benchmark_data --out results/python-arm.md
```

The result board will include `arps_inner_backtest_routed` next to the fixed
variants.

## Decision rule for Bill's expanded board

Before running fresh LLM calls, decide the Python arm using the expanded
snapshot only:

1. Declare the selection metric, preferably pad-level SPEE on oil or an agreed
   all-phase composite.
2. Run the reporting-lag diagnostic.
3. Run the Python-arm experiment board.
4. If `arps_inner_backtest_routed` wins or materially reduces worst-case misses,
   consider freezing it as the deterministic arm for the next LLM comparison.
5. If it does not win, keep `arps_bounded_b` and report that the more adaptive
   logic did not earn its complexity.

## Not implemented now

- Terminal decline switch. Required before EUR/long-horizon claims; low impact
  on the current 12-month board.
- Probabilistic bands. Bill explicitly parked that for the next phase.
- External ML or type-curve/cohort methods. They may matter later, but adding
  them now would blur the clean Python-vs-LLM comparison.
- Live use of private data or vendor tools. The repo remains public-safe.
