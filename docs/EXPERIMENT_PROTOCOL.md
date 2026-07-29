# Candidate Promotion Protocol

## Purpose

Prevent attractive single-seed or synthetic-only results from entering Strict Auto.

## Required evaluation design

A forecast-changing candidate must be compared against the frozen incumbent on identical generated or real wells.

Minimum synthetic promotion run:

```text
30 paired seeds minimum
180 wells per seed
three phases
identical scenario mix and holdout
well-phase bootstrap unit
metric-only epsilon policy disclosed
```

The preferred final run is 50 seeds when schedule permits.

## Required report

Every candidate report must include:

- pooled median log error and standard deviation;
- bias and spread components and shares;
- pooled absolute and relative score delta;
- paired well-phase mean and median delta;
- paired well-phase win rate;
- 95% paired bootstrap interval;
- seed win rate;
- scenario mean, standard deviation, and seeds won;
- clean-history regression;
- worst-scenario regression;
- runtime, determinism, and failure count;
- explicit zero-score policy and counts.

## Promotion gates

Default candidate gates:

| Gate | Default |
|---|---:|
| Mean pooled score delta | `< 0` |
| Seed win rate | `>= 70%` |
| Clean relative regression | `<= 5%` |
| Worst-scenario relative regression | `<= 25%` |
| Runtime | safely inside contest limit |
| Determinism | identical input/config produces identical output |
| Leakage | no future rows or holdout-derived cohorts |
| Failures | zero silent failures |

Passing synthetic gates is necessary, not sufficient. A forecast-changing candidate also requires a real-well holdout result or committee test-data evidence.

## Commands

```bash
python scripts/run_stress_board.py \
  --seeds 1-30 \
  --wells 180 \
  --workers 4 \
  --strict-nonregression-exit
```

The incumbent comparison may omit `--strict-nonregression-exit`; clean and worst-scenario outcomes are still reported.
