# Deterministic Arps research implementation

This note records what was implemented from the deterministic-Arps research pass and what stayed intentionally out of scope.

## Implemented now

### 1. Reporting-lag diagnostic

The research's highest-value claim is that the newest public/state production months can be incomplete and revised upward later. That can bias the deterministic arm through two channels:

1. the latest months anchor fit level/slope; and
2. the uptime haircut reads depressed reported/capacity ratios as recurring downtime.

Implemented:

```bash
python scripts/diagnose_reporting_lag.py benchmark_data
python scripts/diagnose_reporting_lag.py benchmark_data --json --out results/reporting-lag.json
```

The script outputs reported/fitted-capacity ratios by phase and months-before-cutoff. A broad dip in the last 2-3 months is evidence to test lag-skip variants; it is not proof by itself.

### 2. Lag-skip deterministic variants

Implemented in `src/forecast_benchmark/arps.py`:

```text
arps_lag_skip_2
arps_lag_skip_3
```

Both preserve the real forecast origin at the cutoff, but exclude the freshest 2 or 3 reported months from the fit and uptime measurement.

### 3. b-cap deterministic variant

Implemented:

```text
arps_b_cap_1p0
```

This caps the b grid at 1.0 instead of 1.3. It tests the consistency argument that a trailing current-regime window should not need transient-style `b > 1` tail flatness unless the expanded board proves otherwise.

### 4. Combined candidate

Implemented:

```text
arps_lag_skip_2_b_cap_1p0
```

This combines the two highest-signal a-priori variants. It is not the new official arm. It is a candidate to test on the 70-well and 1000-well board.

## Left unchanged

### Relative-error refit

The research supports the existing relative-error refit for multiplicative production/allocation noise. No change.

### Official board arm

`arps_bounded_b` remains the official deterministic contender until the expanded board selects a replacement by a declared metric.

### Terminal decline

Not implemented in this phase. Terminal decline matters for EUR and long-horizon forecasts, but the current benchmark scores a 12-month holdout. It should be required before any EUR or multi-year result is advertised.

## Run order for Bill's next board

1. Build/refresh the larger snapshot.
2. Run:

```bash
python scripts/diagnose_reporting_lag.py benchmark_data --out results/reporting-lag.md
```

3. Run:

```bash
python scripts/run_python_arm_experiments.py benchmark_data --out results/python-arm.md
```

4. Choose the deterministic arm using a declared metric before fresh LLM runs.
5. Freeze that Python arm.
6. Run the API LLM configs.
7. Score Python vs LLM on the same board.

## Selection discipline

Do not pick the Python variant that merely wins the old 20-pad result. The purpose of the workbench is to prevent accidental holdout tuning. The expanded board should be the selection set, and the final LLM comparison should be run after the deterministic rule is frozen.
