> **Historical v1.1 validation record.** Current production posture and v1.3
> gates are documented in `V1_3_ACCURACY_HARDENING.md` and the repository README.

# Release Validation — SmartCast v1.1.0

**Validation date:** July 28, 2026  
**Scope:** validation-hardened source archive for the 2026 SPEE Software Symposium test-data process and bake-off workflow.

## Release gates

| Gate | Result |
|---|---:|
| Python compilation | Passed |
| Automated tests | 78 passed, 2 skipped |
| Public-boundary audit from current tree | Passed |
| Correct cumulative well/phase scoring unit | Passed |
| 30-seed paired synthetic board | Completed; candidate-promotion gate did not pass |
| 50-seed allocation-noise classifier | Passed classifier-only gate |
| 1,000-well / 360-month throughput | Passed: 1,080,000 values, 0 failures, 135.741 s |
| Strict Auto end-to-end smoke test | Passed |
| Vendor Best end-to-end smoke test | Passed |
| Strict Auto determinism | Passed; repeated CSV SHA-256 matched |
| Release manifest self-verification | Passed |
| Retrospective holdout runner smoke test | Passed |

## Correct scoring unit

The stress and real-holdout tools compute one cumulative-volume log error per well/phase:

```text
log(sum(forecast holdout volume) / sum(actual holdout volume))
```

The cross-sectional distribution of those errors is then scored as:

```text
2/3 * |median(log error)| + 1/3 * stdev(log error)
```

This is the current competition-aligned assumption. The 2026 committee's final definition overrides it if the test-data instructions specify a different aggregation unit.

## Paired 30-seed board

Configuration:

- 30 independent paired seeds;
- 180 wells per seed;
- oil, gas, and water;
- 12-month hidden holdout;
- 7–48 visible months;
- clean decline, downtime, terminal dips, shut-ins, reactivation, allocation noise, thin histories, and missing months;
- metric-only epsilon policy for positive actual cumulative volume paired with zero forecast cumulative volume;
- 200 paired well-phase bootstrap replicates per seed and 3,000 aggregate seed bootstrap replicates.

### Aggregate result

| Metric | SmartCast v1 | Legacy Arps |
|---|---:|---:|
| Mean score | 0.072198 | 0.071413 |
| Mean median log error | -0.023407 | -0.011712 |
| Mean log-error stdev | 0.169782 | 0.190233 |
| Mean bias component | 0.015604 | 0.008002 |
| Mean spread component | 0.056594 | 0.063411 |

Comparison:

- mean score delta: `+0.000785`;
- mean relative delta: `+1.54%`;
- median score delta: `+0.001309`;
- seeds won: `11/30`;
- 95% bootstrap interval for mean score delta: `[-0.001062, 0.002515]`.

**Interpretation:** SmartCast reduced spread by roughly 10.8%, but its additional under-forecast bias erased the pooled advantage. The evidence does not show a statistically decisive difference, and the configured 70% seed-win promotion gate failed.

### Scenario evidence

| Scenario | Mean relative delta | Seeds won | Decision |
|---|---:|---:|---|
| Allocation noise | -32.97% | 30/30 | strong synthetic SmartCast advantage |
| Downtime | -27.68% | 30/30 | strong synthetic SmartCast advantage |
| Missing months | -23.60% | 28/30 | replicated SmartCast advantage |
| Thin history | -4.81% | 22/30 | positive but interval touches zero |
| Clean | +4.62% | 10/30 | small regression, within 5% gate |
| Reactivation | +1.05% | 10/30 | inconclusive/slight regression |
| Temporary cutoff dip | +0.39% relative | 2/30 | unstable; no promotion claim |

Scenario scores are nonlinear and do not add to the pooled score. Synthetic scenario wins are not evidence about their frequency or weighting in the hidden data.

## Allocation-noise classifier

The experiment-only classifier was evaluated over 50 seeds and 9,000 wells:

- precision: `98.71%`;
- recall: `97.18%`;
- overall false-positive rate: `0.18%`;
- clean-well flag rate: `0%`;
- true positives: `1,069`;
- false positives: `14`.

This passes the classifier-only gate. It does not promote re-anchoring or any forecast change. The classifier remains isolated under `forecast_benchmark.experiments`.

## Throughput result

The full-scale deterministic run produced 1,080,000 forecast values for 1,000 wells, three phases, and 360 months in `135.741` seconds with zero failures (`7.367` wells/second). This is comfortably below the prior 12-hour contest allowance, but the 2026 committee deadline remains controlling.

## External-data status

The release contains a leakage-controlled real-well retrospective runner and source-attestation requirements. A decision-grade external regulator dataset is **not embedded** in the archive. The included `real_holdout_smoke.json` verifies execution only against the bundled synthetic fixture. A current revised regulator download can support stabilized retrospective validation, but it cannot reconstruct point-in-time reporting lag without archived snapshots.

## Important limitations

- All comparative performance above is synthetic.
- The generator uses known empirical curve families and cannot represent every field or reporting process.
- A revised current public download is not a historical reporting snapshot.
- The exact 2026 schema, horizon, economic cutoff, terminal-decline convention, scoring aggregation, and zero treatment must be frozen from committee instructions.
- Production history alone cannot always distinguish downtime, allocation error, delayed reporting, and permanent shut-in.
- SmartCast should not be described as proven superior to legacy Arps until real or committee data supports that claim.
