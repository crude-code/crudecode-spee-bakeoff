# Designing Backwards from the Scoring Function

## Decision

**SmartCast v1 remains the frozen production incumbent. None of the scoring-derived variants earned promotion.**

This record preserves both the useful reasoning and the failed experiments so they are not repeated. The initial scenario table came from one seed and was not decision-grade. A later ten-seed replication materially revised several scenario conclusions.

## Score decomposition

For a pooled log-error vector `e = log(forecast / actual)`:

```text
SPEE-style score = 2/3 * |median(e)| + 1/3 * stdev(e)
```

Every candidate report must therefore show:

- median log error;
- log-error standard deviation;
- bias and spread components;
- bias and spread shares;
- absolute score delta;
- relative score delta.

On the original synthetic board snapshot:

| Arm | Median log error | Stdev | Score | Bias share | Spread share |
|---|---:|---:|---:|---:|---:|
| SmartCast v1 | -0.016 | 0.177 | 0.0699 | 15% | 85% |
| Legacy Arps | -0.004 | 0.205 | 0.0707 | 3% | 97% |

This decomposition describes that synthetic board only. It does not establish whether bias or spread will dominate the 2026 committee data.

## Idea 1: self-calibration — rejected

The proposal was to estimate model bias from visible-history hindcasts and divide the final forecast by that estimated bias.

Observed result:

- hindcast median bias: `+0.004`;
- final holdout median bias: `-0.056`;
- per-series correlation: `-0.04`;
- corrected score: approximately 2% worse.

The supported conclusion is narrow: **ordinary interior hindcast bias did not transfer to the final evaluation boundary.** Possible explanations include different endpoint operating states, artificial truncation, regime changes, and—in live public data—reporting lag. The experiment did not isolate the cause.

Do not add global bias self-calibration.

## Idea 2: universal recent-month skipping — rejected

The proposal forced every candidate family to ignore the latest three visible months.

A ten-seed replication, 60 wells per seed, found:

| Scenario | Mean relative delta vs SmartCast | Stdev | Seeds won |
|---|---:|---:|---:|
| Allocation noise | -20.7% | 20.7% | 9/10 |
| Clean | +98.5% | 31.1% | 0/10 |
| Reactivation | +21.8% | 9.9% | 0/10 |
| Downtime | -16.0% | 38.0% | 8/10 |
| Temporary cutoff dip | -8.5% | 36.6% | 8/10 |
| Missing months | +24.3% | 54.5% | 3/10 |
| Thin history | +15.6% | 39.4% | 3/10 |

Overall, skip-three **won on 0 of 10 seeds and lost on all 10**, with a mean score approximately 29.5% worse than SmartCast v1.

The earlier single-seed missing-month and thin-history wins reversed under replication. General or `_terminal_disruption`-gated re-anchoring is therefore not promoted. Allocation noise remains only a narrow hypothesis.

## Idea 3: geometric blending — deprioritized

Geometric aggregation is mathematically aligned with log-error loss, but replacing SmartCast's arithmetic cohort and ratio blends changed the score by only about `+0.14%` on the evaluated board. The blend weights were small and the component forecasts were close, so the Jensen gap was negligible.

Keep the reasoning. Do not add the change.

## Idea 4: exact-zero handling — physical forecast preserved

A numerical scoring problem must not silently change the physical forecast.

The release therefore:

- permits exact zero production forecasts;
- does not add a production floor to SmartCast;
- requires the evaluator to declare one of three log-score policies: `drop`, `epsilon`, or `error`;
- reports every nonpositive forecast row dropped or replaced;
- uses metric-only epsilon in the stress and real-holdout tools by default.

The committee's final zero and economic-cutoff rules take precedence when supplied.

## Final conclusion

The useful output was experimental discipline, not a new model arm:

1. SmartCast v1 remains frozen as the incumbent while real-data evidence is collected.
2. Single-seed scenario deltas are smoke tests only.
3. Candidate promotion requires paired multi-seed evidence and real-well validation.
4. Allocation-noise detection may be studied independently, but no forecast change is promoted from it.
