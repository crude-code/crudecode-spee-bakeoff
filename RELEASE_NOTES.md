# SmartCast v1.1.0 — Validation-Hardened SPEE Release

## Production algorithm

The production forecasting behavior remains **SmartCast v1**. No scoring-derived variant was promoted into Strict Auto or Vendor Best.

Rejected or deprioritized ideas are recorded in `docs/SCORING_BACKWARDS_EXPERIMENT.md`:

- global hindcast self-calibration;
- universal recent-month skipping;
- general disruption-gated re-anchoring;
- geometric blending as a standalone improvement;
- altering physical forecasts with an artificial positive floor.

## Scoring correction

The release gate now uses one cumulative holdout log error per well/phase, then computes the cross-sectional median and standard deviation. The prior harness scored month-to-month errors inside each well and was not aligned with the competition objective.

Exact-zero forecasts remain physically valid. Log scoring now requires an explicit `drop`, `epsilon`, or `error` policy and reports every affected observation. Stress and real-holdout validation use metric-only epsilon by default; SmartCast output itself is not changed.

## What the stronger test found

The 30-seed, 180-well paired board did **not** validate the earlier claim that SmartCast beats legacy Arps consistently:

- SmartCast mean score: `0.072198`;
- legacy Arps mean score: `0.071413`;
- mean SmartCast delta: `+0.000785` (`+1.54%`, worse);
- seed wins: `11/30`;
- 95% bootstrap interval for mean delta: `[-0.001062, 0.002515]`.

SmartCast reduced log-error spread but increased under-forecast bias. It produced strong replicated scenario improvements for allocation noise, downtime, and missing months, but those gains did not yield a better pooled score across seeds. The release therefore reports SmartCast as a **candidate with complementary strengths**, not a proven universal winner.

## Validation system

- Paired 30-seed default stress protocol, 180 wells per seed.
- Pooled bias/spread decomposition for every arm.
- Absolute and relative score deltas.
- Paired well-phase absolute-log-error diagnostics.
- Well-phase bootstrap confidence intervals.
- Seed win consistency.
- Scenario means, standard deviations, confidence intervals, and seeds won.
- Clean and worst-scenario regression gates.
- Experiment-only allocation-noise classifier evaluation.
- Leakage-controlled retrospective real-well holdout runner with source attestation.
- Production/experiment import-boundary tests.

## Competition operations retained

- Strict Auto runner with fixed no-touch behavior and complete run evidence.
- Vendor Best runner with ranked QC and fail-closed approved trajectory overrides.
- Broad input aliases, BOM support, duplicate policy, and explicit calendar-gap policy.
- Canonical input validator, submission validator, committee adapter, diagnostics, failure reports, and run logs.
- Uniform 6% effective annual terminal-decline guard across curve families.

See `docs/RELEASE_VALIDATION.md` and `results/` for measured evidence and limitations.
