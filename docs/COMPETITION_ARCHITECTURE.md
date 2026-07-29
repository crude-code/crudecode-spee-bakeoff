# Competition Architecture

## Objective

Maximize point-in-time forecast accuracy for 1,000+ North American horizontal wells while preserving a defensible empirical rate-time basis and finishing comfortably inside the committee deadline.

## SmartCast decision path

1. **Canonical calendar history**
   - every well is sorted and aligned by calendar month;
   - blanks remain NaN unless an explicit zero policy is chosen;
   - duplicates fail or are additively combined by declared policy.
2. **Independent empirical candidates**
   - modified hyperbolic Arps over several trailing windows;
   - exponential decline;
   - stretched-exponential production decline;
   - recovery candidates that exclude the newest disrupted reports while retaining the real calendar forecast origin.
3. **Rolling-origin selection**
   - candidates are tested against multiple internal pre-cutoff holdouts;
   - no committee answer data or future rows are visible.
4. **Operational-state hypotheses**
   - terminal low reports are not forced into a single interpretation;
   - recovery and current-status paths are evaluated using zero-streak length and prior recovery evidence;
   - a depressed but non-zero cutoff report is conservatively anchored to the bounded-Arps capacity path, while multi-month zero streaks retain explicit shut-in/recovery weighting.
5. **Legacy safety ensemble**
   - bounded Arps remains an independent anchor;
   - SmartCast and the anchor are scored on the exact same rolling origins and weights;
   - the richer model only receives material weight when visible-history hindcasts earn it.
6. **Thin-history shrinkage**
   - normalized basin cohort shape when enough basin wells exist;
   - global cohort fallback;
   - blend weight falls to zero as well history becomes sufficient.
7. **Secondary phases**
   - primary phase is selected on BOE-equivalent cumulative history;
   - GOR/CGR and WOR/WGR trends are partially pooled toward cohort slopes;
   - ratio result is blended with the independent phase forecast to avoid ratio overreaction;
   - cohort and ratio adjustments are suppressed when a cutoff disruption is unresolved.
8. **Long horizon**
   - all candidate families transition to at least 6% effective annual terminal decline;
   - vectorized implementation prevents 360-month forecasts from becoming a runtime bottleneck.

## Strict Auto boundary

The algorithm, configuration, and ingestion policy are frozen before execution. Diagnostics may be written, but no person or LLM changes the forecasts after the autocast starts.

## Vendor Best boundary

Vendor Best begins from the same SmartCast run. Human effort is targeted only through the ranked review queue. Any replacement trajectory requires:

- named approver;
- timestamp;
- well and phase;
- engineering reason;
- complete horizon;
- finite non-negative values.

The run log records the override source and count.

## Known residual risks

- A cutoff disruption cannot always be distinguished from permanent shut-in using production alone.
- Cohort quality depends on useful header segmentation and sufficient peer wells.
- Lease-level allocation artifacts can remain internally consistent but wrong.
- Ratio forecasting improves physical consistency but cannot infer facility constraints or unreported processing changes.
- The final committee schema and metrics are not known until the test-data package arrives.
