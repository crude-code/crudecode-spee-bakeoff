# v1.3.1 Independent Confirmation Protocol

## Observed result

The pre-registered selection board chose `cohort_conservative` on 300 tune wells. On 150 separately extracted confirmation wells:

- SciPy control SPEE: 0.1771
- cohort_conservative SPEE: 0.1528
- difference: -0.0244
- clustered 95% interval: [-0.0525, +0.0006]
- bootstrap probability of improvement: 0.971

The point estimate is materially favorable, but the upper interval bound is not below zero. The strict gate therefore remains failed.

## Why a second pool is necessary

The first confirmation result has been viewed. It cannot be reused for profile selection, threshold tuning, or repeated testing without overstating confidence. v1.3.1 appends a distinct `confirm2` role after the frozen eval/cohort/dev/confirm prefix.

## Pre-registration

Before extracting or scoring confirm2:

- fixed profile: `cohort_conservative`;
- comparator: exact original SciPy control;
- primary metric: major-phase monthly SPEE;
- bootstrap unit: well;
- minimum relative improvement: 2%;
- catastrophic-rate tolerance: +1 percentage point;
- supported segment tolerance: +0.05 SPEE;
- cumulative-major tolerance: +0.02 SPEE;
- required model failures: zero;
- required 95% interval upper bound: below zero.

The profile is not reselected after confirm2 is observed.

## Decision rule

Promote only if every gate passes on confirm2 alone. If any gate fails, retain `scipy_control`. The pooled confirm+confirm2 result may be reported descriptively but does not control promotion.
