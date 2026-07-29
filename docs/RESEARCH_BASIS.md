# Research Basis

The implementation follows the contest evidence rather than adding unrelated machine learning.

## SPEE evidence

The official 2024 roadshow reports or depicts:

- 922 real wells plus 100 synthetic wells;
- 7–48 months of visible history;
- 1,000 wells forecast within 12 hours;
- major weaknesses in short histories, operational upsets, b-factor consistency, and secondary phases;
- systematic under-forecasting of GOR on oil wells and over-forecasting of CGR on gas wells;
- a suggestion that a more Bayesian approach could help secondary products;
- common prior parameters of 60 barrels per month or 30 years and, when used, a 6% minimum decline;
- overall scoring defined as `2/3 * |median log error| + 1/3 * standard deviation of log error`.

Primary official sources:

- https://spee.org/2024-software-symposium/
- https://www.spee.org/pdf/2024-Software-Symposium-Roadshow.pdf

The release validation treats each well/phase holdout as one cumulative-volume log error and then computes the cross-sectional score. This is the competition-aligned assumption for the current harness; the 2026 committee's final error definition and aggregation unit must replace it if the test-data instructions differ.

## Model-family evidence

The candidate board is intentionally empirical and rate-time based:

- Arps hyperbolic/exponential decline remains the industry baseline.
- Modified hyperbolic behavior with a terminal decline avoids unbounded or unrealistically flat long tails.
- Stretched-exponential decline provides a bounded unconventional-well alternative with materially different curvature.
- Model selection on pre-cutoff hindcasts avoids selecting a family solely because it fits the visible history best.

Representative references:

- Valkó and Lee, stretched-exponential production decline.
- Shabib-Asl et al., quantitative DCA model selection using information criteria.
- Wahba et al., comparative review of modern unconventional DCA models.
- Tang et al., large-scale shale-oil decline analysis across Bakken, Eagle Ford, and Permian wells.

## Why not deep learning in Strict Auto

The committee explicitly requests empirical rate-time methods such as modified Arps or equivalent. A neural model would add training-data provenance, leakage, reproducibility, and interpretability risk without knowledge of the 2026 test distribution. Cohort shrinkage captures population information while retaining an auditable empirical-curve basis.

## Why ratio pooling is conservative rather than fully Bayesian

A full hierarchical probabilistic implementation would require more validation and calibration than the contest timeline permits. SmartCast implements partial pooling mechanically: local ratio slopes are shrunk toward basin/global medians, bounded, and blended with an independent phase forecast. This targets a documented failure while avoiding an untested probabilistic stack.

## Scoring-derived experiment decisions

The validation-hardened release keeps the score decomposition but rejects global self-calibration, universal recent-month skipping, general disruption-gated re-anchoring, and a physical positive-rate floor. Replication showed that attractive single-seed scenario deltas were unstable. Allocation-noise detection remains an isolated experiment and cannot alter Strict Auto or Vendor Best forecasts.

## Public real-well validation sources

Prefer official regulators that publish monthly well-level production. Colorado ECMC states that oil, gas, and water production has been submitted by well since 1999 and provides downloadable data. Texas production requires special care because important public reporting is lease-level rather than measured individual-well production.

- https://ecmc.colorado.gov/data-maps/downloadable-data-documents
- https://ecmc.state.co.us/cogisdb/Production/ProdSearch
- https://greenport.pa.gov/ReportExtracts/OG/OilGasWellProdReport
- https://www.rrc.state.tx.us/resource-center/research/data-sets-available-for-download/
