# CrudeCode SPEE Bake-Off SmartCast v1.1.0

Competition-focused empirical rate-time forecasting for the **2026 SPEE Software Symposium bake-off**.

The repository produces both allowed entries:

| Submission | Workflow |
|---|---|
| **Strict Auto** | Fixed SmartCast configuration; no review, adjustment, or override after the run starts |
| **Vendor Best** | Same engine plus a ranked QC queue and optional, explicitly approved full-trajectory overrides |

The engine is deterministic. It does not need an LLM, internet access, a database, or CrudeCode private services on bake-off day.

## What changed from the earlier baseline

`smartcast_v1` is not one more Arps knob. It is a controlled empirical ensemble built around the four failure modes SPEE reported after the 2024 bake-off:

1. short production histories;
2. operational upsets near the forecast cutoff;
3. unstable b-factor selection;
4. inconsistent secondary-phase forecasts.

It includes:

- modified Arps, exponential, and stretched-exponential candidate families;
- rolling-origin model selection using visible history only;
- explicit temporary-recovery versus current-status hypotheses;
- basin/global cohort-shape shrinkage for thin histories;
- pooled GOR/CGR/WOR/WGR forecasting for secondary phases;
- a bounded-Arps safety ensemble selected by internal hindcasts;
- a 6% effective annual terminal decline applied to every curve family;
- auditable diagnostics, review priorities, failures, manifests, and run logs;
- explicit missing-month and duplicate-row policies;
- a paired multi-seed stress board, allocation-noise classifier evaluation, real-well holdout runner, and 1,000-well throughput harness.

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
```

Direct script execution also works from a fresh checkout without an editable install.

## Bake-off workflow

### 1. Normalize the committee package

```bash
python scripts/prepare_spee_input.py \
  --monthly raw/monthly_production.csv \
  --headers raw/well_headers.csv \
  --out spee_data \
  --gap-policy nan \
  --duplicate-policy error

python scripts/validate_spee_input.py spee_data
```

`nan` is the conservative default for omitted calendar rows: missing report is not silently treated as zero production.

### 2. Strict Auto

The prior bake-off used up to 30 years, so the default horizon is 360 months. Override it when the committee gives the 2026 specification.

```bash
python scripts/run_spee_strict_auto.py spee_data \
  --horizon 360 \
  --forecast-start YYYY-MM-01

python scripts/validate_spee_input.py spee_data \
  --submission submissions/strict_auto_forecast.csv \
  --horizon 360
```

Outputs include:

- `submissions/strict_auto_forecast.csv`
- `results/strict_auto_run.json`
- `results/strict_auto_diagnostics.json`
- `results/strict_auto_review_queue.csv`
- failure CSV and Markdown report

The review queue is generated for audit and comparison. It is **not** used to change Strict Auto.

### 3. Vendor Best

First run without overrides and inspect the top-ranked wells/phases:

```bash
python scripts/run_spee_vendor_best.py spee_data --horizon 360
```

Approved targeted overrides use `configs/vendor-best-overrides.example.json`:

```bash
python scripts/run_spee_vendor_best.py spee_data \
  --horizon 360 \
  --overrides approved_overrides.json
```

An override package must contain a named approver, ISO timestamp, reason, valid phase, and a complete finite non-negative trajectory for the requested horizon. Duplicate or partial overrides fail closed.

### 4. Adapt to the committee output header

When the test package supplies the exact sample output schema:

```bash
python scripts/render_committee_submission.py \
  submissions/strict_auto_forecast.csv \
  --mapping configs/committee-output-mapping.example.json \
  --out submissions/strict_auto_committee.csv
```

Update only the mapping file unless the committee requires a structurally different wide format.

## Release checks

```bash
make check
make stress
make allocation-noise
make throughput
python scripts/manage_release_manifest.py --verify
```

The synthetic board is a regression gate, not a claim about the hidden committee data. It covers 7–48 month histories, multiple empirical families, downtime, reactivation, terminal dips, shut-ins, missing months, allocation noise, and coupled secondary phases. The release gate uses one cumulative holdout log error per well/phase, then computes the cross-sectional median and standard deviation required by the SPEE-style score.

Candidate promotion requires paired multi-seed evidence, clean and worst-scenario regression checks, confidence intervals, and real-well validation. The completed 30-seed board did **not** establish SmartCast as statistically superior to legacy Arps; SmartCast remains the frozen production incumbent because no tested alternative earned promotion. See `docs/EXPERIMENT_PROTOCOL.md` and `docs/RELEASE_VALIDATION.md`.

The scoring-derived variants are documented as rejected experiments in `docs/SCORING_BACKWARDS_EXPERIMENT.md`. They are not imported by Strict Auto or Vendor Best.

Real-well evaluation is available through:

```bash
python scripts/run_real_holdout.py real_data/colorado \
  --source-name "Colorado ECMC monthly well production" \
  --snapshot-date YYYY-MM-DD
```

See `docs/REAL_DATA_VALIDATION.md` before interpreting the result. The archive includes a runner smoke test, not a decision-grade external-data benchmark; bring an official regulator extract into the canonical format before making real-world performance claims.

## Repository map

```text
src/forecast_benchmark/
  smartcast.py       competition engine, cohort priors, ratios, diagnostics
  overrides.py       fail-closed Vendor Best override audit
  stressboard.py     paired multi-seed synthetic validation and score decomposition
  arps.py            legacy bounded-Arps baseline and ablations
  data.py            canonical loader and explicit gap policies
  spee.py            submission generation and run evidence
scripts/
  prepare_spee_input.py
  run_spee_strict_auto.py
  run_spee_vendor_best.py
  run_stress_board.py
  run_allocation_noise_evaluation.py
  run_real_holdout.py
  run_throughput_test.py
  render_committee_submission.py
docs/
  COMPETITION_ARCHITECTURE.md
  BAKEOFF_DAY_RUNBOOK.md
  RESEARCH_BASIS.md
  RELEASE_VALIDATION.md
  EXPERIMENT_PROTOCOL.md
  SCORING_BACKWARDS_EXPERIMENT.md
  REAL_DATA_VALIDATION.md
```

## Boundaries

- No forecast is represented as reserves certification, title verification, or investment advice.
- No hidden holdout is used for model selection.
- No missing phase is fabricated.
- No input gap, duplicate, forecast failure, or manual override is silent.
- The exact 2026 committee output schema, horizon, cutoff rules, and deadlines must be frozen after the test-data process.
