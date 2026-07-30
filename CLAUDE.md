# CLAUDE.md

## SPEE bake-off version

## v1.5 frozen selective-cohort candidate

The only confirmation candidate is `selective_cohort_v1`: exact SciPy for all
targets except the oil phase of oil-primary DJ, Eagle Ford, and Delaware wells,
where the unchanged conservative cohort layer is allowed. Missing or ambiguous
play metadata fails safe to SciPy. Do not alter the route after confirm3 starts.
Run `scripts/run_v15_selective_confirmation.py`; it is fixed to that profile and
`role=confirm3`.

This branch is specialized for the forecasting bake-off process. The immediate job is no longer a broad research board; it is: normalize the test package, generate a Strict Auto Forecast, optionally generate a Vendor Best Forecast from precomputed LLM outputs, and keep failures/runtime auditable.

Primary commands:

```bash
python scripts/prepare_spee_input.py --monthly raw/monthly.csv --headers raw/headers.csv --out spee_data
python scripts/validate_spee_input.py spee_data
python scripts/run_spee_strict_auto.py spee_data --horizon 12
python scripts/run_spee_vendor_best.py spee_data --llm-dir benchmark_data/runs/<run_id>/forecasts --horizon 12
```

Do not add probabilistic work to this branch yet. Keep the official Strict Auto model stable unless the expanded board promotes a replacement by a declared metric.

## What this is

A 1v1 benchmark: one deterministic decline-curve model against one LLM
forecaster, scored on held-out real production. Research repo, public on
GitHub. The question is whether an LLM reservoir engineer beats a competent
Arps fit, and where the gain comes from.

Read `README.md` for the result and the board.

## Running things

`pip install -e .` **fails on this machine** — Python 3.13 is
externally-managed (PEP 668). Don't fight it; the package doesn't need to
be installed:

```bash
python3 -m pytest -q                          # works — pyproject sets pythonpath=["src"]
PYTHONPATH=src python3 examples/run_benchmark.py
```

Anything outside pytest needs `PYTHONPATH=src`. numpy 2.4, scipy 1.17,
pytest 9 are already available globally.

`make test` works. `make install` does not.

## The one architectural rule

**Both arms are scored by the same code.** `providers.py` defines a
provider as `(split, phase) -> np.ndarray | None`. The deterministic model
is wrapped by `from_model_fn`, the LLM's offline JSON by `precomputed`, and
from there nothing in the scoring path knows which arm it's looking at —
same calendar cutoff, same holdout window, same metrics, same pad summing.

If you find yourself special-casing an arm downstream of the provider seam,
that's a bug in the design, not a feature to add.

## Invariants that will bite you

- **NaN ≠ 0.** NaN means "not reported"; zero means "reported zero". Never
  fill one with the other. `phase_available()` is the single source of
  truth for whether a phase gets scored at all.
- **Skips are loud.** A well that can't be split is reported in
  `n_wells_skipped` / `skipped_well_ids`, never silently dropped. A pad is
  skipped whole if *any* of its wells can't split, and a pad-phase is
  `forecast_missing` if *any* well lacks a forecast — a sum built from a
  subset would read as a forecast miss when it's a bookkeeping hole.
- **The cutoff is a calendar date, not a month index.** Wells on a pad
  start on different dates; an index cutoff blinds each to a different
  period and makes the pad sum meaningless.
- **A short forecast is a broken submission.** `precomputed` raises on a
  length mismatch rather than padding or truncating.
- **Metrics return `None`, not a fake number**, when undefined.

## One deterministic arm, deliberately

There is exactly one: `arps_hyperbolic_bounded_b`. Improving the
deterministic side means **changing that model**, not adding a sibling next
to it. An earlier version carried four baselines plus several research
variants, and with enough arms on the board one of them beats the LLM on
some phase by chance.

`_flat_tail` in `arps.py` is a degenerate fallback for unfittable history,
not an arm.

## The data boundary

`benchmark_data/` (real wells, real API numbers and operators) and `.env`
(database URL) are gitignored and must never be committed. This is a public
repo.

`tests/test_public_repo_hygiene.py` enforces it against **tracked** files
only — the snapshot is supposed to contain blocklisted terms, and scanning
it would fail the build for data that was never at risk. Don't re-widen it
to walk the filesystem.

Note that `git add -f` overrides `.gitignore`. Check `git status` before
committing.

## Adding an LLM run

One config file = one run = one directory under `benchmark_data/runs/`.
`scripts/run_llm.py configs/<name>.json` generates prompts from the
config's skill file, makes one blind Messages API call per pad (no tools,
no system prompt), validates, and writes `config.json` / `prompts/` /
`raw/` / `forecasts/` / `manifest.json` into the run directory. Score with
`run_benchmark.py --llm-dir runs/<run_id>/forecasts`. Needs
`ANTHROPIC_API_KEY` (env or `.env`) and `pip install anthropic`.

Rules the runner enforces, don't work around them: a run directory is
never overwritten (new `run_id` instead), `--resume` refuses a changed
config, refusals/over-length responses fail the pad loudly and are never
retried on a fallback model, and there is no temperature knob — variance
across repeat runs is the measurement, not a nuisance.

The current board's `llm` arm predates this runner (headless `claude -p`,
which added Claude Code's system prompt on top of the skill). Its
artifacts live in the old flat `llm_raw/` / `llm_forecasts/` / `prompts/`
dirs; API runs are a fresh baseline, not comparable to it.

The blinded-database variant (a `SQL_EVIDENCE` prompt block behind an
`AGENT_TOOLS` env var) was removed in b3ebcb1; its runs are archived under
`benchmark_data/archive/`. Recover it from git history if the young-well
retest below ever happens.

## Known gaps

Ordered by how much each would change the conclusion.

1. **One run per arm, no variance estimate.** The LLM arm is
   non-deterministic and has never been repeat-sampled, so no confidence
   interval exists on any reported gap. This became the top item once the
   uptime-fairness question (below) was answered: the remaining margin is
   smaller on some phases, so error bars now decide whether it's real.
2. **Young wells are untested.** Every pad carries 3+ years of history —
   the regime where a well's own curve is most informative and population
   data least. The SQL-access null result shouldn't be believed generally
   until tested on 3–12-month wells.
3. **~18% pad MAPE floor is undiagnosed.** Held across three LLM arms while
   bias and SPEE improved. If it's downtime and allocation noise it's a
   real ceiling; nothing currently distinguishes that from forecastable
   error.
4. **No probabilistic scoring.** Both arms commit to a single P50 path. A
   P10/P50/P90 scorer existed in an earlier version and was removed; it
   would need rebuilding against the provider seam.
5. **A clone can't reproduce anything** — see the data boundary above.

**Answered (2026-07-27): the uptime-fairness question.** The former top
gap — "the LLM's margin came from a mechanical uptime haircut the Arps
side could also apply" — was tested by giving `arps_hyperbolic_bounded_b`
the skill's exact procedure (strike downtime months, refit capacity,
commit capacity × trailing 24-month uptime; parameters fixed a priori,
never tuned on the board). It improved the Arps arm broadly, and on that
board (Arps v2) the LLM still led every phase at both levels, so the
residual margin is not attributable to the haircut alone. The later
trailing-window fit (Arps v3, 2026-07-27) narrowed things further and
flipped pad gas/water MAPE to the Arps side — see README. See "How the deterministic arm got
here" in `README.md`, including the two documented failure modes
(shut-in-at-cutoff, transient dips priced as persistent) that judgment
handles and the mechanical rule deliberately does not.

## Style

Comments explain *why*, not *what*, and the codebase leans on them to
record decisions that would otherwise get relitigated ("this raises instead
of truncating because…"). Match that when adding code. Docstrings carry the
reasoning; inline comments are sparse.
