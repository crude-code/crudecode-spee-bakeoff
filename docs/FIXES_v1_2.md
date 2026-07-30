> **Superseded by v1.3.** This document records Claude's proposed v1.2 fix.
> The valid anchor and NaN repairs were retained, but additional material defects
> are documented in `V1_3_ACCURACY_HARDENING.md`.

# v1.2 fixes — what changed, what did not, and what is still unproven

Driven by the 299-well real development board, where SmartCast scored **0.1643**
major-phase SPEE against original SciPy Arps at **0.1209** — significantly worse
(clustered CI [+0.0016, +0.0584], P(better) = 2%).

## Root cause, confirmed in code (not inferred)

Four linked defects, each verified by reading the source:

1. `smartcast.py:26` imported `arps_hyperbolic_bounded_b` from `arps.py`. In this
   package that is the **fast linearized** implementation — Arm B — not the SciPy
   champion. So the "legacy safety anchor" was anchored to the weaker model.
   Arm B scored 0.2098 with spread 0.5870 (vs A's 0.3595) and cumulative-major
   spread **0.8496**, driven by a few extreme blow-ups.
2. The anchor's **default branch** set `smart_weight = 0.0` — return the anchor,
   discard every candidate / cohort / ratio layer. Measured on real Delaware
   well-phases: `smart_weight` was exactly 0.00 on **27.8%** of them, mean 0.292.
3. `_legacy_backtest_score` **also** used the linearized Arps, so the branch test
   `legacy_bt < 0.92 * bt` was decided by scoring one model while the branch then
   deferred to a different one.
4. `enforce_terminal_decline` mapped any non-finite value to 0.0, and the
   following `np.minimum.accumulate` propagated that zero forward forever.
   `[1000, 900, NaN, 800, 700, 600]` returned `[1000, 900, 0, 0, 0, 0]`.

Consequence: **the 300-well board largely did not measure SmartCast.** It measured
"linearized Arps, with occasional SmartCast." That explains every pattern in it —
C landing between A and B, C worst on long histories (which miss the disruption
and thin-history branches and fall to the default), and C's only win being the
shortest-history bucket, the one path where SmartCast's own logic gets weight.

## Fixed (unambiguous bugs)

- **`arps_scipy_v1.py`** — the original SciPy bounded-b Arps, ported verbatim
  under an explicit name, `arps_bounded_scipy_v1`. Frozen. It is the control; do
  not optimize it.
- **`anchor_impl` config**, default `'scipy'`. Anchoring to the weaker
  implementation was a defect, so pointing at the champion is a fix. Options:
  `'scipy'` / `'linearized'` (reproduces the regression, for ablation) / `'none'`.
- **`_anchor_impl()` resolver** — single source of truth, so the anchor and the
  anchor's own hindcast score can never disagree about which model they mean.
  They did in v1.1.0.
- **NaN propagation fixed** — interior non-finite values are interpolated, ends
  are edge-extrapolated, and only an all-non-finite array degenerates to zeros.
  Verified: interior NaN, leading NaN, trailing NaN, all-NaN, and inf. Terminal
  decline math unchanged (still exactly 6.000% effective annual; steep curves
  still untouched, max diff 0.00e+00).

## Made ablatable, deliberately NOT changed

`default_smart_weight` stays at **0.0** and `depressed_cutoff_smart_weight` stays
at **0.0**, so behaviour is unchanged unless a run turns them up. These are model
choices, not defects — flipping them silently would be exactly the "complexity as
improvement" move the project's own rules forbid. They have to be earned on the
dev board.

New switches, all defaulting to current behaviour: `use_anchor`, `use_recovery`,
`use_cohort`, `use_ratio_coupling`, `use_terminal_decline`.

## New: `scripts/run_ablations.py`

Runs A, B and C0..C6 on one board, paired, with:
- **actual-driven metric mask** — a zero forecast is penalised (floored at 1e-9),
  never silently dropped as it was when the mask required `fc > 0`;
- **common paired well set** across all arms, so a failure cannot shrink an arm
  into an easier subset;
- **clustered bootstrap by well**, since oil/gas/water in one well share downtime;
- **outlier profile** P90/P95/P99/max |log error| — a binary catastrophic rate
  cannot distinguish a 2.1x miss from a 100x miss, and Arm B's huge spread with a
  low catastrophic rate is precisely that signature;
- `--role dev` default; eval requires `--confirm-locked-eval`.

## Verification status

- Existing suite: **78 passed, 2 skipped** — unchanged from v1.1.0, no regressions.
- All three anchor modes produce distinct forecasts (switches are live).
- NaN fix verified across six edge cases.
- Ablation harness runs end to end on a real 12-well Delaware board.

## What is NOT established

**The 12-well pilot numbers contradict the 299-well board** — A_scipy came last
on the pilot and first on the real board. Twelve wells of one play cannot decide
anything; that pilot exists only to prove the harness executes. An earlier
18-well version of the same pilot reported SmartCast significantly *better* with
a confidence interval, and the 299-well board reversed it. Do not repeat that
mistake with these numbers.

**No accuracy claim is made for any fix here.** Repointing the anchor is
justified because anchoring to a measurably worse implementation is a defect, not
because it has been shown to improve the score. That is what the ablation matrix
is for.

## Order of operations

1. Run `run_ablations.py --role dev` on the 300-well dev board.
2. Keep only layers whose removal makes the score worse.
3. A configuration replaces SciPy Arps as the core only if its clustered CI
   against A lies entirely below zero.
4. Leave the 1,000 locked eval wells alone until the code is frozen.

If nothing beats A on the dev board, the honest answer is that **original SciPy
Arps is the Strict Auto core**, with terminal-decline compliance and submission
tooling layered on. SmartCast earns its way back one layer at a time.
