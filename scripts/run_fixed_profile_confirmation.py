#!/usr/bin/env python3
"""Confirm one frozen SmartCast profile on an untouched ``role=confirm2`` pool.

This runner intentionally cannot rank or select profiles.  The profile must be
chosen before the confirm2 answers are inspected.  It compares that fixed
profile with the exact historical SciPy control and applies the same paired,
well-clustered promotion gates used by the v1.3 selection board.

The first confirmation pool is already spent once its aggregate result is
viewed.  Do not combine repeated looks until a threshold passes.  Use confirm2
alone for the promotion decision; any pooled confirm+confirm2 summary is only
descriptive.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

# Running a file from scripts/ places this directory on sys.path.
from run_v13_selection import (  # type: ignore
    CATASTROPHIC,
    PHASES,
    bootstrap_diff,
    cumulative_le,
    load_module,
    major_phase,
    monthly_le,
    read_board,
    segment_scores,
    spee,
)


def main() -> int:
    here = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="board/private-v131")
    ap.add_argument("--legacy-src", default=os.environ.get("LEGACY_FORECAST_SRC", ""))
    ap.add_argument("--smartcast-src", default=str(here / "src"))
    ap.add_argument("--profile", default="cohort_conservative")
    ap.add_argument("--boot", type=int, default=30000)
    ap.add_argument("--out", default="board/results/v131-confirm2")
    ap.add_argument("--min-relative-improvement", type=float, default=0.02)
    ap.add_argument("--max-catastrophic-regression", type=float, default=0.01)
    ap.add_argument("--max-segment-regression", type=float, default=0.05)
    ap.add_argument("--max-cumulative-regression", type=float, default=0.02)
    ap.add_argument(
        "--report-only",
        action="store_true",
        help="write the honest result but exit 0 when promotion fails",
    )
    args = ap.parse_args()

    if args.boot < 1000:
        ap.error("--boot must be at least 1000")
    legacy_src = Path(args.legacy_src).resolve() if args.legacy_src else None
    if legacy_src is None or not (legacy_src / "forecast_benchmark/arps.py").exists():
        ap.error("--legacy-src must point to the ORIGINAL forecast-benchmark src directory")

    root = Path(args.board)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wells, series, manifest = read_board(root)
    board_id = str(manifest.get("board_id", "unknown"))
    confirm2 = sorted(k for k, w in wells.items() if w["role"] == "confirm2")
    cohort = sorted(k for k, w in wells.items() if w["role"] == "cohort")
    if not confirm2:
        raise SystemExit("board requires a nonempty untouched role=confirm2 pool")
    if not cohort:
        raise SystemExit("board requires a nonempty disjoint role=cohort pool")

    smartcast_src = Path(args.smartcast_src).resolve()
    sys.path.insert(0, str(smartcast_src))
    from forecast_benchmark.arps_scipy_v1 import arps_bounded_scipy_v1 as local_scipy  # noqa: E402
    from forecast_benchmark.data import WellSeries  # noqa: E402
    from forecast_benchmark.profiles import get_profile, profile_names  # noqa: E402
    from forecast_benchmark.smartcast import SmartCastProvider  # noqa: E402
    from forecast_benchmark.split import Split  # noqa: E402

    production_profiles = set(profile_names(production_only=True))
    if args.profile not in production_profiles or args.profile == "scipy_control":
        ap.error(
            "--profile must be one frozen experimental production profile; "
            f"choose from {', '.join(sorted(production_profiles - {'scipy_control'}))}"
        )

    original = load_module(legacy_src / "forecast_benchmark/arps.py", "original_scipy_confirm2")

    def to_ws(key: str, tr: dict):
        return WellSeries(
            key,
            list(tr["months"]),
            tr["oil"].copy(),
            tr["gas"].copy(),
            tr["water"].copy(),
        )

    pool = [to_ws(k, series[k]["train"]) for k in cohort]
    metadata = {k: {"basin": wells[k]["play"]} for k in wells}
    provider = SmartCastProvider(pool, metadata, get_profile(args.profile))

    base: dict[str, float] = {}
    chal: dict[str, float] = {}
    base_cum: dict[str, float] = {}
    chal_cum: dict[str, float] = {}
    failures: list[dict[str, str]] = []
    actual_ineligible: list[str] = []
    control_mismatches = 0
    routing = Counter()
    smart_weight_sum = 0.0
    cohort_weight_sum = 0.0
    route_n = 0
    runtime = Counter()

    print(
        f"board_id={board_id} fixed_profile={args.profile} "
        f"confirm2={len(confirm2)} cohort={len(cohort)}"
    )

    for idx, key in enumerate(confirm2, 1):
        tr, hd = series[key]["train"], series[key]["holdout"]
        horizon = len(hd["months"])
        phase = major_phase(tr)
        actual = np.asarray(hd[phase], dtype=float)
        if int(np.sum(np.isfinite(actual) & (actual > 0))) < 2:
            actual_ineligible.append(key)
            continue
        q = np.asarray(tr[phase], dtype=float)

        t0 = time.perf_counter()
        try:
            base_fc = np.asarray(original.arps_hyperbolic_bounded_b(q.copy())(horizon), dtype=float)
        except Exception as exc:
            failures.append({"arm": "A_scipy", "well_key": key, "error": f"{type(exc).__name__}: {exc}"})
            base_fc = None
        runtime["A_scipy"] += time.perf_counter() - t0

        # Exact parity is a release invariant, not an accuracy option.
        try:
            local_fc = np.asarray(local_scipy(q.copy())(horizon), dtype=float)
            if base_fc is None or not np.allclose(
                local_fc, base_fc, rtol=1e-10, atol=1e-10, equal_nan=True
            ):
                control_mismatches += 1
        except Exception:
            control_mismatches += 1

        ws = to_ws(key, tr)
        split = Split(
            key,
            ws,
            list(hd["months"]),
            {p: np.asarray(hd[p], dtype=float) for p in PHASES},
        )
        t0 = time.perf_counter()
        try:
            selected_fc = provider(split, phase)
            if selected_fc is None:
                raise RuntimeError("provider returned None")
            selected_fc = np.asarray(selected_fc, dtype=float)
            diag = provider.diagnostics[(key, horizon)]
            pd = next(p for p in diag.phases if p.phase == phase)
            routing[pd.routing_reason] += 1
            smart_weight_sum += float(pd.smart_weight)
            cohort_weight_sum += float(pd.cohort_weight)
            route_n += 1
        except Exception as exc:
            failures.append({"arm": args.profile, "well_key": key, "error": f"{type(exc).__name__}: {exc}"})
            selected_fc = None
        runtime[args.profile] += time.perf_counter() - t0

        # Fail closed on the same paired well. A missing model result receives
        # the metric floor and cannot disappear into an easier subset.
        eps = max(1e-9, 1e-4 * float(np.median(actual[np.isfinite(actual) & (actual > 0)])))
        if base_fc is None:
            base_fc = np.full(horizon, eps, dtype=float)
        if selected_fc is None:
            selected_fc = np.full(horizon, eps, dtype=float)

        b, bs = monthly_le(base_fc, actual)
        c, cs = monthly_le(selected_fc, actual)
        bc = cumulative_le(base_fc, actual)
        cc = cumulative_le(selected_fc, actual)
        if b is None or c is None or bc is None or cc is None:
            failures.append({
                "arm": "metric",
                "well_key": key,
                "error": f"base={bs.get('status')} selected={cs.get('status')}",
            })
            # Preserve pairing with a deterministic severe penalty.
            b, _ = monthly_le(np.full(horizon, eps), actual)
            c, _ = monthly_le(np.full(horizon, eps), actual)
            bc = cumulative_le(np.full(horizon, eps), actual)
            cc = cumulative_le(np.full(horizon, eps), actual)
        base[key] = float(b)
        chal[key] = float(c)
        base_cum[key] = float(bc)
        chal_cum[key] = float(cc)

        if idx % 25 == 0 or idx == len(confirm2):
            print(f"  scored {idx}/{len(confirm2)} confirm2 wells")

    if control_mismatches:
        raise SystemExit(
            f"ABORT: packaged SciPy control disagreed with original on "
            f"{control_mismatches} well(s)"
        )

    keys = sorted(base)
    if len(keys) < 30:
        raise SystemExit(f"only {len(keys)} actual-eligible confirm2 wells; require at least 30")

    base_score = spee([base[k] for k in keys])
    chal_score = spee([chal[k] for k in keys])
    base_cum_score = spee([base_cum[k] for k in keys])
    chal_cum_score = spee([chal_cum[k] for k in keys])
    boot = bootstrap_diff(base, chal, keys, args.boot)
    diff = float(chal_score["spee"]) - float(base_score["spee"])
    rel = -diff / max(float(base_score["spee"]), 1e-12)
    cat_base = float(np.mean([abs(base[k]) > CATASTROPHIC for k in keys]))
    cat_chal = float(np.mean([abs(chal[k]) > CATASTROPHIC for k in keys]))
    cat_delta = cat_chal - cat_base
    cum_delta = float(chal_cum_score["spee"]) - float(base_cum_score["spee"])

    segment_gate = True
    segments: dict[str, dict] = {}
    for field in ("play", "bucket"):
        bseg = segment_scores(base, keys, wells, field)
        cseg = segment_scores(chal, keys, wells, field)
        rows = {}
        for group in bseg:
            n = int(bseg[group]["n"])
            delta = float(cseg[group]["spee"]) - float(bseg[group]["spee"])
            rows[group] = {"n": n, "base": bseg[group], "profile": cseg[group], "delta": delta}
            if n >= 10 and delta > args.max_segment_regression:
                segment_gate = False
        segments[field] = rows

    selected_failure_count = sum(1 for r in failures if r["arm"] == args.profile)
    gates = {
        "no_control_mismatch": control_mismatches == 0,
        "no_profile_failures": selected_failure_count == 0,
        "point_estimate_better": diff < 0,
        "ci_entirely_below_zero": float(boot["ci95"][1]) < 0,
        "minimum_relative_improvement": rel >= args.min_relative_improvement,
        "catastrophic_non_regression": cat_delta <= args.max_catastrophic_regression,
        "cumulative_major_non_regression": cum_delta <= args.max_cumulative_regression,
        "segment_non_regression": segment_gate,
    }
    promote = all(gates.values())
    recommendation = args.profile if promote else "scipy_control"

    report = {
        "schema_version": 1,
        "protocol": (
            "single pre-registered fixed profile on untouched confirm2; "
            "confirm2 alone controls promotion; prior confirmation is not pooled for the gate"
        ),
        "board_id": board_id,
        "profile": args.profile,
        "confirm2_total": len(confirm2),
        "confirm2_eligible": len(keys),
        "actual_ineligible": len(actual_ineligible),
        "scores": {
            "A_scipy": {**base_score, "catastrophic_rate": cat_base, "cum_major": base_cum_score},
            args.profile: {**chal_score, "catastrophic_rate": cat_chal, "cum_major": chal_cum_score},
        },
        "comparison": {
            "score_diff_profile_minus_A": diff,
            "relative_improvement": rel,
            "catastrophic_rate_delta": cat_delta,
            "cumulative_score_delta": cum_delta,
            **boot,
        },
        "promotion_gates": gates,
        "promotion_pass": promote,
        "recommended_profile": recommendation,
        "segments": segments,
        "routing": {
            "routing_reason_counts": dict(routing),
            "mean_smart_weight": smart_weight_sum / max(route_n, 1),
            "mean_cohort_weight": cohort_weight_sum / max(route_n, 1),
        },
        "runtime_s": {k: round(float(v), 3) for k, v in runtime.items()},
        "failure_count": len(failures),
        "profile_failure_count": selected_failure_count,
    }
    (out / "fixed_confirmation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / "failures.json").write_text(
        json.dumps(failures, indent=2) + "\n", encoding="utf-8"
    )
    (out / "deployment_profile.json").write_text(
        json.dumps({
            "board_id": board_id,
            "promotion_pass": promote,
            "fixed_profile": args.profile,
            "recommended_profile": recommendation,
            "strict_auto_argument": f"--profile {recommendation}",
            "reason": (
                "fixed profile passed independent confirm2 gates"
                if promote else
                "fixed profile did not pass every independent confirm2 gate; deploy exact SciPy control"
            ),
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"Confirm2 A={float(base_score['spee']):.4f} "
        f"profile={float(chal_score['spee']):.4f} diff={diff:+.4f}"
    )
    print(
        f"Confirm2 CI [{boot['ci95'][0]:+.4f}, {boot['ci95'][1]:+.4f}] "
        f"P(better)={boot['p_better']:.3f}"
    )
    print(f"Promotion: {'PASS -> ' + args.profile if promote else 'FAIL -> scipy_control'}")
    print(f"Wrote {out / 'fixed_confirmation.json'}")
    return 0 if (promote or args.report_only) else 2


if __name__ == "__main__":
    raise SystemExit(main())
