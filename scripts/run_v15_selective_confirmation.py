#!/usr/bin/env python3
"""One-look confirmation of the frozen ``selective_cohort_v1`` profile.

The role is fixed to ``confirm3`` and the profile is fixed in code. This runner
cannot rank profiles or reuse earlier confirmation pools. The gate is matched
to the competition loss: a directional improvement probability plus explicit
tail, cumulative, segment, failure, parity, and runtime guardrails.
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

PROFILE = "selective_cohort_v1"
ROLE = "confirm3"


def main() -> int:
    here = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="board/private-v15")
    ap.add_argument("--legacy-src", default=os.environ.get("LEGACY_FORECAST_SRC", ""))
    ap.add_argument("--smartcast-src", default=str(here / "src"))
    ap.add_argument("--boot", type=int, default=30000)
    ap.add_argument("--out", default="board/results/v15-selective-confirm3")
    ap.add_argument("--min-relative-improvement", type=float, default=0.01)
    ap.add_argument("--min-probability-better", type=float, default=0.95)
    ap.add_argument("--max-catastrophic-regression", type=float, default=0.005)
    ap.add_argument("--max-p95-abs-regression", type=float, default=0.02)
    ap.add_argument("--max-segment-regression", type=float, default=0.03)
    ap.add_argument("--max-cumulative-regression", type=float, default=0.01)
    ap.add_argument("--max-runtime-ratio", type=float, default=2.5)
    ap.add_argument("--report-only", action="store_true")
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
    targets = sorted(k for k, w in wells.items() if w["role"] == ROLE)
    cohort = sorted(k for k, w in wells.items() if w["role"] == "cohort")
    if len(targets) < 100:
        raise SystemExit(f"board requires at least 100 untouched role={ROLE} wells")
    if not cohort:
        raise SystemExit("board requires a disjoint cohort role")

    smartcast_src = Path(args.smartcast_src).resolve()
    sys.path.insert(0, str(smartcast_src))
    from forecast_benchmark.arps_scipy_v1 import arps_bounded_scipy_v1 as local_scipy  # noqa: E402
    from forecast_benchmark.data import WellSeries  # noqa: E402
    from forecast_benchmark.profiles import get_profile  # noqa: E402
    from forecast_benchmark.smartcast import SmartCastProvider  # noqa: E402
    from forecast_benchmark.split import Split  # noqa: E402

    original = load_module(legacy_src / "forecast_benchmark/arps.py", "original_scipy_v15_confirm3")

    def to_ws(key: str, tr: dict) -> WellSeries:
        return WellSeries(key, list(tr["months"]), tr["oil"].copy(), tr["gas"].copy(), tr["water"].copy())

    pool = [to_ws(k, series[k]["train"]) for k in cohort]
    metadata = {k: {"basin": wells[k]["play"]} for k in wells}
    provider = SmartCastProvider(pool, metadata, get_profile(PROFILE))

    base: dict[str, float] = {}
    chal: dict[str, float] = {}
    base_cum: dict[str, float] = {}
    chal_cum: dict[str, float] = {}
    failures: list[dict[str, str]] = []
    runtime = Counter()
    routing = Counter()
    control_mismatches = 0

    print(f"board_id={manifest.get('board_id')} fixed_profile={PROFILE} {ROLE}={len(targets)} cohort={len(cohort)}")
    for idx, key in enumerate(targets, 1):
        tr, hd = series[key]["train"], series[key]["holdout"]
        horizon = len(hd["months"])
        phase = major_phase(tr)
        actual = np.asarray(hd[phase], dtype=float)
        if int(np.sum(np.isfinite(actual) & (actual > 0))) < 2:
            continue
        q = np.asarray(tr[phase], dtype=float)
        t0 = time.perf_counter()
        try:
            base_fc = np.asarray(original.arps_hyperbolic_bounded_b(q.copy())(horizon), dtype=float)
        except Exception as exc:
            failures.append({"arm": "scipy_control", "well_key": key, "error": f"{type(exc).__name__}: {exc}"})
            base_fc = None
        runtime["scipy_control"] += time.perf_counter() - t0
        try:
            packaged = np.asarray(local_scipy(q.copy())(horizon), dtype=float)
            if base_fc is None or not np.allclose(packaged, base_fc, rtol=1e-10, atol=1e-10, equal_nan=True):
                control_mismatches += 1
        except Exception:
            control_mismatches += 1

        ws = to_ws(key, tr)
        split = Split(key, ws, list(hd["months"]), {p: np.asarray(hd[p], dtype=float) for p in PHASES})
        t0 = time.perf_counter()
        try:
            profile_fc = provider(split, phase)
            if profile_fc is None:
                raise RuntimeError("provider returned None")
            profile_fc = np.asarray(profile_fc, dtype=float)
            pd = next(d for d in provider.diagnostics[(key, horizon)].phases if d.phase == phase)
            routing[pd.cohort_gate_reason or pd.routing_reason] += 1
        except Exception as exc:
            failures.append({"arm": PROFILE, "well_key": key, "error": f"{type(exc).__name__}: {exc}"})
            profile_fc = None
        runtime[PROFILE] += time.perf_counter() - t0

        eps = max(1e-9, 1e-4 * float(np.median(actual[np.isfinite(actual) & (actual > 0)])))
        if base_fc is None:
            base_fc = np.full(horizon, eps)
        if profile_fc is None:
            profile_fc = np.full(horizon, eps)
        b, _ = monthly_le(base_fc, actual)
        c, _ = monthly_le(profile_fc, actual)
        bc = cumulative_le(base_fc, actual)
        cc = cumulative_le(profile_fc, actual)
        if None in (b, c, bc, cc):
            failures.append({"arm": "metric", "well_key": key, "error": "metric failure"})
            continue
        base[key], chal[key] = float(b), float(c)
        base_cum[key], chal_cum[key] = float(bc), float(cc)
        if idx % 50 == 0 or idx == len(targets):
            print(f"  scored {idx}/{len(targets)} {ROLE} wells")

    if control_mismatches:
        raise SystemExit(f"ABORT: packaged control mismatch on {control_mismatches} wells")
    keys = sorted(set(base) & set(chal))
    if len(keys) < 100:
        raise SystemExit(f"only {len(keys)} eligible paired wells")

    base_score = spee([base[k] for k in keys])
    chal_score = spee([chal[k] for k in keys])
    boot = bootstrap_diff(base, chal, keys, args.boot)
    diff = float(chal_score["spee"]) - float(base_score["spee"])
    rel = -diff / max(float(base_score["spee"]), 1e-12)
    base_abs = np.abs([base[k] for k in keys])
    chal_abs = np.abs([chal[k] for k in keys])
    p95_delta = float(np.percentile(chal_abs, 95) - np.percentile(base_abs, 95))
    cat_delta = float(np.mean(chal_abs > CATASTROPHIC) - np.mean(base_abs > CATASTROPHIC))
    cum_delta = float(spee([chal_cum[k] for k in keys])["spee"] - spee([base_cum[k] for k in keys])["spee"])
    runtime_ratio = float(runtime[PROFILE] / max(runtime["scipy_control"], 1e-9))

    segments = {}
    segment_gate = True
    for field in ("play", "bucket"):
        bs = segment_scores(base, keys, wells, field)
        cs = segment_scores(chal, keys, wells, field)
        rows = {}
        for group in bs:
            delta = float(cs[group]["spee"] - bs[group]["spee"])
            n = int(bs[group]["n"])
            rows[group] = {"n": n, "base": bs[group], "profile": cs[group], "delta": delta}
            if n >= 20 and delta > args.max_segment_regression:
                segment_gate = False
        segments[field] = rows

    profile_failures = sum(r["arm"] == PROFILE for r in failures)
    gates = {
        "exact_control_parity": control_mismatches == 0,
        "zero_profile_failures": profile_failures == 0,
        "minimum_relative_improvement": rel >= args.min_relative_improvement,
        "directional_probability": float(boot["p_better"]) >= args.min_probability_better,
        "catastrophic_non_regression": cat_delta <= args.max_catastrophic_regression,
        "p95_abs_error_non_regression": p95_delta <= args.max_p95_abs_regression,
        "cumulative_non_regression": cum_delta <= args.max_cumulative_regression,
        "segment_non_regression": segment_gate,
        "runtime_guardrail": runtime_ratio <= args.max_runtime_ratio,
    }
    promote = all(gates.values())
    recommendation = PROFILE if promote else "scipy_control"
    report = {
        "schema_version": 1,
        "protocol": "one frozen selective-cohort policy, one untouched confirm3 look, one-sided decision with tail guardrails",
        "board_id": manifest.get("board_id"),
        "role": ROLE,
        "profile": PROFILE,
        "n": len(keys),
        "scores": {"scipy_control": base_score, PROFILE: chal_score},
        "comparison": {
            "score_diff": diff,
            "relative_improvement": rel,
            "p95_abs_error_delta": p95_delta,
            "catastrophic_rate_delta": cat_delta,
            "cumulative_score_delta": cum_delta,
            "runtime_ratio": runtime_ratio,
            **boot,
        },
        "gates": gates,
        "promotion_pass": promote,
        "recommended_profile": recommendation,
        "segments": segments,
        "routing": dict(routing),
        "runtime_s": {k: round(float(v), 3) for k, v in runtime.items()},
        "failures": len(failures),
    }
    (out / "v15_selective_confirmation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (out / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    (out / "deployment_profile.json").write_text(json.dumps({
        "board_id": manifest.get("board_id"),
        "promotion_pass": promote,
        "fixed_profile": PROFILE,
        "recommended_profile": recommendation,
        "strict_auto_argument": f"--profile {recommendation}",
    }, indent=2) + "\n")
    print(f"{ROLE} A={base_score['spee']:.4f} profile={chal_score['spee']:.4f} diff={diff:+.4f} P(better)={boot['p_better']:.3f}")
    print(f"Promotion: {'PASS' if promote else 'FAIL'} -> {recommendation}")
    return 0 if (promote or args.report_only) else 2


if __name__ == "__main__":
    raise SystemExit(main())
