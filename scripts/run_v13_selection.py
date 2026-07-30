#!/usr/bin/env python3
"""Select a SmartCast v1.3 profile without touching the locked eval set.

Workflow
--------
1. Read the frozen real board and use only role=dev plus disjoint role=cohort.
2. Prefer a separately extracted role=confirm pool. If it is absent,
   deterministically split dev wells within play x history bucket into tune and
   confirmation subsets.
3. Penalize provider failures on the same actual-eligible wells; never shrink an
   arm into an easier subset.
4. Require a clustered-bootstrap win, a minimum effect, and no material segment
   regression before recommending replacement of the SciPy control.

This is deliberately a small profile board, not a hyperparameter optimizer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PHASES = ("oil", "gas", "water")
CATASTROPHIC = float(np.log(2.0))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_board(root: Path):
    wells: dict[str, dict[str, str]] = {}
    with (root / "wells.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            wells[row["well_key"]] = row

    temp: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    with (root / "series.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            temp[row["well_key"]][row["split"]].append(row)

    series: dict[str, dict[str, dict[str, Any]]] = {}
    for key, splits in temp.items():
        series[key] = {}
        for split, rows in splits.items():
            rows.sort(key=lambda r: r["month"])
            months = [r["month"] for r in rows]
            if len(months) != len(set(months)):
                raise ValueError(f"duplicate calendar month for {key}/{split}")
            series[key][split] = {
                "months": months,
                **{p: np.asarray([float(r[p]) for r in rows], dtype=float) for p in PHASES},
            }
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    return wells, series, manifest


def major_phase(train: dict[str, Any]) -> str:
    oil = float(np.nansum(np.maximum(train["oil"], 0.0)))
    gas_boe = float(np.nansum(np.maximum(train["gas"], 0.0))) / 6.0
    return "oil" if oil >= gas_boe else "gas"


def monthly_le(fc: np.ndarray, act: np.ndarray) -> tuple[float | None, dict[str, Any]]:
    fc = np.asarray(fc, dtype=float)
    act = np.asarray(act, dtype=float)
    if fc.size != act.size:
        return None, {"status": "length_mismatch"}
    valid = np.isfinite(act) & (act > 0)
    if int(valid.sum()) < 2:
        return None, {"status": "actual_ineligible"}
    f = fc[valid]
    a = act[valid]
    if np.any(~np.isfinite(f)):
        return None, {"status": "nonfinite_forecast"}
    eps = max(1e-9, 1e-4 * float(np.median(a)))
    floored = int(np.sum(f <= 0))
    f = np.where(f > 0, f, eps)
    return float(np.mean(np.log(f / a))), {"status": "ok", "floored": floored}


def cumulative_le(fc: np.ndarray, act: np.ndarray) -> float | None:
    fc = np.asarray(fc, dtype=float)
    act = np.asarray(act, dtype=float)
    if fc.size != act.size or np.any(~np.isfinite(fc)):
        return None
    valid = np.isfinite(act)
    if not np.any(valid):
        return None
    sa = float(np.sum(np.maximum(act[valid], 0.0)))
    if sa <= 0:
        return None
    sf = max(float(np.sum(np.maximum(fc[valid], 0.0))), 1e-6 * sa)
    return float(np.log(sf / sa))


def spee(values: list[float]) -> dict[str, float | int]:
    x = np.asarray(values, dtype=float)
    if x.size < 2:
        return {"spee": float("nan"), "median_le": float("nan"), "stdev_le": float("nan"), "n": int(x.size)}
    med = float(np.median(x)); sd = float(np.std(x, ddof=0))
    return {"spee": (2/3)*abs(med) + (1/3)*sd, "median_le": med, "stdev_le": sd, "n": int(x.size)}


def deterministic_partition(keys: list[str], wells: dict[str, dict[str, str]], board_id: str):
    cells: dict[tuple[str, str], list[str]] = defaultdict(list)
    for key in keys:
        cells[(wells[key]["play"], wells[key]["bucket"])].append(key)
    tune, confirm = [], []
    for cell in sorted(cells):
        ordered = sorted(cells[cell], key=lambda k: hashlib.sha256(f"{board_id}|v13-select|{k}".encode()).hexdigest())
        for i, key in enumerate(ordered):
            (confirm if i % 3 == 0 else tune).append(key)
    tune, confirm = sorted(tune), sorted(confirm)
    # Tiny smoke boards can have one well per cell. Preserve a usable split
    # without changing the deterministic real-board partition.
    if not tune or not confirm:
        ordered = sorted(keys, key=lambda k: hashlib.sha256(f"{board_id}|v13-fallback|{k}".encode()).hexdigest())
        cut = max(1, min(len(ordered) - 1, int(round(2 * len(ordered) / 3))))
        tune, confirm = ordered[:cut], ordered[cut:]
    return sorted(tune), sorted(confirm)


def bootstrap_diff(base: dict[str, float], chal: dict[str, float], keys: list[str], reps: int):
    rng = np.random.default_rng(1307)
    diffs = np.empty(reps, dtype=float)
    n = len(keys)
    for b in range(reps):
        sample = [keys[i] for i in rng.integers(0, n, n)]
        sb = float(spee([base[k] for k in sample])["spee"])
        sc = float(spee([chal[k] for k in sample])["spee"])
        diffs[b] = sc - sb
    return {
        "ci95": [float(v) for v in np.percentile(diffs, [2.5, 97.5])],
        "p_better": float(np.mean(diffs < 0)),
    }


def segment_scores(values: dict[str, float], keys: list[str], wells: dict[str, dict[str, str]], field: str):
    groups: dict[str, list[float]] = defaultdict(list)
    for key in keys:
        groups[wells[key][field]].append(values[key])
    return {g: spee(v) for g, v in sorted(groups.items())}


def main() -> int:
    here = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="board/private")
    ap.add_argument("--legacy-src", default=os.environ.get("LEGACY_FORECAST_SRC", ""), required=False)
    ap.add_argument("--smartcast-src", default=str(here / "src"))
    ap.add_argument("--boot", type=int, default=20000)
    ap.add_argument("--out", default="board/results/v13-selection")
    ap.add_argument("--min-relative-improvement", type=float, default=0.02)
    ap.add_argument("--max-catastrophic-regression", type=float, default=0.01)
    ap.add_argument("--max-segment-regression", type=float, default=0.05)
    ap.add_argument(
        "--report-only",
        action="store_true",
        help="write the honest promotion result but return exit code 0 when no profile passes",
    )
    args = ap.parse_args()

    if args.boot < 1000:
        ap.error("--boot must be at least 1000")
    legacy_src = Path(args.legacy_src).resolve() if args.legacy_src else None
    if legacy_src is None or not (legacy_src / "forecast_benchmark/arps.py").exists():
        ap.error("--legacy-src must point to the ORIGINAL forecast-benchmark src directory")

    root = Path(args.board)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    wells, series, manifest = read_board(root)
    board_id = str(manifest.get("board_id", "unknown"))
    dev = sorted(k for k, w in wells.items() if w["role"] == "dev")
    external_confirm = sorted(k for k, w in wells.items() if w["role"] == "confirm")
    cohort = sorted(k for k, w in wells.items() if w["role"] == "cohort")
    if not dev or not cohort:
        raise SystemExit("board requires nonempty dev and disjoint cohort roles")
    if external_confirm:
        tune, confirm = dev, external_confirm
        protocol = "pre-registered profiles; full dev tune plus separately extracted untouched confirmation"
    else:
        tune, confirm = deterministic_partition(dev, wells, board_id)
        protocol = "pre-registered profiles; stratified deterministic 2/3 tune, 1/3 confirmation fallback"
    targets = sorted(set(tune) | set(confirm))
    print(f"board_id={board_id}")
    print(
        f"targets={len(targets)} tune={len(tune)} confirm={len(confirm)} "
        f"external_confirm={bool(external_confirm)} cohort={len(cohort)}"
    )

    original = load_module(legacy_src / "forecast_benchmark/arps.py", "original_scipy_control")
    smartcast_src = Path(args.smartcast_src).resolve()
    sys.path.insert(0, str(smartcast_src))
    from forecast_benchmark.arps import arps_hyperbolic_bounded_b as linearized  # noqa: E402
    from forecast_benchmark.arps_scipy_v1 import arps_bounded_scipy_v1 as local_scipy  # noqa: E402
    from forecast_benchmark.data import WellSeries  # noqa: E402
    from forecast_benchmark.profiles import profile_payload, smartcast_profiles  # noqa: E402
    from forecast_benchmark.smartcast import SmartCastProvider  # noqa: E402
    from forecast_benchmark.split import Split  # noqa: E402

    def to_ws(key: str, tr: dict[str, Any]):
        return WellSeries(key, list(tr["months"]), tr["oil"].copy(), tr["gas"].copy(), tr["water"].copy())

    pool = [to_ws(k, series[k]["train"]) for k in cohort]
    metadata = {k: {"basin": wells[k]["play"]} for k in wells}
    all_profiles = smartcast_profiles()
    selection_names = [
        "candidate_gate", "recovery_conservative", "cohort_conservative",
        "major_conservative", "v12_fixed",
    ]
    profiles = {name: all_profiles[name] for name in selection_names}
    providers = {name: SmartCastProvider(pool, metadata, cfg) for name, cfg in profiles.items()}
    arms = ["A_scipy", "B_linearized", "anchor_only", *selection_names]
    values: dict[str, dict[str, float]] = {a: {} for a in arms}
    cum_values: dict[str, dict[str, float]] = {a: {} for a in arms}
    failures: list[dict[str, str]] = []
    route_counts: dict[str, Counter[str]] = {name: Counter() for name in profiles}
    route_weight_sum = Counter(); route_weight_n = Counter(); cohort_weight_sum = Counter()
    local_control_mismatches = 0
    actual_ineligible: list[str] = []
    runtimes = Counter()

    for idx, key in enumerate(targets, 1):
        tr, hd = series[key]["train"], series[key]["holdout"]
        H = len(hd["months"])
        major = major_phase(tr)
        actual = np.asarray(hd[major], dtype=float)
        if int(np.sum(np.isfinite(actual) & (actual > 0))) < 2:
            actual_ineligible.append(key)
            continue
        ws = to_ws(key, tr)
        split = Split(key, ws, list(hd["months"]), {p: np.asarray(hd[p], dtype=float) for p in PHASES})
        q = np.asarray(tr[major], dtype=float)

        forecasts: dict[str, np.ndarray] = {}
        for name, fn in (
            ("A_scipy", lambda: original.arps_hyperbolic_bounded_b(q.copy())(H)),
            ("B_linearized", lambda: linearized(q.copy())(H)),
        ):
            t0 = time.perf_counter()
            try:
                forecasts[name] = np.asarray(fn(), dtype=float)
            except Exception as exc:
                failures.append({"arm": name, "well_key": key, "error": f"{type(exc).__name__}: {exc}"})
            runtimes[name] += time.perf_counter() - t0

        # The packaged frozen control must reproduce the historical source before
        # it is allowed to serve as SmartCast's anchor.
        try:
            local = np.asarray(local_scipy(q.copy())(H), dtype=float)
            if "A_scipy" not in forecasts or not np.allclose(local, forecasts["A_scipy"], rtol=1e-10, atol=1e-10, equal_nan=True):
                local_control_mismatches += 1
        except Exception:
            local_control_mismatches += 1

        # Fast safety profile: the verified SciPy control plus the common 6%
        # terminal rule. Avoid recomputing the entire candidate board merely to
        # assign it zero weight.
        try:
            from forecast_benchmark.smartcast import enforce_terminal_decline
            forecasts["anchor_only"] = enforce_terminal_decline(local.copy(), 0.06)
        except Exception as exc:
            failures.append({"arm": "anchor_only", "well_key": key, "error": f"{type(exc).__name__}: {exc}"})

        for name, provider in providers.items():
            t0 = time.perf_counter()
            try:
                fc = provider(split, major)
                if fc is None:
                    raise RuntimeError("provider returned None")
                forecasts[name] = np.asarray(fc, dtype=float)
                diag = provider.diagnostics[(key, H)]
                pd = next(p for p in diag.phases if p.phase == major)
                route_counts[name][pd.routing_reason] += 1
                route_weight_sum[name] += pd.smart_weight
                route_weight_n[name] += 1
                cohort_weight_sum[name] += pd.cohort_weight
            except Exception as exc:
                failures.append({"arm": name, "well_key": key, "error": f"{type(exc).__name__}: {exc}"})
            runtimes[name] += time.perf_counter() - t0

        for arm in arms:
            fc = forecasts.get(arm)
            if fc is None:
                # Fail closed on the same well: a missing forecast receives the
                # metric floor rather than disappearing into an easier subset.
                eps = max(1e-9, 1e-4 * float(np.median(actual[np.isfinite(actual) & (actual > 0)])))
                fc = np.full(H, eps, dtype=float)
            mle, status = monthly_le(fc, actual)
            cle = cumulative_le(fc, actual)
            if mle is None or cle is None:
                failures.append({"arm": arm, "well_key": key, "error": f"metric failure: {status}"})
                eps = max(1e-9, 1e-4 * float(np.median(actual[np.isfinite(actual) & (actual > 0)])))
                fc = np.full(H, eps, dtype=float)
                mle, _ = monthly_le(fc, actual); cle = cumulative_le(fc, actual)
            values[arm][key] = float(mle)
            cum_values[arm][key] = float(cle)

        if idx % 25 == 0 or idx == len(targets):
            print(f"  scored {idx}/{len(targets)} tune/confirmation wells")

    if local_control_mismatches:
        raise SystemExit(
            f"ABORT: packaged SciPy anchor disagreed with original control on "
            f"{local_control_mismatches} well(s); do not trust profile results"
        )

    eligible = sorted(set(targets) - set(actual_ineligible))
    tune_e = [k for k in tune if k in values["A_scipy"]]
    confirm_e = [k for k in confirm if k in values["A_scipy"]]

    def summarize(keys: list[str]):
        payload = {}
        for arm in arms:
            vals = [values[arm][k] for k in keys]
            row = spee(vals)
            row["catastrophic_rate"] = float(np.mean(np.abs(vals) > CATASTROPHIC))
            row["cum_major"] = spee([cum_values[arm][k] for k in keys])
            payload[arm] = row
        return payload

    tune_scores = summarize(tune_e)
    # Selection excludes known regression/control arms and secondary-only profile.
    selectable = ["anchor_only", "candidate_gate", "recovery_conservative", "cohort_conservative", "major_conservative", "v12_fixed"]
    complexity_order = {name: i for i, name in enumerate(selectable)}
    selected = min(selectable, key=lambda a: (float(tune_scores[a]["spee"]), complexity_order[a]))

    confirm_scores = summarize(confirm_e)
    full_scores = summarize(eligible)
    boot = bootstrap_diff(values["A_scipy"], values[selected], confirm_e, args.boot)
    diff = float(confirm_scores[selected]["spee"]) - float(confirm_scores["A_scipy"]["spee"])
    rel = -diff / max(float(confirm_scores["A_scipy"]["spee"]), 1e-12)
    cat_delta = float(confirm_scores[selected]["catastrophic_rate"]) - float(confirm_scores["A_scipy"]["catastrophic_rate"])

    segment_gate = True
    segment_report: dict[str, Any] = {}
    for field in ("play", "bucket"):
        base_seg = segment_scores(values["A_scipy"], confirm_e, wells, field)
        chal_seg = segment_scores(values[selected], confirm_e, wells, field)
        rows = {}
        for group in base_seg:
            n = int(base_seg[group]["n"])
            delta = float(chal_seg[group]["spee"]) - float(base_seg[group]["spee"])
            rows[group] = {"n": n, "base": base_seg[group], "selected": chal_seg[group], "delta": delta}
            if n >= 10 and delta > args.max_segment_regression:
                segment_gate = False
        segment_report[field] = rows

    selected_failure_count = sum(1 for r in failures if r["arm"] == selected)
    gate = {
        "no_control_mismatch": local_control_mismatches == 0,
        "no_selected_model_failures": selected_failure_count == 0,
        "point_estimate_better": diff < 0,
        "ci_entirely_below_zero": boot["ci95"][1] < 0,
        "minimum_relative_improvement": rel >= args.min_relative_improvement,
        "catastrophic_non_regression": cat_delta <= args.max_catastrophic_regression,
        "segment_non_regression": segment_gate,
    }
    promote = all(gate.values())
    recommendation = selected if promote else "A_scipy_control"

    routing = {}
    for name in profiles:
        n = max(1, route_weight_n[name])
        routing[name] = {
            "routing_reason_counts": dict(route_counts[name]),
            "mean_smart_weight": float(route_weight_sum[name] / n),
            "mean_cohort_weight": float(cohort_weight_sum[name] / n),
        }

    report = {
        "schema_version": 1,
        "board_id": board_id,
        "dev_counts": {
            "tune_total": len(tune), "confirm_total": len(confirm),
            "total_scored_population": len(targets), "eligible": len(eligible),
            "tune_eligible": len(tune_e), "confirm_eligible": len(confirm_e),
            "actual_ineligible": len(actual_ineligible),
            "external_confirmation_pool": bool(external_confirm),
        },
        "selection_protocol": protocol,
        "selected_on_tune": selected,
        "recommendation": recommendation,
        "promotion_pass": promote,
        "promotion_gates": gate,
        "confirmation_comparison": {
            "score_diff_selected_minus_A": diff,
            "relative_improvement": rel,
            "catastrophic_rate_delta": cat_delta,
            **boot,
        },
        "tune_scores": tune_scores,
        "confirm_scores": confirm_scores,
        "combined_tune_confirmation_descriptive_scores": full_scores,
        "segments_confirm": segment_report,
        "routing_aggregates": routing,
        "runtime_s": {k: round(float(v), 3) for k, v in runtimes.items()},
        "failure_count": len(failures),
        "selected_failure_count": selected_failure_count,
        "profiles": profile_payload(),
        "profiles_scored": ["anchor_only", *selection_names],
    }
    (out / "selection.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "failures.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
    deployment_profile = recommendation if recommendation != "A_scipy_control" else "scipy_control"
    (out / "deployment_profile.json").write_text(
        json.dumps({
            "board_id": board_id,
            "promotion_pass": promote,
            "selected_on_tune": selected,
            "recommended_profile": deployment_profile,
            "strict_auto_argument": f"--profile {deployment_profile}",
            "reason": (
                "confirmation promotion gates passed"
                if promote
                else "no experimental profile passed; deploy exact frozen SciPy control"
            ),
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("\nTune ranking (major-phase SPEE)")
    for arm in sorted(selectable, key=lambda a: float(tune_scores[a]["spee"])):
        print(f"  {arm:<24} {float(tune_scores[arm]['spee']):.4f}")
    print(f"\nSelected on tune: {selected}")
    print(f"Confirm A={float(confirm_scores['A_scipy']['spee']):.4f} selected={float(confirm_scores[selected]['spee']):.4f} diff={diff:+.4f}")
    print(f"Confirm CI [{boot['ci95'][0]:+.4f}, {boot['ci95'][1]:+.4f}] P(better)={boot['p_better']:.3f}")
    print(f"Promotion: {'PASS' if promote else 'FAIL'} -> {recommendation}")
    print(f"Wrote {out/'selection.json'}")
    return 0 if (promote or args.report_only) else 2


if __name__ == "__main__":
    raise SystemExit(main())
