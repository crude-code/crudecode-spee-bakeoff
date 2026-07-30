#!/usr/bin/env python3
"""Legacy exploratory ablation matrix C0..C6.

SUPERSEDED: use ``scripts/run_v13_selection.py`` for promotion decisions. That
runner uses actual calendar labels, failure penalties, pre-registered profiles,
and a tune/confirmation split. This file remains only for historical diagnosis.


    python scripts/run_ablations.py --board board/private --role dev \
        --legacy-src "C:\\path\\to\\forecast-benchmark-main\\src"

WHY
On the 299-well real development board, SmartCast scored 0.1643 major-phase
SPEE against the original SciPy Arps at 0.1209 — significantly WORSE
(clustered CI [+0.0016, +0.0584], P(better) = 2%).

Root cause, confirmed in code:
  * smartcast.py imported `arps_hyperbolic_bounded_b` from arps.py, which is the
    fast LINEARIZED implementation (Arm B), not the SciPy champion (Arm A);
  * Arm B scored 0.2098 with spread 0.5870 vs A's 0.3595, and cum-major spread
    0.8496 driven by a few extreme blow-ups;
  * the anchor's DEFAULT branch set smart_weight = 0.0, i.e. return the anchor
    and discard every candidate/cohort/ratio layer;
  * the default routing branch could assign 100% weight to that weaker anchor;
  * `_legacy_backtest_score` ALSO used the linearized Arps, so the branch test
    `legacy_bt < 0.92*bt` was decided on the wrong model.

So the 300-well board largely did not measure SmartCast. It measured "linearized
Arps, with occasional SmartCast." This matrix separates the layers so each one
is judged on its own.

THE ARMS
  A     original SciPy bounded-b Arps          — the champion / control
  B     fast linearized Arps                  — kept visible; it is the regression
  C0    candidates only, no anchor, no post   — is the candidate board any good?
  C1    SmartCast + SciPy anchor              — highest-value single test
  C2    SmartCast, anchor OFF entirely        — does the anchor help or poison?
  C3    C1 with cohort OFF                    — isolates cohort shrinkage
  C4    C1 with recovery OFF                  — isolates endpoint/status blending
  C5    C1 with ratio coupling OFF            — isolates phase coupling
  C6    C1 with default_smart_weight = 0.5    — does letting SmartCast actually
                                                run on the default path help?

PROMOTION RULE
A layer is kept only if removing it makes the score WORSE on the dev board, and
the whole configuration only replaces A if its clustered CI against A lies
entirely below zero. Nothing here is promoted on a point estimate.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_module(path: str, name: str, src_root: str):
    if src_root not in sys.path:
        sys.path.insert(0, src_root)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_board(root: Path, role: str):
    wells = {}
    with (root / "wells.csv").open() as f:
        for r in csv.DictReader(f):
            wells[r["well_key"]] = r
    series = defaultdict(lambda: {"train": defaultdict(list), "holdout": defaultdict(list)})
    with (root / "series.csv").open() as f:
        for r in csv.DictReader(f):
            s = series[r["well_key"]][r["split"]]
            for ph in ("oil", "gas", "water"):
                s[ph].append(float(r[ph]))
    target = [k for k, w in wells.items() if w["role"] == role]
    cohort = [k for k, w in wells.items() if w["role"] == "cohort"]
    return wells, series, target, cohort


def spee(vals):
    x = np.array([v for v in vals if v is not None and np.isfinite(v)])
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan"), len(x)
    return ((2/3)*abs(np.median(x)) + (1/3)*np.std(x),
            float(np.median(x)), float(np.std(x)), len(x))


def monthly_avg_le(fc, act):
    fc = np.asarray(fc, float); act = np.asarray(act, float)
    n = min(len(fc), len(act)); fc, act = fc[:n], act[:n]
    v = np.isfinite(act) & (act > 0)          # ACTUAL-driven mask, identical for all arms
    if v.sum() < 2:
        return None
    f = np.where(np.isfinite(fc) & (fc > 0), fc, 1e-9)   # a zero forecast is penalised,
    return float(np.mean(np.log(f[v] / act[v])))          # never silently dropped


def outlier_profile(vals):
    """P90/P95/P99/max |log error| — a binary catastrophic rate cannot tell a
    2.1x miss from a 100x miss, and Arm B's huge spread with a low catastrophic
    rate is exactly that signature."""
    x = np.abs(np.array([v for v in vals if v is not None and np.isfinite(v)]))
    if len(x) < 2:
        return {}
    return {"p90": float(np.percentile(x, 90)), "p95": float(np.percentile(x, 95)),
            "p99": float(np.percentile(x, 99)), "max": float(x.max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="board/private")
    ap.add_argument("--role", default="dev", choices=["dev", "eval"])
    ap.add_argument("--confirm-locked-eval", action="store_true")
    ap.add_argument("--legacy-src", required=True,
                    help="src/ of the historical benchmark repo (SciPy Arps)")
    ap.add_argument("--smartcast-src", default="./src")
    ap.add_argument("--boot", type=int, default=20000)
    ap.add_argument("--out", default="board/results/ablations")
    args = ap.parse_args()

    if args.role == "eval" and not args.confirm_locked_eval:
        sys.exit("refusing to touch the locked eval set without --confirm-locked-eval")

    root = Path(args.board)
    wells, series, target, cohort = read_board(root, args.role)
    print(f"role={args.role}: {len(target)} wells | cohort={len(cohort)} wells")

    armA = load_module(f"{args.legacy_src}/forecast_benchmark/arps.py", "armA_arps", args.legacy_src)
    sys.path.insert(0, args.smartcast_src)
    import forecast_benchmark.arps as armB                  # noqa: E402
    import forecast_benchmark.smartcast as SC               # noqa: E402
    from forecast_benchmark.data import WellSeries          # noqa: E402
    from forecast_benchmark.split import Split              # noqa: E402

    C = SC.SmartCastConfig
    configs = {
        "C0_candidates_only": C(anchor_impl="none", use_recovery=False,
                                use_cohort=False, use_ratio_coupling=False),
        "C1_scipy_anchor":    C(anchor_impl="scipy"),
        "C2_no_anchor":       C(anchor_impl="none"),
        "C3_no_cohort":       C(anchor_impl="scipy", use_cohort=False),
        "C4_no_recovery":     C(anchor_impl="scipy", use_recovery=False),
        "C5_no_ratio":        C(anchor_impl="scipy", use_ratio_coupling=False),
        "C6_default_w05":     C(anchor_impl="scipy", default_smart_weight=0.5),
        "C_v11_baseline":     C(anchor_impl="linearized"),   # reproduces the regression
    }

    def month_labels(n, y=2015):
        out, m = [], 1
        for _ in range(n):
            out.append(f"{y}-{m:02d}-01"); m += 1
            if m > 12: m, y = 1, y+1
        return out

    def to_ws(key, tr):
        n = len(tr["oil"])
        return WellSeries(key, month_labels(n), np.array(tr["oil"]),
                          np.array(tr["gas"]), np.array(tr["water"]))

    cohort_pool = [to_ws(k, series[k]["train"]) for k in cohort]
    cohort_meta = {k: {"basin": wells[k]["play"]} for k in wells}

    providers = {}
    for name, cfg in configs.items():
        providers[name] = SC.SmartCastProvider(cohort_pool, cohort_meta, cfg) if cohort_pool \
                          else SC.SmartCastProvider([to_ws(k, series[k]["train"]) for k in target],
                                                    {k: {"basin": wells[k]["play"]} for k in target}, cfg)

    recs = defaultdict(dict)
    for key in target:
        tr, hd = series[key]["train"], series[key]["holdout"]
        H = len(hd["oil"])
        if H < 2:
            continue
        ws = to_ws(key, tr)
        sp = Split(key, ws, ["m"]*H, {p: np.array(hd[p]) for p in ("oil", "gas", "water")})
        major = "oil" if np.nansum(tr["oil"]) >= np.nansum(tr["gas"])/6.0 else "gas"

        recs["A_scipy"][key] = monthly_avg_le(
            armA.arps_hyperbolic_bounded_b(np.array(tr[major]))(H), hd[major])
        recs["B_linearized"][key] = monthly_avg_le(
            armB.arps_hyperbolic_bounded_b(np.array(tr[major]))(H), hd[major])
        for name, prov in providers.items():
            fc = prov(sp, major)
            recs[name][key] = monthly_avg_le(fc, hd[major]) if fc is not None else None

    # common paired set across ALL arms — a failure must never shrink an arm's
    # well set into an easier subset
    common = [k for k in target
              if all(recs[a].get(k) is not None and np.isfinite(recs[a][k]) for a in recs)]
    print(f"common paired wells across all arms: {len(common)} of {len(target)}\n")

    base = "A_scipy"
    rows = []
    for arm in recs:
        s, med, sd, n = spee([recs[arm][k] for k in common])
        rows.append((s, arm, med, sd, n, outlier_profile([recs[arm][k] for k in common])))
    rows.sort()

    print(f"{'arm':<22}{'n':>5}{'median LE':>11}{'stdev':>9}{'SPEE':>9}{'P95|LE|':>9}{'max':>8}")
    print("-" * 73)
    for s, arm, med, sd, n, op in rows:
        print(f"{arm:<22}{n:>5}{med:>+11.4f}{sd:>9.4f}{s:>9.4f}"
              f"{op.get('p95', float('nan')):>9.3f}{op.get('max', float('nan')):>8.2f}")

    def clustered(chal, B):
        rng = np.random.default_rng(0); d = []
        for _ in range(B):
            idx = rng.integers(0, len(common), len(common))
            kb = [common[i] for i in idx]
            sa = spee([recs[base][k] for k in kb])[0]
            sc = spee([recs[chal][k] for k in kb])[0]
            if np.isfinite(sa) and np.isfinite(sc):
                d.append(sc - sa)
        lo, hi = np.percentile(d, [2.5, 97.5])
        return lo, hi, float(np.mean(np.array(d) < 0))

    sA = spee([recs[base][k] for k in common])[0]
    print(f"\nclustered bootstrap by well vs {base} (B={args.boot})")
    report = {"n_common": len(common), "base": base, "arms": {}, "paired": {}}
    for s, arm, med, sd, n, op in rows:
        report["arms"][arm] = {"spee": s, "median_le": med, "stdev_le": sd,
                               "n": n, "outliers": op}
        if arm == base:
            continue
        lo, hi, pb = clustered(arm, args.boot)
        verdict = "BETTER" if hi < 0 else ("WORSE" if lo > 0 else "inconclusive")
        report["paired"][arm] = {"diff": s - sA, "ci95": [float(lo), float(hi)],
                                 "p_better": pb, "verdict": verdict}
        print(f"  {arm:<22}{s-sA:+.4f}  CI [{lo:+.4f},{hi:+.4f}]  P(better)={pb:.3f}  {verdict}")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out)/"ablations.json").write_text(json.dumps(report, indent=2, default=float)+"\n")
    print(f"\nwrote {args.out}/ablations.json")
    print("Aggregate metrics only. Well IDs and per-well forecasts stay private.")


if __name__ == "__main__":
    main()
