#!/usr/bin/env python3
"""Score original Arps, fast Arps and SmartCast on a frozen real-well board.

Default role is DEV. Running the locked EVAL role requires
--role eval --confirm-locked-eval.

Reported metric families:
  major      mean monthly ln(F/A), major phase, actual-positive months
  allph      mean of available phase-level monthly log errors per well
  cum_major  ln(sum F / sum A), major phase
  cum_allph  mean of available phase cumulative log errors per well

The exact SPEE per-well aggregation remains unconfirmed, so every result is
labelled by metric family. Bootstrap resampling is clustered by well. Metric
eligibility is determined only from actuals and applied symmetrically to all
arms; provider failures still fail closed.
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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

CATASTROPHIC = float(np.log(2.0))
PHASES = ("oil", "gas", "water")
METRICS = ("major", "allph", "cum_major", "cum_allph")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_board(root: Path) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
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
                raise ValueError(f"Duplicate calendar month in board for {key}/{split}")
            series[key][split] = {
                "months": months,
                **{phase: np.array([float(r[phase]) for r in rows], dtype=float) for phase in PHASES},
            }

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return wells, series, manifest


def spee(vals: list[float | None]) -> tuple[float, float, float, int]:
    x = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    if x.size < 2:
        return float("nan"), float("nan"), float("nan"), int(x.size)
    med = float(np.median(x))
    sd = float(np.std(x, ddof=0))
    return (2.0 / 3.0) * abs(med) + (1.0 / 3.0) * sd, med, sd, int(x.size)


def monthly_avg_le(fc: np.ndarray, act: np.ndarray) -> tuple[float | None, dict[str, int | str]]:
    """Use an actual-only eligibility mask and penalize forecast zeros.

    Eligibility is determined only from the answer series and is therefore the
    same for every arm. A phase with fewer than two positive actual months is
    reported as metric-ineligible rather than as a model failure. Invalid or
    wrong-length forecasts remain hard failures.
    """
    fc = np.asarray(fc, dtype=float)
    act = np.asarray(act, dtype=float)
    if fc.size != act.size:
        return None, {"status": "length_mismatch", "actual_positive": 0,
                      "forecast_floored": 0, "nonfinite_forecast": 0}
    valid_actual = np.isfinite(act) & (act > 0)
    n = int(valid_actual.sum())
    if n < 2:
        return None, {"status": "insufficient_actual_positive", "actual_positive": n,
                      "forecast_floored": 0, "nonfinite_forecast": 0}
    f = fc[valid_actual]
    a = act[valid_actual]
    nonfinite = int(np.sum(~np.isfinite(f)))
    if nonfinite:
        return None, {"status": "nonfinite_forecast", "actual_positive": n,
                      "forecast_floored": 0, "nonfinite_forecast": nonfinite}
    eps = max(1e-9, 1e-4 * float(np.median(a)))
    floored = int(np.sum(f <= 0))
    f = np.where(f > 0, f, eps)
    return float(np.mean(np.log(f / a))), {
        "status": "ok", "actual_positive": n, "forecast_floored": floored,
        "nonfinite_forecast": 0,
    }


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
    sf = float(np.sum(np.maximum(fc[valid], 0.0)))
    sf = max(sf, 1e-6 * sa)
    return float(np.log(sf / sa))


def parse_arms(value: str) -> list[str]:
    arms = [a.strip().upper() for a in value.split(",") if a.strip()]
    if not arms or any(a not in {"A", "B", "C"} for a in arms):
        raise argparse.ArgumentTypeError("arms must be a comma-separated subset of A,B,C")
    return arms


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    local_src = repo_root / "src"
    if str(local_src) not in sys.path:
        sys.path.insert(0, str(local_src))
    from forecast_benchmark.profiles import get_profile, profile_names
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="board/private")
    ap.add_argument("--role", choices=("dev", "eval"), default="dev")
    ap.add_argument("--confirm-locked-eval", action="store_true")
    ap.add_argument("--legacy-src", default=os.environ.get("LEGACY_FORECAST_SRC", ""))
    ap.add_argument("--smartcast-src", default=os.environ.get("SMARTCAST_SRC", str(local_src)))
    ap.add_argument(
        "--profile", choices=profile_names(production_only=False), default="scipy_control",
        help="configuration used for Arm C",
    )
    ap.add_argument("--arms", type=parse_arms, default=parse_arms("A,B,C"))
    ap.add_argument("--boot", type=int, default=10_000)
    ap.add_argument("--out", default="")
    ap.add_argument("--allow-major-failures", action="store_true")
    args = ap.parse_args()

    if args.role == "eval" and not args.confirm_locked_eval:
        ap.error("Locked eval requires --confirm-locked-eval after code and sampler are frozen")
    if args.boot < 100:
        ap.error("--boot must be at least 100")

    root = Path(args.board)
    outdir = Path(args.out) if args.out else Path("board/results") / args.role
    outdir.mkdir(parents=True, exist_ok=True)

    legacy_src = Path(args.legacy_src).resolve() if args.legacy_src else None
    smartcast_src = Path(args.smartcast_src).resolve()
    if "A" in args.arms and (legacy_src is None or not (legacy_src / "forecast_benchmark/arps.py").exists()):
        ap.error("Arm A requires --legacy-src PATH_TO_ORIGINAL_REPO/src or LEGACY_FORECAST_SRC")
    if any(a in args.arms for a in ("B", "C")) and not (smartcast_src / "forecast_benchmark/arps.py").exists():
        ap.error("--smartcast-src must point to the SmartCast repository src directory")

    wells, series, manifest = read_board(root)
    board_id = manifest.get("board_id", "unknown")
    print(f"board_id={board_id}")

    armA = None
    if "A" in args.arms:
        armA = load_module(legacy_src / "forecast_benchmark/arps.py", "frozen_legacy_arps_v1")

    # SmartCast package is inserted only after Arm A's standalone file is loaded.
    sys.path.insert(0, str(smartcast_src))
    import forecast_benchmark.arps as armB  # noqa: E402
    import forecast_benchmark.smartcast as SC  # noqa: E402
    from forecast_benchmark.data import WellSeries  # noqa: E402
    from forecast_benchmark.split import Split  # noqa: E402

    def to_ws(key: str, tr: dict[str, Any]):
        return WellSeries(
            well_id=key,
            months=list(tr["months"]),
            oil=np.asarray(tr["oil"], dtype=float),
            gas=np.asarray(tr["gas"], dtype=float),
            water=np.asarray(tr["water"], dtype=float),
        )

    target_keys = sorted(k for k, w in wells.items() if w["role"] == args.role)
    cohort_keys = sorted(k for k, w in wells.items() if w["role"] == "cohort")
    if not target_keys:
        raise SystemExit(f"No wells with role={args.role}")
    if "C" in args.arms and not cohort_keys:
        raise SystemExit("SmartCast requires a nonempty disjoint role=cohort pool; eval fallback is prohibited")
    print(f"role={args.role}: {len(target_keys)} wells | cohort={len(cohort_keys)} wells")

    smart = None
    if "C" in args.arms:
        pool = [to_ws(k, series[k]["train"]) for k in cohort_keys]
        metadata = {k: {"basin": wells[k]["play"]} for k in wells}
        smart = SC.SmartCastProvider(pool, metadata, get_profile(args.profile))

    recs: dict[str, dict[str, dict[str, Any]]] = {a: {} for a in args.arms}
    runtime = {a: 0.0 for a in args.arms}
    failure_records: list[dict[str, str]] = []
    metric_ineligibility: dict[str, dict[str, dict[str, int | str]]] = {}

    for key in target_keys:
        if key not in series or "train" not in series[key] or "holdout" not in series[key]:
            raise ValueError(f"Missing train/holdout series for {key}")
        tr, hd = series[key]["train"], series[key]["holdout"]
        H = len(hd["months"])
        if H != 12:
            raise ValueError(f"Expected 12 calendar holdout months for {key}, found {H}")
        ws = to_ws(key, tr)
        split = Split(key, ws, list(hd["months"]), {p: np.asarray(hd[p], dtype=float) for p in PHASES})
        major = wells[key].get("major_phase") or (
            "oil" if np.nansum(np.maximum(tr["oil"], 0.0)) >= np.nansum(np.maximum(tr["gas"], 0.0)) / 6.0 else "gas"
        )
        metric_ineligibility[key] = {}
        for phase in PHASES:
            actual = np.asarray(hd[phase], dtype=float)
            positive = int(np.sum(np.isfinite(actual) & (actual > 0)))
            total_positive = float(np.nansum(np.maximum(actual, 0.0)))
            if positive < 2:
                metric_ineligibility[key][f"monthly_{phase}"] = {
                    "status": "insufficient_actual_positive",
                    "actual_positive_months": positive,
                }
            if total_positive <= 0:
                metric_ineligibility[key][f"cumulative_{phase}"] = {
                    "status": "nonpositive_actual_total",
                    "actual_positive_months": positive,
                }

        for arm in args.arms:
            t0 = time.perf_counter()
            d: dict[str, Any] = {"_play": wells[key]["play"], "_bucket": wells[key]["bucket"], "_major_phase": major}
            monthly_values: list[float] = []
            cumulative_values: list[float] = []
            phase_failures = 0
            for phase in PHASES:
                try:
                    if arm == "A":
                        fc = armA.arps_hyperbolic_bounded_b(np.asarray(tr[phase], dtype=float))(H)
                    elif arm == "B":
                        fc = armB.arps_hyperbolic_bounded_b(np.asarray(tr[phase], dtype=float))(H)
                    else:
                        fc = smart(split, phase)
                    if fc is None:
                        raise RuntimeError("provider returned None")
                    fc = np.asarray(fc, dtype=float)
                    actual = np.asarray(hd[phase], dtype=float)
                    mle, mdiag = monthly_avg_le(fc, actual)
                    cle = cumulative_le(fc, actual)
                    d[f"m_{phase}"] = mle
                    d[f"c_{phase}"] = cle
                    d[f"metric_diag_{phase}"] = mdiag
                    status = str(mdiag.get("status"))
                    if status in {"length_mismatch", "nonfinite_forecast"}:
                        raise RuntimeError(f"invalid forecast for metric: {status}; diag={mdiag}")
                    if mle is not None:
                        monthly_values.append(mle)
                    if cle is not None:
                        cumulative_values.append(cle)
                    elif float(np.nansum(np.maximum(actual, 0.0))) > 0:
                        raise RuntimeError("cumulative metric unavailable despite positive actual total")
                except Exception as exc:  # preserve exact failure, never silently omit
                    phase_failures += 1
                    failure_records.append({"well_key": key, "arm": arm, "phase": phase, "error": f"{type(exc).__name__}: {exc}"})
                    d[f"m_{phase}"] = None
                    d[f"c_{phase}"] = None
            runtime[arm] += time.perf_counter() - t0
            d["major"] = d.get(f"m_{major}")
            d["allph"] = float(np.mean(monthly_values)) if monthly_values else None
            d["cum_major"] = d.get(f"c_{major}")
            d["cum_allph"] = float(np.mean(cumulative_values)) if cumulative_values else None
            d["phase_failures"] = phase_failures
            recs[arm][key] = d

    # Distinguish actual-only metric ineligibility from provider/model failure.
    major_actual_ineligible = [
        key for key in target_keys
        if f"monthly_{recs[args.arms[0]][key]['_major_phase']}" in metric_ineligibility.get(key, {})
    ]
    major_model_missing = [
        (arm, key) for arm in args.arms for key in target_keys
        if key not in major_actual_ineligible and recs[arm].get(key, {}).get("major") is None
    ]
    diagnostics_payload = {
        "model_failures": failure_records,
        "major_actual_ineligible_wells": [
            {
                "well_key": key,
                "major_phase": recs[args.arms[0]][key]["_major_phase"],
                "reason": metric_ineligibility[key][f"monthly_{recs[args.arms[0]][key]['_major_phase']}"],
            }
            for key in major_actual_ineligible
        ],
        "all_actual_metric_ineligibility": metric_ineligibility,
    }
    (outdir / "failures.json").write_text(json.dumps(diagnostics_payload, indent=2) + "\n", encoding="utf-8")
    if major_model_missing and not args.allow_major_failures:
        raise SystemExit(
            f"Major-phase MODEL forecast/metric missing for {len(major_model_missing)} arm-well pairs; "
            f"see {outdir/'failures.json'}"
        )
    if major_actual_ineligible:
        print(
            f"NOTE: {len(major_actual_ineligible)} well(s) are major-monthly-metric ineligible "
            "for every arm because the holdout has fewer than two positive actual months; "
            "they remain available for eligible cumulative/all-phase metrics."
        )

    report: dict[str, Any] = {
        "schema_version": 2,
        "board_id": board_id,
        "role": args.role,
        "metric_warning": "provisional metric families; exact committee per-well aggregation unconfirmed",
        "arms": {}, "paired": {}, "segments": {}, "influence": {},
        "runtime_s": runtime,
        "failure_count": len(failure_records),
        "major_actual_ineligible_count": len(major_actual_ineligible),
        "major_actual_ineligible_wells": major_actual_ineligible,
        "smartcast_profile": args.profile if "C" in args.arms else None,
    }

    print(f"\n{'metric':<12}{'arm':<5}{'n':>6}{'median LE':>12}{'stdev':>10}{'SPEE':>10}{'catastrophic':>15}")
    print("-" * 70)
    common_by_metric: dict[str, list[str]] = {}
    for metric in METRICS:
        common = [k for k in target_keys if all(recs[a][k].get(metric) is not None for a in args.arms)]
        common_by_metric[metric] = common
        report["arms"][metric] = {}
        for arm in args.arms:
            vals = [recs[arm][k].get(metric) for k in common]
            s, med, sd, n = spee(vals)
            cat = float(np.mean([abs(float(v)) > CATASTROPHIC for v in vals])) if n else float("nan")
            report["arms"][metric][arm] = {
                "spee": s, "median_le": med, "stdev_le": sd, "n_common": n,
                "coverage": n / len(target_keys), "catastrophic_rate": cat,
                "runtime_s": round(runtime[arm], 3),
            }
            print(f"{metric:<12}{arm:<5}{n:>6}{med:>+12.4f}{sd:>10.4f}{s:>10.4f}{100*cat:>14.1f}%")
        print()

    def clustered(metric: str, base: str, challenge: str, B: int):
        keys = [k for k in target_keys if recs[base][k].get(metric) is not None and recs[challenge][k].get(metric) is not None]
        if len(keys) < 2:
            return keys, (float("nan"), float("nan")), float("nan")
        rng = np.random.default_rng(20260729)
        diffs = []
        for _ in range(B):
            sampled = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
            sb = spee([recs[base][k][metric] for k in sampled])[0]
            sc = spee([recs[challenge][k][metric] for k in sampled])[0]
            if np.isfinite(sb) and np.isfinite(sc):
                diffs.append(sc - sb)
        arr = np.asarray(diffs, dtype=float)
        if arr.size == 0:
            return keys, (float("nan"), float("nan")), float("nan")
        return keys, np.percentile(arr, [2.5, 97.5]), float(np.mean(arr < 0))

    print(f"Clustered bootstrap by well (B={args.boot})")
    for metric in METRICS:
        report["paired"][metric] = {}
        for challenge in [a for a in args.arms if a != "A"]:
            keys, (lo, hi), pbetter = clustered(metric, "A", challenge, args.boot)
            base_score = spee([recs["A"][k][metric] for k in keys])[0]
            challenge_score = spee([recs[challenge][k][metric] for k in keys])[0]
            diff = challenge_score - base_score
            verdict = "SIG better" if hi < 0 else ("SIG worse" if lo > 0 else "inconclusive")
            report["paired"][metric][f"{challenge}_vs_A"] = {
                "n_wells": len(keys), "diff": diff, "ci95": [float(lo), float(hi)],
                "bootstrap_probability_better": pbetter, "verdict": verdict,
            }
            print(f"  {metric:<10} {challenge} vs A: {diff:+.4f} CI [{lo:+.4f},{hi:+.4f}] P(better)={pbetter:.3f} {verdict}")

    # Leave-one-well-out influence for the production candidate versus A.
    if "A" in args.arms and "C" in args.arms:
        for metric in ("major", "allph"):
            keys = [k for k in target_keys if recs["A"][k].get(metric) is not None and recs["C"][k].get(metric) is not None]
            diffs = []
            for omitted in keys:
                kept = [k for k in keys if k != omitted]
                if len(kept) < 2:
                    continue
                diffs.append({
                    "omitted": omitted,
                    "diff": spee([recs["C"][k][metric] for k in kept])[0] - spee([recs["A"][k][metric] for k in kept])[0],
                })
            if diffs:
                values = [d["diff"] for d in diffs]
                report["influence"][metric] = {
                    "min_diff": min(values), "max_diff": max(values),
                    "removals_where_C_better": sum(v < 0 for v in values),
                    "n_removals": len(values),
                    "most_influential": max(diffs, key=lambda d: abs(d["diff"])),
                }

    # Segment scores use the same common wells across arms and carry n.
    for seg_name, field in (("play", "_play"), ("history_bucket", "_bucket")):
        report["segments"][seg_name] = {}
        groups = sorted({recs[args.arms[0]][k][field] for k in common_by_metric["major"]})
        print(f"\nBy {seg_name} (major metric; descriptive only)")
        for group in groups:
            keys = [k for k in common_by_metric["major"] if recs[args.arms[0]][k][field] == group]
            row = {"n_wells": len(keys), "scores": {}}
            for arm in args.arms:
                row["scores"][arm] = spee([recs[arm][k]["major"] for k in keys])[0]
            report["segments"][seg_name][group] = row
            scores = " ".join(f"{a}={row['scores'][a]:.4f}" for a in args.arms)
            print(f"  {group:<16} n={len(keys):>4} {scores}")

    report["provenance"] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scorer_sha256": file_sha256(Path(__file__)),
        "board_manifest_sha256": file_sha256(root / "manifest.json") if (root / "manifest.json").exists() else None,
        "legacy_arps_sha256": file_sha256(legacy_src / "forecast_benchmark/arps.py") if legacy_src else None,
        "smartcast_arps_sha256": file_sha256(smartcast_src / "forecast_benchmark/arps.py"),
        "smartcast_core_sha256": file_sha256(smartcast_src / "forecast_benchmark/smartcast.py"),
    }

    (outdir / "comparison.json").write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")
    (outdir / "failures.json").write_text(json.dumps(diagnostics_payload, indent=2) + "\n", encoding="utf-8")
    if args.role == "eval":
        (outdir / "LOCKED_EVAL_RECEIPT.json").write_text(json.dumps({
            "board_id": board_id,
            "created_utc": report["provenance"]["created_utc"],
            "provenance": report["provenance"],
            "warning": "Do not tune models against this locked evaluation result.",
        }, indent=2) + "\n", encoding="utf-8")

    print(f"\nWrote {outdir/'comparison.json'}")
    print("Aggregate metrics only. Raw well IDs and per-well forecasts remain private.")


if __name__ == "__main__":
    main()
