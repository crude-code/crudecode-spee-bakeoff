#!/usr/bin/env python3
"""Diagnose the dispersion tail on already-spent real-board roles.

This script is explicitly diagnostic. It compares the exact SciPy control,
the replicated ``cohort_conservative`` challenger, and the bounded
``gated_cohort_v1`` challenger on dev/confirm/confirm2. It never reads eval and
never makes a deployment decision.

Private output contains well keys and must remain under board/results/.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from run_v13_selection import (  # type: ignore
    CATASTROPHIC,
    PHASES,
    bootstrap_diff,
    major_phase,
    monthly_le,
    read_board,
    spee,
)


def endpoint_features(q: np.ndarray) -> tuple[float | None, float | None, str]:
    q = np.asarray(q, dtype=float)
    valid = np.flatnonzero(np.isfinite(q) & (q > 0))
    if valid.size < 4:
        return None, None, "sparse"
    tail_idx = valid[-min(6, valid.size):]
    tail = q[tail_idx]
    prior = q[valid[-min(4, valid.size):-1]]
    ratio = float(tail[-1] / max(float(np.median(prior)), 1e-9)) if prior.size else None
    vol = float(np.std(np.diff(np.log(np.maximum(tail, 1e-9))))) if tail.size >= 4 else None
    if ratio is not None and ratio < 0.60:
        band = "low"
    elif ratio is not None and ratio > 1.60:
        band = "high"
    elif vol is not None and vol > 0.45:
        band = "volatile"
    else:
        band = "normal"
    return ratio, vol, band


def tail_summary(values: dict[str, float]) -> dict[str, object]:
    keys = sorted(values)
    x = np.asarray([values[k] for k in keys], dtype=float)
    result = dict(spee(x.tolist()))
    ax = np.abs(x)
    result.update({
        "catastrophic_rate": float(np.mean(ax > CATASTROPHIC)),
        "abs_le_p90": float(np.percentile(ax, 90)),
        "abs_le_p95": float(np.percentile(ax, 95)),
        "abs_le_p99": float(np.percentile(ax, 99)),
        "abs_le_max": float(np.max(ax)),
    })
    full = float(result["spee"])
    influence = []
    if len(keys) >= 3:
        for i, key in enumerate(keys):
            without = np.delete(x, i)
            score_without = float(spee(without.tolist())["spee"])
            influence.append((full - score_without, key, float(x[i]), score_without))
    influence.sort(reverse=True)
    result["top_influence"] = [
        {"well_key": key, "influence": float(inf), "le": le, "score_without": sw}
        for inf, key, le, sw in influence[:10]
    ]
    if influence:
        ordered_keys = [key for _, key, _, _ in influence]
        remove1 = set(ordered_keys[:1])
        remove_pct = set(ordered_keys[:max(1, int(np.ceil(0.01 * len(keys))))])
        result["score_without_top1"] = float(spee([values[k] for k in keys if k not in remove1])["spee"])
        result["score_without_top1pct"] = float(spee([values[k] for k in keys if k not in remove_pct])["spee"])
    return result


def main() -> int:
    here = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="board/private-v131")
    ap.add_argument("--smartcast-src", default=str(here / "src"))
    ap.add_argument("--roles", default="dev,confirm,confirm2")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out", default="board/results/v14-tail")
    args = ap.parse_args()
    if args.boot < 1000:
        ap.error("--boot must be at least 1000")

    root = Path(args.board)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wells, series, manifest = read_board(root)
    roles = tuple(x.strip() for x in args.roles.split(",") if x.strip())
    invalid = set(roles) - {"dev", "confirm", "confirm2"}
    if invalid:
        ap.error(f"diagnostic roles only; invalid={sorted(invalid)}")

    sys.path.insert(0, str(Path(args.smartcast_src).resolve()))
    from forecast_benchmark.arps_scipy_v1 import arps_bounded_scipy_v1  # noqa: E402
    from forecast_benchmark.data import WellSeries  # noqa: E402
    from forecast_benchmark.profiles import get_profile  # noqa: E402
    from forecast_benchmark.smartcast import SmartCastProvider  # noqa: E402
    from forecast_benchmark.split import Split  # noqa: E402

    def to_ws(key: str, tr: dict) -> WellSeries:
        return WellSeries(key, list(tr["months"]), tr["oil"].copy(), tr["gas"].copy(), tr["water"].copy())

    cohort_keys = sorted(k for k, w in wells.items() if w["role"] == "cohort")
    pool = [to_ws(k, series[k]["train"]) for k in cohort_keys]
    metadata = {k: {"basin": wells[k]["play"]} for k in wells}
    profiles = ("cohort_conservative", "gated_cohort_v1")
    providers = {name: SmartCastProvider(pool, metadata, get_profile(name)) for name in profiles}
    arms = ("scipy_control", *profiles)

    values: dict[str, dict[str, dict[str, float]]] = {
        role: {arm: {} for arm in arms} for role in roles
    }
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    routing: dict[str, Counter[str]] = {name: Counter() for name in profiles}

    targets = sorted(k for k, w in wells.items() if w["role"] in roles)
    print(f"board_id={manifest.get('board_id')} roles={roles} targets={len(targets)} cohort={len(cohort_keys)}")
    for idx, key in enumerate(targets, 1):
        role = wells[key]["role"]
        tr, hd = series[key]["train"], series[key]["holdout"]
        phase = major_phase(tr)
        actual = np.asarray(hd[phase], dtype=float)
        if int(np.sum(np.isfinite(actual) & (actual > 0))) < 2:
            continue
        horizon = len(actual)
        q = np.asarray(tr[phase], dtype=float)
        ratio, vol, endpoint_band = endpoint_features(q)
        forecasts: dict[str, np.ndarray] = {}
        try:
            forecasts["scipy_control"] = np.asarray(arps_bounded_scipy_v1(q.copy())(horizon), dtype=float)
        except Exception as exc:
            failures.append({"arm": "scipy_control", "well_key": key, "error": f"{type(exc).__name__}: {exc}"})
            continue
        ws = to_ws(key, tr)
        split = Split(key, ws, list(hd["months"]), {p: np.asarray(hd[p], dtype=float) for p in PHASES})
        diag_by_profile = {}
        for name, provider in providers.items():
            try:
                fc = provider(split, phase)
                if fc is None:
                    raise RuntimeError("provider returned None")
                forecasts[name] = np.asarray(fc, dtype=float)
                wd = provider.diagnostics[(key, horizon)]
                pd = next(d for d in wd.phases if d.phase == phase)
                diag_by_profile[name] = pd
                routing[name][pd.cohort_gate_reason or pd.routing_reason] += 1
            except Exception as exc:
                failures.append({"arm": name, "well_key": key, "error": f"{type(exc).__name__}: {exc}"})

        if set(forecasts) != set(arms):
            continue
        record: dict[str, object] = {
            "well_key": key,
            "role": role,
            "play": wells[key]["play"],
            "bucket": wells[key]["bucket"],
            "major_phase": phase,
            "endpoint_ratio": ratio,
            "endpoint_volatility": vol,
            "endpoint_band": endpoint_band,
        }
        for arm, fc in forecasts.items():
            le, status = monthly_le(fc, actual)
            if le is None:
                failures.append({"arm": arm, "well_key": key, "error": str(status)})
                break
            values[role][arm][key] = float(le)
            record[f"le_{arm}"] = float(le)
        else:
            for name, pd in diag_by_profile.items():
                record[f"weight_{name}"] = float(pd.cohort_weight)
                record[f"gate_{name}"] = pd.cohort_gate_reason
                record[f"support_{name}"] = int(pd.cohort_support)
                record[f"dispersion_{name}"] = pd.cohort_log_mad
                record[f"similarity_{name}"] = pd.cohort_similarity_error
            rows.append(record)
        if idx % 50 == 0 or idx == len(targets):
            print(f"  scored {idx}/{len(targets)}")

    report: dict[str, object] = {
        "schema_version": 1,
        "protocol": "diagnostic only; spent dev/confirm/confirm2; eval never read",
        "board_id": manifest.get("board_id"),
        "roles": roles,
        "arms": {},
        "pairwise": {},
        "segments": {},
        "routing": {name: dict(counts) for name, counts in routing.items()},
        "failure_count": len(failures),
    }
    for role in roles:
        common = sorted(set.intersection(*(set(values[role][a]) for a in arms)))
        report["arms"][role] = {arm: tail_summary({k: values[role][arm][k] for k in common}) for arm in arms}
        report["pairwise"][role] = {}
        for name in profiles:
            report["pairwise"][role][f"{name}_vs_scipy"] = bootstrap_diff(
                values[role]["scipy_control"], values[role][name], common, args.boot
            )

    segment_fields = ("play", "bucket", "major_phase", "endpoint_band")
    for field in segment_fields:
        grouped = defaultdict(list)
        for row in rows:
            grouped[(str(row["role"]), str(row[field]))].append(row)
        field_payload = {}
        for (role, group), group_rows in sorted(grouped.items()):
            entry = {"n": len(group_rows)}
            for arm in arms:
                vals = [float(r[f"le_{arm}"]) for r in group_rows]
                entry[arm] = spee(vals)
            field_payload[f"{role}|{group}"] = entry
        report["segments"][field] = field_payload

    (out / "tail_diagnostics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (out / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    if rows:
        fields = sorted({key for row in rows for key in row})
        with (out / "per_well_private.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {out / 'tail_diagnostics.json'}")
    print("DIAGNOSTIC ONLY: do not use trimmed scores as the competition metric.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
