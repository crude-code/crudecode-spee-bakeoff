# SmartCast v1.5 — Selective Cohort Protocol

## Frozen policy

`selective_cohort_v1` returns exact frozen SciPy for every target/phase except:

- target primary phase is oil;
- play metadata is `DJ`, `EAGLE_FORD`, or `DELAWARE`; and
- forecasted phase is oil.

For that one route, it applies the unchanged `cohort_conservative` settings:
play-specific cohort, no global fallback, minimum 8 peers, full support at 30
peers, linear blending, and the historical thin-history weight.

The route does not use reporting provenance, producing days, actual holdout
values, operator, or a learned classifier. Secondary phases remain exact SciPy
because the policy selection was based on major-phase evidence.

## Development evidence

On 749 spent wells, the selected policy changed major-phase SPEE from 0.1719
to 0.1592, catastrophic rate from 7.076% to 6.676%, and p95 absolute log error
from 0.8805 to 0.8595. The policy passed the predeclared spent-role guardrails
and was selected for one fresh confirmation. These are not production claims.

## Confirmation

Use a board with a previously untouched `role=confirm3` and a disjoint cohort.
Run only:

```powershell
python scripts/run_v15_selective_confirmation.py `
  --board "board/private-v15" `
  --legacy-src "C:\path\to\forecast-benchmark-main\src" `
  --smartcast-src ".\src" `
  --boot 30000 `
  --out "board/results/v15-selective-confirm3" `
  --report-only
```

The runner cannot rank profiles. Promotion requires improvement probability,
minimum practical improvement, catastrophic/p95/cumulative/segment/runtime
guardrails, zero profile failures, and exact SciPy parity.

Until every gate passes, deploy `scipy_control`.
