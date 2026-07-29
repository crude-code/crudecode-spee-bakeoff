# Public history audit

`forecast-benchmark` is public. `benchmark_data/` is not. The benchmark snapshot
can contain real wells, API numbers, operators, and private database-derived
production histories. `.env` can contain database or API credentials. Those two
paths must never be tracked, even once.

## What to run before pushing

```bash
git fetch --all --tags
python scripts/audit_public_history.py
python -m pytest -q
```

The audit checks two things:

1. the current tracked tree has no `.env` and nothing under `benchmark_data/`
2. the available Git history has never touched `.env` or `benchmark_data/`

The history scan only sees commits present in your local clone. For a complete
check, fetch first.

## If it fails

Do not simply delete the file and push. If a real snapshot or credential reached
public history, the current tree is clean but the old commit still exists.
Escalate before rewriting history.

Safe cases:

- a false positive in a local throwaway branch before push
- a dummy test file created only inside a temporary test repo

Unsafe cases:

- real `benchmark_data/*.csv` or JSON committed to the public repo
- `.env` committed with a real database URL or API key
- private vendor/fund/internal text committed into a tracked doc

## Why this exists

The normal hygiene test blocks current tracked leaks. This script is the
maintainer pre-push check that also answers the second question: did anything
walk through the door before the guard existed?
