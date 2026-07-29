# Allocation-Noise Classifier Experiment

## Boundary

The classifier is under `forecast_benchmark.experiments`. Production SmartCast, Strict Auto, and Vendor Best do not import it. It does not modify forecasts.

## Why it exists

Universal recent-month skipping failed decisively. The only replicated positive signal from that line of work appeared in the synthetic allocation-noise scenario. Before testing a forecast intervention, the system must first show that allocation-like noise can be detected with high precision and without flagging clean wells.

## Features

The fixed detector uses robust log-trend residuals, repeated sign reversals, large month-to-month jumps, and second differences across available phases. It requires a conjunction of signals so terminal dips and isolated downtime do not trigger it easily.

## Run

```bash
python scripts/run_allocation_noise_evaluation.py \
  --seeds 1-50 \
  --wells 180
```

Default classifier gates:

- precision at least 90%;
- clean-well false-positive rate at most 5%.

Passing this classifier gate does **not** promote re-anchoring. A separate paired forecast experiment would still need to pass the full candidate promotion protocol.
