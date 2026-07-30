PYTHON ?= python3

.PHONY: install compile test audit check bench arps-experiments lag-diagnostic spee-validate spee-strict-auto spee-vendor-best stress stress-quick allocation-noise throughput real-holdout real-select

install:
	$(PYTHON) -m pip install -e ".[dev]"

compile:
	$(PYTHON) -m compileall -q src tests examples scripts

test:
	$(PYTHON) -m pytest -q

audit:
	$(PYTHON) scripts/audit_public_history.py --current-only

check: compile test audit

bench:
	PYTHONPATH=src $(PYTHON) examples/run_benchmark.py

arps-experiments:
	PYTHONPATH=src $(PYTHON) scripts/run_python_arm_experiments.py

lag-diagnostic:
	PYTHONPATH=src $(PYTHON) scripts/diagnose_reporting_lag.py

spee-validate:
	PYTHONPATH=src $(PYTHON) scripts/validate_spee_input.py spee_data

spee-strict-auto:
	PYTHONPATH=src $(PYTHON) scripts/run_spee_strict_auto.py spee_data

spee-vendor-best:
	PYTHONPATH=src $(PYTHON) scripts/run_spee_vendor_best.py spee_data

stress:
	PYTHONPATH=src $(PYTHON) scripts/run_stress_board.py --wells 180 --seeds 1-30

stress-quick:
	PYTHONPATH=src $(PYTHON) scripts/run_stress_board.py --wells 24 --seeds 1-3 --workers 1 --bootstrap-reps 50 --aggregate-bootstrap-reps 100

allocation-noise:
	PYTHONPATH=src $(PYTHON) scripts/run_allocation_noise_evaluation.py --wells 180 --seeds 1-50

throughput:
	PYTHONPATH=src $(PYTHON) scripts/run_throughput_test.py --wells 1000 --horizon 360

real-holdout:
	@echo "Use scripts/run_real_holdout.py with source attestation; see docs/REAL_DATA_VALIDATION.md"

real-select:
	@echo "Set LEGACY_FORECAST_SRC to the original forecast-benchmark src directory"
	PYTHONPATH=src $(PYTHON) scripts/run_v13_selection.py --legacy-src "$(LEGACY_FORECAST_SRC)" --boot 20000 --report-only
