# PYTHONPATH=scripts lets every script find the shared modules (common.py, model_spec.py).
PY := env PYTHONPATH=scripts .venv/bin/python
# Keep macOS awake during long runs (no-op elsewhere).
AWAKE := $(shell command -v caffeinate >/dev/null 2>&1 && echo "caffeinate -dimsu")

# One production model (config/model.toml).
RAW := data/raw/geo_all_channels.csv
MODEL := models/model/meridian_model.binpb
FOLDS := 104 117 130 143
FOLD_MODELS := $(foreach t,$(FOLDS),models/cv/train$(t)/meridian_model.binpb)
CALIBRATED := models/model_calibrated/meridian_model.binpb
LIFT := data/lift_tests/lift_tests.csv

.PHONY: all data eda fit convergence cv-fit validate \
        reports-meridian budget gates results reports calibration docker clean-outputs

# Production pipeline. Stops if the model fails its convergence gate. The acceptance
# gates may fail (blocking failures are reported in reports/results.md); reports still build.
all: eda fit convergence validate reports-meridian budget
	-$(MAKE) gates
	$(MAKE) reports

# The raw data file is committed in data/raw/; its checksum is verified when it is loaded.
data:
	@test -f $(RAW) || { echo "Missing $(RAW)"; exit 1; }

# EDA: data checks and exploratory tests -> outputs/01_eda/ and reports/eda.md.
eda: data
	$(PY) scripts/01_eda/eda.py
	$(PY) scripts/01_eda/eda_findings.py

# Fit the production model (skill script); skipped if already saved.
fit: $(MODEL)

$(MODEL): $(RAW) config/model.toml
	mkdir -p models/model
	$(AWAKE) $(PY) scripts/02_fit/build_model.py > models/model/fit.log 2>&1

# Convergence gate on the production fit (config/gates.toml); exits non-zero on failure.
convergence: $(MODEL)
	$(PY) scripts/02_fit/check_convergence.py

# Validation: refits of the same model on weeks 1-N (the 143-week refit is the
# 13-week holdout), each forecasting the next 13 weeks.
cv-fit: $(FOLD_MODELS)

models/cv/train%/meridian_model.binpb: $(RAW) config/model.toml
	mkdir -p models/cv/train$*
	$(AWAKE) $(PY) scripts/02_fit/build_model.py --train-weeks $* > models/cv/train$*/fit.log 2>&1

validate: cv-fit
	$(PY) scripts/03_validate/validate_cv.py

# Meridian health check + Model Results Summary (skill script).
reports-meridian: $(MODEL)
	$(AWAKE) $(PY) scripts/03_validate/results_report.py

# Budget scenarios (skill script).
budget: $(MODEL)
	$(AWAKE) $(PY) scripts/04_scenarios/run_budget_optimization.py

# Acceptance gates (exits non-zero on any blocking failure). Rules and thresholds:
# config/gates.toml; documentation: Part 3 of reports/results.md.
gates:
	$(PY) scripts/03_validate/gates.py

# Channel results file + the results report (the README is written by hand).
results:
	$(PY) scripts/05_report/analysis_data.py
	$(PY) scripts/05_report/channel_results.py

reports: results
	$(PY) scripts/05_report/make_reports.py

# Optional, not part of production: refit with a HYPOTHETICAL lift test.
$(LIFT):
	$(PY) scripts/04_scenarios/make_hypothetical_lift_test.py

$(CALIBRATED): $(RAW) $(LIFT) config/model.toml
	mkdir -p models/model_calibrated
	$(AWAKE) $(PY) scripts/02_fit/build_model.py --calibrated > models/model_calibrated/fit.log 2>&1

calibration: $(CALIBRATED)
	$(PY) scripts/04_scenarios/calibration.py

# Container: build from the lockfile; verify the EDA reproduces and sampling runs.
# The image's default command runs the production pipeline (make all).
IMAGE := mmm-geo-toy-meridian
docker:
	docker build -t $(IMAGE) .
	docker run --rm $(IMAGE) sh -c 'cp outputs/01_eda/eda_summary.json /tmp/ref.json && make eda && .venv/bin/python docker/check_eda_reproduces.py /tmp/ref.json outputs/01_eda/eda_summary.json'
	docker run --rm $(IMAGE) sh -c 'make data && PYTHONPATH=scripts .venv/bin/python docker/check_sampling.py'

clean-outputs:
	rm -rf outputs reports/figures
