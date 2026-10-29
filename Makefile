PY := .venv/bin/python
# Keep macOS awake during long runs (no-op elsewhere).
AWAKE := $(shell command -v caffeinate >/dev/null 2>&1 && echo "caffeinate -dimsu")

# One production model (config/model.toml).
RAW := data/raw/geo_all_channels.csv
MODEL := model_build/model/meridian_model.binpb
FOLDS := 104 117 130 143
FOLD_MODELS := $(foreach t,$(FOLDS),model_build/cv/train$(t)/meridian_model.binpb)
CALIBRATED := model_build/model_calibrated/meridian_model.binpb
LIFT := data/lift_tests/lift_tests.csv

.PHONY: all data eda eda-meridian eda-extended fit convergence cv-fit validate \
        reports-meridian budget gates results reports calibration docker clean-outputs

# Production pipeline. Stops if the model fails its convergence gate. The acceptance
# gates may fail (blocking failures are reported in reports/results.md); reports still build.
all: eda fit convergence validate reports-meridian budget
	-$(MAKE) gates
	$(MAKE) reports

# The raw data file is committed in data/raw/; its checksum is verified when it is loaded.
data:
	@test -f $(RAW) || { echo "Missing $(RAW)"; exit 1; }

# EDA: Meridian's EDA report for the production spec (prior draws only) + extended EDA.
eda: eda-meridian eda-extended

eda-meridian: data
	$(AWAKE) $(PY) model_build/build_model.py --eda-only

eda-extended: data
	$(PY) scripts/eda.py
	$(PY) scripts/eda_findings.py

# Fit the production model (skill script); skipped if already saved.
fit: $(MODEL)

$(MODEL): $(RAW) config/model.toml
	mkdir -p model_build/model
	$(AWAKE) $(PY) model_build/build_model.py > model_build/model/fit.log 2>&1

# Convergence gate on the production fit (config/gates.toml); exits non-zero on failure.
convergence: $(MODEL)
	$(PY) scripts/check_convergence.py

# Validation: refits of the same model on weeks 1-N (the 143-week refit is the
# 13-week holdout), each forecasting the next 13 weeks.
cv-fit: $(FOLD_MODELS)

model_build/cv/train%/meridian_model.binpb: $(RAW) config/model.toml
	mkdir -p model_build/cv/train$*
	$(AWAKE) $(PY) model_build/build_model.py --train-weeks $* > model_build/cv/train$*/fit.log 2>&1

validate: cv-fit
	$(PY) scripts/validate_cv.py

# Meridian health check + Model Results Summary (skill script).
reports-meridian: $(MODEL)
	$(AWAKE) $(PY) model_build/results_report.py

# Budget scenarios (skill script).
budget: $(MODEL)
	$(AWAKE) $(PY) model_build/run_budget_optimization.py

# Acceptance gates (exits non-zero on any blocking failure). Rules and thresholds:
# config/gates.toml; documentation: Part 3 of reports/results.md.
gates:
	$(PY) scripts/gates.py

# Channel results file + markdown reports + README.
results:
	$(PY) scripts/analysis_data.py
	$(PY) scripts/channel_results.py

reports: results
	$(PY) scripts/make_reports.py
	$(PY) scripts/make_readme.py

# Optional, not part of production: refit with a HYPOTHETICAL lift test.
$(LIFT):
	$(PY) scripts/make_hypothetical_lift_test.py

$(CALIBRATED): $(RAW) $(LIFT) config/model.toml
	mkdir -p model_build/model_calibrated
	$(AWAKE) $(PY) model_build/build_model.py --calibrated > model_build/model_calibrated/fit.log 2>&1

calibration: $(CALIBRATED)
	$(PY) scripts/calibration.py

# Container: build from the lockfile; verify the EDA reproduces and sampling runs.
# The image's default command runs the production pipeline (make all).
IMAGE := mmm-geo-toy-meridian
docker:
	docker build -t $(IMAGE) .
	docker run --rm $(IMAGE) sh -c 'cp outputs/eda/eda_summary.json /tmp/ref.json && make eda-extended && .venv/bin/python scripts/compare_json.py /tmp/ref.json outputs/eda/eda_summary.json'
	docker run --rm $(IMAGE) sh -c 'make data && .venv/bin/python scripts/docker_smoke.py'

clean-outputs:
	rm -rf outputs reports/figures
