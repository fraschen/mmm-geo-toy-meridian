PY := .venv/bin/python
# Keep macOS awake during long runs (no-op elsewhere).
AWAKE := $(shell command -v caffeinate >/dev/null 2>&1 && echo "caffeinate -dimsu")

RAW := data/raw/geo_all_channels.csv

.PHONY: fit-round1 results reports docker calibration budget gates reports-meridian all data eda eda-meridian eda-extended fit select cv-fit validate clean-outputs

CANDIDATES := linear_strong_pool_lag4 linear_strong_pool
ROUND1 := aks_default_pool linear_default_pool aks_strong_pool
MODELS := $(foreach c,$(CANDIDATES),model_build/$(c)/meridian_model.binpb)

# Every phase. Gates may fail (blocking failures are reported); the reports still build.
all: eda fit select validate reports-meridian budget calibration
	-$(MAKE) gates
	$(MAKE) reports

data: $(RAW)

$(RAW):
	$(PY) scripts/download_data.py

# Phase 1: Meridian's EDA report (skill-generated script, default spec,
# prior draws only) + the extended EDA (summary JSON + figures).
eda: eda-meridian eda-extended

eda-meridian: data
	$(AWAKE) $(PY) model_build/build_model.py --spec eda_default --eda-only

eda-extended: data
	$(PY) scripts/eda.py
	$(PY) scripts/eda_findings.py

# Phase 3: round 2 (R1, R2; default sampler). Round 1 (C1-C3) diagnostics are committed;
# `make fit-round1` refits those models.
fit: $(MODELS)

ROUND1_MODELS := $(foreach c,$(ROUND1),model_build/$(c)/meridian_model.binpb)
fit-round1: $(ROUND1_MODELS)

$(MODELS) $(ROUND1_MODELS): model_build/%/meridian_model.binpb: $(RAW)
	$(AWAKE) $(PY) model_build/build_model.py --spec $* > model_build/$*.log 2>&1

select: $(MODELS)
	$(PY) scripts/select_model.py

# Phase 4: rolling-origin CV folds for the selected spec (the 143-week fold is
# the 13-week holdout), then forecast scoring.
SELECTED := linear_strong_pool_lag4
FOLDS := 104 117 130 143
FOLD_MODELS := $(foreach t,$(FOLDS),model_build/cv/$(SELECTED)_train$(t)/meridian_model.binpb)

cv-fit: $(FOLD_MODELS)

model_build/cv/$(SELECTED)_train%/meridian_model.binpb: $(RAW)
	mkdir -p model_build/cv
	$(AWAKE) $(PY) model_build/build_model.py --spec $(SELECTED) --train-weeks $* > model_build/cv/train$*.log 2>&1

validate: cv-fit
	$(PY) scripts/validate_cv.py

# Phase 5: Meridian health check + Model Results Summary (skill script).
reports-meridian: model_build/linear_strong_pool_lag4/meridian_model.binpb
	$(AWAKE) $(PY) model_build/results_report.py

# Phase 5: automated gates (exits non-zero on any blocking failure). Rules and
# thresholds: config/gates.toml; documentation: reports/gates.md (make reports).
gates:
	$(PY) scripts/gates.py

# Phase 7: budget scenarios (skill script).
budget: model_build/linear_strong_pool_lag4/meridian_model.binpb
	$(AWAKE) $(PY) model_build/run_budget_optimization.py

# Phase 8: HYPOTHETICAL lift test, calibrated refit, comparison.
LIFT := data/lift_tests/lift_tests.csv
CALIBRATED := model_build/linear_strong_pool_lag4_calibrated/meridian_model.binpb

$(LIFT):
	$(PY) scripts/make_hypothetical_lift_test.py

$(CALIBRATED): $(RAW) $(LIFT)
	$(AWAKE) $(PY) model_build/build_model.py --spec linear_strong_pool_lag4_calibrated > model_build/linear_strong_pool_lag4_calibrated.log 2>&1

calibration: $(CALIBRATED)
	$(PY) scripts/calibration.py

# Phase 6: channel results file + markdown reports (+ README).
results:
	$(PY) scripts/analysis_data.py
	$(PY) scripts/channel_results.py

reports: results
	$(PY) scripts/make_reports.py
	$(PY) scripts/make_gates_report.py
	$(PY) scripts/make_readme.py

# Phase 9: Docker image from the lockfile; verifies EDA reproduces and sampling runs.
IMAGE := mmm-geo-toy-meridian
docker:
	docker build -t $(IMAGE) .
	docker run --rm $(IMAGE) sh -c 'cp outputs/eda/eda_summary.json /tmp/ref.json && make eda-extended && .venv/bin/python scripts/compare_json.py /tmp/ref.json outputs/eda/eda_summary.json'
	docker run --rm $(IMAGE) sh -c 'make data && .venv/bin/python scripts/docker_smoke.py'

clean-outputs:
	rm -rf outputs reports/figures model_build/eda_default
