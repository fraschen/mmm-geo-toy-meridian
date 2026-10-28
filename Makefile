PY := .venv/bin/python
# Keep macOS awake during long runs (no-op elsewhere).
AWAKE := $(shell command -v caffeinate >/dev/null 2>&1 && echo "caffeinate -dimsu")

RAW := data/raw/geo_all_channels.csv

.PHONY: all data eda eda-meridian eda-extended fit clean-outputs

CANDIDATES := aks_default_pool linear_default_pool aks_strong_pool \
              linear_strong_pool_lag4 linear_strong_pool
MODELS := $(foreach c,$(CANDIDATES),model_build/$(c)/meridian_model.binpb)

all: eda fit

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

# Phase 3: fit each approved candidate (skill script); skipped if already saved.
fit: $(MODELS)

model_build/%/meridian_model.binpb: $(RAW)
	$(AWAKE) $(PY) model_build/build_model.py --spec $* > model_build/$*.log 2>&1

clean-outputs:
	rm -rf outputs reports/figures model_build/eda_default
