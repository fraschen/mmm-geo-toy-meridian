PY := .venv/bin/python
# Keep macOS awake during long runs (no-op elsewhere).
AWAKE := $(shell command -v caffeinate >/dev/null 2>&1 && echo "caffeinate -dimsu")

RAW := data/raw/geo_all_channels.csv

.PHONY: all data eda eda-meridian eda-extended fit fit-long select cv-fit validate clean-outputs

CANDIDATES := aks_default_pool linear_default_pool aks_strong_pool \
              linear_strong_pool_lag4 linear_strong_pool
MODELS := $(foreach c,$(CANDIDATES),model_build/$(c)/meridian_model.binpb)

all: eda fit fit-long select validate

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

$(MODELS): model_build/%/meridian_model.binpb: $(RAW)
	$(AWAKE) $(PY) model_build/build_model.py --spec $* > model_build/$*.log 2>&1

# Round 3 (post hoc): C1-C3 refit with the longer warm-up.
ROUND3 := $(foreach c,aks_default_pool linear_default_pool aks_strong_pool,model_build/$(c)__long/meridian_model.binpb)

fit-long: $(ROUND3)

$(ROUND3): model_build/%__long/meridian_model.binpb: $(RAW)
	$(AWAKE) $(PY) model_build/build_model.py --spec $* --sampler long > model_build/$*__long.log 2>&1

select: $(MODELS) $(ROUND3)
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

clean-outputs:
	rm -rf outputs reports/figures model_build/eda_default
