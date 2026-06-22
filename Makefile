.PHONY: help setup data features forecast uncertainty optimize backtest all test clean

PYTHON ?= python

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Install pinned dependencies
	$(PYTHON) -m pip install -r requirements.txt

data:  ## [Layer 0] Download & clean the ETF universe -> data/processed/
	$(PYTHON) -m src.etl.download

features:  ## [Layer 0] Build model features from the clean panel -> data/processed/
	$(PYTHON) -m src.features.build

forecast:  ## [Layer 1] Walk-forward forecasts (headline + annex) -> data/processed/
	$(PYTHON) -m src.models.walkforward

uncertainty:  ## [Layer 2] Bootstrap-ensemble forecast uncertainty -> data/processed/
	$(PYTHON) -m src.models.uncertainty

optimize:  ## [Layer 3] Solve uncertainty-aware allocation (cvxpy) (TODO)
	@echo "TODO: implement src/optimization"

backtest:  ## [Layer 4] Backtest vs 1/N and naive Markowitz; write results/ (TODO)
	@echo "TODO: implement src/evaluation"

all: data features forecast uncertainty optimize backtest  ## Run the full pipeline

test:  ## Run the test suite
	$(PYTHON) -m pytest -q

clean:  ## Remove generated data & results (keeps .gitkeep)
	find data/raw data/processed results/figures results/metrics \
		-type f ! -name '.gitkeep' -delete
