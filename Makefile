PYTHON := python

.PHONY: help data train forecast

help:
	@echo "Usage:"
	@echo "  make data     # generate synthetic dataset"
	@echo "  make train    # train models and evaluate"
	@echo "  make forecast # generate 28-day forecast"

data:
	$(PYTHON) -m src.data

train:
	$(PYTHON) -m src.train

forecast:
	$(PYTHON) -m src.forecast
