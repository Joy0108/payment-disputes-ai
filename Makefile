.PHONY: help install data build train eval test lint serve clean docker deploy

PY ?= python
export PYTHONPATH := src

help:
	@echo "install  install the package with dev and serving extras"
	@echo "data     regenerate the raw corpora from the seed"
	@echo "build    build the bronze, silver and gold layers"
	@echo "train    train the four models"
	@echo "eval     full evaluation and promotion gates"
	@echo "test     run the test suite"
	@echo "serve    run the FastAPI service"

install:
	$(PY) -m pip install -e ".[dev,serving]"

data:
	$(PY) scripts/build_data.py

build: data
	$(PY) -m disputes.cli build

train: build
	$(PY) -m disputes.cli train

eval:
	$(PY) -m disputes.cli eval

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests scripts

serve:
	$(PY) -m disputes.cli serve --reload

clean:
	rm -rf artifacts reports data/silver data/gold .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

docker:
	docker compose up --build

deploy:
	kubectl apply -k deploy/
