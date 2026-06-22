PYTHON ?= python3

.PHONY: help dev compile lint test ci nox-tests

help:
	@echo "Targets:"
	@echo "  dev        Install editable + dev deps"
	@echo "  compile    Byte-compile src/"
	@echo "  lint       Run ruff checks"
	@echo "  test       Run pytest (PYTHONPATH=src)"
	@echo "  ci         compile + lint + test"
	@echo "  nox-tests  Run tests in isolated nox env"

dev:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -e '.[dev]'

compile:
	$(PYTHON) -m compileall -q src

lint:
	$(PYTHON) -m ruff check src tests

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

ci: compile lint test

nox-tests:
	$(PYTHON) -m nox -s tests
