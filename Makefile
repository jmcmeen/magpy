# MagPy -- uv-based developer tasks.
# Run `make help` for the list.

UV ?= uv

.PHONY: help install sync lock run cli test lint fmt fmt-check check clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: sync ## Alias for sync

sync: ## Create/refresh the venv and install deps (incl. dev group)
	$(UV) sync

lock: ## Resolve and write uv.lock
	$(UV) lock

run: ## Launch the GUI
	$(UV) run magpy-gui

cli: ## Run the CLI (pass args via ARGS=...)
	$(UV) run magpy $(ARGS)

test: ## Run the test suite
	$(UV) run pytest

lint: ## Lint with ruff
	$(UV) run ruff check src tests

fmt: ## Auto-format and apply lint fixes
	$(UV) run ruff format src tests
	$(UV) run ruff check --fix src tests

fmt-check: ## Check formatting without modifying files
	$(UV) run ruff format --check src tests

check: lint fmt-check test ## Lint, format check, then test

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
