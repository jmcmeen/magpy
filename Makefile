# MagPy -- uv-based developer tasks.
# Run `make help` for the list.

UV ?= uv

.PHONY: help sync lock run cli test lint format check clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

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
	$(UV) run ruff check src

format: ## Format with black and apply ruff autofixes
	$(UV) run black src
	$(UV) run ruff check --fix src

check: lint test ## Lint then test

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
