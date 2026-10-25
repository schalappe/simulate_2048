.PHONY: install dev lint format check test clean play train train-tiny train-small train-full eval typecheck test-cov eval-run

# Install dependencies
install:
	uv sync

# Install with dev dependencies
dev:
	uv sync --group dev

# Lint code
lint:
	uv run ruff check reinforce/ tests/ twentyfortyeight/

# Format code
format:
	uv run ruff format reinforce/ tests/ twentyfortyeight/
	uv run ruff check --fix reinforce/ tests/ twentyfortyeight/

# Type check
typecheck:
	uv run pyrefly check twentyfortyeight reinforce tests

# Run all checks (lint + typecheck)
check: lint typecheck

# Run tests
test:
	uv run pytest tests/

# Run tests with coverage
test-cov:
	uv run pytest tests/ --cov=twentyfortyeight --cov=reinforce --cov-report=term-missing

# Play the game manually
play:
	uv run python manuals_control.py

# --- Training (JAX) ---
# Each target trains in its own run directory under runs/ and resumes it when rerun.

# Train with tiny config on the local CPU (debugging, about 10 seconds for 100 steps)
train-tiny:
	uv run python -m reinforce.train --mode tiny --steps 100 --run-dir runs/tiny --allow-cpu

# Train with small config (needs a GPU; see docs/runpod.md)
train-small:
	uv run python -m reinforce.train --mode small --run-dir runs/small

# Train with full paper configuration (needs a GPU; runs for days, see docs/runpod.md)
train-full:
	uv run python -m reinforce.train --mode full --run-dir runs/full

# Default training target
train: train-tiny

# --- Evaluation (JAX) ---

# Evaluate an untrained network (smoke check of the search)
eval:
	uv run python -m reinforce.evaluate --mode tiny --games 2 --simulations 8

# Evaluate a training run
# Usage: make eval-run RUN_DIR=runs/full
eval-run:
	uv run python -m reinforce.evaluate --run-dir $(RUN_DIR) --games 10

# Clean up cache files
clean:
	rm -rf .ruff_cache .mypy_cache .pytest_cache __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
