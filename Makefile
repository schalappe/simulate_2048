.PHONY: install dev lint format check test clean play train train-tiny train-small train-full eval

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

# Train with tiny config (for debugging, ~1 minute)
train-tiny:
	uv run python -m reinforce.train --mode tiny --steps 100

# Train with small config (for experimentation, ~10 minutes)
train-small:
	uv run python -m reinforce.train --mode small --steps 10000

# Train with full paper configuration (~hours/days)
train-full:
	uv run python -m reinforce.train --mode full

# Default training target
train: train-small

# --- Evaluation (JAX) ---

# Evaluate Stochastic MuZero agent
# Usage: make eval CHECKPOINT=checkpoints
eval:
	uv run python -m reinforce.evaluate --games 10 --simulations 50

# Evaluate with custom checkpoint
# Usage: make eval-checkpoint CHECKPOINT=checkpoints/step_10000
eval-checkpoint:
	uv run python -m reinforce.evaluate --games 10 --checkpoint $(CHECKPOINT)

# Clean up cache files
clean:
	rm -rf .ruff_cache .mypy_cache .pytest_cache __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
