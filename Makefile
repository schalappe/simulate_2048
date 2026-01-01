.PHONY: install dev lint format check test clean play train train-small train-full eval eval-mcts eval-muzero

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

# --- Training ---

# Train with small config (for testing/debugging)
train-small:
	uv run python reinforce/train.py --mode small --steps 1000

# Train with full paper configuration
train-full:
	uv run python reinforce/train.py --mode full --steps 100000

# Default training target (alias for train-small)
train: train-small

# --- Evaluation ---

# Evaluate basic MCTS agent (no neural network)
eval-mcts:
	uv run python reinforce/evaluate.py --method mcts --length 10 --simulations 10

# Evaluate Stochastic MuZero with a checkpoint
# Usage: make eval-muzero CHECKPOINT=checkpoints/step_10000
eval-muzero:
	uv run python reinforce/evaluate.py --method stochastic_muzero --length 10 --checkpoint $(CHECKPOINT)

# Default evaluation target (basic MCTS)
eval: eval-mcts

# Clean up cache files
clean:
	rm -rf .ruff_cache .mypy_cache .pytest_cache __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
