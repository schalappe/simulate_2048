.PHONY: install dev lint format check test clean play

# Install dependencies
install:
	uv sync

# Install with dev dependencies
dev:
	uv sync --group dev

# Lint code
lint:
	uv run ruff check .

# Format code
format:
	uv run ruff format .
	uv run ruff check --fix .

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

# Clean up cache files
clean:
	rm -rf .ruff_cache .mypy_cache .pytest_cache __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
