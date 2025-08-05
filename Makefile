.PHONY: help install fmt lint test ci clean

# Default target
help:
	@echo "Available targets:"
	@echo "  install  - Install package and dev dependencies"
	@echo "  fmt      - Format code with Black and Ruff"
	@echo "  lint     - Run all linting checks"
	@echo "  test     - Run tests with coverage"
	@echo "  ci       - Run full CI pipeline"
	@echo "  clean    - Clean build artifacts and cache"

# Install package and dev dependencies
install:
	pip install -e ".[dev]"
	pre-commit install

# Format code
fmt:
	black .
	ruff format .

# Run linting checks
lint:
	ruff check .
	mypy stock_analyzer/

# Run tests with coverage
test:
	pytest

# Run full CI pipeline (format, lint, test)
ci: fmt lint test

# Clean build artifacts and cache
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete 