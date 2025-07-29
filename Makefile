# Makefile for continual-tiny-transformer development
# Usage: make <target>
# Example: make install, make test, make lint

.PHONY: help install install-dev clean test test-unit test-integration test-gpu lint format type-check security docs build release

# Default target
help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation targets
install: ## Install production dependencies
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e ".[dev,test,docs]"
	pre-commit install

install-full: ## Install all dependencies including benchmarking tools
	pip install -e ".[dev,test,docs,benchmark]"
	pre-commit install

# Environment setup
setup: ## Initial setup for development
	python -m venv venv
	. venv/bin/activate && pip install --upgrade pip setuptools wheel
	. venv/bin/activate && make install-dev
	@echo "Environment setup complete. Activate with: source venv/bin/activate"

# Cleaning targets
clean: ## Clean build artifacts and cache files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .tox/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

clean-data: ## Clean generated data and model files  
	rm -rf data/
	rm -rf models/
	rm -rf checkpoints/
	rm -rf outputs/
	rm -rf wandb/
	rm -rf lightning_logs/

# Testing targets
test: ## Run all tests
	pytest

test-unit: ## Run unit tests only
	pytest tests/unit/

test-integration: ## Run integration tests only
	pytest tests/integration/

test-benchmarks: ## Run benchmark tests
	pytest tests/benchmarks/ --benchmark-only

test-gpu: ## Run GPU tests (requires CUDA)
	pytest -m gpu

test-coverage: ## Run tests with coverage reporting
	pytest --cov=continual_transformer --cov-report=html --cov-report=term

test-tox: ## Run tests across multiple Python versions with tox
	tox

# Code quality targets
lint: ## Run linting checks
	ruff check src/ tests/
	black --check src/ tests/
	isort --check src/ tests/

format: ## Auto-format code
	black src/ tests/
	isort src/ tests/
	ruff check --fix src/ tests/

type-check: ## Run type checking
	mypy src/

security: ## Run security checks
	bandit -r src/ -f json

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

# Documentation targets
docs: ## Build documentation
	cd docs && sphinx-build -b html . _build/html

docs-serve: ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8000

docs-clean: ## Clean documentation build
	rm -rf docs/_build/

# Development targets
dev-server: ## Start development server (if applicable)
	@echo "Starting development environment..."
	# Add your development server command here

notebook: ## Start Jupyter notebook server
	jupyter notebook

# Build and release targets
build: ## Build package for distribution
	python -m build

build-clean: ## Clean build and rebuild package
	make clean
	make build

release-test: ## Upload to test PyPI
	twine upload --repository testpypi dist/*

release: ## Upload to PyPI
	twine upload dist/*

# Docker targets  
docker-build: ## Build Docker image
	docker build -t continual-tiny-transformer .

docker-run: ## Run Docker container
	docker run -it --rm continual-tiny-transformer

docker-dev: ## Run development environment with Docker Compose
	docker-compose up -d

docker-down: ## Stop Docker Compose services
	docker-compose down

# Utility targets
requirements: ## Update requirements files
	pip-compile requirements.in
	pip-compile requirements-dev.in

upgrade-deps: ## Upgrade all dependencies
	pip install --upgrade pip setuptools wheel
	pip install --upgrade -e ".[dev,test,docs]"

check-deps: ## Check for dependency security issues
	safety check
	pip-audit

profile: ## Run profiling on sample code
	python -m cProfile -s cumulative scripts/profile_example.py

benchmark: ## Run performance benchmarks
	python scripts/benchmark.py

# Git hooks
install-hooks: ## Install git hooks
	pre-commit install
	pre-commit install --hook-type commit-msg

# Environment info
info: ## Show environment information
	@echo "Python version:"
	@python --version
	@echo "\nPip version:"
	@pip --version
	@echo "\nInstalled packages:"
	@pip list | head -20
	@echo "\nGit status:"
	@git status --short

# Quick development workflow
dev: ## Quick development setup (install + pre-commit + test)
	make install-dev
	make pre-commit
	make test-unit

# CI simulation
ci: ## Simulate CI pipeline locally
	make lint
	make type-check
	make security
	make test-coverage
	make build

# Performance monitoring
memory-profile: ## Profile memory usage
	mprof run python scripts/memory_example.py
	mprof plot

line-profile: ## Profile line-by-line execution
	kernprof -l -v scripts/line_profile_example.py