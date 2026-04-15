.PHONY: setup test lint clean

setup:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/ && mypy src/

clean:
	rm -rf .pytest_cache __pycache__ src/pet_eval/__pycache__ \
		src/pet_eval/**/__pycache__ tests/__pycache__ tests/**/__pycache__ \
		*.egg-info src/*.egg-info .mypy_cache .ruff_cache \
		outputs/ wandb/ benchmark_cache/
