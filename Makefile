install:
	pip install -e ".[dev,docs]"

test:
	pytest us_imputation_benchmarking/tests/ --cov=us_imputation_benchmarking --cov-report=xml --maxfail=0

check-format:
	linecheck .
	isort --check-only --profile black us_imputation_benchmarking/
	black . -l 79 --check

format:
	linecheck . --fix
	isort --profile black us_imputation_benchmarking/
	black . -l 79

documentation:
	cd docs && jupyter-book build .

build:
	pip install build
	python -m build

clean:
	rm -rf dist/ build/ *.egg-info/
	rm -rf docs/_build/
