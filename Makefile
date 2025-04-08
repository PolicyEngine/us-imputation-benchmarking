install:
	pip install -e ".[dev,docs]"

test:
	pytest microimpute/tests/ --cov=microimpute --cov-report=xml --maxfail=0

check-format:
	linecheck .
	isort --check-only --profile black microimpute/
	black . -l 79 --check

format:
	linecheck . --fix
	isort --profile black microimpute/
	black . -l 79

documentation:
	cd docs && jupyter-book build .

build:
	pip install build
	python -m build

clean:
	rm -rf dist/ build/ *.egg-info/
	rm -rf docs/_build/
