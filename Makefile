install:
	pip install -e ".[dev,docs]"

test:
	pytest tests/ --cov=microimpute --cov-report=xml --maxfail=0

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
	python docs/add_plotly_to_book.py docs/_build/html

build:
	pip install build
	python -m build

clean:
	rm -rf dist/ build/ *.egg-info/
	rm -rf docs/_build/

paper: main

main: $(wildcard paper/sections/**/*.tex) $(wildcard paper/bibliography/*.bib) paper/main.tex paper/macros.tex
	cd paper && \
	BIBINPUTS=./bibliography pdflatex main && \
	BIBINPUTS=./bibliography bibtex main && \
	pdflatex main && \
	pdflatex main

clean-paper:
	rm -f paper/*.aux paper/*.bbl paper/*.blg paper/*.log paper/*.out paper/*.toc paper/main.pdf paper/sections/**/*.aux
