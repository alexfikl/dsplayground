PYTHON?=python

all: flake8 pylint mypy

flake8:
	$(PYTHON) -m flake8 dsplayground tests examples
	@echo -e "\e[1;32mflake8 clean!\e[0m"

pylint:
	PYTHONWARNINGS=ignore $(PYTHON) -m pylint dsplayground tests/*.py examples/*.py
	@echo -e "\e[1;32mpylint clean!\e[0m"

mypy:
	$(PYTHON) -m mypy --show-error-codes dsplayground tests examples
	@echo -e "\e[1;32mmypy clean!\e[0m"

mypy-strict:
	$(PYTHON) -m mypy --strict --show-error-codes dsplayground tests
	@echo -e "\e[1;32mmypy (strict) clean!\e[0m"

tags:
	ctags -R

.PHONY: all flake8 pylint mypy mypy-strict
