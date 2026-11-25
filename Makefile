.PHONY: check test analysis

check: analysis test
	python -m compileall -q scripts tests

analysis:
	python scripts/cost_model.py
	python scripts/savings_model.py

test:
	python -m pytest -q
