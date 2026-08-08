# Local checks — the replacement for the removed GitHub Actions workflows.
#
# The venv is created with `uv venv .venv311 && uv pip install --python .venv311/bin/python -e ".[dev,docs]"`.
# `make check` is what CI used to be: lint, types, tests. Run it before pushing.

PY := .venv311/bin

.PHONY: test lint types check docs docs-serve docs-deploy

test:
	$(PY)/python -m pytest -q

lint:
	$(PY)/ruff check src tests

types:
	$(PY)/mypy

check: lint types test

docs:
	$(PY)/mkdocs build --strict

docs-serve:
	$(PY)/mkdocs serve

# Deploys the MkDocs site to the gh-pages branch. Manual on purpose: the Actions
# workflow that did this on every push to main was removed, so publishing the site is
# now a deliberate act rather than a side effect.
docs-deploy:
	$(PY)/mkdocs gh-deploy --force
