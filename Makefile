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

# Deploys the site to the gh-pages branch. Manual on purpose: the Actions workflow
# that did this on every push to main was removed, so publishing is a deliberate act.
# Layout: web/ is the landing page at the site root; the MkDocs build lands under
# /docs — which is why site_url in mkdocs.yml carries the /docs/ suffix.
docs-deploy:
	$(PY)/mkdocs build
	rm -rf .site-root
	mkdir .site-root
	cp -R web/. .site-root/
	mkdir .site-root/docs
	cp -R site/. .site-root/docs/
	$(PY)/ghp-import -n -p -f -m "deploy site (landing + docs)" .site-root
	rm -rf .site-root
