# Local checks — the replacement for the removed GitHub Actions workflows.
#
# The venv is created with `uv venv .venv311 && uv pip install --python .venv311/bin/python -e ".[dev,docs]"`.
# `make check` is what CI used to be: lint, types, tests. Run it before pushing.

PY := .venv311/bin

.PHONY: test lint types check docs docs-serve site preview docs-deploy clean build

test:
	$(PY)/python -m pytest -q

lint:
	$(PY)/ruff check src tests

types:
	$(PY)/mypy

check: lint types test

# setuptools reuses a stale build/lib without pruning it: a wheel built here once carried
# six retired v1 modules (guardopt.optimizer, guardopt.router, ...) that no longer exist
# in src/. Build only from a clean tree, and only through `python -m build`.
clean:
	rm -rf build dist src/*.egg-info .site-root

build: clean
	$(PY)/python -m build

docs:
	$(PY)/mkdocs build --strict

docs-serve:
	$(PY)/mkdocs serve

# Assemble the full site into .site-root — the EXACT tree gh-pages serves: web/ is the
# landing page at the root; the MkDocs build lands under /docs (which is why site_url in
# mkdocs.yml carries the /docs/ suffix). Both `preview` and `docs-deploy` build from this,
# so what you preview locally is what ships. Without it, the landing page's `docs/` links
# 404 in local preview because the docs only exist under /docs once assembled here.
site:
	$(PY)/mkdocs build
	rm -rf .site-root
	mkdir .site-root
	cp -R web/. .site-root/
	mkdir .site-root/docs
	cp -R site/. .site-root/docs/

# Serve the assembled site locally so the landing page AND /docs both work, exactly as on
# the live host. Visit http://localhost:8000/ — the Docs links resolve to /docs/.
preview: site
	@echo "→ full site at http://localhost:8000/  (docs at http://localhost:8000/docs/)"
	$(PY)/python -m http.server 8000 --directory .site-root

# Deploys the assembled site to the gh-pages branch. Manual on purpose: the Actions
# workflow that did this on every push to main was removed, so publishing is a deliberate
# act. Note: this pushes to the gh-pages branch, but the site only goes live once GitHub
# Pages is enabled for the repo (Settings → Pages → Source: gh-pages branch).
docs-deploy: site
	$(PY)/ghp-import -n -p -f -m "deploy site (landing + docs)" .site-root
	rm -rf .site-root
