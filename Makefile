# AI Engineering Atlas — knowledge-base, content and site tooling.
#
# `make check` is the command that must pass before anything is described as verified.
# The PDF cross-check and the content checks need the development virtualenv (see
# requirements-dev.txt); the extraction, indexing, coverage and validation steps use the
# standard library only.
#
# Dependency order matters and is encoded below:
#   content index (registry, placeholders) -> coverage (reads the registry) -> validate
# Reversing that order produces a staleness warning, which the validator reports rather than hides.

PY  ?= python3
VENV := .venv/bin/python

.PHONY: help setup install dev extract index verify-pdf sync content-index demo-data coverage content-checks validate check site start report clean

help:
	@echo "make setup          create .venv and install development tools"
	@echo "make extract        extract the two roadmap websites into knowledge/roadmaps/"
	@echo "make index          index the supplied book markdown into knowledge/sources/mml/"
	@echo "make verify-pdf     cross-check the markdown extraction against the supplied PDF"
	@echo "make sync           reconcile the two directions of the curriculum mappings"
	@echo "make demo-data      rebuild the data behind the interactive demonstrations"
	@echo "make content-index  build knowledge/lessons/registry.json and the placeholder file"
	@echo "make coverage       recompute the coverage report"
	@echo "make content-checks run the executable checks behind the lessons (needs numpy, sympy)"
	@echo "make validate       run the automated knowledge-base and content checks"
	@echo "make check          run everything that can be automated, in dependency order"
	@echo "make install        install the site's dependencies (npm install)"
	@echo "make dev            run the site on http://localhost:3000 with live reload"
	@echo "make site           type-check and build the Next.js application"
	@echo "make start          serve the production build on http://localhost:3000"
	@echo "make report         print the current curriculum and content state summary"

setup:
	$(PY) -m venv .venv
	$(VENV) -m pip install --quiet --upgrade pip
	$(VENV) -m pip install --quiet -r requirements-dev.txt

extract:
	$(PY) tools/extract_roadmaps.py

index:
	$(PY) tools/index_mml.py

verify-pdf:
	@if [ -x $(VENV) ]; then $(VENV) tools/verify_source_against_pdf.py 2>/dev/null; \
	else echo "no .venv found: run 'make setup' first. The check was NOT performed."; exit 1; fi

sync:
	$(PY) tools/sync_mappings.py

demo-data:
	@if [ -x $(VENV) ]; then $(VENV) tools/build_demo_data.py; \
	else echo "no .venv found: run 'make setup' first. The data was NOT rebuilt."; exit 1; fi

content-index:
	@if [ -x $(VENV) ]; then $(VENV) tools/build_content_index.py; \
	else echo "no .venv found: run 'make setup' first (PyYAML is needed to read lesson front matter)."; exit 1; fi

coverage:
	$(PY) tools/build_coverage.py

content-checks:
	@if [ -x $(VENV) ]; then $(VENV) tools/run_content_checks.py; \
	else echo "no .venv found: run 'make setup' first. No check was run."; exit 1; fi

validate:
	$(PY) tools/validate_knowledge.py

check: extract index sync demo-data content-index coverage content-checks validate
	@echo "Automated checks finished. The PDF cross-check is separate: make verify-pdf"
	@echo "Now run \`make dev\` (or \`make site\`) to see it; the site reads these files at build time."
	@echo "Passing these checks does not mean any content is verified: lessons carry their own"
	@echo "verification records, and a lesson with an unrun core check is shown as needing revision."

install:
	npm install

# The interface reads knowledge/ and content/ at build time, so run `make check`
# first if those have changed. Next.js will otherwise serve the previous snapshot.
dev:
	npm run dev

# WARNING: `next dev` and `next build` both write .next/. Running the build while the dev
# server is up corrupts the dev server's client manifest, and every page then fails with
# "Could not find the module ... in the React Client Manifest" until .next is deleted and
# the dev server restarted. Stop the dev server before building.
site:
	npm run typecheck
	npm run build

# Serves what `make site` produced. Requires `make site` to have run.
start:
	npm run start

report:
	@$(PY) -c "import json;\
c=json.load(open('knowledge/curriculum/coverage.json'))['summary'];\
r=json.load(open('knowledge/lessons/registry.json'));\
print(json.dumps(c['totals'],indent=2));\
print('competency coverage:',json.dumps(c['competency_coverage_counts'],indent=2));\
print('content state:',json.dumps(r['current_state'],indent=2));\
print('open integrity issues:',len(json.load(open('knowledge/curriculum/coverage.json'))['issues']))"

clean:
	@echo "Nothing to clean: generated knowledge and content files are committed on purpose, so"
	@echo "that a reader can see the state of the curriculum and the lessons without running code."
