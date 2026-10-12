# AI Engineering Atlas — knowledge-base and content tooling.
#
# `make check` is the command that must pass before anything is described as verified.
# The PDF cross-check needs the development virtualenv (see requirements-dev.txt);
# everything else is standard-library Python.

PY  ?= python3
VENV := .venv/bin/python

.PHONY: help setup extract index verify-pdf coverage sync validate check report clean

help:
	@echo "make setup        create .venv and install development tools (pypdf)"
	@echo "make extract      extract the two roadmap websites into knowledge/roadmaps/"
	@echo "make index        index the supplied book markdown into knowledge/sources/mml/"
	@echo "make verify-pdf   cross-check the markdown extraction against the supplied PDF"
	@echo "make sync         reconcile the two directions of the curriculum mappings"
	@echo "make coverage     recompute the coverage report"
	@echo "make validate     run the automated knowledge-base checks"
	@echo "make check        run everything that can be automated, in dependency order"
	@echo "make report       print the current curriculum state summary"

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

coverage:
	$(PY) tools/build_coverage.py

validate:
	$(PY) tools/validate_knowledge.py

check: extract index sync coverage validate
	@echo "Automated checks finished. The PDF cross-check is separate: make verify-pdf"
	@echo "Note: passing these checks does not mean any content is verified. No lesson exists yet."

report:
	@$(PY) -c "import json;d=json.load(open('knowledge/curriculum/coverage.json'));s=d['summary'];print(json.dumps(s['totals'],indent=2));print('coverage states:',json.dumps(s['competency_coverage_counts'],indent=2));print('open integrity issues:',len(d['issues']))"

clean:
	@echo "Nothing to clean: generated knowledge files are committed on purpose, so that a"
	@echo "reader can see the state of the curriculum without running any code."
