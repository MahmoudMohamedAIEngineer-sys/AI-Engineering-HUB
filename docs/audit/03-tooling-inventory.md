# Tooling inventory

The master brief lists many tools, skills and reference collections. This document records what
is **actually** available in this environment, what was used, what was deliberately not used, and
what is missing. Nothing is claimed to have been used that was not.

## 1. What exists in this environment

| Tool | Present | Used for |
| --- | --- | --- |
| Python 3.11.2 (standard library) | yes | all knowledge-base tooling: extraction, indexing, verification, coverage, validation |
| `python3 -m venv` + pip | yes (with PEP 668 restrictions) | a project virtualenv holding development-only packages |
| pypdf 6.20.0 | installed into `.venv` from PyPI | cross-checking the markdown extraction against the supplied PDF |
| Node.js 22.22.3 / npm 10.9.8 | yes | available for the application phase; not used yet |
| git 2.39.5 | yes | version control of the knowledge base and future application |
| gh 2.23.0 | yes, authenticated | pull-request workflow |
| Outbound network to PyPI, npm and general web | yes | one package install; web research for the competency audit |

## 2. What the brief mentions that is **not** present

Checked rather than assumed. None of the following exist in this repository or environment:

- **Agent skills** (Refactoring UI, Taste, web design guidelines, Awesome Design, Image-to-code,
  REA, Componentry) — no skill directories, no skill manifests, no MCP configuration of any kind
  were found. If these are expected to be provided, they have not been supplied yet.
- **Playwright CLI or any browser-testing tool** — no `playwright`, no browser binaries, no
  `package.json` at all. Browser validation therefore cannot be performed in this phase, and no
  claim of browser testing is made. It becomes relevant in the interface phase, where
  `npx playwright` can be installed on request.
- **Scrapy or any crawler** — not installed, and not needed: the only web inputs are two local
  HTML files and targeted research lookups.
- **React Three Fiber, Lenis, shader or liquid-glass implementations** — nothing is installed and
  nothing was added. Whether they earn a place is a decision for the interface phase, where the
  test is whether a specific educational idea needs them (for example a manipulable linear
  transformation), not whether they look impressive.
- **The supplied reference websites** (DesignSpells, Refero, Details Vault, portfolio and studio
  sites) — not present in the repository. Their URLs were not supplied, so no design reference
  analysis could be performed yet; see "What is needed from you" below.

## 3. Tools written for this project

All under `tools/`, all standard-library Python except where noted, all runnable via `make`.

| Tool | What it does | Reads | Writes |
| --- | --- | --- | --- |
| `tools/htmldom.py` | small ordered-text DOM reader used by the roadmap extractor | — | — |
| `tools/extract_roadmaps.py` | extracts the 44 skills, 132 resources, day counts, schedules and the partial day-by-day plan from the two sites; cross-checks the two sites against each other | the two roadmap HTML files (read-only) | `knowledge/roadmaps/*.json`, including a data-quality report |
| `tools/index_mml.py` | indexes the book: TOC, sections, printed pages, anchors, figures/tables/definitions/theorems/examples, exercise sets, extraction defects | the three markdown files (read-only) | `knowledge/sources/mml/{manifest,structure,inventory,defects}.json` |
| `tools/verify_source_against_pdf.py` | checks the markdown against the original PDF: identity, page offset, TOC agreement, exercise-set inventories, and whether missing items survive in mangled form | `mml-book.pdf` via pypdf | `knowledge/sources/mml/pdf-verification.json` |
| `tools/sync_mappings.py` | reconciles the two directions of the source↔curriculum and roadmap↔curriculum mappings, printing every change | authored knowledge files | the same files, updated |
| `tools/build_coverage.py` | computes coverage per competency from evidence (source sections, roadmap skills, lessons), compares it with the declared status, and reports disagreements | the whole knowledge base | `knowledge/curriculum/coverage.{json,md}` |
| `tools/validate_knowledge.py` | 34 check types / ~1,400 assertions: unique identifiers, referential integrity, prerequisite cycles, status vocabularies, source-hash drift, verification records, coverage freshness | the whole knowledge base | exit code and a report |

The design rule throughout: **a check that cannot run must say so**, never report success by
default. `verify_source_against_pdf.py` writes a "not verified" report if pypdf is absent, and
the validator distinguishes warnings from errors.

## 4. Dependencies added, and why

Only one, deliberately:

- **pypdf 6.20.0** (development only, `.venv`, recorded in `requirements-dev.txt`). Justification:
  the usefulness of everything built on the book depends on knowing precisely what the book
  contains — in particular that chapters 8-12 have no exercise sets and that six items are
  damaged in the extraction. Without the check those would be guesses. It is not a runtime
  dependency of any future application, and authentication, fonts or content of the PDF are
  never needed at serve time.

No other package was installed. The platform's runtime stack is a separate decision (see
`docs/adr/0002-application-stack.md`) and has not been implemented yet.

## 5. What is needed from you before the interface phase

1. **The design reference links** the brief refers to (DesignSpells, Refero, Details Vault, the
   portfolio and studio sites, the Refactoring UI / Taste skills). Without them, design references
   cannot be inspected and a "reference analysis" section would be fabricated.
2. **Confirmation of the application stack** (or a preference), so that the interface phase does
   not rebuild anything.
3. **A decision on browser testing**: whether to install Playwright in this environment so that
   the interface can be validated as a real user would experience it, rather than only by reading
   source.
