# AI Engineering Atlas

A source-grounded, interactive curriculum for becoming a professional AI engineer, built from the
books the learner supplies and the two roadmaps the learner already wrote.

The product is the knowledge, not the website: everything here exists to make a rigorously verified
body of teaching material navigable, practiceable and honest about what has and has not been
checked.

## Current state (honest summary)

| Layer | State |
| --- | --- |
| Repository audit | complete — `docs/audit/` |
| Source discovery and indexing | complete for the one supplied book (Mathematics for Machine Learning, 2024-01-15 draft): 12 chapters, 81 sections located, printed-page and file/line anchors, object inventory, defect register, independent PDF cross-check |
| Curriculum model | 10 domains, 82 competencies with observable outcomes and justified prerequisites, 44 roadmap skills mapped, 10 audit findings |
| Coverage | 21 competencies fully sourced, 8 partly sourced, 52 with **no source yet** |
| Learning content | 3 lessons, 20 exercises, 24 executable checks, 3 verification records — **0 lessons verified**, because source fidelity needs a human reader and the records say so |
| Waiting for material | 59 generated placeholder cards stating scope, prerequisites, roadmap position and the source each topic needs (`content/curriculum/placeholders.json`) |
| Application | Next.js App Router over the knowledge base, 92 pre-rendered pages; browser verification not yet run (`docs/plan/implementation-plan.md`, Phase 4) |

`make check` re-derives everything that can be automated and validates it. `make site` type-checks
and builds the application.

## Repository layout

```
Mathematics for ML/                 the supplied book markdown (untouched; canonical source)
mml-book.pdf                        the supplied original PDF (untouched)
perfect_resources_44_skills.html    roadmap site 1 (untouched)
study_timetable_8hrs.html           roadmap site 2 (untouched)

knowledge/                          the curriculum knowledge base — see knowledge/README.md
  states.json                       status vocabularies and rules
  roadmaps/                         generated extraction of the two roadmap sites
  sources/mml/                      generated source index + authored section-to-competency map
  curriculum/                       authored competencies/domains/skill map + generated coverage
  lessons/registry.json             generated index of lessons, exercises, records, placeholders
  lessons/checks-report.json        generated record of every executable check and its output
  references/external.json          external sources consulted, with retrieval status

content/                            the authored teaching material
  lessons/*.md                      lessons with YAML front matter (the schema is in the registry)
  solutions/*.md                    solutions, with the verification each answer passed
  verification/*.json               one verification record per lesson
  checks/*.py                       the executable checks those records cite as evidence
  demos/                            data behind the interactive demonstrations
  curriculum/placeholders.json      generated cards for every topic with no source yet

app/ components/ lib/               the Next.js application (ADR 0002)
public/                             verbatim copies of the two roadmap pages, served unmodified

tools/                              extraction, indexing, verification and validation scripts
docs/
  audit/                            phase 1 deliverables: repository, source and tooling reports
  plan/implementation-plan.md       the seven phases and their acceptance criteria
  notation.md                       where this platform's notation differs from the book's
  adr/                              decisions with the alternatives that were considered
```

## Commands

```bash
make setup          # create .venv and install development tooling (pypdf, PyYAML, numpy, sympy)
make extract        # rebuild knowledge/roadmaps/ from the two roadmap HTML files
make index          # rebuild knowledge/sources/mml/ from the book markdown
make verify-pdf     # cross-check the extraction against the original PDF (needs .venv)
make demo-data      # rebuild the data behind the interactive demonstrations
make content-index  # rebuild the lesson registry and the placeholder cards
make coverage       # recompute the coverage report
make content-checks # run the executable checks behind the lessons
make check          # everything above, in dependency order
make site           # type-check and build the application
make report         # print the current curriculum and content state
```

## Principles this repository holds itself to

1. **Education first.** A topic without a source is recorded as a gap, never filled with
   plausible-sounding generated text. 59 placeholder cards exist for exactly this reason, and they
   contain no teaching content.
2. **Provenance over polish.** Every claim in a lesson is traceable to a section and printed page of
   a supplied book, or is marked as established theory, or is marked as not established.
3. **Verification is evidence, not assertion.** A lesson's displayed state is derived from its
   verification record. A record with an unrun core check cannot produce a "verified" badge, and the
   interface counts zero verified lessons accordingly.
4. **The learner's files are never modified.** The books, the PDF, and both roadmap HTML files are
   inputs. The served roadmap pages are byte-identical copies, checked on every validation run.
5. **Defects are recorded, not repaired.** The extraction registers for the roadmap pages and for the
   source index keep what was actually found, including what could not be located.
6. **No template-shaped teaching.** The three lessons share the metadata and evidence machinery and
   nothing else; one is derivation-led, one proof-led, one measurement-led.

## What this repository does not claim

- That the curriculum is complete. It is the learner's roadmap made machine-readable, with its
  defects listed rather than smoothed over.
- That any lesson is verified. Three lessons exist; all three are marked as needing revision, and
  each record lists the specific checks that were not performed.
- That the interactive material has been tested in a browser. It has been type-checked and built,
  which is not the same thing.
