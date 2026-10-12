# AI Engineering Atlas

A source-grounded, interactive curriculum for becoming a professional AI engineer, built from
the books the learner supplies and the two roadmaps the learner already wrote.

The product is the knowledge, not the website: everything here exists to make a rigorously
verified body of teaching material navigable, practiceable and honest about what has and has not
been checked.

## Current state (honest summary)

| Layer | State |
| --- | --- |
| Repository audit | complete — `docs/audit/` |
| Source discovery and indexing | complete for the one supplied book (Mathematics for Machine Learning, 2024-01-15 draft): 12 chapters, 81 sections, printed-page and file/line anchors, object inventory, defect register, independent PDF cross-check |
| Curriculum model | 10 domains, 82 competencies with observable outcomes and justified prerequisites, 44 roadmap skills mapped, 10 audit findings |
| Coverage | 23 competencies fully sourced, 7 partially, 52 with **no source yet**; 0 lessons |
| Learning content | **none written yet** — the registry is empty by design, so nothing can be mistaken for a finished course |
| Application | not started; stack proposed in `docs/adr/0002-application-stack.md` |

`make check` re-derives everything that can be automated and validates it.

## Repository layout

```
Mathematics for ML/            the supplied book markdown (untouched; canonical source)
mml-book.pdf                   the supplied original PDF (untouched)
perfect_resources_44_skills.html   roadmap site 1 (untouched)
study_timetable_8hrs.html          roadmap site 2 (untouched)

knowledge/                     the curriculum knowledge base — see knowledge/README.md
  states.json                  status vocabularies and rules
  roadmaps/                    generated extraction of the two roadmap sites
  sources/mml/                 generated source index + authored section-to-competency map
  curriculum/                  authored competencies/domains/skill map + generated coverage
  lessons/registry.json        lessons, exercises, verification records (empty)
  references/external.json     external sources consulted, with retrieval status

tools/                         extraction, indexing, verification and validation scripts
docs/
  audit/                       phase 1 deliverables: repository, source and tooling reports
  plan/implementation-plan.md  the seven phases and their acceptance criteria
  adr/                         decisions with the alternatives that were considered
```

## Commands

```bash
make setup      # create .venv and install development tooling (pypdf)
make extract    # rebuild knowledge/roadmaps/ from the two roadmap HTML files
make index      # rebuild knowledge/sources/mml/ from the book markdown
make verify-pdf # cross-check the extraction against the original PDF (needs .venv)
make check      # extract + index + reconcile + coverage + validate
make report     # print the current curriculum totals
```

## Principles this repository holds itself to

1. **Education first.** A topic without a source is recorded as a gap, never filled with
   plausible-sounding generated text.
2. **Traceability.** Every claim in future lessons will name its source section and printed page;
   every exercise will declare whether it is sourced, adapted, external or original.
3. **Verification is data.** Material is published as verified only with a verification record
   naming the checks that ran, their methods and their results — including the checks that could
   not run.
4. **Nothing is regenerated wholesale.** Changing a source marks the dependent lessons stale; it
   does not rebuild the curriculum.
5. **The originals are canonical.** The supplied books and the two roadmap sites are read, never
   rewritten.
