# The curriculum knowledge base

This directory holds the durable, machine-readable model of the curriculum: what is to be
learned, where it comes from, what has been checked, and what is still missing. It is the
thing that survives between development sessions. Nothing here is decoration for a website;
the website is expected to be a view over this data.

## Layout

```
knowledge/
  states.json                      status vocabularies and transition rules
  roadmaps/
    perfect-resources.raw.json     generated: extraction of roadmap site 1
    study-timetable.raw.json       generated: extraction of roadmap site 2
    consolidated-inventory.json    generated: the 44 skills merged from both sites
    data-quality-report.json       generated: defects found in the two roadmap sites
  sources/
    mml/                           the supplied Mathematics for Machine Learning material
      manifest.json                generated: identity, files, hashes, provenance limits
      structure.json               generated: parts, chapters, sections, printed pages, anchors
      inventory.json               generated: figures, tables, definitions, theorems, examples, exercises
      defects.json                 generated: extraction-defect register
      pdf-verification.json        generated: independent cross-check against the supplied PDF
      section-concept-map.json     AUTHORED: which concepts and competencies each section supports
  curriculum/
    domains.json                   AUTHORED: ten domains
    competencies.json              AUTHORED: the competency framework (outcomes, prerequisites, source coverage)
    skill-competency-map.json      AUTHORED: roadmap skills mapped to competencies, plus audit findings
    coverage.json                  generated: computed coverage per competency
    coverage-report.md             generated: the human-readable version of the same
  lessons/
    registry.json                  AUTHORED: lessons, exercises and verification records (currently empty, by design)
  references/
    external.json                  AUTHORED: external sources consulted, with retrieval status
```

Generated files carry `generated_by` and must not be hand-edited: re-run the tool instead.
Authored files carry `version` and `last_reviewed`.

## Identifiers

| Prefix | Meaning | Example |
| --- | --- | --- |
| `SK-NN` | A roadmap skill, numbered as the user's own two roadmap sites number it | `SK-03` Linear Algebra |
| `MATH-…`, `PROD-…` | A competency, `DOMAIN-NN`, stable for life | `MATH-07` Matrix decompositions |
| `SRC-MML` | A registered source work | the supplied MML draft |
| `REF-NNN` | An external reference consulted during research | `REF-002` |
| `AF-NN` | An audit finding about the roadmaps | `AF-02` |
| `LSN-…`, `EX-…`, `VER-…` | Lesson, exercise and verification-record identifiers, defined in `lessons/registry.json` | not yet used |

A competency keeps its identifier even if its title is refined, so that links from lessons,
exercises and reports do not break.

## Statuses

Two vocabularies are defined in `states.json` and they are deliberately separate:

- **`topic_status`** describes how far a *curriculum item* has been developed
  (`identified` → `awaiting_sources` → `sources_indexed` → `content_in_development` →
  `in_verification` → `verified` → `published`, with `needs_revision` and `stale` as
  corrective states).
- **`publication_state`** describes what a *content artefact* may claim to a learner
  (`draft`, `in_verification`, `needs_revision`, `verified`, `published`, `stale`).

A curriculum item can be `sources_indexed` while every lesson under it is still `draft`.
The interface must read the artefact's own field before showing a verification badge.

## Which direction is authoritative

Two relationships are recorded in both directions, deliberately, because each view is used
for a different task:

| Relationship | View A (used when writing lessons) | View B (used for planning) |
| --- | --- | --- |
| source → curriculum | `sources/mml/section-concept-map.json` | `competencies[].source_coverage.mml_sections` |
| roadmap → curriculum | `skill-competency-map.json [].competencies` | `competencies[].roadmap_skills` |

`tools/build_coverage.py` reports any disagreement between the two views, and
`tools/sync_mappings.py` resolves disagreements by union, printing every change it makes.

## Routine work

Adding a new book:

1. Put the files under a directory of your choosing. Do not rename or edit the originals.
2. Write a `tools/index_<work>.py` in the style of `tools/index_mml.py` that hashes the files
   and produces a `manifest.json` and a `structure.json` under `knowledge/sources/<work>/`.
3. Author `section-concept-map.json` for the new work: one entry per section with concepts and
   competency links. Add its section identifiers to the affected competencies.
4. Run `tools/sync_mappings.py`, then `tools/build_coverage.py`, then
   `tools/validate_knowledge.py`. The coverage report will now show which competencies gained
   a source and which skills can move from `awaiting_sources` to `sources_indexed`.

Changing a source file:

1. `tools/validate_knowledge.py` fails with `manifest-hash-matches`, naming the changed file.
2. Re-run the indexer. Diff `structure.json` to see which sections moved or disappeared.
3. Re-review the section map entries for the affected sections, and mark every lesson that
   cites them `stale`. Re-verify those lessons; do not regenerate the curriculum.

Adding a lesson:

1. Add an entry to `lessons/registry.json` with `publication_state: "draft"`.
2. Every outcome the lesson assesses must be copied from the competency's `outcomes` list, so
   that "which lesson assesses this outcome" stays answerable.
3. Each exercise declares its `origin` (`source`, `adapted`, `original`, `external`). Source
   exercises keep the book's own numbering and page; they are never renumbered.
4. When the checks have been run, add a verification record naming each check, its method and
   its result, including the checks that could **not** be run.
5. Only then may `publication_state` become `verified`.

## What this knowledge base does not claim

- It does not claim that any lesson exists. `lessons/registry.json` is empty and the coverage
  report says so on every line.
- It does not claim that a book's coverage of a topic equals teaching at the target level. The
  `source_coverage.status` field distinguishes `sourced` from `partial`, and the gap list in
  `coverage-report.md` names every competency without a source.
- It does not treat the supplied book as complete for the skills in the roadmaps around it.
  52 of 82 competencies currently have no source at all.
