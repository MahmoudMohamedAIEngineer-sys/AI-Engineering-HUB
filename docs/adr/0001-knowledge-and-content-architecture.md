# ADR 0001 — Knowledge base as the source of truth, with content rendered from a typed model

**Status:** accepted (phase 2) · **Date:** 2026-10-12

## Context

The platform must eventually present a curriculum whose claims are traceable: which source
supports this lesson, which competencies a lesson covers, which exercises assess an outcome,
which lessons become stale when a source changes. At the same time, lessons must read like
prose written by a teacher, not like database rows.

Two naive designs were rejected. The first is "content as markdown files and nothing else":
prose is pleasant but the traceability relationships live in the author's head, and there is no
mechanical way to answer the questions above. The second is "everything in a database with an
admin editor": traceable, but it makes the content unreviewable in the medium where diffs,
reviews and version history actually work, and it moves the project's knowledge out of the
repository into an opaque store — exactly the failure mode this project is meant to avoid.

## Decision

1. **The knowledge base under `knowledge/` is the source of truth for structure**: sources,
   sections, domains, competencies, prerequisites, skill mappings, references, and the registry
   of lessons, exercises and verification records. It is plain JSON in git, diffable and
   reviewable.
2. **Structural data is authored in one direction and computed in the other.** A relationship
   recorded twice (source→competency and competency→source) is checked mechanically by
   `tools/build_coverage.py` and reconciled by `tools/sync_mappings.py`, which prints every change
   it makes. Where the two disagree, a report is produced; nothing is silently overwritten.
3. **Generated files are marked as generated** and carry the hashes of their inputs, so that a
   stale artefact can be detected instead of trusted.
4. **Lesson prose will be authored content with typed front matter** (schema defined in
   `knowledge/lessons/registry.json`), and the application will render lessons from that model.
   Prose lives in files a human reviews; the relationships it asserts live in data the validator
   can check.
5. **Verification state is data, not decoration.** `publication_state` on an artefact plus a
   verification record with per-check results determines what any interface may claim. The rule is
   enforced by `tools/validate_knowledge.py`: a lesson marked verified without a record, or with
   core checks unrun, fails validation.

## Consequences

- Adding a book is a bounded, scripted operation with a review step, not a rewrite.
- The repository cannot be "finished": the coverage report prints the gaps every time it runs,
  including when the gap is that no lesson exists at all.
- Two vocabularies of state must be maintained (`topic_status` for curriculum items,
  `publication_state` for artefacts). This is deliberate: conflating them is how drafts acquire
  the appearance of verified courses.
- Authoring requires running `make check` before claiming anything, which is slower than writing
  a summary and calling it a course.

## Alternatives considered

| Alternative | Why not |
| --- | --- |
| Markdown-only, no model | Cannot answer "which lessons depend on this source" without manual bookkeeping; the first change to a book would invalidate that bookkeeping silently |
| Database-first with admin editor | Loses reviewable diffs and git history for the intellectual content; adds an auth/authorisation surface that has no purpose until there is more than one author |
| Embed curriculum data inside the front-end code | Makes the curriculum unreadable outside the app and couples content to a framework version; the extraction and verification tooling would have to import from the UI |
| Single directional mapping only | The two directions are used by different tasks (grounding a lesson versus planning coverage); recording one and deriving the other loses the ability to detect drift, which is precisely what caught 21 inconsistencies here |
