# Implementation plan

Seven phases. Each phase states its objective, its artefacts and the criteria that decide
whether it is done. A phase is not "in progress" if its acceptance criteria are unmet; it is
either done or it is not.

Status at the end of this working session is recorded per phase.

---

## Phase 1 — Repository audit

**Objective.** Establish what exists, what the source material is, what tooling is available, and
what the outward risks are, before changing anything.

**Acceptance criteria**

- [x] Every file in the repository inspected and described (`docs/audit/01-repository-audit.md`).
- [x] Both existing roadmap websites located, read and their structure documented, without
      modifying them.
- [x] The book identified from its own title page, author line, draft footer, table of contents
      and PDF metadata rather than from its filename (`docs/audit/02-source-discovery.md`).
- [x] Tooling classified as present, absent or not yet needed (`docs/audit/03-tooling-inventory.md`).
- [x] Outward risks recorded with their consequences (R1-R9 in the audit).
- [x] Nothing destructive done: no file moved, renamed, rewritten or deleted.

**Status: complete.**

---

## Phase 2 — Source discovery and curriculum design

**Objective.** Turn the supplied material and the two roadmaps into a machine-readable curriculum
model with traceable provenance and visible gaps.

**Acceptance criteria**

- [x] Roadmap data extracted by a script, reproducibly, including a report of the defects found
      inside the roadmap pages themselves.
- [x] Source indexed to section level with printed pages, file/line anchors, object inventory and
      an extraction-defect register.
- [x] Independent verification of source claims against the original PDF (page offset, TOC
      agreement, exercise inventory, and what "absent" means).
- [x] A competency framework with observable outcomes and justified prerequisites
      (`knowledge/curriculum/competencies.json`, 82 competencies across 10 domains).
- [x] A documented comparison of the roadmaps against that framework, with findings classified as
      essential / recommended / role-specific / optional and each carrying its evidence
      (`skill-competency-map.json` → `audit_findings`, AF-01 to AF-10).
- [x] Section-level mapping from source to competencies, in both directions, with the two
      directions reconciled mechanically rather than by hand.
- [x] Generated coverage report separating what is sourced, what is partially sourced and what has
      no source, and naming the lessons that do not exist.
- [x] Automated validation of the whole knowledge base: 34 check types, ~1,400 assertions, no
      errors, one warning (five competencies are audit additions rather than roadmap topics, which
      is intended and listed).
- [x] Explicit statement that no lesson exists yet, so nothing could be falsely reported as taught.

**Status: complete for the supplied source (MML).** The same machinery is ready for the next book;
what it needs is the material.

---

## Phase 3 — Content-engine prototype (the pilot module)

**Objective.** Demonstrate the teaching standard on one bounded set of topics, with everything a
learner needs: objectives, prerequisites, a real explanation, derivations, worked examples,
implementation, practice, solutions and a verification record.

**Proposed scope** (to be confirmed): matrix decompositions and the low-rank approximation —
MML 4.1-4.6 — because it has 12 source exercise items, both derivation and algorithm content, a
genuine geometric payoff (the SVD as rotation-scaling-rotation), and a direct downstream use in
PCA and compression. Alternative: probability and the Gaussian (MML 6.1-6.5).

**Acceptance criteria**

- [ ] Every lesson objective traced to a specific section and printed page of the source.
- [ ] At least one full derivation written out step by step, with the steps checked independently
      (numerically or symbolically) and the check recorded.
- [ ] At least one implementation, executed and tested in this environment, with its output
      captured.
- [ ] Exercises with declared origin: source items keep their numbering and page; authored items
      are labelled `original`.
- [ ] Every solution verified by a method named in the verification record, including which checks
      could not be run.
- [ ] A verification record per lesson naming the checks, the methods and the results.
- [ ] A written statement of what the lesson does not cover and where the source is silent.

**Not before:** the pilot must not become a template that every later lesson is forced into.

---

## Phase 4 — Educational interface

**Objective.** Build the real interface over the content model: curriculum map, source library,
lesson rendering with mathematics, exercises and feedback, prerequisite navigation, progress.

**Acceptance criteria**

- [ ] The interface renders lesson, curriculum, source and reference data from the knowledge base;
      no hardcoded course content.
- [ ] Verification badges derive from `publication_state`; a lesson with no verification record
      renders as unverified, and the platform says so.
- [ ] Mathematics renders correctly (display and inline), including the source's notation, with a
      documented convention where the source's conventions collide.
- [ ] Keyboard navigation, visible focus, contrast, reduced-motion support and responsive layout
      checked on desktop and mobile widths.
- [ ] Loading, empty and error states exist for every data-driven view — including the empty
      state that this repository is actually in today.
- [ ] The two original roadmap pages remain reachable and unmodified.

---

## Phase 5 — Verification and quality controls

**Objective.** Make correctness checkable, and make unverified material impossible to mistake for
verified material.

**Acceptance criteria**

- [ ] Automated: knowledge-base validation, source-hash drift detection, coverage freshness,
      internal link checks, content-schema validation of lessons, and execution of every code
      example in the lessons.
- [ ] Automated where possible: mathematical checks via independent numeric evaluation of the
      derivations' results.
- [ ] Browser checks through Playwright once it is installed; until then, no browser-testing claim
      is made anywhere.
- [ ] Manual: source fidelity and pedagogical review, recorded per lesson rather than asserted
      globally.
- [ ] A published list of what is automated, what is manual, and what remains unchecked.

---

## Phase 6 — Expansion

**Objective.** Add the remaining books and domains as material arrives, reusing the pipeline
without flattening every subject into one lesson shape.

**Acceptance criteria**

- [ ] Each new source indexed, mapped and reconciled with the existing framework.
- [ ] Competencies without a source stay visibly unsourced until one arrives.
- [ ] Existing verified lessons are never regenerated wholesale; only affected ones become `stale`
      and are re-verified.
- [ ] Interview preparation and project work added with rubrics, once the foundational path is
      reliable.

---

## Phase 7 — Final validation

**Objective.** Confirm end to end that a learner can actually learn here, and that a maintainer can
add a source without losing verified work.

**Acceptance criteria**

- [ ] A learner can: discover a topic, follow prerequisites, read the explanation, run an
      interactive demonstration, do exercises, receive feedback that identifies the misconception,
      and reach the next related competency.
- [ ] A maintainer can add a source, re-run the pipeline and see exactly which lessons became
      stale and why.
- [ ] The completion report quotes the actual test results and lists what is still unresolved.

---

## Standing rules for every phase

1. Nothing is marked verified without a record of what was checked.
2. Nothing is generated for a topic that has no source, other than an honest statement of the gap.
3. Claims that depend on dates or versions carry both.
4. Source files are never modified; generated artefacts say which source and which version produced
   them.
5. When a design decision is consequential, the alternatives considered are written down (see
   `docs/adr/`).
