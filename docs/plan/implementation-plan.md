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

**Scope, as executed.** Matrix decompositions and the low-rank approximation, MML 4.1-4.6, covering
competencies MATH-06, MATH-07 and MATH-08, delivered as three lessons:

| Lesson | Source | Competency | Exercises |
| --- | --- | --- | --- |
| `LSN-MATH-06-01` determinants, eigenvalues, eigenvectors | MML 4.1-4.2, printed pp.99-113 | MATH-06 | 7 |
| `LSN-MATH-07-01` matrix decompositions and the SVD | MML 4.3-4.5, 4.7, printed pp.114-134 | MATH-07 | 7 |
| `LSN-MATH-08-01` low-rank approximation | MML 4.6, 10.4, printed pp.129-134, 333-335 | MATH-08 | 6 |

**Acceptance criteria**

- [x] Every lesson objective traced to a specific section and printed page of the source
      (`source_anchors` in each lesson's front matter, validated against the source index).
- [x] At least one full derivation written out step by step, with the steps checked independently
      (numerically or symbolically) and the check recorded. All three lessons qualify: the
      trace/determinant invariants and the diagonalisation criterion; the constructive SVD proof
      with its five steps; and the optimality proof of the Eckart-Young theorem by a dimension
      argument, which the supplied source states but does not prove.
- [x] At least one implementation, executed and tested in this environment, with its output
      captured. `content/checks/` holds 24 checks; the measured outputs (including the accuracy
      collapse of the textbook SVD construction at a singular-value ratio of 1e10) are recorded in
      `knowledge/lessons/checks-report.json`.
- [x] Exercises with declared origin: source items keep their numbering and page; authored items
      are labelled `original`; material adapted from a source item records the item, the page and
      what was changed.
- [x] Every solution verified by a method named in the verification record, including which checks
      could not be run. Each lesson's record lists the checks that were *not* performed.
- [x] A verification record per lesson naming the checks, the methods and the results
      (`content/verification/LSN-MATH-06-01.json`, `-07-01`, `-08-01`).
- [x] A written statement of what the lesson does not cover and where the source is silent. Two
      substantive findings came out of this: the supplied text never states the spectral-norm form
      of the Eckart-Young theorem's Frobenius companion (the word "Frobenius" does not occur in it),
      and the SVD is derived from `A'A` without any discussion of computation — so the numerical
      material is recorded as a gap rather than filled from memory.
- [x] A placeholder layer for what cannot be written yet: 59 competencies with no source get a
      generated, visibly incomplete entry stating their scope, prerequisites, position in the
      learner's own roadmap and the source they are waiting for
      (`content/curriculum/placeholders.json`), with no invented content.
- [x] The engine refuses to overstate: a lesson's displayed state is derived from its verification
      record, and the validator fails if a lesson claims a state its record does not support.

**Status: complete.** Three lessons, 20 exercises, 24 executable checks, three verification records.
None of the three lessons is marked verified, and the records say why: source fidelity needs a human
reader with the book open, and two of the lessons reference material the supplied source does not
contain. The home page counts zero verified lessons accordingly.

**Not before:** the pilot must not become a template that every later lesson is forced into. The
three lessons differ in structure deliberately — one is derivation-led, one proof-led, one
measurement-led — while sharing only the metadata and evidence machinery.

---

## Phase 4 — Educational interface

**Objective.** Build the real interface over the content model: curriculum map, source library,
lesson rendering with mathematics, exercises and feedback, prerequisite navigation, progress.

**What exists.** A Next.js App Router application (ADR 0002) that reads `knowledge/` and `content/`
at build time and pre-renders 92 pages: home, curriculum index and one page per competency, lesson
index, lesson pages, source library, roadmap inventory, method. No course content is hardcoded in
the interface; when a file it needs is missing, the loader fails with the name of the file rather
than falling back to invented defaults.

**Acceptance criteria**

- [x] The interface renders lesson, curriculum, source and reference data from the knowledge base;
      no hardcoded course content.
- [x] Verification badges derive from the verification record (stricter than `publication_state`: a
      record whose outcome is `needs_revision` cannot render as verified), and the home page counts
      zero verified lessons because that is what the evidence says.
- [x] Mathematics renders (display and inline) through KaTeX, with the source's notation preserved
      inside the mathematics. The convention where the two systems collide is documented in
      `docs/notation.md` and summarised on the method page.
- [x] Loading, empty and error states exist: the lesson index names every competency with no lesson
      and why, the placeholder cards state what is missing, and missing knowledge files raise a named
      error instead of rendering a blank page.
- [x] The two original roadmap pages are reachable unmodified: they are served verbatim from
      `public/` and the validator fails the build if either served copy stops being byte-identical to
      the file the learner supplied.
- [x] Interactive components are islands loaded on demand, and the only one shipped is the
      rank-truncation demonstration, which shows the size/error trade-off that the text can only
      tabulate.
- [ ] Keyboard navigation, visible focus, contrast, reduced-motion support and responsive layout
      verified in a real browser. The CSS implements all five (skip link, `:focus-visible`, relative
      colour, `prefers-reduced-motion`, a mobile breakpoint), and a type check and a production build
      pass — but **no browser test has been run**, so this criterion is not met and no claim is made
      that it is.

**Status: implemented, with the browser-verification criterion explicitly unmet.** The remaining work
is Phase 5's job: browser checks through Playwright, which is not installed in this environment.

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
