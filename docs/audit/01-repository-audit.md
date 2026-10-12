# Repository audit

Phase 1 deliverable. Written after inspecting every file in the repository; nothing in this
document is assumed. Where a check was performed, the result is stated; where something could
not be checked, it says so.

## 1. What the repository contains

At the start of this work the repository held five files and no application code:

| Path | Size | What it is |
| --- | --- | --- |
| `Mathematics for ML/Mathematics for ML part 1.md` | 375 KB, 4,970 lines | Markdown extracted from the supplied MML PDF; covers the front matter, the table of contents, and chapters 1 through section 4.6 |
| `Mathematics for ML/Mathematics for ML part 2.md` | 379 KB, 5,126 lines | Sections 4.7 through 8.4 (starts mid-chapter 4, ends mid-chapter 8) |
| `Mathematics for ML/Mathematics for ML part 3.md` | 411 KB, 5,694 lines | Section 8.4.1 through chapter 12, the references and the index |
| `mml-book.pdf` | 17.6 MB, 417 pages | The original book PDF. Metadata: title *Mathematics for Machine Learning*, authors Deisenroth, Faisal and Ong, produced with pdfTeX, created 2024-01-15 |
| `perfect_resources_44_skills.html` | 165 KB, 2,776 lines | First roadmap website (see section 3) |
| `study_timetable_8hrs.html` | 139 KB, 2,204 lines | Second roadmap website |

There is no `package.json`, no build system, no test suite, no CI configuration, no README, no
licence file and no existing application to preserve or extend. Git history is a single commit
(`9222896 Add files via upload`), so there is no earlier architecture to respect and no previous
work that a new implementation could accidentally duplicate.

The two roadmap HTML files are hand-written, self-contained pages (inline `<style>`, one
`<script>`, no external JavaScript dependencies, web fonts from Google Fonts). They are
first-party work by the user and are treated as source material to be preserved untouched.

## 2. Environment

| Tool | Version found | Notes |
| --- | --- | --- |
| Python | 3.11.2 | System interpreter is PEP 668 managed (externally managed environment), so `pip install` outside a virtualenv is refused |
| Node.js | 22.22.3 | Supports the current Next.js/React toolchain |
| npm | 10.9.8 | |
| git | 2.39.5 | |
| gh | 2.23.0 | GitHub CLI available; authentication is configured for the session |

Because the system Python refuses package installs, a project virtualenv (`.venv`, git-ignored)
holds development-only tooling. `requirements-dev.txt` records it; `make setup` recreates it.

## 3. The two existing roadmap websites

Both were located and read. They are not interchangeable: each carries a different half of the
user's plan, and they share the same 44-skill numbering, which is what makes a stable join
possible.

**`perfect_resources_44_skills.html` — "The Perfect Resource".** The resource-selection layer.
44 skill cards (`id="s1"` … `id="s44"`), each with a phase tag, a "what you learn" summary, and
exactly three resources: one book, one free course or series, one paid course. Each resource
carries a platform, title, author/year and a stated rationale. Counts: 44 skills × 3 = 132
resources, matching the page's own claim.

**`study_timetable_8hrs.html` — "Master Study Timetable".** The scheduling layer. A sidebar with
all 44 skills, each with a study-day count, grouped into five phases; a hero row claiming 339
study days, 2,712 hours, 57 weeks and about 14 months; a standard daily schedule; and, for part
of the curriculum, day-by-day blocks.

Findings that matter for everything built next:

- **The day-level plan is incomplete.** Detailed day blocks exist for skills 01-16 plus two
  project sprints (116 dated day entries). The other 28 skills exist only as sidebar entries
  with day counts. A learner interface must not imply that a day plan exists where it does not.
- **The roadmaps' own numbers disagree with their content.** The timetable header states 2,720
  total hours while its stat row states 2,712; the resource index filter legend states
  "SWE (10), Data Eng (10), AI/ML (10), Specialization (6)" while the page contains 8, 8, 10,
  10 and 8 skill cards; SK-01 declares 14 study days while its plan numbers 13. The full list is
  generated into `knowledge/roadmaps/data-quality-report.json`. These are recorded, not
  silently corrected: the page content is treated as canonical and the interface will derive
  counts from data rather than from hand-written text.
- **Skill names differ between the two pages** for 24 of the 44 skills (for example
  "Data Warehouses (Snowflake / BigQuery / Redshift)" versus "Data Warehouses"). The longer
  form from the resource index is preserved as the canonical name, and both are kept in the
  extracted data.
- Day counts sum to exactly 339 study days including the two sprints, which reconciles with the
  hero figure and confirms the extraction is complete.

The extraction is reproducible: `make extract` rewrites `knowledge/roadmaps/*.json` and the
data-quality report from the two HTML files, which are left untouched.

## 4. Risks and constraints identified at this stage

| # | Risk | Consequence if ignored | Response in this phase |
| --- | --- | --- | --- |
| R1 | Only one of ten curriculum domains has any source material | Building lessons for the other domains would mean inventing content or presenting generic summaries as a course | Competencies without sources are recorded as gaps, and the coverage report prints them; nothing in those domains is marked taught |
| R2 | The supplied book's figures are referenced as external CDN URLs (`cdn-mineru.openxlab.org.cn`), and no image file exists in the repository | Lessons that assume the original figures would break, and copying the book's figures raises copyright issues | The figure inventory records 243 references with zero local assets. Lessons will use original diagrams generated for this platform; the book's figures are cited by number and page rather than reproduced |
| R3 | The book's exercise sets have no solutions in this draft | Presenting its exercises as self-checkable would be false, or would require unverified answers | The 69 located items are indexed with their identifiers and pages; any use requires our own solutions, verified independently and labelled as ours |
| R4 | Chapters 8-12 contain no exercise sets at all (verified against the PDF) | Practice coverage would silently stop after chapter 7 | Recorded as a defect with the verification evidence attached; practice for those chapters will be authored and labelled `original` |
| R5 | Six exercise items are mangled in the extraction (2.7, 5.7, 6.3, 6.4, 7.9, 7.10) | An exercise could be quoted incorrectly or an item missed | Recorded with per-item evidence about whether the wording survives; each must be checked against the PDF page before use |
| R6 | The copyright position of the book (free to read, not free to redistribute) | A public site that reproduces the book's text or figures would be a legal problem and would also teach worse | Lessons are authored explanations that cite the book; the platform does not host the book's text or figures |
| R7 | The roadmap's pace (8 hours/day for ~14 months) is presented with day counts for skills whose target level is `deep` | A learner could believe a deep competency was completed in three days | Depth and effort are modelled separately; the audit raises this explicitly (finding AF-10) rather than redefining the target level to fit the calendar |
| R8 | Fast-moving domains (LLM tooling, lakehouse formats, cloud services) | Content silently becomes wrong | Those competencies are marked version-sensitive; references must carry retrieval dates and the version described |
| R9 | No application, tests or CI exist yet | Any claim about the platform would be unverifiable | Phase 1 makes no platform claims; the tooling that does exist is runnable and its output is committed |

## 5. What was deliberately not done

- No book file was rewritten, moved or renamed. The originals are the canonical artefacts.
- Neither roadmap site was modified. The extraction reads them; it does not write to them.
- No dependency was added to the runtime of a platform that does not exist yet. `pypdf` is
  development-only and confined to `.venv`.
- No lesson was written. Writing lessons before the curriculum model and the verification
  machinery existed would have produced exactly the kind of unverifiable content this project
  is meant to avoid.

## 6. Open questions for the user

These change architecture or legal exposure and are therefore asked rather than assumed
(see the accompanying message for the same list):

1. **Visibility.** Is the platform personal, or will it be published publicly? It changes how
   conservative the citation and figure policy must be.
2. **First module.** Which topic should be the pilot that demonstrates the teaching standard?
   The strongest candidates given the supplied source are matrix decompositions and the SVD
   (MML 4.1-4.6, with 12 exercise items and two chapters of downstream use) or probability and
   the Gaussian (MML 6.1-6.5, with 11 items).
3. **Roadmap sites.** Confirmed default: keep both files untouched and build a new interface
   that renders the extracted data, linking back to the originals. Confirm if the intent is
   instead to freeze them as-is and never re-present their content.
