# Source discovery report

How the supplied material was identified, what is in it, and what is wrong with it. Every claim
below is produced by a script in `tools/` and written to `knowledge/sources/`, so it can be
re-checked rather than believed.

## 1. Identification — no filename was assumed

The directory is named `Mathematics for ML` and the PDF `mml-book.pdf`, which is suggestive but
not evidence. The work was identified from its own contents:

| Evidence | Where it was found |
| --- | --- |
| Title page: "MATHEMATICS FOR MACHINE LEARNING" | `part 1`, line 1 |
| Author line: "Marc Peter Deisenroth · A. Aldo Faisal · Cheng Soon Ong" | `part 1`, line 3 |
| The book's own table of contents, 111 lines with printed page numbers | `part 1`, lines 5-115 |
| Running footer on every page: `Draft (2024-01-15) of "Mathematics for Machine Learning". Feedback: https://mml-book.com.` | 50 occurrences across the three files |
| Copyright line: "©2024 M. P. Deisenroth, A. A. Faisal, C. S. Ong. Published by Cambridge University Press (2020)" | `part 3`, index region |
| PDF metadata: title, authors, creation date 2024-01-15, producer pdfTeX | `mml-book.pdf` |

The draft footer and the PDF creation date agree, so the markdown is an extraction of exactly
this PDF edition. The three markdown files are consecutive page ranges, not chapters:

| File | Content | First located section | Last content |
| --- | --- | --- | --- |
| `part 1` | front matter, TOC, chapters 1-4.6 | 1.1 (line 210) | the SVD movie-ratings example at the end of section 4.6 |
| `part 2` | sections 4.7 - 8.4 | 4.7 (line 44) | section 8.4, immediately before 8.4.1 |
| `part 3` | sections 8.4.1 - 12.6, references, index | 8.4.1 (line 1) | the index |

`knowledge/sources/mml/structure.json` records, for every one of the 81 table-of-contents
sections, its printed page number, the file it was found in, its line, and whether it was found
at all. All 81 were located.

## 2. Structure found

- 2 parts (Part I Mathematical Foundations p.9-248, Part II Central Machine Learning Problems
  p.249-394), 12 chapters, 81 TOC sections, 122 subsection headings in the body, 290 numbered
  equations, 143 figure references, 121 examples, 67 definitions, 32 theorems, 4 table
  references.
- Chapter titles in the body are sometimes printed without their number ("Analytic Geometry",
  "Probability and Distributions", and four others). Chapter anchors are therefore derived from
  the first located section and that derivation is recorded in the structure file rather than
  left implicit.
- Printed page numbers survive only in the table of contents; the body has no page markers.
  Citations are therefore section-level with printed page, plus a file/line anchor for
  sub-sections. No page numbers are invented.

## 3. Extraction defects (generated into `knowledge/sources/mml/defects.json`)

| Defect | Severity | Evidence |
| --- | --- | --- |
| 243 figure references point at 243 external CDN URLs; zero local image files | warning | the figure URLs are all on one host and no image exists in the repository |
| 35 heading occurrences are duplicated in the body | info | the printed running header is interleaved into the text, so a naive scan counts a section twice |
| Six LaTeX token corruptions (for example `\ b { S }` where a bold macro was intended, `\stackrel{\cdot}` where the book prints `\stackrel{!}`), plus 33 broken roman-text runs inside math | warning | pattern matches over the extracted text |
| 10 tables were emitted as raw HTML `<table>` markup rather than markdown | info | counted directly |
| Three printed page numbers were emitted as level-2 headings ("172", "370", "407") | warning | line-level inspection |
| One ISBN-like digit string ("0123956389") was emitted as a heading | warning | line-level inspection |
| Six exercise identifiers lost their item boundary | warning | see section 5 |

The corruptions matter because a lesson that quotes a formula must not propagate them. Each
occurrence is located by file and line, so a lesson author can check the original.

## 4. Exercise sets — what the source actually provides

This is the part most likely to be got wrong by assumption, so it was verified twice: once by
indexing the markdown, once by searching the supplied PDF.

| Chapter | Printed page | Items in PDF | Items located in markdown | Gap |
| --- | --- | --- | --- | --- |
| 2 | 64 | 20 | 19 | 2.7 |
| 3 | 96 | 10 | 10 | — |
| 4 | 137 | 12 | 12 | — |
| 5 | 170 | 9 | 8 | 5.7 |
| 6 | 222 | 13 | 11 | 6.3, 6.4 |
| 7 | 247 | 11 | 9 | 7.9, 7.10 |
| 8-12 | — | 0 | 0 | no sets in this draft |

The PDF was searched page by page for exercise sections: they occur on printed pages 64, 96, 137,
170, 222 and 247 only, and the printed-page to PDF-page offset of 6 was confirmed consistent
across those six occurrences. **Chapters 8-12 contain no exercise sets in this draft** — this is
a property of the book, not a defect in the extraction, and it is recorded as such.

For the six missing items, the verifier checks whether the wording of the exercise survives
anywhere in the markdown rather than just the identifier:

| Item | Wording found in the extraction? | Interpretation |
| --- | --- | --- |
| 6.3, 6.4 | yes | the item boundary was lost; the content is present in mangled form |
| 7.10 | yes (matches more than one place, so the mapping is not certain) | needs a page-level check before use |
| 2.7, 5.7, 7.9 | no | apparently absent or too corrupted to match |

The verified artefact is `knowledge/sources/mml/pdf-verification.json`, produced by
`tools/verify_source_against_pdf.py`. It records what was checked, what was not, and the
limitations observed while checking (pypdf skipped some form content, so "absent in the PDF" is
only ever concluded from pages that extracted cleanly).

## 5. Copyright position

The book is free to download but remains the authors' and publisher's copyright. The platform
therefore:

- cites the book by section and printed page, and does not host or reproduce its text;
- does not redistribute its figures; lessons use diagrams authored for this platform;
- may use the book's exercise *statements* as specifications for practice, with attribution and
  with solutions that are ours and that are marked as ours;
- keeps the supplied markdown and PDF in the repository unchanged as the canonical source, and
  separates them from generated content under `knowledge/`.

## 6. Coverage that the supplied book can support

Of the 82 competencies defined in the framework, the supplied book fully sources 23 and partially
sources 7. The remaining 52 have no source in the repository. The mapping is section-level:
`knowledge/sources/mml/section-concept-map.json` states, for each of the 81 sections, which
concepts it teaches and which competencies it supports; `knowledge/curriculum/coverage-report.md`
now reports the result per competency.

Notable strengths: matrix decompositions (MML 4.x → MATH-07, MATH-08), probability and the
Gaussian (6.x → MATH-12 to MATH-15), gradients and backpropagation (5.x → MATH-09 to MATH-11),
estimation and model selection (8.x-9.x → MATH-18, ML-04, ML-06), PCA and mixture models
(10.x-11.x → ML-07). Notable absences even inside the book's own subjects: numerical stability
and conditioning (MATH-19 is only `partial`, with the analysis needing a numerical-linear-algebra
source), tree-based methods and applied model selection (inside ML-02), and deep learning itself
(the book teaches the calculus behind training, not the field).
