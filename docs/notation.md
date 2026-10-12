# Notation conventions

Two notation systems meet in this project: the supplied book's, and the one this platform writes in.
They are close but not identical, and where they disagree the platform follows the book inside
quotations of it and this file everywhere else. Anything not listed here is written the way the book
writes it.

## Delimiters in lesson source files

| Form | Meaning |
| --- | --- |
| `$...$` | inline mathematics, rendered by KaTeX |
| `$$...$$` | display mathematics, rendered by KaTeX, on its own lines |
| backticks | code, identifiers, and short expressions that are *not* mathematics (a function name, a file path, a literal string) |

Prose mathematics in these lessons is mostly written as display equations or as backticked symbols
rather than as inline mathematics. That is deliberate: an inline `$k$` inside a sentence that says "k
components" adds a rendering dependency for no gain, while a display equation earns its space. Inline
mathematics is used where a symbol appears inside a sentence and the alternative would be a
parenthetical definition.

## Where this platform's conventions differ from the book's

| Concept | The book writes | These lessons write | Why |
| --- | --- | --- | --- |
| Matrix transpose | `A^⊤` | `A'`, and `Aᵀ` in display math | `A'` survives copy-paste and plain-text search; the display form matches the book |
| Vectors | bold lowercase, `x` | `x`, `v`, `v_i` in italics | the book's boldface does not survive markdown extraction readably, and the loss is only cosmetic where the type is already clear from context |
| Matrices | bold uppercase, `A` | `A` | as above |
| Inner product | `⟨x, y⟩` and `x^⊤y` interchangeably | `⟨x, y⟩` in prose, `x'y` in computations | the book uses both; the distinction here is prose versus line-by-line algebra |
| Element of a set | `∈` | `∈` | unchanged |
| The reals | `ℝ` | `R` in plain text, `\mathbb{R}` in display math | matches the source file's own ASCII form and renders identically |
| Norms | `‖·‖` | `‖·‖` with a subscript for the norm used (`‖A‖₂`, `‖A‖_F`) | the book's `‖A‖_2` and `‖A‖_F` are kept; the subscript is never omitted when two norms appear in one sentence |
| Rank | `rk(A)` | `rk(A)`, `rank A` in prose | unchanged |
| Kernel / null space | `ker(A)` | `ker(A)` | unchanged |
| Diagonal matrix | `diag(λ₁, …, λₙ)` | `diag(λ₁, …, λₙ)` | unchanged |

## Symbols that mean more than one thing

These collisions are real in the source and in the wider literature. Each lesson states which reading
is in force where it could be ambiguous.

| Symbol | Readings | Rule in these lessons |
| --- | --- | --- |
| `λ` | an eigenvalue; a Lagrange multiplier; a regularisation strength | context states which; in a chapter about eigenvalues it is always the eigenvalue unless the sentence says otherwise |
| `σ` | a singular value; a standard deviation; a logistic function | `σ_i` with an index is a singular value in this module; `σ²` in a statistics lesson is a variance |
| `k` | a rank of a truncation; an index; the number of neighbours or clusters | stated in every exercise statement |
| `D` | a diagonal matrix of eigenvalues; a dataset; a dimensionality | stated where used |
| `p` | a probability; the number of features | stated where used |
| `X` | a data matrix; a random variable | a capital letter in a data context is the matrix when it has a shape given beside it |
| `'` | transpose; a derivative | never a derivative in these lessons (the derivative is written `d/dx` or with Lagrange notation) |

## Provenance markers used in prose

| Marker | Meaning |
| --- | --- |
| "MML 4.6, printed p.129" | a specific place in the supplied book — section and printed page, never a PDF page |
| "the source" | the book named in the lesson's front matter, i.e. `SRC-MML` |
| `[REF-004]` | an entry in `knowledge/references/external.json`, with its authority and retrieval status |
| "not established here" | a statement this platform has not verified and is not asking the reader to take on trust |

## Why this file exists

A curriculum that reuses one book's notation without saying so forces the learner to reverse-engineer
conventions from context, and it silently breaks when a second book arrives with different ones. A
curriculum that invents its own notation uniformly makes comparison with the source harder. This file
is the compromise: the book's notation inside the mathematics, this platform's simplified forms in
prose and code, and a written record of every place the two differ.
