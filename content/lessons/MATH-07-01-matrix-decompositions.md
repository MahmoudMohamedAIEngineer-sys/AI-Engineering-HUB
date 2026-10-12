---
lesson:
  id: LSN-MATH-07-01
  title: "Matrix decompositions: Cholesky, eigendecomposition, and the SVD that works for every matrix"
  competencies: [MATH-07]
  covers_outcomes:
    - "State the existence conditions of the Cholesky, eigendecomposition and SVD, and give an example that violates each condition."
    - "Use the SVD to describe the action of a matrix as rotation, scaling and rotation."
    - "Relate the eigendecomposition of a symmetric positive-definite matrix to its Cholesky factor."
  sources:
    - work_id: SRC-MML
      sections: ["4.3", "4.4", "4.5", "4.7"]
      pages: "114-134"
      role: primary
  source_anchors:
    sections:
      - {id: "4.3", file: "Mathematics for ML part 1.md", line: 4216, printed_page: 114}
      - {id: "4.4", file: "Mathematics for ML part 1.md", line: 4264, printed_page: 115}
      - {id: "4.5", file: "Mathematics for ML part 1.md", line: 4423, printed_page: 119}
      - {id: "4.7", file: "Mathematics for ML part 1.md", line: 6, printed_page: 134}
    numbered_objects:
      - "Definition 4.19 (diagonalizable)"
      - "Definition 4.23 (spectral norm of a matrix)"
      - "Theorem 4.18 (Cholesky decomposition)"
      - "Theorem 4.20 (eigendecomposition)"
      - "Theorem 4.21 (a symmetric matrix can always be diagonalized)"
      - "Theorem 4.22 (SVD theorem)"
      - "Theorem 4.24 (the spectral norm of A is its largest singular value)"
      - "Figure 4.8 (SVD as sequential transformations)"
  supplemental_references: []
  prerequisites:
    - competency: MATH-06
      need: "Eigenvalues, eigenspaces and the spectral theorem. The SVD is constructed from the eigendecomposition of A'A, so the whole derivation rests on the symmetric case."
    - competency: MATH-04
      need: "Inner products, orthogonality and orthonormal bases. Without them 'orthogonal matrices' and 'the columns of U are orthonormal' carry no meaning."
    - competency: MATH-02
      need: "Matrices as linear mappings, for the interpretation of the SVD as basis change, scaling and basis change."
  estimated_effort: {reading_minutes: 95, exercise_minutes: 165}
  content_path: content/lessons/MATH-07-01-matrix-decompositions.md
  publication_state: draft
  verification_record_id: VER-LSN-MATH-07-01-01
  last_verified: 2026-10-12
  provenance:
    source_supported: >
      The Cholesky decomposition and its existence for symmetric positive-definite matrices
      (Theorem 4.18), the eigendecomposition and its existence conditions (Theorems 4.20, 4.21,
      Definition 4.19), the SVD theorem for any rectangular matrix (Theorem 4.22), the
      spectral norm as the largest singular value (Theorem 4.24) and the matrix phylogeny of
      section 4.7 are all taken from MML sections 4.3 to 4.7, with printed pages recorded above.
    established_theory: >
      The constructive proof of the SVD from the eigendecomposition of A'A, the induction proving
      existence of the Cholesky factorisation, the uniqueness argument for the Cholesky factor, the
      fast powers-of-sigma bound used to derive the spectral norm, and the worked examples are
      standard results written out in full here. Where the source leaves a proof to the reader or
      states a result without proof, the lesson says so rather than presenting the gap as covered.
    original_examples: >
      Every numerical instance (the 2x2 SPD matrix, the rectangular 3x2 matrix, the
      non-diagonalisable counterexample, the badly conditioned matrix in the code section, and all
      code) was produced for this lesson. The source's Figure 4.8 is cited by number and not
      reproduced, and the book's movie-ratings example is not reused.
    not_covered_here: >
      Numerical computation of the SVD (Golub-Kahan bidiagonalisation), the cost and conditioning
      of each decomposition (MATH-19, no source in the repository), and the low-rank approximation
      theorem, which is the subject of LSN-MATH-08-01.
exercises:
  - id: EX-LSN-MATH-07-01-01
    competencies: [MATH-07]
    origin: original
    origin_detail: {}
    difficulty: guided
    prompt: "Compute the Cholesky factor of A = [[9, 6, 3], [6, 5, 2], [3, 2, 2]] by hand using the column formulas, verify LL' = A, and explain which of the three diagonal entries of L would be imaginary if a single entry of A were changed, and why the factorisation would then not exist over the reals."
    expected_outcome: "A correct lower-triangular L with positive diagonal, an exact verification of LL' = A, and an explanation that the third pivot is the Schur complement whose positivity is exactly the positive-definiteness condition."
    solution_path: content/solutions/MATH-07-01-solutions.md#ex-01
    solution_verified_by: "independent Cholesky computation with numpy.linalg.cholesky and exact symbolic verification of LL' = A"
  - id: EX-LSN-MATH-07-01-02
    competencies: [MATH-07]
    origin: adapted
    origin_detail:
      work_id: SRC-MML
      section: "Chapter 4 exercises 4.8 and 4.9 (printed p.138): 'Find the SVD of the matrix' / 'Find the singular value decomposition of'"
      printed_page: 138
      note: "The task is the source's, which asks for the full SVD of a given matrix; the matrix here is a 3x2 rectangle chosen by this lesson so that U must be completed with a third orthogonal direction, a step the source's own matrices do not force."
    difficulty: independent
    prompt: "Compute the full SVD of A = [[1, 0], [0, 1], [1, 1]]. Work through the construction: form A'A, find its eigenvalues and an orthonormal eigenbasis, obtain the singular values and the right-singular vectors, then compute u_i = Av_i/sigma_i and extend the set to an orthonormal basis of the left space. Verify A = U Sigma V' by multiplying out, and state which singular values are zero and why."
    expected_outcome: "sigma = (sqrt(3), 1), an explicit U (3x3, including the direction orthogonal to the columns of A), Sigma 3x2, V 2x2, a verified reconstruction, and the observation that no singular value is zero because the columns of A are independent."
    solution_path: content/solutions/MATH-07-01-solutions.md#ex-02
    solution_verified_by: "exact symbolic reconstruction with sympy and comparison against numpy.linalg.svd"
  - id: EX-LSN-MATH-07-01-03
    competencies: [MATH-07]
    origin: original
    origin_detail: {}
    difficulty: independent
    prompt: "For each of the three decompositions in this lesson, give a matrix that violates its existence condition, with a one-line reason: a symmetric matrix with no Cholesky factor, a square matrix with no eigendecomposition, and — if you can — a matrix with no SVD. Then verify each failure numerically and explain what the failing computation actually does."
    expected_outcome: "A valid non-positive-definite symmetric matrix, a defective square matrix, the correct statement that no matrix lacks an SVD, and observations about what the library routines return or raise in the failing cases."
    solution_path: content/solutions/MATH-07-01-solutions.md#ex-03
    solution_verified_by: "the failures were executed and their outcomes recorded (numpy raises LinAlgError for the Cholesky failure; eig returns fewer independent eigenvectors)"
  - id: EX-LSN-MATH-07-01-04
    competencies: [MATH-07]
    origin: original
    origin_detail: {}
    difficulty: multi_step
    prompt: "Prove that ||A||_F^2 equals the sum of the squared singular values of A, and that ||A x|| <= sigma_1 ||x|| for every x. Use the SVD, not the entries. Then explain in one sentence why the second result is the reason sigma_1 is called the spectral norm."
    expected_outcome: "Both derivations from the SVD expansion of A, correct handling of the orthogonality of U and V, and the identification of sigma_1 as the operator norm."
    solution_path: content/solutions/MATH-07-01-solutions.md#ex-04
    solution_verified_by: "both identities checked numerically on 200 random matrices, recording the maximum deviation"
  - id: EX-LSN-MATH-07-01-05
    competencies: [MATH-07]
    origin: original
    origin_detail: {}
    difficulty: multi_step
    prompt: "A'A and AA' have the same nonzero eigenvalues (Chapter 4 exercise 4.11, printed p.138, which you should do separately). Instead, explain why their zero eigenvalues can differ in multiplicity: give a rectangular example where one of the two matrices is singular with a large null space and the other is not singular at all, and connect the null space of A'A to the null space of A."
    expected_outcome: "A correct example (for instance a short-and-wide or tall-and-narrow full-rank matrix), the identity ker(A'A) = ker(A) proved in a line, and the resulting multiplicities counted."
    solution_path: content/solutions/MATH-07-01-solutions.md#ex-05
    solution_verified_by: "null spaces and eigenvalue counts computed symbolically for the example"
  - id: EX-LSN-MATH-07-01-06
    competencies: [MATH-07]
    origin: original
    origin_detail: {}
    difficulty: transfer
    prompt: "Take an SPD matrix and compare two different square roots of it: the triangular Cholesky factor L and the symmetric square root A^{1/2} = Q Lambda^{1/2} Q'. Show both satisfy SS' = A, show they are different, and characterise every square root of A as Q Lambda^{1/2} W' for an orthogonal W. Then say which square root a sampling routine should use if it wants x = A^{1/2} z to have covariance A with the fewest operations, and why."
    expected_outcome: "Both factors verified, the general form proved from the spectral decomposition, and the correct practical answer with a reason (the triangular factor is cheaper for a single solve, and its structure is what makes the sampling transform cheap)."
    solution_path: content/solutions/MATH-07-01-solutions.md#ex-06
    solution_verified_by: "both square roots computed exactly, the orthogonal W extracted from the relation and verified to be orthogonal, and both products confirmed to equal A"
  - id: EX-LSN-MATH-07-01-07
    competencies: [MATH-07]
    origin: original
    origin_detail: {}
    difficulty: open_ended
    prompt: "Implement the textbook construction of the SVD: eigen-decompose A'A, take square roots of the eigenvalues, and build U from Av_i/sigma_i. Compare with numpy's SVD on 100 random matrices, then on matrices whose singular values span a wide range (epsilon chosen so that the ratio is around 1e8). Report at which ratio the construction stops delivering accurate small singular values, and explain why forming A'A is the cause."
    expected_outcome: "A working implementation, measured agreement on well-conditioned inputs, a measured loss of accuracy on ill-conditioned ones, and the explanation that forming A'A squares the ratio of largest to smallest singular value, which is exactly the information the small singular values carry."
    rubric: "Full credit requires: (1) the implementation handles rectangular matrices, not only square ones; (2) the comparison is quantitative (relative error in each singular value, not just 'it matched'); (3) the ill-conditioned experiment is run with a stated epsilon and the resulting error is reported; (4) the answer states clearly that numpy does not use this construction, so agreement tells you the mathematics is right, not that the algorithm is good."
    solution_path: content/solutions/MATH-07-01-solutions.md#ex-07
    solution_verified_by: "the reference implementation was executed and its measured relative errors are the recorded output of check_ex07_gram_svd_accuracy"
---

# Matrix decompositions

A matrix carries `mn` numbers, and most of what you want to know about it is not visible in those
numbers. Decompositions replace the matrix with a product of structured factors — triangular,
orthogonal, diagonal — and the structure is what carries the answer to the question you actually
have. In this lesson one question drives everything: **what does this mapping do, and how much of
it matters?**

Three decompositions answer variants of that question, and they have different domains:

| Decomposition | Applies to | Form | The structure it hands you |
| --- | --- | --- | --- |
| Cholesky | symmetric, positive definite | `A = LL'` | triangular factor, cheap to use |
| Eigendecomposition | diagonalisable square matrices | `A = PDP^{-1}` | invariant directions when the matrix is symmetric: `A = QΛQ'` |
| SVD | **every** matrix, any shape | `A = UΣV'` | a graded list of directions and their importance |

The SVD is the one that always exists, and the reason it is the workhorse of applied linear
algebra. MML develops the three in sections 4.3 to 4.5 and then, in section 4.7, draws a "matrix
phylogeny" that shows how the special classes nest inside each other. This lesson follows that
order, and derives the SVD rather than stating it, because the derivation is what tells you where
the singular values come from and why they are always real and non-negative.

Source: MML sections 4.3, 4.4, 4.5 and 4.7 (printed pages 114-134). The construction of the SVD
also depends on the eigenvalue material of sections 4.1-4.2, covered in `LSN-MATH-06-01`.

---

## Part 1 — Cholesky: the decomposition that requires the most and costs the least

### 1.1 The condition: symmetric positive definite

A square matrix `A` is **symmetric positive definite** (SPD) when `A = A'` and

$$
x^{\top} A x > 0 \quad \text{for every } x \neq 0.
$$

For a symmetric matrix this is equivalent to all eigenvalues being positive, which is worth
remembering because it is usually easier to check than the definition: by the spectral theorem
(`MATH-06`) a symmetric matrix has an orthonormal eigenbasis, so writing `x` in that basis gives
`x'Ax = Σ λᵢ cᵢ²`, which is positive for every nonzero `x` exactly when every eigenvalue is
positive.

Where SPD matrices come from, and why this matters for learning systems rather than only for
mathematics:

- **Covariance matrices.** `Cov(x) = E[(x-μ)(x-μ)']` is symmetric and, unless the data is
  degenerate, positive definite. Every Gaussian distribution is parameterised by one.
- **The Hessian of a convex function.** A twice-differentiable function is strictly convex at a
  point exactly when its Hessian there is positive definite (`MATH-17`).
- **Gram matrices and kernels.** `X'X` is positive semi-definite by construction; adding a ridge
  term makes it positive definite, which is precisely why ridge regularisation makes a problem
  solvable.

### 1.2 Constructing the factor, column by column

**Theorem 4.18 in the source** states that a symmetric positive-definite matrix can be factorised
as `A = LL'` with `L` lower triangular and with strictly positive diagonal entries.

Read the equation entrywise, from the top-left corner. For the first column of `L`:

$$
A_{11} = L_{11}^2 \quad\Rightarrow\quad L_{11} = \sqrt{A_{11}}, \qquad
A_{i1} = L_{i1} L_{11} \quad\Rightarrow\quad L_{i1} = A_{i1} / L_{11}.
$$

For the second column, the entries `L_{22}`, `L_{32}`, ... appear in the same way once the already
computed entries are subtracted:

$$
L_{jj} = \sqrt{\,A_{jj} - \sum_{k<j} L_{jk}^2\,}, \qquad
L_{ij} = \frac{1}{L_{jj}}\Big(A_{ij} - \sum_{k<j} L_{ik} L_{jk}\Big) \quad (i > j).
$$

Two things about these formulas are the whole content of the theorem:

1. **The square root argument is positive.** The quantity inside the square root is the `j`-th
   pivot of symmetric Gaussian elimination, and positive definiteness is exactly what keeps every
   pivot strictly positive. If `A` is not positive definite, some pivot is non-positive and the
   algorithm has no real answer — which is why *attempting* a Cholesky factorisation is a standard
   practical test of definiteness.
2. **The result is unique.** Both formulas determine each entry from previously computed ones, and
   the diagonal is fixed by the requirement that it be positive rather than negative.

The source proves the existence half by induction on the dimension: partition
`A = [[a, b'], [b, C]]`, take the Schur complement `C - bb'/a`, show it is itself SPD — for any
`y ≠ 0`, the vector `x = (-b'y/a, y)` is non-zero and gives `x'Ax = y'(C - bb'/a)y` — and apply the
induction hypothesis to it. That is the argument to reproduce when you want to know why the
theorem is true, and it also shows where it fails: the induction needs `a > 0` at every step.

### 1.3 A worked factorisation

Take

$$
A = \begin{bmatrix} 9 & 6 & 3 \\ 6 & 5 & 2 \\ 3 & 2 & 2 \end{bmatrix},
$$

which is symmetric, and whose leading minors are `9 > 0`, `9·5 - 36 = 9 > 0`, and
`det A = 9(10 - 4) - 6(12 - 6) + 3(12 - 15) = 54 - 36 - 9 = 9 > 0` — all positive, so by Sylvester's
criterion it is positive definite.

**First column.** `L₁₁ = √9 = 3`, `L₂₁ = 6/3 = 2`, `L₃₁ = 3/3 = 1`.

**Second column.** `L₂₂ = √(5 - 2²) = √1 = 1`, and `L₃₂ = (2 - 1·2)/1 = 0`.

**Third column.** `L₃₃ = √(2 - 1² - 0²) = 1`.

$$
L = \begin{bmatrix} 3 & 0 & 0 \\ 2 & 1 & 0 \\ 1 & 0 & 1 \end{bmatrix}.
$$

**Verification by multiplication** — this is the check to run every time, and it is cheap:

$$
LL' = \begin{bmatrix} 3 & 0 & 0 \\ 2 & 1 & 0 \\ 1 & 0 & 1 \end{bmatrix}
       \begin{bmatrix} 3 & 2 & 1 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}
     = \begin{bmatrix} 9 & 6 & 3 \\ 6 & 5 & 2 \\ 3 & 2 & 2 \end{bmatrix} = A. \quad\checkmark
$$

A structural remark worth noticing: `L₃₂ = 0` was not an accident of the arithmetic but a
consequence of `A₃₂ = L₃₁L₂₁ + L₃₂L₂₂ = 1·2 + 0·1 = 2`. The factor's sparsity is inherited from the
matrix's, which is why sparse Cholesky is the standard tool for very large systems.

### 1.4 Why the Cholesky factor is the useful one

- **Solving systems.** For SPD `A`, `Ax = b` becomes two triangular solves, `Ly = b` then
  `L'x = y`, each costing `n²` operations after a factorisation costing about `n³/6` — half of what
  a general LU factorisation costs, and without any pivoting decisions.
- **Sampling from a Gaussian.** If `z` has independent standard normal entries, then `x = μ + Lz`
  has covariance `LL' = A`. This is *the* reason generative models and Bayesian samplers contain a
  Cholesky call: it turns "draw from `N(μ, Σ)`" into "draw independent numbers and apply a
  triangular matrix".
- **A test that does not lie.** A failed Cholesky is proof of non-definiteness, and a successful
  one is proof of definiteness (up to floating-point effects). Eigenvalue computation answers the
  same question more slowly.

A caution that belongs to `MATH-19` rather than here: for badly conditioned `A`, computing `A'A`
and then factorising squares the condition number, so routines such as least squares should not be
solved by forming normal equations and applying Cholesky when `A` itself is ill-conditioned. The
supplied source does not develop this; it is recorded as a coverage gap.

---

## Part 2 — Eigendecomposition: the decomposition that requires the matrix to be nice

**Theorem 4.20 in the source** gives the eigendecomposition: a square matrix `A` factorises as

$$
A = PDP^{-1}
$$

with `D` diagonal if and only if `A` is **diagonalisable** (Definition 4.19). The diagonal entries
are the eigenvalues and the columns of `P` are corresponding eigenvectors, so the decomposition is
exactly the statement "there is a basis of eigenvectors". `LSN-MATH-06-01` derived the criterion:
every eigenvalue's geometric multiplicity must equal its algebraic multiplicity, which fails
precisely for defective matrices such as `[[1,1],[0,1]]`.

Two restrictions make the eigendecomposition less useful than the SVD for data work, and both are
worth stating as limitations rather than facts to memorise:

- **It only exists for some square matrices.** A matrix with a repeated eigenvalue and too few
  eigenvectors has no such factorisation at all.
- **`P` need not be orthogonal.** Even when the decomposition exists, `P^{-1}` is a general inverse.
  The exception is the symmetric case, where Theorem 4.21 plus the spectral theorem give

$$
S = Q\Lambda Q^{\top} \quad \text{with } Q \text{ orthogonal, } \Lambda \text{ real diagonal,}
$$

and that orthogonal factorisation is what the SVD is built from. In that case `Q^{-1} = Q'`, which
means no linear solve is needed to invert the mapping — a practical difference, not a cosmetic one.

**The link to Cholesky, made precise.** Both decompositions produce what can loosely be called a
square root of a matrix, and they are *different* square roots unless the matrix is diagonal:

- Cholesky gives the unique **lower-triangular** factor with positive diagonal.
- The spectral decomposition gives the **symmetric** square root `A^{1/2} = QΛ^{1/2}Q'`.

Both satisfy `SS' = A`. Neither is "the" square root: every square root of an SPD matrix has the
form `QΛ^{1/2}W'` for some orthogonal `W`, and the triangular one is singled out by that extra
structural requirement rather than by the decomposition. Exercise 6 verifies all of this on a small
matrix. This distinction is where a common misstatement lives — that the Cholesky factor equals
`QΛ^{1/2}`. It does not; `QΛ^{1/2}` is not even triangular in general.

---

## Part 3 — The SVD: existence for every matrix

### 3.1 The theorem

**Theorem 4.22 in the source.** Let `A ∈ R^{m×n}` be a rectangular matrix of rank `r`. The SVD of
`A` is a factorisation

$$
A = U\Sigma V^{\top},
$$

where `U ∈ R^{m×m}` and `V ∈ R^{n×n}` are orthogonal matrices, `Σ ∈ R^{m×n}` has non-negative
entries on its diagonal in decreasing order, and the diagonal entries `σ₁ ≥ σ₂ ≥ ... ≥ 0` are the
**singular values** of `A`. The columns of `U` and `V` are the left- and right-singular vectors.

No hypothesis is needed beyond `A` being a real matrix. That is the theorem's entire point.

### 3.2 Deriving it, step by step

The source constructs the SVD from the eigendecompositions of `A'A` and `AA'`. Here is that
construction with the reasoning made explicit, because every step uses something from earlier in
this curriculum.

**Step 1: `A'A` is symmetric positive semi-definite.** Symmetry: `(A'A)' = A'A`. Positive
semi-definiteness: for any `x`, `x'A'Ax = (Ax)'(Ax) = ‖Ax‖² ≥ 0`. By the spectral theorem
(`MATH-06`), `A'A` therefore has an orthonormal eigenbasis with only non-negative eigenvalues. Write

$$
A^{\top}A v_i = \lambda_i v_i, \qquad \|v_i\| = 1, \qquad \lambda_1 \geq \dots \geq \lambda_n \geq 0 .
$$

The non-negativity is not assumed, it is derived: taking inner products with `v_i` gives
`λᵢ = vᵢ'A'Avᵢ = ‖Avᵢ‖² ≥ 0`.

**Step 2: define the singular values.** Set $\sigma_i = \sqrt{\lambda_i}$. Then $\sigma_i \ge 0$, and
$\sigma_i = 0$ exactly when $Av_i = 0$. The rank $r$ of $A$ is the number of strictly positive
$\sigma_i$ — because $\|Av_i\|^2 = \lambda_i = 0$ shows that the eigenvectors for zero eigenvalues span the null space of `A`, a
fact used again in exercise 5.

**Step 3: build the left-singular vectors.** For `i ≤ r` define

$$
u_i = \frac{1}{\sigma_i} A v_i .
$$

These are unit vectors, they are pairwise orthogonal, and the computation shows both facts at once:
using $A^{\top}A v_j = \lambda_j v_j$,

$$
\langle u_i, u_j \rangle = \frac{1}{\sigma_i \sigma_j} v_i^{\top} A^{\top} A v_j
= \frac{\lambda_j}{\sigma_i \sigma_j}\, v_i^{\top} v_j
= \frac{\sigma_j}{\sigma_i}\, \delta_{ij},
$$

which is `1` when `i = j` and `0` otherwise. The same manipulation reveals *why* the same `σᵢ`
appear in the other eigenproblem:

$$
AA^{\top} u_i = \frac{1}{\sigma_i} A A^{\top} A v_i = \frac{\lambda_i}{\sigma_i} A v_i = \lambda_i u_i = \sigma_i^2 u_i,
$$

so the left-singular vectors are eigenvectors of $AA^{\top}$, with the same eigenvalues $\sigma_i^2$. That is
the reason `A'A` and `AA'` share their nonzero eigenvalues, which is exactly what Chapter 4's
exercise 4.11 (printed p.138) asks you to prove; you should do that one yourself, and exercise 5
below takes the part it does not ask.

**Step 4: complete `U`.** The vectors `u₁, ..., u_r` are orthonormal in `R^m`, and an orthonormal
set always extends to an orthonormal basis: append `m - r` unit vectors each orthogonal to all the
previous ones (`MATH-04` covers the construction). For those extra directions `A'Avᵢ = 0`, hence
`Avᵢ = 0`, which is what makes the next step consistent.

**Step 5: verify the factorisation.** Compare `UΣV'` with `A` on a basis of input space. For each
right-singular vector `v_j`, the right-hand side `UΣV'v_j = UΣe_j` equals `σ_j u_j`, while the
left-hand side is `Av_j` — equal by definition of `u_j` when `σ_j > 0`, and `0 = 0` when
`σ_j = 0`. Two linear maps that agree on a basis are equal, so `A = UΣV'`. ∎

**The same theorem in outer-product form.** Writing the product column by column gives the
identity that the rest of this chapter and the next lesson depend on:

$$
A = \sum_{i=1}^{r} \sigma_i u_i v_i^{\top}.
$$

`A` is a sum of `r` rank-1 matrices, ordered by importance — the first term is the single best
rank-1 piece of `A`, and each subsequent term is the best possible addition. MML section 4.6 uses
this form (its equation 4.90) to build image approximations; `LSN-MATH-08-01` proves what "best"
means.

### 3.3 Reading the factors as a recipe

An orthogonal matrix with determinant `+1` is a rotation, and one with determinant `-1` is a
rotation composed with a reflection. Reading `A = UΣV'` right to left, the source's Figure 4.8
shows the mapping as: change basis with `V'`, scale (and add or drop dimensions) with `Σ`, then
change basis again with `U`. In the square case that is exactly "rotation, scaling, rotation",
where the first and last rotations differ.

The scaling step is a *diagonal* matrix acting in the rotated coordinate system, so the image of
the unit sphere is another sphere stretched along the coordinate axes: a hyper-ellipsoid whose
semi-axis lengths are `σ₁, ..., σ_min(m,n)`. Two consequences to carry forward:

- **The singular values are the stretch factors, sorted.** Nothing else about the matrix tells you
  how much it stretches in each direction.
- **Rank is visible in `Σ`.** The number of nonzero `σᵢ` is the rank, and the directions where the
  stretch is zero are the null space. A rank-deficient matrix squashes at least one dimension to
  nothing, which is why it cannot be inverted.

### 3.4 A worked 2x2 SVD

Take `A = [[2, 2], [-1, 1]]`, following the five steps above.

**Step 1.** $A^{\top}A = \begin{bmatrix} 2 & -1 \\ 2 & 1 \end{bmatrix}\begin{bmatrix} 2 & 2 \\ -1 & 1 \end{bmatrix} = \begin{bmatrix} 5 & 3 \\ 3 & 5 \end{bmatrix}$.

**Step 2.** `det(A'A - λI) = (5-λ)² - 9 = λ² - 10λ + 16 = (λ-8)(λ-2)`, so `λ₁ = 8`, `λ₂ = 2` and
hence `σ₁ = 2√2 ≈ 2.828`, `σ₂ = √2 ≈ 1.414`.

**Step 3.** Eigenvector for `λ₁ = 8`: `(5-8)v₁ + 3v₂ = 0` gives `v₁ = v₂`, so
`v₁ = (1,1)'/√2`. For `λ₂ = 2`: `(5-2)v₁ + 3v₂ = 0` gives `v₁ = -v₂`, so `v₂ = (1,-1)'/√2`. These
are orthogonal, as the spectral theorem promises they can be chosen to be.

**Step 4.** `u₁ = Av₁/σ₁ = (4, 0)'/√2 / (2√2) = (1, 0)'` and
`u₂ = Av₂/σ₂ = (0, -2)'/√2 / √2 = (0, -1)'`.

**Step 5.** With `U = [[1,0],[0,-1]]`, `Σ = diag(2√2, √2)` and `V = (1/√2)[[1,1],[1,-1]]`,
multiplying back gives `A` exactly:

$$
U\Sigma V^{\top} = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}
\begin{bmatrix} 2\sqrt{2} & 0 \\ 0 & \sqrt{2} \end{bmatrix}
\frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}
= \frac{1}{\sqrt{2}}\begin{bmatrix} 2\sqrt{2} & 2\sqrt{2} \\ -\sqrt{2} & \sqrt{2} \end{bmatrix}
= \begin{bmatrix} 2 & 2 \\ -1 & 1 \end{bmatrix}. \quad\checkmark
$$

Three observations the example is designed to show:

- **The `U` factor can contain a reflection**: here `det U = -1`. That is consistent with
  `det A = 4 > 0`, because `det V = -1` too, and `det A = det U · det Σ · det V = (-1)(4)(-1) = 4`.
  The product of singular values always equals `|det A|` for a square matrix; the sign lives in the
  orthogonal factors.
- **The singular values are not the eigenvalues.** `A` has `det(A - λI) = (2-λ)(1-λ) + 2 =
  λ² - 3λ + 4`, whose roots are complex. There are no real eigenvalues, yet the SVD exists with a
  perfectly real, positive `Σ`. This is the concrete reason the SVD is the right tool for data
  matrices.
- **The ellipse picture.** The unit circle maps to an ellipse with semi-axes `2√2` and `√2`, and
  the directions of those axes are the columns of `U`.

### 3.5 What the SVD settles that the determinant could not

For a square matrix, `det A = 0` is a strictly logical test of invertibility, but a terrible
numerical one: `det` is a volume, and volumes of `n x n` matrices in `R^n` underflow long before
the matrix becomes ill-conditioned. The SVD gives the calibrated version. The **spectral norm**
(Definition 4.23) is defined as

$$
\|A\|_2 = \max_{x \neq 0} \frac{\|Ax\|}{\|x\|},
$$

and Theorem 4.24 states that this equals `σ₁`. The proof is three lines once you have the SVD:
write $x = \sum_i c_i v_i$, so $Ax = \sum_i c_i \sigma_i u_i$, and

$$
\|Ax\|^2 = \sum_i c_i^2 \sigma_i^2 \leq \sigma_1^2 \sum_i c_i^2 = \sigma_1^2 \|x\|^2,
$$

with equality at $x = v_1$. Dividing by $\|x\|$ and taking the maximum gives $\sigma_1$. Exercise 4 asks you
to complete this argument and its Frobenius-norm counterpart.

The smallest singular value then has a concrete meaning that no determinant provides: it is the
spectral-norm distance from `A` to the nearest singular matrix. That is a corollary of the
Eckart-Young theorem proved in `LSN-MATH-08-01`, and it is the reason `σ₁/σ_n` — the condition
number — is the quantity that appears in every discussion of numerical stability, while `det A`
does not.

---

## Part 4 — Choosing between them (the matrix phylogeny)

MML section 4.7 arranges the matrix classes in a tree, and the picture is worth internalising as a
decision procedure. Read it as a set of inclusions, from the most restricted class outwards:

```
symmetric positive definite  ⊂  symmetric  ⊂  normal  ⊂  all square matrices  ⊂  all matrices
        Cholesky              eigen (QΛQ')   eigen     eigen (if diagonalisable)      SVD
```

- **All matrices** have an SVD. If you have a choice, use it.
- **Symmetric matrices** additionally have an orthonormal eigendecomposition (`S = QΛQ'`) and no
  defective cases, and their singular values are the absolute values of their eigenvalues.
- **Positive definite matrices** additionally have a Cholesky factorisation, which is the cheapest
  of the three to compute and the most convenient to use when it exists.
- **Diagonalisable but not symmetric matrices** have an eigendecomposition `PDP^{-1}` whose `P` is
  not orthogonal, so the basis is not orthonormal and the factors do not behave as rotations. For
  numerical work this is a real loss: algorithm design usually prefers the SVD even for square
  matrices, because it trades a smaller set of "nice" matrices for stability on all of them.

A useful way to remember the practical rule: **ask what the matrix is, then ask what it does.**
Cholesky answers "is this SPD and how do I solve with it cheaply"; the eigendecomposition answers
"which directions are invariant"; the SVD answers "which directions matter, in what order, and how
much".

---

## Part 5 — Doing this with a computer

### 5.1 The three calls

```python
import numpy as np

A = np.array([[9.0, 6.0, 3.0],
              [6.0, 5.0, 2.0],
              [3.0, 2.0, 2.0]])

L = np.linalg.cholesky(A)                     # lower triangular, A = L @ L.T
w, Q = np.linalg.eigh(A)                      # symmetric: use eigh, not eig
U, s, Vt = np.linalg.svd(A)                   # s is decreasing, Vt is V-transposed
print(np.allclose(L @ L.T, A), np.allclose(Q @ np.diag(w) @ Q.T, A))
print("singular values:", s)
```

Three habits, each of which prevents a specific class of error:

1. **Use the symmetric routines when you know the matrix is symmetric.** `eigh` exploits the
   structure, returns real eigenvalues sorted ascending, and is faster and more accurate than
   `eig`. Choosing `eig` for a symmetric matrix is a missed opportunity, not a bug.
2. **`np.linalg.svd` returns `Vᵀ`, not `V`.** Naming the variable `Vt` rather than `V` makes the
   transposition visible at every use site; a silent transpose error is a classic source of wrong
   results that still type-check.
3. **Never index `s[0]` as "the largest" unless you checked.** `svd` returns singular values in
   decreasing order, while `eig` and `eigh` make no such promise (`eigh` ascending). When in doubt,
   sort explicitly.

### 5.2 The warning that the source leaves out

The derivation above computes the SVD from `A'A`. **Real libraries do not.** Forming `A'A` squares
the condition number, because `σ(A'A) = σ(A)²`, so the small singular values — the ones that decide
the numerical rank — lose about half their significant digits. Libraries use the Golub-Kahan
bidiagonalisation instead, which never forms `A'A`.

This is not a historical footnote; it is measurable, and exercise 7 asks you to measure it. The
experiment below was run in this project's environment; the figures are the recorded output of the
lesson's checks (`knowledge/lessons/checks-report.json`), not estimates.

**First, the construction is right.** Over 100 random matrices of sizes 3 to 11, the largest
relative difference between the singular values from `A'A` and those from the library is
**4.3e-13** — floating-point round-off. So the mathematics of the derivation holds, and any
difficulty is numerical.

**Second, the construction is fragile.** Take a 40x40 matrix whose singular values are chosen to
run geometrically from 1 down to `1e-8`:

| | smallest singular value | relative error |
| --- | --- | --- |
| library SVD (Golub-Kahan) | 1.000e-08 | 4.7e-10 |
| eigenvalues of `A'A`, square-rooted | 1.080e-08 | **8.0e-02** |

The `A'A` route gets the smallest singular value wrong by 8%. Now sweep the ratio between the
largest and smallest singular value and watch the error grow:

| ratio σ₁/σ_n | relative error of the smallest singular value, `A'A` route |
| --- | --- |
| 1e2 | 1.2e-13 |
| 1e4 | 2.5e-09 |
| 1e6 | 1.9e-06 |
| 1e8 | 8.0e-02 |
| 1e10 | 1.0 (no correct digits) |
| 1e12 | 1.0 |
| 1e16 | 1.0 |

The pattern is the prediction made concrete: squaring the condition number costs about one decimal
digit per two orders of magnitude, and by a ratio of `1e10` the smallest singular value — the one
that carries the rank decision — is entirely lost. This is why the SVD is computed by
bidiagonalising `A` and never by forming `A'A`, and why `MATH-19` is on the curriculum map with a
recorded gap: the supplied source derives the SVD but does not discuss its computation.

### 5.3 What to check after any decomposition

```python
def svd_checks(A, U, s, Vt, tol=1e-10):
    m, n = A.shape
    assert np.allclose(U @ U.T, np.eye(m)), "U is not orthogonal"
    assert np.allclose(Vt @ Vt.T, np.eye(n)), "V is not orthogonal"
    assert np.all(s[:-1] >= s[1:] - tol), "singular values are not sorted"
    assert np.all(s >= -tol), "a singular value is negative"
    # Reconstruct with the MINIMUM number of terms: r nonzero singular values.
    r = int((s > tol * s[0]).sum())
    approx = (U[:, :r] * s[:r]) @ Vt[:r, :]
    assert np.allclose(approx, A), f"reconstruction failed, rank {r}"
    return r
```

The rank returned here is the **numerical rank**: the number of singular values above a threshold
relative to `σ₁`. It is a decision, not a fact about the matrix — the choice of threshold is
exactly what a numerical-linear-algebra source would explain, and it is currently a gap. Saying
"the rank is the number of nonzero singular values" is mathematically right and computationally
useless, because computed singular values are almost never exactly zero.

---

## Common misconceptions, with counterexamples

**1. "The singular values are the absolute values of the eigenvalues."** True only for symmetric
matrices. The worked example in section 3.4 has singular values `2√2, √2` and no real eigenvalues
at all.

**2. "The SVD requires the matrix to be square / full rank / symmetric."** None of those. Theorem
4.22 needs nothing but a real matrix, which is exactly why the SVD is used on data matrices whose
shape changes between projects.

**3. "The Cholesky factor is `QΛ^{1/2}`, from the eigendecomposition."** It is not: `QΛ^{1/2}` is
symmetric, while `L` is triangular, and they are equal only in diagonal cases. Both are square
roots; the triangular one is unique.

**4. "`U` and `V` are unique."** The singular values are unique. The vectors are not: each pair
`(uᵢ, vᵢ)` can be replaced by `(-uᵢ, -vᵢ)`, and when a singular value repeats, any orthonormal basis
of that eigenspace works. Code that compares `U` from two libraries element-by-element is testing
something that is not determined.

**5. "`det A ≠ 0` means the matrix is safe to invert."** It means the matrix is invertible, not
that it is well conditioned. The SVD gives the number that tells you how safe it is: `σ₁/σ_n`.

**6. "Forming `A'A` is how you compute an SVD."** It is how you *derive* one. See section 5.2 for
the measurement.

**7. "The rank is the number of nonzero singular values, so `matrix_rank` is exact."** Computed
singular values are rarely exactly zero; `matrix_rank` compares against a tolerance and therefore
returns a decision. It is the right decision to use, but it is not a property read off the matrix.

---

## Practice

| # | Focus | Difficulty | Origin |
| --- | --- | --- | --- |
| 1 | Cholesky by hand, verified, with the pivot that carries the condition | guided | original |
| 2 | Full SVD of a 3x2 matrix, including completing `U` | independent | adapted (MML 4.8/4.9, p.138) |
| 3 | Existence conditions, each violated by a concrete matrix | independent | original |
| 4 | Prove `‖A‖_F² = Σσᵢ²` and `‖Ax‖ ≤ σ₁‖x‖`, then name the norm | multi-step | original |
| 5 | Why `A'A` and `AA'` differ in their zero eigenvalues | multi-step | original (source 4.11 cited, done separately) |
| 6 | Cholesky factor versus symmetric square root, and all square roots | transfer | original |
| 7 | Implement the textbook SVD and measure where it breaks down | open-ended | original |

**Left for you, deliberately**, because they are good practice exactly as written and this platform
does not publish solutions where the book publishes none: **MML exercises 4.8** and **4.9**
(printed p.138, full SVDs of given matrices), **4.11** (equal nonzero eigenvalues of `A'A` and
`AA'`) and **4.12** (prove the spectral-norm formula of Theorem 4.24 for `x ≠ 0`). Exercise 4 above
proves the Frobenius-norm companion result and the bound, which the source does not ask; that is a
complement to 4.12, not a substitute for it. Exercise 10 of the same set, the rank-1 approximation,
belongs to `LSN-MATH-08-01` and is left for there.

---

## Connections

- **Backwards.** `LSN-MATH-06-01` supplies the eigenvalues, eigenspaces, multiplicity test and the
  spectral theorem that this lesson's construction depends on. `MATH-04` supplies orthogonality and
  orthonormal bases; `MATH-02` the mapping interpretation behind "rotation, scaling, rotation".
- **Forwards.** `LSN-MATH-08-01` proves the Eckart-Young theorem for the truncation and turns
  `A = Σσᵢuᵢvᵢ'` into compression, using `‖A‖_F² = Σσᵢ²` from exercise 4. The PCA lesson (`ML-07`)
  is the eigendecomposition of a covariance matrix, with section 10.4 of the source making the link
  to low-rank approximation explicitly. `MATH-19` is the recorded gap covering cost, conditioning
  and algorithm choice.
- **Sideways.** The Cholesky factor appears again in the Gaussian material: a covariance matrix is
  SPD, and sampling uses its triangular factor (`MATH-14`).

## Further reading

- **Primary source.** MML Section 4.3 (printed pp.114-115) for Cholesky with Theorem 4.18, Section
  4.4 (pp.115-119) for the eigendecomposition with Theorems 4.20 and 4.21 and Definition 4.19,
  Section 4.5 (pp.119-129) for the SVD with Theorem 4.22, Figure 4.8 and Theorem 4.24, and Section
  4.7 (p.134) for the phylogeny. Figures 4.8 to 4.10 in your copy show the sequential-transformation
  reading and are worth comparing against section 3.3 above.
- **Your roadmap's visual companion.** The timetable maps this material to 3Blue1Brown's *Essence
  of Linear Algebra* for the geometric picture (chapter numbers as given in your plan; not verified
  against the series in this session) and to StatQuest for the applied intuition.
- **Not available in this repository.** Cost and stability of the three algorithms, the Golub-Kahan
  computation, and numerical rank thresholds. These are recorded as gaps in the coverage report
  rather than filled from memory.
