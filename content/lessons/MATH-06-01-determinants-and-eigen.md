---
lesson:
  id: LSN-MATH-06-01
  title: "Determinants, eigenvalues and eigenvectors: what a matrix does to its own directions"
  competencies: [MATH-06]
  covers_outcomes:
    - "Compute eigenvalues and eigenvectors of a small matrix by hand and verify the result by substitution."
    - "Explain what the determinant says about invertibility and volume scaling, including its sign."
    - "Determine whether a matrix is diagonalisable and explain what failure to be diagonalisable means."
  sources:
    - work_id: SRC-MML
      sections: ["4.1", "4.2"]
      pages: "99-113"
      role: primary
  source_anchors:
    sections:
      - {id: "4.1", file: "Mathematics for ML part 1.md", line: 3614, printed_page: 99}
      - {id: "4.2", file: "Mathematics for ML part 1.md", line: 3818, printed_page: 105}
    numbered_objects:
      - "Definition 4.4 (determinant)"
      - "Definition 4.5 (characteristic polynomial)"
      - "Definition 4.9 (algebraic multiplicity)"
      - "Definition 4.10 (eigenspace and eigenspectrum)"
      - "Theorem 4.12 (eigenvectors with distinct eigenvalues are linearly independent)"
      - "Theorem 4.15 (spectral theorem)"
      - "Figure 4.7 (eigendecomposition as sequential transformations)"
  supplemental_references: []
  prerequisites:
    - competency: MATH-01
      need: "Linear independence. The test for a nonzero solution of (A - λI)v = 0 is a statement about dependence of the columns."
    - competency: MATH-02
      need: "Matrices as linear mappings. Without the mapping reading, 'the direction it does not rotate' is not a meaningful question."
    - competency: MATH-03
      need: "Solvability of systems. You need the fact that a square homogeneous system has a nonzero solution exactly when its matrix is singular."
  estimated_effort: {reading_minutes: 80, exercise_minutes: 150}
  content_path: content/lessons/MATH-06-01-determinants-and-eigen.md
  publication_state: draft
  verification_record_id: VER-LSN-MATH-06-01-01
  last_verified: 2026-10-12
  provenance:
    source_supported: >
      The determinant as a volume-scaling and orientation factor, the characteristic polynomial,
      algebraic multiplicity, eigenspaces, the independence of eigenvectors with distinct
      eigenvalues, and the spectral theorem are all taken from MML sections 4.1 and 4.2, with the
      printed pages recorded above.
    established_theory: >
      The derivation of the 2x2 determinant from the area of the image parallelogram, the trace and
      determinant invariants proved from the characteristic polynomial, the diagonalisation argument
      for repeated application, and the counterexamples in the misconception section are standard
      results written out here in full.
    original_examples: >
      Every numerical instance used for teaching (the matrix [[2,1],[1,2]], the defective matrix
      [[1,1],[0,1]], the non-defective repeated-eigenvalue matrix, the exercise matrices, and all
      code) was chosen for this lesson. The book's own examples were not copied, and the book's
      figures are cited by number and page rather than reproduced.
    not_covered_here: >
      Computation of the determinant of large matrices, numerical conditioning of eigenproblems
      (MATH-19), and the eigendecomposition as a tool (MATH-07, next lesson). The Cholesky
      decomposition (MML 4.3) is deliberately skipped here.
exercises:
  - id: EX-LSN-MATH-06-01-01
    competencies: [MATH-06]
    origin: original
    origin_detail: {}
    difficulty: recognition
    prompt: "For B = [[2, 0, 3], [1, 4, -1], [0, 2, 1]], compute the determinant twice: by Sarrus' rule and by Laplace expansion along the first row. Confirm that the two agree, and state whether B is invertible."
    expected_outcome: "Correct determinant by two independent expansions, and the conclusion det B != 0 so B is invertible."
    solution_path: content/solutions/MATH-06-01-solutions.md#ex-01
    solution_verified_by: "independent symbolic computation with sympy (tools/run_content_checks.py, check math06_ex01_determinant)"
  - id: EX-LSN-MATH-06-01-02
    competencies: [MATH-06]
    origin: original
    origin_detail: {}
    difficulty: guided
    prompt: "Explain, without a full expansion, why det([[1,2,3],[4,5,6],[1,2,3]]) = 0, and name the property of the determinant you used. Then construct a 4x4 matrix whose determinant is 0 for a different reason."
    expected_outcome: "Identification of the alternating (or multilinearity) property when two rows coincide, plus a second, structurally different reason for a zero determinant."
    solution_path: content/solutions/MATH-06-01-solutions.md#ex-02
    solution_verified_by: "independent symbolic computation with sympy and manual reasoning recorded in the solution"
  - id: EX-LSN-MATH-06-01-03
    competencies: [MATH-06]
    origin: adapted
    origin_detail:
      work_id: SRC-MML
      section: "4.3 exercises (printed p.139), exercise 4.3 in the source's numbering asks the same question for two 2x2 matrices"
      printed_page: 139
      note: "The task is the source's; the matrices are ours, so that the book's own exercise remains unsolved here for independent practice."
    difficulty: independent
    prompt: "Compute all eigenvalues and the corresponding eigenspaces of A = [[3,1],[1,3]] and of C = [[2,2],[1,3]] by hand. Verify each eigenpair by substitution, and verify both eigenvalues against the trace and the determinant."
    expected_outcome: "Four correct eigenpairs with eigenspaces given as spans, each verified by substitution and by the trace/determinant invariants."
    solution_path: content/solutions/MATH-06-01-solutions.md#ex-03
    solution_verified_by: "independent symbolic computation with sympy (check math06_ex03_eigenpairs)"
  - id: EX-LSN-MATH-06-01-04
    competencies: [MATH-06]
    origin: adapted
    origin_detail:
      work_id: SRC-MML
      section: "4.5 exercises (printed p.138), exercise 4.5 in the source's numbering uses four matrices for the same question"
      printed_page: 138
      note: "The question is the source's (diagonalisability is unrelated to invertibility); the five matrices are ours."
    difficulty: multi_step
    prompt: "Complete a 5x2 table for the matrices listed in the solution file: for each, decide whether it is invertible and whether it is diagonalisable over R, and give the reason for each answer. Two of the five require the same matrix to be invertible but not diagonalisable over R."
    expected_outcome: "Correct invertibility from the determinant, correct diagonalisability from the algebraic-versus-geometric multiplicity comparison, and reasons that name the multiplicities rather than the eigenvalues alone."
    solution_path: content/solutions/MATH-06-01-solutions.md#ex-04
    solution_verified_by: "independent computation of multiplicities with numpy/sympy and manual reasoning recorded in the solution"
  - id: EX-LSN-MATH-06-01-05
    competencies: [MATH-06]
    origin: original
    origin_detail: {}
    difficulty: multi_step
    prompt: "Prove that if v is an eigenvector of A with eigenvalue λ, then v is an eigenvector of A^k with eigenvalue λ^k for every k >= 1. Then show that the converse fails: give a real 2x2 matrix A for which A^2 has a real eigenvalue that is not the square of any real eigenvalue of A."
    expected_outcome: "A correct induction argument, and a counterexample (a rotation matrix) together with the explanation that its eigenvalues are complex, which is what makes the real converse fail."
    solution_path: content/solutions/MATH-06-01-solutions.md#ex-05
    solution_verified_by: "symbolic verification of the counterexample in sympy (check math06_ex05_converse)"
  - id: EX-LSN-MATH-06-01-06
    competencies: [MATH-06]
    origin: original
    origin_detail: {}
    difficulty: transfer
    prompt: "Show that every real 3x3 matrix has at least one real eigenvalue, and give an example of a real 3x3 matrix with exactly one. Explain how this argument changes in dimension 2."
    expected_outcome: "The odd-degree real polynomial argument, a valid example (a rotation in a plane plus a scaling along the third axis), and the correct statement that the argument fails in even dimension with the rotation in 2D as the witness."
    solution_path: content/solutions/MATH-06-01-solutions.md#ex-06
    solution_verified_by: "symbolic computation of the characteristic polynomials (check math06_ex06_odd_degree)"
  - id: EX-LSN-MATH-06-01-07
    competencies: [MATH-06]
    origin: original
    origin_detail: {}
    difficulty: open_ended
    prompt: "Implement a function eigen_2x2(A) that returns the eigenvalues and eigenvectors of a real 2x2 matrix using the quadratic formula rather than a library decomposition. Test it against numpy.linalg.eig on 1000 random matrices and report the maximum residual ||A v - λ v||. Explain which matrices make the residual large and why the comparison is being made against a library at all."
    expected_outcome: "A working implementation, a measured residual, an identification of the ill-conditioned cases (nearly repeated eigenvalues), and a statement of what the test does and does not establish."
    rubric: "Full credit requires: (1) the implementation handles both real and complex eigenvalue cases rather than ignoring one; (2) the residual is measured rather than asserted; (3) at least one concrete example of a matrix where the two disagree is shown, with the reason; (4) the answer states that agreement with numpy establishes consistency, not correctness, unless the mathematics is checked independently too."
    solution_path: content/solutions/MATH-06-01-solutions.md#ex-07
    solution_verified_by: "the reference implementation was executed by check_ex07_reference_implementation; the reported residuals and the near-defective case are its recorded output"
---

# Determinants, eigenvalues and eigenvectors

You already know two things from earlier work. A matrix is a linear mapping
(`MATH-02`), and a square system `Ax = b` has a unique solution exactly when `A` is
invertible (`MATH-03`). This lesson answers two questions about that mapping that turn out
to be the same question in two costumes:

1. **Does the mapping preserve volume, and does it flip orientation?** The answer is a single
   number, the **determinant**, and it also decides invertibility.
2. **Are there directions the mapping does not rotate, only stretch?** Those directions are the
   **eigenvectors**, and the stretch factors are the **eigenvalues**.

The second question is the reason this chapter exists, because it turns a matrix into something
you can reason about: a set of independent directions with known behaviour. Every result in
the rest of this chapter — the eigendecomposition, the SVD, the low-rank approximation, and,
two chapters later, PCA — is a consequence of the answers here. The two questions meet at one
equation, `det(A - λI) = 0`, which is why they are taught together.

The source for this lesson is *Mathematics for Machine Learning* (MML) sections 4.1 and 4.2
(printed pages 99-113). The printed pages are recorded in the front matter of this file; the
book's figures are cited by number and are not reproduced here, so that you read them in your
own copy.

---

## Part 1 — The determinant as a volume scaling factor

### 1.1 The 2x2 case, from geometry rather than from a formula

Take the mapping `A = [[a, b], [c, d]]`. Its two columns are the images of the standard basis
vectors, and the unit square is spanned by those two basis vectors. So the parallelogram spanned
by the two columns is exactly the image of the unit square.

![The unit square mapped by A: its area becomes |det A|](assets/MATH-06-01-parallelogram.svg)

What is the area of that parallelogram? Put `u = (a, c)` and `v = (b, d)`. In two dimensions the
signed area of the parallelogram spanned by `u` and `v` can be written in terms of their
coordinates: it equals `ad - bc`. One way to see it: subtract the parallelogram parts from the
bounding rectangle of side lengths `(a + b)` and `(c + d)`, giving
`(a+b)(c+d) - ac - bd - 2bc = ad - bc`. The sign is the part that matters for us: it records
whether the mapping reversed orientation.

So for a 2x2 matrix,

$$
\det A = ad - bc = \text{signed area of the image of the unit square.}
$$

Three consequences follow immediately, before any general theory:

- `det A = 0` means the image has zero area, which means the two columns are **linearly
  dependent**, which means the mapping collapses the plane onto a line. That is exactly what
  being non-invertible means: information has been destroyed and cannot be recovered.
- `det A < 0` means orientation was reversed, which is what a reflection does.
- The magnitude of the determinant tells you how much area is scaled.

Check the last claim on a scaling matrix: `diag(2, 3)` maps the unit square to a 2-by-3
rectangle of area 6, and `det diag(2, 3) = 6`. Check it on a rotation: rotating does not change
area, and every rotation matrix has determinant 1. Check a shear: `[[1, 1], [0, 1]]` maps the
square to a parallelogram of the same area, and `det = 1`. None of these needed a formula.

### 1.2 The general definition, and the one property that really matters

For an `n x n` matrix, the determinant is the unique function of the columns that is:

1. **multilinear** — linear in each column separately;
2. **alternating** — zero whenever two columns are equal, which (with multilinearity) forces a
   sign change when two columns are swapped;
3. **normalised** — `det I = 1`.

These three properties are the definition; the familiar formulas are consequences. The reason to
state it this way is that property 2 is what makes the determinant useful: it is a machine for
detecting **dependence**, and dependence is what "no volume" means.

The property used constantly in practice is multiplicativity:

$$
\det(AB) = (\det A)(\det B).
$$

Why it must hold: applying `B` and then `A` scales volume by `det B` and then by `det A`, and the
result of applying both is the map `AB`. Volume scaling factors multiply because they are
ratios of volumes. (MML proves this from the definition; the geometric reading is the reason to
believe it before seeing the proof.)

A warning that will matter later (and is a genuine gap in the supplied source, tracked as
`MATH-19`): the textbook expansions below are combinatorial. Laplace expansion of an `n x n`
determinant costs on the order of `n!` operations, so for `n = 20` it is hopeless. Any real
computation factorises the matrix and reads the determinant off the factors. You should be able
to compute a determinant by hand for `n = 2` and `n = 3`; you should never do it by Laplace
expansion for `n = 15`.

### 1.3 A worked computation, and what the answer tells you

Compute

```
A = [[1, 3, 5],
     [2, 4, 6],
     [0, 2, 4]]
```

**By Sarrus' rule** (three down-right products minus three down-left products):

```
det A =  1·4·4 + 3·6·0 + 5·2·2  −  5·4·0 + 1·6·2 + 3·2·4
      = (16 + 0 + 20) − (0 + 12 + 24)
      = 36 − 36
      = 0.
```

**By Laplace expansion along the first row**, expanding each 2x2 minor with its sign
(`+, −, +`):

```
det A = 1·det[[4,6],[2,4]] − 3·det[[2,6],[0,4]] + 5·det[[2,4],[0,2]]
      = 1·(16 − 12) − 3·(8 − 0) + 5·(4 − 0)
      = 4 − 24 + 20
      = 0.
```

Both methods agree, and the answer is zero, which is telling you something: `A` is singular. You
can confirm it without any expansion at all, because the rows are dependent:
`2·row1 − row2 = (2, 6, 10) − (2, 4, 6) = (0, 2, 4) = row3`. So one row is a combination of the
others, the columns cannot span three dimensions, and the volume is zero.

This is the pattern to notice: when a determinant problem has an answer of zero, the intended
insight is almost always a dependency you could have spotted, and checking that dependency is a
fast way to be sure of your arithmetic. This particular matrix is the one MML uses in its
Exercise 4.1 (printed page 137); if you work that exercise, you are doing this computation
twice, in the two ways the book asks for.

### 1.4 Where the determinant appears in the rest of this curriculum

- **Solvability.** `det A != 0` is the invertibility test used by the characteristic polynomial
  below, and by the change-of-variables formula for probability densities (MML 6.7, `MATH-12`),
  where the determinant of the Jacobian is exactly the local volume scaling factor.
- **Volume as a model of capacity.** The determinant of a covariance matrix measures the spread
  of a Gaussian in the way a volume does (MML 6.5, `MATH-14`).
- **A warning about numerical use.** For large matrices, `det` is a bad way to test
  invertibility: it can overflow or underflow, and a value near zero gives no calibrated sense
  of how close to singular the matrix is. The singular values of the next lesson give a better
  answer, and that is one reason the SVD is everywhere.

---

## Part 2 — Eigenvalues and eigenvectors

### 2.1 The question, made precise

A linear mapping moves almost every vector: it changes both its length and its direction. The
structures that survive the mapping — the subspaces that are mapped into themselves — are what we
can reason about. A one-dimensional invariant subspace is the simplest such structure, and it
exists exactly when the mapping sends some nonzero vector `v` to a multiple of itself:

```
A v = λ v,   v != 0.
```

`λ` is an **eigenvalue** and `v` an **eigenvector** (MML Section 4.2; Definition 4.9 covers the
algebraic multiplicity of an eigenvalue, and Definition 4.10 the eigenspace and eigenspectrum).
Note what the definition insists on: `v != 0`. The zero vector satisfies any such equation
trivially and carries no information about the mapping.

Geometrically: an eigenvector is a direction that the mapping does not rotate. If `λ > 1` the
vector is stretched, `0 < λ < 1` shrinks it, `λ < 0` reverses it, and `λ = 0` annihilates it —
which means `A` is singular and the eigenvector spans the null space.

### 2.2 Deriving the characteristic polynomial — and why the determinant returns here

The eigenvalue equation is a homogeneous system in disguise:

```
A v = λ v   ⇔   A v − λ v = 0   ⇔   (A − λI) v = 0.
```

By the result you already have from `MATH-03` (a square homogeneous system has a nonzero
solution exactly when its matrix is singular):

$$
(A - \lambda I)v = 0 \ \text{has a nonzero solution}
\iff A - \lambda I \ \text{is singular}
\iff \det(A - \lambda I) = 0.
$$

So the eigenvalues are precisely the roots of the **characteristic polynomial**

```
p_A(λ) = det(A − λI),
```

which has degree `n` in `λ` for an `n x n` matrix (MML Definition 4.5). Over the complex numbers
it has exactly `n` roots when counted with multiplicity, by the fundamental theorem of algebra —
so every complex matrix has `n` eigenvalues, though over the reals it may have fewer.

That is the first honest connection between the two halves of this lesson: the determinant,
which measured volume, is now a polynomial whose roots are the scaling factors of the invariant
directions. If `det(A − λI) = 0`, the matrix `A − λI` collapses space, which is precisely why a
nonzero vector can survive the collapse.

### 2.3 A full worked example, by hand

Take

```
A = [[2, 1],
     [1, 2]]
```

**Step 1 — the characteristic polynomial.**

```
A − λI = [[2 − λ, 1],
          [1, 2 − λ]]

det(A − λI) = (2 − λ)(2 − λ) − 1·1 = λ² − 4λ + 3 = (λ − 3)(λ − 1).
```

So `λ₁ = 3` and `λ₂ = 1`.

**Step 2 — the eigenspace for `λ = 3`.** Solve `(A − 3I)v = 0`:

```
A − 3I = [[−1, 1],
          [ 1, −1]]
```

The two rows are negatives of each other, so the system reduces to the single equation
`−v₁ + v₂ = 0`, i.e. `v₁ = v₂`. The eigenspace is
`E₃ = span{(1, 1)}`, one-dimensional.

**Step 3 — the eigenspace for `λ = 1`.** Solve `(A − I)v = 0`:

```
A − I = [[1, 1],
         [1, 1]]
```

Reduces to `v₁ + v₂ = 0`, so `E₁ = span{(1, −1)}`, also one-dimensional.

**Step 4 — verify by substitution.** This step is not optional; it is how you catch arithmetic
errors before they propagate into everything downstream.

```
A(1, 1)ᵀ = (2·1 + 1·1, 1·1 + 2·1)ᵀ = (3, 3)ᵀ = 3·(1, 1)ᵀ   ✓
A(1, −1)ᵀ = (2·1 + 1·(−1), 1·1 + 2·(−1))ᵀ = (1, −1)ᵀ = 1·(1, −1)ᵀ   ✓
```

**Step 5 — check against the invariants** (next section). `tr A = 4 = 3 + 1` ✓ and
`det A = 3 = 3·1` ✓.

**Interpretation.** `A` maps the line `span{(1,1)}` to itself, stretching it by 3, and leaves
the line `span{(1,−1)}` fixed. Every other vector is a combination of those two directions and
is therefore stretched between the two factors. Two numbers and two directions have replaced
what looked like four arbitrary entries.

### 2.4 Two free checks: trace and determinant

Write the characteristic polynomial as `(λ₁ − λ)(λ₂ − λ) ... (λₙ − λ)` and compare with the
expansion of `det(A − λI)`. For the 2x2 case the comparison is short enough to do completely:

```
det(A − λI) = λ² − (tr A)λ + det A,
```

where `tr A = a + d`. On the other hand, if `λ₁, λ₂` are the roots, then

```
(λ − λ₁)(λ − λ₂) = λ² − (λ₁ + λ₂)λ + λ₁λ₂.
```

Matching coefficients gives, for every 2x2 matrix,

```
tr A = λ₁ + λ₂   and   det A = λ₁ λ₂.
```

**Consequences for the sign of the determinant.** Because `det A` is the product of the
eigenvalues, `det A = 0` means at least one eigenvalue is zero (singular), and `det A < 0` means
the eigenvalues have opposite signs — the mapping stretches in one direction and reverses in
another, which is exactly what reversing orientation means. This is the shape of the general
result too: the trace is the sum of all eigenvalues and the determinant is their product, with
multiplicities (MML Theorem 4.20 is the version for the eigendecomposition; Section 4.2 states
the invariants).

Use these as checks on every hand computation. If your eigenvalues are `{2, 5}` for a matrix
whose trace is 9 and determinant is 6, you have made an arithmetic error somewhere and you have
not yet had to re-derive anything to find out.

### 2.5 When there are not enough eigenvectors

Three situations need to be distinguished, and conflating them is the most common error in this
chapter.

**(a) Not enough independent eigenvectors: defective matrices.** Consider the "shear-ish"
triangular matrix

```
J = [[1, 1],
     [0, 1]]
```

The characteristic polynomial is `(1 − λ)²`, so `λ = 1` appears twice: its **algebraic
multiplicity** is 2. Now find the eigenspace: `J − I = [[0, 1], [0, 0]]` has rank 1, so its
null space is one-dimensional, spanned by `(1, 0)`. The **geometric multiplicity** — the
dimension of the eigenspace, `dim ker(A − λI)` — is 1. Since 1 < 2, there are not enough
independent eigenvectors to span the plane, and `J` is called **defective**: it cannot be
diagonalised. Geometrically it stretches along one direction and shears everything else.

**(b) No real eigenvalues: rotations.** Take the 90-degree rotation

```
R = [[0, −1],
     [1,  0]]
```

`det(R − λI) = λ² + 1`, whose roots are `λ = ±i`. Over the real numbers there are no eigenvalues
at all, and that is the correct answer, not a failure of the method: a rotation by 90 degrees
rotates every real direction, so there is no invariant line. Note that the invariants still hold
in the complex sense: `tr R = 0 = i + (−i)` ✓ and `det R = 1 = i·(−i)` ✓. Over the complex
numbers the rotation is diagonalisable after all, and the complex eigenvectors describe the
rotation as a pair of 90-degree phase advances.

**(c) Repeated eigenvalues that are perfectly fine.** Repetition alone is not the problem — the
multiplicity comparison is. Take

```
B = [[2, 0, 0],
     [0, 2, 1],
     [0, 0, 1]]
```

`B` is block upper triangular, so its characteristic polynomial is the product of the blocks'
polynomials: `(2 − λ)²(1 − λ)`. So `λ = 2` has algebraic multiplicity 2 and `λ = 1` has
algebraic multiplicity 1. Now the eigenspace of `λ = 2`:

```
B − 2I = [[0, 0, 0],
          [0, 0, 1],
          [0, 0, −1]]
```

which has rank 1, so its null space has dimension `3 − 1 = 2`. The geometric multiplicity is
therefore 2, equal to the algebraic multiplicity, and `B` **is** diagonalisable despite the
repeated eigenvalue.

**The criterion, stated once.** `A` is diagonalisable if and only if, for every eigenvalue, the
geometric multiplicity equals the algebraic multiplicity — equivalently, if there are `n`
linearly independent eigenvectors in total. A sufficient (not necessary) condition is that the
eigenvalues are all distinct: MML Theorem 4.12 proves that eigenvectors belonging to distinct
eigenvalues are linearly independent, which gives `n` of them and a basis.

### 2.6 What diagonalisation buys you: repeated application

Suppose `A` has `n` linearly independent eigenvectors, collected as the columns of `V`, with
eigenvalues on the diagonal of `Λ`. Then `AV = VΛ`, hence

$$
A = V\Lambda V^{-1}
\qquad\text{and, by induction,}\qquad
A^{k} = V \Lambda^{k} V^{-1}.
$$

The second identity is the useful one: applying a matrix `k` times is the same as expressing the
input in the eigenbasis, raising `n` numbers to the `k`-th power, and converting back. Powers of
matrices are what iterative methods compute, and this reduces them to powers of scalars.

Watch it happen with `A = [[2,1],[1,2]]` from section 2.3. Decompose `(2, 0)ᵀ` in the eigenbasis:
`(2, 0) = 1·(1, 1) + 1·(1, −1)`. Applying `A` repeatedly gives

```
Aᵏ (2, 0)ᵀ = 3ᵏ (1, 1)ᵀ + 1ᵏ (1, −1)ᵀ = (3ᵏ + 1, 3ᵏ − 1)ᵀ.
```

Because `3ᵏ` grows and `1ᵏ` does not, the direction of the result converges to `(1, 1)ᵀ`: the
dominant eigenvector. That is the mechanism behind power iteration for computing a dominant
eigenpair, behind the convergence of PageRank-style arguments in the book's further reading
(MML Section 4.8), and behind every "the system settles into its dominant mode" statement you
will meet later. The `k = 1` case is checked numerically in the code section below.

### 2.7 Symmetric matrices: why the rest of the chapter works

Real symmetric matrices are better behaved than arbitrary ones, and the difference is what makes
the next two lessons possible:

**Spectral theorem (MML Theorem 4.15).** A real symmetric matrix has only real eigenvalues and
has an orthonormal basis of eigenvectors. Equivalently,

$$
S = Q \Lambda Q^{\top}, \quad Q \ \text{orthogonal} \ (Q^{\top}Q = I), \quad \Lambda \ \text{real diagonal.}
$$

Two consequences worth stating explicitly, because they are the reason the SVD exists:

- No symmetric matrix is defective, so the awkward cases of section 2.5(a) cannot arise for the
  matrices that PCA and the SVD depend on.
- The eigendecomposition of a symmetric matrix is numerically pleasant to compute, and its
  inverse is `Q Λ⁻¹ Qᵀ` — no linear solve required.

Both facts are used immediately: `AᵀA` and `AAᵀ` are symmetric by construction, so the singular
values of `A` are the square roots of the eigenvalues of these two symmetric matrices. That is
the next lesson (`LSN-MATH-07-01`).

---

## Part 3 — Doing this with a computer

Hand computation builds the intuition and the checks. Real work is done in code, with different
risks. The code below was executed in the project environment; the output shown is the actual
output, recorded in `content/checks/` and summarised in the verification record for this lesson.

### 3.1 Compute, then verify — never just compute

```python
import numpy as np

A = np.array([[2.0, 1.0],
              [1.0, 2.0]])

w, V = np.linalg.eig(A)
print("eigenvalues:", w)              # [3. 1.]
print("eigenvectors:\n", V)

# The verification that matters: the residual of the defining equation.
for i in range(len(w)):
    v = V[:, i]
    residual = np.linalg.norm(A @ v - w[i] * v)
    print(f"lambda={w[i]:+.6f}  ||Av - lambda v|| = {residual:.3e}")

# The invariants, as an independent check on the decomposition itself.
print("trace:", np.trace(A), "sum of eigenvalues:", w.sum())
print("det:  ", np.linalg.det(A), "product of eigenvalues:", np.prod(w))
```

Three habits are worth taking from this snippet:

1. **Check the residual, not the eigenvector.** Eigenvectors are only defined up to scale, and
   library vectors are normalised to unit length; comparing vectors element by element against
   your hand answer is the wrong test. `||Av − λv||` should be around machine precision times
   the size of `A`, and if it is not, something is wrong with the eigenvalue.
2. **Use the invariants as a cheap global check.** Trace and determinant are independent of the
   decomposition, so agreement is real evidence.
3. **Do not assume an order.** The library returns eigenvalues in an unspecified order, and pairs
   complex eigenvalues into conjugate pairs. Code that indexes `w[0]` as "the largest eigenvalue"
   is wrong; use `np.argsort` explicitly.

A fourth habit, specific to non-symmetric matrices: computed eigenvalues of a non-symmetric
matrix can be much less accurate than the matrix entries suggest, especially when eigenvalues are
close together. The relevant quantity is the conditioning of the eigenproblem, which is not the
same as the conditioning of the matrix — this is one of the gaps tracked as `MATH-19`.

### 3.2 Check your hand work symbolically

For small matrices, symbolic computation is the right tool for checking a derivation you did by
hand, because it can factor the characteristic polynomial exactly:

```python
import sympy as sp

lam = sp.symbols("lambda")
M = sp.Matrix([[2, 1], [1, 2]])

char_poly = (M - lam * sp.eye(2)).det()
print(sp.factor(char_poly))          # (lambda - 3)*(lambda - 1)
print(sp.solve(char_poly, lam))      # [1, 3]

# Eigenspaces exactly, including the defect detection in section 2.5(a).
J = sp.Matrix([[1, 1], [0, 1]])
print(J.eigenvects())                # [(1, 2, [Matrix([[1],[0]])])]
```

That last line reports the eigenvalue `1`, its algebraic multiplicity `2`, and a list containing
**one** eigenvector: the printed pair of numbers is the defect made visible. This is the check to
run when you are unsure whether a repeated eigenvalue is the harmless kind or the defective kind.

### 3.3 The one-line summary of the diagonalisation test in code

```python
def is_diagonalisable(A: np.ndarray) -> bool:
    """Geometric multiplicity equals algebraic multiplicity for every eigenvalue."""
    w, V = np.linalg.eig(A)
    # count of independent eigenvectors the decomposition actually produced
    return V.shape[1] == A.shape[0] and np.linalg.matrix_rank(V) == A.shape[0]
```

This ranking test is a numerical proxy, not a proof: near-defective matrices produce nearly
dependent eigenvectors, and the rank computation has a tolerance. It is honest to say that this
function answers "did the decomposition produce a basis?" rather than "is the matrix
diagonalisable?", and that the two differ close to the boundary. Exercise 7 asks you to measure
how badly.

---

## Common misconceptions, with counterexamples

**1. `det(A + B) = det A + det B`.** False. In 2D take `A = I` and `B = −I`: `det A = 1`,
`det B = 1`, but `A + B = 0`, so `det(A + B) = 0`. The determinant is multiplicative, not
additive, and it is not linear in the matrix — only in each *column*, with the others held fixed.

**2. "An eigenvalue of zero means something is broken."** It means the matrix is singular, and
the corresponding eigenvector spans the null space. Zero eigenvalues are the normal case for
projection matrices and for covariance matrices of rank-deficient data.

**3. "`det A = 0` means the system has no solution."** Only for a *square* system with a nonzero
right-hand side. A homogeneous system `Ax = 0` always has the trivial solution, and `det A = 0`
means it has many more.

**4. "Repeated eigenvalues mean the matrix is not diagonalisable."** Section 2.5(c) is the
counterexample: `B` has eigenvalue 2 with algebraic multiplicity 2 and is still diagonalisable.

**5. "Eigenvectors are the important features of the data."** Eigenvectors are the invariant
directions of *a specific linear map that you chose*. Change the map — standardise the data
differently, use the covariance matrix instead of the Gram matrix — and you get different
eigenvectors. In the PCA lesson (`ML-07`) that choice is the subject of the lesson, and it
matters more than the eigen-decomposition itself.

**6. "Eigenvalues of `A` and of `A + cI` are related by scaling."** No: adding `cI` shifts every
eigenvalue by `c` and leaves the eigenvectors unchanged, because `(A + cI)v = Av + cv = (λ + c)v`.
This is worth knowing because it is how a shift is used to accelerate eigenvalue computation.

**7. Computing eigenvalues from the characteristic polynomial is a good algorithm.** It is not,
beyond tiny matrices: the roots of a polynomial can be far more sensitive to perturbations than
the eigenvalues of the matrix, and forming the polynomial coefficients can itself lose accuracy.
The library does not do this.

---

## Practice

Difficulty rises from recognition to open-ended work. Exercises marked `adapted` take their
*question* from the source but use instances chosen here, so that the book's own exercises remain
available for independent practice; the source's page and item number are recorded so you can
compare. Exercise statements, hints and full solutions are in
`content/solutions/` and are also listed in `knowledge/lessons/registry.json`.

| # | Focus | Difficulty | Origin |
| --- | --- | --- | --- |
| 1 | Determinant two ways, then interpret a nonzero result | recognition | original |
| 2 | Recognise why a determinant is zero from structure | guided | original |
| 3 | Eigenpairs and eigenspaces of two 2x2 matrices, with checks | independent | adapted (MML 4.3, p.139) |
| 4 | Invertibility versus diagonalisability, five matrices | multi-step | adapted (MML 4.5, p.138) |
| 5 | Prove the power rule for eigenpairs; break the converse | multi-step | original |
| 6 | Why an odd-dimensional real matrix must have a real eigenvalue | transfer | original |
| 7 | Implement 2x2 eigendecomposition and measure its residual | open-ended | original |

Two source exercises are deliberately left unsolved here, because they are good practice exactly
as written and doing them yourself is the point: **MML Exercise 4.2** (printed page 137) on
computing a determinant efficiently, where the intended skill is spotting exploitable structure
rather than expanding, and **MML Exercise 4.4** (printed page 137) on computing all eigenspaces
of a 4x4 matrix, where working with a larger instance is the point. Solutions to those two are
not provided anywhere in this platform, by design: with no published solutions in the source,
any answer we wrote would be ours rather than the authors', and it is more honest to let you
check your work against the invariants in section 2.4 and the substitution test in section 2.3.

### Solutions and what they establish

Every solution in this lesson states its method and the checks it passed. The solution files
record, for each exercise, (a) the reasoning, (b) the answer, (c) the verification that was
actually run, and (d) what that verification does not establish. Exercise 7's solution is code,
and its reported residual is the measured output of an execution, not an estimate.

---

## Connections

- **Backwards.** `MATH-01` (independence: the determinant detects it), `MATH-02` (matrices as
  mappings: the geometric reading of eigenvectors), `MATH-03` (the homogeneous-system fact that
  produces `det(A − λI) = 0`).
- **Forwards.** `LSN-MATH-07-01` uses the symmetric case (`S = QΛQᵀ`) to construct the SVD from
  `AᵀA` and `AAᵀ`, and reuses the determinant through multiplicativity of volume scaling. The PCA
  lesson (`ML-07`) is the eigendecomposition of a covariance matrix, with the standardisation
  decision as its real content. `MATH-17` uses the signs of the eigenvalues of a Hessian to
  classify stationary points, which is definiteness in the language of this lesson.
- **Sideways.** `MATH-19` (the gap): conditioning of eigenproblems, cost of decompositions, and
  why `det` is not a stability measure. The supplied book does not teach this analysis; a
  numerical-linear-algebra source is required, and the coverage report records that need.

## Further reading

- **Primary source.** MML Section 4.1 (printed pages 99-104) for the determinant as a volume
  scaling factor, Definition 4.4 for the determinant, Section 4.2 (printed pages 105-113) for
  eigenvalues, the characteristic polynomial, multiplicities and diagonalisation, with
  Definitions 4.5, 4.9, 4.10 and Theorems 4.12 and 4.15. Figure 4.7 in your copy shows the
  eigendecomposition as sequential transformations and is worth comparing against section 2.6
  above.
- **Your roadmap's visual companion.** Your own timetable maps determinant and basis-change
  intuition to chapters 6-7 of 3Blue1Brown's *Essence of Linear Algebra* (day 27 of your plan)
  and eigenvectors to chapter 14 (day 29). Those chapter numbers come from your roadmap; I have
  not verified them against the series in this session, so treat the mapping as a suggestion
  rather than a citation.
- **Not yet available.** For the numerical material flagged as `MATH-19` above, no source is in
  the repository. The coverage report lists it as a gap with an explicit statement of what is
  missing, rather than recommending something unverified.
