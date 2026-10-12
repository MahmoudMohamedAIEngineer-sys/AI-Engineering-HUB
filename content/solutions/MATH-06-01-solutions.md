# Solutions — Determinants, eigenvalues and eigenvectors (`LSN-MATH-06-01`)

Every solution below states four things: the reasoning, the answer, **the verification that was
actually run**, and what that verification does not establish. The checks live in
`content/checks/math06_eigen_checks.py` and their recorded output is in
`knowledge/lessons/checks-report.json`.

Where a solution quotes a number, that number came out of a computation, not out of the prose.

---

## EX-LSN-MATH-06-01-01 — determinant two ways

<a id="ex-01"></a>

**Method.** Sarrus applies to 3x3 matrices only: three "down-right" products minus three
"down-left" products. Laplace expansion along the first row multiplies each entry by the
determinant of the 2x2 minor obtained by deleting its row and column, with alternating signs.

With `B = [[2, 0, 3], [1, 4, -1], [0, 2, 1]]`:

```
Sarrus, down-right:   2·4·1 + 0·(−1)·0 + 3·1·2 = 8 + 0 + 6 = 14
Sarrus, down-left:    3·4·0 + 0·1·1 + 2·(−1)·2 = 0 + 0 − 4 = −4
det B = 14 − (−4) = 18

Laplace along row 1 (signs +, −, +):
  2 · det[[4, −1], [2, 1]]  = 2 · (4·1 − (−1)·2)  = 2 · 6  = 12
− 0 · det[[1, −1], [0, 1]]  = 0
+ 3 · det[[1, 4], [0, 2]]   = 3 · (1·2 − 4·0)     = 3 · 2  =  6
det B = 12 + 0 + 6 = 18
```

**Answer.** `det B = 18`, from both expansions. Since `18 ≠ 0`, `B` is invertible.

**Verified.** Three independent methods agree: the Sarrus formula evaluated step by step, the
Laplace expansion, `sympy.Matrix.det()` (exact rational arithmetic) and `numpy.linalg.det`
(floating point), the last returning `18.0`.

**Not established by that.** Agreement between two algorithms that both compute a determinant
does not by itself tell you the determinant is *meaningful* for the decision you are making.
`det B = 18` says `B` is invertible; it does not say `B` is well conditioned. A matrix with
`det = 10⁻¹⁰` is technically invertible and numerically close to useless. That distinction is
tracked as a separate competency (`MATH-19`) because the supplied book does not teach it.

---

## EX-LSN-MATH-06-01-02 — why a determinant is zero, from structure

<a id="ex-02"></a>

**Method.** The determinant is alternating: swapping two rows changes its sign, so a determinant
with two equal rows equals its own negative and is therefore zero. More generally, any two
columns being equal makes it zero, and by multilinearity any column that is a combination of the
others does too.

**Answer, part 1.** For `[[1, 2, 3], [4, 5, 6], [1, 2, 3]]`, rows 1 and 3 are identical. The
alternating property forces `det = 0`. No expansion is needed and none should be done.

**Answer, part 2.** One valid construction with a *different* reason: a 4x4 matrix whose third row
is exactly twice its first row,

```
[[1, 2, 3, 4],
 [0, 1, 0, 1],
 [2, 4, 6, 8],     ← = 2 × row 1, so the rows are dependent
 [1, 0, 1, 0]]
```

Here no two rows are equal, so the alternating property alone does not apply; the reason is
**dependence**: one row lies in the span of the others, the rank is at most 3, and the volume the
mapping spans is therefore zero. A third, structurally different reason would be a column of
zeros (the image is contained in a hyperplane).

**Verified.** All three structures were checked: the determinants are exactly zero
(`sympy.Matrix.det()`), and the ranks are 2, 3 and 2 respectively out of 3, 4 and 3 — confirming
in each case that the matrix is singular for the reason claimed.

**Not established by that.** Spotting a dependency is a *sufficient* way to prove a determinant is
zero; failing to spot one is not evidence that the determinant is nonzero. When structure is not
visible, compute it.

---

## EX-LSN-MATH-06-01-03 — eigenvalues and eigenspaces

<a id="ex-03"></a>

**Method.** Form `det(A − λI) = 0`, factor, then for each root solve the homogeneous system
`(A − λI)v = 0` by reading off the dependent rows. Finish with the two invariants and a
substitution check.

### `A = [[3, 1], [1, 3]]`

```
det(A − λI) = (3 − λ)² − 1 = λ² − 6λ + 8 = (λ − 4)(λ − 2)     → λ = 4, 2

λ = 4:  A − 4I = [[−1, 1], [1, −1]]  →  v₁ = v₂   → E₄ = span{(1, 1)}
λ = 2:  A − 2I = [[ 1, 1], [1,  1]]  →  v₁ = −v₂  → E₂ = span{(1, −1)}
```

Checks: `tr A = 6 = 4 + 2` ✓, `det A = 9 − 1 = 8 = 4 · 2` ✓, and by substitution
`A(1,1)ᵀ = (4,4)ᵀ = 4(1,1)ᵀ`, `A(1,−1)ᵀ = (2,−2)ᵀ = 2(1,−1)ᵀ` ✓.

### `C = [[2, 2], [1, 3]]`

```
det(C − λI) = (2 − λ)(3 − λ) − 2 = λ² − 5λ + 4 = (λ − 4)(λ − 1)   → λ = 4, 1

λ = 4:  C − 4I = [[−2, 2], [1, −1]]  →  v₁ = v₂   → E₄ = span{(1, 1)}
λ = 1:  C −  I = [[ 1, 2], [1,  2]]  →  v₁ = −2v₂ → E₁ = span{(2, −1)}
```

Checks: `tr C = 5 = 4 + 1` ✓, `det C = 6 − 2 = 4 = 4 · 1` ✓, and by substitution
`C(1,1)ᵀ = (4,4)ᵀ`, `C(2,−1)ᵀ = (2,−1)ᵀ = 1·(2,−1)ᵀ` ✓.

For `C`, notice that the eigenspace for `λ = 1` is `span{(2, −1)}` — not `span{(1, −1)}`, which
would be the answer for a symmetric matrix. Reading the eigenspace off the "shape" of the matrix
instead of solving the system is the most common error here.

**Verified.** The characteristic polynomials were factored exactly and the eigenpairs substituted
back symbolically (`A v − λ v = 0` as an exact expression, not a floating-point comparison).

**Not established by that.** Exact symbolic verification confirms the algebra. It does not confirm
that you can *do* this reliably by hand under time pressure, which is what the local curriculum
target for this competency asks for; that is what repeated practice is for.

---

## EX-LSN-MATH-06-01-04 — invertible versus diagonalisable

<a id="ex-04"></a>

**Method.** Invertibility is `det ≠ 0`. Diagonalisability is decided by comparing, for every
eigenvalue, the **algebraic multiplicity** (how often the root appears in the characteristic
polynomial) with the **geometric multiplicity** (the dimension of the eigenspace, i.e.
`n − rank(A − λI)`). Equality for all eigenvalues means diagonalisable. A matrix with a non-real
eigenvalue cannot be diagonalised over the real numbers even if it is diagonalisable over the
complex numbers.

| # | Matrix | det | Invertible | Diagonalisable over ℝ | Competing multiplicities |
| --- | --- | --- | --- | --- | --- |
| 1 | `[[1, 1], [0, 1]]` shear | 1 | **yes** | **no** | `λ = 1`: algebraic 2, geometric 1 |
| 2 | `[[0, 1], [0, 0]]` nilpotent | 0 | no | no | `λ = 0`: algebraic 2, geometric 1 |
| 3 | `[[2, 0], [0, 2]]` scaled identity | 4 | yes | yes | `λ = 2`: algebraic 2, geometric 2 |
| 4 | `[[0, −1], [1, 0]]` rotation by 90° | 1 | **yes** | **no** (over ℂ: yes) | `λ = ±i`, each algebraic 1, geometric 1 |
| 5 | `[[1, 0], [0, 0]]` projection | 0 | no | yes | `λ = 0` and `λ = 1`, each algebraic 1, geometric 1 |

**Reasoning for the two rows that matter.**

- Row 1: `det = 1 ≠ 0` so it is invertible, yet `A − I = [[0, 1], [0, 0]]` has rank 1, giving a
  one-dimensional eigenspace against algebraic multiplicity 2. Defective, so not diagonalisable.
  **Invertibility and diagonalisability are independent properties.**
- Row 4: invertible (a rotation preserves area), and it has two distinct complex eigenvalues, so
  it is diagonalisable over ℂ. Over ℝ it has no eigenvalues at all, so no real eigenbasis exists.
  This is the case that separates the two fields and the reason the phrase "diagonalisable" needs
  a field attached to it.

**Verified.** `sympy` computed the determinant, the characteristic polynomial, the algebraic
multiplicity, the rank of `A − λI` (hence the geometric multiplicity) and diagonalisability over
the algebraic closure, for all five matrices. The verdicts above are the recorded output.

**Not established by that.** Rows 1 and 2 are *numerically* delicate: a matrix that is defective in
exact arithmetic has eigenvalues that are not differentiable functions of its entries. If you
perturb `[[1, 1], [0, 1]]` by `10⁻¹⁰` the two eigenvalues separate and the two eigenvectors become
almost parallel rather than identical. Symbolic verification cannot see that; a floating-point
experiment can, and this is why exercise 7 asks for measured residuals.

---

## EX-LSN-MATH-06-01-05 — the power rule, and its failed converse

<a id="ex-05"></a>

**Claim 1.** If `Av = λv` with `v ≠ 0`, then `Aᵏv = λᵏv` for all `k ≥ 1`.

*Proof by induction.* Base case `k = 1` is the hypothesis. Inductive step: assume `Aᵏv = λᵏv`.
Then

```
A^(k+1) v = A (Aᵏ v) = A (λᵏ v) = λᵏ (A v) = λᵏ (λ v) = λ^(k+1) v,
```

using linearity of `A` for the third equality. Since `v ≠ 0` by hypothesis, the resulting
eigenvector is legitimate. ∎

**Claim 2.** The converse fails over ℝ: there is a real matrix `A` such that `A²` has a real
eigenvalue that is not the square of any real eigenvalue of `A`.

**Counterexample.** Take the rotation `R = [[0, −1], [1, 0]]`. Then

```
R² = [[−1, 0], [0, −1]] = −I,
```

whose only eigenvalue is `−1` with multiplicity 2. But `R` has characteristic polynomial
`λ² + 1`, so its eigenvalues are `±i` and it has **no real eigenvalues at all**. There is
therefore no real `λ` of `R` with `λ² = −1`.

**The important qualification.** Over ℂ the converse does hold here: `R` has eigenvalues `i` and
`−i`, and `(±i)² = −1`. So the failure is not a failure of the power rule but of restricting
attention to the reals — a concrete illustration of why eigenvalues are defined over ℂ and why
"this matrix has no eigenvalues" is really "it has none in the field we are working over".

**Verified.** The power rule was checked exactly for `A = [[2, 1], [1, 2]]`, `v = (1,1)ᵀ` and
`k ∈ {1, 2, 3, 5, 8}`: `Aᵏv − 3ᵏv = 0` as an exact expression each time. `R² = −I`, the
eigenvalues of `R²` (`{−1}`) and of `R` (`{i, −i}`) were computed symbolically.

**Not established by that.** Checking finitely many values of `k` is not the proof; the induction
above is. The check tests the claim the proof makes, which is what a check is for.

---

## EX-LSN-MATH-06-01-06 — one real eigenvalue in odd dimension

<a id="ex-06"></a>

**Claim.** Every real 3x3 matrix has at least one real eigenvalue.

**Reasoning.** The characteristic polynomial `p(λ) = det(A − λI)` is a real polynomial of degree 3.
For large positive `λ` the leading term dominates with a sign determined by `−λ³`, so `p(λ) → −∞`;
for large negative `λ`, `p(λ) → +∞`. A polynomial that is continuous and takes both signs must
have a real root (the intermediate value theorem). Every real root of the characteristic
polynomial is a real eigenvalue — and note that the argument never needed the entries of `A`, only
the degree.

The same argument works in every odd dimension and fails in even dimensions, because a polynomial
of even degree can take the same sign at both ends.

**Example with exactly one real eigenvalue.**

```
A = [[0, −1, 0],
     [1,  0, 0],
     [0,  0, 2]]
```

This is a rotation in the `(x₁, x₂)`-plane together with a stretch in the `x₃`-direction. Its
characteristic polynomial factors as

```
det(A − λI) = −(λ − 2)(λ² + 1) = (2 − λ)(λ² + 1),
```

whose only real root is `λ = 2`; the remaining eigenvalues are the complex pair `±i`, produced by
the rotation block. So exactly one real eigenvalue, as required.

**The 2D contrast.** In dimension 2 the argument cannot be made, and the rotation
`[[0, −1], [1, 0]]` is the witness: `λ² + 1` has no real root. An even-dimensional matrix can have
no real eigenvalues at all.

**Verified.** The characteristic polynomial of the 3x3 example was factored exactly and its roots
solved for: exactly one real root, `λ = 2`, the other two being `±i`. The 2x2 rotation's
polynomial `λ² + 1` was confirmed to have no real roots.

**Not established by that.** The intermediate value theorem argument is a statement about real
polynomials; its applicability here depends on the characteristic polynomial having real
coefficients, which follows from `A` having real entries. If `A` were complex, nothing in this
exercise would apply.

---

## EX-LSN-MATH-06-01-07 — reference implementation and its residual

<a id="ex-07"></a>

**Method.** For a real 2x2 matrix, the characteristic polynomial is
`λ² − (tr A)λ + det A`, so `λ = (tr A ± √(tr²A − 4 det A))/2`; the square root may be negative, in
which case the eigenvalues are complex and the implementation must not drop them. An eigenvector
for a computed `λ` is any nonzero vector in the null space of `A − λI`, which in two dimensions
can be read off directly from the rows of the shifted matrix.

**Reference implementation.** See `content/checks/math06_eigen_checks.py`,
`check_ex07_reference_implementation` — the function `eigen_2x2` is the reference solution, and
running the check executes it.

**Measured results** (recorded output, `knowledge/lessons/checks-report.json`):

| Quantity | Measured |
| --- | --- |
| Matrices tested | 1000 random 2x2, fixed seed `7`, standard normal entries |
| Maximum `‖Av − λv‖` over all eigenpairs | `3.803e-15` |
| Maximum `‖Av − λv‖` among matrices with complex eigenvalues | `8.680e-16` |
| Maximum disagreement with `numpy.linalg.eigvals` | below `1e-8` (asserted as a bound) |
| Residual for the nearly defective `[[1, 1], [0, 1+10⁻¹⁰]]` | `2.500e-21` |

**Interpretation.** The residual is the right test, because eigenvectors are only defined up to
scale and library eigenvectors are normalised: comparing components directly is meaningless.
Residuals around `10⁻¹⁵` on unit-scale data are what floating-point arithmetic allows.

The nearly defective case is the interesting one. Its residual is *smaller* than the random
average, even though the matrix is the worst-behaved in the set: the eigenvalues are within
`10⁻¹⁰` of each other, so the two eigenvectors are almost parallel, and a small residual tells you
only that each returned pair solves its own equation. It says nothing about whether the
eigenvectors are a useful basis. This is why the answer must state what the test does *not*
establish:

- agreement with `numpy` establishes **consistency between two implementations**;
- the residual establishes that **each returned pair solves `Av = λv`** to that precision;
- neither establishes that the computation is **stable** for nearly defective matrices, and
  neither is a proof of correctness in exact arithmetic.

That last gap is what a full treatment of conditioning would fill (`MATH-19`, currently without a
source in this repository — see the coverage report).

**What would earn full credit on the rubric.** Handling the complex case rather than assuming real
eigenvalues; measuring the residual rather than asserting it; exhibiting a concrete
disagreement-adjacent case (near-repeated eigenvalues) with an explanation; and stating the limits
of the verification honestly, as above.

---

## Source exercises left unsolved, deliberately

MML exercises 4.2 and 4.4 (printed page 137) are not solved anywhere in this platform. Reasons,
stated plainly:

1. The supplied source contains **no solutions** to its own exercises, so any answer written here
   would be ours, not the authors', and it would be dishonest to present it differently without a
   separate verification of the exercise's intent.
2. Both are good practice exactly as written: 4.2 teaches you to exploit structure instead of
   expanding, and 4.4 asks you to compute every eigenspace of a 4x4 matrix, where the arithmetic
   is the point.
3. You can check your own work on them completely: the substitution test (section 2.3, step 4) and
   the trace/determinant invariants (section 2.4) detect any arithmetic error, and
   `sympy.eigenvects` will settle an argument.

The matrices used in exercises 3 and 4 above were therefore chosen to be different from the
book's, so the service of providing worked examples does not accidentally give away the practice.
