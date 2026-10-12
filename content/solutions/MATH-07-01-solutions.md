# Solutions — Matrix decompositions (`LSN-MATH-07-01`)

Each solution states the method, the answer, the verification that was actually run, and what that
verification does not establish. The checks are in `content/checks/math07_svd_checks.py`; their
recorded output is `knowledge/lessons/checks-report.json`.

---

## EX-LSN-MATH-07-01-01 — Cholesky by hand

<a id="ex-01"></a>

**Method.** Apply the column formulas in order. For column `j`: first the diagonal entry
`L_jj = sqrt(A_jj - Σ_{k<j} L_jk²)`, then the entries below it
`L_ij = (A_ij - Σ_{k<j} L_ik L_jk) / L_jj`.

With `A = [[9, 6, 3], [6, 5, 2], [3, 2, 2]]`:

```
column 1:  L11 = sqrt(9) = 3
           L21 = 6/3 = 2            L31 = 3/3 = 1
column 2:  L22 = sqrt(5 - 2²) = 1
           L32 = (2 - 1·2)/1 = 0
column 3:  L33 = sqrt(2 - 1² - 0²) = 1
```

**Answer.**

```
L = [[3, 0, 0],
     [2, 1, 0],
     [1, 0, 1]]
```

**Verification by multiplication** (do this every time — it is four multiplications):

```
LL' = [[3,0,0],[2,1,0],[1,0,1]] · [[3,2,1],[0,1,0],[0,0,1]]
    = [[9,6,3],[6,5,2],[3,2,2]] = A   ✓
```

**The pivot question.** The relevant question is not "which entry would be imaginary" but "which
pivot stops being positive". For this matrix, the third pivot is
`A₃₃ - L₃₁² - L₃₂² = 2 - 1 - 0 = 1`, so `L₃₃ = 1`.

- Lower `A₃₃` from 2 to **1**: the third pivot becomes `1 - 1 - 0 = 0`, so `L₃₃ = 0`. The matrix is
  then positive *semi*-definite (its determinant is 0) and the factorisation exists but has a zero
  on the diagonal, which violates the strict positivity that Theorem 4.18 requires.
- Lower it to **0**: the third pivot becomes `-1`, and the formula demands the square root of a
  negative number. The determinant is then `-9`, so the matrix is indefinite and no real
  factorisation exists.

That is the whole content of the existence condition: a symmetric factorisation with a real
positive diagonal exists exactly when every pivot stays positive, and the pivots are the leading
principal minors in disguise (Sylvester's criterion).

**Verified.** `LL' = A` exactly in symbolic arithmetic; the leading principal minors of `A` are
`9, 9, 9`; `sympy` confirms positive definiteness; the perturbed matrices were confirmed to be
non-definite with determinants `0` and `-9`; `numpy.linalg.cholesky` was confirmed to raise
`LinAlgError` on a non-positive-definite matrix.

**Not established by that.** The hand computation is verified for *this* matrix. Numerical Cholesky
implementations differ in pivoting and in what they do with near-zero pivots; the algorithm's
behaviour on badly conditioned input belongs to the recorded `MATH-19` gap.

---

## EX-LSN-MATH-07-01-02 — full SVD of a rectangular matrix

<a id="ex-02"></a>

**Method.** The construction of section 3.2, in order: form `A'A`, find its eigenvalues and an
orthonormal eigenbasis, set `σᵢ = √λᵢ`, compute `uᵢ = Avᵢ/σᵢ`, then extend the `uᵢ` to an
orthonormal basis of the left space.

With `A = [[1, 0], [0, 1], [1, 1]]` (3x2):

**Step 1.** `A'A = [[2, 1], [1, 2]]`.

**Step 2.** `det(A'A - λI) = (2-λ)² - 1 = (λ-3)(λ-1)`, so `λ₁ = 3`, `λ₂ = 1`, hence
`σ₁ = √3`, `σ₂ = 1`.

**Step 3.** Eigenvector for `λ₁ = 3`: `(2-3)v₁ + v₂ = 0` gives `v₁ = (1,1)'/√2`. For `λ₂ = 1`:
`v₁ + v₂ = 0` gives `v₂ = (1,-1)'/√2`.

**Step 4.**

```
u₁ = A v₁ / σ₁ = (1/√2, 1/√2, 2/√2)' / √3 = (1, 1, 2)' / √6
u₂ = A v₂ / σ₂ = (1/√2, -1/√2, 0)' / 1   = (1, -1, 0)' / √2
```

**Step 5, completing `U`.** `u₁` and `u₂` span a plane in `R³`; the third direction must be
orthogonal to both. Solve `u₁'x = 0` and `u₂'x = 0`:

```
x₁ + x₂ + 2x₃ = 0   and   x₁ - x₂ = 0   →   x = (1, 1, -1)/√3
```

**Answer.**

```
U = [[1/√6,  1/√2,  1/√3],
     [1/√6, -1/√2,  1/√3],
     [2/√6,      0, -1/√3]]

Σ = [[√3, 0],
     [0,  1],
     [0,  0]]          V = (1/√2)[[1, 1], [1, -1]]
```

**Verification.**

```
U Σ V' = A exactly (symbolic arithmetic)
U'U = I₃  (U is orthogonal)
numpy.linalg.svd returns σ = (√3, 1) to 1e-12
```

**Answer to the last question.** No singular value is zero, and the reason is that the two columns
of `A` are linearly independent: `σᵢ = 0` would mean `Avᵢ = 0` for a unit vector `vᵢ`, which would
make the columns dependent. The number of nonzero singular values equals the rank, here 2 = min(3,2).

**Not established by that.** The `U` computed here is one of many valid choices: `(uᵢ, vᵢ)` can be
flipped together, and the third column of `U` could be any unit vector orthogonal to the first two
corresponding to a zero row of `Σ`. Comparing your `U` element by element with a library's is
therefore not a valid test; compare `Σ` and `UΣV'` instead.

---

## EX-LSN-MATH-07-01-03 — existence conditions, each violated

<a id="ex-03"></a>

| Decomposition | Condition | A matrix that violates it | Why it fails |
| --- | --- | --- | --- |
| Cholesky `A = LL'` | symmetric **and** positive definite | `[[1, 2], [2, 1]]` | symmetric, eigenvalues `3` and `-1` — not positive definite, so a real pivot turns negative |
| Eigendecomposition `A = PDP^{-1}` | square and diagonalisable | `[[1, 1], [0, 1]]` | defective: eigenvalue 1 has algebraic multiplicity 2 and geometric multiplicity 1 |
| SVD `A = UΣV'` | none | — | no such matrix exists; the theorem has no hypothesis beyond real entries |

**What the failing computations actually do** (executed, not assumed):

- `numpy.linalg.cholesky([[1,2],[2,1]])` raises `LinAlgError: Matrix is not positive definite`.
  The routine does not return a complex factor even though `L` with complex entries would satisfy
  the equations; the failure is how the routine reports the violated hypothesis.
- `numpy.linalg.eig([[1,1],[0,1]])` returns eigenvalues `[1, 1]` and an eigenvector matrix of
  **rank 1**: both returned columns are the same direction (up to numerical noise). The routine
  does not raise. This is the dangerous failure mode — a decomposition that silently does not
  exist, because the library always returns `n` vectors and it is your job to notice that they are
  not independent.
- The SVD was computed for 15 matrices of shapes (3,5), (5,3), (4,4), (6,2), (2,6) including
  rank-deficient and all-zero matrices; the maximum reconstruction error was `0.00e+00` to double
  precision. There is no failing case to report, which is the point.

**Not established by that.** "No matrix lacks an SVD" is a theorem (4.22), and 15 matrices are a
check of the implementation, not a proof. The proof is the derivation in section 3.2, which never
uses anything about `A` beyond its existence.

---

## EX-LSN-MATH-07-01-04 — the two norm identities

<a id="ex-04"></a>

**Claim 1.** `‖A‖_F² = Σᵢ σᵢ²` (the sum runs over all `min(m,n)` singular values).

*Proof.* Write the outer-product form `A = Σᵢ σᵢ uᵢvᵢ'`. The Frobenius norm is
`‖A‖_F² = tr(A'A)`, and

```
A'A = (Σ σᵢ vᵢuᵢ')(Σ σⱼ uⱼvⱼ') = Σᵢ Σⱼ σᵢσⱼ vᵢ (uᵢ'uⱼ) vⱼ' = Σᵢ σᵢ² vᵢvᵢ',
```

because `uᵢ'uⱼ = δᵢⱼ`. Taking traces and using `tr(vᵢvᵢ') = ‖vᵢ‖² = 1` gives
`tr(A'A) = Σᵢ σᵢ²`. ∎

The same identity follows from `‖A‖_F² = Σλᵢ(A'A)`, since the `σᵢ²` *are* the eigenvalues of `A'A`.

**Claim 2.** `‖Ax‖ ≤ σ₁‖x‖` for every `x`, with equality at `x = v₁`.

*Proof.* Expand `x = Σᵢ cᵢvᵢ` (the `vᵢ` are an orthonormal basis of the input space). Then
`Ax = Σᵢ cᵢσᵢuᵢ`, and because the `uᵢ` are orthonormal,

```
‖Ax‖² = Σᵢ cᵢ²σᵢ² ≤ σ₁² Σᵢ cᵢ² = σ₁²‖x‖².
```

Taking square roots gives the bound; at `x = v₁` the inequality is an equality because
`Av₁ = σ₁u₁` and `‖u₁‖ = 1`. ∎

**Why this names the norm.** Since `σ₁` is the maximum of `‖Ax‖/‖x‖`, it is the operator norm of
the mapping — the worst-case stretch factor, which is what "spectral norm" means (Definition 4.23
and Theorem 4.24 in the source). The Frobenius norm, by contrast, is a norm on the *array* of
entries: it is not the norm induced by a vector norm, and that distinction is exactly why the two
behave differently in the next lesson's approximation theorem.

**Verified.** Over 200 random matrices (sizes 2-9) and 800 random vectors: the maximum deviation in
the Frobenius identity was `3.6e-13`, the maximum violation of the bound was `0.0`, and the bound
was attained at `v₁` to `4.5e-16`.

**Not established by that.** The checks sample matrices; the proofs above are what establish the
claims. The numerical agreement confirms that both were implemented and stated correctly.

---

## EX-LSN-MATH-07-01-05 — why the zero eigenvalues differ

<a id="ex-05"></a>

**The identity.** `ker(A'A) = ker(A)`, and therefore `ker(AA') = ker(A')`.

*Proof.* If `Ax = 0` then `A'Ax = A'(Ax) = 0`, so `ker(A) ⊆ ker(A'A)`. Conversely, if `A'Ax = 0`,
then `0 = x'A'Ax = ‖Ax‖²`, and a vector of norm zero is zero, so `Ax = 0` and
`ker(A'A) ⊆ ker(A)`. ∎

The identity is the reason the derivation works at all: the eigenvectors of `A'A` for zero
eigenvalues are exactly the null space directions of `A`.

**The example.** Take the tall matrix

```
A = [[1, 0],
     [0, 1],
     [0, 0]]        (3x2, rank 2)
```

Then `A'A = I₂`, which has **no** zero eigenvalues, while `AA' = diag(1, 1, 0)` has a zero
eigenvalue of multiplicity 1.

**The counting rule.** The dimensions come out of the rank-nullity theorem applied to each side:

```
dim ker(A'A) = n - rank(A'A) = n - rank A      (since rank(A'A) = rank A)
dim ker(AA') = m - rank(AA') = m - rank A
```

So the two null spaces contain the same `rank A` directions in a formal sense, but they live in
spaces of different dimensions, and their zero eigenvalues have multiplicities `n - rank A` and
`m - rank A`. For the example: `2 - 2 = 0` and `3 - 2 = 1`.

**Verified.** `A'A = I₂` and `AA' = diag(1,1,0)` computed exactly; eigenvalue dictionaries
`{1: 2}` and `{1: 2, 0: 1}`; the null-space dimension equality `dim ker(A'A) = dim ker(A)` checked
symbolically on random integer matrices of shapes (3,2), (2,3) and (4,2).

**Not established by that.** The source's own exercise (4.11, printed p.138) asks you to prove that
the *nonzero* eigenvalues coincide. That is deliberately left for you: the proof is a two-line
computation of `A'A v = λv ⇒ AA'(Av) = λ(Av)`, together with the observation that `Av ≠ 0` whenever
`λ ≠ 0`, and doing it yourself is the point.

---

## EX-LSN-MATH-07-01-06 — two square roots, and all of them

<a id="ex-06"></a>

**Setup.** Take `A = [[2, 1], [1, 2]]`, which is symmetric with eigenvalues `3` and `1`, hence
positive definite.

**The symmetric square root.** An orthonormal eigenbasis is `q₁ = (1,1)'/√2` (for `λ = 3`) and
`q₂ = (1,-1)'/√2` (for `λ = 1`), so

```
S = Q Λ^{1/2} Q' = [[1/2 + √3/2, -1/2 + √3/2],
                    [-1/2 + √3/2, 1/2 + √3/2]]
```

**The Cholesky factor.**

```
L = [[√2,        0     ],
     [1/√2, √(3/2)     ]]
```

**Both are roots, and they differ.** `SS' = A` and `LL' = A` (both verified exactly). `S` is
symmetric with no zero entries; `L` is lower triangular. Neither is "the" square root.

**All square roots.** Let `A = QΛQ'` with `Q` orthogonal. For any square root `Z` (meaning
`ZZ' = A`), put `Y = Q'Z`. Then

```
YY' = Q'ZZ'Q = Q'AQ = Λ.
```

The diagonal entries of `YY' = Λ` say that the rows of `Y` are mutually orthogonal with norms
`√λᵢ`; dividing each row by `√λᵢ` produces a matrix with orthonormal rows, which is `W'` for some
orthogonal `W`. Hence `Y = Λ^{1/2}W'` and therefore

```
Z = Q Λ^{1/2} W'  with W orthogonal.
```

With `L` in the role of `Z`, solving gives `W' = Λ^{-1/2}Q'L`, i.e.

```
W = [[-1/2, √3/2],
     [√3/2,  1/2]]
```

which is orthogonal (it is a reflection), and `QΛ^{1/2}W' = L` **exactly**. So the family of square
roots is parameterised by the orthogonal group, and the Cholesky factor is the member that happens
to be triangular with a positive diagonal.

**Which one should a sampling routine use?** The triangular one. To generate `x = μ + Sz` with
`z ~ N(0, I)` you need one matrix-vector product either way, but the construction of `S` matters:
Cholesky costs about `n³/6` operations, while the spectral route costs an eigendecomposition plus a
matrix product. Since the result has the same distribution — the covariance is `SS' = A` in both
cases — the cheaper factor is the right one. (A subtlety worth knowing: the two square roots give
*different* samples for the same `z`, and both are correct draws from `N(μ, A)`. The distribution
does not determine the transform.)

**Verified.** Both factors verified exactly against `A`; `Q` verified orthogonal; the extracted `W`
verified orthogonal and `QΛ^{1/2}W' = L` verified to be exactly zero differences; the reflected
factor `S·diag(1,-1)` confirmed to be a third square root that is neither symmetric nor triangular.

**Not established by that.** The uniqueness of the *triangular* factor (that only one `L` with
positive diagonal works) was asserted rather than proved here; the forward-substitution argument is
the proof, and it is short enough to write out — do that if you want the exercise's full value.

---

## EX-LSN-MATH-07-01-07 — implementing the textbook SVD

<a id="ex-07"></a>

**Reference implementation.**

```python
def svd_via_gram(matrix):
    """Singular values from the eigendecomposition of A'A (the construction of section 3.2)."""
    gram = matrix.T @ matrix
    values = np.linalg.eigvalsh(gram)        # symmetric: real, ascending
    return np.sqrt(np.maximum(values, 0.0))[::-1]
```

**Measured results** (recorded output of `check_ex07_gram_svd_accuracy`):

| Experiment | Result |
| --- | --- |
| 100 random matrices, sizes 3-11 | largest relative difference in singular values: **4.3e-13** |
| 40x40, singular values 1 down to 1e-8: library | smallest σ = 1.000e-08, relative error 4.7e-10 |
| 40x40, same matrix: `A'A` route | smallest σ = 1.080e-08, relative error **8.0e-02** |

And the sweep over the ratio `σ₁/σ_n`:

| ratio | relative error, `A'A` route |
| --- | --- |
| 1e2 | 1.2e-13 |
| 1e4 | 2.5e-09 |
| 1e6 | 1.9e-06 |
| 1e8 | 8.0e-02 |
| 1e10 | 1.0 |
| 1e12 | 1.0 |
| 1e16 | 1.0 |

**Interpretation, and what the test does and does not establish.**

- The construction is **mathematically correct**: on well-conditioned inputs it reproduces the
  singular values to round-off, which is the strongest agreement one can ask for.
- Forming `A'A` **squares** the ratio of extreme singular values, because `σ(A'A) = σ(A)²`. At a
  ratio of `1e10` the smallest singular value of `A'A` is below the round-off level of the largest,
  so no digits survive — and that value is exactly the one that decides the numerical rank.
- Agreement with `numpy` on easy problems is therefore evidence about the *mathematics*, not about
  the *algorithm*. `numpy` does not use this construction, and the lesson's section 5.2 says why.
- The rubric's fourth point: the honest conclusion is that the attempt was worth making — it proves
  the derivation — and will never be used in production code for anything ill-conditioned.

**The reference solution is the check itself.** `content/checks/math07_svd_checks.py`,
`check_ex07_gram_svd_accuracy`, contains the function and runs both experiments; there is no
separate copy of the code to drift out of date.

---

## Source exercises left unsolved, deliberately

**MML 4.8, 4.9** (full SVDs, printed p.138), **4.11** (equal nonzero eigenvalues of `A'A` and
`AA'`) and **4.12** (the spectral-norm formula for `x ≠ 0`) are left as practice. The reasons are
the same as in `LSN-MATH-06-01`: the source publishes no solutions, so any answer written here
would be ours rather than the authors'; each of them is good practice exactly as written; and each
can be checked independently — `numpy.linalg.svd` settles a computed SVD, and the identities in
exercise 4 above settle 4.12.

**MML 4.10** (the rank-1 approximation) belongs to the next lesson and is left for there.
