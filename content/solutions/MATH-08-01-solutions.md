# Solutions — Low-rank approximation (`LSN-MATH-08-01`)

Method, answer, verification actually run, and what the verification does not establish. The checks
are in `content/checks/math08_lowrank_checks.py`; recorded output in
`knowledge/lessons/checks-report.json`.

---

## EX-LSN-MATH-08-01-01 — the best rank-1 approximation

<a id="ex-01"></a>

**Method.** The best rank-1 approximation is the first term of the SVD, `σ₁u₁v₁'`. So compute the
SVD, keep one term, and measure what is left using the identities
`‖A − Â(1)‖₂ = σ₂` and `‖A − Â(1)‖_F² = σ₂²`.

With `A = [[3, 0], [4, 5]]`:

```
A'A = [[9+16, 20], [20, 25]] = [[25, 20], [20, 25]]
det(A'A - λI) = (25-λ)² - 400 = λ² - 50λ + 225 = (λ - 45)(λ - 5)
σ₁ = √45 = 3√5 ≈ 6.7082        σ₂ = √5 ≈ 2.2361

λ₁ = 45:  -20v₁ + 20v₂ = 0  →  v₁ = (1, 1)'/√2
u₁ = A v₁ / σ₁ = (3, 9)'/(√2 · 3√5) = (1, 3)'/√10
```

**Answer.**

```
Â(1) = σ₁ u₁ v₁' = 3√5 · (1,3)'(1,1) / √20 = (3/2)·[[1, 1],
                                                    [3, 3]]
```

**The residual, and both norms.**

```
C = A - Â(1) = [[3 - 3/2,  0 - 3/2],      = (1/2)·[[3, -3],
                [4 - 9/2,  5 - 9/2]]              [-1, 1]]

C'C = (1/4)·[[10, -10], [-10, 10]] = (5/2)·[[1, -1], [-1, 1]]
det(C'C - λI) = λ(λ - 5)   →   singular values of C are (√5, 0)
```

So `‖C‖₂ = √5 = σ₂` and `‖C‖_F² = 2·(9/4 + 1/4)·... = 5 = σ₂²`. Both match the theorem, because the
residual of a rank-1 truncation *is* the second singular term.

**Consistency checks.** `tr A = 8`, `det A = 15`. The eigenvalues of `A` are the roots of
`λ² - 8λ + 15 = (λ-3)(λ-5)`, so `{3, 5}`; then `σ₁σ₂ = 3√5·√5 = 15 = |det A|` ✓ and
`σ₁² + σ₂² = 45 + 5 = 50 = tr(A'A) = 9 + 16 + 25 + 20 - 20... = 3² + 0² + 4² + 5² = 50` ✓.

**Interesting: the singular values are not the modulus of the eigenvalues here.** `A` has real
eigenvalues `3, 5`, yet `σ = (6.708, 2.236)`. The relation `σᵢ = |λᵢ|` holds only for symmetric
matrices, and `A` is not symmetric. This is worth noticing precisely because the eigenvalues happen
to be real — reality of the spectrum does not give you the singular values.

**Verified.** The SVD, the rank-1 term, the residual, and the characteristic polynomial `λ(λ-5)` of
`C'C` were computed exactly in symbolic arithmetic; `det A = σ₁σ₂` and `tr(A'A) = Σσᵢ²` were
confirmed.

**Not established by that.** That this rank-1 matrix is *optimal* is the Eckart-Young theorem; the
check confirms the value it attains. Exercise 2 is where the optimality claim is dealt with, and it
is proved there, not assumed.

---

## EX-LSN-MATH-08-01-02 — proving the optimality bound

<a id="ex-02"></a>

**Claim.** If `rk(B) ≤ k` then `‖A − B‖₂ ≥ σ_{k+1}`.

**Proof.** Three facts do the work: the null space of a rank-`k` matrix is large, the span of the
first `k+1` right-singular vectors is small but non-trivial, and two subspaces of a finite-dimensional
space must intersect when their dimensions are large enough.

1. `dim ker B = n − rk(B) ≥ n − k`.
2. `V = span{v₁, ..., v_{k+1}}` has dimension `k+1`.
3. `dim(ker B ∩ V) ≥ dim ker B + dim V − n ≥ (n − k) + (k + 1) − n = 1`. Pick a unit vector `x` in
   the intersection, so `Bx = 0` and `x = Σ_{i=1}^{k+1} cᵢvᵢ` with `Σcᵢ² = 1`.
4. `Ax = Σ cᵢσᵢuᵢ`, so `‖Ax‖² = Σ cᵢ²σᵢ² ≥ σ_{k+1}² Σ cᵢ² = σ_{k+1}²`, since `σᵢ ≥ σ_{k+1}` for
   `i ≤ k+1` and the `uᵢ` are orthonormal.
5. Therefore `‖A − B‖₂ ≥ ‖(A − B)x‖ = ‖Ax‖ ≥ σ_{k+1}`. ∎

**Why the bound is attained.** `Â(k)` has `rk ≤ k` and `A − Â(k) = Σ_{i>k}σᵢuᵢvᵢ'`, whose largest
singular value is `σ_{k+1}`; so `‖A − Â(k)‖₂ = σ_{k+1}` exactly. Combined with the claim, `Â(k)` is
a minimiser.

**A sharper consequence (worth keeping).** The same argument applied at every index gives the
interlacing inequality `σ_{k+i}(A) ≤ σᵢ(A − B)`. That is stronger than the single bound and is what
one uses when studying perturbation of singular values.

**Verified.** For 80 random matrices of shapes up to 8x7 with `k` from 1 to 3: the truncation's
spectral error equalled `σ_{k+1}` to 1e-10 every time, and 200 random rank-`k` candidates per matrix
never produced a smaller error. The smallest margin found above `σ_{k+1}` was recorded in the report.

**Not established by that.** Random search cannot prove a lower bound — it can only fail to
disprove one. The proof above is what establishes it; the experiment is a check that the
implementation and the statement agree, and it would have caught a wrong index (`k` versus `k+1`)
immediately, which is the most common error in this theorem.

---

## EX-LSN-MATH-08-01-03 — the demonstration matrix and the error identities

<a id="ex-03"></a>

**Method.** Read the stored SVD from `content/demos/rank-truncation.json`, reconstruct at each `k`
as `(U[:, :k] * s[:k]) @ Vt[:k, :]`, and compare the measured error with the closed form.

**Recorded and recomputed figures.**

| k | relative Frobenius error | energy retained | spectral error |
| --- | --- | --- | --- |
| 1 | 0.4213 | 0.822464 | 2.0376 = σ₂ |
| 2 | 0.3091 | 0.904461 | 1.3520 = σ₃ |
| 3 | 0.2438 | 0.940561 | 0.9011 = σ₄ |
| 5 | 0.1669 | 0.972136 | 0.8730 = σ₆ |
| 8 | 0.0555 | 0.996916 | 0.3301 = σ₉ |
| 12 | 0.0048 | 0.999977 | 0.0298 = σ₁₃ |

**The two relations to take away.**

```
relative error = sqrt(1 - energy)      (exactly, by the definitions in the lesson)
spectral error = σ_(k+1)               (exactly, by the structure of the truncation)
```

The first is why "99% energy" and "1% error" are different requirements: `sqrt(1 - 0.99) ≈ 0.1`.

**Verified.** The stored figures were recomputed from the stored factors and agree to 3.2e-8, which
is the six-decimal storage precision of the file — not machine precision, and the check says so.
The spectral error was confirmed to equal the next stored singular value in every case.

**Not established by that.** The stored factors are a genuine SVD of the stored matrix, which is
checked separately (`check_gram_matrix_and_demo_integrity`), including U and V orthogonality to
1.3e-6 — again limited by storage precision rather than by the decomposition.

---

## EX-LSN-MATH-08-01-04 — the storage break-even rule

<a id="ex-04"></a>

**Derivation.** The factors of a rank-`k` representation store `k` columns of `U`, `k` rows of `V'`,
and `k` singular values: `k(m + n + 1)` numbers against `mn` for the full matrix. The representation
is smaller exactly when

```
k(m + n + 1) < mn        ⇔        k < mn / (m + n + 1).
```

**For a 1432 x 1910 matrix** (the size of the image matrix in the source's Figure 4.11 discussion,
cited rather than reproduced):

```
mn = 2,735,120                m + n + 1 = 3,343
largest compressing k = floor(2,735,120 / 3,343) = 818
storage ratio at k = 50:  50 · 3343 / 2735120 = 0.061
storage ratio at k = 200: 200 · 3343 / 2735120 = 0.244
```

So the break-even rank for that shape is 818 — a large fraction of the 1432 possible values, which is
why image compression by SVD is described as "keep a few hundred components" and works at exactly the
scales where the spectrum decays fast.

**For the 24x24 demonstration matrix**: `mn = 576`, `m + n + 1 = 49`, break-even `k = 11`. At
`k = 12` the representation stores 588 numbers, more than the matrix. The demo file's recorded
ratios agree with the rule to 1e-4.

**Verified.** The break-even formula was derived symbolically (`k = mn/(m+n+1)` as the unique
solution), the arithmetic was computed exactly, and the ratios in the demonstration file were checked
against the rule.

**Not established by that.** The rule assumes each stored number costs the same. Real formats use
different widths, headers and per-array metadata, so `k(m+n+1)/mn` is an upper bound on the saving,
not a measured file size. The interactive component says the same thing next to the numbers it
displays.

---

## EX-LSN-MATH-08-01-05 — numerical rank is a decision

<a id="ex-05"></a>

**Construction.** A 6x6 matrix with singular values `[1, 1e-1, 1e-2, 1e-3, 1e-8, 1e-14]`, built as
`L diag(s) R'` with `L`, `R` orthogonal.

**Measured ranks at three thresholds** (relative to `σ₁`):

| threshold | reported rank | induced relative Frobenius error |
| --- | --- | --- |
| 1e-3 | 3 | 9.9e-04 |
| 1e-8 | 4 | 9.9e-09 |
| 1e-12 | 5 | 1.4e-14 |

`numpy.linalg.matrix_rank` with its default threshold (`max(m,n)·ε·σ₁ ≈ 1.3e-15`) reports 6, the
mathematical rank.

**One paragraph, as asked.** When the small singular values are tiny rather than exactly zero, the
matrix is not rank-deficient in exact arithmetic and *is* effectively rank-deficient for any purpose
that has an error budget. The honest formulation is: the matrix is within distance `σ_{k+1}` (in the
spectral norm, by the theorem of exercise 2) of a matrix of rank `k`, so discarding the tail costs at
most that much in the worst direction and `Σ_{i>k}σᵢ²` in total squared error. The rank you report
should therefore be the one whose induced error you can defend, and the number to report alongside it
is that error — which is precisely why the table above has three rows rather than one answer. Calling
a matrix "rank 5" without saying "at a threshold of 1e-12, with a relative error of 1.4e-14" is
stating a convention as if it were a fact.

**Verified.** The three ranks and their induced errors were computed; the default-threshold rank was
confirmed to be 6.

**Not established by that.** No threshold is claimed to be correct. The lesson's coverage report
records that the supplied source does not discuss threshold selection, so this is recorded as a gap
(`MATH-19`) rather than presented as best practice.

---

## EX-LSN-MATH-08-01-06 — two criteria for choosing k

<a id="ex-06"></a>

**Results on the demonstration matrix.**

| criterion | smallest k | measured outcome | storage ratio |
| --- | --- | --- | --- |
| retain 99% of the energy | **7** | energy 99.4689%, relative error 0.0729 | 60% of 576 entries |
| relative Frobenius error below 1% | **11** | energy 99.9953%, relative error 0.0068 | 94% of 576 entries |

The two criteria differ by a factor of more than three in `k`, and the second one saves almost
nothing in storage.

**The argument.** For a downstream linear model I would use the error criterion, not the energy
criterion, and I would tune `k` by cross-validation rather than by either closed-form rule. Reasons,
in order of importance:

1. **Neither criterion knows the task.** The retained components are chosen to minimise
   reconstruction error, which is not the same objective as prediction error. A direction that
   contributes little variance can be the direction that predicts the target — the classic case being
   a binary outcome whose class separation lies in a low-variance direction.
2. **Neither criterion knows what is noise.** Discarding a component with a small singular value
   discards whatever signal it carried, and there is no guarantee in the theorem that small means
   noisy. The energy criterion is more permissive about this in appearance only: it also discards a
   tail, just a shorter one.
3. **The energy criterion is scale-blind in the wrong way.** Energy is dominated by the largest
   components, so it will happily accept a `k` at which a component with real predictive content has
   been dropped.
4. **What the error criterion buys.** A bound on total squared reconstruction error, which is at
   least a quantity you can state and defend. That is why it is a better starting point — not because
   1% is a magic number.

**What both criteria share, and why it matters.** Both are *unsupervised*: they never look at the
response variable. Anything chosen this way has to be validated against the task afterwards, and the
validation must not reuse the data that chose `k`. If the SVD is computed on the full dataset before
splitting, the test set has influenced the features and every reported metric is optimistic.

**Verified.** Both criteria were evaluated on the recorded SVD; the `k` values, energies, relative
errors and storage ratios above are the recorded output of
`check_ex06_selection_criteria`, not estimates.

**Not established by that.** The claim that cross-validation beats a closed-form rule on an unseen
task is a statement about practice, not something these checks establish; it is the kind of claim
that needs a dataset and an experiment, and this lesson has neither. The argument above is reasoning,
and it is labelled as such.
