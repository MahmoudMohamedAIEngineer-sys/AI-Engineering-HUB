---
id: math.linalg.matrices
book: mathematics-for-machine-learning
chapter: 2
section: 2.3.4
tag: must_know
prerequisites: [math.linalg.matrices]
leads_to: [math.linalg.eigen]
---
# Computing the Inverse by Elimination, and How Systems Are Solved in Practice

## 1. What it is (plain-language definition, then the formal definition)

In plain language: to invert a matrix, set up the question "which matrix X undoes A?" as n ordinary systems (one per column of the identity), and solve them all at once by running Gaussian elimination on the augmented matrix [A | I] until the left half becomes I; the right half is then A^{-1}. And when the goal is not the inverse itself but one solution of A x = b, practice usually avoids forming A^{-1} at all — direct elimination or iterative methods are cheaper and numerically safer. [BOOK: mathematics-for-machine-learning, ch.2, §2.3.4 and "Calculating the Inverse" in §2.3]

Formally: for A ∈ R^{n×n}, the inverse is the matrix X with A X = I_n, X = [x_1 | ... | x_n] by columns. Column j of this matrix equation is the system A x_j = e_j, so computing A^{-1} *is* solving n systems with the same coefficient matrix. Because elementary row operations act identically on all right-hand sides carried along, elimination on [A | I_n] reaching [I_n | X] certifies X = A^{-1} — *provided* the left half can be reduced to I_n, which works exactly when A is invertible; a pivot failure (a zero column where a pivot is needed) is the elimination-level proof that A is singular. The book also records the closed form via the **Moore–Penrose pseudo-inverse**: for A with linearly independent columns, A x = b (when consistent, and in least-squares form generally — Chapter 9) is solved by x = (AᵀA)^{-1} Aᵀ b, where (AᵀA)^{-1} Aᵀ is the pseudo-inverse; the same paragraph warns that forming inverses or pseudo-inverses explicitly is generally not recommended for numerical precision and costs many matrix–matrix computations. [BOOK: mathematics-for-machine-learning, ch.2, §2.3.4]

The algorithmic landscape the section sketches: Gaussian elimination costs on the order of n³ operations — fine for thousands of unknowns, impractical for millions; large-scale practice therefore uses iterative methods of the form x^{(k+1)} = C x^{(k)} + d (stationary methods such as Richardson, Jacobi, Gauss–Seidel, successive over-relaxation, or Krylov methods such as conjugate gradients), designed so the residual error ‖x^{(k+1)} − x*‖ shrinks every iteration. [BOOK: mathematics-for-machine-learning, ch.2, §2.3.4]

## 2. Why it exists (the problem it solves; what breaks without it)

The 2×2 formula from §2.2.2 gives no path to a 4×4 or 400×400 inverse, and "solve A x = b" is the most frequently needed computation in applied mathematics — something must scale. This section's first contribution is the reduction: inversion is not a new problem, it is n copies of the solving problem you already own, batched through one elimination. Its second contribution is judgment about what *not* to compute. The mathematically clean formula x = A^{-1} b tempts beginners to invert matrices in code; the book explicitly steers away: computing A^{-1} does strictly more work than solving for the one b you have (it answers for all e_j right-hand sides whether you need them or not), and floating-point formation of an inverse amplifies rounding error compared with solving the system directly. Likewise the pseudo-inverse formula (AᵀA)^{-1} Aᵀ b, while theoretically central (it is the least-squares solution of Chapter 9 and the minimum-norm choice among Fate-3 families), squares the condition trouble by forming AᵀA and then inverts it — fine for reasoning, avoided in production code. What breaks without this judgment: correct-looking code that is slow, memory-hungry, and mysteriously inaccurate on ill-conditioned data, plus the false belief that non-square systems "cannot be solved" — they can, in the least-squares/minimum-norm sense the pseudo-inverse names. [BOOK: mathematics-for-machine-learning, ch.2, §2.3.4; the reasoning expansion is DERIVED.]

## 3. Intuition (a physical or visual analogy)

Inverting by elimination is like figuring out a combination lock's inverse operations by filming yourself locking it: apply operations to A until it becomes I (the unlocked state); the same operations, applied to I, spell out the unlocking sequence as a matrix. Do the operations on both halves simultaneously and the film and the recipe are the same object, [A | I] → [I | A^{-1}]. The cost argument has a household version: x = A^{-1} b by "first compute the inverse" is like printing a full conversion table for every conceivable amount of money when you need to convert exactly one amount — the table (inverse) is only worth building if many conversions (many b's) will reuse it. Iterative methods are the opposite philosophy: instead of solving exactly in one long procedure, start with a guess and keep polishing it, like adjusting a shower's temperature by feel — each iteration is cheap, and you stop when the residual is small enough for the purpose. For a million-variable system, "cheap and repeatable" beats "exact but cubic". [DERIVED analogies.]

## 4. The mechanics (every formula, every derivation step, every theorem and its proof or proof idea, step-by-step logic)

**Why [A | I] → [I | A^{-1}] works (proof idea).** Elementary row operations on a matrix are equivalent to left-multiplication by invertible "operation matrices", and applying a sequence of them that turns A into I means the combined operation matrix, call it E, satisfies E A = I — so E is a left inverse of A, and for square matrices a left inverse is the inverse (its two-sidedness was part of Definition 2.3; the elimination here constructs it). Applying the identical operations to the right half I produces E I = E = A^{-1} on the right side of the bar. That is the whole justification: the right half accumulates the recorded operations, and the record *is* the inverse. Conversely, if elimination stalls — a column with no available nonzero pivot — then A cannot reach I, mirroring §2.2.2: a singular A (for instance one with proportional rows) has no inverse to find. [DERIVED justification of the book's procedure (2.56).]

**The book's Example 2.9, stated and audited.** A = [[1,0,2,0],[1,1,0,0],[1,2,0,1],[1,1,1,1]]. Elimination of [A | I_4] yields the claimed inverse A^{-1} = [[-1,2,-2,2],[1,-1,2,-2],[1,-1,1,-1],[-1,0,-1,2]]. Audit (one entry by hand, the rest by code): (A A^{-1})_{11} = 1·(-1) + 0·1 + 2·1 + 0·(-1) = 1. (A A^{-1})_{12} = 1·2 + 0·(-1) + 2·(-1) + 0·0 = 0. The code below multiplies the full product and obtains I_4 exactly in integer arithmetic. [BOOK: mathematics-for-machine-learning, ch.2, §2.3, Example 2.9; audit DERIVED.]

**The pseudo-inverse formula and its place.** If A's columns are linearly independent, AᵀA is invertible (its invertibility fails exactly when columns are dependent — a thread §2.5/§2.6 will make precise), and x̂ = (AᵀA)^{-1} Aᵀ b satisfies the normal equations AᵀA x̂ = Aᵀ b. When b lies in the column span, x̂ solves A x = b; otherwise x̂ is the least-squares solution (Chapter 9 proves optimality via projection, §3.8/§9.4), and among least-squares solutions this formula returns the minimum-norm one. The book's warning stands: existence of the formula is a reasoning tool; computing it literally (form AᵀA, invert it, multiply) is discouraged numerically in favor of factorization- or iteration-based solvers. [BOOK: mathematics-for-machine-learning, ch.2, §2.3.4; interpretation forward-references the book's Ch. 9.]

**Cost accounting.** Dense Gaussian elimination on n×n: on the order of n³ operations (the book's statement) — for n = 10³, ~10⁹ (seconds-scale territory); n = 10⁶, ~10¹⁸ (impractical). Iterative methods replace one cubic procedure by a sequence of matrix–vector products whose per-step cost follows the matrix's size (and sparsity), converging when the iteration matrix C is chosen so errors contract; the book names Richardson, Jacobi, Gauss–Seidel, successive over-relaxation, and the Krylov family (conjugate gradients, GMRES, biconjugate gradients) as the standard shelf, with details deferred to its cited references. [BOOK: mathematics-for-machine-learning, ch.2, §2.3.4]

## 5. Worked examples (full numbers, nothing skipped; redo the author's examples and add your own)

Example 2.9 is audited above: two entries by hand ((1,1) = 1, (1,2) = 0) and the complete A A^{-1} = I_4 by the code in §6, in exact integer arithmetic. [BOOK example; DERIVED audit.]

A complete small inversion by elimination, every step shown. Invert A = [[2, 1], [5, 3]] (whose determinant ad - bc = 1, so §2.2.2 gives A^{-1} = [[3, -1], [-5, 2]] to check against). Augmented: [2 1 | 1 0; 5 3 | 0 1]. Step 1: row1 /= 2: [1 1/2 | 1/2 0; 5 3 | 0 1]. Step 2: row2 -= 5·row1: [1 1/2 | 1/2 0; 0 1/2 | -5/2 1]. Step 3: row2 *= 2: [0 1 | -5 2] in row 2. Step 4: row1 -= (1/2)·row2: [1 0 | 3 -1; 0 1 | -5 2]. Left half is I_2, right half reads [[3, -1], [-5, 2]] — exactly the §2.2.2 formula's answer, obtained mechanically. [DERIVED worked example confirming BOOK §2.2.2/§2.3 agreement.]

Pseudo-inverse sanity check in miniature: A = [[1], [1]] (2×1, one independent column), b = (3, 5). No exact solution exists (x would need to be both 3 and 5). AᵀA = [2], (AᵀA)^{-1} Aᵀ = (1/2)[1, 1], so x̂ = (3+5)/2 = 4 — the average, hence the least-squares answer: errors (-1, +1), squared-error sum 2, and any other x does worse (at x = 3: errors (0,-2), sum 4). The formula does the right thing in the inconsistent case, which is its real workplace. [DERIVED]

## 6. Code (runnable, with saved output)

```python
def matmul(A, B):
    return [[sum(A[i][l] * B[l][j] for l in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]

A = [[1,0,2,0],[1,1,0,0],[1,2,0,1],[1,1,1,1]]
Ainv = [[-1,2,-2,2],[1,-1,2,-2],[1,-1,1,-1],[-1,0,-1,2]]
print(matmul(A, Ainv))
# Gauss-Jordan inverse for [[2,1],[5,3]]
from fractions import Fraction as F
M = [[F(2),F(1),F(1),F(0)],[F(5),F(3),F(0),F(1)]]
def rref(M):
    r = 0
    for c in range(2):
        p = next(i for i in range(r, 2) if M[i][c] != 0)
        M[r], M[p] = M[p], M[r]
        pv = M[r][c]; M[r] = [v/pv for v in M[r]]
        for i in range(2):
            if i != r and M[i][c] != 0:
                f = M[i][c]; M[i] = [v - f*w for v, w in zip(M[i], M[r])]
        r += 1
    return M
for row in rref(M): print([str(v) for v in row])
```
```text
[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
['1', '0', '3', '-1']
['0', '1', '-5', '2']
```

## 7. Where it shows up in AI/ML (concrete uses)

Nobody inverts matrices in production ML code, and that is this section's lesson operating as intended: NumPy's `solve`, PyTorch's triangular solvers, and Kalman/least-squares implementations all run elimination-family factorizations (LU, Cholesky, QR) that embody [A | I] thinking without materializing A^{-1}. Closed-form learning algorithms (ridge regression's (XᵀX + λI)^{-1} Xᵀ y) are *stated* with an inverse and *computed* with a solver; Gaussian-process regression and Bayesian linear regression (Ch. 9) are built on such solves. At LLM scale, only the iterative philosophy survives, generalized: conjugate gradients solves the linear systems inside second-order optimization and Gaussian-process approximations, and first-order training itself is an iteration x^{(k+1)} = x^{(k)} - λ∇L in the x^{(k+1)} = C x^{(k)} + d spirit. Recognizing "an inverse in a formula means a solve in code" prevents a whole class of slow, inaccurate implementations. [DERIVED applications.]

## 8. Common mistakes and misconceptions

(1) Computing A^{-1} explicitly to solve one system — more work, worse numerics; solve instead. (2) Forming AᵀA to "help": it squares the sensitivity to rounding and can turn a solvable problem into a numerically singular one; the formula is for reasoning and small exact work. (3) Assuming elimination failing means your arithmetic slipped: a missing pivot can be the correct report that A is singular. (4) Believing non-square systems are unsolvable tout court: the pseudo-inverse gives the least-squares/minimum-norm solution, the workhorse of Chapter 9. (5) Inverting to "check" invertibility: attempting the reduction *is* the check. (6) Quoting O(n³) as a reason to fear modest systems: thousands of unknowns are routine; the cubic wall matters at millions, where iterative/sparse methods take over.

## 9. Interview angle (how it is asked, what a strong answer includes)

Asked as: "How would you solve A x = b in code?" A strong answer refuses the bait "multiply by the inverse": describe elimination/factorization (solve, never invert), state the O(n³) cost and when it bites, name the least-squares/pseudo-inverse route for inconsistent or underdetermined systems while noting you would compute it via QR/SVD rather than forming (AᵀA)^{-1}, and mention iterative methods (conjugate gradients) for very large sparse cases. The sentence "an inverse in a formula means a solve in code" lands the practical point this section exists to teach.

## 10. Visualization spec (what should be drawn or interactive, with parameters the user can change)

Side-by-side elimination film: [A | I] grid where each row operation highlights simultaneously on both halves, with a playhead stepping through Example 2.9 until [I | A^{-1}] appears and the closing multiplication A A^{-1} = I animates entry by entry. A second control panel compares strategies as n grows: a log-scale cost curve for cubic elimination vs per-iteration cost of an iterative method, with a slider for n from 10 to 10⁶. User parameters: chosen example (2×2 by hand system or Example 2.9), playback speed, n slider.

## 11. Sources and verification (provenance tags and proof file paths)

[BOOK: mathematics-for-machine-learning, ch.2, §2.3 "Calculating the Inverse"] for the [A | I] method (2.56) and Example 2.9 (2.57)–(2.58); [BOOK: mathematics-for-machine-learning, ch.2, §2.3.4] for the pseudo-inverse formula (2.59), the numerical warning, the cubic-cost statement, and the iterative method taxonomy (2.60). [DERIVED]: justification proof idea, audits, the 2×2 elimination, and the mini least-squares check — machine-verified by §6 code via scripts/code_check.py.
