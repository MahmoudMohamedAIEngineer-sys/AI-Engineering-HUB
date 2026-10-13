---
id: math.linalg.matrices
book: mathematics-for-machine-learning
chapter: 2
section: 2.2.2
tag: must_know
prerequisites: [math.linalg.matrices]
leads_to: [math.linalg.linear-maps, math.linalg.eigen]
---
# Inverse and Transpose

## 1. What it is (plain-language definition, then the formal definition)

The transpose in plain language: flip the table across its main diagonal, so rows become columns and columns become rows. The inverse in plain language: the matrix that undoes another matrix — multiply by it and you are back where you started, as if nothing happened. [BOOK: mathematics-for-machine-learning, ch.2, §2.2.2]

Formally: for A in R^{m x n}, the **transpose** A^T in R^{n x m} is defined entrywise by (A^T)_ij = a_ji. [BOOK: mathematics-for-machine-learning, ch.2, §2.2.2] For a square matrix A in R^{n x n}, an **inverse** is a matrix B in R^{n x n} with AB = I_n = BA; it is denoted A^{-1}. If such a B exists, A is called regular / invertible / nonsingular; if not, A is singular / noninvertible, and when the inverse exists it is unique. A square matrix A with A = A^T is **symmetric**. [BOOK: mathematics-for-machine-learning, ch.2, §2.2.2] Two warnings are part of the definitions, not footnotes: not every square matrix has an inverse (existence is a property to check, not a given), and inversion does not distribute over addition — in general (A + B)^{-1} ≠ A^{-1} + B^{-1}, the matrix echo of 1/(2+4) ≠ 1/2 + 1/4. [BOOK: mathematics-for-machine-learning, ch.2, §2.2.2]

## 2. Why it exists (the problem it solves; what breaks without it)

The inverse exists because solving is undoing. The equation A x = b asks "which x did A produce b from?" If an undoing matrix A^{-1} exists, the answer is immediate: x = A^{-1} b, one multiplication. Every closed-form solution in this field — least squares via the inverse of X^T X, Gaussian conditioning formulas, the Kalman gain — is at heart an appeal to an inverse (modern practice computes the *solution* without literally forming the inverse, for numerical reasons this hub returns to, but the object being reasoned about is the inverse). When A is singular, undoing is impossible for a structural reason: A has collapsed some direction of information away (two different inputs map to the same output), and no operation can recover what was destroyed. Singularity is therefore not a technical annoyance; it is the precise statement "this transformation loses information", and §2.6's rank will measure exactly how much. Without the inverse concept there is no way to state when a system has a unique solution in matrix language: A x = b has exactly one solution for every b precisely when A^{-1} exists.

The transpose exists because the same data needs reading in both directions. A data matrix X (examples x features) sometimes must act as features x examples (X^T X accumulates feature-by-feature co-occurrence; X X^T accumulates example-by-example similarity). The transpose also manufactures symmetric matrices on demand: X^T X is always symmetric, and symmetric matrices own the friendliest theory in the book (Chapter 4's eigendecomposition guarantees). In calculus (Chapter 5), transposes appear every time a gradient changes between row and column layout. Without A^T, the product A^T A — arguably the most computed product in statistics — cannot even be written. [DERIVED: role analysis; X^T X symmetry proved in §4.]

## 3. Intuition (a physical or visual analogy)

Transpose is turning a class roster sideways: the spreadsheet that listed students in rows and subjects in columns becomes subjects in rows and students in columns. No number changes; what changes is which collection a "row read" hands you. This is why shape (m x n) becomes (n x m): the roster's width and height trade places. Inverse is a two-way door versus a one-way door. An invertible matrix is a two-way door between spaces: walk through with A, walk back with A^{-1}, end up home. A singular matrix is a door that leads into a room with no exit back — concretely, a projection that flattens 3D points onto a plane: infinitely many points along each vertical line land on the same image point, and from the flat image you cannot know the height. That picture explains uniqueness of the inverse too: if two different "ways back" both restored every point, apply one then... the proof in §4 makes it airtight, but the door picture already tells you there cannot be two distinct universal undo buttons. Symmetry then reads as: the roster looks the same after turning it sideways — entry (i,j) equals entry (j,i) — which is why adjacency matrices of undirected friendships and covariance matrices are symmetric: the relationship recorded is mutual by nature. [DERIVED analogies.]

## 4. The mechanics (every formula, every derivation step, every theorem and its proof or proof idea, step-by-step logic)

**The 2x2 inverse, derived, not memorized.** Let A = [[a,b],[c,d]] (renaming the book's a_11 etc.). Its determinant-to-be is the number ad − bc (Chapter 4 names it). Claim: if ad − bc ≠ 0 then A^{-1} = (1/(ad − bc)) [[d, −b],[−c, a]]; if ad − bc = 0, no inverse exists. Verification by multiplication, entry by entry: A times [[d,−b],[−c,a]] gives entry (1,1): a d + b (−c) = ad − bc; entry (1,2): a(−b) + b a = 0; entry (2,1): c d + d (−c) = 0; entry (2,2): c(−b) + d a = ad − bc. So the product is (ad − bc) I, and dividing by the nonzero scalar ad − bc yields I. The same computation run in the other order also gives (ad − bc) I, satisfying the two-sided requirement. Why the condition cannot be dropped: if ad − bc = 0, the displayed product is the zero matrix no matter what stands in the [[d,−b],[−c,a]] slot, and a short argument (if B were an inverse, then I = AB forces the rows of A to be proportional-free, contradicting ad = bc) shows nothing else can work either; Chapter 4's determinant machinery generalizes exactly this test. [BOOK: mathematics-for-machine-learning, ch.2, §2.2.2 for the statement and product; proof-completion DERIVED.]

**Uniqueness of the inverse (proof).** Suppose B and C both satisfy the inverse property for A: AB = BA = I and AC = CA = I. Then B = B I = B (A C) = (B A) C = I C = C. Equal. The middle step uses associativity (§2.2.1) — a first payoff of those laws: uniqueness of inverses is not obvious until you see that associativity is exactly the tool that forces it. [DERIVED from the book's definitions and §2.2.1.]

**Transpose laws, each by entry inspection.** (A^T)^T = A: transposing twice returns entry (i,j) to a_ij. (A + B)^T = A^T + B^T: entry (i,j) of the left is (A+B)_ji = a_ji + b_ji, matching the right. (AB)^T = B^T A^T — note the order flip: entry (i,j) of (AB)^T is (AB)_ji = sum_l a_jl b_li; entry (i,j) of B^T A^T is sum_l (B^T)_il (A^T)_lj = sum_l b_li a_jl, the same sum. The order flip is forced, and forgetting it is a classic error: (AB)^T ≠ A^T B^T in general. If A is invertible, so is A^T, with (A^{-1})^T = (A^T)^{-1}, written A^{-T}. **Symmetric matrices:** sums of symmetric matrices are symmetric ((A+B)^T = A+B), but products generally are not — the book's counterexample: [[1,0],[0,0]] [[1,1],[1,1]] = [[1,1],[0,0]], whose transpose [[1,0],[1,0]] differs. However A^T A is always symmetric: (A^T A)^T = A^T (A^T)^T = A^T A, using both transpose laws in one breath. That one-line fact guarantees covariance and Gram matrices are symmetric, which is what lets Chapter 4 apply its symmetric-matrix theory to them. [BOOK: mathematics-for-machine-learning, ch.2, §2.2.2 for laws and counterexample; proofs DERIVED by entries.]

## 5. Worked examples (full numbers, nothing skipped; redo the author's examples and add your own)

The book's inverse pair, audited. A = [[1,2,1],[4,4,5],[6,7,7]] and B = [[-7,-7,6],[2,1,-1],[4,5,-4]] are claimed mutually inverse. Check AB entry (1,1): 1*(-7) + 2*2 + 1*4 = -7 + 4 + 4 = 1. Entry (1,2): 1*(-7) + 2*1 + 1*5 = -7 + 2 + 5 = 0. Entry (2,1): 4*(-7) + 4*2 + 5*4 = -28 + 8 + 20 = 0. Entry (3,3): 6*6 + 7*(-1) + 7*(-4) = 36 - 7 - 28 = 1. The pattern (1s on diagonal, 0s off) continues for the remaining entries — all nine are computed by the §6 code, which prints the full identity matrix, completing the audit rather than trusting the claim. [BOOK: mathematics-for-machine-learning, ch.2, §2.2.2, audited by DERIVED code.]

2x2 formula in action: A = [[2,1],[5,3]], ad − bc = 6 − 5 = 1, so A^{-1} = [[3,−1],[−5,2]]. Check: A A^{-1} entry (1,1) = 2*3 + 1*(−5) = 1; (1,2) = 2*(−1) + 1*2 = 0; (2,1) = 5*3 + 3*(−5) = 0; (2,2) = 5*(−1)+3*2 = 1. Identity confirmed. A singular specimen: A = [[1,2],[2,4]] has ad − bc = 4 − 4 = 0; indeed row 2 is twice row 1, the map collapses the plane onto a line, and no inverse exists — connecting singularity back to §2.1's Fate 1/Fate 3 geometry. Transpose example: A = [[1,2,3],[4,5,6]] (2x3) transposes to the 3x2 matrix [[1,4],[2,5],[3,6]]; entry (2,3) of A, the number 6, lands at position (3,2) of A^T, exactly as (A^T)_ij = a_ji dictates. [DERIVED.]

## 6. Code (runnable, with saved output)

```python
def matmul(A, B):
    return [[sum(A[i][l] * B[l][j] for l in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]

def transpose(A):
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]

A = [[1, 2, 1], [4, 4, 5], [6, 7, 7]]
B = [[-7, -7, 6], [2, 1, -1], [4, 5, -4]]
print(matmul(A, B))
print(transpose([[1, 2, 3], [4, 5, 6]]))
print(matmul([[2, 1], [5, 3]], [[3, -1], [-5, 2]]))
```
```text
[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
[[1, 4], [2, 5], [3, 6]]
[[1, 0], [0, 1]]
```

## 7. Where it shows up in AI/ML (concrete uses)

X^T X and its inverse define ordinary least squares (theta = (X^T X)^{-1} X^T y when the inverse exists) and ridge regression exists precisely to repair the singular case (X^T X + lambda I is invertible). Attention uses K^T inside QK^T. Gradients of products trigger the (AB)^T = B^T A^T order flip in every backpropagation derivation (Chapter 5). Symmetric matrices built as A^T A underpin PCA's covariance eigendecomposition (Chapter 10) and kernel Gram matrices (Chapter 12), and the failure of X^T X to be invertible is the practical face of multicollinearity — features that duplicate each other, §2.1's redundancy wearing statistical clothes. [DERIVED applications tied to the book's later chapters.]

## 8. Common mistakes and misconceptions

(1) Computing (AB)^T as A^T B^T without flipping the order. (2) Believing every square matrix is invertible; proportional rows already kill invertibility. (3) Believing (A + B)^{-1} = A^{-1} + B^{-1} — the book's scalar counterexample 1/(2+4) ≠ 1/2 + 1/4 kills the pattern. (4) Assuming a product of symmetric matrices is symmetric (counterexample in §4). (5) Solving A x = b by explicitly forming A^{-1} in code when a direct solver is stabler and cheaper — the inverse is a reasoning tool first, a computational last resort. (6) Forgetting that symmetry requires square shape.

## 9. Interview angle (how it is asked, what a strong answer includes)

Asked as: "When is a matrix invertible, and how do you check?" Strong answer: square and A^{-1} exists iff the determinant is nonzero (Chapter 4), equivalently iff A x = b has a unique solution for every b, equivalently iff no row is a combination of others; give the 2x2 formula and a singular example like [[1,2],[2,4]]. A second frequent probe: "Is X^T X always symmetric / invertible?" — symmetric yes (one-line proof), invertible only when X has independent columns; that distinction earns the point.

## 10. Visualization spec (what should be drawn or interactive, with the parameters the user can change)

Two panels. (a) Transpose flipper: hover a cell in A and watch its mirror cell light in A^T as the grid rotates around the diagonal. (b) Invertibility explorer: a 2x2 matrix whose columns are draggable arrows; as the arrows approach parallel, ad − bc approaches 0, the transformed unit square flattens toward a line, and the "inverse exists" indicator switches off at the singular limit. User parameters: the four entries / arrow endpoints.

## 11. Sources and verification (provenance tags and proof file paths)

[BOOK: mathematics-for-machine-learning, ch.2, §2.2.2] Definitions 2.3 (inverse), 2.4 (transpose), 2.5 (symmetric), the 2x2 formula, the book's inverse pair and symmetric-product counterexample, and all stated laws. [DERIVED]: uniqueness proof, entry-level law proofs, singularity narrative, and numeric audits — machine-checked by §6 code via scripts/code_check.py.
