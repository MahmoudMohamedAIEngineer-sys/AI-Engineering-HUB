---
id: math.linalg.matrices
book: mathematics-for-machine-learning
chapter: 2
section: 2.2.1
tag: must_know
prerequisites: [math.linalg.matrices]
leads_to: [math.linalg.linear-maps]
---
# Matrix Addition and Multiplication

## 1. What it is (plain-language definition, then the formal definition)

Addition in plain language: line two tables of the same shape up and add corresponding cells. Multiplication in plain language: each entry of the result is a whole row of the left matrix dotted against a whole column of the right matrix — multiply matching entries and add them up. [BOOK: mathematics-for-machine-learning, ch.2, §2.2.1]

Formally: for A, B in R^{m x n}, the sum A + B is the matrix with (A + B)_ij = a_ij + b_ij, defined only when the shapes agree. For A in R^{m x n} and B in R^{n x k}, the product C = AB in R^{m x k} has entries c_ij = sum over l = 1..n of a_il * b_lj. [BOOK: mathematics-for-machine-learning, ch.2, §2.2.1] Read c_ij slowly: fix row i of A and column j of B, walk along them together (l is the walking index), multiply pairs, accumulate. The inner dimensions must match (A's column count n equals B's row count n) because the walk needs both sequences to have the same length; the outer dimensions survive into the result's shape (m x k). This section also defines the **identity matrix** I_n in R^{n x n}: 1s on the main diagonal, 0s elsewhere, and records the algebra: associativity (AB)C = A(BC), distributivity (A+B)C = AC + BC and A(C+D) = AC + AD, and identity IA = AI = A with shape-appropriate identity matrices on each side. [BOOK: mathematics-for-machine-learning, ch.2, §2.2.1]

## 2. Why it exists (the problem it solves; what breaks without it)

Addition exists because models combine like-shaped tables constantly: adding a bias is broadcast addition, a residual connection adds a layer's input to its output, and gradient accumulation adds per-batch gradients. Without a defined, entrywise, shape-checked addition, none of those sentences can even be written compactly. Multiplication exists for a deeper reason: it is the operation that makes composition of linear processes a single object. If one linear process turns inputs into hidden features and a second turns hidden features into predictions, their composite is again a linear process, and matrix multiplication is *defined* so that its matrix is the product of the two matrices. That is why the strange row-times-column rule is not arbitrary: it is forced by the requirement (AB)x = A(Bx), apply B first, then A. The book signals this by noting matrices represent linear mappings (§2.7 proves it fully). Without multiplication you cannot write a two-layer linear network, cannot write the normal equations, cannot express "rotate then scale" as one transformation. And without the identity matrix you cannot even state what an inverse means in the next topic ("the matrix that multiplies back to I"), so I_n is infrastructure laid one section before it is needed. [DERIVED: the composition forcing argument is the standard reading of the book's forward reference to §2.7; it is verified concretely in §5.]

A concrete failure without these rules: suppose you try to "multiply" two matrices entrywise (a tempting guess, since addition is entrywise). Then applying the "product" to a vector does not correspond to applying the two transformations in sequence, the dimensions of intermediate quantities stop tracking anything meaningful, and the representation of a linear system A x as a combination of columns (§2.1) breaks. Entrywise multiplication exists in practice (the Hadamard product, used for gating in LSTMs) but it is a different operation with a different job. [DERIVED.]

## 3. Intuition (a physical or visual analogy)

For multiplication, picture a factory with two stages. B is the first machine: it takes n input dials and produces k intermediate parts (column j of B says how much of each input goes into part j... more precisely, B's columns are recipes). A is the second machine: it takes those n-dimensioned intermediates and produces m final goods. The product AB is the direct recipe book from raw inputs to final goods, with every intermediate substitution already carried out: c_ij answers "how much raw input j ends up in final good i, through all intermediate paths at once", and the sum over l is exactly "sum over all intermediate parts l of (input j into part l) times (part l into good i)". That is why the walk index l runs over the shared middle dimension and why that dimension disappears from the result. Addition's picture is simpler: two spreadsheets of measurements, combined cell by cell because the cells mean the same thing in both. Identity is the factory machine that passes everything through unchanged — pointless alone, indispensable as the reference point for "undo this machine" (the inverse). [DERIVED analogy.]

## 4. The mechanics (every formula, every derivation step, every theorem and its proof or proof idea, step-by-step logic)

**Computing one entry.** c_ij = a_i1 b_1j + a_i2 b_2j + ... + a_in b_nj. Nothing is hidden in the sum: n multiplications and n−1 additions per entry, mk entries for the full product, so an (m x n)(n x k) product costs on the order of mnk operations — the counting that later explains why matrix multiplication dominates training cost and why its shape order matters via associativity (you may parenthesize (AB)C either way, but the cost differs enormously; the values do not). [DERIVED cost count from the book's formula.]

**Why associativity is believable, entry level.** ((AB)C)_ij = sum_p (sum_l a_il b_lp) c_pj = sum over l, p of a_il b_lp c_pj, and (A(BC))_ij expands to the same double sum with the grouping moved. Since real-number addition and multiplication associate, the two groupings of the same total agree. The book states the law; this is its proof idea in one line, and the same entry-expansion proves distributivity: ((A+B)C)_ij = sum_l (a_il + b_il) c_lj = sum_l a_il c_lj + sum_l b_il c_lj = (AC + BC)_ij. Identity: (I A)_ij = sum_l delta_il a_lj = a_ij, because the sum's only surviving term is l = i, where delta_ii = 1; A I = A identically on the other side. Note the book's warning embedded in the identity law: for A in R^{m x n} with m ≠ n, the left identity is I_m and the right one is I_n — different matrices, both legitimately called "the identity" by context. [BOOK: mathematics-for-machine-learning, ch.2, §2.2.1, laws; proofs DERIVED by entry expansion.]

**Non-commutativity is the default.** In general AB ≠ BA, and often one of them does not even exist: if A is 2x3 and B is 3x4, AB exists (2x4) while BA would need B's 4 columns to meet A's 2 rows and is undefined. Even when both exist (square matrices of equal size), the book's worked product pair shows the two orders giving different matrices. Order in a product is *sequence order* — rightmost factor acts first on a vector — so treating multiplication as commutative would mean claiming transformation order never matters, which rotating-then-translating a point already refutes geometrically. [BOOK: mathematics-for-machine-learning, ch.2, §2.2.1]

## 5. Worked examples (full numbers, nothing skipped; redo the author's examples and add your own)

Full product by hand. Let A = [[1,2,3],[4,5,6]] (2x3) and B = [[7,8],[9,10],[11,12]] (3x2). Then AB is 2x2. c_11 = row 1 of A against column 1 of B: 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58. c_12 = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64. c_21 = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139. c_22 = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154. So AB = [[58,64],[139,154]]. BA also exists here (3x3): its (1,1) entry is 7*1 + 8*4 = 39, already different in shape from AB, demonstrating non-commutativity before any equality of values could even be asked. [DERIVED worked example in the book's format.]

Composition, verifying why the rule exists. Let B scale the first coordinate by 2: B = [[2,0],[0,1]], and A rotate the plane 90 degrees counterclockwise: A = [[0,-1],[1,0]]. Take x = (1, 1). Apply B first: Bx = (2, 1). Then A: A(Bx) = (0*2 + -1*1, 1*2 + 0*1) = (-1, 2). Now the product AB = [[0*2 + -1*0, 0*0 + -1*1],[1*2 + 0*0, 1*0 + 0*1]] = [[0,-1],[2,0]], and (AB)x = (0*1 + -1*1, 2*1 + 0*1) = (-1, 2). Same answer, one operation: the product really is "do B, then A" frozen into a single matrix. The book's Figure 2.5 makes the companion point about shapes: even when both orders are defined, AB and BA can have different dimensions. [BOOK: mathematics-for-machine-learning, ch.2, §2.2.1; numbers DERIVED and machine-checked in §6.]

Identity check: for the A above, I_2 A = [[1,0],[0,1]] [[1,2,3],[4,5,6]]... that pairing is shape-illegal (2x2 times 2x3 is legal: inner 2 = 2, result 2x3): row 1: 1*1+0*4 = 1, 1*2+0*5 = 2, 1*3+0*6 = 3; row 2: 0*1+1*4 = 4, 5, 6 — A returned unchanged, and A I_3 on the right likewise. [DERIVED.]

## 6. Code (runnable, with saved output)

```python
def matmul(A, B):
    m, n, k = len(A), len(B), len(B[0])
    return [[sum(A[i][l] * B[l][j] for l in range(n)) for j in range(k)] for i in range(m)]

A = [[1, 2, 3], [4, 5, 6]]
B = [[7, 8], [9, 10], [11, 12]]
print(matmul(A, B))
rot = [[0, -1], [1, 0]]
scale = [[2, 0], [0, 1]]
print(matmul(rot, matmul(scale, [[1], [1]])), matmul(matmul(rot, scale), [[1], [1]]))
```
```text
[[58, 64], [139, 154]]
[[-1], [2]] [[-1], [2]]
```

## 7. Where it shows up in AI/ML (concrete uses)

One neural network layer is exactly an affine map built from this multiplication: H = XW + b for a batch X (examples x features), W (features x hidden units). Attention scores are QK^T, a product with a transpose. The prediction of linear regression is X theta. Covariance estimation computes X^T X. Even the cost conversation is this section: multiplying (N x D)(D x K) costs about NDK multiply-adds, which is why batch sizes, feature counts, and hidden widths are the knobs of training budgets. Associativity with cost-awareness is why libraries choose multiplication order and why (X^T X) theta can be far cheaper than X^T (X theta) or vice versa depending on shapes. [DERIVED applications.]

## 8. Common mistakes and misconceptions

(1) Multiplying entrywise and calling it matrix multiplication. (2) Forgetting the inner-dimension check, or "fixing" a mismatch by silently transposing the wrong factor. (3) Assuming AB = BA, especially for square matrices where both exist and look comparable. (4) Reading (AB)x as "A first" — the factor nearest x acts first. (5) Using one identity matrix for both sides of a rectangular A; the two identities have different sizes. (6) Believing associativity means order of evaluation is irrelevant computationally: values agree, runtimes can differ by orders of magnitude.

## 9. Interview angle (how it is asked, what a strong answer includes)

Asked as: "What does this line compute and what are the shapes?" (for instance H = X @ W with X: (128, 784), W: (784, 256)). A strong answer: result (128, 256); walk one entry (row of X against column of W, summed over 784 products); state the cost (~128*784*256 operations); and note the multiplication's meaning (each hidden unit is a weighted combination of all input features). Volunteering the composition meaning of products ("rightmost applies first") marks real understanding.

## 10. Visualization spec (what should be drawn or interactive, with the parameters the user can change)

Row-by-column multiplication animator: two grids A and B; selecting result cell (i, j) highlights row i and column j and steps through the l-walk, accumulating products term by term. A second panel composes transformations: pick "scale" then "rotate" (and swap the order) and watch a test point and the grid transform, showing the two orders landing differently. User parameters: matrix entries, selected cell, transformation order, test point.

## 11. Sources and verification (provenance tags and proof file paths)

[BOOK: mathematics-for-machine-learning, ch.2, §2.2.1] for both definitions, identity matrix, and the stated laws. [DERIVED]: entry-level proof ideas for the laws, cost count, composition verification, and all worked numbers — each machine-checked by the §6 code via scripts/code_check.py.
