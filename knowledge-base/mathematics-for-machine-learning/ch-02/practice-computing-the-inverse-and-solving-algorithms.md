# Practice: Computing the Inverse; Solving Algorithms

All problems are original (never copied from the book), fully solved, and sourced below.

### Problem 1 (recall)
source_topics: [math.linalg.matrices]
State the [A | I] -> [I | A^{-1}] method and when it fails.

**Solution.** Run Gaussian elimination on [A | I_n]; if the left half reaches I_n, the right half is A^{-1}. It fails (a needed pivot is missing) exactly when A is singular. [BOOK: mathematics-for-machine-learning, ch.2, §2.3 'Calculating the Inverse']

### Problem 2 (recall)
source_topics: [math.linalg.matrices]
State the pseudo-inverse solution formula, its assumption, and the book's numerical warning.

**Solution.** x = (A^T A)^{-1} A^T b, assuming A has linearly independent columns. Warning: forming inverses/pseudo-inverses explicitly costs many computations and is generally not recommended for numerical precision. [BOOK: mathematics-for-machine-learning, ch.2, §2.3.4]

### Problem 3 (computation)
source_topics: [math.linalg.matrices]
Invert A=[[1,2],[1,3]] by [A|I] elimination.

**Solution.** Augmented [1 2|1 0; 1 3|0 1]. Row2-=Row1: [1 2|1 0; 0 1|-1 1]. Row1-=2 Row2: [1 0|3 -2; 0 1|-1 1]. A^{-1}=[[3,-2],[-1,1]]. Check (1,1): 1*3+2*(-1)=1. [DERIVED]

### Problem 4 (computation)
source_topics: [math.linalg.matrices]
Estimate elimination cost ratio: a system grows from n=100 to n=10000 (100x). By what factor do operations grow under cubic scaling?

**Solution.** (100)^3 = 1,000,000 times more operations. This is why million-variable systems abandon direct elimination. [BOOK: mathematics-for-machine-learning, ch.2, §2.3.4 cubic scaling]

### Problem 5 (derivation)
source_topics: [math.linalg.matrices]
Justify the [A|I] method: why does the right half become A^{-1}?

**Solution.** Row operations are left-multiplication by invertible operation matrices; if their composition E sends A to I, then E A = I, so E is (for square A) the inverse. The same operations applied to the right half I produce E I = E = A^{-1}. [DERIVED from BOOK §2.3 (2.56)]

### Problem 6 (application (ML))
source_topics: [math.linalg.matrices]
Your code computes theta = inv(X^T X) @ X^T @ y and is slow and occasionally singular. Give the two fixes this section prescribes.

**Solution.** (1) Do not form the inverse: solve (X^T X) theta = X^T y with a solver/factorization. (2) Singularity means dependent columns: drop a column or regularize (ridge adds lambda I). In production code even X^T X is usually avoided via QR/SVD. [DERIVED from BOOK §2.3.4]

### Problem 7 (application (ML))
source_topics: [math.linalg.matrices]
A consistent system has A of shape 20x100 (more unknowns than equations). Can the pseudo-inverse formula still select a solution, and which one?

**Solution.** Yes under independent rows/columns conditions in its transposed form: the pseudo-inverse selects the minimum-norm solution among the infinite family — the family member closest to the origin. [BOOK: mathematics-for-machine-learning, ch.2, §2.3.4 minimum-norm statement]

### Problem 8 (interview)
source_topics: [math.linalg.matrices]
Interview-style: 'How do you solve A x = b in code?' Model and weak answers.

**Solution.** Model: 'Never invert: use an elimination/factorization solve, O(n^3); for inconsistent/underdetermined cases use least squares via QR/SVD (pseudo-inverse conceptually); for very large sparse systems use iterative methods like conjugate gradients.' Weak: 'Multiply both sides by A inverse' — states the formula, ignores cost and numerics. source_topics: [math.linalg.matrices]
