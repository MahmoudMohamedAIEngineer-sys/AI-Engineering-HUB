# Practice: Inverse and Transpose

Source topics for all problems: `math.linalg.matrices`. Every problem is original (never copied from the book), with a full worked solution.

### Problem 1 (recall)
source_topics: [math.linalg.matrices]
Define transpose, inverse, regular/singular, and symmetric matrix.

**Solution.** (A^T)_ij=a_ji. B=A^{-1} iff AB=I=BA (square). A is regular/invertible/nonsingular if A^{-1} exists, else singular. A symmetric iff A=A^T (square only). [BOOK: mathematics-for-machine-learning, ch.2, §2.2.2]

### Problem 2 (recall)
source_topics: [math.linalg.matrices]
State the transpose laws and the two warnings about inverses from this section.

**Solution.** (A^T)^T=A; (AB)^T=B^T A^T (order flips); (A+B)^T=A^T+B^T; (A^{-1})^T=(A^T)^{-1}. Warnings: not every square matrix is invertible; (A+B)^{-1} != A^{-1}+B^{-1} in general. [BOOK: mathematics-for-machine-learning, ch.2, §2.2.2]

### Problem 3 (computation)
source_topics: [math.linalg.matrices]
Find the inverse of A=[[3,1],[2,1]] using the 2x2 formula and verify one diagonal and one off-diagonal entry of A A^{-1}.

**Solution.** ad-bc=3-2=1, so A^{-1}=[[1,-1],[-2,3]]. Check (1,1): 3*1+1*(-2)=1. (1,2): 3*(-1)+1*3=0. [DERIVED from BOOK §2.2.2 formula]

### Problem 4 (computation)
source_topics: [math.linalg.matrices]
Transpose A=[[1,0,2],[3,4,5]] and state both shapes; then give (A^T) entry (3,2).

**Solution.** A is 2x3; A^T=[[1,3],[0,4],[2,5]] is 3x2. (A^T)_{3,2}=a_{2,3}=5. [DERIVED]

### Problem 5 (derivation)
source_topics: [math.linalg.matrices]
Prove that A^T A is symmetric for any A in R^{m x n}, using only the transpose laws.

**Solution.** (A^T A)^T = (A)^T (A^T)^T by (XY)^T=Y^T X^T with X=A^T, Y=A; = A^T A. A matrix equal to its transpose is symmetric; A^T A is n x n, square, so symmetry applies. [DERIVED from BOOK §2.2.2 laws]

### Problem 6 (application (ML))
source_topics: [math.linalg.matrices]
In least squares, X is (500, 4) with two identical feature columns. What goes wrong with (X^T X)^{-1} and which property from this section explains it?

**Solution.** X^T X becomes singular (duplicate columns make one row/column a copy of another, like [[1,2],[2,4]] scaled up), so the inverse does not exist and the closed form fails. Remedy direction: drop a column or use ridge (X^T X + lambda I). [DERIVED]

### Problem 7 (application (ML))
source_topics: [math.linalg.matrices]
Attention computes scores Q K^T. If Q and K are both (seq, d), what shape are the scores and why is the transpose the operation that appears?

**Solution.** Scores are (seq, seq): entry (i,j) must pair query i with key j, i.e. dot rows of Q with rows of K; K^T turns K's rows into columns so the product's c_ij = sum_l Q_il (K^T)_lj = sum_l Q_il K_jl. [DERIVED]

### Problem 8 (interview)
source_topics: [math.linalg.matrices]
Interview-style: 'When is a matrix invertible?' Model answer and weak-answer example.

**Solution.** Model: 'For square A: iff A^{-1} exists, equivalently determinant != 0 (Ch.4), equivalently A x=b has a unique solution for all b, equivalently no row is a combination of others; 2x2 check: ad-bc != 0. [[1,2],[2,4]] is singular.' Weak: 'When you can divide by it' — no criterion, no example. source_topics: [math.linalg.matrices]
