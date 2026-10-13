# Practice: Matrices (definition, scalar multiplication, compact form)

Source topics for all problems: `math.linalg.matrices`. Every problem is original (never copied from the book), with a full worked solution.

### Problem 1 (recall)
source_topics: [math.linalg.matrices]
Give the formal definition of an (m,n)-matrix and the meaning of a_ij, R^{m x n}, row vector, column vector.

**Solution.** An m-by-n tuple of reals a_ij (i=1..m rows, j=1..n columns) in a rectangular scheme; R^{m x n} is the set of all of them; (1,n) is a row vector, (m,1) a column vector. [BOOK: mathematics-for-machine-learning, ch.2, §2.2]

### Problem 2 (recall)
source_topics: [math.linalg.matrices]
State the three scalar-multiplication laws used in this topic and what (lambda A)_ij is.

**Solution.** (lambda A)_ij = lambda a_ij. Laws: (lp)A = l(pA); (l+p)A = lA + pA; l(A+B) = lA + lB; plus l(BC)=(lB)C=B(lC). [BOOK: mathematics-for-machine-learning, ch.2, §2.2.3]

### Problem 3 (computation)
source_topics: [math.linalg.matrices]
Let A=[[2,-1],[0,3]] and lambda=-2. Compute lambda A and then (lambda A) with lambda replaced by (2 + -4)... i.e. verify (l+p)A = lA+pA for l=2, p=-4 on entry (1,2).

**Solution.** lambda A = [[-4,2],[0,-6]]. Entry (1,2): (2 + -4)*(-1) = 2; 2*(-1) + -4*(-1) = -2+4 = 2. Equal. [DERIVED]

### Problem 4 (computation)
source_topics: [math.linalg.matrices]
Write the system x1+2x2=7; 3x1-x2=5 in compact form A x = b, giving A, x, b explicitly.

**Solution.** A=[[1,2],[3,-1]], x=(x1,x2), b=(7,5). Expanding A x row-wise returns the two equations. [BOOK: mathematics-for-machine-learning, ch.2, §2.2.4 pattern]

### Problem 5 (derivation)
source_topics: [math.linalg.matrices]
Prove (lambda + psi) A = lambda A + psi A directly from the entry definition of scalar multiplication.

**Solution.** Entry (i,j) of LHS = (lambda+psi) a_ij = lambda a_ij + psi a_ij (real distributivity) = entry (i,j) of lambda A plus entry (i,j) of psi A = entry (i,j) of RHS. All entries agree, so the matrices are equal. [DERIVED from BOOK §2.2.3]

### Problem 6 (application (ML))
source_topics: [math.linalg.matrices]
A dataset matrix X has shape (1000, 20). You stack its columns into one vector per the book's convention. What is the vector's length, and what ML object is usually (1000,20) instead?

**Solution.** Stacking a single matrix's columns gives length 1000*20 = 20000 in R^20000. Usually (1000,20) itself is the dataset: 1000 examples x 20 features, kept unstacked so rows stay examples. [DERIVED]

### Problem 7 (application (ML))
source_topics: [math.linalg.matrices]
In gradient descent the update is theta := theta - lambda * g where g is a gradient matrix of the same shape as theta. Which operations from this chapter are being used, and why must shapes match?

**Solution.** Scalar multiplication (lambda*g scales every entry) and the (later) same-shape addition/subtraction. Shapes must match because entrywise operations pair corresponding entries; a mismatched gradient cannot be subtracted entry-by-entry. [DERIVED]

### Problem 8 (interview)
source_topics: [math.linalg.matrices]
Interview-style: 'What is a matrix, really?' Model answer and weak-answer example.

**Solution.** Model: 'A rectangular array in R^{m x n} with entries a_ij addressed by row i and column j; it compacts a linear system as A x = b and represents linear mappings (Ch.2 §2.7).' Weak: 'A table of numbers you can multiply' — no addresses, no shape, no purpose. source_topics: [math.linalg.matrices]
