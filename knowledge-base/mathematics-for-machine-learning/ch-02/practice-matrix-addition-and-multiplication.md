# Practice: Matrix Addition and Multiplication

Source topics for all problems: `math.linalg.matrices`. Every problem is original (never copied from the book), with a full worked solution.

### Problem 1 (recall)
source_topics: [math.linalg.matrices]
State the entry formulas for A+B and C=AB, including when each is defined.

**Solution.** (A+B)_ij=a_ij+b_ij, same shape only. c_ij = sum_l a_il b_lj for A in R^{m x n}, B in R^{n x k}, giving C in R^{m x k}; inner dimensions must agree. [BOOK: mathematics-for-machine-learning, ch.2, §2.2.1]

### Problem 2 (recall)
source_topics: [math.linalg.matrices]
State associativity, both distributivity laws, and the identity law, with the book's caveat about identity sizes.

**Solution.** (AB)C=A(BC); (A+B)C=AC+BC; A(C+D)=AC+AD; I_m A = A I_n = A for A in R^{m x n}, and I_m != I_n when m!=n. [BOOK: mathematics-for-machine-learning, ch.2, §2.2.1]

### Problem 3 (computation)
source_topics: [math.linalg.matrices]
Compute AB for A=[[2,0],[1,3]], B=[[1,4],[2,5]].

**Solution.** c11=2*1+0*2=2; c12=2*4+0*5=8; c21=1*1+3*2=7; c22=1*4+3*5=19. AB=[[2,8],[7,19]]. [DERIVED]

### Problem 4 (computation)
source_topics: [math.linalg.matrices]
A is 3x5, B is 5x2, C is 2x7. Which of AB, BA, (AB)C, A(BC) exist, and what are the result shapes?

**Solution.** AB exists: 3x2. BA: (5x2)(3x5) inner 2 vs 3 — undefined. (AB)C: (3x2)(2x7) = 3x7. BC: (5x2)(2x7)=5x7, then A(BC): (3x5)(5x7)=3x7. Both associations exist and agree in value. [DERIVED]

### Problem 5 (derivation)
source_topics: [math.linalg.matrices]
Prove (A+B)C = AC + BC entry-by-entry from the definitions.

**Solution.** ((A+B)C)_ij = sum_l (a_il+b_il) c_lj = sum_l a_il c_lj + sum_l b_il c_lj = (AC)_ij + (BC)_ij = (AC+BC)_ij for all i,j. [DERIVED from BOOK §2.2.1 definitions]

### Problem 6 (application (ML))
source_topics: [math.linalg.matrices]
X is (128, 784) and W is (784, 256). What is the shape and approximate multiply-add count of XW, and what does one output entry mean?

**Solution.** (128,256); about 128*784*256 = 25,690,112 multiply-adds. Entry (i,j) = example i's 784 features dotted with hidden unit j's weight column: one hidden unit's pre-activation for one example. [DERIVED]

### Problem 7 (application (ML))
source_topics: [math.linalg.matrices]
A two-layer linear network computes A(Bx) for x in R^4, B in R^{3x4}, A in R^{2x3}. Rewrite as one matrix times x, give its shape, and state which transformation is applied first.

**Solution.** (AB)x with AB in R^{2x4} (compute B first: (2x3)(3x4) inner 3 agrees, result 2x4). B is applied first because it stands nearest x. [DERIVED]

### Problem 8 (interview)
source_topics: [math.linalg.matrices]
Interview-style: 'Is matrix multiplication commutative?' Model answer and weak-answer example.

**Solution.** Model: 'No. Often BA is not even defined (shape mismatch); for square factors the values generally differ because order encodes composition order — rightmost factor acts first. Example: rotate-after-scale vs scale-after-rotate give different results.' Weak: 'No, AB != BA' with no reason or example. source_topics: [math.linalg.matrices]
