# Practice: Solving Systems: REF, RREF, Pivots

All problems are original (never copied from the book), fully solved, and sourced below.

### Problem 1 (recall)
source_topics: [math.linalg.systems]
State the three elementary transformations and the one restriction on the scaling move, with the reason for the restriction.

**Solution.** Swap two rows; multiply a row by lambda != 0; add a (multiple of one) row to another. Lambda must be nonzero because multiplying by 0 erases the equation's information and can invent solutions; nonzero scaling is reversible and preserves the solution set exactly. [BOOK: mathematics-for-machine-learning, ch.2, §2.3.2]

### Problem 2 (recall)
source_topics: [math.linalg.systems]
Define pivot, row-echelon form, reduced row-echelon form, basic and free variables.

**Solution.** Pivot = first nonzero entry of a row. REF: zero rows at bottom, each pivot strictly right of the one above. RREF: REF plus every pivot is 1 and alone in its column. Basic variables = pivot-column variables; free variables = the rest. [BOOK: mathematics-for-machine-learning, ch.2, Def. 2.6 and remarks]

### Problem 3 (computation)
source_topics: [math.linalg.systems]
Bring [2 4 | 10; 1 3 | 7] to REF and solve.

**Solution.** Row1/2: [1 2 | 5; 1 3 | 7]. Row2 -= Row1: [1 2 |5; 0 1 |2]. Back-substitute: x2=2, x1=5-4=1. Solution (1,2). Check: 2+8=10; 1+6=7. [DERIVED]

### Problem 4 (computation)
source_topics: [math.linalg.systems]
For REF [1 -2 1 | 0; 0 0 1 | -2; 0 0 0 | a+1] (miniature of Example 2.6), state pivots, basic/free variables, and the value(s) of a for which solutions exist.

**Solution.** Pivots in columns 1 and 3; basic x1,x3; free x2. Solutions exist only when a+1=0, i.e. a=-1. [BOOK: mathematics-for-machine-learning, ch.2, Example 2.6 pattern]

### Problem 5 (derivation)
source_topics: [math.linalg.systems]
Prove that adding lambda times row i to row j (i != j) never changes the solution set.

**Solution.** Any x satisfying both original equations satisfies the new row j because it is a sum of two equalities that hold. Conversely the operation is reversible: subtract lambda times row i from the new row j to recover the old system, so any x satisfying the new system satisfies the old. Same solution set both ways. [DERIVED from BOOK §2.3.2]

### Problem 6 (application (ML))
source_topics: [math.linalg.systems]
An REF of a training system has a row [0 0 0 | 5]. What does training code conclude before computing any weights, and which fate is this?

**Solution.** The row asserts 0 = 5, impossible: the system is inconsistent (no-solution fate). Exact solving stops; the ML response is a least-squares/approximate fit (book Ch. 9), not more elimination. [DERIVED]

### Problem 7 (application (ML))
source_topics: [math.linalg.systems]
A system for 4 model weights reduces to RREF with pivots in columns 1,2,4 only. How many weights does the data determine, how many remain free, and what is the standard ML response?

**Solution.** 3 pivot columns determine 3 basic weights; 1 free weight remains undetermined by the equations. Response: add a selection principle — regularization, a prior, or more data. [DERIVED]

### Problem 8 (interview)
source_topics: [math.linalg.systems]
Interview-style: 'How do you detect infinitely many solutions during Gaussian elimination?' Model and weak answers.

**Solution.** Model: 'After REF/RREF, if a consistent system has a column without a pivot, that variable is free and parametrizes infinitely many solutions; count = number of free columns.' Weak: 'If there are more unknowns than equations' — counts alone do not decide it (a 3x3 system showed all three fates in §2.1). source_topics: [math.linalg.systems]
