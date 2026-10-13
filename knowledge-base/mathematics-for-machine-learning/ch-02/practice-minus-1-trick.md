# Practice: The Minus-1 Trick

All problems are original (never copied from the book), fully solved, and sourced below.

### Problem 1 (recall)
source_topics: [math.linalg.systems]
State the minus-1 trick recipe in four steps.

**Solution.** (1) Reduce A to RREF. (2) Mark pivot and free columns. (3) Pad to square A-tilde inserting a -1 row at each free diagonal position. (4) Read each free column of A-tilde as one homogeneous solution direction. [BOOK: mathematics-for-machine-learning, ch.2, §2.3.3]

### Problem 2 (recall)
source_topics: [math.linalg.systems]
What do the read-off columns form, and what is that solution space later called?

**Solution.** A basis of the solution space of A x = 0 (book §2.3.3), later called the kernel or null space (§2.7.3). [BOOK: mathematics-for-machine-learning, ch.2, §2.3.3]

### Problem 3 (computation)
source_topics: [math.linalg.systems]
Apply the trick to A = [[1,2,0],[0,0,1]] (RREF) and verify the direction.

**Solution.** Pivots columns 1,3; free column 2. A-tilde inserts -1 row at position 2: [[1,2,0],[0,-1,0],[0,0,1]]. Read free column 2: (2,-1,0). Check: A(2,-1,0) = (2-2, 0) = (0,0). [DERIVED from BOOK §2.3.3]

### Problem 4 (computation)
source_topics: [math.linalg.systems]
For A in Example 2.8 (3 pivots, 5 columns), how many homogeneous directions are there and why?

**Solution.** 5 - 3 = 2 directions, one per free column (columns 2 and 5). [BOOK: mathematics-for-machine-learning, ch.2, Example 2.8]

### Problem 5 (derivation)
source_topics: [math.linalg.systems]
Show why a read-off vector v for free column f satisfies A v = 0, using the RREF column structure.

**Solution.** In RREF each pivot column is a unit vector e_i, and free column f equals sum_i a_if * (pivot column for row i). The read-off v places a_if in pivot slots, -1 in slot f, 0 in other free slots, so A v = (that combination of pivot columns) - (column f) = column f - column f = 0. [DERIVED from BOOK §2.3.3]

### Problem 6 (application (ML))
source_topics: [math.linalg.systems]
A linear classifier's weight matrix maps R^5 inputs to 3 class scores and its RREF has 3 pivots. How many independent input perturbation directions leave all scores exactly unchanged, and why do security analysts care?

**Solution.** 5 - 3 = 2 independent kernel directions: perturbations along them change the input without changing any score — invisible directions, the linear case of adversarial blind spots. [DERIVED]

### Problem 7 (application (ML))
source_topics: [math.linalg.systems]
You read off kernel directions from a matrix in REF but not RREF and get wrong answers. What exactly failed?

**Solution.** The read-off coefficients assume pivot columns are unit vectors alone in their columns (RREF). In mere REF, pivot columns still contain entries above/below pivots, so copying free-column entries does not produce A v = 0. Finish the reduction first. [DERIVED from BOOK §2.3.3 preconditions]

### Problem 8 (interview)
source_topics: [math.linalg.systems]
Interview-style: 'Find the null space of this RREF matrix.' Model approach and weak-answer example.

**Solution.** Model: mark pivots/free columns, one direction per free column (pivot coefficients copied, -1 in own slot), state dimension = n - pivots, verify one direction against A. Weak: solving by back-substitution from scratch and dropping a sign — the trick exists to prevent exactly that. source_topics: [math.linalg.systems]
