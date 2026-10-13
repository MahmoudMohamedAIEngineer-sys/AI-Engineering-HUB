# Practice: Systems of Linear Equations

Source topics for all problems: `math.linalg.systems`. Every problem is original (never copied from the book), with a full worked solution.

### Problem 1 (recall)
source_topics: [math.linalg.systems]
State the three possible fates of a real system of linear equations and the geometric picture for two unknowns.

**Solution.** No solution, exactly one solution, infinitely many. Two unknowns: lines intersect in zero points (parallel distinct), one point, or all points (same line). [BOOK: mathematics-for-machine-learning, ch.2, §2.1]

### Problem 2 (recall)
source_topics: [math.linalg.systems]
What makes an equation 'linear' in this section, and what is a free variable?

**Solution.** Each unknown appears only to the first power, never multiplied by another unknown or inside a nonlinear function. A free variable is an unknown not determined by the system, set to an arbitrary parameter a in R, generating the solution family. [BOOK: mathematics-for-machine-learning, ch.2, §2.1]

### Problem 3 (computation)
source_topics: [math.linalg.systems]
Solve or diagnose: x1+x2=5; x1-x2=1.

**Solution.** Add: 2x1=6 so x1=3; then x2=2. Check: 3+2=5, 3-2=1. Unique solution (3,2). [DERIVED]

### Problem 4 (computation)
source_topics: [math.linalg.systems]
Solve or diagnose: x1+x2+x3=6; x2+x3=4; x1+2x2+2x3=10. Hint: compare the third equation with the first two.

**Solution.** Third = first + (x2+x3 component): (x1+x2+x3)+(x2+x3)=6+4=10, so eq.3 is redundant. Set x3=a: x2=4-a, x1=6-(x2+x3)=6-4=2. Family: (2, 4-a, a), infinitely many solutions. [DERIVED in book's Fate-3 pattern]

### Problem 5 (derivation)
source_topics: [math.linalg.systems]
Prove the trichotomy for a system with two distinct solutions: if u and v (u!=v) both solve A x = b in equation form, then every point u + t(v-u), t in R, is also a solution. (Use the equation form, not matrices.)

**Solution.** Each equation is linear: sum_j a_ij (u_j + t(v_j-u_j)) = sum a_ij u_j + t(sum a_ij v_j - sum a_ij u_j) = b_i + t(b_i - b_i) = b_i. So the whole line through u,v solves the system; distinct u,v give infinitely many solutions. Hence solution counts are 0, 1, or infinite. [DERIVED; formalizes BOOK ch.2 §2.1]

### Problem 6 (application (ML))
source_topics: [math.linalg.systems]
Training data for linear regression gives 1,000 equations (examples) in 5 unknowns (weights) with no exact solution. Which fate is this, and what does Chapter 9 do about it (as foreshadowed in §2.1)?

**Solution.** No-solution (overdetermined) fate. Chapter 9 relaxes 'solve exactly' to least squares: choose weights minimizing total squared error — the book states regression solves a version of such systems when they cannot be solved exactly. [BOOK: mathematics-for-machine-learning, ch.2, §2.1]

### Problem 7 (application (ML))
source_topics: [math.linalg.systems]
A model has 10,000 parameters and only 300 training examples, and many very different parameter vectors fit the training data perfectly. Which fate, and what is the standard remedy direction?

**Solution.** Infinitely-many-solutions (underdetermined) fate. Remedy direction: add constraints/preferences among solutions — regularization or more data — to select one; §2.3/§2.6 formalize via rank and free variables. [DERIVED]

### Problem 8 (interview)
source_topics: [math.linalg.systems]
Interview-style: 'When does A x = b have infinitely many solutions?' Model answer and weak-answer example.

**Solution.** Model: 'When the equations contain redundancy relative to the unknowns — some unknown stays free; equivalently rank < number of unknowns (Ch.2 §2.6). Example: (5/2-3a/2, 1/2+a/2, a).' Weak: 'When there are more variables than equations' — count alone does not decide; independence does. source_topics: [math.linalg.systems]
