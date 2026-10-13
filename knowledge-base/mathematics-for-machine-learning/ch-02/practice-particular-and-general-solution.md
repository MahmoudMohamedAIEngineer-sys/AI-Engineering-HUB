# Practice: Particular and General Solutions

All problems are original (never copied from the book), fully solved, and sourced below.

### Problem 1 (recall)
source_topics: [math.linalg.systems]
State the book's three steps for obtaining all solutions of A x = b, and the non-uniqueness remark.

**Solution.** (1) Find one particular solution of A x = b. (2) Find all solutions of A x = 0. (3) General solution = particular + homogeneous solutions. Neither particular nor general solution is unique. [BOOK: mathematics-for-machine-learning, ch.2, §2.3.1]

### Problem 2 (recall)
source_topics: [math.linalg.systems]
In the book's (2.38) example, which columns are pivot columns, which variables are free, and what is the particular solution?

**Solution.** Pivot columns 1,2 (they are e1,e2); free variables x3,x4; particular solution (42, 8, 0, 0) from b = 42 c1 + 8 c2. [BOOK: mathematics-for-machine-learning, ch.2, §2.3.1]

### Problem 3 (computation)
source_topics: [math.linalg.systems]
For A=[[1,0,8,-4],[0,1,2,12]], b=(42,8), write the solution at lambda1=2, lambda2=-1 and verify row 1.

**Solution.** x = (42,8,0,0) + 2(8,2,-1,0) -1(-4,12,0,-1) = (42+16+4, 8+4-12, -2, 1) = (62, 0, -2, 1). Row1: 62 + 8*(-2) + (-4)(1) = 62-16-4 = 42. OK. [DERIVED from BOOK §2.3.1]

### Problem 4 (computation)
source_topics: [math.linalg.systems]
A x = b has particular solution (1,2,3) and homogeneous directions (1,0,-1) and (0,1,1). Give two distinct solutions different from the particular one and verify the first against the homogeneous direction property (A applied gives 0 for directions — state what you can check without A).

**Solution.** E.g. (1,2,3)+(1,0,-1)=(2,2,2) and (1,2,3)+(0,1,1)=(1,3,4). Without A we can check the construction: each differs from the particular solution by a stated homogeneous direction, so by the structure theorem each is a solution iff the directions satisfy A h = 0. [DERIVED]

### Problem 5 (derivation)
source_topics: [math.linalg.systems]
Prove: if x1 and x2 both solve A x = b, then x1 - x2 solves A x = 0, and conclude the solution set is either empty, a single point, or infinite.

**Solution.** A(x1-x2) = A x1 - A x2 = b - b = 0. If two distinct solutions exist, their nonzero difference h gives infinitely many solutions x1 + t h (each checks: A(x1 + t h)=b+0). So counts are 0, 1, or infinite. [DERIVED; formalizes BOOK §2.1 trichotomy]

### Problem 6 (application (ML))
source_topics: [math.linalg.systems]
An overparameterized model fits training data exactly with weights w*. A colleague says 'the solution is unique, training converged to it.' Use this section to correct them in two sentences.

**Solution.** Convergence found one particular solution; the homogeneous directions (weight changes leaving training predictions unchanged) were never shown to be trivial, so uniqueness is unproven and generally false for overparameterized models. Which member of the family you get is decided by initialization/optimizer, not by the data alone. [DERIVED]

### Problem 7 (application (ML))
source_topics: [math.linalg.systems]
Underdetermined regression: X is 10x50 and X w = y is consistent. How many free parameters does the solution family have at minimum, and what does minimum-norm selection add?

**Solution.** At least 50-10 = 40 free directions (pivots <= 10). Minimum-norm selection picks the unique family member closest to 0, computed via the pseudo-inverse (algorithms topic). [DERIVED]

### Problem 8 (interview)
source_topics: [math.linalg.systems]
Interview-style: 'Describe all solutions of an underdetermined consistent system.' Model and weak answers.

**Solution.** Model: 'One particular solution plus the full homogeneous solution space (null space), parametrized by one lambda per free variable; dimension = unknowns - pivots.' Weak: 'There are infinitely many solutions' — true but describes none of them. source_topics: [math.linalg.systems]
