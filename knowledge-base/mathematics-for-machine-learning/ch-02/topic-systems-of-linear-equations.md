---
id: math.linalg.systems
book: mathematics-for-machine-learning
chapter: 2
section: 2.1
tag: must_know
prerequisites: [ml.foundations.data-models-learning]
leads_to: [math.linalg.matrices, math.calc.gradients]
---
# Systems of Linear Equations

## 1. What it is (plain-language definition, then the formal definition)

In plain language, a system of linear equations is a list of demands that must all hold at the same time, where each demand says "these unknowns, weighted and added, equal this target". A solution is one assignment to all unknowns that satisfies every demand simultaneously. [BOOK: mathematics-for-machine-learning, ch.2, §2.1]

Formally, with unknowns x_1, ..., x_n, coefficients a_ij in R and targets b_i in R, a system of m linear equations is: for each i = 1..m, a_i1 x_1 + ... + a_in x_n = b_i. The word "linear" has a precise job here: every unknown appears only to the first power, never multiplied by another unknown and never inside a nonlinear function, which is exactly what makes the geometry below (lines, planes) and the later matrix algebra work. An n-tuple (x_1, ..., x_n) in R^n that satisfies all m equations at once is a **solution**; the set of all such tuples is the solution set. The book's motivating instance is a production plan: products N_1..N_n each consume a_ij units of resource R_i per unit produced, b_i units of resource i are available, and the optimal plan (x_1..x_n) uses every resource exactly, giving one equation per resource. [BOOK: mathematics-for-machine-learning, ch.2, §2.1] The first fact to internalize, proved by the examples below rather than assumed: a real linear system has exactly one of three fates — no solution, exactly one solution, or infinitely many solutions. Never two, never seventeen. [BOOK: mathematics-for-machine-learning, ch.2, §2.1]

## 2. Why it exists (the problem it solves; what breaks without it)

Almost every quantitative question in machine learning, asked plainly, is one of these systems wearing a costume. "Which weights make the model's predictions match the training labels?" is a system (Chapter 9 solves its least-squares relaxation). "Which production plan exhausts the resources?" is a system. "Which vector, written in the new basis, is the same arrow?" is a system (§2.7). If you cannot state a problem as simultaneous linear demands and reason about whether the demands conflict, duplicate each other, or pin the unknowns down, you are reduced to guessing assignments and checking them one at a time — exponential trial and error with no insight into why an assignment works or whether another one would too. The three-fates theorem is the diagnostic payoff: when a system misbehaves you ask which fate you are in and why — conflicting demands (no solution), redundant demands (infinitely many), or exactly enough independent demands (one solution) — and each fate has a different engineering response. When there is no exact solution, Chapter 9's move is to relax "solve" to "come as close as possible", which is where linear regression lives. The book signals this already in this section: linear regression solves a version of the production example precisely when the system itself cannot be solved. [BOOK: mathematics-for-machine-learning, ch.2, §2.1]

## 3. Intuition (a physical or visual analogy)

With two unknowns, each equation is a line drawn on the x_1–x_2 plane: every point on the line is one way to satisfy that single demand. A solution of the system must lie on all lines at once, so the solution set is the intersection of the lines. Two lines in a plane do exactly three things: cross at one point, lie on top of each other (same line), or run parallel and never meet. Those are the three fates, seen. With three unknowns each equation is a plane in 3D space, and intersecting planes can give a plane, a line, a point, or nothing — again matching the trichotomy, and the book's figure of two intersecting lines for the system 4x_1 + 4x_2 = 5, 2x_1 − 4x_2 = 1, meeting at (1, 1/4), is the two-dimensional case drawn once and for all. [BOOK: mathematics-for-machine-learning, ch.2, §2.1] The analogy scales in spirit if not in eyesight: an equation is a constraint surface, a solution is a point lying on all surfaces at once, and "solving" is computing that intersection instead of drawing it.

## 4. The mechanics (every formula, every derivation step, every theorem and its proof or proof idea, step-by-step logic)

The mechanics of this section are three worked diagnoses, each teaching one fate, plus the compact rewriting the rest of the chapter builds on.

**Fate 1 — no solution (contradiction between demands).** System: x_1+x_2+x_3 = 3; x_1−x_2+2x_3 = 2; 2x_1+3x_3 = 1. Add the first two equations, left sides and right sides separately: (x_1+x_1) + (x_2−x_2) + (x_3+2x_3) = 3+2, giving 2x_1 + 3x_3 = 5. But the third equation demands 2x_1 + 3x_3 = 1. The same quantity cannot equal both 5 and 1, so no assignment can satisfy all three: the solution set is empty. Why is adding equations a legal move? Because equality is preserved under adding equal quantities: if A = B and C = D then A + C = B + D. Every solving method in §2.3 is built from this and two sibling moves. [BOOK: mathematics-for-machine-learning, ch.2, §2.1]

**Fate 2 — exactly one solution.** Change the third equation to x_2 + x_3 = 2. From equations (1) and (3): x_1 = 3 − x_2 − x_3 and x_2 = 2 − x_3, so x_1 = 3 − (2 − x_3) − x_3 = 1. Then (1)+(2) gives 2x_1 + 3x_3 = 5, and with x_1 = 1 we get 3x_3 = 3, so x_3 = 1, and back-substituting, x_2 = 2 − 1 = 1. Candidate (1,1,1). A candidate found by deriving consequences is only a candidate until checked — plugging back: 1+1+1 = 3 OK; 1−1+2 = 2 OK; 1+1 = 2 OK — so (1,1,1) is a solution, and the derivation shows every solution must equal it, hence uniqueness. [BOOK: mathematics-for-machine-learning, ch.2, §2.1]

**Fate 3 — infinitely many solutions (a redundant demand).** Change the third equation instead to 2x_1 + 3x_3 = 5. Now (1)+(2) reproduces equation (3) exactly, so (3) adds no information and can be dropped. From (1),(2): adding gives 2x_1 = 5 − 3x_3, i.e. x_1 = 5/2 − (3/2)x_3; subtracting appropriately gives x_2 = 1/2 + (1/2)x_3. Nothing determines x_3; declare it a **free variable** x_3 = a, a in R. The solution set is the one-parameter family (5/2 − 3a/2, 1/2 + a/2, a). One redundant equation converted one would-be-determined unknown into a free parameter — this trade, rank versus free variables, is formalized in §2.3 and §2.6. [BOOK: mathematics-for-machine-learning, ch.2, §2.1]

**Compact rewriting.** Collect the coefficients column-wise: the system equals x_1·(first column) + x_2·(second column) + ... + x_n·(nth column) = b, and stacking the columns side by side produces the matrix form A x = b that the next sections define properly. Read this way, solving A x = b asks: which weights on the columns of A rebuild the target b? That single reading powers §2.5 (linear combinations), §2.6 (rank), and least squares. [BOOK: mathematics-for-machine-learning, ch.2, §2.1]

## 5. Worked examples (full numbers, nothing skipped; redo the author's examples and add your own)

The book's three systems are worked in full in §4 above (no solution; unique solution (1,1,1); family (5/2 − 3a/2, 1/2 + a/2, a)). [BOOK: mathematics-for-machine-learning, ch.2, §2.1] One further example from the section's geometric remark, solved completely: 4x_1 + 4x_2 = 5 and 2x_1 − 4x_2 = 1. Add the equations: 6x_1 = 6, so x_1 = 1. Substitute into the first: 4 + 4x_2 = 5, so x_2 = 1/4. Check in the second: 2 − 1 = 1 OK. Unique solution (1, 1/4) — the crossing point of the two lines in the book's figure. [BOOK: mathematics-for-machine-learning, ch.2, §2.1]

An added example in the production setting, small enough to do by hand: a workshop makes tables (x_1) and chairs (x_2); each table uses 3 units of wood and 1 of labor, each chair 1 of wood and 2 of labor; 12 wood and 10 labor are available and must be used exactly. System: 3x_1 + x_2 = 12; x_1 + 2x_2 = 10. Double the second: 2x_1 + 4x_2 = 20; subtract twice the first: (2x_1+4x_2) − (6x_1+2x_2) = 20 − 24 gives −4x_1 + 2x_2 = −4, so x_2 = 2x_1 − 2. Substitute into 3x_1 + x_2 = 12: 3x_1 + 2x_1 − 2 = 12, 5x_1 = 14, x_1 = 14/5 = 2.8, x_2 = 3.6. Check labor: 2.8 + 7.2 = 10 OK; wood: 8.4 + 3.6 = 12 OK. Note the modeling moral: the mathematics happily returns fractional furniture; the real problem needs an integer constraint the linear system does not know about — an early sighting of the gap between a model and the world it models (Chapter 8). [DERIVED in the book's production-example setting; arithmetic verified in §6 code.]

## 6. Code (runnable, with saved output)

An exact solver for small systems using fractions (no floating-point error, no external libraries), applied to the book's unique-solution system and the production example:

```python
from fractions import Fraction as F

def solve(A, b):
    n = len(A)
    M = [[F(v) for v in row] + [F(b[i])] for i, row in enumerate(A)]
    for col in range(n):
        piv = next(r for r in range(col, n) if M[r][col] != 0)
        M[col], M[piv] = M[piv], M[col]
        for r in range(n):
            if r != col and M[r][col] != 0:
                factor = M[r][col] / M[col][col]
                M[r] = [v - factor * w for v, w in zip(M[r], M[col])]
    return [M[i][n] / M[i][i] for i in range(n)]

print(solve([[1, 1, 1], [1, -1, 2], [0, 1, 1]], [3, 2, 2]))
print(solve([[3, 1], [1, 2]], [12, 10]))
```
```text
[Fraction(1, 1), Fraction(1, 1), Fraction(1, 1)]
[Fraction(14, 5), Fraction(18, 5)]
```

The algorithm is Gauss-Jordan elimination, formalized as elementary transformations in §2.3 of the book; here it serves as a numerical check on the hand derivations. [DERIVED: code output confirms §4 Fate 2 and §5.]

## 7. Where it shows up in AI/ML (concrete uses)

Solving A x = b is the computational core of: ordinary least-squares regression (Chapter 9, where the system is usually inconsistent and the closest answer is taken); the normal equations inside ridge regression; Gaussian process prediction; Kalman filtering updates; and the linear solves inside every Newton step of optimization. The three fates reappear as: overdetermined training data (no exact solution — fit approximately), underdetermined problems with more parameters than data (infinitely many fits — regularize to choose one), and exactly determined interpolation. Recognizing which fate your training problem is in tells you, before any code, whether to reach for least squares, regularization, or more data. [DERIVED from the book's forward pointer to Chapter 9 in §2.1.]

## 8. Common mistakes and misconceptions

(1) Believing a system can have exactly two or exactly five solutions — for real linear systems the count is 0, 1, or infinite, because if two distinct assignments both work, the whole line between them works (proved in §2.3 topics via the free-variable structure). (2) Forgetting to verify a candidate by substitution: derivation by consequences can introduce or lose candidates, substitution is the audit. (3) Treating a redundant equation as a contradiction: redundancy (Fate 3) and contradiction (Fate 1) look similar — both mean an equation is a combination of others — but in Fate 1 the combination disagrees on the right-hand side, in Fate 3 it agrees. (4) Assuming the number of equations decides solvability; three equations in three unknowns exhibited all three fates above with the same count. What matters is independence of the demands, formalized as rank in §2.6.

## 9. Interview angle (how it is asked, what a strong answer includes)

Asked as: "When does A x = b have no solution / one solution / infinitely many?" A strong answer gives the trichotomy, one tiny example of each (the three systems in §4 are ready-made), the geometric reading (intersecting lines/planes), and the ML consequence (overdetermined → least squares; underdetermined → regularization). Mentioning that "number of equations vs unknowns does not decide it; independence (rank) does" is the sentence that separates memorized from understood.

## 10. Visualization spec (what should be drawn or interactive, with the parameters the user can change)

Two-line intersection explorer: the learner drags two lines (each controlled by slope and intercept sliders) in the x_1–x_2 plane and watches the intersection point, the parallel case, and the coincident case; displayed alongside are the two equations and the computed solution or fate label. User parameters: slope_1, intercept_1, slope_2, intercept_2; a "make parallel" and "make coincident" button snap to Fate 1 and Fate 3.

## 11. Sources and verification (provenance tags and proof file paths)

[BOOK: mathematics-for-machine-learning, ch.2, §2.1] for definitions, the production example, all three systems, the trichotomy, the geometric remark, and the compact rewriting. [DERIVED] items show their arithmetic in place and are re-checked by the §6 code, executed by scripts/code_check.py. No web sources.
