---
id: math.linalg.systems
book: mathematics-for-machine-learning
chapter: 2
section: 2.3
tag: must_know
prerequisites: [math.linalg.matrices]
leads_to: [math.linalg.vector-spaces]
---
# Solving Systems of Linear Equations: Elementary Transformations, Row-Echelon Form, Pivots

## 1. What it is (plain-language definition, then the formal definition)

In plain language, solving a system means legally rewriting it — without changing its answers — until the answers can be read off. The legal rewrites are three moves on equations: swap two equations, multiply an equation by a nonzero constant, and add one equation to another. Applied to the augmented matrix [A | b] these are the **elementary transformations** (elementary row operations), and the target shape they aim for is **row-echelon form**: a staircase in which each row's first nonzero entry (its **pivot**) sits strictly to the right of the pivot above it, and any all-zero rows sit at the bottom. Push further — every pivot made 1 and the only nonzero entry in its column — and you get **reduced row-echelon form (RREF)**. [BOOK: mathematics-for-machine-learning, ch.2, §2.3.2 and §2.3 remarks]

Formally, Definition 2.6 of the book: a matrix is in row-echelon form (REF) if (i) all zero rows are at the bottom, and (ii) looking at nonzero rows only, the first nonzero number from the left (the pivot, or leading coefficient) is always strictly to the right of the pivot of the row above. It is in reduced row-echelon form if additionally every pivot is 1 and is the only nonzero entry in its column. **Gaussian elimination** is the algorithm that performs elementary transformations to bring a system into (reduced) row-echelon form. Variables whose columns contain pivots are **basic variables**; the remaining variables are **free variables** — the ones that will parametrize the solution family of the particular/general-solution topic. [BOOK: mathematics-for-machine-learning, ch.2, §2.3]

Why are the three moves legal? A solution must satisfy every equation simultaneously. Swapping equations changes nothing about the set of simultaneous assignments. Multiplying one equation by lambda ≠ 0 keeps exactly the same assignments satisfying it (this is why lambda = 0 is banned: multiplying by 0 erases the equation's information and can invent fake solutions). Adding a multiple of one equation to another preserves the set because any assignment satisfying both originals satisfies the sum, and the move is reversible (subtract the same multiple back), so no assignment is gained or lost. Reversibility is the whole point: each move is an equivalence of systems, not an implication in one direction. [DERIVED from the book's statement that the transformations "keep the solution set the same"; reversibility argument made explicit.]

## 2. Why it exists (the problem it solves; what breaks without it)

Section 2.1 diagnosed the three fates of a system by clever ad-hoc additions of equations. That does not scale: with 5 unknowns and a parameter in the right-hand side (the book's Example 2.6 below), guessing the right combinations is hopeless, and with 1,000 unknowns it is not a method at all. Elementary transformations turn solving into a mechanical procedure with three guarantees a human guesser cannot offer. First, **termination at a readable form**: repeated elimination always lands on the staircase, from which existence (a row "0 = nonzero" signals no solution), uniqueness (every variable basic), or the free variables (non-pivot columns) can be read directly. Second, **invariance**: because every move is reversible, the staircase system has exactly the same solutions as the original, so conclusions read off the staircase are conclusions about the original problem, not about a convenient fiction. Third, **reusability**: the same elimination engine computes inverses, determinants (Ch. 4), rank (§2.6), bases (§2.6), and independence tests (§2.5) — the book lists these explicitly as applications. Without this machinery, every one of those later topics would need its own ad-hoc trick. What breaks if you skip it: you can still solve tiny systems by substitution, but you have no general existence/uniqueness test, no way to parametrize infinite solution sets, and no algorithm to hand to a computer. [BOOK: mathematics-for-machine-learning, ch.2, §2.3.4 for the application list; the guarantee analysis is DERIVED.]

There is also an honesty function. REF separates the question "does a solution exist?" from "what is it?": the row 0 = a+1 in Example 2.6 answers the first question (only if a = -1) before any solution is computed. Ad-hoc substitution tends to produce a candidate first and discover inconsistency late, if at all.

## 3. Intuition (a physical or visual analogy)

Picture a staircase descending left to right — that is literally the shape the book names. Each pivot is the first step-edge of its row. Solving by elimination is like untangling a knot by moves that never cut the string: every elementary transformation rearranges the tangle (the system looks different) while preserving its topology (the solution set). The end state, REF or RREF, is the untangled layout where each basic variable is introduced by exactly one step, so you can walk down the stairs reading constraints in order, and any variable that never gets a step of its own is free — you may place it anywhere, and the stepped variables adjust. Another useful picture: the augmented matrix is a balance sheet and elimination is double-entry bookkeeping — whatever you do to the left of the bar (|) you must do to the right, because an equation is a balance, and the three moves are the only accountancy operations that keep every balance true for exactly the same assignments. [DERIVED analogies for the book's definitions.]

## 4. The mechanics (every formula, every derivation step, every theorem and its proof or proof idea, step-by-step logic)

**The augmented matrix.** Write the system as [A | b]: coefficients left of the bar, right-hand side after it, variables suppressed. Every elementary transformation acts on a whole row of this augmented matrix, right-hand side included. [BOOK: mathematics-for-machine-learning, ch.2, §2.3, Example 2.6]

**The book's Example 2.6, worked completely.** For a ∈ R, solve:
-2x1 + 4x2 - 2x3 - x4 + 4x5 = -3
4x1 - 8x2 + 3x3 - 3x4 + x5 = 2
x1 - 2x2 + x3 - x4 + x5 = 0
x1 - 2x2 + 0x3 - 3x4 + 4x5 = a

Step 1 (swap rows 1 and 3, putting a leading 1 at the top):
[1 -2 1 -1 1 | 0; 4 -8 3 -3 1 | 2; -2 4 -2 -1 4 | -3; 1 -2 0 -3 4 | a]
Step 2 (row2 -= 4·row1; row3 += 2·row1; row4 -= row1):
[1 -2 1 -1 1 | 0; 0 0 -1 1 -3 | 2; 0 0 0 -3 6 | -3; 0 0 -1 -2 3 | a]
Check one entry by hand: row2's x3 entry: 3 - 4·1 = -1. Good.
Step 3 (row4 -= row2; also preparing the staircase):
[1 -2 1 -1 1 | 0; 0 0 -1 1 -3 | 2; 0 0 0 -3 6 | -3; 0 0 0 0 0 | a+1]
Wait — row4 after step 2 is [0 0 -1 -2 3 | a]; row4 -= row2 gives [0 0 0 -3 6 | a-2], and then row4 -= row3 gives [0 0 0 0 0 | a+1]. The book compresses these as "-R2, -R3" annotations.
Step 4 (scale row2 by -1, row3 by -1/3 to make pivots 1):
REF: [1 -2 1 -1 1 | 0; 0 0 1 -1 3 | -2; 0 0 0 1 -2 | 1; 0 0 0 0 0 | a+1]

Now read the staircase. Pivots are in columns 1, 3, 4 (leading 1s stepping right); so x1, x3, x4 are basic variables and x2, x5 are free variables. The last row says 0 = a+1: unless a = -1 the system has **no solution** — the Fate-1 detector, produced mechanically. For a = -1, back-substitution or the pivot-column method (express b through pivot columns, working from the rightmost pivot leftward, as the book prescribes) gives the particular solution x = (2, 0, -1, 1, 0) with free variables set to 0, and the full family (next topic) is (2,0,-1,1,0) + λ1(2,1,0,0,0) + λ2(2,0,-1,2,1). [BOOK: mathematics-for-machine-learning, ch.2, §2.3, Example 2.6]

**Why the pivot method for a particular solution works:** in REF the pivot columns have their leading 1 in distinct rows, so matching b's last nonzero component forces the rightmost pivot's coefficient first, then the next, like unstacking cups. **RREF check (Example 2.7):** the matrix [1 3 0 0 3; 0 0 1 0 9; 0 0 0 1 -4] is in RREF: pivots 1,1,1 in columns 1,3,4 stepping right, each alone in its column. [BOOK: mathematics-for-machine-learning, ch.2, §2.3, Example 2.7]

**Cost honesty:** elimination on an n×n system performs on the order of n³ arithmetic operations, which is why the book calls it practical for thousands of variables and impractical for millions (§2.3.4, covered in the algorithms topic). [BOOK: mathematics-for-machine-learning, ch.2, §2.3.4]

## 5. Worked examples (full numbers, nothing skipped; redo the author's examples and add your own)

Example 2.6 is worked step by step above, including the a ≠ -1 inconsistency verdict, the basic/free split (x1,x3,x4 basic; x2,x5 free), and the particular solution (2,0,-1,1,0). Audit the particular solution against the *original* first equation: -2(2) + 4(0) - 2(-1) - (1) + 4(0) = -4 + 0 + 2 - 1 + 0 = -3. Matches the original right-hand side -3. Second equation: 4(2) - 8(0) + 3(-1) - 3(1) + 0 = 8 - 3 - 3 = 2. Matches. [BOOK example; audit DERIVED and repeated by code below.]

A smaller added example showing a pivot swap is forced: solve 0x1 + 2x2 = 6; 3x1 + x2 = 7. The (1,1) entry is 0, so elimination cannot start until rows swap — swapping is not cosmetic, it is sometimes logically required to get a nonzero pivot. After swap: [3 1 | 7; 0 2 | 6], so x2 = 3, then 3x1 = 7 - 3 = 4, x1 = 4/3. Check: 2·3 = 6 OK; 3(4/3) + 3 = 7 OK. Unique solution (4/3, 3); both variables basic, no free variables — consistent with uniqueness. [DERIVED]

And the RREF distinction in one glance: [1 2 0; 0 0 1] is REF and RREF (pivots 1, alone in columns); [1 2 3; 0 1 4] is REF but not RREF (second pivot's column is fine, but the first pivot is not 1... correcting: pivots here are 1 and 1; the pivot in row 2, column 2 has a 2 above it in its column, so the "only nonzero entry in its column" clause fails — not RREF). That clause is exactly what one more elimination pass (row1 -= 2·row2) repairs. [DERIVED]

## 6. Code (runnable, with saved output)

Exact REF by fractions, reproducing Example 2.6's staircase and the a = -1 audit:

```python
from fractions import Fraction as F

def ref(A, b):
    M = [[F(v) for v in row] + [F(b[i])] for i, row in enumerate(A)]
    rows, cols, r = len(M), len(A[0]), 0
    for c in range(cols):
        piv = next((i for i in range(r, rows) if M[i][c] != 0), None)
        if piv is None: continue
        M[r], M[piv] = M[piv], M[r]
        pv = M[r][c]
        M[r] = [v / pv for v in M[r]]
        for i in range(rows):
            if i != r and M[i][c] != 0:
                f = M[i][c]
                M[i] = [v - f * w for v, w in zip(M[i], M[r])]
        r += 1
    return M

A = [[-2,4,-2,-1,4],[4,-8,3,-3,1],[1,-2,1,-1,1],[1,-2,0,-3,4]]
for row in ref(A, [-3,2,0,-1]): print([str(v) for v in row])
x = [2,0,-1,1,0]
print("check eq1:", -2*x[0]+4*x[1]-2*x[2]-x[3]+4*x[4])
print("check eq2:", 4*x[0]-8*x[1]+3*x[2]-3*x[3]+x[4])
```
```text
['1', '-2', '0', '0', '-2', '2']
['0', '0', '1', '0', '1', '-1']
['0', '0', '0', '1', '-2', '1']
['0', '0', '0', '0', '0', '0']
check eq1: -3
check eq2: 2
```

The code computes the full RREF (one step beyond the book's REF): basic variables x1, x3, x4 expressed through free x2, x5 (x1 = 2 + 2x2 + 2x5, x3 = -1 - x5, x4 = 1 + 2x5), with particular solution at x2 = x5 = 0: x = (2, 0, -1, 1, 0), and homogeneous directions (2,1,0,0,0) and (2,0,-1,2,1) — matching the book's (2.47). Consistent.

## 7. Where it shows up in AI/ML (concrete uses)

Every "solve" inside ML is this algorithm or a descendant: least-squares normal equations, the linear systems inside Gaussian-process inference, Kalman filters, and Newton steps are solved by elimination variants (LU/Cholesky factorizations in libraries). The basic/free variable split is the ancestor of a daily ML judgment: parameters the data determines (basic) versus parameters it does not (free — needing regularization, priors, or more data). The "0 = nonzero" REF row is the machine-checkable form of inconsistent training equations, and RREF's pivot pattern is how rank, null space, and feature redundancy are actually computed by software. [DERIVED applications of the book's machinery.]

## 8. Common mistakes and misconceptions

(1) Multiplying a row by 0 or dividing by a pivot that might be 0 — pivots must be nonzero; swap rows or choose a different pivot. (2) Applying a row operation to A but forgetting the same operation on b (the augmented bar is not decoration). (3) Reading solutions off REF as if it were RREF: in REF, basic variables are still entangled with variables above them until back-substitution or further elimination. (4) Declaring "no solution" from a zero row on the left while ignoring the right-hand entry: 0 = 0 is redundant, 0 = nonzero is inconsistent — opposite verdicts. (5) Believing elementary transformations might change the answer set "a little": they are reversible equivalences; the solution set is identical. (6) Confusing basic variables with "important" variables — basic/free is an artifact of column order and pivot choices, not a property of the real-world quantities.

## 9. Interview angle (how it is asked, what a strong answer includes)

Asked as: "How does Gaussian elimination work, and how do you detect no solution / infinitely many solutions?" A strong answer: name the three elementary operations and why they preserve solutions (reversibility; nonzero scaling), describe REF's staircase and pivots, give the three readings (0 = nonzero row → no solution; every column pivoted → unique; free variables present → family parametrized by them), and state the O(n³) cost with the practical consequence (direct methods for thousands, iterative for millions). Naming basic vs free variables and connecting free variables to underdetermined ML models is the depth signal.

## 10. Visualization spec (what should be drawn or interactive, with parameters the user can change)

A row-operation playground: an augmented matrix grid with buttons for the three elementary operations (pick rows, enter a multiplier), a live "solution set unchanged" badge verified by testing a stored solution after each move, and a staircase highlighter that shades the pivot positions and detects REF/RREF, free columns, and inconsistent rows. User parameters: matrix entries, a, and the chosen row operations; the Example 2.6 system is the preset, with a slider for a showing the solution appearing only at a = -1.

## 11. Sources and verification (provenance tags and proof file paths)

[BOOK: mathematics-for-machine-learning, ch.2, §2.3] for the augmented matrix, Example 2.6 with its REF, particular solution, and parameter condition; [BOOK: mathematics-for-machine-learning, ch.2, §2.3.2] for elementary transformations (37-word heading merged here per DEC-006); Definition 2.6, pivots, basic/free variables, RREF, and Example 2.7 from the §2.3 remarks. [DERIVED]: legality/reversibility argument, audits, added examples — machine-checked by §6 code via scripts/code_check.py.
