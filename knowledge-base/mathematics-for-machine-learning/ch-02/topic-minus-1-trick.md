---
id: math.linalg.systems
book: mathematics-for-machine-learning
chapter: 2
section: 2.3.3
tag: must_know
prerequisites: [math.linalg.matrices]
leads_to: [math.linalg.vector-spaces]
---
# The Minus-1 Trick: Reading the Homogeneous Solutions Straight off the RREF

## 1. What it is (plain-language definition, then the formal definition)

In plain language, the minus-1 trick is a reading device: once a matrix is in reduced row-echelon form, you pad it into a square matrix by inserting a row containing a single -1 wherever a pivot is "missing" on the diagonal, and then the columns that contain a pivot 1 can be read downward as solutions of A x = 0 — no back-substitution, no sign puzzles per problem. [BOOK: mathematics-for-machine-learning, ch.2, §2.3.3]

Formally: let A ∈ R^{k×n} be in RREF with no zero rows, pivot columns j_1, ..., j_k. In RREF each pivot column is a standard unit vector e_i (1 in its pivot row, 0 elsewhere in the column), and every non-pivot (free) column can be written as a combination of the pivot columns to its left. Extend A to an n×n matrix Ã by adding, for each free column position, a row whose only nonzero entry is -1 in that free column, inserted so that Ã's diagonal contains only 1s (in pivot positions) and -1s (in free positions). Then: the columns of Ã that contain a diagonal 1 — equivalently, the columns indexed by free variables — are solutions of A x = 0, and together they form a basis (in the sense of §2.6.1) of the solution space of A x = 0, later called the kernel or null space (§2.7.3). [BOOK: mathematics-for-machine-learning, ch.2, §2.3.3]

The book's example: A = [[1, 3, 0, 0, 3], [0, 0, 1, 0, 9], [0, 0, 0, 1, -4]] (already RREF, pivots in columns 1, 3, 4). Free positions are columns 2 and 5, so insert a -1 row at diagonal position 2 and another at position 5:
Ã = [[1, 3, 0, 0, 3], [0, -1, 0, 0, 0], [0, 0, 1, 0, 9], [0, 0, 0, 1, -4], [0, 0, 0, 0, -1]].
Reading the columns that contain a diagonal 1... precisely: read the **free columns** of Ã (columns 2 and 5): column 2 = (3, -1, 0, 0, 0), column 5 = (3, 0, 9, -4, -1). These are the homogeneous directions: solutions of A x = 0 are exactly λ1 (3,-1,0,0,0) + λ2 (3,0,9,-4,-1). [BOOK: mathematics-for-machine-learning, ch.2, §2.3.3, Example 2.8]

## 2. Why it exists (the problem it solves; what breaks without it)

The previous topic produced homogeneous directions by hand-expressing each free column through pivot columns and juggling signs — workable for two free variables, error-prone for ten, and miserable under time pressure (in an exam or an interview whiteboard). The minus-1 trick exists to convert that reasoning into a *mechanical read-off* with a fixed sign convention, so the only remaining task is reaching RREF. What breaks without it is reliability: the most common student error in this whole chapter is a sign or index slip in exactly one homogeneous direction, which then poisons basis, rank, and kernel computations downstream (§2.6, §2.7 all consume these directions). The trick also reveals structure the hand method hides: the homogeneous directions, read as columns of Ã, expose that the solution set's dimension is n - k (columns minus pivots) and that each direction "belongs" to one free variable — it carries -1 (the inserted marker) in its own free coordinate and the free column's pivot-combination coefficients elsewhere. That is the rank-nullity relationship in embryo, seen before it is named. [DERIVED: motivation and structural reading of the book's construction.]

It is worth saying what the trick is not: it is not a new theorem about which vectors solve A x = 0 — the previous topic's theorem already decides that — it is notation engineered so the answer can be read without recomputation, and its correctness is established by verifying the read-off vectors directly (§4).

## 3. Intuition (a physical or visual analogy)

Think of RREF as a seating chart where pivot columns are assigned seats (unit vectors, one per row). Free columns are unseated guests standing in the aisle, each described by how they lean on the seated guests ("column 5 leans 3 on seat 1, 9 on seat 3, -4 on seat 4"). The minus-1 insertion gives each standing guest their own marked floor spot (-1), turning the chart into a full square where every column is one person's complete description. Reading a free column from top to bottom then tells a complete story: "to cancel out, take 3 of pivot 1, 9 of pivot 3, -4 of pivot 4, and -1 of myself" — a recipe for producing zero, written vertically. The reason the read-off works is that in RREF the pivot columns contribute to exactly one equation each, so the bookkeeping that back-substitution would do row-by-row has already been done by the form itself; the -1 is just the guest's own coefficient in their cancellation recipe, fixed at -1 by the way we inserted the row. [DERIVED analogy.]

## 4. The mechanics (every formula, every derivation step, every theorem and its proof or proof idea, step-by-step logic)

**Why the read-off vectors solve A x = 0 (proof idea).** Let f be a free column index. In RREF, column f of A has entries only in pivot rows; say its entry in the row of pivot column j_i is a_{i,f}, which by the RREF structure means column f of A equals Σ_i a_{i,f} · (pivot column j_i), because pivot column j_i is exactly the unit vector e_i. Form the vector v whose pivot-coordinate j_i is a_{i,f}, whose f-coordinate is -1, and whose other free coordinates are 0 — this is precisely the f-th column of Ã (the inserted -1 row contributes the -1 at position f and 0s elsewhere). Then A v = Σ_i a_{i,f} (A e_{j_i} as a combination...) — cleanly: A v equals (combination of pivot columns with coefficients a_{i,f}) plus (-1)·(column f of A) = column f - column f = 0. So every free column of Ã solves the homogeneous system. The book further states these columns form a **basis** of the solution space: they are independent (each has -1 where the others have 0, so no one is a combination of the rest) and they span (any homogeneous solution is determined by its free coordinates, and matching them with the λs then forces the pivot coordinates). Basis and span are defined properly in §2.6.1; the independence-from-the--1-pattern argument can already be verified by eye. [BOOK: mathematics-for-machine-learning, ch.2, §2.3.3 for the construction and basis claim; proof expanded DERIVED.]

**Step-by-step recipe.** (1) Row-reduce A to RREF; delete zero rows conceptually. (2) Mark pivot columns (contain a leading 1 alone in its column) and free columns. (3) Build Ã: keep A's rows in place; insert, in each free diagonal position, a row that is all zeros except -1 in that free column. (4) Read each **free column** of Ã top-to-bottom: that is one homogeneous direction. (5) General homogeneous solution: all λ-combinations of those columns. (6) Audit: multiply A by each read-off vector; the result must be the zero vector. The book's Example 2.7/2.8 pair shows the same answer obtained first "by insight" (expressing non-pivot columns through pivot columns) and then by the trick — they agree, as they must. [BOOK: mathematics-for-machine-learning, ch.2, §2.3.3, Examples 2.7–2.8]

**Consistency with the previous topic:** the (8, 2, -1, 0) direction there came from c3 = 8c1 + 2c2: pivot coefficients (8, 2) in slots 1, 2, then -1 in its own slot 3. The minus-1 trick produces exactly this pattern from the RREF of that matrix. One subtlety the book is careful about and readers often miss: the columns to read are those corresponding to free variables (equivalently, columns that in Ã contain... the columns whose diagonal entry is -1 are the *sources* of directions; the direction itself is that whole column of Ã). Read column 5 of the example again: (3, 0, 9, -4, -1) — pivot slots get (3, 9, -4) from A's column 5 (which reads "3 of pivot col 1, 9 of pivot col 3, -4 of pivot col 4"), free slot 2 gets 0, own slot 5 gets -1. [BOOK: §2.3.3; reading guidance DERIVED.]

## 5. Worked examples (full numbers, nothing skipped; redo the author's examples and add your own)

Book's Example 2.8 audited column by column. Direction v1 = (3, -1, 0, 0, 0): A v1 = row1: 1·3 + 3·(-1) + 0 + 0 + 3·0 = 0; row2: 0 + 0 + 1·0 + 0 + 9·0 = 0; row3: 0. Zero vector, as promised. Direction v2 = (3, 0, 9, -4, -1): row1: 3 + 0 + 0 + 0 + 3·(-1) = 0; row2: 9 + 9·(-1) = 0; row3: -4 + (-4)(-1) = -4 + 4 = 0. Both directions verified against the original A, not just Ã. [BOOK example; audit DERIVED and coded below.]

Why the audit uses A and not Ã: Ã is a reading device with extra invented rows; the claim to check is about A's kernel. Multiplying by Ã's added rows would test nothing about the original system (each added row just returns the -1-scaled free coordinate). [DERIVED]

Added example, built from the previous topic's system A = [[1, 0, 8, -4], [0, 1, 2, 12]], b = (42, 8): A is already RREF (pivots columns 1, 2); free columns 3, 4. Ã inserts -1 rows at positions 3 and 4: Ã = [[1,0,8,-4],[0,1,2,12],[0,0,-1,0],[0,0,0,-1]]. Read free columns: column 3 = (8, 2, -1, 0), column 4 = (-4, 12, 0, -1) — identical to the directions derived by hand in the particular/general topic. The trick and the theory agree on both available examples. [DERIVED application of BOOK §2.3.3 to BOOK §2.3.1.]

A dimension count to internalize: directions = n - (#pivots). Example 2.8: 5 - 3 = 2 directions. Previous topic: 4 - 2 = 2. Example 2.6: 5 - 3 = 2 (λ1, λ2 there). When this count is 0 (pivots in every column), the homogeneous system has only the zero solution and the system's solution, if it exists, is unique — the corollary from the previous topic, now countable at a glance. [DERIVED]

## 6. Code (runnable, with saved output)

```python
def matvec(A, x):
    return [sum(a * v for a, v in zip(row, x)) for row in A]

A = [[1, 3, 0, 0, 3], [0, 0, 1, 0, 9], [0, 0, 0, 1, -4]]
Atilde = [[1, 3, 0, 0, 3], [0, -1, 0, 0, 0], [0, 0, 1, 0, 9], [0, 0, 0, 1, -4], [0, 0, 0, 0, -1]]
cols = [[Atilde[i][j] for i in range(5)] for j in range(5)]
v1, v2 = cols[1], cols[4]   # free columns 2 and 5 (0-indexed 1 and 4)
print("read-off directions:", v1, v2)
print("A v1:", matvec(A, v1), " A v2:", matvec(A, v2))
```
```text
read-off directions: [3, -1, 0, 0, 0] [3, 0, 9, -4, -1]
A v1: [0, 0, 0]  A v2: [0, 0, 0]
```

## 7. Where it shows up in AI/ML (concrete uses)

The solution space of A x = 0 — the kernel/null space — is where this read-off gets used constantly: it is the set of input perturbations a linear model cannot see (adversarial directions for a linear classifier live here), the unidentifiable parameter directions in an underdetermined regression, and the null space whose dimension (nullity) balances rank in the rank-nullity theorem of §2.7. In numerical software the same object is computed by SVD rather than RREF (§4.5), but the interpretation — "these basis directions produce exactly zero output" — is the one learned here, and debugging a model's blind spots still starts by asking what its kernel contains. [DERIVED applications; SVD forward-reference is the book's §4.5.]

## 8. Common mistakes and misconceptions

(1) Reading the pivot columns of Ã instead of the free columns: pivot columns of Ã describe the seated guests, not the cancellation recipes. (2) Applying the trick to a matrix that is only in REF, not RREF: without "pivot alone in its column", the read-off coefficients are wrong; finish the reduction. (3) Dropping the inserted rows' purpose and testing directions against Ã instead of A. (4) Sign slips: the own-coordinate is -1 while pivot coordinates copy A's free column as-is; "fixing" signs by habit breaks the cancellation. (5) Forgetting zero rows of A must be absent/ignored when padding to square — the count of inserted rows is n - k where k counts pivots, not original rows. (6) Believing the trick fails when the first column is free: it does not; the insertion logic is positional, and the first inserted -1 row simply becomes the top row of Ã.

## 9. Interview angle (how it is asked, what a strong answer includes)

Asked as: "Given this RREF matrix, find the null space." A strong candidate marks pivot vs free columns, writes one direction per free variable with the free column's pivot-coefficients copied and -1 (or 1, with consistent sign convention) in the own slot, states the dimension as n minus the number of pivots, and verifies one direction against A. Speed matters here, and that is exactly what the trick buys; narrating the verification step signals rigor rather than memorized pattern-matching.

## 10. Visualization spec (what should be drawn or interactive, with parameters the user can change)

An RREF highlighter: the matrix renders with pivot columns shaded and free columns outlined; clicking "apply minus-1 trick" animates the -1 rows sliding into their diagonal positions to form the square Ã, then each free column lifts out as a direction vector card with a live "A·v = 0" check badge. User parameters: matrix entries (presets: Examples 2.7/2.8 and §2.3.1), a "reduce to RREF" button, λ sliders forming combinations of the read-off directions with residual display.

## 11. Sources and verification (provenance tags and proof file paths)

[BOOK: mathematics-for-machine-learning, ch.2, §2.3.3] for the construction (2.51)–(2.52), the basis claim, and Example 2.8 (2.53)–(2.55); Example 2.7's RREF from §2.3. [DERIVED]: correctness proof idea, audits, cross-check against §2.3.1 — machine-checked by §6 code via scripts/code_check.py.
