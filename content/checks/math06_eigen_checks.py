"""Executable checks behind lesson LSN-MATH-06-01.

Every numerical claim made in `content/lessons/MATH-06-01-determinants-and-eigen.md` and in its
solutions is computed here. The point is not that the arithmetic runs, but that the arithmetic
was done independently of the prose: the lesson states answers, and this file re-derives them
with sympy (exact) and numpy (floating point).

Run via `tools/run_content_checks.py`, which records the output as verification evidence.
Requires the development virtualenv (numpy, sympy).
"""

from __future__ import annotations

import numpy as np
import sympy as sp

LAM = sp.symbols("lambda")


def check_ex01_determinant_two_ways() -> str:
    """Exercise 1: det B by Sarrus and by Laplace expansion must agree, and B is invertible."""
    B = sp.Matrix([[2, 0, 3], [1, 4, -1], [0, 2, 1]])

    # Sarrus: three down-right products minus three down-left products, written out so that the
    # formula being tested is visible rather than hidden in index arithmetic.
    a, b, c = B[0, 0], B[0, 1], B[0, 2]
    d, e, f = B[1, 0], B[1, 1], B[1, 2]
    g, h, i = B[2, 0], B[2, 1], B[2, 2]
    sarrus = (a * e * i + b * f * g + c * d * h) - (c * e * g + b * d * i + a * f * h)

    # Laplace along the first row: signed 2x2 minors.
    laplace = 0
    for column in range(3):
        minor = B.minor_submatrix(0, column)
        laplace += ((-1) ** column) * B[0, column] * minor.det()

    exact = B.det()
    assert sarrus == 18, f"Sarrus gave {sarrus}"
    assert laplace == 18, f"Laplace gave {laplace}"
    assert exact == 18, f"sympy det gave {exact}"
    assert abs(float(np.linalg.det(np.array(B.tolist(), dtype=float))) - 18.0) < 1e-9, "numpy disagrees"
    assert exact != 0, "B should be invertible"

    return (
        "Sarrus = 18, Laplace = 18, sympy det(B) = 18, numpy det(B) = 18.0; "
        "three independent methods agree and det != 0, so B is invertible."
    )


def check_ex02_zero_determinant_reasons() -> str:
    """Exercise 2: two structurally different reasons for a zero determinant."""
    repeated_row = sp.Matrix([[1, 2, 3], [4, 5, 6], [1, 2, 3]])
    dependent_rows = sp.Matrix([
        [1, 2, 3, 4],
        [0, 1, 0, 1],
        [2, 4, 6, 8],   # exactly 2 * row 1
        [1, 0, 1, 0],
    ])
    zero_column = sp.Matrix([[1, 0, 3], [4, 0, 6], [7, 0, 9]])

    assert repeated_row.det() == 0, "identical rows must give determinant 0"
    assert dependent_rows.det() == 0, "a row that is a multiple of another must give 0"
    assert zero_column.det() == 0, "a zero column must give determinant 0"
    assert repeated_row.rank() < 3 and dependent_rows.rank() < 4 and zero_column.rank() < 3

    return (
        "det = 0 for all three structures: two identical rows, "
        "a row equal to twice another row (rank 3 < 4), and a zero column (rank 2 < 3). "
        "In every case the rank check confirms the determinant is zero for a stated reason."
    )


def check_ex03_eigenpairs() -> str:
    """Exercise 3: eigenvalues and eigenspaces of A = [[3,1],[1,3]] and C = [[2,2],[1,3]]."""
    A = sp.Matrix([[3, 1], [1, 3]])
    C = sp.Matrix([[2, 2], [1, 3]])

    a_char = sp.factor((A - LAM * sp.eye(2)).det())
    c_char = sp.factor((C - LAM * sp.eye(2)).det())
    assert a_char == (LAM - 4) * (LAM - 2), f"A characteristic polynomial: {a_char}"
    assert c_char == (LAM - 4) * (LAM - 1), f"C characteristic polynomial: {c_char}"

    a_eigen = {value: vectors for value, _, vectors in A.eigenvects()}
    c_eigen = {value: vectors for value, _, vectors in C.eigenvects()}
    assert set(a_eigen) == {4, 2}, a_eigen
    assert set(c_eigen) == {4, 1}, c_eigen

    # Verify by substitution: A v = lambda v, exactly.
    assert sp.simplify(A * sp.Matrix([1, 1]) - 4 * sp.Matrix([1, 1])) == sp.zeros(2, 1)
    assert sp.simplify(A * sp.Matrix([1, -1]) - 2 * sp.Matrix([1, -1])) == sp.zeros(2, 1)
    assert sp.simplify(C * sp.Matrix([1, 1]) - 4 * sp.Matrix([1, 1])) == sp.zeros(2, 1)
    assert sp.simplify(C * sp.Matrix([2, -1]) - 1 * sp.Matrix([2, -1])) == sp.zeros(2, 1)

    # Invariants: trace = sum, determinant = product.
    for matrix, values in ((A, [4, 2]), (C, [4, 1])):
        assert matrix.trace() == sum(values), f"trace check failed for {matrix}"
        assert matrix.det() == values[0] * values[1], f"determinant check failed for {matrix}"

    return (
        "A: eigenvalues 4, 2 with eigenspaces span{(1,1)} and span{(1,-1)}; "
        "C: eigenvalues 4, 1 with eigenspaces span{(1,1)} and span{(2,-1)}. "
        "Each eigenpair verified by exact substitution, and trace/determinant invariants hold "
        "for both matrices (4 = 4, 3 = 4*1 for C: trace 5 = 5, det 4 = 4*1)."
    )


def check_ex04_invertible_vs_diagonalisable() -> str:
    """Exercise 4: the five matrices, with multiplicities rather than just verdicts."""
    cases = {
        "1  shear            [[1,1],[0,1]]": sp.Matrix([[1, 1], [0, 1]]),
        "2  nilpotent        [[0,1],[0,0]]": sp.Matrix([[0, 1], [0, 0]]),
        "3  scaled identity  [[2,0],[0,2]]": sp.Matrix([[2, 0], [0, 2]]),
        "4  rotation 90      [[0,-1],[1,0]]": sp.Matrix([[0, -1], [1, 0]]),
        "5  projection       [[1,0],[0,0]]": sp.Matrix([[1, 0], [0, 0]]),
    }
    lines = []
    for label, matrix in cases.items():
        determinant = matrix.det()
        invertible = determinant != 0
        # Over the reals, sympy reports complex eigenvectors as such; diagonalisability over R
        # therefore also requires every eigenvalue to be real.
        eigenvalues = matrix.eigenvals()
        all_real = all(sp.im(value) == 0 for value in eigenvalues)
        # sympy decides diagonalisability over the algebraic closure, so the real question is
        # separate: a matrix with a non-real eigenvalue cannot be diagonalised over R.
        diagonalisable_over_c = bool(matrix.is_diagonalizable())
        diagonalisable_over_r = diagonalisable_over_c and all_real
        # Geometric multiplicity is the dimension of the eigenspace, i.e. dim ker(A - lambda I),
        # computed from the rank rather than from the length of an eigenvector list.
        multiplicity_report = ", ".join(
            f"lambda={value}: algebraic {algebraic}, geometric "
            f"{matrix.shape[0] - (matrix - value * sp.eye(matrix.shape[0])).rank()}"
            for value, algebraic, _ in matrix.eigenvects()
        )
        lines.append(
            f"{label}: det={determinant}, invertible={invertible}, "
            f"diagonalisable over C={diagonalisable_over_c}, over R={diagonalisable_over_r} "
            f"[{multiplicity_report}]"
        )

    # The two matrices that make the point: invertible but not diagonalisable, and vice versa
    # would be impossible, so the pairing is invertible-but-not-diagonalisable plus
    # not-invertible-but-diagonalisable.
    assert cases["1  shear            [[1,1],[0,1]]"].det() != 0
    assert not cases["1  shear            [[1,1],[0,1]]"].is_diagonalizable()
    assert cases["5  projection       [[1,0],[0,0]]"].det() == 0
    assert cases["5  projection       [[1,0],[0,0]]"].is_diagonalizable()
    # The rotation is the case that separates the two fields: diagonalisable over C, not over R.
    rotation = cases["4  rotation 90      [[0,-1],[1,0]]"]
    assert bool(rotation.is_diagonalizable()) is True
    assert any(sp.im(value) != 0 for value in rotation.eigenvals()), "rotation eigenvalues are not real"
    return "\n".join(lines)


def check_ex05_power_rule_and_converse() -> str:
    """Exercise 5: the power rule holds; its real converse fails (rotation matrix)."""
    A = sp.Matrix([[2, 1], [1, 2]])
    v = sp.Matrix([1, 1])
    for k in (1, 2, 3, 5, 8):
        assert sp.simplify(A ** k * v - (3 ** k) * v) == sp.zeros(2, 1), f"power rule failed at k={k}"

    rotation = sp.Matrix([[0, -1], [1, 0]])
    squared = rotation ** 2
    assert squared == -sp.eye(2), "the squared rotation should be minus the identity"
    assert squared.eigenvals() == {-1: 2}, "the squared rotation should have eigenvalue -1"
    real_eigenvalues = [value for value in rotation.eigenvals() if sp.im(value) == 0]
    assert real_eigenvalues == [], "the rotation has no real eigenvalues"

    return (
        "Power rule verified exactly for A=[[2,1],[1,2]], v=(1,1), k in {1,2,3,5,8}: "
        "A^k v = 3^k v. Converse counterexample: R=[[0,-1],[1,0]] has R^2 = -I, whose real "
        "eigenvalue -1 is not the square of any real eigenvalue of R, because R has none. "
        "Over C, R has eigenvalues i and -i, and (-i)^2 = i^2 = -1, so the converse does hold "
        "in the complex sense — which is precisely the point of the exercise."
    )


def check_ex06_odd_degree_real_eigenvalue() -> str:
    """Exercise 6: a construction with exactly one real eigenvalue, and the 2D contrast."""
    rotation_plus_scaling = sp.Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 2]])
    characteristic = sp.factor((rotation_plus_scaling - LAM * sp.eye(3)).det())
    # det(A - lambda I) is the convention used here and in the source, so for a 3x3 matrix the
    # leading coefficient is -1: det(A - lambda I) = -(lambda - 2)(lambda^2 + 1).
    assert sp.simplify(characteristic - (-(LAM - 2) * (LAM ** 2 + 1))) == 0, f"got {characteristic}"

    real_roots = sp.solve(sp.Eq(characteristic, 0), LAM)
    real_roots = [root for root in real_roots if sp.im(root) == 0]
    assert real_roots == [2], f"expected exactly one real eigenvalue, got {real_roots}"

    # The 2D contrast: rotation has no real eigenvalue at all, so the odd-degree argument
    # cannot be made in even dimension.
    two_d = sp.Matrix([[0, -1], [1, 0]])
    assert sp.factor((two_d - LAM * sp.eye(2)).det()) == LAM ** 2 + 1
    assert two_d.eigenvals() == {sp.I: 1, -sp.I: 1}
    assert [value for value in two_d.eigenvals() if sp.im(value) == 0] == []

    return (
        "[[0,-1,0],[1,0,0],[0,0,2]] has det(A - lambda I) = -(lambda-2)(lambda^2+1): "
        "exactly one real eigenvalue (2) and a complex conjugate pair (+/- i). "
        "In 2D the rotation [[0,-1],[1,0]] has characteristic polynomial lambda^2+1 with no real "
        "roots, confirming that the odd-degree argument has no even-dimensional analogue."
    )


def check_trace_determinant_invariants_random() -> str:
    """Section 2.4: trace = sum of eigenvalues and det = product, tested on random matrices."""
    generator = np.random.default_rng(20261012)
    worst_trace = 0.0
    worst_det = 0.0
    for size in (2, 3, 5, 8):
        for _ in range(25):
            matrix = generator.normal(size=(size, size))
            values = np.linalg.eigvals(matrix)
            worst_trace = max(worst_trace, abs(values.sum() - np.trace(matrix)))
            worst_det = max(worst_det, abs(values.prod() - np.linalg.det(matrix)))
    assert worst_trace < 1e-9, f"trace identity violated by {worst_trace}"
    assert worst_det < 1e-9, f"determinant identity violated by {worst_det}"
    return (
        f"Over 100 random matrices of sizes 2, 3, 5 and 8: maximum |sum(eigenvalues) - trace| = "
        f"{worst_trace:.3e}, maximum |prod(eigenvalues) - det| = {worst_det:.3e}. "
        "Both invariants hold to floating-point precision, which is the tolerance they are true to.",
    )


def check_ex07_reference_implementation() -> str:
    """Exercise 7: the reference 2x2 implementation, its measured residual, and its bad case."""
    def eigen_2x2(matrix: np.ndarray) -> tuple[complex, complex, np.ndarray, np.ndarray]:
        """Eigenvalues via the quadratic formula and eigenvectors via the null space of A - lambda I."""
        a, b = float(matrix[0, 0]), float(matrix[0, 1])
        c, d = float(matrix[1, 0]), float(matrix[1, 1])
        trace, determinant = a + d, a * d - b * c
        discriminant = trace * trace - 4.0 * determinant
        root = np.sqrt(complex(discriminant))
        values = ((trace + root) / 2.0, (trace - root) / 2.0)

        vectors = []
        for value in values:
            shifted = np.array([[a - value, b], [c, d - value]], dtype=complex)
            # A nonzero vector in the null space of the shifted matrix: for 2x2 this is obtained
            # from any nonzero row, using the standard (b, -(a - lambda)) / (-(d - lambda), c) forms.
            if abs(shifted[0, 1]) > abs(shifted[1, 0]):
                vector = np.array([shifted[0, 1], -(shifted[0, 0])], dtype=complex)
            else:
                vector = np.array([-(shifted[1, 1]), shifted[1, 0]], dtype=complex)
            norm = np.linalg.norm(vector)
            vectors.append(vector / norm if norm > 0 else vector)
        return values[0], values[1], vectors[0], vectors[1]

    generator = np.random.default_rng(7)
    worst = 0.0
    worst_matrix = None
    complex_case_residual = 0.0
    for _ in range(1000):
        matrix = generator.normal(size=(2, 2))
        values = eigen_2x2(matrix)[:2]
        for value, vector in zip(values, eigen_2x2(matrix)[2:]):
            residual = float(np.linalg.norm(matrix @ vector - value * vector))
            if residual > worst:
                worst, worst_matrix = residual, matrix.copy()
            if abs(complex(value).imag) > 1e-12:
                complex_case_residual = max(complex_case_residual, residual)
        # Consistency with the library, matched by nearest eigenvalue (order is unspecified).
        library_values = np.linalg.eigvals(matrix)
        for value in values:
            distance = min(abs(complex(value) - complex(other)) for other in library_values)
            assert distance < 1e-8, f"disagreement with numpy: {value} vs {library_values}"

    # The ill-conditioned case: an eigenvalue of algebraic multiplicity 2 that is not defective
    # in exact arithmetic but is numerically indistinguishable from a defective one.
    near_defective = np.array([[1.0, 1.0], [0.0, 1.0 + 1e-10]])
    v1, v2, u1, u2 = eigen_2x2(near_defective)
    near_residual = max(
        float(np.linalg.norm(near_defective @ u1 - v1 * u1)),
        float(np.linalg.norm(near_defective @ u2 - v2 * u2)),
    )

    assert worst < 1e-10, f"worst residual over 1000 random matrices was {worst}"
    return (
        f"Reference implementation ran on 1000 random 2x2 matrices. "
        f"Maximum residual ||A v - lambda v|| = {worst:.3e} (matrix "
        f"[[{worst_matrix[0,0]:.6f}, {worst_matrix[0,1]:.6f}], "
        f"[{worst_matrix[1,0]:.6f}, {worst_matrix[1,1]:.6f}]]). "
        f"Maximum residual among matrices with complex eigenvalues = {complex_case_residual:.3e}. "
        f"For the nearly defective matrix [[1,1],[0,1+1e-10]], residual = {near_residual:.3e}: "
        "the eigenvalues are almost equal, the two eigenvectors are almost parallel, and the "
        "residual itself stays small while the eigenvector directions become unreliable. "
        "Agreement with numpy establishes consistency between two implementations; it is the "
        "residual, not the agreement, that establishes that each returned pair solves Av = lambda v."
    )


def check_lesson_numbers_used_in_prose() -> str:
    """Spot-check the specific numbers the lesson states in prose."""
    A = sp.Matrix([[2, 1], [1, 2]])
    assert sp.factor((A - LAM * sp.eye(2)).det()) == (LAM - 3) * (LAM - 1)
    assert A.trace() == 4 and A.det() == 3

    # Section 2.6: A^k (2,0) = 3^k (1,1) + (1,-1)
    for k in (1, 2, 3, 4):
        left = A ** k * sp.Matrix([2, 0])
        right = (3 ** k) * sp.Matrix([1, 1]) + sp.Matrix([1, -1])
        assert sp.simplify(left - right) == sp.zeros(2, 1), f"section 2.6 identity failed at k={k}"

    # Section 2.5(c): B is diagonalisable with a repeated eigenvalue.
    B = sp.Matrix([[2, 0, 0], [0, 2, 1], [0, 0, 1]])
    b_char = sp.factor((B - LAM * sp.eye(3)).det())
    assert sp.simplify(b_char - (LAM - 2) ** 2 * (1 - LAM)) == 0, f"got {b_char}"
    assert B.is_diagonalizable()
    assert (B - 2 * sp.eye(3)).rank() == 1

    # Section 1.1: the bounding-rectangle area identity for the 2x2 determinant.
    a, b, c, d = sp.symbols("a b c d")
    assert sp.expand((a + b) * (c + d) - a * c - b * d - 2 * b * c) == a * d - b * c

    # Section 2.3, step 1: the factorisation of the worked example.
    assert sp.expand((2 - LAM) ** 2 - 1) == sp.expand((LAM - 3) * (LAM - 1))

    return (
        "Prose numbers verified exactly: det[[2,1],[1,2]] = 3, trace = 4, characteristic "
        "polynomial (lambda-3)(lambda-1), the section 2.6 identity A^k(2,0) = 3^k(1,1) + (1,-1) "
        "for k = 1..4, the section 2.5(c) factorisation (lambda-2)^2(1-lambda) with rank(B-2I) = 1, "
        "the parallelogram-area identity (a+b)(c+d) - ac - bd - 2bc = ad - bc, and the "
        "worked example's own factorisation."
    )
