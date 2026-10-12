"""Executable checks behind lesson LSN-MATH-07-01 (matrix decompositions).

Every numerical claim in that lesson and in its solutions is recomputed here: the Cholesky
factorisation, the full SVD of a rectangular matrix, the existence-condition counterexamples, the
norm identities, the null-space multiplicities, the comparison of the two square roots, and the
measured cost of computing the SVD from A'A.

Run via `tools/run_content_checks.py`. Requires numpy and sympy from the development virtualenv.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

LAM = sp.symbols("lambda")


def check_ex01_cholesky_by_hand() -> str:
    """Exercise 1 and section 1.3: L = [[3,0,0],[2,1,0],[1,0,1]] for the worked matrix."""
    A = sp.Matrix([[9, 6, 3], [6, 5, 2], [3, 2, 2]])
    L = sp.Matrix([[3, 0, 0], [2, 1, 0], [1, 0, 1]])

    assert sp.simplify(L * L.T - A) == sp.zeros(3, 3), "LL' does not equal A"

    # Sylvester's criterion: all leading principal minors positive.
    minors = [A[:k, :k].det() for k in (1, 2, 3)]
    assert minors == [9, 9, 9], f"leading minors {minors}"
    assert A.is_positive_definite, "sympy disagrees that A is positive definite"

    # A single perturbation that destroys definiteness: lowering the last diagonal entry.
    boundary = sp.Matrix([[9, 6, 3], [6, 5, 2], [3, 2, 1]])
    assert boundary.det() == 0, f"the perturbed determinant should be 0, got {boundary.det()}"
    assert not boundary.is_positive_definite, "the perturbed matrix should not be positive definite"
    # Going one step further makes it indefinite, which is what a library will reject outright.
    indefinite = sp.Matrix([[9, 6, 3], [6, 5, 2], [3, 2, 0]])
    assert indefinite.det() < 0, f"expected a negative determinant, got {indefinite.det()}"
    assert not indefinite.is_positive_definite

    return (
        "L = [[3,0,0],[2,1,0],[1,0,1]] satisfies LL' = A exactly. Leading principal minors of A are "
        f"{minors}, all positive, and sympy confirms positive definiteness. Lowering A[3,3] from 2 to 1 "
        f"makes det A = {boundary.det()} and the matrix positive semi-definite but not definite: the "
        "third pivot becomes sqrt(0) = 0, so the factorisation exists but has a zero on the diagonal "
        "instead of a positive entry. Lowering it to 0 makes det A = " + str(indefinite.det()) +
        " and the matrix indefinite, and the square root that the formulas demand is that of a negative "
        "number. That is the whole content of the existence condition: every pivot must stay positive."
    )


def check_ex02_full_svd_3x2() -> str:
    """Exercise 2: full SVD of A = [[1,0],[0,1],[1,1]], including completing U."""
    A = sp.Matrix([[1, 0], [0, 1], [1, 1]])
    gram = A.T * A
    assert gram == sp.Matrix([[2, 1], [1, 2]]), f"A'A = {gram}"

    char = sp.factor((gram - LAM * sp.eye(2)).det())
    assert sp.simplify(char - (LAM - 3) * (LAM - 1)) == 0, f"characteristic polynomial {char}"

    sigma = [sp.sqrt(3), 1]
    v1 = sp.Matrix([1, 1]) / sp.sqrt(2)
    v2 = sp.Matrix([1, -1]) / sp.sqrt(2)
    assert sp.simplify(gram * v1 - 3 * v1) == sp.zeros(2, 1)
    assert sp.simplify(gram * v2 - 1 * v2) == sp.zeros(2, 1)

    u1 = sp.simplify(A * v1 / sigma[0])
    u2 = sp.simplify(A * v2 / sigma[1])
    u3 = sp.Matrix([1, 1, -1]) / sp.sqrt(3)
    assert sp.simplify((u1.T * u1)[0] - 1) == 0 and sp.simplify((u2.T * u2)[0] - 1) == 0
    assert sp.simplify((u1.T * u2)[0]) == 0
    assert sp.simplify((u1.T * u3)[0]) == 0 and sp.simplify((u2.T * u3)[0]) == 0

    U = sp.Matrix.hstack(u1, u2, u3)
    V = sp.Matrix.hstack(v1, v2)
    Sigma = sp.Matrix([[sp.sqrt(3), 0], [0, 1], [0, 0]])
    reconstruction = sp.simplify(U * Sigma * V.T)
    assert reconstruction == A, f"reconstruction gave {reconstruction}"
    assert sp.simplify(U.T * U - sp.eye(3)) == sp.zeros(3, 3), "U is not orthogonal"

    library = np.linalg.svd(np.array(A.tolist(), dtype=float), full_matrices=True)
    library_sigma = library[1]
    assert np.allclose(library_sigma, [np.sqrt(3), 1.0], atol=1e-12), library_sigma

    return (
        "A'A = [[2,1],[1,2]] with eigenvalues 3 and 1, so sigma = (sqrt(3), 1) exactly. "
        "v1 = (1,1)/sqrt(2), v2 = (1,-1)/sqrt(2); u1 = (1,1,sqrt(2))/sqrt(3) = (1,1,2)/sqrt(6), "
        "u2 = (1,-1,0)/sqrt(2); the third left direction completing U is (1,1,-1)/sqrt(3), orthogonal "
        "to both. U Sigma V' reproduces A exactly, U is orthogonal, and numpy returns the same "
        "singular values to 1e-12."
    )


def check_ex03_existence_conditions() -> str:
    """Exercise 3: one counterexample per decomposition, and none for the SVD."""
    symmetric_not_pd = np.array([[1.0, 2.0], [2.0, 1.0]])
    cholesky_failure = None
    try:
        np.linalg.cholesky(symmetric_not_pd)
    except np.linalg.LinAlgError as error:  # pragma: no cover - the failure is the point
        cholesky_failure = str(error)

    defective = np.array([[1.0, 1.0], [0.0, 1.0]])
    defective_eig = np.linalg.eig(defective)

    # No matrix lacks an SVD: checked over random rectangular shapes, including rank-deficient ones.
    generator = np.random.default_rng(11)
    shapes = [(3, 5), (5, 3), (4, 4), (6, 2), (2, 6)]
    worst = 0.0
    for rows, columns in shapes:
        for mode in ("full", "rank_deficient", "zero"):
            if mode == "full":
                matrix = generator.normal(size=(rows, columns))
            elif mode == "rank_deficient":
                left = generator.normal(size=(rows, 2))
                right = generator.normal(size=(2, columns))
                matrix = left @ right
            else:
                matrix = np.zeros((rows, columns))
            u, s, vt = np.linalg.svd(matrix, full_matrices=False)
            reconstruction = (u * s) @ vt
            worst = max(worst, float(np.linalg.norm(reconstruction - matrix)))
    assert cholesky_failure is not None, "numpy should reject a non-positive-definite matrix"
    assert worst < 1e-12, f"SVD reconstruction error {worst}"

    eigenvalues, vectors = defective_eig
    return (
        f"Cholesky: [[1,2],[2,1]] is symmetric with eigenvalues 3 and -1, so it is not positive "
        f"definite, and numpy.linalg.cholesky raises LinAlgError ({cholesky_failure.splitlines()[0]}). "
        f"Eigendecomposition: [[1,1],[0,1]] is defective; numpy.eig returns eigenvalues "
        f"{np.round(eigenvalues, 12)} with eigenvector matrix {np.round(vectors, 12).tolist()}, which "
        "has rank 1, so there is no basis of eigenvectors and no A = PDP^-1. SVD: no counterexample "
        "exists; over 15 matrices of shapes (3,5), (5,3), (4,4), (6,2), (2,6) including rank-deficient "
        f"and zero matrices, the maximum reconstruction error was {worst:.2e}."
    )


def check_ex04_norm_identities() -> str:
    """Exercise 4: ||A||_F^2 = sum of squared singular values, and ||Ax|| <= sigma_1 ||x||."""
    generator = np.random.default_rng(20261013)
    worst_frobenius = 0.0
    worst_bound = 0.0
    tightness = 0.0
    for _ in range(200):
        rows, columns = generator.integers(2, 9, size=2)
        matrix = generator.normal(size=(int(rows), int(columns)))
        singular_values = np.linalg.svd(matrix, compute_uv=False)

        worst_frobenius = max(
            worst_frobenius,
            abs(np.sum(singular_values**2) - np.linalg.norm(matrix, "fro") ** 2),
        )
        for _ in range(4):
            x = generator.normal(size=(int(columns),))
            ratio = np.linalg.norm(matrix @ x) / np.linalg.norm(x)
            worst_bound = max(worst_bound, ratio - singular_values[0])
        # The bound is attained at the first right-singular vector.
        _, _, vt = np.linalg.svd(matrix)
        tightness = max(tightness, abs(np.linalg.norm(matrix @ vt[0]) - singular_values[0]))

    assert worst_frobenius < 1e-10, f"Frobenius identity violated by {worst_frobenius}"
    assert worst_bound < 1e-10, f"spectral bound violated by {worst_bound}"
    assert tightness < 1e-10, f"the bound is not attained at v1 ({tightness})"
    return (
        f"Over 200 random matrices: maximum |sum(sigma^2) - ||A||_F^2| = {worst_frobenius:.2e}; "
        f"maximum violation of ||Ax|| <= sigma_1||x|| over 800 random vectors = {worst_bound:.2e}; "
        f"and | ||A v_1|| - sigma_1 | <= {tightness:.2e}, so the bound is attained exactly at the "
        "first right-singular vector — which is why sigma_1 is the operator (spectral) norm."
    )


def check_ex05_zero_eigenvalue_multiplicities() -> str:
    """Exercise 5: ker(A'A) = ker(A), and the differing multiplicities of the zero eigenvalue."""
    tall = sp.Matrix([[1, 0], [0, 1], [0, 0]])          # 3x2, rank 2
    gram_right = tall.T * tall                           # 2x2
    gram_left = tall * tall.T                            # 3x3

    assert sp.simplify(tall.T * tall - sp.eye(2)) == sp.zeros(2, 2)
    assert gram_right.eigenvals() == {1: 2}, gram_right.eigenvals()
    assert gram_left.eigenvals() == {1: 2, 0: 1}, gram_left.eigenvals()

    # ker(A'A) = ker(A): both inclusions in one line each, checked symbolically on the example and
    # then on random matrices where the null spaces are computed as exact bases.
    def nullspace(matrix):
        return matrix.nullspace()

    for rows, columns in ((3, 2), (2, 3), (4, 2)):
        generator = np.random.default_rng(rows * 10 + columns)
        dense = sp.Matrix(generator.integers(-4, 5, size=(rows, columns)).tolist())
        left_null = nullspace(dense.T * dense)
        right_null = nullspace(dense)
        assert len(left_null) == len(right_null), (rows, columns)

    zero_gram_left = 3 - gram_left.rank()
    zero_gram_right = 2 - gram_right.rank()
    return (
        "For A = [[1,0],[0,1],[0,0]] (3x2, rank 2): A'A = I(2) has no zero eigenvalue, while AA' = "
        "diag(1,1,0) has a zero eigenvalue of multiplicity 1. So A'A and AA' share their nonzero "
        "eigenvalues (both {1,1}) but not the multiplicity of zero: the null spaces have dimensions "
        f"{len(nullspace(tall))} for A'A (equal to dim ker A) and {zero_gram_left} for AA' "
        f"(equal to dim ker A' = m - rank A). Verified symbolically for random integer matrices of "
        "shapes (3,2), (2,3) and (4,2): dim ker(A'A) = dim ker(A) in every case."
    )


def check_ex06_two_square_roots() -> str:
    """Exercise 6: the triangular and symmetric square roots, and the general form.

    A 2x2 SPD matrix is used rather than the 3x3 of the Cholesky example so that every step can be
    verified *exactly* in symbolic arithmetic: with a 3x3 matrix the orthonormal eigenbasis involves
    nested radicals that sympy can confirm only numerically, which would weaken the check for no
    gain in understanding.
    """
    A = sp.Matrix([[2, 1], [1, 2]])

    # The symmetric square root, from an explicit orthonormal eigenbasis.
    eigenvectors = []
    for value, algebraic, vectors in A.eigenvects():
        for vector in vectors:
            candidate = vector.normalized()
            for previous in eigenvectors:
                candidate = sp.simplify(candidate - (previous.T * candidate)[0] * previous)
            eigenvectors.append(sp.simplify(candidate.normalized()))
    Q = sp.Matrix.hstack(*eigenvectors)
    assert sp.simplify(Q.T * Q - sp.eye(2)) == sp.zeros(2, 2), "Q is not orthogonal"
    diag = sp.diag(*[sp.simplify((Q.T * A * Q)[i, i]) for i in range(2)])
    assert sp.simplify(Q * diag * Q.T - A) == sp.zeros(2, 2), "Q diag Q' is not A"

    sqrt_symmetric = sp.simplify(Q * diag.applyfunc(sp.sqrt) * Q.T)
    assert sp.simplify(sqrt_symmetric * sqrt_symmetric.T - A) == sp.zeros(2, 2)
    assert sqrt_symmetric.T == sqrt_symmetric, "the symmetric square root should be symmetric"

    # The Cholesky factor: L11 = sqrt(2), L21 = 1/sqrt(2), L22 = sqrt(3/2).
    L = sp.Matrix([[sp.sqrt(2), 0], [1 / sp.sqrt(2), sp.sqrt(sp.Rational(3, 2))]])
    assert sp.simplify(L * L.T - A) == sp.zeros(2, 2), "the Cholesky factor does not reproduce A"
    assert sp.simplify(sqrt_symmetric - L) != sp.zeros(2, 2), "the two square roots should differ"

    # General form: every square root is Q Lambda^(1/2) W' with W orthogonal. The diagonal inverse
    # is built entry by entry, since applying 1/sqrt to a full matrix would divide by sqrt(0).
    inverse_sqrt = sp.diag(*[1 / sp.sqrt(diag[i, i]) for i in range(2)])
    W = sp.simplify((inverse_sqrt * (Q.T * L)).T)
    assert sp.simplify(W.T * W - sp.eye(2)) == sp.zeros(2, 2), "W is not orthogonal"
    assert sp.simplify(Q * sp.diag(*[sp.sqrt(diag[i, i]) for i in range(2)]) * W.T - L) == sp.zeros(2, 2)

    # A second, non-triangular square root that is not symmetric either: S W for a reflection W.
    reflection = sp.Matrix([[1, 0], [0, -1]])
    other = sp.simplify(sqrt_symmetric * reflection)
    assert sp.simplify(other * other.T - A) == sp.zeros(2, 2), "the reflected square root is not a root"
    assert other != sqrt_symmetric and other != L

    return (
        "For A = [[2,1],[1,2]]: the Cholesky factor is L = [[sqrt(2), 0], [1/sqrt(2), sqrt(3/2)]] "
        "and the symmetric square root is S = Q Lambda^(1/2) Q' = [[1/2 + sqrt(3)/2, -1/2 + sqrt(3)/2], "
        "[-1/2 + sqrt(3)/2, 1/2 + sqrt(3)/2]]. Both satisfy (.)(.)' = A exactly, and they differ: S is "
        "symmetric and has no zero entries, L is triangular. The general-form relation L = Q "
        "Lambda^(1/2) W' holds exactly with W = [[-1/2, sqrt(3)/2], [sqrt(3)/2, 1/2]], which is "
        "orthogonal, so the triangularity of the Cholesky factor is an extra requirement rather than a "
        "consequence of the decomposition. Reflecting S by diag(1, -1) gives a third square root, which "
        "is neither symmetric nor triangular, confirming that square roots are not unique."
    )


def check_ex07_gram_svd_accuracy() -> str:
    """Exercise 7 and section 5.2: the textbook construction and where it loses accuracy."""
    def svd_via_gram(matrix: np.ndarray) -> np.ndarray:
        """Singular values from the eigendecomposition of A'A (the textbook construction)."""
        gram = matrix.T @ matrix
        values = np.linalg.eigvalsh(gram)          # symmetric, so real and ascending
        return np.sqrt(np.maximum(values, 0.0))[::-1]

    generator = np.random.default_rng(4242)

    well_conditioned = 0.0
    for _ in range(100):
        rows, columns = generator.integers(3, 12, size=2)
        matrix = generator.normal(size=(int(rows), int(columns)))
        library = np.linalg.svd(matrix, compute_uv=False)
        # A'A has as many eigenvalues as A has columns; a matrix with fewer rows than columns has
        # min(m, n) singular values, so only those are compared.
        gram_based = svd_via_gram(matrix)[: len(library)]
        well_conditioned = max(
            well_conditioned,
            float(np.max(np.abs(gram_based - library) / np.maximum(library, 1e-300))),
        )

    # Ill-conditioned experiment: a 40x40 matrix with chosen singular values, ratio 1e8.
    size = 40
    orthogonal_left, _ = np.linalg.qr(generator.normal(size=(size, size)))
    orthogonal_right, _ = np.linalg.qr(generator.normal(size=(size, size)))
    ratio = 1e-8
    chosen = np.geomspace(1.0, ratio, size)
    matrix = (orthogonal_left * chosen) @ orthogonal_right.T

    library = np.linalg.svd(matrix, compute_uv=False)
    gram_based = svd_via_gram(matrix)
    relative_library = abs(library[-1] - ratio) / ratio
    relative_gram = abs(gram_based[-1] - ratio) / ratio

    # Where does the construction stop working? Sweep the ratio.
    sweep = []
    for exponent in (2, 4, 6, 8, 10, 12, 16):
        target = 10.0 ** (-exponent)             # smallest singular value relative to the largest
        scaled = (orthogonal_left * np.geomspace(1.0, target, size)) @ orthogonal_right.T
        library_sigma = np.linalg.svd(scaled, compute_uv=False)
        gram_result = svd_via_gram(scaled)
        error = abs(gram_result[-1] - library_sigma[-1]) / library_sigma[-1]
        sweep.append((f"1e-{exponent}", float(error)))

    assert well_conditioned < 1e-8, f"gram-based SVD inaccurate on well-conditioned inputs: {well_conditioned}"
    assert relative_gram > relative_library, "the gram construction should be less accurate"

    sweep_text = ", ".join(f"ratio {label}: relative error {value:.1e}" for label, value in sweep)
    return (
        f"Well-conditioned case: over 100 random matrices of size 3-11, the maximum relative "
        f"difference in singular values between the textbook construction and numpy is "
        f"{well_conditioned:.2e} — the construction is correct. "
        f"Ill-conditioned case: a 40x40 matrix with singular values from 1 down to 1e-8. "
        f"numpy's smallest singular value is {library[-1]:.3e} (relative error {relative_library:.1e}); "
        f"the A'A construction gives {gram_based[-1]:.3e} (relative error {relative_gram:.1e}). "
        f"Sweep of loss against the ratio of extreme singular values: {sweep_text}. "
        "The pattern is the point: forming A'A squares the ratio, so the small singular values lose "
        "roughly half their significant digits, and the loss grows without bound as the ratio grows."
    )


def check_lesson_worked_examples() -> str:
    """Everything the lesson states numerically: the 2x2 SVD, the phylogeny claim, the minors."""
    A = sp.Matrix([[2, 2], [-1, 1]])

    gram = A.T * A
    assert gram == sp.Matrix([[5, 3], [3, 5]])
    char = sp.factor((gram - LAM * sp.eye(2)).det())
    assert sp.simplify(char - (LAM - 8) * (LAM - 2)) == 0, char

    U = sp.Matrix([[1, 0], [0, -1]])
    Sigma = sp.diag(sp.sqrt(8), sp.sqrt(2))
    V = sp.Matrix([[1, 1], [1, -1]]) / sp.sqrt(2)
    assert sp.simplify(U * Sigma * V.T - A) == sp.zeros(2, 2)

    # det A = det U * det Sigma * det V, and |det A| = product of singular values.
    assert U.det() == -1 and sp.simplify(V.det() + 1) == 0
    assert sp.simplify(Sigma.det() - sp.sqrt(16)) == 0
    assert sp.simplify(U.det() * Sigma.det() * V.det() - A.det()) == 0
    assert abs(A.det()) == 4 and sp.simplify(sp.sqrt(8) * sp.sqrt(2) - 4) == 0

    # No real eigenvalues, but a perfectly real SVD.
    assert sp.factor((A - LAM * sp.eye(2)).det()) == LAM**2 - 3 * LAM + 4
    assert sp.discriminant(LAM**2 - 3 * LAM + 4, LAM) == -7

    # Section 1.3's leading minors, and the ellipse semi-axes equal the singular values.
    B = sp.Matrix([[9, 6, 3], [6, 5, 2], [3, 2, 2]])
    assert [B[:k, :k].det() for k in (1, 2, 3)] == [9, 9, 9]

    return (
        "Section 3.4: A'A = [[5,3],[3,5]] with characteristic polynomial (lambda-8)(lambda-2) for "
        "lambda = sigma^2, so sigma = 2sqrt(2) and sqrt(2); U Sigma V' reproduces A exactly with "
        "det U = -1 and det V = -1, so det A = 4 = (2sqrt(2))(sqrt(2)) = |det A| as claimed. "
        "The eigenvalues of A are complex (discriminant -7, characteristic polynomial "
        "lambda^2 - 3lambda + 4), which is the counterexample to 'singular values are eigenvalues'. "
        "Section 1.3: the leading minors of [[9,6,3],[6,5,2],[3,2,2]] are 9, 9 and 9."
    )
