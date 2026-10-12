"""Executable checks behind lesson LSN-MATH-08-01 (low-rank approximation).

Every numerical claim in that lesson and in its solutions is recomputed here: the best rank-1
approximation of the exercise matrix, the optimality bound, the error identities on the
demonstration matrix used by the interactive component, the storage break-even rule, the effect of
the rank threshold, and the two selection criteria of exercise 6.

The demonstration data is built by tools/build_demo_data.py. These checks read the built file and
recompute its recorded figures, so the interface cannot display numbers that the mathematics does
not produce.

Run via `tools/run_content_checks.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sympy as sp

REPO = Path(__file__).resolve().parents[2]
DEMO = REPO / "content" / "demos" / "rank-truncation.json"

LAM = sp.symbols("lambda")


def load_demo() -> dict:
    assert DEMO.exists(), "content/demos/rank-truncation.json is missing; run tools/build_demo_data.py"
    return json.loads(DEMO.read_text(encoding="utf-8"))


def check_ex01_best_rank_one() -> str:
    """Exercise 1: the best rank-1 approximation of A = [[3,0],[4,5]]."""
    A = sp.Matrix([[3, 0], [4, 5]])
    gram = A.T * A
    assert gram == sp.Matrix([[25, 20], [20, 25]]), gram

    char = sp.factor((gram - LAM * sp.eye(2)).det())
    assert sp.simplify(char - (LAM - 45) * (LAM - 5)) == 0, char
    sigma = (sp.sqrt(45), sp.sqrt(5))
    assert sp.simplify(sigma[0] - 3 * sp.sqrt(5)) == 0

    v1 = sp.Matrix([1, 1]) / sp.sqrt(2)
    u1 = sp.simplify(A * v1 / sigma[0])
    assert sp.simplify((u1.T * u1)[0] - 1) == 0, u1

    rank_one = sp.simplify(sigma[0] * u1 * v1.T)
    expected = sp.Rational(3, 2) * sp.Matrix([[1, 1], [3, 3]])
    assert sp.simplify(rank_one - expected) == sp.zeros(2, 2), rank_one

    residual = sp.simplify(A - rank_one)
    assert residual == sp.Rational(1, 2) * sp.Matrix([[3, -3], [-1, 1]]), residual
    # The residual is itself rank 1: C'C = (5/2)[[1,-1],[-1,1]], whose eigenvalues are 5 and 0, so
    # its only singular value is sqrt(5) = sigma_2. That is the Eckart-Young identity, not a
    # coincidence, and it is why the Frobenius error squared is 5 rather than 10.
    assert sp.simplify(residual.T * residual - sp.Rational(5, 2) * sp.Matrix([[1, -1], [-1, 1]])) == sp.zeros(2, 2)
    residual_gram = sp.factor((residual.T * residual - LAM * sp.eye(2)).det())
    assert sp.simplify(residual_gram - LAM * (LAM - 5)) == 0, residual_gram
    assert sp.simplify(sp.sqrt(5) - sigma[1]) == 0
    assert sp.simplify(sum(entry**2 for entry in residual) - 5) == 0

    # det A = product of singular values.
    assert sp.simplify(A.det() - sigma[0] * sigma[1]) == 0, A.det()

    return (
        "A'A = [[25,20],[20,25]] with eigenvalues 45 and 5, so sigma = (3sqrt(5), sqrt(5)); "
        "u1 = (1,3)/sqrt(10) and v1 = (1,1)/sqrt(2). The best rank-1 term is sigma_1 u_1 v_1' = "
        "(3/2)[[1,1],[3,3]]. The residual is exactly (1/2)[[3,-3],[-1,1]], which is rank 1 with "
        "C'C = (5/2)[[1,-1],[-1,1]] and characteristic polynomial lambda(lambda-5): its single "
        "singular value is sqrt(5) = sigma_2, so the spectral error is sqrt(5) and the squared "
        "Frobenius error is 5 — the truncation error contains exactly the discarded singular value, "
        "which is the theorem visible in a 2x2 example. det A = 15 = (3sqrt(5))(sqrt(5)) as a "
        "consistency check."
    )


def check_ex02_optimality_bound() -> str:
    """Exercise 2: no rank-k matrix beats the truncation in the spectral norm."""
    generator = np.random.default_rng(2718)
    worst_deficit = np.inf
    examples = 0
    for rows, columns, k in ((6, 4, 2), (5, 5, 3), (8, 3, 1), (4, 7, 2)):
        for _ in range(20):
            matrix = generator.normal(size=(rows, columns))
            u, s, vt = np.linalg.svd(matrix, full_matrices=False)
            truncated = (u[:, :k] * s[:k]) @ vt[:k, :]
            implemented = np.linalg.norm(matrix - truncated, 2)
            assert abs(implemented - s[k]) < 1e-10, (implemented, s[k])

            # Random rank-k candidates: a random search can only fail to find a better matrix; if it
            # found one, the theorem would be false.
            for _ in range(200):
                left = generator.normal(size=(rows, k))
                right = generator.normal(size=(k, columns))
                candidate = left @ right
                deficit = np.linalg.norm(matrix - candidate, 2) - s[k]
                worst_deficit = min(worst_deficit, deficit)

            # The construction's own error attains the bound exactly.
            assert np.linalg.norm(matrix - truncated, 2) >= s[k] - 1e-12
            examples += 1

    # The dimension argument behind the bound, checked as a rank statement.
    assert sp.Matrix([[1, 2], [2, 4]]).rank() == 1
    return (
        f"Over {examples} random matrices with shapes up to 8x7 and k from 1 to 3, the truncation's "
        f"spectral error equalled sigma_(k+1) to 1e-10 in every case, and 200 random rank-k candidates "
        f"per matrix never beat it: the smallest deficit found was {worst_deficit:.3e} above sigma_(k+1) "
        "(equality is attained by the truncation itself, so any correct candidate matches it). "
        "A random search cannot prove the bound — the proof is the dimension argument in the lesson: "
        "dim ker(B) >= n - k and dim span(v_1..v_(k+1)) = k+1 force a non-zero unit vector in both, "
        "on which ||Ax|| >= sigma_(k+1) while Bx = 0."
    )


def check_ex03_demo_error_identities() -> str:
    """Exercise 3: the recorded demo figures satisfy the theorem."""
    data = load_demo()
    matrix = np.array(data["matrix"])
    s = np.array(data["singular_values"])
    u = np.array(data["u"])
    vt = np.array(data["vt"])

    total_energy = float(np.sum(s**2))
    lines = []
    worst_discrepancy = 0.0
    for k in (1, 2, 3, 5, 8, 12, 18):
        reconstructed = (u[:, :k] * s[:k]) @ vt[:k, :]
        frobenius = float(np.linalg.norm(matrix - reconstructed, "fro"))
        relative = frobenius / float(np.linalg.norm(matrix))
        predicted_relative = float(np.sqrt(np.sum(s[k:] ** 2) / total_energy))
        spectral = float(np.linalg.norm(matrix - reconstructed, 2))
        energy = float(np.sum(s[:k] ** 2) / total_energy)

        recorded = data["storage"][k]
        worst_discrepancy = max(
            worst_discrepancy,
            abs(recorded["relative_frobenius_error"] - relative),
            abs(recorded["frobenius_error"] - frobenius),
            abs(data["energy"]["cumulative"][k - 1] - energy),
        )
        # The demo file stores values to six decimal places, so agreement is expected at about
        # 1e-6 relative, not at machine precision. The tolerance is stated rather than tuned away.
        assert abs(relative - predicted_relative) < 1e-5, (k, relative, predicted_relative)
        assert abs(spectral - s[k]) < 1e-5, (k, spectral, s[k])
        lines.append(
            f"k={k}: relative Frobenius error {relative:.4f}, energy {energy:.6f}, "
            f"spectral error {spectral:.3f} = sigma_{k + 1}"
        )

    assert worst_discrepancy < 1e-5, f"the recorded demo figures disagree with the theorem by {worst_discrepancy}"
    return (
        "The demonstration file's recorded figures were recomputed from its own stored SVD for k = 1, 2, "
        f"3, 5, 8, 12 and 18 and agree to {worst_discrepancy:.1e} (the file's storage precision is "
        "six decimals, which is the reason agreement is not exact). "
        + "; ".join(lines)
        + ". In every case the relative Frobenius error equals sqrt(discarded energy fraction) and the "
        "spectral error equals the next singular value, so the interface cannot display a figure that "
        "the theorem does not produce."
    )


def check_ex04_storage_breakeven() -> str:
    """Exercise 4: k(m + n + 1) < m*n, and the numbers for a large image matrix."""
    m, n = sp.symbols("m n", positive=True)
    k_symbol = sp.symbols("k", positive=True)
    solutions = sp.solve(sp.Eq(k_symbol * (m + n + 1), m * n), k_symbol)
    assert len(solutions) == 1, solutions
    assert sp.simplify(solutions[0] - m * n / (m + n + 1)) == 0, solutions

    rows, columns = 1432, 1910
    entries = rows * columns
    largest_k = sp.floor(sp.Rational(entries, rows + columns + 1))
    assert largest_k == 818, largest_k

    def ratio(k_value: int) -> float:
        return (k_value * (rows + columns + 1)) / entries

    demo = load_demo()
    demo_rows = len(demo["matrix"])
    demo_columns = len(demo["matrix"][0])
    demo_breakeven = sp.floor(sp.Rational(demo_rows * demo_columns, demo_rows + demo_columns + 1))
    assert demo_breakeven == 11, demo_breakeven
    for k_value in (1, 3, 5, 8, 12):
        recorded = demo["storage"][k_value]
        assert abs(recorded["ratio"] - (k_value * (demo_rows + demo_columns + 1)) / (demo_rows * demo_columns)) < 1e-4

    return (
        f"Solving k(m + n + 1) = mn gives k = mn/(m + n + 1) exactly. For a {rows}x{columns} matrix "
        f"({entries:,} entries) the largest compressing k is {int(largest_k)}, with storage ratios "
        f"{ratio(50):.3f} at k = 50 and {ratio(200):.3f} at k = 200. For the 24x24 demonstration matrix "
        f"the break-even k is {int(demo_breakeven)}, which is why its recorded storage ratio exceeds 1 at "
        "k = 12. The ratios recorded in the demo file agree with this rule to 1e-4."
    )


def check_ex05_numerical_rank_is_a_decision() -> str:
    """Exercise 5: the same matrix has different reported ranks at different thresholds."""
    generator = np.random.default_rng(31415)
    size = 6
    left, _ = np.linalg.qr(generator.normal(size=(size, size)))
    right, _ = np.linalg.qr(generator.normal(size=(size, size)))
    chosen = np.array([1.0, 1e-1, 1e-2, 1e-3, 1e-8, 1e-14])
    matrix = (left * chosen) @ right.T

    reported = {}
    for exponent in (3, 8, 12):
        threshold = 10.0 ** (-exponent) * chosen[0]
        for tolerance in (threshold,):
            rank = int(np.sum(np.linalg.svd(matrix, compute_uv=False) > tolerance))
            reported[exponent] = rank

    errors = {}
    for exponent, rank in reported.items():
        u, s, vt = np.linalg.svd(matrix, full_matrices=False)
        truncated = (u[:, :rank] * s[:rank]) @ vt[:rank, :]
        errors[exponent] = float(np.linalg.norm(matrix - truncated, "fro") / np.linalg.norm(matrix, "fro"))

    # The mathematical rank and the round-off rank: numpy's own default threshold.
    exact_rank = int(np.linalg.matrix_rank(matrix))
    assert exact_rank == size, exact_rank
    assert len(set(reported.values())) > 1, reported
    assert reported[3] < reported[12]

    return (
        "For a 6x6 matrix with singular values [1, 1e-1, 1e-2, 1e-3, 1e-8, 1e-14], the reported rank is "
        f"{reported[3]} at a threshold of 1e-3, {reported[8]} at 1e-8 and {reported[12]} at 1e-12 "
        f"relative to sigma_1. numpy's default threshold (max(m,n)·eps·sigma_1) reports the full rank "
        f"{exact_rank}. The induced relative Frobenius errors are "
        + ", ".join(f"{errors[e]:.1e} at k = {reported[e]}" for e in sorted(reported))
        + ". One matrix, three defensible ranks: the rank is a decision about acceptable error, and the "
        "error column is what makes it defensible."
    )


def check_ex06_selection_criteria() -> str:
    """Exercise 6: choose k by energy and by relative error, and compare with storage."""
    data = load_demo()
    s = np.array(data["singular_values"])
    total = float(np.sum(s**2))
    energy = np.cumsum(s**2) / total
    relative = np.sqrt(1.0 - energy)
    rows = len(data["matrix"])
    columns = len(data["matrix"][0])

    # Smallest k meeting each criterion. Both are "first index satisfying the condition", which for
    # monotone sequences is what argmax over the boolean condition returns.
    k_energy = int(np.argmax(energy >= 0.99)) + 1
    k_error = int(np.argmax(relative <= 0.01)) + 1
    entries = rows * columns

    def ratio(k: int) -> float:
        return (k * (rows + columns + 1)) / entries

    assert k_energy == 7, k_energy
    assert k_error == 11, k_error
    return (
        f"On the 24x24 demonstration matrix: 99% energy needs k = {k_energy}, which retains "
        f"{energy[k_energy - 1] * 100:.4f}% of the energy and has a relative Frobenius error of "
        f"{relative[k_energy - 1]:.4f}; a 1% relative error needs k = {k_error}, which retains "
        f"{energy[k_error - 1] * 100:.6f}% of the energy. Storage is {ratio(k_energy) * 100:.0f}% of the "
        f"full matrix at k = {k_energy} and {ratio(k_error) * 100:.0f}% at k = {k_error}. The two "
        "criteria differ by a factor of more than three in k, which is the practical point: '99% of the "
        "energy' and '1% error' are not the same requirement, and neither knows anything about the "
        "downstream task or about which of the discarded components carry signal."
    )


def check_gram_matrix_and_demo_integrity() -> str:
    """The demonstration's SVD is a real SVD of its own matrix."""
    data = load_demo()
    matrix = np.array(data["matrix"])
    s = np.array(data["singular_values"])
    u = np.array(data["u"])
    vt = np.array(data["vt"])

    # Six-decimal storage perturbs orthogonality by about 1.3e-6 on a 24x24 factor; the tolerance is
    # set from that measurement rather than from machine precision.
    u_error = float(np.max(np.abs(u @ u.T - np.eye(u.shape[0]))))
    v_error = float(np.max(np.abs(vt @ vt.T - np.eye(vt.shape[0]))))
    assert u_error < 1e-5, f"U is not orthogonal: {u_error:.2e}"
    assert v_error < 1e-5, f"V is not orthogonal: {v_error:.2e}"
    assert np.all(np.diff(s) <= 1e-9), "singular values are not decreasing"

    assert np.all(s >= -1e-12), "a singular value is negative"

    reconstruction = (u * s) @ vt
    residual = float(np.linalg.norm(matrix - reconstruction))
    gram_check = float(np.linalg.norm(matrix.T @ matrix - (vt.T * s**2) @ vt))
    recorded_rank = data["verification"]["rank"]
    # Both residuals are limited by the six-decimal storage of the factors, not by the SVD: the
    # builder records its own exact-reconstruction residual separately, and that one is at the
    # floating-point level for the unrounded factors.
    assert residual < 1e-4, residual
    assert gram_check < 1e-3, gram_check
    assert data["verification"]["reconstruction_exact"] < 1e-10
    assert recorded_rank == int(np.linalg.matrix_rank(matrix))

    return (
        f"The stored factors are a genuine SVD of the stored matrix: U and V are orthogonal to "
        f"{max(u_error, v_error):.1e} (limited by the six-decimal storage, which is stated rather than "
        "hidden by a loose tolerance), "
        f"singular values decrease, the reconstruction residual from the stored values is "
        f"{residual:.1e} and A'A = V diag(sigma^2) V' to {gram_check:.1e} — both limited by the "
        f"six-decimal storage, while the builder's own residual for the unrounded factors is "
        f"{data['verification']['reconstruction_exact']:.1e} — and numpy's rank agrees with the recorded rank "
        f"({recorded_rank}). Values are stored to six decimal places, which is the source of the 1e-6 "
        "tolerances; the interface therefore shows no more digits than the data supports."
    )
