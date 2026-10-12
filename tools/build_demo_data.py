#!/usr/bin/env python3
"""Build the data for the rank-truncation interactive demonstration.

The demonstration shows what the Eckart-Young theorem means in practice: what a matrix looks like
when you keep only its k largest singular values, and what that costs and saves.

The matrix is **constructed from an explicit formula** rather than taken from a dataset. That is a
deliberate choice: a synthetic matrix whose singular values we can also analyse on paper teaches
the theorem without pretending that a toy is a real workload. Its three ingredients are chosen to
decay at different rates, so the reconstruction shows *what* survives truncation rather than only
that something does:

  1. a smooth Gaussian bump        -> one large singular value
  2. a diagonal ridge              -> a few moderate singular values
  3. deterministic fine structure  -> a long tail of small singular values

Output: content/demos/rank-truncation.json (U, s, V^T for the truncated reconstruction, plus the
matrix itself and the exact storage arithmetic).

Run: `.venv/bin/python tools/build_demo_data.py`
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "content" / "demos" / "rank-truncation.json"
SIZE = 24


def build_matrix(size: int = SIZE) -> tuple[np.ndarray, dict[str, str]]:
    grid = np.linspace(-1.0, 1.0, size)
    x, y = np.meshgrid(grid, grid)

    gaussian = np.exp(-((x - 0.35) ** 2 + (y + 0.3) ** 2) / 0.08)
    ridge = np.exp(-((x * 0.7 + y * 0.7) ** 2) / 0.02) * (1.0 - 0.3 * np.abs(x - y))
    texture = 0.25 * np.sin(9.0 * x) * np.sin(7.0 * y) + 0.12 * np.cos(15.0 * (x + y))

    matrix = gaussian + 0.6 * ridge + texture
    # Normalise to [0, 1] so the rendering is a greyscale image without a rescaling decision hidden
    # inside the component.
    matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min())

    ingredients = {
        "gaussian_bump": "exp(-((x-0.35)^2 + (y+0.3)^2)/0.08) — smooth, so it is close to rank 1",
        "diagonal_ridge": "exp(-((x+y)^2 * 0.49)/0.02) * (1 - 0.3|x-y|) — a line, so it needs a few components",
        "fine_texture": "0.25 sin(9x) sin(7y) + 0.12 cos(15(x+y)) — high frequency, so it fills the tail",
    }
    return matrix, ingredients


def main() -> int:
    matrix, ingredients = build_matrix()
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)

    rows, columns = matrix.shape
    full_entries = rows * columns
    # Storage for the rank-k factorisation: k columns of U, k singular values, k rows of V^T.
    storage = [
        {
            "k": int(k),
            "entries": int(k * (rows + columns) + k),
            "full_entries": int(full_entries),
            "ratio": round((k * (rows + columns) + k) / full_entries, 4),
            "frobenius_error": round(float(np.linalg.norm(matrix - (u[:, :k] * s[:k]) @ vt[:k, :])), 6),
            "relative_frobenius_error": round(
                float(np.linalg.norm(matrix - (u[:, :k] * s[:k]) @ vt[:k, :]) / np.linalg.norm(matrix)), 6
            ),
            "spectral_error_equals_next_singular_value": round(float(s[k]) if k < len(s) else 0.0, 6),
        }
        for k in range(0, len(s) + 1)
    ]

    payload = {
        "generated_by": "tools/build_demo_data.py",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "what_this_is": (
            "A synthetic 24x24 matrix built from three explicit formulas, its full SVD, and the "
            "storage/error arithmetic for every truncation rank. It is a teaching object, not a "
            "dataset, and nothing here is measured from real data."
        ),
        "matrix": [[round(float(value), 6) for value in row] for row in matrix.tolist()],
        "image_ingredients": ingredients,
        "u": [[round(float(value), 6) for value in row] for row in u.tolist()],
        "singular_values": [round(float(value), 6) for value in s.tolist()],
        "vt": [[round(float(value), 6) for value in row] for row in vt.tolist()],
        "storage": storage,
        "energy": {
            "note": "Fraction of the squared Frobenius norm captured by the first k singular values.",
            "cumulative": [
                round(float(np.sum(s[: k + 1] ** 2) / np.sum(s**2)), 6) for k in range(len(s))
            ],
        },
        "verification": {
            "reconstruction_exact": round(float(np.linalg.norm(matrix - u @ np.diag(s) @ vt)), 12),
            "frobenius_norm": round(float(np.linalg.norm(matrix)), 6),
            "rank": int(np.linalg.matrix_rank(matrix)),
            "note": (
                "The exact-reconstruction residual is the floating-point round-off of the library "
                "SVD on this matrix; it is recorded so the demonstration cannot claim more accuracy "
                "than it has."
            ),
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")

    print(f"wrote {OUT.relative_to(REPO)} ({OUT.stat().st_size / 1024:.1f} kB)")
    print(f"singular values: {[round(float(v), 3) for v in s[:8]]} ... {round(float(s[-1]), 4)}")
    print(f"rank: {payload['verification']['rank']}  exact-reconstruction residual: {payload['verification']['reconstruction_exact']:.2e}")
    print("storage at k=1,3,5,10: " + ", ".join(f"{row['ratio']:.2f}x" for row in storage if row["k"] in (1, 3, 5, 10)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
