"""Exact star discrepancy and support-set utilities.

This module should own the mathematical target used by the rest of the project:

- computing candidate corners from support sets;
- evaluating the minus star discrepancy objective;
- enumerating support sets for small ``n`` and ``d``;
- extracting optimal or top-scoring support sets;
- producing the alpha/beta supervision labels described in ``spec.md``.

Keep this module independent of model code. It should be easy to test on small
hand-checkable point sets.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def as_point_array(points: ArrayLike) -> NDArray[np.float64]:
    """Return points as a validated ``(n, d)`` floating point array."""
    point_array = np.asarray(points, dtype=np.float64)
    if point_array.ndim != 2:
        raise ValueError("points must have shape (n, d)")
    if point_array.shape[0] == 0:
        raise ValueError("points must contain at least one point")
    if point_array.shape[1] == 0:
        raise ValueError("points must contain at least one coordinate")
    return point_array


def support_corner(
    points: ArrayLike,
    support_indices: tuple[int, ...] | list[int],
) -> NDArray[np.float64]:
    """Compute ``y(S) = max_{p in S} p`` coordinate-wise."""
    point_array = as_point_array(points)
    if len(support_indices) == 0:
        raise ValueError("support_indices must be non-empty")
    return np.max(point_array[list(support_indices)], axis=0)


def delta_minus(points: ArrayLike, corner: ArrayLike) -> float:
    """Evaluate ``|{p_i <= y}| / n - prod_j y_j`` for a candidate corner."""
    point_array = as_point_array(points)
    corner_array = np.asarray(corner, dtype=np.float64)
    if corner_array.shape != (point_array.shape[1],):
        raise ValueError("corner must have shape (d,)")

    dominated_count = np.all(point_array <= corner_array, axis=1).sum()
    volume = np.prod(corner_array)
    return float(dominated_count / point_array.shape[0] - volume)


def enumerate_supports(n: int, d: int):
    """Yield support index tuples with ``1 <= |S| <= min(n, d)``."""
    if n <= 0:
        raise ValueError("n must be positive")
    if d <= 0:
        raise ValueError("d must be positive")

    for support_size in range(1, min(n, d) + 1):
        yield from combinations(range(n), support_size)


def score_support(
    points: ArrayLike,
    support_indices: tuple[int, ...] | list[int],
) -> dict[str, Any]:
    """Compute corner and discrepancy score for one support set."""
    support = tuple(support_indices)
    corner = support_corner(points, support)
    return {
        "support": support,
        "corner": corner,
        "score": delta_minus(points, corner),
    }


def find_top_supports(
    points: ArrayLike,
    top_b: int = 1,
    *,
    include_ties: bool = True,
    atol: float = 1e-12,
) -> list[dict[str, Any]]:
    """Return the best-scoring support sets for the minus discrepancy problem.

    If ``include_ties`` is true, every support whose score ties the last retained
    score up to ``atol`` is returned. With ``top_b=1`` this gives all optimal
    support sets.
    """
    point_array = as_point_array(points)
    if top_b <= 0:
        raise ValueError("top_b must be positive")

    scored = [
        score_support(point_array, support)
        for support in enumerate_supports(*point_array.shape)
    ]
    scored.sort(key=lambda item: item["score"], reverse=True)

    if len(scored) <= top_b or not include_ties:
        return scored[:top_b]

    cutoff = scored[top_b - 1]["score"]
    return [item for item in scored if item["score"] >= cutoff - atol]
