"""Non-learned baselines for point scoring."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from grl.discrepancy import as_point_array, delta_minus


def local_discrepancy_scores(points: ArrayLike) -> NDArray[np.float64]:
    """Score each point by using the point itself as the discrepancy corner."""
    point_array = as_point_array(points)
    return np.array(
        [delta_minus(point_array, point) for point in point_array],
        dtype=np.float64,
    )


def select_top_k(scores: ArrayLike, k: int) -> NDArray[np.int64]:
    """Return indices of the top ``k`` scores, sorted best first."""
    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if not 1 <= k <= score_array.shape[0]:
        raise ValueError("k must satisfy 1 <= k <= len(scores)")

    return np.argsort(-score_array, kind="stable")[:k].astype(np.int64)


def random_subset(
    n: int,
    k: int,
    rng: np.random.Generator | int | None = None,
) -> NDArray[np.int64]:
    """Return ``k`` uniformly sampled point indices without replacement."""
    if not 1 <= k <= n:
        raise ValueError("k must satisfy 1 <= k <= n")

    generator = np.random.default_rng(rng)
    return np.sort(generator.choice(n, size=k, replace=False)).astype(np.int64)
