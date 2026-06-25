"""Point-set generation and labeled dataset construction.

This module should collect the data side of the experiments:

- generating uniform point sets of configurable size and dimension;
- calling the exact discrepancy/support-set code to produce labels;
- saving and loading generated samples in a simple research-friendly format;
- later adding non-uniform point-set families if useful.

The initial target is correctness and reproducibility, not a general dataset
framework.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from grl.discrepancy import find_top_supports, support_corner

PointFamily = Literal["uniform", "jittered_grid"]


def make_uniform_points(
    n: int,
    d: int,
    rng: np.random.Generator | int | None = None,
) -> NDArray[np.float64]:
    """Generate a uniform point set in ``[0, 1)^d`` with shape ``(n, d)``."""
    if n <= 0:
        raise ValueError("n must be positive")
    if d <= 0:
        raise ValueError("d must be positive")

    generator = np.random.default_rng(rng)
    return generator.random((n, d), dtype=np.float64)


def make_jittered_grid_points(
    grid_size: int,
    d: int,
    rng: np.random.Generator | int | None = None,
) -> NDArray[np.float64]:
    """Generate one jittered point per cell of a regular ``grid_size^d`` grid."""
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    if d <= 0:
        raise ValueError("d must be positive")

    generator = np.random.default_rng(rng)
    cell_indices = np.indices((grid_size,) * d).reshape(d, -1).T
    jitter = generator.random(cell_indices.shape, dtype=np.float64)
    return (cell_indices + jitter) / grid_size


def make_points(
    n: int,
    d: int,
    rng: np.random.Generator | int | None = None,
    *,
    family: PointFamily = "uniform",
    grid_size: int | None = None,
) -> NDArray[np.float64]:
    """Generate a point set from one of the project benchmark families."""
    if family == "uniform":
        return make_uniform_points(n, d, rng)
    if family == "jittered_grid":
        if grid_size is None:
            raise ValueError("grid_size is required for jittered_grid")
        expected_n = grid_size**d
        if n != expected_n:
            raise ValueError(
                f"jittered_grid requires n == grid_size**d; got n={n}, "
                f"grid_size**d={expected_n}"
            )
        return make_jittered_grid_points(grid_size, d, rng)
    raise ValueError(f"unknown point family: {family}")


def labels_from_support(
    points: ArrayLike,
    support_indices: tuple[int, ...] | list[int],
    *,
    atol: float = 1e-12,
) -> dict[str, NDArray[np.float64]]:
    """Create alpha/beta labels for one support set.

    ``alpha[i, j]`` is one when ``p_i <= y(S)`` and ``p_ij = y(S)_j``.
    ``beta[i]`` is one when point ``i`` contributes at least one coordinate.
    """
    point_array = np.asarray(points, dtype=np.float64)
    corner = support_corner(point_array, support_indices)

    dominated = np.all(point_array <= corner, axis=1)
    touches_corner_coordinate = np.isclose(point_array, corner, atol=atol)
    alpha = (dominated[:, None] & touches_corner_coordinate).astype(np.float64)
    beta = alpha.max(axis=1)
    return {
        "alpha": alpha,
        "beta": beta,
    }


def make_labeled_sample(
    n: int,
    d: int,
    rng: np.random.Generator | int | None = None,
    *,
    family: PointFamily = "uniform",
    grid_size: int | None = None,
    top_b: int = 1,
    include_ties: bool = True,
) -> dict[str, Any]:
    """Generate one point set with exact top support labels."""
    points = make_points(n, d, rng, family=family, grid_size=grid_size)
    supports = find_top_supports(points, top_b=top_b, include_ties=include_ties)
    labels = [
        {
            **support,
            **labels_from_support(points, support["support"]),
        }
        for support in supports
    ]
    return {
        "points": points,
        "supports": labels,
    }


def make_labeled_samples(
    num_samples: int,
    n: int,
    d: int,
    rng: np.random.Generator | int | None = None,
    *,
    family: PointFamily = "uniform",
    grid_size: int | None = None,
    top_b: int = 1,
    include_ties: bool = True,
    show_progress: bool = False,
) -> list[dict[str, Any]]:
    """Generate multiple labeled point-set samples."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    generator = np.random.default_rng(rng)
    return [
        make_labeled_sample(
            n,
            d,
            generator,
            family=family,
            grid_size=grid_size,
            top_b=top_b,
            include_ties=include_ties,
        )
        for _ in tqdm(
            range(num_samples),
            desc=f"generate {family} n={n} d={d}",
            disable=not show_progress,
        )
    ]


def samples_to_arrays(samples: list[dict[str, Any]]) -> dict[str, NDArray[Any]]:
    """Pack labeled samples into padded arrays for ``.npz`` storage."""
    if not samples:
        raise ValueError("samples must be non-empty")

    points = np.stack([sample["points"] for sample in samples])
    num_samples, n, d = points.shape
    max_supports = max(len(sample["supports"]) for sample in samples)

    support_mask = np.zeros((num_samples, max_supports, n), dtype=np.bool_)
    alpha = np.zeros((num_samples, max_supports, n, d), dtype=np.float32)
    beta = np.zeros((num_samples, max_supports, n), dtype=np.float32)
    scores = np.full((num_samples, max_supports), np.nan, dtype=np.float64)
    corners = np.full((num_samples, max_supports, d), np.nan, dtype=np.float64)
    support_count = np.zeros(num_samples, dtype=np.int64)

    for sample_index, sample in enumerate(samples):
        support_count[sample_index] = len(sample["supports"])
        for support_index, support in enumerate(sample["supports"]):
            support_mask[sample_index, support_index, list(support["support"])] = True
            alpha[sample_index, support_index] = support["alpha"]
            beta[sample_index, support_index] = support["beta"]
            scores[sample_index, support_index] = support["score"]
            corners[sample_index, support_index] = support["corner"]

    return {
        "points": points.astype(np.float32),
        "support_mask": support_mask,
        "alpha": alpha,
        "beta": beta,
        "scores": scores,
        "corners": corners,
        "support_count": support_count,
    }


def save_labeled_samples(
    path: str | Path,
    samples: list[dict[str, Any]],
    *,
    seed: int | None = None,
    family: PointFamily = "uniform",
    grid_size: int | None = None,
    top_b: int = 1,
    include_ties: bool = True,
) -> None:
    """Save labeled samples as a compressed ``.npz`` file."""
    arrays = samples_to_arrays(samples)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **arrays,
        seed=-1 if seed is None else seed,
        family=family,
        grid_size=-1 if grid_size is None else grid_size,
        top_b=top_b,
        include_ties=include_ties,
    )


def load_labeled_samples(path: str | Path) -> dict[str, NDArray[Any]]:
    """Load arrays produced by ``save_labeled_samples``."""
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


class LabeledPointSetDataset(Dataset):
    """Torch dataset wrapper around saved labeled point-set arrays."""

    def __init__(self, path: str | Path):
        self.arrays = load_labeled_samples(path)

    def __len__(self) -> int:
        return int(self.arrays["points"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "points": torch.as_tensor(self.arrays["points"][index]),
            "support_mask": torch.as_tensor(self.arrays["support_mask"][index]),
            "alpha": torch.as_tensor(self.arrays["alpha"][index]),
            "beta": torch.as_tensor(self.arrays["beta"][index]),
            "scores": torch.as_tensor(self.arrays["scores"][index]),
            "corners": torch.as_tensor(self.arrays["corners"][index]),
            "support_count": torch.as_tensor(self.arrays["support_count"][index]),
        }
