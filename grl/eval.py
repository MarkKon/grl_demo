"""Evaluation metrics and checks.

This module should implement the project metrics from ``spec.md``:

- recall of the predicted subset against optimal support sets;
- regret for a predicted budget ``K``;
- coordinate-permutation symmetry checks;
- small reporting helpers for comparing graph construction ablations.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
import torch

from grl.baseline import local_discrepancy_scores, random_subset, select_top_k
from grl.data import load_labeled_samples
from grl.discrepancy import as_point_array, score_support
from grl.models import CoordinateKnnGraphPointScorer, FlatSetMlpPointScorer


def support_masks_to_indices(
    support_mask: ArrayLike,
    support_count: int,
) -> list[tuple[int, ...]]:
    """Convert padded boolean support masks to index tuples."""
    masks = np.asarray(support_mask, dtype=np.bool_)
    return [
        tuple(np.flatnonzero(masks[index]).astype(int).tolist())
        for index in range(support_count)
    ]


def recall_at_k(
    selected_indices: ArrayLike,
    support_indices: tuple[int, ...] | list[tuple[int, ...]],
) -> float:
    """Compute best support recall achieved by the selected subset."""
    selected = set(np.asarray(selected_indices, dtype=np.int64).tolist())
    supports = (
        [tuple(support_indices)]
        if support_indices and isinstance(support_indices[0], int)
        else list(support_indices)
    )
    if not supports:
        raise ValueError("support_indices must be non-empty")

    return max(len(selected.intersection(support)) / len(support) for support in supports)


def restricted_discrepancy(
    points: ArrayLike,
    selected_indices: ArrayLike,
) -> dict[str, Any]:
    """Compute the best discrepancy using supports inside selected indices."""
    point_array = as_point_array(points)
    selected = tuple(np.asarray(selected_indices, dtype=np.int64).tolist())
    if not selected:
        raise ValueError("selected_indices must be non-empty")

    max_support_size = min(point_array.shape[1], len(selected))
    best: dict[str, Any] | None = None
    for support_size in range(1, max_support_size + 1):
        for support in combinations(selected, support_size):
            scored = score_support(point_array, support)
            if best is None or scored["score"] > best["score"]:
                best = scored

    if best is None:
        raise ValueError("could not score any selected support")
    return best


def regret_at_k(
    points: ArrayLike,
    selected_indices: ArrayLike,
    optimal_score: float,
) -> float:
    """Return restricted discrepancy divided by the exact optimal score."""
    if optimal_score == 0:
        return float("nan")
    return restricted_discrepancy(points, selected_indices)["score"] / optimal_score


def evaluate_local_baseline_arrays(
    arrays: dict[str, NDArray[Any]],
    *,
    k: int,
) -> dict[str, float]:
    """Evaluate the local-discrepancy baseline on loaded dataset arrays."""
    recalls = []
    regrets = []

    for sample_index, points in enumerate(arrays["points"]):
        scores = local_discrepancy_scores(points)
        selected = select_top_k(scores, k)
        support_count = int(arrays["support_count"][sample_index])
        supports = support_masks_to_indices(
            arrays["support_mask"][sample_index],
            support_count,
        )
        optimal_score = float(np.nanmax(arrays["scores"][sample_index, :support_count]))

        recalls.append(recall_at_k(selected, supports))
        regrets.append(regret_at_k(points, selected, optimal_score))

    return {
        "num_samples": float(arrays["points"].shape[0]),
        "k": float(k),
        "mean_recall": float(np.mean(recalls)),
        "mean_regret": float(np.mean(regrets)),
    }


def evaluate_local_baseline_file(path: str, *, k: int) -> dict[str, float]:
    """Load a saved dataset and evaluate the local-discrepancy baseline."""
    return evaluate_local_baseline_arrays(load_labeled_samples(path), k=k)


def evaluate_random_baseline_arrays(
    arrays: dict[str, NDArray[Any]],
    *,
    k: int,
    seed: int = 0,
    repeats: int = 1,
) -> dict[str, float]:
    """Evaluate uniformly random subsets of size ``k``."""
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    recalls = []
    regrets = []
    generator = np.random.default_rng(seed)

    for _ in range(repeats):
        for sample_index, points in enumerate(arrays["points"]):
            selected = random_subset(points.shape[0], k, generator)
            support_count = int(arrays["support_count"][sample_index])
            supports = support_masks_to_indices(
                arrays["support_mask"][sample_index],
                support_count,
            )
            optimal_score = float(np.nanmax(arrays["scores"][sample_index, :support_count]))

            recalls.append(recall_at_k(selected, supports))
            regrets.append(regret_at_k(points, selected, optimal_score))

    return {
        "num_samples": float(arrays["points"].shape[0]),
        "k": float(k),
        "repeats": float(repeats),
        "mean_recall": float(np.mean(recalls)),
        "mean_regret": float(np.mean(regrets)),
    }


def evaluate_random_baseline_file(
    path: str,
    *,
    k: int,
    seed: int = 0,
    repeats: int = 1,
) -> dict[str, float]:
    """Load a saved dataset and evaluate the random subset baseline."""
    return evaluate_random_baseline_arrays(
        load_labeled_samples(path),
        k=k,
        seed=seed,
        repeats=repeats,
    )


def evaluate_score_arrays(
    arrays: dict[str, NDArray[Any]],
    score_matrix: NDArray[np.float64],
    *,
    k: int,
) -> dict[str, float]:
    """Evaluate externally produced point scores on saved dataset arrays."""
    recalls = []
    regrets = []

    for sample_index, points in enumerate(arrays["points"]):
        selected = select_top_k(score_matrix[sample_index], k)
        support_count = int(arrays["support_count"][sample_index])
        supports = support_masks_to_indices(
            arrays["support_mask"][sample_index],
            support_count,
        )
        optimal_score = float(np.nanmax(arrays["scores"][sample_index, :support_count]))

        recalls.append(recall_at_k(selected, supports))
        regrets.append(regret_at_k(points, selected, optimal_score))

    return {
        "num_samples": float(arrays["points"].shape[0]),
        "k": float(k),
        "mean_recall": float(np.mean(recalls)),
        "mean_regret": float(np.mean(regrets)),
    }


def checkpoint_model(checkpoint: dict[str, Any]) -> torch.nn.Module:
    """Construct the model described by a saved checkpoint."""
    model_name = checkpoint.get("model", "graph")
    if model_name == "graph":
        return CoordinateKnnGraphPointScorer(
            hidden_dim=int(checkpoint["hidden_dim"]),
            num_layers=int(checkpoint["graph_layers"]),
            k=int(checkpoint["graph_k"]),
            metric=str(checkpoint["graph_metric"]),
            graph_kind=str(checkpoint.get("graph_kind", f"knn_{checkpoint['graph_metric']}")),
            coordinate_shared=bool(checkpoint.get("coordinate_shared", True)),
            input_dim=int(checkpoint.get("input_d", 0)) or None,
        )
    if model_name == "flat_mlp":
        return FlatSetMlpPointScorer(
            n=int(checkpoint["input_n"]),
            d=int(checkpoint["input_d"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
        )
    raise ValueError(f"unknown checkpoint model: {model_name}")


def checkpoint_scores(
    arrays: dict[str, NDArray[Any]],
    checkpoint_path: str,
    *,
    batch_size: int = 256,
    device: str | None = None,
) -> NDArray[np.float64]:
    """Run a saved point-scorer checkpoint and return point probabilities."""
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
    model = checkpoint_model(checkpoint).to(resolved_device)
    model.load_state_dict(checkpoint.get("best_state_dict", checkpoint["state_dict"]))
    model.eval()

    points = torch.as_tensor(arrays["points"]).float()
    scores = []
    with torch.no_grad():
        for start in range(0, points.shape[0], batch_size):
            batch = points[start : start + batch_size].to(resolved_device)
            scores.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(scores, axis=0)


def evaluate_checkpoint_file(
    dataset_path: str,
    checkpoint_path: str,
    *,
    k: int,
    batch_size: int = 256,
    device: str | None = None,
) -> dict[str, float]:
    """Evaluate any saved point-scorer checkpoint on a saved dataset."""
    arrays = load_labeled_samples(dataset_path)
    scores = checkpoint_scores(
        arrays,
        checkpoint_path,
        batch_size=batch_size,
        device=device,
    )
    return evaluate_score_arrays(arrays, scores, k=k)
