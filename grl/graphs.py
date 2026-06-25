"""Graph construction for edge-ablation experiments.

Nodes are the input points. The intentionally variable part is how edges are
constructed, because the project should compare graph choices such as:

- Euclidean k-nearest-neighbor graphs;
- l-infinity k-nearest-neighbor graphs;
- rank-adjacency graphs;
- rank-space k-nearest-neighbor graphs.

For now this file only documents the intended responsibility. Implementation
should stay small and avoid committing to a heavy graph abstraction too early.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
import torch

from grl.discrepancy import as_point_array

KnnMetric = Literal["euclidean", "linf"]
GraphKind = Literal[
    "knn_euclidean",
    "knn_linf",
    "rank_knn_euclidean",
    "rank_knn_linf",
    "rank_adjacency",
]


@dataclass(frozen=True)
class TensorGraph:
    """Gather-ready graph tensors for batched torch point sets."""

    kind: GraphKind
    neighbors: torch.Tensor
    edge_mask: torch.Tensor
    edge_attr: torch.Tensor


def coordinate_ranks(points: ArrayLike) -> NDArray[np.float64]:
    """Return zero-based coordinate ranks normalized by ``max(n - 1, 1)``."""
    point_array = as_point_array(points)
    n = point_array.shape[0]
    ranks = np.argsort(np.argsort(point_array, axis=0), axis=0).astype(np.float64)
    return ranks / max(n - 1, 1)


def pairwise_distances(points: ArrayLike, metric: KnnMetric) -> NDArray[np.float64]:
    """Compute pairwise Euclidean or l-infinity distances."""
    point_array = as_point_array(points)
    differences = point_array[:, None, :] - point_array[None, :, :]
    if metric == "euclidean":
        return np.linalg.norm(differences, axis=-1)
    if metric == "linf":
        return np.max(np.abs(differences), axis=-1)
    raise ValueError(f"unknown kNN metric: {metric}")


def build_knn_edges(points: ArrayLike, k: int, metric: KnnMetric) -> NDArray[np.int64]:
    """Build directed kNN edges with shape ``(2, n * k)``.

    Edge direction is ``source -> target`` where each source point connects to
    its ``k`` nearest other points.
    """
    point_array = as_point_array(points)
    n = point_array.shape[0]
    if not 1 <= k < n:
        raise ValueError("k must satisfy 1 <= k < n")

    distances = pairwise_distances(point_array, metric)
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argsort(distances, axis=1, kind="stable")[:, :k]
    sources = np.repeat(np.arange(n), k)
    targets = neighbors.reshape(-1)
    return np.stack([sources, targets]).astype(np.int64)


def build_rank_knn_edges(points: ArrayLike, k: int, metric: KnnMetric) -> NDArray[np.int64]:
    """Build directed kNN edges after mapping points to normalized rank space."""
    return build_knn_edges(coordinate_ranks(points), k=k, metric=metric)


def build_rank_adjacency_edges(points: ArrayLike) -> NDArray[np.int64]:
    """Connect points that are adjacent in rank order along at least one coordinate."""
    point_array = as_point_array(points)
    n, d = point_array.shape
    edge_set: set[tuple[int, int]] = set()

    for coord in range(d):
        order = np.argsort(point_array[:, coord], kind="stable")
        for left, right in zip(order[:-1], order[1:], strict=True):
            edge_set.add((int(left), int(right)))
            edge_set.add((int(right), int(left)))

    if not edge_set:
        return np.zeros((2, 0), dtype=np.int64)
    return np.array(sorted(edge_set), dtype=np.int64).T


def edge_features(points: ArrayLike, edge_index: ArrayLike) -> NDArray[np.float64]:
    """Build coordinate-wise edge features from the spec.

    The returned array has shape ``(m, d, 6)`` with features:

    - source coordinate value;
    - target coordinate value;
    - target minus source;
    - absolute coordinate difference;
    - indicator that source coordinate is <= target coordinate;
    - target rank minus source rank.
    """
    point_array = as_point_array(points)
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, m)")

    sources = edges[0]
    targets = edges[1]
    source_points = point_array[sources]
    target_points = point_array[targets]
    differences = target_points - source_points
    ranks = coordinate_ranks(point_array)
    rank_differences = ranks[targets] - ranks[sources]

    return np.stack(
        [
            source_points,
            target_points,
            differences,
            np.abs(differences),
            (source_points <= target_points).astype(np.float64),
            rank_differences,
        ],
        axis=-1,
    )


def build_knn_graph(
    points: ArrayLike,
    *,
    k: int,
    metric: KnnMetric = "euclidean",
    include_edge_features: bool = True,
) -> dict[str, NDArray[np.float64] | NDArray[np.int64] | str]:
    """Build a directed kNN graph for a point set."""
    edge_index = build_knn_edges(points, k=k, metric=metric)
    graph: dict[str, NDArray[np.float64] | NDArray[np.int64] | str] = {
        "kind": f"knn_{metric}",
        "edge_index": edge_index,
    }
    if include_edge_features:
        graph["edge_attr"] = edge_features(points, edge_index)
    return graph


def build_rank_knn_graph(
    points: ArrayLike,
    *,
    k: int,
    metric: KnnMetric = "euclidean",
    include_edge_features: bool = True,
) -> dict[str, NDArray[np.float64] | NDArray[np.int64] | str]:
    """Build a directed kNN graph in normalized coordinate-rank space."""
    edge_index = build_rank_knn_edges(points, k=k, metric=metric)
    graph: dict[str, NDArray[np.float64] | NDArray[np.int64] | str] = {
        "kind": f"rank_knn_{metric}",
        "edge_index": edge_index,
    }
    if include_edge_features:
        graph["edge_attr"] = edge_features(points, edge_index)
    return graph


def build_rank_adjacency_graph(
    points: ArrayLike,
    *,
    include_edge_features: bool = True,
) -> dict[str, NDArray[np.float64] | NDArray[np.int64] | str]:
    """Build an undirected graph over adjacent points in each coordinate rank."""
    edge_index = build_rank_adjacency_edges(points)
    graph: dict[str, NDArray[np.float64] | NDArray[np.int64] | str] = {
        "kind": "rank_adjacency",
        "edge_index": edge_index,
    }
    if include_edge_features:
        graph["edge_attr"] = edge_features(points, edge_index)
    return graph


def build_euclidean_knn_graph(
    points: ArrayLike,
    *,
    k: int,
    include_edge_features: bool = True,
) -> dict[str, NDArray[np.float64] | NDArray[np.int64] | str]:
    """Build a directed Euclidean kNN graph."""
    return build_knn_graph(
        points,
        k=k,
        metric="euclidean",
        include_edge_features=include_edge_features,
    )


def build_linf_knn_graph(
    points: ArrayLike,
    *,
    k: int,
    include_edge_features: bool = True,
) -> dict[str, NDArray[np.float64] | NDArray[np.int64] | str]:
    """Build a directed l-infinity kNN graph."""
    return build_knn_graph(
        points,
        k=k,
        metric="linf",
        include_edge_features=include_edge_features,
    )


def build_rank_euclidean_knn_graph(
    points: ArrayLike,
    *,
    k: int,
    include_edge_features: bool = True,
) -> dict[str, NDArray[np.float64] | NDArray[np.int64] | str]:
    """Build a directed Euclidean kNN graph in normalized rank space."""
    return build_rank_knn_graph(
        points,
        k=k,
        metric="euclidean",
        include_edge_features=include_edge_features,
    )


def build_rank_linf_knn_graph(
    points: ArrayLike,
    *,
    k: int,
    include_edge_features: bool = True,
) -> dict[str, NDArray[np.float64] | NDArray[np.int64] | str]:
    """Build a directed l-infinity kNN graph in normalized rank space."""
    return build_rank_knn_graph(
        points,
        k=k,
        metric="linf",
        include_edge_features=include_edge_features,
    )


def build_graph(
    points: ArrayLike,
    *,
    kind: GraphKind,
    k: int = 8,
    include_edge_features: bool = True,
) -> dict[str, NDArray[np.float64] | NDArray[np.int64] | str]:
    """Build one of the graph families used in the ablation suite."""
    if kind == "knn_euclidean":
        return build_euclidean_knn_graph(points, k=k, include_edge_features=include_edge_features)
    if kind == "knn_linf":
        return build_linf_knn_graph(points, k=k, include_edge_features=include_edge_features)
    if kind == "rank_knn_euclidean":
        return build_rank_euclidean_knn_graph(points, k=k, include_edge_features=include_edge_features)
    if kind == "rank_knn_linf":
        return build_rank_linf_knn_graph(points, k=k, include_edge_features=include_edge_features)
    if kind == "rank_adjacency":
        return build_rank_adjacency_graph(points, include_edge_features=include_edge_features)
    raise ValueError(f"unknown graph kind: {kind}")


def tensor_coordinate_ranks(points: torch.Tensor) -> torch.Tensor:
    """Return zero-based coordinate ranks normalized by ``max(n - 1, 1)``."""
    n = points.shape[1]
    ranks = torch.argsort(torch.argsort(points, dim=1), dim=1).float()
    return ranks / max(n - 1, 1)


def _knn_metric_from_kind(kind: GraphKind) -> KnnMetric:
    if kind.endswith("euclidean"):
        return "euclidean"
    if kind.endswith("linf"):
        return "linf"
    raise ValueError(f"{kind} is not a kNN graph kind")


def _tensor_knn_neighbors(
    points: torch.Tensor,
    *,
    kind: GraphKind,
    k: int,
) -> torch.Tensor:
    """Return target neighbor indices with shape ``(batch, n, k)``."""
    n = points.shape[1]
    if not 1 <= k < n:
        raise ValueError("k must satisfy 1 <= k < number of points")

    basis = tensor_coordinate_ranks(points) if kind.startswith("rank_knn") else points
    differences = basis[:, :, None, :] - basis[:, None, :, :]
    metric = _knn_metric_from_kind(kind)
    if metric == "euclidean":
        distances = torch.linalg.vector_norm(differences, dim=-1)
    else:
        distances = differences.abs().amax(dim=-1)

    diagonal = torch.eye(n, dtype=torch.bool, device=points.device).unsqueeze(0)
    distances = distances.masked_fill(diagonal, torch.inf)
    return torch.topk(distances, k=k, dim=-1, largest=False, sorted=True).indices


def _tensor_rank_adjacency_neighbors(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return padded rank-adjacency neighbors and a boolean edge mask."""
    batch_size, n, d = points.shape
    max_degree = max(1, 2 * d)
    if n == 1:
        neighbors = torch.zeros(
            batch_size,
            n,
            max_degree,
            dtype=torch.long,
            device=points.device,
        )
        edge_mask = torch.zeros(batch_size, n, max_degree, dtype=torch.bool, device=points.device)
        return neighbors, edge_mask

    order = torch.argsort(points, dim=1)
    candidates = torch.full(
        (batch_size, n, max_degree),
        n,
        dtype=torch.long,
        device=points.device,
    )

    lower_rank_points = order[:, :-1, :]
    higher_rank_points = order[:, 1:, :]
    candidates[:, :, 0::2].scatter_(dim=1, index=lower_rank_points, src=higher_rank_points)
    candidates[:, :, 1::2].scatter_(dim=1, index=higher_rank_points, src=lower_rank_points)

    sorted_candidates = torch.sort(candidates, dim=2).values
    edge_mask = sorted_candidates != n
    is_first_occurrence = torch.ones_like(edge_mask)
    is_first_occurrence[:, :, 1:] = sorted_candidates[:, :, 1:] != sorted_candidates[:, :, :-1]
    keep = edge_mask & is_first_occurrence

    positions = torch.cumsum(keep.long(), dim=2) - 1
    overflow_positions = max_degree + torch.arange(max_degree, device=points.device)
    compact_keys = torch.where(keep, positions, overflow_positions.view(1, 1, -1))
    compact_order = torch.argsort(compact_keys, dim=2)
    neighbors = torch.gather(sorted_candidates, dim=2, index=compact_order)
    edge_mask = torch.gather(keep, dim=2, index=compact_order)
    neighbors = neighbors.masked_fill(~edge_mask, 0)
    return neighbors, edge_mask


def tensor_edge_features(points: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
    """Build edge features with shape ``(batch, n, neighbors, d, 6)``."""
    batch_size, n, d = points.shape
    neighbor_count = neighbors.shape[2]
    gather_index = neighbors.unsqueeze(-1).expand(batch_size, n, neighbor_count, d)
    target_points = torch.gather(
        points.unsqueeze(1).expand(batch_size, n, n, d),
        dim=2,
        index=gather_index,
    )
    source_points = points.unsqueeze(2).expand_as(target_points)
    differences = target_points - source_points

    ranks = tensor_coordinate_ranks(points)
    target_ranks = torch.gather(
        ranks.unsqueeze(1).expand(batch_size, n, n, d),
        dim=2,
        index=gather_index,
    )
    source_ranks = ranks.unsqueeze(2).expand_as(target_ranks)

    return torch.stack(
        [
            source_points,
            target_points,
            differences,
            differences.abs(),
            (source_points <= target_points).float(),
            target_ranks - source_ranks,
        ],
        dim=-1,
    )


def build_tensor_graph(points: torch.Tensor, *, kind: GraphKind, k: int = 8) -> TensorGraph:
    """Build graph tensors used by torch point scorers."""
    if points.ndim != 3:
        raise ValueError("points must have shape (batch, n, d)")
    if kind == "rank_adjacency":
        neighbors, edge_mask = _tensor_rank_adjacency_neighbors(points)
    elif kind in {"knn_euclidean", "knn_linf", "rank_knn_euclidean", "rank_knn_linf"}:
        neighbors = _tensor_knn_neighbors(points, kind=kind, k=k)
        edge_mask = torch.ones_like(neighbors, dtype=torch.bool)
    else:
        raise ValueError(f"unknown graph kind: {kind}")
    return TensorGraph(
        kind=kind,
        neighbors=neighbors,
        edge_mask=edge_mask,
        edge_attr=tensor_edge_features(points, neighbors),
    )
