"""Tests for graph construction behavior and edge-ablation invariants."""

import numpy as np
import pytest
import torch

from grl.graphs import (
    build_graph,
    build_euclidean_knn_graph,
    build_knn_edges,
    build_linf_knn_graph,
    build_rank_adjacency_edges,
    build_rank_adjacency_graph,
    build_rank_knn_edges,
    build_rank_linf_knn_graph,
    build_tensor_graph,
    coordinate_ranks,
    edge_features,
    pairwise_distances,
    tensor_coordinate_ranks,
)


def test_coordinate_ranks_are_normalized_per_dimension():
    points = np.array(
        [
            [0.2, 0.7],
            [0.5, 0.1],
            [0.9, 0.4],
        ]
    )

    ranks = coordinate_ranks(points)

    np.testing.assert_allclose(
        ranks,
        np.array(
            [
                [0.0, 1.0],
                [0.5, 0.0],
                [1.0, 0.5],
            ]
        ),
    )


def test_pairwise_distances_support_euclidean_and_linf():
    points = np.array(
        [
            [0.0, 0.0],
            [3.0, 4.0],
        ]
    )

    euclidean = pairwise_distances(points, "euclidean")
    linf = pairwise_distances(points, "linf")

    assert euclidean[0, 1] == 5.0
    assert linf[0, 1] == 4.0


def test_knn_edges_have_n_times_k_directed_edges_without_self_loops():
    points = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.9, 0.0],
            [1.0, 0.0],
        ]
    )

    edge_index = build_knn_edges(points, k=2, metric="euclidean")

    assert edge_index.shape == (2, 8)
    assert not np.any(edge_index[0] == edge_index[1])
    for source in range(points.shape[0]):
        assert np.sum(edge_index[0] == source) == 2


def test_knn_edges_reject_invalid_k():
    points = np.array([[0.0], [1.0]])

    with pytest.raises(ValueError):
        build_knn_edges(points, k=0, metric="euclidean")
    with pytest.raises(ValueError):
        build_knn_edges(points, k=2, metric="euclidean")


def test_rank_knn_edges_use_rank_space_not_coordinate_space():
    points = np.array(
        [
            [0.63696169, 0.26978671, 0.04097352],
            [0.01652764, 0.81327024, 0.91275558],
            [0.60663578, 0.72949656, 0.54362499],
            [0.93507242, 0.81585355, 0.00273850],
            [0.85740428, 0.03358558, 0.72965545],
        ]
    )

    coordinate_edges = build_knn_edges(points, k=1, metric="euclidean")
    rank_edges = build_rank_knn_edges(points, k=1, metric="euclidean")

    assert coordinate_edges[:, 0].tolist() == [0, 3]
    assert rank_edges[:, 0].tolist() == [0, 2]


def test_rank_adjacency_edges_connect_adjacent_rank_neighbors():
    points = np.array(
        [
            [0.1, 0.8],
            [0.4, 0.2],
            [0.9, 0.5],
        ]
    )

    edge_index = build_rank_adjacency_edges(points)
    edge_set = set(map(tuple, edge_index.T.tolist()))

    assert edge_set == {
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
        (1, 2),
        (2, 1),
        (2, 0),
        (0, 2),
    }


def test_euclidean_and_linf_can_choose_different_neighbors():
    points = np.array(
        [
            [0.0, 0.0],
            [0.7, 0.0],
            [0.5, 0.5],
        ]
    )

    euclidean = build_knn_edges(points, k=1, metric="euclidean")
    linf = build_knn_edges(points, k=1, metric="linf")

    assert euclidean[:, 0].tolist() == [0, 1]
    assert linf[:, 0].tolist() == [0, 2]


def test_edge_features_match_spec_coordinates_and_ranks():
    points = np.array(
        [
            [0.2, 0.7],
            [0.5, 0.1],
            [0.9, 0.4],
        ]
    )
    edge_index = np.array([[0], [1]])

    features = edge_features(points, edge_index)

    assert features.shape == (1, 2, 6)
    np.testing.assert_allclose(features[0, :, 0], points[0])
    np.testing.assert_allclose(features[0, :, 1], points[1])
    np.testing.assert_allclose(features[0, :, 2], points[1] - points[0])
    np.testing.assert_allclose(features[0, :, 3], np.abs(points[1] - points[0]))
    np.testing.assert_allclose(features[0, :, 4], np.array([1.0, 0.0]))
    np.testing.assert_allclose(features[0, :, 5], np.array([0.5, -1.0]))


def test_knn_graph_contains_kind_edges_and_optional_features():
    points = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.9, 0.0],
        ]
    )

    euclidean = build_euclidean_knn_graph(points, k=1)
    linf = build_linf_knn_graph(points, k=1, include_edge_features=False)

    assert euclidean["kind"] == "knn_euclidean"
    assert euclidean["edge_index"].shape == (2, 3)
    assert euclidean["edge_attr"].shape == (3, 2, 6)
    assert linf["kind"] == "knn_linf"
    assert "edge_attr" not in linf


def test_rank_graph_builders_contain_kind_edges_and_features():
    points = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.8],
            [0.9, 0.1],
        ]
    )

    rank_knn = build_rank_linf_knn_graph(points, k=1)
    rank_adjacency = build_rank_adjacency_graph(points)
    generic = build_graph(points, kind="rank_adjacency")

    assert rank_knn["kind"] == "rank_knn_linf"
    assert rank_knn["edge_index"].shape == (2, 3)
    assert rank_knn["edge_attr"].shape == (3, 2, 6)
    assert rank_adjacency["kind"] == "rank_adjacency"
    assert rank_adjacency["edge_index"].shape[0] == 2
    assert generic["kind"] == "rank_adjacency"


def test_tensor_coordinate_ranks_match_numpy_ranks():
    points = np.array(
        [
            [0.2, 0.7],
            [0.5, 0.1],
            [0.9, 0.4],
        ],
        dtype=np.float32,
    )

    ranks = tensor_coordinate_ranks(torch.as_tensor(points)[None, :, :])

    torch.testing.assert_close(
        ranks,
        torch.as_tensor(coordinate_ranks(points), dtype=torch.float32)[None, :, :],
    )


@pytest.mark.parametrize(
    "kind",
    [
        "knn_euclidean",
        "knn_linf",
        "rank_knn_euclidean",
        "rank_knn_linf",
        "rank_adjacency",
    ],
)
def test_tensor_graph_contains_gather_ready_neighbors_masks_and_edge_features(kind):
    points = torch.as_tensor(
        np.random.default_rng(0).random((2, 5, 3), dtype=np.float32)
    )

    graph = build_tensor_graph(points, kind=kind, k=2)

    assert graph.kind == kind
    assert graph.neighbors.shape[:2] == (2, 5)
    assert graph.edge_mask.shape == graph.neighbors.shape
    assert graph.edge_attr.shape == (*graph.neighbors.shape, 3, 6)
    assert graph.edge_mask.dtype == torch.bool


def test_tensor_knn_graph_matches_numpy_edges_for_one_sample():
    points = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.9, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    graph = build_tensor_graph(torch.as_tensor(points)[None, :, :], kind="knn_euclidean", k=2)
    edge_index = build_knn_edges(points, k=2, metric="euclidean")

    np.testing.assert_array_equal(
        graph.neighbors[0].numpy(),
        edge_index[1].reshape(points.shape[0], 2),
    )


def test_tensor_rank_adjacency_graph_matches_numpy_edges_for_one_sample():
    points = np.array(
        [
            [0.1, 0.8, 0.3],
            [0.4, 0.2, 0.7],
            [0.9, 0.5, 0.1],
            [0.3, 0.4, 0.9],
        ],
        dtype=np.float32,
    )

    graph = build_tensor_graph(torch.as_tensor(points)[None, :, :], kind="rank_adjacency")
    edge_index = build_rank_adjacency_edges(points)
    expected = {source: [] for source in range(points.shape[0])}
    for source, target in edge_index.T.tolist():
        expected[source].append(target)

    for source in range(points.shape[0]):
        count = int(graph.edge_mask[0, source].sum().item())
        assert graph.neighbors[0, source, :count].tolist() == sorted(expected[source])
