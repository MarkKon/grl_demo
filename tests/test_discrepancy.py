"""Tests for exact discrepancy and support-label behavior."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from grl.baseline import local_discrepancy_scores, random_subset, select_top_k
from grl.data import (
    LabeledPointSetDataset,
    labels_from_support,
    load_labeled_samples,
    make_jittered_grid_points,
    make_labeled_sample,
    make_labeled_samples,
    make_uniform_points,
    samples_to_arrays,
    save_labeled_samples,
)
from grl.models import CoordinateKnnGraphPointScorer, FlatSetMlpPointScorer
from grl.train import alpha_union_target, beta_union_target, train_point_scorer
from grl.discrepancy import (
    delta_minus,
    enumerate_supports,
    find_top_supports,
    score_support,
    support_corner,
)
from grl.eval import (
    evaluate_checkpoint_file,
    evaluate_local_baseline_arrays,
    evaluate_random_baseline_arrays,
    recall_at_k,
    regret_at_k,
    restricted_discrepancy,
)


def test_support_corner_uses_coordinatewise_maximum():
    points = np.array(
        [
            [0.2, 0.8],
            [0.7, 0.3],
            [0.4, 0.5],
        ]
    )

    corner = support_corner(points, (0, 1))

    np.testing.assert_allclose(corner, np.array([0.7, 0.8]))


def test_delta_minus_counts_dominated_points_minus_volume():
    points = np.array(
        [
            [0.2, 0.2],
            [0.4, 0.5],
            [0.9, 0.1],
        ]
    )

    score = delta_minus(points, np.array([0.5, 0.5]))

    assert score == 2 / 3 - 0.25


def test_enumerate_supports_uses_sizes_up_to_dimension():
    supports = list(enumerate_supports(n=4, d=2))

    assert supports == [
        (0,),
        (1,),
        (2,),
        (3,),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    ]


def test_find_top_supports_matches_manual_scores():
    points = np.array(
        [
            [0.2, 0.2],
            [0.4, 0.5],
            [0.9, 0.1],
        ]
    )

    top = find_top_supports(points, top_b=1)
    manual = max(
        (score_support(points, support) for support in enumerate_supports(3, 2)),
        key=lambda item: item["score"],
    )

    assert top[0]["support"] == manual["support"]
    assert top[0]["score"] == manual["score"]


def test_labels_from_support_marks_corner_coordinate_contributors():
    points = np.array(
        [
            [0.2, 0.2],
            [0.4, 0.5],
            [0.4, 0.1],
            [0.7, 0.5],
        ]
    )

    labels = labels_from_support(points, (1,))

    np.testing.assert_array_equal(
        labels["alpha"],
        np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [1.0, 0.0],
                [0.0, 0.0],
            ]
        ),
    )
    np.testing.assert_array_equal(labels["beta"], np.array([0.0, 1.0, 1.0, 0.0]))


def test_make_uniform_points_is_seed_reproducible():
    points_a = make_uniform_points(5, 2, rng=123)
    points_b = make_uniform_points(5, 2, rng=123)

    np.testing.assert_allclose(points_a, points_b)


def test_make_jittered_grid_points_places_one_point_per_cell_reproducibly():
    points_a = make_jittered_grid_points(2, 3, rng=123)
    points_b = make_jittered_grid_points(2, 3, rng=123)

    np.testing.assert_allclose(points_a, points_b)
    assert points_a.shape == (8, 3)
    cell_indices = np.floor(points_a * 2).astype(int)
    assert sorted(map(tuple, cell_indices.tolist())) == sorted(
        map(tuple, np.indices((2, 2, 2)).reshape(3, -1).T.tolist())
    )
    assert np.all((0.0 <= points_a) & (points_a < 1.0))


def test_make_labeled_sample_supports_jittered_grid_family():
    sample = make_labeled_sample(
        8,
        3,
        rng=123,
        family="jittered_grid",
        grid_size=2,
        top_b=1,
    )

    assert sample["points"].shape == (8, 3)
    assert len(sample["supports"]) >= 1


def test_jittered_grid_family_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        make_labeled_sample(7, 3, rng=123, family="jittered_grid", grid_size=2)


def test_make_labeled_sample_contains_points_and_support_labels():
    sample = make_labeled_sample(6, 2, rng=123, top_b=1)

    assert sample["points"].shape == (6, 2)
    assert len(sample["supports"]) >= 1
    assert sample["supports"][0]["alpha"].shape == (6, 2)
    assert sample["supports"][0]["beta"].shape == (6,)


def test_save_load_and_torch_dataset_roundtrip(tmp_path):
    path = tmp_path / "samples.npz"
    samples = make_labeled_samples(3, 5, 2, rng=123, top_b=1)

    save_labeled_samples(path, samples, seed=123, top_b=1)
    arrays = load_labeled_samples(path)
    dataset = LabeledPointSetDataset(path)

    assert arrays["points"].shape == (3, 5, 2)
    assert arrays["support_mask"].shape[:2] == arrays["scores"].shape
    assert len(dataset) == 3
    assert tuple(dataset[0]["points"].shape) == (5, 2)


def test_local_discrepancy_baseline_scores_each_point_corner():
    points = np.array(
        [
            [0.2, 0.2],
            [0.4, 0.5],
            [0.9, 0.1],
        ]
    )

    scores = local_discrepancy_scores(points)
    selected = select_top_k(scores, 2)

    np.testing.assert_allclose(
        scores,
        np.array(
            [
                delta_minus(points, points[0]),
                delta_minus(points, points[1]),
                delta_minus(points, points[2]),
            ]
        ),
    )
    assert selected.tolist() == np.argsort(-scores, kind="stable")[:2].tolist()


def test_random_subset_samples_k_distinct_indices_reproducibly():
    selected_a = random_subset(10, 4, rng=123)
    selected_b = random_subset(10, 4, rng=123)

    np.testing.assert_array_equal(selected_a, selected_b)
    assert selected_a.shape == (4,)
    assert len(set(selected_a.tolist())) == 4
    assert np.all((0 <= selected_a) & (selected_a < 10))


def test_restricted_discrepancy_scores_only_selected_supports():
    points = np.array(
        [
            [0.2, 0.2],
            [0.4, 0.5],
            [0.9, 0.1],
        ]
    )

    restricted = restricted_discrepancy(points, [0, 1])

    manual = max(
        [score_support(points, (0,)), score_support(points, (1,)), score_support(points, (0, 1))],
        key=lambda item: item["score"],
    )
    assert restricted["support"] == manual["support"]
    assert restricted["score"] == manual["score"]


def test_recall_and_regret_at_k():
    points = np.array(
        [
            [0.2, 0.2],
            [0.4, 0.5],
            [0.9, 0.1],
        ]
    )
    optimal = find_top_supports(points, top_b=1)[0]

    assert recall_at_k([0, 1], [(0, 2), (1,)]) == 1.0
    assert regret_at_k(points, [0, 1, 2], optimal["score"]) == 1.0


def test_evaluate_local_baseline_arrays_returns_metrics():
    samples = make_labeled_samples(2, 5, 2, rng=123, top_b=1)

    metrics = evaluate_local_baseline_arrays(samples_to_arrays(samples), k=2)

    assert metrics["num_samples"] == 2.0
    assert metrics["k"] == 2.0
    assert 0.0 <= metrics["mean_recall"] <= 1.0


def test_evaluate_random_baseline_arrays_returns_reproducible_metrics():
    samples = make_labeled_samples(2, 5, 2, rng=123, top_b=1)
    arrays = samples_to_arrays(samples)

    metrics_a = evaluate_random_baseline_arrays(arrays, k=2, seed=123, repeats=3)
    metrics_b = evaluate_random_baseline_arrays(arrays, k=2, seed=123, repeats=3)

    assert metrics_a == metrics_b
    assert metrics_a["num_samples"] == 2.0
    assert metrics_a["k"] == 2.0
    assert metrics_a["repeats"] == 3.0
    assert 0.0 <= metrics_a["mean_recall"] <= 1.0


def test_zero_layer_graph_forward_returns_one_logit_per_point():
    model = CoordinateKnnGraphPointScorer(hidden_dim=8, num_layers=0, k=0)
    points = np.random.default_rng(0).random((4, 5, 2), dtype=np.float32)

    logits = model(torch.as_tensor(points))

    assert tuple(logits.shape) == (4, 5)
    assert tuple(model.last_coordinate_logits.shape) == (4, 5, 2)


def test_zero_layer_graph_forward_accepts_different_dimensions():
    model = CoordinateKnnGraphPointScorer(hidden_dim=8, num_layers=0, k=0)
    points_2d = torch.as_tensor(np.random.default_rng(0).random((4, 5, 2), dtype=np.float32))
    points_3d = torch.as_tensor(np.random.default_rng(1).random((4, 5, 3), dtype=np.float32))

    assert tuple(model(points_2d).shape) == (4, 5)
    assert tuple(model(points_3d).shape) == (4, 5)


def test_zero_layer_graph_is_coordinate_permutation_invariant_for_point_scores():
    model = CoordinateKnnGraphPointScorer(hidden_dim=8, num_layers=0, k=0)
    points = torch.as_tensor(np.random.default_rng(0).random((4, 5, 3), dtype=np.float32))
    permutation = torch.tensor([2, 0, 1])

    logits = model(points)
    permuted_logits = model(points[:, :, permutation])

    torch.testing.assert_close(logits, permuted_logits)


def test_zero_layer_graph_does_not_use_knn_constraints():
    model = CoordinateKnnGraphPointScorer(hidden_dim=8, num_layers=0, k=5)
    points = torch.as_tensor(np.random.default_rng(0).random((4, 5, 2), dtype=np.float32))

    assert tuple(model(points).shape) == (4, 5)


def test_coordinate_knn_graph_forward_returns_one_logit_per_point():
    model = CoordinateKnnGraphPointScorer(hidden_dim=8, num_layers=1, k=2)
    points = torch.as_tensor(np.random.default_rng(0).random((4, 5, 2), dtype=np.float32))

    logits = model(points)

    assert tuple(logits.shape) == (4, 5)
    assert tuple(model.last_coordinate_logits.shape) == (4, 5, 2)


@pytest.mark.parametrize(
    "graph_kind",
    [
        "knn_euclidean",
        "knn_linf",
        "rank_knn_euclidean",
        "rank_knn_linf",
        "rank_adjacency",
    ],
)
def test_coordinate_graph_forward_supports_graph_kinds(graph_kind):
    model = CoordinateKnnGraphPointScorer(
        hidden_dim=8,
        num_layers=1,
        k=2,
        graph_kind=graph_kind,
    )
    points = torch.as_tensor(np.random.default_rng(0).random((4, 5, 3), dtype=np.float32))

    logits = model(points)

    assert tuple(logits.shape) == (4, 5)
    assert tuple(model.last_coordinate_logits.shape) == (4, 5, 3)


def test_coordinate_knn_graph_forward_accepts_different_dimensions():
    model = CoordinateKnnGraphPointScorer(hidden_dim=8, num_layers=1, k=2)
    points_2d = torch.as_tensor(np.random.default_rng(0).random((4, 5, 2), dtype=np.float32))
    points_3d = torch.as_tensor(np.random.default_rng(1).random((4, 5, 3), dtype=np.float32))

    assert tuple(model(points_2d).shape) == (4, 5)
    assert tuple(model(points_3d).shape) == (4, 5)


def test_coordinate_knn_graph_is_coordinate_permutation_invariant_for_point_scores():
    model = CoordinateKnnGraphPointScorer(hidden_dim=8, num_layers=1, k=2)
    points = torch.as_tensor(np.random.default_rng(0).random((4, 5, 3), dtype=np.float32))
    permutation = torch.tensor([2, 0, 1])

    logits = model(points)
    permuted_logits = model(points[:, :, permutation])

    torch.testing.assert_close(logits, permuted_logits)


def test_coordinate_knn_graph_rejects_k_at_forward_when_too_large():
    model = CoordinateKnnGraphPointScorer(hidden_dim=8, num_layers=1, k=5)
    points = torch.as_tensor(np.random.default_rng(0).random((4, 5, 2), dtype=np.float32))

    with pytest.raises(ValueError):
        model(points)


def test_coordinate_unshared_graph_is_coordinate_order_sensitive():
    model = CoordinateKnnGraphPointScorer(
        hidden_dim=8,
        num_layers=0,
        k=0,
        coordinate_shared=False,
        input_dim=3,
    )
    points = torch.as_tensor(np.random.default_rng(0).random((4, 5, 3), dtype=np.float32))
    permutation = torch.tensor([2, 0, 1])

    logits = model(points)
    permuted_logits = model(points[:, :, permutation])

    assert not torch.allclose(logits, permuted_logits)


def test_coordinate_unshared_graph_rejects_different_dimensions():
    model = CoordinateKnnGraphPointScorer(
        hidden_dim=8,
        num_layers=0,
        k=0,
        coordinate_shared=False,
        input_dim=3,
    )
    points = torch.as_tensor(np.random.default_rng(0).random((4, 5, 4), dtype=np.float32))

    with pytest.raises(ValueError):
        model(points)


def test_flat_mlp_is_point_order_sensitive():
    model = FlatSetMlpPointScorer(n=5, d=3, hidden_dim=8)
    points = torch.as_tensor(np.random.default_rng(0).random((4, 5, 3), dtype=np.float32))
    permutation = torch.tensor([4, 3, 2, 1, 0])

    logits = model(points)
    permuted_logits = model(points[:, permutation, :])

    assert not torch.allclose(logits[:, permutation], permuted_logits)


def test_beta_union_target_collapses_tied_supports(tmp_path):
    path = tmp_path / "samples.npz"
    samples = make_labeled_samples(2, 5, 2, rng=123, top_b=1)
    save_labeled_samples(path, samples)
    batch = next(iter(DataLoader(LabeledPointSetDataset(path), batch_size=2)))

    target = beta_union_target(batch)

    assert tuple(target.shape) == (2, 5)
    assert target.max().item() <= 1.0
    assert target.min().item() >= 0.0


def test_alpha_union_target_collapses_tied_supports(tmp_path):
    path = tmp_path / "samples.npz"
    samples = make_labeled_samples(2, 5, 2, rng=123, top_b=1)
    save_labeled_samples(path, samples)
    batch = next(iter(DataLoader(LabeledPointSetDataset(path), batch_size=2)))

    target = alpha_union_target(batch)

    assert tuple(target.shape) == (2, 5, 2)
    assert target.max().item() <= 1.0
    assert target.min().item() >= 0.0


def test_train_and_evaluate_zero_layer_graph_smoke(tmp_path):
    dataset_path = tmp_path / "samples.npz"
    transfer_dataset_path = tmp_path / "transfer_samples.npz"
    checkpoint_path = tmp_path / "graph_zero_layer.pt"
    samples = make_labeled_samples(4, 5, 2, rng=123, top_b=1)
    transfer_samples = make_labeled_samples(3, 5, 3, rng=456, top_b=1)
    save_labeled_samples(dataset_path, samples)
    save_labeled_samples(transfer_dataset_path, transfer_samples)

    checkpoint = train_point_scorer(
        dataset_path,
        checkpoint_path,
        model_name="graph",
        epochs=1,
        batch_size=2,
        hidden_dim=8,
        graph_layers=0,
        graph_k=0,
        alpha_weight=1.0,
        seed=123,
        device="cpu",
        show_progress=False,
    )
    metrics = evaluate_checkpoint_file(
        str(dataset_path),
        str(checkpoint_path),
        k=2,
        device="cpu",
    )
    transfer_metrics = evaluate_checkpoint_file(
        str(transfer_dataset_path),
        str(checkpoint_path),
        k=3,
        device="cpu",
    )

    assert checkpoint_path.exists()
    assert checkpoint["model"] == "graph"
    assert checkpoint["graph_layers"] == 0
    assert "train_alpha_loss" in checkpoint["history"][0]
    assert metrics["num_samples"] == 4.0
    assert transfer_metrics["num_samples"] == 3.0
    assert 0.0 <= metrics["mean_recall"] <= 1.0


def test_train_and_evaluate_graph_smoke(tmp_path):
    dataset_path = tmp_path / "samples.npz"
    checkpoint_path = tmp_path / "graph.pt"
    samples = make_labeled_samples(4, 5, 2, rng=123, top_b=1)
    save_labeled_samples(dataset_path, samples)

    checkpoint = train_point_scorer(
        dataset_path,
        checkpoint_path,
        model_name="graph",
        epochs=1,
        batch_size=2,
        hidden_dim=8,
        graph_layers=1,
        graph_k=2,
        seed=123,
        device="cpu",
        show_progress=False,
    )
    metrics = evaluate_checkpoint_file(
        str(dataset_path),
        str(checkpoint_path),
        k=2,
        device="cpu",
    )

    assert checkpoint_path.exists()
    assert checkpoint["model"] == "graph"
    assert metrics["num_samples"] == 4.0
    assert 0.0 <= metrics["mean_recall"] <= 1.0


def test_train_and_evaluate_coordinate_unshared_graph_smoke(tmp_path):
    dataset_path = tmp_path / "samples.npz"
    checkpoint_path = tmp_path / "unshared_graph.pt"
    samples = make_labeled_samples(4, 5, 2, rng=123, top_b=1)
    save_labeled_samples(dataset_path, samples)

    checkpoint = train_point_scorer(
        dataset_path,
        checkpoint_path,
        model_name="graph",
        epochs=1,
        batch_size=2,
        hidden_dim=8,
        graph_layers=0,
        graph_k=0,
        coordinate_shared=False,
        alpha_weight=0.0,
        seed=123,
        device="cpu",
        show_progress=False,
    )
    metrics = evaluate_checkpoint_file(
        str(dataset_path),
        str(checkpoint_path),
        k=2,
        device="cpu",
    )

    assert checkpoint_path.exists()
    assert checkpoint["model"] == "graph"
    assert checkpoint["coordinate_shared"] is False
    assert checkpoint["input_n"] == 5
    assert checkpoint["input_d"] == 2
    assert metrics["num_samples"] == 4.0


def test_train_with_validation_tracks_best_checkpoint(tmp_path):
    dataset_path = tmp_path / "samples.npz"
    val_path = tmp_path / "val_samples.npz"
    checkpoint_path = tmp_path / "graph_zero_layer.pt"
    save_labeled_samples(dataset_path, make_labeled_samples(4, 5, 2, rng=123, top_b=1))
    save_labeled_samples(val_path, make_labeled_samples(3, 5, 2, rng=456, top_b=1))

    checkpoint = train_point_scorer(
        dataset_path,
        checkpoint_path,
        val_dataset_path=val_path,
        model_name="graph",
        epochs=1,
        batch_size=2,
        hidden_dim=8,
        graph_layers=0,
        graph_k=0,
        seed=123,
        device="cpu",
        show_progress=False,
    )

    assert checkpoint_path.exists()
    assert "val_loss" in checkpoint["history"][0]
    assert checkpoint["best_metric"] == "val_loss"
    assert "best_state_dict" in checkpoint
