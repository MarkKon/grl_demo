"""Run one hard-coded ablation config: train if needed, evaluate, write CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grl.eval import (  # noqa: E402
    evaluate_checkpoint_file,
    evaluate_local_baseline_file,
    evaluate_random_baseline_file,
)
from grl.train import train_point_scorer  # noqa: E402


TRAIN = "data/uniform_d3_n64_train.npz"
VAL = "data/uniform_d3_n64_val.npz"

DATASETS = {
    "d3_n64": ("data/uniform_d3_n64_test.npz", (4, 8, 16)),
    "d3_n128": ("data/uniform_d3_n128_test.npz", (4, 8, 16)),
    "d4_n64": ("data/uniform_d4_n64_test.npz", (4, 8, 16)),
    "jitter_d3_n64": ("data/jittered_grid4_d3_n64_test.npz", (4, 8, 16)),
}

SCOPES = {
    "d_ge_3": ("d3_n64", "d3_n128", "d4_n64", "jitter_d3_n64"),
    "d3_only": ("d3_n64", "d3_n128", "jitter_d3_n64"),
    "d3_n64_only": ("d3_n64", "jitter_d3_n64"),
}


@dataclass(frozen=True)
class AblationConfig:
    method: str
    kind: Literal["local", "random", "graph"]
    scope: str = "d_ge_3"
    checkpoint: str | None = None
    train_kwargs: dict[str, Any] = field(default_factory=dict)


CONFIGS = [
    AblationConfig("local", "local"),
    AblationConfig("random", "random"),
    AblationConfig(
        "depth0_no_graph",
        "graph",
        checkpoint="checkpoints/ablations/depth0_no_graph.pt",
        train_kwargs={"model_name": "graph", "graph_layers": 0},
    ),
    AblationConfig(
        "depth1_rank_knn_linf",
        "graph",
        checkpoint="checkpoints/ablations/depth1_rank_knn_linf.pt",
        train_kwargs={
            "model_name": "graph",
            "graph_layers": 1,
            "graph_k": 8,
            "graph_kind": "rank_knn_linf",
        },
    ),
    AblationConfig(
        "depth2_rank_knn_linf",
        "graph",
        checkpoint="checkpoints/ablations/depth2_rank_knn_linf.pt",
        train_kwargs={
            "model_name": "graph",
            "graph_layers": 2,
            "graph_k": 8,
            "graph_kind": "rank_knn_linf",
        },
    ),
    AblationConfig(
        "knn_euclidean",
        "graph",
        checkpoint="checkpoints/ablations/knn_euclidean.pt",
        train_kwargs={
            "model_name": "graph",
            "graph_layers": 2,
            "graph_k": 8,
            "graph_kind": "knn_euclidean",
        },
    ),
    AblationConfig(
        "knn_linf",
        "graph",
        checkpoint="checkpoints/ablations/knn_linf.pt",
        train_kwargs={
            "model_name": "graph",
            "graph_layers": 2,
            "graph_k": 8,
            "graph_kind": "knn_linf",
        },
    ),
    AblationConfig(
        "rank_knn_euclidean",
        "graph",
        checkpoint="checkpoints/ablations/rank_knn_euclidean.pt",
        train_kwargs={
            "model_name": "graph",
            "graph_layers": 2,
            "graph_k": 8,
            "graph_kind": "rank_knn_euclidean",
        },
    ),
    AblationConfig(
        "rank_knn_linf",
        "graph",
        checkpoint="checkpoints/ablations/rank_knn_linf.pt",
        train_kwargs={
            "model_name": "graph",
            "graph_layers": 2,
            "graph_k": 8,
            "graph_kind": "rank_knn_linf",
        },
    ),
    AblationConfig(
        "rank_adjacency",
        "graph",
        checkpoint="checkpoints/ablations/rank_adjacency.pt",
        train_kwargs={
            "model_name": "graph",
            "graph_layers": 2,
            "graph_kind": "rank_adjacency",
        },
    ),
    AblationConfig(
        "coord_unshared_rank_knn_linf",
        "graph",
        scope="d3_only",
        checkpoint="checkpoints/ablations/coord_unshared_rank_knn_linf.pt",
        train_kwargs={
            "model_name": "graph",
            "graph_layers": 2,
            "graph_k": 8,
            "graph_kind": "rank_knn_linf",
            "coordinate_shared": False,
        },
    ),
    AblationConfig(
        "flat_mlp",
        "graph",
        scope="d3_n64_only",
        checkpoint="checkpoints/ablations/flat_mlp.pt",
        train_kwargs={"model_name": "flat_mlp", "hidden_dim": 128, "alpha_weight": 0.0},
    ),
]

CONFIG_BY_METHOD = {config.method: config for config in CONFIGS}
FIELDNAMES = [
    "method",
    "kind",
    "dataset_label",
    "dataset",
    "k",
    "num_samples",
    "mean_recall",
    "mean_regret",
    "checkpoint",
    "seed",
    "repeats",
]


def dataset_label(path: str) -> str:
    return Path(path).stem


def train_if_needed(config: AblationConfig, args: argparse.Namespace) -> None:
    if config.kind != "graph":
        return
    if config.checkpoint is None:
        raise ValueError(f"{config.method} has no checkpoint path")
    train_point_scorer(
        TRAIN,
        config.checkpoint,
        val_dataset_path=VAL,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        show_progress=not args.no_progress,
        **config.train_kwargs,
    )


def evaluate_config(
    config: AblationConfig, args: argparse.Namespace
) -> list[dict[str, Any]]:
    rows = []
    for dataset_key in SCOPES[config.scope]:
        dataset, ks = DATASETS[dataset_key]
        for k in ks:
            if config.kind == "local":
                metrics = evaluate_local_baseline_file(dataset, k=k)
                checkpoint = ""
                repeats = ""
            elif config.kind == "random":
                metrics = evaluate_random_baseline_file(
                    dataset,
                    k=k,
                    seed=args.seed,
                    repeats=args.random_repeats,
                )
                checkpoint = ""
                repeats = args.random_repeats
            else:
                if config.checkpoint is None:
                    raise ValueError(f"{config.method} has no checkpoint path")
                metrics = evaluate_checkpoint_file(
                    dataset,
                    config.checkpoint,
                    k=k,
                    batch_size=args.eval_batch_size,
                    device=args.device,
                )
                checkpoint = config.checkpoint
                repeats = ""
            rows.append(
                {
                    "method": config.method,
                    "kind": config.kind,
                    "dataset_label": dataset_label(dataset),
                    "dataset": dataset,
                    "k": k,
                    "checkpoint": checkpoint,
                    "seed": args.seed,
                    "repeats": repeats,
                    **metrics,
                }
            )
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("method", nargs="?")
    parser.add_argument("--list-methods", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-repeats", type=int, default=1)
    parser.add_argument("--device")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_methods:
        for config in CONFIGS:
            print(config.method)
        return
    if args.method not in CONFIG_BY_METHOD:
        methods = ", ".join(CONFIG_BY_METHOD)
        raise SystemExit(f"method must be one of: {methods}")
    if args.output is None:
        raise SystemExit("--output is required")

    config = CONFIG_BY_METHOD[args.method]
    train_if_needed(config, args)
    rows = evaluate_config(config, args)
    write_rows(Path(args.output), rows)
    print(f"{config.method}: wrote {args.output}")


if __name__ == "__main__":
    main()
