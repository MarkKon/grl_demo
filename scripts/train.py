"""Run a training job using the code in ``grl.train``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grl.train import train_point_scorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("output")
    parser.add_argument("--val")
    parser.add_argument(
        "--model",
        choices=["graph", "flat_mlp"],
        default="graph",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--alpha-weight", type=float, default=1.0)
    parser.add_argument("--graph-layers", type=int, default=0)
    parser.add_argument("--graph-k", type=int, default=8)
    parser.add_argument("--graph-metric", choices=["euclidean", "linf"], default="euclidean")
    parser.add_argument(
        "--graph-kind",
        choices=[
            "knn_euclidean",
            "knn_linf",
            "rank_knn_euclidean",
            "rank_knn_linf",
            "rank_adjacency",
        ],
    )
    parser.add_argument("--coordinate-unshared", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = train_point_scorer(
        args.dataset,
        args.output,
        val_dataset_path=args.val,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        alpha_weight=args.alpha_weight,
        graph_layers=args.graph_layers,
        graph_k=args.graph_k,
        graph_metric=args.graph_metric,
        graph_kind=args.graph_kind,
        coordinate_shared=not args.coordinate_unshared,
        seed=args.seed,
        device=args.device,
        show_progress=not args.no_progress,
    )
    print(f"saved: {args.output}")
    print(f"final_train_loss: {checkpoint['history'][-1]['train_loss']}")
    if "val_loss" in checkpoint["history"][-1]:
        print(f"final_val_loss: {checkpoint['history'][-1]['val_loss']}")
        print(f"best_val_loss: {checkpoint['best_value']}")


if __name__ == "__main__":
    main()
