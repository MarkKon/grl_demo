"""Evaluate a trained model or baseline using ``grl.eval``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grl.eval import (
    evaluate_checkpoint_file,
    evaluate_local_baseline_file,
    evaluate_random_baseline_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument(
        "--baseline",
        choices=["local", "random", "graph"],
        default="local",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.baseline == "local":
        metrics = evaluate_local_baseline_file(args.dataset, k=args.k)
    elif args.baseline == "random":
        metrics = evaluate_random_baseline_file(
            args.dataset,
            k=args.k,
            seed=args.seed,
            repeats=args.repeats,
        )
    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for learned baselines")
        metrics = evaluate_checkpoint_file(
            args.dataset,
            args.checkpoint,
            k=args.k,
            batch_size=args.batch_size,
            device=args.device,
        )
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
