"""Evaluate one ablation task and write machine-readable metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grl.eval import (  # noqa: E402
    evaluate_checkpoint_file,
    evaluate_local_baseline_file,
    evaluate_random_baseline_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--method", required=True)
    parser.add_argument("--kind", choices=["local", "random", "graph"], required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.kind == "local":
        metrics = evaluate_local_baseline_file(args.dataset, k=args.k)
    elif args.kind == "random":
        metrics = evaluate_random_baseline_file(
            args.dataset,
            k=args.k,
            seed=args.seed,
            repeats=args.repeats,
        )
    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for graph evaluations")
        metrics = evaluate_checkpoint_file(
            args.dataset,
            args.checkpoint,
            k=args.k,
            batch_size=args.batch_size,
            device=args.device,
        )

    output = {
        "method": args.method,
        "kind": args.kind,
        "dataset": args.dataset,
        "dataset_label": Path(args.dataset).stem,
        "checkpoint": args.checkpoint,
        "k": args.k,
        "seed": args.seed,
        "repeats": args.repeats if args.kind == "random" else None,
        **metrics,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(
        f"{args.method:32s} {Path(args.dataset).stem:30s} "
        f"K={args.k:<3d} recall={metrics['mean_recall']:.4f} "
        f"regret={metrics['mean_regret']:.4f}"
    )


if __name__ == "__main__":
    main()
