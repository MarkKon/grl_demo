"""Generate labeled point-set data for local experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grl.data import make_labeled_samples, save_labeled_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--family", choices=["uniform", "jittered_grid"], default="uniform")
    parser.add_argument("--grid-size", type=int)
    parser.add_argument("--top-b", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-include-ties", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    include_ties = not args.no_include_ties
    samples = make_labeled_samples(
        args.num_samples,
        args.n,
        args.d,
        rng=args.seed,
        family=args.family,
        grid_size=args.grid_size,
        top_b=args.top_b,
        include_ties=include_ties,
        show_progress=not args.no_progress,
    )
    save_labeled_samples(
        args.output,
        samples,
        seed=args.seed,
        family=args.family,
        grid_size=args.grid_size,
        top_b=args.top_b,
        include_ties=include_ties,
    )


if __name__ == "__main__":
    main()
