# Learning Subset Heuristics for Star Discrepancy with GNNs

This repository contains the code and experiment artifacts for a study of
inductive biases in graph neural networks. The model learns to rank points in a
point set so that a small selected subset retains a high-quality candidate for
the minus star discrepancy.

## Research question

How do point-permutation equivariance, coordinate equivariance, message
passing, and informed graph connectivity affect generalization of a learned
subset heuristic?

The experiments compare non-learned baselines, a flat MLP, a model without
message passing, multiple message-passing depths, coordinate-shared and
coordinate-unshared models, and several graph constructions. Transfer is
measured across point-set size, dimension, and data generation method.

## Main findings

- Point-permutation equivariance and message passing are the useful inductive
  biases in this setting.
- Two message-passing layers consistently improve over no message passing,
  especially under distribution shift.
- The tested graph constructions perform similarly; informed connectivity is
  not a strong differentiator once nearby points can exchange information.
- Coordinate sharing enables transfer to unseen dimensions, but does not
  improve the in-distribution result in this experiment.

The main metric is the regret ratio: the discrepancy found from the selected
subset divided by the exact discrepancy. Higher is better and `1.0` is exact.
Representative mean regret ratios at subset budget `K = 8` are:

| Method | d=3, n=64 | d=3, n=128 | d=4, n=64 | Jittered |
| --- | ---: | ---: | ---: | ---: |
| Random | 0.609 | 0.546 | 0.570 | 0.595 |
| Local discrepancy | 0.793 | 0.782 | 0.718 | 0.769 |
| Flat MLP | 0.611 | - | - | 0.631 |
| No message passing | 0.894 | 0.773 | 0.842 | 0.853 |
| 1-layer rank kNN | 0.963 | 0.895 | 0.926 | 0.936 |
| 2-layer rank kNN | 0.970 | 0.916 | 0.933 | 0.948 |

The complete result table is in
[`results/paper/metrics.csv`](results/paper/metrics.csv). Report figures are
generated locally in `results/paper/figures/` and are intentionally ignored by
Git.

## Setup and verification

The project uses Python 3.13 and [uv](https://docs.astral.sh/uv/). From the
repository root:

```bash
uv sync --dev
uv run pytest
```

Regenerate all report figures from the committed metrics:

```bash
uv run python scripts/make_paper_plots.py
```

This writes PDF, SVG, and 320 DPI PNG versions to
`results/paper/figures/`. PDF is the intended LaTeX input format.

## Repository layout

```text
grl/                    Core discrepancy, graph, model, training, and evaluation code
scripts/                Data generation, experiment, summary, and plotting entrypoints
data/                   Fixed train, validation, and test datasets used in the study
results/paper/          Committed metrics; generated report figures are ignored
tests/                  Mathematical and graph-construction tests
```

The code is intentionally compact research code rather than a general-purpose
library. The experiment configurations are explicit in
`scripts/run_ablation_config.py` so the compared methods can be audited in one
place.

## Experimental setup

The training distribution contains 10,000 uniform point sets with shape
`(n=64, d=3)`, with 2,000 validation and 2,000 in-distribution test examples.
Transfer tests contain 1,000 examples each for larger point sets `(128, 3)`,
higher dimension `(64, 4)`, and jittered sampling on a `4 x 4 x 4` grid.

The reported learned models use 300 epochs, batch size 32, Adam with learning
rate `1e-3`, hidden size 64 (128 for the flat MLP), and seed 0. The evaluation
budgets are `K in {4, 8, 16}`. Exact support sets provide both point labels and
coordinate-level labels; the graph models optimize the sum of their binary
cross-entropy losses.

## Reproducing the full ablation

Inspect the fixed experiment plan without starting training:

```bash
./scripts/run_ablations.sh --mode plan --run-id reproduction
```

Run all configurations sequentially (a CUDA GPU is recommended):

```bash
./scripts/run_ablations.sh --mode local --run-id reproduction \
  --epochs 300 --batch-size 32 --eval-batch-size 256 \
  --seed 0 --device cuda
```

The run writes per-method metrics, an aggregate CSV, Markdown tables, and plots
under `results/ablations/reproduction/`. To regenerate the paper-style figures
from that run:

```bash
uv run python scripts/make_paper_plots.py \
  --metrics results/ablations/reproduction/metrics.csv \
  --output-dir results/ablations/reproduction/paper_plots
```

For a Slurm cluster, the same explicit configuration list can be emitted as one
array job:

```bash
./scripts/run_ablations.sh --mode slurm --run-id reproduction \
  --epochs 300 --batch-size 32 --eval-batch-size 256 \
  --seed 0 --device cuda --slurm-header scripts/slurm/l40s.header
```

Add `--submit` after reviewing the generated task table and Slurm scripts.

## Key entrypoints

- `scripts/make_data.py`: generate exactly labelled point-set datasets.
- `scripts/train.py`: train one learned point scorer.
- `scripts/eval.py`: evaluate a learned or non-learned baseline.
- `scripts/run_ablation_config.py`: run one named experiment configuration.
- `scripts/run_ablations.sh`: run or schedule the complete fixed ablation.
- `scripts/summarize_ablations.py`: aggregate per-method metrics.
- `scripts/make_paper_plots.py`: regenerate the report figures.
