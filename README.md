# GRL Project

This repository contains early research code for graph representation learning
heuristics for star discrepancy search.

The detailed project specification lives outside this repository in the
zettelkasten. It is linked here as:

```text
spec.md -> /Users/kmark/zettelkasten/20 Projects/GRL_project/GRL project idea.md
```

Treat `spec.md` as the source of truth for the mathematical formulation,
training targets, benchmark ideas, and candidate model architectures.

## Repository Structure

The repository is intentionally small for now. This is research code, not a
general-purpose library.

```text
grl/
  __init__.py
  discrepancy.py
  data.py
  baseline.py
  graphs.py
  models.py
  train.py
  eval.py

scripts/
  make_data.py
  train.py
  eval.py
  eval_to_json.py
  run_ablation_config.py
  run_ablations.sh
  summarize_ablations.py
  slurm/l40s.header

tests/
  test_discrepancy.py
  test_graphs.py
```

## File Responsibilities

`grl/discrepancy.py`

Exact star discrepancy and support-set utilities. This should contain the core
mathematical target: candidate corners, minus star discrepancy evaluation,
support-set enumeration, top support extraction, and alpha/beta label creation.

`grl/data.py`

Point-set generation and labeled dataset construction. This should start with
uniform point sets and small exact labels, then later grow to additional point
set families if needed.

`grl/baseline.py`

Non-learned baseline scoring methods. The first baseline scores each point by
the local discrepancy obtained when using that point itself as the candidate
corner.

`grl/graphs.py`

Graph construction for ablations. The nodes are always the input points, while
edge construction is the main variable: Euclidean kNN, l-infinity kNN,
rank-space kNN, rank-adjacency graphs, and later dominance-style graph
families. This module owns both NumPy graph builders for inspection/tests and
the torch graph tensor builder used by learned point scorers. The torch
interface returns gather-ready neighbor indices, edge masks, and coordinate-wise
edge features so model code does not duplicate graph topology rules.

`grl/models.py`

Model definitions. The learned model is a coordinate-shared graph point scorer.
With `--graph-layers 0`, it performs no edge construction or message passing and
acts as the no-connectivity learned baseline. With positive graph layers, it
uses graph tensors from `grl/graphs.py` for coordinate-wise message passing.
Both modes are intended to transfer across point-set size and dimension.

`grl/train.py`

Reusable training code called by scripts: configuration handling, model setup,
loss calculation, checkpointing, and the training loop.

`grl/eval.py`

Evaluation metrics and checks: recall against optimal support sets, regret for a
chosen budget `K`, coordinate-permutation checks, and comparison helpers for
graph ablations.

`scripts/make_data.py`

Thin command-line entrypoint for generating labeled data.

`scripts/train.py`

Thin command-line entrypoint for running training.

`scripts/eval.py`

Thin command-line entrypoint for evaluating a model or baseline.

`scripts/run_ablation_config.py`

Runs one hard-coded ablation config. A config is a single model or baseline
method, such as `local`, `random`, `rank_knn_linf`, or `flat_mlp`. Graph configs
train first, then evaluate all datasets and budgets assigned to that config,
and write one CSV file of metrics.

`scripts/run_ablations.sh`

Builds the reproducible ablation run directory and, in local or Slurm mode,
runs one job per hard-coded model/baseline config. The shell script is only the
job-array driver; the exact ablation plan lives in `scripts/run_ablation_config.py`.

`scripts/summarize_ablations.py`

Aggregates per-config metric CSV files into `metrics.csv`, Markdown tables, SVG
plots, and a summary file.

`scripts/slurm/l40s.header`

Cluster-specific Slurm settings for one L40S GPU per config job. The ablation
runner injects job names, array indices, log paths, and working directory.

`tests/test_discrepancy.py`

Correctness tests for exact discrepancy and support-label behavior.

`tests/test_graphs.py`

Tests for graph construction behavior and invariants needed by edge-ablation
experiments.

## Current Baseline Flow

Generate a tiny labeled dataset:

```bash
uv run python scripts/make_data.py data/smoke.npz --num-samples 10 --n 6 --d 2 --seed 0
```

Evaluate the local-discrepancy baseline:

```bash
uv run python scripts/eval.py data/smoke.npz --k 2
```

Evaluate the random subset baseline:

```bash
uv run python scripts/eval.py data/smoke.npz --baseline random --k 2 --seed 0 --repeats 20
```

Train the no-connectivity learned baseline:

```bash
uv run python scripts/train.py data/smoke.npz checkpoints/graph_zero_layer.pt \
  --val data/smoke_val.npz --epochs 20 --batch-size 32 --graph-layers 0
```

Evaluate the no-connectivity learned baseline:

```bash
uv run python scripts/eval.py data/smoke.npz --baseline graph --checkpoint checkpoints/graph_zero_layer.pt --k 2
```

Train the coordinate-wise kNN graph model:

```bash
uv run python scripts/train.py data/smoke.npz checkpoints/graph.pt \
  --val data/smoke_val.npz --epochs 20 --batch-size 32 \
  --graph-layers 2 --graph-k 8 --graph-metric euclidean
```

Evaluate the graph model:

```bash
uv run python scripts/eval.py data/smoke.npz --baseline graph --checkpoint checkpoints/graph.pt --k 2
```

Training uses progress bars and reports train/validation losses. When `--val`
is provided, checkpoints include the best validation state and evaluation uses
that state by default. The learned models train with both beta point loss and
alpha coordinate loss; `--alpha-weight` controls the coordinate-loss weight.

## Ablation Suite

The ablation suite is intentionally hard-coded for reproducibility. Each method
config is one job:

- non-learned baselines: `local`, `random`
- depth ablations: `depth0_no_graph`, `depth1_rank_knn_linf`, `depth2_rank_knn_linf`
- connectivity ablations: `knn_euclidean`, `knn_linf`, `rank_knn_euclidean`,
  `rank_knn_linf`, `rank_adjacency`
- transfer/equivariance checks: `coord_unshared_rank_knn_linf`, `flat_mlp`

Create a readable plan without running jobs:

```bash
./scripts/run_ablations.sh --mode plan --run-id smoke_plan
```

Run every config sequentially on the local machine:

```bash
./scripts/run_ablations.sh --mode local --run-id smoke_local \
  --epochs 20 --batch-size 32 --seed 0
```

Write a Slurm config array, one array task per method config:

```bash
./scripts/run_ablations.sh --mode slurm --run-id main_ablation
```

Submit the Slurm config array and dependent summary job:

```bash
./scripts/run_ablations.sh --mode slurm --submit --run-id main_ablation \
  --device cuda --slurm-header scripts/slurm/l40s.header
```

Each config job writes `results/ablations/<run-id>/metrics/<method>.csv`.
After all config jobs finish, `scripts/summarize_ablations.py` aggregates those
CSVs into:

```text
results/ablations/<run-id>/
  metrics.csv
  tables/
  plots/
  summary.md
```

To inspect or run one config directly:

```bash
uv run python scripts/run_ablation_config.py --list-methods

uv run python scripts/run_ablation_config.py rank_knn_linf \
  --output results/ablations/manual/metrics/rank_knn_linf.csv \
  --epochs 20 --batch-size 32 --seed 0
```

## Cluster Runbook

The intended cluster flow is: clone or sync the repo, install with `uv`, copy
the `.npz` datasets, submit the hard-coded config array, then copy the run
directory back.

On the cluster, clone and install:

```bash
git clone <repo-url> grl_project
cd grl_project
uv sync --dev
```

If the repo is not remote-hosted yet, sync it from this machine instead:

```bash
rsync -av --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
  /Users/kmark/Documents/Repositories/_lectures/grl_project/ \
  <cluster-host>:~/grl_project/
```

Copy the data needed by the hard-coded suite:

```bash
rsync -av data/ <cluster-host>:~/grl_project/data/
```

On the cluster, check the run plan before submitting:

```bash
cd ~/grl_project
./scripts/run_ablations.sh --mode plan --run-id main_ablation \
  --device cuda --slurm-header scripts/slurm/l40s.header
```

Submit the full ablation:

```bash
./scripts/run_ablations.sh --mode slurm --submit --run-id main_ablation \
  --device cuda --slurm-header scripts/slurm/l40s.header \
  --epochs 20 --batch-size 32 --eval-batch-size 256 --seed 0
```

Monitor jobs:

```bash
squeue -u "$USER"
tail -f results/ablations/main_ablation/logs/config_<jobid>_<taskid>.out
```

After the dependent report job finishes, copy results back locally:

```bash
rsync -av <cluster-host>:~/grl_project/results/ablations/main_ablation/ \
  /Users/kmark/Documents/Repositories/_lectures/grl_project/results/ablations/main_ablation/
```

The files to inspect locally are:

```text
results/ablations/main_ablation/metrics.csv
results/ablations/main_ablation/summary.md
results/ablations/main_ablation/tables/
results/ablations/main_ablation/plots/
results/ablations/main_ablation/logs/
```

If a config job fails, rerun that method directly on the cluster after fixing
the issue:

```bash
uv run python scripts/run_ablation_config.py rank_knn_linf \
  --output results/ablations/main_ablation/metrics/rank_knn_linf.csv \
  --epochs 20 --batch-size 32 --eval-batch-size 256 --seed 0 --device cuda

uv run python scripts/summarize_ablations.py results/ablations/main_ablation
```
