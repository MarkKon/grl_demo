# TODO

## Training

- Add weighted BCE for sparse positive beta/alpha targets.
- Add a full experiment-suite runner that trains and evaluates all baseline and
  graph variants across the saved train/validation/test splits.

## Experiments

- Compare local discrepancy, zero-layer graph, Euclidean kNN graph, and
  l-infinity kNN graph baselines.
- Report metrics across multiple budgets `K` in one evaluation run.
- Write experiment outputs as JSON or CSV for later plotting.

## Graphs

- Add dominance graph builders after the kNN experiments are working.
- Decide whether graph edges should be precomputed/cached for larger runs.
