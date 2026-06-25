#!/usr/bin/env bash
set -euo pipefail

mkdir -p data

# d=2, n=32: cheap enough to make fairly large.
echo "Generating d=2, n=32 train split: 20000 samples"
uv run python scripts/make_data.py data/uniform_d2_n32_train.npz \
  --num-samples 25000 --n 32 --d 2 --seed 100

echo "Generating d=2, n=32 validation split: 1000 samples"
uv run python scripts/make_data.py data/uniform_d2_n32_val.npz \
  --num-samples 5000 --n 32 --d 2 --seed 101

echo "Generating d=2, n=32 test split: 1000 samples"
uv run python scripts/make_data.py data/uniform_d2_n32_test.npz \
  --num-samples 5000 --n 32 --d 2 --seed 102

# d=3, n=64: more expensive, but still reasonable.
echo "Generating d=3, n=64 train split: 1000 samples"
uv run python scripts/make_data.py data/uniform_d3_n64_train.npz \
  --num-samples 10000 --n 64 --d 3 --seed 200

echo "Generating d=3, n=64 validation split: 200 samples"
uv run python scripts/make_data.py data/uniform_d3_n64_val.npz \
  --num-samples 2000 --n 64 --d 3 --seed 201

echo "Generating d=3, n=64 test split: 200 samples"
uv run python scripts/make_data.py data/uniform_d3_n64_test.npz \
  --num-samples 2000 --n 64 --d 3 --seed 202

# d=3, n=128: test-only transfer/stress set.
echo "Generating d=3, n=128 test-only split: 100 samples"
uv run python scripts/make_data.py data/uniform_d3_n128_test.npz \
  --num-samples 1000 --n 128 --d 3 --seed 300

# d=4, n=64: test-only transfer/stress set.
echo "Generating d=4, n=64 test-only split: 1000 samples"
uv run python scripts/make_data.py data/uniform_d4_n64_test.npz \
  --num-samples 1000 --n 64 --d 4 --seed 400

# d=3, n=64: test-only distribution-transfer set on a 4x4x4 grid.
echo "Generating d=3, n=64 jittered-grid test-only split: 1000 samples"
uv run python scripts/make_data.py data/jittered_grid4_d3_n64_test.npz \
  --num-samples 1000 --n 64 --d 3 --family jittered_grid --grid-size 4 --seed 500

echo "Done."
