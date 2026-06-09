#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

OUT_DIR="${1:-benchmarking/results/full_eval_gpu7_$(date +%Y%m%d_%H%M%S)}"
GPU_ID="${2:-${EG_EVAL_GPU:-7}}"
EASYGRAPH_REPO="${EASYGRAPH_REPO:-$(cd .. && pwd)/Easy-Graph}"
EASYGRAPH_GPU_BACKEND="${EASYGRAPH_GPU_BACKEND:-mine}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

"${PYTHON_BIN}" benchmarking/run_full_baselines.py \
  --gpu "$GPU_ID" \
  --out-dir "$OUT_DIR" \
  --library-timeout 180 \
  --pr-alpha 0.75 \
  --pr-eps 1e-6 \
  --pr-max-iter 200 \
  --easygraph-repo "$EASYGRAPH_REPO" \
  --easygraph-gpu-backend "$EASYGRAPH_GPU_BACKEND"
