#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 PHYSICAL_GPU" >&2
  exit 2
fi

GPU="$1"
ROOT="/home/dataset-assist-0/einwang/workspace/haorandu/EGGPU_Paper_Repo/EG_Evaluation"
PYTHON_BIN="/home/dataset-assist-0/einwang/conda_cache/conda_env/EGGPU/bin/python"
CUDA_ROOT="/home/dataset-assist-0/einwang/conda_cache/conda_env/tongyideepresearch"
RESULT_ROOT="${ROOT}/benchmarking/results/nxcugraph_strict_main99_20260730"
ORKUT_OUT="${RESULT_ROOT}/large_com_orkut"
TWITTER_OUT="${RESULT_ROOT}/large_gap_twitter"

mkdir -p \
  "${ORKUT_OUT}/cupy_cache" \
  "${TWITTER_OUT}/cupy_cache"

export CUDA_VISIBLE_DEVICES="${GPU}"
export EGGPU_CUDA_ROOT="${CUDA_ROOT}"
export CUDA_PATH="${CUDA_ROOT}"
export CUDA_HOME="${CUDA_ROOT}"
export CUPY_CUDA_PATH="${CUDA_ROOT}"
export CUDAToolkit_ROOT="${CUDA_ROOT}"
export CONDA_PREFIX="${CUDA_ROOT}"

echo "[large 1/2] com-Orkut: eight successful nx-cugraph functions"
CUPY_CACHE_DIR="${ORKUT_OUT}/cupy_cache" \
  "${PYTHON_BIN}" benchmarking/run_nxcugraph_large_matrix.py \
    --manifests datasets/scaling/csr/com-Orkut/com-Orkut.json \
    --functions \
      PageRank LCC WCC BFS Dijkstra BellmanFord SSSP KCore \
    --gpu "${GPU}" \
    --repeat 5 \
    --memory-repeat 0 \
    --source-count 8 \
    --warmup-calls 3 \
    --timeout 600 \
    --load-timeout 900 \
    --out-dir "${ORKUT_OUT}" \
    |& tee "${ORKUT_OUT}/run.log"

echo "[large 2/2] GAP-twitter: three successful nx-cugraph functions"
CUPY_CACHE_DIR="${TWITTER_OUT}/cupy_cache" \
  "${PYTHON_BIN}" benchmarking/run_nxcugraph_large_matrix.py \
    --manifests datasets/scaling/csr/GAP-twitter/GAP-twitter.json \
    --functions PageRank BFS Dijkstra \
    --gpu "${GPU}" \
    --repeat 5 \
    --memory-repeat 0 \
    --source-count 8 \
    --warmup-calls 3 \
    --timeout 600 \
    --load-timeout 900 \
    --out-dir "${TWITTER_OUT}" \
    |& tee "${TWITTER_OUT}/run.log"

echo "Strict nx-cugraph large 11-cell run complete: ${RESULT_ROOT}"
