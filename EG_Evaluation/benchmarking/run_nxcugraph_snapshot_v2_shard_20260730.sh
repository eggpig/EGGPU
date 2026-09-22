#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 PHYSICAL_GPU SHARD_LABEL DATASET_CSV" >&2
  exit 2
fi

GPU="$1"
LABEL="$2"
DATASETS="$3"
ROOT="/home/dataset-assist-0/einwang/workspace/haorandu/EGGPU_Paper_Repo/EG_Evaluation"
PYTHON_BIN="/home/dataset-assist-0/einwang/conda_cache/conda_env/EGGPU/bin/python"
RESULT_ROOT="${ROOT}/benchmarking/results/nxcugraph_strict_main99_20260730/snapshot_v2"
OUT_DIR="${RESULT_ROOT}/${LABEL}"
FUNCTIONS="PageRank,LCC,WCC,BFS,Dijkstra,BellmanFord,SSSP,KCore"

mkdir -p "${OUT_DIR}/cupy_cache"
cd "${ROOT}"

CUDA_VISIBLE_DEVICES="${GPU}" \
CUPY_CACHE_DIR="${OUT_DIR}/cupy_cache" \
EGGPU_EXTERNAL_VISIBILITY_MARKER=TRUE \
"${PYTHON_BIN}" benchmarking/run_full_baselines.py \
  --gpu "${GPU}" \
  --out-dir "${OUT_DIR}" \
  --repeat 5 \
  --datasets "${DATASETS}" \
  --functions "${FUNCTIONS}" \
  --baselines nx-cugraph \
  --measurement-mode timing \
  --nx-cugraph-warmup 3 \
  --library-timeout 600 \
  --inter-run-cooldown 0.1 \
  --sssp-sources 8 \
  >"${OUT_DIR}/run.log" 2>&1
