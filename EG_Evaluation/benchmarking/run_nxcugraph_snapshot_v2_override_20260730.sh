#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "usage: $0 PHYSICAL_GPU DATASET FUNCTION ATTEMPT" >&2
  exit 2
fi

GPU="$1"
DATASET="$2"
FUNCTION="$3"
ATTEMPT="$4"
ROOT="/home/dataset-assist-0/einwang/workspace/haorandu/EGGPU_Paper_Repo/EG_Evaluation"
PYTHON_BIN="/home/dataset-assist-0/einwang/conda_cache/conda_env/EGGPU/bin/python"
RESULT_ROOT="${ROOT}/benchmarking/results/nxcugraph_strict_main99_20260730/snapshot_v2_overrides"
OUT_DIR="${RESULT_ROOT}/${DATASET}_${FUNCTION}_${ATTEMPT}"

mkdir -p "${OUT_DIR}/cupy_cache"
cd "${ROOT}"

CUDA_VISIBLE_DEVICES="${GPU}" \
CUPY_CACHE_DIR="${OUT_DIR}/cupy_cache" \
EGGPU_EXTERNAL_VISIBILITY_MARKER=TRUE \
"${PYTHON_BIN}" benchmarking/run_full_baselines.py \
  --gpu "${GPU}" \
  --out-dir "${OUT_DIR}" \
  --repeat 5 \
  --datasets "${DATASET}" \
  --functions "${FUNCTION}" \
  --baselines nx-cugraph \
  --measurement-mode timing \
  --nx-cugraph-warmup 3 \
  --library-timeout 600 \
  --inter-run-cooldown 0.1 \
  --sssp-sources 8 \
  >"${OUT_DIR}/run.log" 2>&1
