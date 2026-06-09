#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
GPU="${GPU:-0}"

CUDA_ROOT="${EGGPU_CUDA_ROOT:-${CUDA_PATH:-${CUDA_HOME:-${CUDAToolkit_ROOT:-${CONDA_PREFIX:-}}}}}"
LD_PATH="${LD_LIBRARY_PATH:-}"
if [[ -n "${CUDA_ROOT}" ]]; then
  for candidate in "${CUDA_ROOT}/lib" "${CUDA_ROOT}/targets/x86_64-linux/lib"; do
    if [[ -d "${candidate}" ]]; then
      LD_PATH="${candidate}:${LD_PATH}"
    fi
  done
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
export EGGPU_MONITOR_GPU_INDEX="${GPU}"
export EASYGRAPH_ENABLE_GPU=TRUE
export EASYGRAPH_GPU_BACKEND=mine
export EASYGRAPH_GPU_STRICT_ERRORS=TRUE
export EASYGRAPH_GPU_CONSTRAINT_SMALLER_INTERSECTION=AUTO
export PYTHONPATH="${ROOT}/Easy-Graph:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${LD_PATH}"

SKIP_IDLE_CHECK=0
case "${EGGPU_ALLOW_BUSY_GPU:-${ALLOW_BUSY_GPU:-}}" in
  1|true|TRUE|yes|YES|on|ON)
    SKIP_IDLE_CHECK=1
    ;;
esac
if [[ "${SKIP_IDLE_CHECK}" -ne 1 ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_ROW="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F',' -v gpu="${GPU}" '$1 ~ "^[[:space:]]*" gpu "[[:space:]]*$" {gsub(/[[:space:]]/, "", $2); gsub(/[[:space:]]/, "", $3); print $2 "," $3; exit}')"
    if [[ -n "${GPU_ROW}" ]]; then
      IFS=',' read -r GPU_MEM GPU_UTIL <<< "${GPU_ROW}"
      if [[ "${GPU_MEM}" -gt "${EGGPU_IDLE_MAX_MEMORY_MB:-1024}" || "${GPU_UTIL}" -gt "${EGGPU_IDLE_MAX_UTILIZATION:-5}" ]]; then
        echo "GPU=${GPU} is busy for smoke (memory=${GPU_MEM} MiB, utilization=${GPU_UTIL}%)." >&2
        echo "Pick an idle GPU, or set EGGPU_ALLOW_BUSY_GPU=1 only for a deliberate debug smoke." >&2
        exit 2
      fi
    fi
  fi
fi

cd "${ROOT}/EG_Evaluation"
"${PYTHON_BIN}" benchmarking/preflight_full_eval_ready.py

OUT="benchmarking/results/repro_smoke_full_$(date +%Y%m%d_%H%M%S)"
"${PYTHON_BIN}" benchmarking/run_full_baselines.py \
  --gpu "${GPU}" \
  --repeat 1 \
  --warmup 0 \
  --easygraph-warmup 1 \
  --library-timeout 60 \
  --inter-run-cooldown 0.1 \
  --pr-alpha 0.75 \
  --pr-eps 1e-6 \
  --pr-max-iter 20 \
  --sssp-sources 2 \
  --bc-sources 2 \
  --datasets LastFM \
  --functions PageRank \
  --out-dir "${OUT}"

ABL_OUT="benchmarking/results/repro_smoke_ablation_$(date +%Y%m%d_%H%M%S).csv"
"${PYTHON_BIN}" benchmarking/run_eggpu_ablations.py \
  --experiment workflow \
  --variant full \
  --edge-path datasets/undirected/LastFM.txt \
  --dataset-name LastFM \
  --graph-type undirected \
  --functions PageRank,Constraint \
  --repeat 1 \
  --warmup 1 \
  --gpu "${GPU}" \
  --out "${ABL_OUT}"

echo "Smoke full result: ${OUT}"
echo "Smoke ablation result: ${ABL_OUT}"
