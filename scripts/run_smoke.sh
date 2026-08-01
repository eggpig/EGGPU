#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
GPU="${GPU:-0}"

CUDA_ROOT="${EGGPU_CUDA_ROOT:-${CUDA_PATH:-${CUDA_HOME:-${CUDAToolkit_ROOT:-${CONDA_PREFIX:-}}}}}"
LD_PATH="${LD_LIBRARY_PATH:-}"
if [[ -n "${CUDA_ROOT}" ]]; then
  for candidate in "${CUDA_ROOT}/lib64" "${CUDA_ROOT}/lib" "${CUDA_ROOT}/targets/x86_64-linux/lib"; do
    if [[ -d "${candidate}" ]]; then
      LD_PATH="${candidate}:${LD_PATH}"
    fi
  done
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
export EGGPU_MONITOR_GPU_INDEX=0
export EASYGRAPH_ENABLE_GPU=TRUE
export EASYGRAPH_GPU_STRICT_ERRORS=TRUE
export PYTHONPATH="${ROOT}/Easy-Graph:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${LD_PATH}"

"${PYTHON_BIN}" "${ROOT}/scripts/correctness_smoke.py"
