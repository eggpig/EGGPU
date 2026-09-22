#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${FINAL13_GPU:-7}"
CORE_ROOT="${FINAL13_RESULT_ROOT:-${ROOT}/benchmarking/results/final13_missing_gpu7_20260716_212844}"
FOLLOWUP_ROOT="${FINAL13_FOLLOWUP_ROOT:-${ROOT}/benchmarking/results/final13_followup_gpu7_20260717_complete}"
ASSET_ROOT="${FINAL13_ASSET_ROOT:-${ROOT}/../writing/EGGPU_FINAL_EXPERIMENT_ASSETS_13_COMPLETE_20260717}"
POLL_SECONDS="${FINAL13_WAIT_POLL_SECONDS:-20}"

echo "[chain] waiting for the already-running core experiment: ${CORE_ROOT}"
while [[ ! -s "${CORE_ROOT}/phase_status.json" ]]; do
  if ! pgrep -f "run_final_13_missing_core.sh|run_eggpu_scaling.py|run_nxcugraph_large_matrix.py" >/dev/null; then
    echo "[chain] core status is absent and no matching core process is alive" >&2
    exit 3
  fi
  sleep "${POLL_SECONDS}"
done

echo "[chain] core completed; starting repair and remaining experiments"
FINAL13_GPU="${GPU}" \
FINAL13_RESULT_ROOT="${CORE_ROOT}" \
FINAL13_FOLLOWUP_ROOT="${FOLLOWUP_ROOT}" \
PAPER_REPEAT="${PAPER_REPEAT:-5}" \
PAPER_MEMORY_REPEAT="${PAPER_MEMORY_REPEAT:-3}" \
PAPER_TIMEOUT="${PAPER_TIMEOUT:-100}" \
PAPER_LOAD_TIMEOUT="${PAPER_LOAD_TIMEOUT:-900}" \
COMMON_PY="${COMMON_PY:-/home/dataset-assist-0/einwang/conda_cache/conda_env/EGGPU/bin/python}" \
EGGPU_CUDA_ROOT="${EGGPU_CUDA_ROOT:-/home/dataset-assist-0/einwang/conda_cache/conda_env/sglang}" \
bash "${ROOT}/run_final_13_recovery_and_followup.sh" \
  |& tee "${CORE_ROOT}/recovery_and_followup.launcher.log"

echo "[chain] experiments completed; generating final paper bundle"
FINAL13_RESULT_ROOT="${CORE_ROOT}" \
FINAL13_FOLLOWUP_ROOT="${FOLLOWUP_ROOT}" \
FINAL13_ASSET_ROOT="${ASSET_ROOT}" \
COMMON_PY="${COMMON_PY:-/home/dataset-assist-0/einwang/conda_cache/conda_env/EGGPU/bin/python}" \
bash "${ROOT}/finalize_final_13_paper_assets.sh" \
  |& tee "${CORE_ROOT}/finalize_final_13_paper_assets.log"

echo "[chain] complete"
echo "[chain] core=${CORE_ROOT}"
echo "[chain] followup=${FOLLOWUP_ROOT}"
echo "[chain] assets=${ASSET_ROOT}"
