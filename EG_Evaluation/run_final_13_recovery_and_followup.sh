#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${COMMON_PY:-/home/dataset-assist-0/einwang/conda_cache/conda_env/EGGPU/bin/python}"
CUDA_ROOT="${EGGPU_CUDA_ROOT:-/home/dataset-assist-0/einwang/conda_cache/conda_env/sglang}"
GPU="${FINAL13_GPU:-7}"
REPEAT="${PAPER_REPEAT:-5}"
MEMORY_REPEAT="${PAPER_MEMORY_REPEAT:-3}"
TIMEOUT="${PAPER_TIMEOUT:-100}"
LOAD_TIMEOUT="${PAPER_LOAD_TIMEOUT:-900}"
CORE_ROOT="${FINAL13_RESULT_ROOT:-${ROOT}/benchmarking/results/final13_missing_gpu7_20260716_212844}"
FOLLOWUP_ROOT="${FINAL13_FOLLOWUP_ROOT:-${ROOT}/benchmarking/results/final13_followup_gpu${GPU}_$(date +%Y%m%d_%H%M%S)}"
EASYGRAPH_OUTPUT="${CORE_ROOT}/eggpu_large_matrix"
ORKUT="${ROOT}/datasets/scaling/csr/com-Orkut/com-Orkut.json"
TWITTER="${ROOT}/datasets/scaling/csr/GAP-twitter/GAP-twitter.json"

if [[ ! -s "${CORE_ROOT}/phase_status.json" ]]; then
  echo "Core experiment has not completed: ${CORE_ROOT}/phase_status.json is absent." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
export EGGPU_MONITOR_GPU_INDEX="${GPU}"
export EASYGRAPH_ENABLE_GPU=TRUE
export EASYGRAPH_GPU_STRICT_ERRORS=TRUE
export EASYGRAPH_GPU_RESULT_CACHE=FALSE
export EASYGRAPH_GPU_ADAPTIVE_HOST=FALSE
export EGGPU_ALLOW_CUDA_SYNC=TRUE
export EGGPU_GPU_VISIBILITY_MARKER=FALSE
export EGGPU_CUDA_ROOT="${CUDA_ROOT}"
export PYTHONPATH="${ROOT}/../Easy-Graph:${ROOT}/benchmarking${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${CUDA_ROOT}/lib:${CUDA_ROOT}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo "[recovery 1/7] incrementally rebuild the changed EGGPU extension"
PYTHON_BIN="${PYTHON_BIN}" \
EGGPU_CUDA_ROOT="${CUDA_ROOT}" \
EGGPU_CUDA_ARCHITECTURES=80 \
bash "${ROOT}/../scripts/build_eggpu.sh" \
  |& tee "${CORE_ROOT}/recovery_build.log"

echo "[recovery 2/7] tiny GPU contract gate for bulk BC, MST, and SCC"
"${PYTHON_BIN}" "${ROOT}/benchmarking/validate_bulk_mst_bc_scc_regression.py" \
  --output "${CORE_ROOT}/recovery_bulk_contract_gate.json" \
  |& tee "${CORE_ROOT}/recovery_bulk_contract_gate.log"

echo "[recovery 3/7] resume the complete EGGPU 2 x 16 matrix"
# Successful fresh-process samples are reused.  Only failed/missing samples are
# recomputed; the all-function invocation then regenerates a complete summary.
# The original large-matrix process loaded the pre-fix PID-namespace monitor.
# Preserve its valid timing samples, but force every publishable memory sample
# through the namespace-aware coordinator monitor loaded by this new process.
find "${EASYGRAPH_OUTPUT}/raw" -maxdepth 1 -type f -name '*_memory_*.json' -delete
"${PYTHON_BIN}" "${ROOT}/benchmarking/run_eggpu_scaling.py" \
  --manifests "${ORKUT}" "${TWITTER}" \
  --functions all \
  --output-dir "${EASYGRAPH_OUTPUT}" \
  --repeat "${REPEAT}" --warmup 2 --memory-repeat "${MEMORY_REPEAT}" \
  --source-count 8 --bc-source-count 16 --closeness-source-count 16 \
  --memory-poll-ms 2 --timeout "${TIMEOUT}" --load-timeout "${LOAD_TIMEOUT}" \
  --retry-cells com-Orkut:MST,com-Orkut:SCC,com-Orkut:BC,GAP-twitter:BC \
  --hard-call-timeout --continue-on-failure \
  |& tee "${CORE_ROOT}/eggpu_large_matrix_recovery.log"

echo "[recovery 4/7] four-function R-MAT and cumulative workflow supplements"
FINAL13_GPU="${GPU}" \
PAPER_REPEAT="${REPEAT}" \
PAPER_MEMORY_REPEAT="${MEMORY_REPEAT}" \
PAPER_TIMEOUT="${TIMEOUT}" \
PAPER_LOAD_TIMEOUT="${LOAD_TIMEOUT}" \
FINAL13_FOLLOWUP_ROOT="${FOLLOWUP_ROOT}" \
COMMON_PY="${PYTHON_BIN}" \
EGGPU_CUDA_ROOT="${CUDA_ROOT}" \
bash "${ROOT}/run_final_13_missing_followup.sh" \
  |& tee "${FOLLOWUP_ROOT}.launcher.log"

echo "[recovery 5/7] prepare normalized weighted MatrixMarket inputs for Gunrock"
GUNROCK_INPUT_DIR="${CORE_ROOT}/gunrock_inputs"
"${PYTHON_BIN}" "${ROOT}/benchmarking/prepare_gunrock_bulk_mtx.py" \
  "${ORKUT}" "${TWITTER}" \
  --output-dir "${GUNROCK_INPUT_DIR}" \
  |& tee "${CORE_ROOT}/prepare_gunrock_bulk_mtx.log"

echo "[recovery 6/7] pinned native Gunrock executables on both scale anchors"
"${PYTHON_BIN}" "${ROOT}/benchmarking/run_gunrock_large_matrix.py" \
  --manifests "${ORKUT}" "${TWITTER}" \
  --functions \
    PageRank MST LCC WCC SCC BFS Dijkstra BellmanFord SSSP KCore \
    BC Closeness EffectiveSize Efficiency Constraint Hierarchy \
  --input-dir "${GUNROCK_INPUT_DIR}" \
  --eggpu-result-dir "${EASYGRAPH_OUTPUT}" \
  --gpu "${GPU}" --repeat "${REPEAT}" --memory-repeat "${MEMORY_REPEAT}" \
  --source-count 8 --bc-source-count 16 --timeout "${TIMEOUT}" \
  --out-dir "${CORE_ROOT}/gunrock_large_matrix" \
  |& tee "${CORE_ROOT}/gunrock_large_matrix.log"

echo "[recovery 7/7] sparse-native qualification of CPU baselines on both scale anchors"
# This phase starts only after every GPU timing/memory experiment above has
# completed, so large host-graph construction cannot perturb GPU measurements.
"${PYTHON_BIN}" "${ROOT}/benchmarking/run_cpu_large_matrix.py" \
  --manifests "${ORKUT}" "${TWITTER}" \
  --baselines igraph networkx easygraph-cpu easygraph-cpp \
  --functions \
    PageRank MST LCC WCC SCC BFS Dijkstra BellmanFord SSSP KCore \
    BC Closeness EffectiveSize Efficiency Constraint Hierarchy \
  --repeat "${REPEAT}" --memory-repeat "${MEMORY_REPEAT}" \
  --source-count 8 --bc-source-count 16 --closeness-source-count 16 \
  --timeout "${TIMEOUT}" --load-timeout "${LOAD_TIMEOUT}" \
  --worker-memory-limit-gb 128 --gpu "${GPU}" \
  --out-dir "${CORE_ROOT}/cpu_large_matrix" \
  |& tee "${CORE_ROOT}/cpu_large_matrix.log"

cat >"${CORE_ROOT}/recovery_and_followup_status.json" <<EOF
{
  "status": "completed",
  "gpu": ${GPU},
  "core_result": "${CORE_ROOT}",
  "followup_result": "${FOLLOWUP_ROOT}",
  "cpu_large_matrix": "${CORE_ROOT}/cpu_large_matrix",
  "gunrock_large_matrix": "${CORE_ROOT}/gunrock_large_matrix",
  "gunrock_inputs": "${GUNROCK_INPUT_DIR}",
  "bulk_contract_gate": "${CORE_ROOT}/recovery_bulk_contract_gate.json"
}
EOF

echo "Recovery and follow-up complete."
echo "Core: ${CORE_ROOT}"
echo "Follow-up: ${FOLLOWUP_ROOT}"
