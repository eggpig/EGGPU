#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${COMMON_PY:-/home/dataset-assist-0/einwang/conda_cache/conda_env/EGGPU/bin/python}"
EASYGRAPH_REPO="${ROOT}/../Easy-Graph"
GPU="${FINAL13_GPU:-7}"
REPEAT="${PAPER_REPEAT:-5}"
MEMORY_REPEAT="${PAPER_MEMORY_REPEAT:-3}"
TIMEOUT="${PAPER_TIMEOUT:-100}"
LOAD_TIMEOUT="${PAPER_LOAD_TIMEOUT:-900}"
RUN_TS="${FINAL13_TS:-$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${FINAL13_RESULT_ROOT:-${ROOT}/benchmarking/results/final13_missing_gpu${GPU}_${RUN_TS}}"

ORKUT="${ROOT}/datasets/scaling/csr/com-Orkut/com-Orkut.json"
TWITTER="${ROOT}/datasets/scaling/csr/GAP-twitter/GAP-twitter.json"
RMAT_ROOT="${ROOT}/datasets/scaling/csr"
RMAT_MANIFESTS=(
  "${RMAT_ROOT}/R-MAT-S20-EF16/R-MAT-S20-EF16.json"
  "${RMAT_ROOT}/R-MAT-S22-EF16/R-MAT-S22-EF16.json"
  "${RMAT_ROOT}/R-MAT-S24-EF16/R-MAT-S24-EF16.json"
  "${RMAT_ROOT}/R-MAT-S26-EF16/R-MAT-S26-EF16.json"
)

for required in "${ORKUT}" "${TWITTER}" "${RMAT_MANIFESTS[@]}"; do
  if [[ ! -s "${required}" ]]; then
    echo "missing required CSR manifest: ${required}" >&2
    exit 2
  fi
done

export CUDA_VISIBLE_DEVICES="${GPU}"
export EGGPU_MONITOR_GPU_INDEX="${GPU}"
export EASYGRAPH_ENABLE_GPU=TRUE
export EASYGRAPH_GPU_STRICT_ERRORS=TRUE
export EASYGRAPH_GPU_RESULT_CACHE=FALSE
export EASYGRAPH_GPU_ADAPTIVE_HOST=FALSE
export EGGPU_ALLOW_CUDA_SYNC=TRUE
export EGGPU_GPU_VISIBILITY_MARKER=FALSE
export EGGPU_CUDA_ROOT="${EGGPU_CUDA_ROOT:-/home/dataset-assist-0/einwang/conda_cache/conda_env/sglang}"
export PYTHONPATH="${EASYGRAPH_REPO}:${ROOT}/benchmarking${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${EGGPU_CUDA_ROOT}/lib:${EGGPU_CUDA_ROOT}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

mkdir -p "${RESULT_ROOT}"
cat >"${RESULT_ROOT}/protocol.json" <<EOF
{
  "protocol": "final_13_missing_matrix_v1",
  "gpu": ${GPU},
  "timing_repeat": ${REPEAT},
  "memory_repeat": ${MEMORY_REPEAT},
  "eggpu_warmup": 2,
  "competitor_warmup": 0,
  "per_call_timeout_seconds": ${TIMEOUT},
  "timing_processes": "fresh_process_per_sample",
  "weighted_edge_semantics": "1 + (src * dst) % num_nodes",
  "large_closeness_semantics": "exact on 16 deterministic evenly-spaced sources",
  "path_source_count": 8,
  "bc_source_count": 16,
  "status_policy": "every dataset/function/system cell is ok, timeout, oom, unsupported_api, semantic_mismatch, representation_limit, resource_limit, or execution_error"
}
EOF

echo "[final13 1/4] deterministic weighted CSR siblings"
"${PYTHON_BIN}" "${ROOT}/benchmarking/prepare_bulk_csr_weights.py" \
  "${ORKUT}" "${TWITTER}" \
  |& tee "${RESULT_ROOT}/prepare_weights.log"
WEIGHTS_RC="${PIPESTATUS[0]}"

echo "[final13 2/4] EGGPU: two real scale graphs x all 16 functions"
"${PYTHON_BIN}" "${ROOT}/benchmarking/run_eggpu_scaling.py" \
  --manifests "${ORKUT}" "${TWITTER}" \
  --functions all \
  --output-dir "${RESULT_ROOT}/eggpu_large_matrix" \
  --repeat "${REPEAT}" --warmup 2 --memory-repeat "${MEMORY_REPEAT}" \
  --source-count 8 --bc-source-count 16 --closeness-source-count 16 \
  --memory-poll-ms 2 --timeout "${TIMEOUT}" --load-timeout "${LOAD_TIMEOUT}" \
  --hard-call-timeout --continue-on-failure \
  |& tee "${RESULT_ROOT}/eggpu_large_matrix.log"
EGGPU_RC="${PIPESTATUS[0]}"

echo "[final13 3/4] R-MAT S20/S22/S24/S26: missing fourth function KCore"
"${PYTHON_BIN}" "${ROOT}/benchmarking/run_eggpu_scaling.py" \
  --manifests "${RMAT_MANIFESTS[@]}" \
  --functions KCore \
  --output-dir "${RESULT_ROOT}/rmat_kcore" \
  --repeat "${REPEAT}" --warmup 2 --memory-repeat "${MEMORY_REPEAT}" \
  --source-count 8 --memory-poll-ms 2 \
  --timeout "${TIMEOUT}" --load-timeout "${LOAD_TIMEOUT}" \
  --hard-call-timeout --continue-on-failure \
  |& tee "${RESULT_ROOT}/rmat_kcore.log"
RMAT_RC="${PIPESTATUS[0]}"

echo "[final13 4/4] strict nx-cugraph: two real scale graphs x all 16 functions"
"${PYTHON_BIN}" "${ROOT}/benchmarking/run_nxcugraph_large_matrix.py" \
  --manifests "${ORKUT}" "${TWITTER}" \
  --functions \
    PageRank MST LCC WCC SCC BFS Dijkstra BellmanFord SSSP KCore \
    BC Closeness EffectiveSize Efficiency Constraint Hierarchy \
  --gpu "${GPU}" --repeat "${REPEAT}" --memory-repeat "${MEMORY_REPEAT}" \
  --source-count 8 --timeout "${TIMEOUT}" --load-timeout "${LOAD_TIMEOUT}" \
  --out-dir "${RESULT_ROOT}/nxcugraph_large_matrix" \
  |& tee "${RESULT_ROOT}/nxcugraph_large_matrix.log"
NXCG_RC="${PIPESTATUS[0]}"

cat >"${RESULT_ROOT}/phase_status.json" <<EOF
{
  "weights_exit_code": ${WEIGHTS_RC},
  "eggpu_exit_code": ${EGGPU_RC},
  "rmat_kcore_exit_code": ${RMAT_RC},
  "nxcugraph_exit_code": ${NXCG_RC},
  "result_root": "${RESULT_ROOT}"
}
EOF

echo "Final-13 missing core experiment finished: ${RESULT_ROOT}"
echo "Individual function failures are expected to remain as structured result rows."
exit 0
