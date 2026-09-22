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
RESULT_ROOT="${FINAL13_FOLLOWUP_ROOT:-${ROOT}/benchmarking/results/final13_followup_gpu${GPU}_${RUN_TS}}"
RMAT_ROOT="${ROOT}/datasets/scaling/csr"
RMAT_MANIFESTS=(
  "${RMAT_ROOT}/R-MAT-S20-EF16/R-MAT-S20-EF16.json"
  "${RMAT_ROOT}/R-MAT-S22-EF16/R-MAT-S22-EF16.json"
  "${RMAT_ROOT}/R-MAT-S24-EF16/R-MAT-S24-EF16.json"
  "${RMAT_ROOT}/R-MAT-S26-EF16/R-MAT-S26-EF16.json"
)

for required in "${RMAT_MANIFESTS[@]}"; do
  if [[ ! -s "${required}" ]]; then
    echo "missing required R-MAT CSR manifest: ${required}" >&2
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
  "protocol": "final_13_missing_followup_v1",
  "gpu": ${GPU},
  "timing_repeat": ${REPEAT},
  "memory_repeat": ${MEMORY_REPEAT},
  "eggpu_warmup": 2,
  "competitor_warmup": 0,
  "per_call_timeout_seconds": ${TIMEOUT},
  "rmat_functions": ["PageRank", "WCC", "BFS", "SSSP"],
  "cumulative_workflow": ["WCC", "PageRank", "BFS"],
  "cumulative_scope": "analysis calls only; host graph construction is reported separately",
  "status_policy": "all attempted cells retain ok or a structured failure reason"
}
EOF

echo "[followup 1/5] deterministic R-MAT weights for weighted SSSP"
"${PYTHON_BIN}" "${ROOT}/benchmarking/prepare_bulk_csr_weights.py" \
  "${RMAT_MANIFESTS[@]}" \
  |& tee "${RESULT_ROOT}/prepare_rmat_weights.log"
WEIGHTS_RC="${PIPESTATUS[0]}"

echo "[followup 2/5] EGGPU R-MAT S20/S22/S24/S26 four-function matrix"
"${PYTHON_BIN}" "${ROOT}/benchmarking/run_eggpu_scaling.py" \
  --manifests "${RMAT_MANIFESTS[@]}" \
  --functions PageRank,WCC,BFS,SSSP \
  --output-dir "${RESULT_ROOT}/eggpu_rmat_four_functions" \
  --repeat "${REPEAT}" --warmup 2 --memory-repeat "${MEMORY_REPEAT}" \
  --source-count 8 --memory-poll-ms 2 \
  --timeout "${TIMEOUT}" --load-timeout "${LOAD_TIMEOUT}" \
  --hard-call-timeout --continue-on-failure \
  |& tee "${RESULT_ROOT}/eggpu_rmat_sssp.log"
EGGPU_RMAT_RC="${PIPESTATUS[0]}"

echo "[followup 3/5] strict nx-cugraph R-MAT four-function comparison"
"${PYTHON_BIN}" "${ROOT}/benchmarking/run_nxcugraph_large_matrix.py" \
  --manifests "${RMAT_MANIFESTS[@]}" \
  --functions PageRank WCC BFS SSSP \
  --gpu "${GPU}" --repeat "${REPEAT}" --memory-repeat "${MEMORY_REPEAT}" \
  --source-count 8 --timeout "${TIMEOUT}" --load-timeout "${LOAD_TIMEOUT}" \
  --out-dir "${RESULT_ROOT}/nxcugraph_rmat_four_functions" \
  |& tee "${RESULT_ROOT}/nxcugraph_rmat_four_functions.log"
NXCG_RMAT_RC="${PIPESTATUS[0]}"

echo "[followup 4/5] cumulative WCC -> PageRank -> BFS workflow"
"${PYTHON_BIN}" "${ROOT}/benchmarking/run_cumulative_workflow_comparison.py" \
  --datasets \
    ca-HepTh LastFM p2p-Gnutella04 ca-HepPh email-Enron \
    ca-CondMat soc-Epinions1 com-youtube ER-100k soc-Slashdot0811 \
  --baselines EGGPU EGGPU-isolated igraph nx-cugraph \
  --repeat "${REPEAT}" --sources 8 --timeout "${TIMEOUT}" \
  --load-timeout "${LOAD_TIMEOUT}" --gpu "${GPU}" \
  --output-dir "${RESULT_ROOT}/cumulative_workflow" \
  |& tee "${RESULT_ROOT}/cumulative_workflow.log"
CUMULATIVE_RC="${PIPESTATUS[0]}"

echo "[followup 5/5] sparse-native igraph R-MAT four-function comparison"
# Keep the CPU comparison after every GPU timing and memory pass so host graph
# construction and CPU contention cannot perturb the authoritative GPU data.
"${PYTHON_BIN}" "${ROOT}/benchmarking/run_cpu_large_matrix.py" \
  --manifests "${RMAT_MANIFESTS[@]}" \
  --baselines igraph \
  --functions PageRank WCC BFS SSSP \
  --repeat "${REPEAT}" --memory-repeat "${MEMORY_REPEAT}" \
  --source-count 8 --timeout "${TIMEOUT}" --load-timeout "${LOAD_TIMEOUT}" \
  --worker-memory-limit-gb 128 --gpu "${GPU}" \
  --out-dir "${RESULT_ROOT}/igraph_rmat_four_functions" \
  |& tee "${RESULT_ROOT}/igraph_rmat_four_functions.log"
IGRAPH_RMAT_RC="${PIPESTATUS[0]}"

cat >"${RESULT_ROOT}/phase_status.json" <<EOF
{
  "weights_exit_code": ${WEIGHTS_RC},
  "eggpu_rmat_four_functions_exit_code": ${EGGPU_RMAT_RC},
  "nxcugraph_rmat_exit_code": ${NXCG_RMAT_RC},
  "cumulative_workflow_exit_code": ${CUMULATIVE_RC},
  "igraph_rmat_exit_code": ${IGRAPH_RMAT_RC},
  "result_root": "${RESULT_ROOT}"
}
EOF

echo "Final-13 follow-up experiment finished: ${RESULT_ROOT}"
echo "Structured per-cell failures remain in the corresponding CSV/JSON ledgers."
exit 0
