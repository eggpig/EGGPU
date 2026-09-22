#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${COMMON_PY:-/home/dataset-assist-0/einwang/conda_cache/conda_env/EGGPU/bin/python}"
CUDA_ROOT="${EGGPU_CUDA_ROOT:-/home/dataset-assist-0/einwang/conda_cache/conda_env/sglang}"
GPU="${RECOVERY_GPU:-7}"
REPEAT="${PAPER_REPEAT:-5}"
MEMORY_REPEAT="${PAPER_MEMORY_REPEAT:-3}"
TIMEOUT="${PAPER_TIMEOUT:-100}"
RUN_TS="${RECOVERY_TS:-$(date +%Y%m%d_%H%M%S)}"
OUT="${GPU_BASELINE_RECOVERY_ROOT:-${ROOT}/benchmarking/results/gpu_baseline_gap_recovery_gpu${GPU}_${RUN_TS}}"

MAINTAINED_BIN="/home/dataset-assist-0/einwang/workspace/haorandu/EG_Evaluation/gunrock_latest/build_cuda132_a100_migrated/bin"
LEGACY_LCC_BIN="/home/dataset-assist-0/einwang/workspace/haorandu/EG_Evaluation/gunrock_legacy_master/build_lcc_result_cuda128_clean/bin"
CORE="${ROOT}/benchmarking/results/final13_missing_gpu7_20260716_212844"
GUNROCK_INPUTS="${CORE}/gunrock_inputs"
EGGPU_LARGE="${CORE}/eggpu_large_matrix"
REFERENCE_STANDARD="${ROOT}/benchmarking/results/full_eval_gpu0_20260712_144427_repeat5_splitmem_exact_mst_strict_nxcg"
ORKUT="${ROOT}/datasets/scaling/csr/com-Orkut/com-Orkut.json"
TWITTER="${ROOT}/datasets/scaling/csr/GAP-twitter/GAP-twitter.json"
STANDARD_DATASETS="ca-HepTh,LastFM,p2p-Gnutella04,ca-HepPh,email-Enron,ca-CondMat,soc-Epinions1,soc-Slashdot0811,ER-100k,web-NotreDame,com-youtube"
RECOVERY_FUNCTIONS="PageRank,LCC,BFS,Dijkstra,SSSP,KCore"
QUALIFICATION_EXCLUSIONS="BellmanFord:no_aligned_implementation;BC-16:full_vector_semantic_mismatch"

for required in \
  "${PYTHON_BIN}" \
  "${MAINTAINED_BIN}/pr" \
  "${MAINTAINED_BIN}/sssp" \
  "${MAINTAINED_BIN}/bfs" \
  "${MAINTAINED_BIN}/kcore" \
  "${LEGACY_LCC_BIN}/lcc" \
  "${GUNROCK_INPUTS}/com-Orkut.aligned-weighted.mtx" \
  "${GUNROCK_INPUTS}/GAP-twitter.aligned-weighted.mtx" \
  "${ORKUT}" "${TWITTER}"; do
  if [[ ! -e "${required}" ]]; then
    echo "required recovery artifact is missing: ${required}" >&2
    exit 2
  fi
done

if ! command -v nvidia-smi >/dev/null 2>&1 || \
   ! nvidia-smi -i "${GPU}" --query-gpu=index --format=csv,noheader >/dev/null 2>&1; then
  echo "GPU ${GPU} is unavailable on this host; no benchmark was started." >&2
  exit 3
fi

mkdir -p "${OUT}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export EGGPU_MONITOR_GPU_INDEX="${GPU}"
export EGGPU_CUDA_ROOT="${CUDA_ROOT}"
export EGGPU_CHILD_PYTHON="${PYTHON_BIN}"
export COMMON_PY="${PYTHON_BIN}"
export EG_GUNROCK_BIN_PATHS="${MAINTAINED_BIN}:${LEGACY_LCC_BIN}"
export PYTHONPATH="${ROOT}/../Easy-Graph:${ROOT}/benchmarking${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${CUDA_ROOT}/lib:${CUDA_ROOT}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cat >"${OUT}/protocol.json" <<EOF
{
  "scope": "targeted_gpu_baseline_gap_recovery",
  "gpu": ${GPU},
  "timing_repeat": ${REPEAT},
  "memory_repeat": ${MEMORY_REPEAT},
  "timeout_seconds": ${TIMEOUT},
  "standard_datasets": "${STANDARD_DATASETS}",
  "functions": "${RECOVERY_FUNCTIONS}",
  "qualification_exclusions": "${QUALIFICATION_EXCLUSIONS}",
  "baseline": "Gunrock",
  "timing_memory_isolation": true,
  "original_results_overwritten": false,
  "semantic_policy": "PageRank parameters aligned; LCC full-vector validation; BFS/SSSP/Dijkstra/KCore CPU validation and result export are isolated from timing; Dijkstra is the maintained single-source SSSP implementation alias; BellmanFord and BC-16 retain qualification evidence but are not repeated because they cannot produce aligned publishable results"
}
EOF

echo "[1/4] static semantic-adapter regression tests"
"${PYTHON_BIN}" -m unittest \
  benchmarking.tests.test_gunrock_semantic_adapters \
  benchmarking.tests.test_gunrock_large_semantics \
  |& tee "${OUT}/static_tests.log"

echo "[2/4] 11 cross-library datasets, split timing and memory passes"
"${PYTHON_BIN}" "${ROOT}/benchmarking/run_split_full_baselines.py" \
  --gpu "${GPU}" \
  --out-dir "${OUT}/standard" \
  --repeat "${REPEAT}" --memory-repeat "${MEMORY_REPEAT}" \
  --warmup 0 --easygraph-warmup 0 \
  --library-timeout "${TIMEOUT}" --inter-run-cooldown 1.0 \
  --pr-alpha 0.75 --pr-eps 1e-6 --pr-max-iter 200 \
  --easygraph-repo "${ROOT}/../Easy-Graph" \
  --sssp-sources 8 --bc-sources 16 \
  --datasets "${STANDARD_DATASETS}" \
  --functions "${RECOVERY_FUNCTIONS}" \
  --baselines Gunrock \
  --eggpu-execution-protocol steady-state \
  |& tee "${OUT}/standard.log"

echo "[3/4] com-Orkut and GAP-twitter with the same semantic adapters"
"${PYTHON_BIN}" "${ROOT}/benchmarking/run_gunrock_large_matrix.py" \
  --manifests "${ORKUT}" "${TWITTER}" \
  --functions PageRank LCC BFS Dijkstra SSSP KCore \
  --input-dir "${GUNROCK_INPUTS}" \
  --eggpu-result-dir "${EGGPU_LARGE}" \
  --gpu "${GPU}" --repeat "${REPEAT}" --memory-repeat "${MEMORY_REPEAT}" \
  --source-count 8 --bc-source-count 16 \
  --pagerank-alpha 0.75 --pagerank-tolerance 1e-6 \
  --timeout "${TIMEOUT}" --no-resume \
  --out-dir "${OUT}/large" \
  |& tee "${OUT}/large.log"

echo "[4/4] completeness and publication-eligibility audit"
set +e
"${PYTHON_BIN}" "${ROOT}/benchmarking/audit_gpu_baseline_gap_recovery.py" \
  --standard-result "${OUT}/standard" \
  --large-result "${OUT}/large" \
  --output-dir "${OUT}/audit" \
  --datasets "${STANDARD_DATASETS}" \
  --functions "${RECOVERY_FUNCTIONS}" \
  --reference-result "${REFERENCE_STANDARD}" \
  --repeat "${REPEAT}" \
  |& tee "${OUT}/audit.log"
AUDIT_RC="${PIPESTATUS[0]}"
set -e

cat >"${OUT}/completion.json" <<EOF
{
  "status": "completed",
  "audit_exit_code": ${AUDIT_RC},
  "standard_result": "${OUT}/standard",
  "large_result": "${OUT}/large",
  "audit_result": "${OUT}/audit",
  "note": "Nonzero audit means a structural/sample contract issue. Timeouts and validated semantic mismatches remain recorded outcomes."
}
EOF

echo "GPU baseline recovery completed: ${OUT}"
echo "Audit: ${OUT}/audit/GPU_BASELINE_RECOVERY_AUDIT.md"
