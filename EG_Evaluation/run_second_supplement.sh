#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${COMMON_PY:-/home/dataset-assist-0/einwang/conda_cache/conda_env/EGGPU/bin/python}"
EASYGRAPH_REPO="${ROOT}/../Easy-Graph"
GPU="${SUPPLEMENT_GPU:-0}"
REPEAT="${PAPER_REPEAT:-5}"
MEMORY_REPEAT="${PAPER_MEMORY_REPEAT:-3}"
TIMEOUT="${PAPER_TIMEOUT:-100}"
SCALING_TIMEOUT="${SCALING_TIMEOUT:-1800}"
RUN_TS="${SUPPLEMENT_TS:-$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${SECOND_SUPPLEMENT_RESULT_ROOT:-${ROOT}/benchmarking/results/second_supplement_gpu${GPU}_${RUN_TS}}"
COLD_OUT="${RESULT_ROOT}/cold_start"
SCALING_OUT="${RESULT_ROOT}/scaling"
RMAT_OUT="${RESULT_ROOT}/controlled_rmat"
PAPER_OUT="${RESULT_ROOT}/paper_artifacts"
MAIN10_OUT="${RESULT_ROOT}/main_10_paper_artifacts"
STEADY_DIR="${STEADY_DIR:-${ROOT}/benchmarking/results/full_eval_gpu0_20260712_144427_repeat5_splitmem_exact_mst_strict_nxcg}"
ABLATION_DIR="${ABLATION_DIR:-${ROOT}/benchmarking/results/ablation_gpu0_20260712_144427_repeat5_strengthened}"
FIRST_USE_DIR="${FIRST_USE_DIR:-${ROOT}/benchmarking/results/eggpu_first_use_repaired_gpu0_20260715_121333_repeat5}"
BALANCED_DATASETS="ca-HepTh,LastFM,p2p-Gnutella04,ca-HepPh,email-Enron,ca-CondMat,soc-Epinions1,com-youtube,ER-100k,soc-Slashdot0811"
ORKUT_MANIFEST="${ROOT}/datasets/scaling/csr/com-Orkut/com-Orkut.json"
TWITTER_MANIFEST="${ROOT}/datasets/scaling/csr/GAP-twitter/GAP-twitter.json"
GATE_MANIFEST="${ROOT}/datasets/scaling/csr/com-youtube/com-youtube.json"
GATE_EDGE_PATH="${ROOT}/datasets/undirected/com-youtube.ungraph.txt"
RMAT_S10_MANIFEST="${ROOT}/datasets/scaling/csr/R-MAT-S10-EF16/R-MAT-S10-EF16.json"
RMAT_S20_MANIFEST="${ROOT}/datasets/scaling/csr/R-MAT-S20-EF16/R-MAT-S20-EF16.json"
RMAT_S22_MANIFEST="${ROOT}/datasets/scaling/csr/R-MAT-S22-EF16/R-MAT-S22-EF16.json"
RMAT_S24_MANIFEST="${ROOT}/datasets/scaling/csr/R-MAT-S24-EF16/R-MAT-S24-EF16.json"
RMAT_S26_MANIFEST="${ROOT}/datasets/scaling/csr/R-MAT-S26-EF16/R-MAT-S26-EF16.json"

for required in \
  "${GATE_MANIFEST}" "${ORKUT_MANIFEST}" "${TWITTER_MANIFEST}" \
  "${RMAT_S10_MANIFEST}" "${RMAT_S20_MANIFEST}" "${RMAT_S22_MANIFEST}" \
  "${RMAT_S24_MANIFEST}" "${RMAT_S26_MANIFEST}"; do
  if [[ ! -s "${required}" ]]; then
    echo "missing scaling manifest: ${required}" >&2
    echo "run: bash ${ROOT}/prepare_scaling_datasets.sh" >&2
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
export PYTHONPATH="${EASYGRAPH_REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${EGGPU_CUDA_ROOT}/lib:${EGGPU_CUDA_ROOT}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

mkdir -p "${RESULT_ROOT}"
cp "${ROOT}/benchmarking/second_supplement_protocol_20260715.json" "${RESULT_ROOT}/protocol.json"

echo "[1/8] validating real and controlled bulk CSR artifacts outside every measured window"
"${PYTHON_BIN}" "${ROOT}/benchmarking/validate_bulk_csr_manifests.py" \
  "${GATE_MANIFEST}" "${ORKUT_MANIFEST}" "${TWITTER_MANIFEST}" \
  "${RMAT_S10_MANIFEST}" "${RMAT_S20_MANIFEST}" "${RMAT_S22_MANIFEST}" \
  "${RMAT_S24_MANIFEST}" "${RMAT_S26_MANIFEST}" \
  |& tee "${RESULT_ROOT}/csr_validation.log"

echo "[2/8] correctness and positive-performance gates"
"${PYTHON_BIN}" "${ROOT}/benchmarking/run_bulk_csr_performance_gate.py" \
  --edge-path "${GATE_EDGE_PATH}" --bulk-manifest "${GATE_MANIFEST}" \
  --graph-type undirected --warmup 2 --repeat 5 \
  --minimum-geomean-speedup "${BULK_GATE_MIN_SPEEDUP:-1.02}" \
  --output "${RESULT_ROOT}/bulk_csr_performance_gate.json" \
  |& tee "${RESULT_ROOT}/bulk_csr_performance_gate.log"
"${PYTHON_BIN}" "${ROOT}/benchmarking/validate_rmat_reference_gate.py" \
  "${RMAT_S10_MANIFEST}" --source-count 4 \
  --output "${RESULT_ROOT}/rmat_reference_gate.json" \
  |& tee "${RESULT_ROOT}/rmat_reference_gate.log"

echo "[3/8] assemble audited cold-start comparison without repeating the ten-graph run"
"${PYTHON_BIN}" "${ROOT}/benchmarking/assemble_reused_cold_start.py" \
  --main-dir "${STEADY_DIR}" \
  --first-use-dir "${FIRST_USE_DIR}" \
  --output-dir "${COLD_OUT}" \
  |& tee "${RESULT_ROOT}/cold_start.log"

echo "[4/8] native-CSR real-graph scaling bridge: 6M/234M/1.468B stored entries"
set +e
"${PYTHON_BIN}" "${ROOT}/benchmarking/run_eggpu_scaling.py" \
  --manifests "${GATE_MANIFEST}" "${ORKUT_MANIFEST}" "${TWITTER_MANIFEST}" \
  --functions PageRank,WCC,BFS,KCore \
  --output-dir "${SCALING_OUT}" \
  --repeat "${REPEAT}" --warmup 2 --memory-repeat "${MEMORY_REPEAT}" \
  --source-count 4 --memory-poll-ms 2 --timeout "${SCALING_TIMEOUT}" \
  |& tee "${RESULT_ROOT}/scaling.log"
SCALING_RC="${PIPESTATUS[0]}"
set -e

echo "[5/8] controlled R-MAT scaling: S20/S22/S24/S26, edge factor 16"
set +e
"${PYTHON_BIN}" "${ROOT}/benchmarking/run_eggpu_scaling.py" \
  --manifests \
    "${RMAT_S20_MANIFEST}" "${RMAT_S22_MANIFEST}" \
    "${RMAT_S24_MANIFEST}" "${RMAT_S26_MANIFEST}" \
  --functions PageRank,WCC,BFS \
  --output-dir "${RMAT_OUT}" \
  --repeat "${REPEAT}" --warmup 2 --memory-repeat "${MEMORY_REPEAT}" \
  --source-count 4 --memory-poll-ms 2 --timeout "${SCALING_TIMEOUT}" \
  |& tee "${RESULT_ROOT}/controlled_rmat.log"
RMAT_RC="${PIPESTATUS[0]}"
set -e

echo "[6/8] paired statistics and paper artifacts"
"${PYTHON_BIN}" "${ROOT}/benchmarking/generate_final_paper_bundle.py" \
  --result-dir "${STEADY_DIR}" --ablation-dir "${ABLATION_DIR}" \
  --datasets "${BALANCED_DATASETS}" --out-dir "${MAIN10_OUT}"
"${PYTHON_BIN}" "${ROOT}/benchmarking/generate_second_supplement_artifacts.py" \
  --result-root "${RESULT_ROOT}" --steady-dir "${STEADY_DIR}" \
  --out-dir "${PAPER_OUT}"

echo "[7/8] completeness, correctness, statistics, and memory audit"
set +e
"${PYTHON_BIN}" "${ROOT}/benchmarking/audit_second_supplement.py" \
  --result-root "${RESULT_ROOT}" \
  |& tee "${RESULT_ROOT}/audit.log"
AUDIT_RC="${PIPESTATUS[0]}"
set -e

echo "[8/8] final protocol status"
echo "Done: ${RESULT_ROOT}"
if [[ "${SCALING_RC}" -ne 0 ]]; then
  echo "Scaling recorded one or more failures; rerun the same RESULT_ROOT to resume." >&2
  exit "${SCALING_RC}"
fi
if [[ "${RMAT_RC}" -ne 0 ]]; then
  echo "Controlled R-MAT scaling recorded failures; rerun the same RESULT_ROOT to resume." >&2
  exit "${RMAT_RC}"
fi
if [[ "${AUDIT_RC}" -ne 0 ]]; then
  echo "Second-supplement audit failed; all raw results remain available." >&2
  exit "${AUDIT_RC}"
fi
