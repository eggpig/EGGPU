#!/usr/bin/env bash
set -euo pipefail

EVAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${EVAL_ROOT}"

GPU="${FINAL_GPU:-5}"
FINAL_BENCHMARK_REPEAT="${FINAL_BENCHMARK_REPEAT:-5}"
FINAL_MEMORY_REPEAT="${FINAL_MEMORY_REPEAT:-1}"
FINAL_SMOKE_TIMEOUT="${FINAL_SMOKE_TIMEOUT:-20}"
COMMON_PY="${COMMON_PY:-/home/dataset-assist-0/einwang/conda_cache/conda_env/EGGPU/bin/python}"
CUDA_ROOT="${EGGPU_CUDA_ROOT:-/home/dataset-assist-0/einwang/conda_cache/conda_env/sglang}"
EG_REPO="${EVAL_ROOT}/../Easy-Graph"
LEGACY_EVAL_ROOT="${EVAL_ROOT}/../../EG_Evaluation"
GUNROCK_PATHS="${EG_GUNROCK_BIN_PATHS:-${LEGACY_EVAL_ROOT}/gunrock_latest/build_cuda132_a100_migrated/bin:${LEGACY_EVAL_ROOT}/gunrock_legacy_master/build_lcc_result_cuda128_clean/bin}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
GATE_OUT="benchmarking/results/final_protocol_gate_gpu${GPU}_${RUN_TS}"
SMOKE_MAIN="${GATE_OUT}/main_split_smoke"
SMOKE_ABL="${GATE_OUT}/ablation_smoke"
MARKER_DECISION_FILE="${FINAL_MARKER_DECISION_FILE:-${GATE_OUT}/visibility_marker_enabled.txt}"
MARKER_ENABLED="TRUE"
MARKER_MB=256
mkdir -p "${SMOKE_MAIN}" "${SMOKE_ABL}"
mkdir -p "$(dirname "${MARKER_DECISION_FILE}")"

COMMON_ENV=(
  "CONDA_EXE=/opt/conda/bin/conda"
  "EGGPU_CHILD_PYTHON=${COMMON_PY}"
  "EGGPU_USE_CONDA_RUN=FALSE"
  "CUDA_VISIBLE_DEVICES=${GPU}"
  "EGGPU_MONITOR_GPU_INDEX=${GPU}"
  "EGGPU_CUDA_ROOT=${CUDA_ROOT}"
  "CUDA_PATH=${CUDA_ROOT}"
  "CONDA_PREFIX=${CUDA_ROOT}"
  "EASYGRAPH_ENABLE_GPU=TRUE"
  "EASYGRAPH_GPU_STRICT_ERRORS=TRUE"
  "EG_GUNROCK_BIN_PATHS=${GUNROCK_PATHS}"
  "EGGPU_GPU_VISIBILITY_MARKER=TRUE"
  "EGGPU_GPU_VISIBILITY_MARKER_MB=256"
  "EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB=AUTO"
  "EASYGRAPH_GPU_ADAPTIVE_POLICY=TRUE"
  "EASYGRAPH_GPU_COMPONENT_DENSE_RETURN=FALSE"
  "EASYGRAPH_GPU_SCC_ACTIVE_TRIM=TRUE"
  "EASYGRAPH_GPU_SCC_ACTIVE_TRIM_MAX_ITERS=16"
  "EASYGRAPH_GPU_SCC_DEGREE_PIVOT=TRUE"
  "EASYGRAPH_GPU_SCC_HOST_ENABLE=FALSE"
  "EASYGRAPH_GPU_KCORE_HOST_ENABLE=FALSE"
  "EASYGRAPH_GPU_SSSP_HOST_ENABLE=FALSE"
  "EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_MIN_AVG_DEGREE=10"
  "EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_MIN_MAX_DEGREE=AUTO"
  "EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_THREADS=1024"
  "EASYGRAPH_GPU_BC_WARP_SIZE=AUTO"
  "EASYGRAPH_GPU_CONSTRAINT_SMALLER_INTERSECTION=AUTO"
  "EGGPU_CLOSENESS_EXACT_MAX_NODES=1000000"
  "EGGPU_CLOSENESS_EXACT_MAX_WORK=50000000000"
  "PYTHONPATH=${EG_REPO}:${PYTHONPATH:-}"
  "LD_LIBRARY_PATH=${CUDA_ROOT}/lib:${CUDA_ROOT}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
)

echo "[gate] CPU/static contracts"
PYTHONPATH="${EG_REPO}:${PYTHONPATH:-}" "${COMMON_PY}" -m py_compile \
  benchmarking/library_baselines.py \
  benchmarking/run_full_baselines.py \
  benchmarking/run_split_full_baselines.py \
  benchmarking/run_eggpu_first_use.py \
  benchmarking/run_eggpu_natural_workflow.py \
  benchmarking/run_eggpu_ablations.py \
  benchmarking/summarize_ablation_system.py \
  benchmarking/audit_full_result.py
bash -n run_main_and_ablation.sh
bash -n run_complete_paper_experiments.sh
bash -n run_final_gpu5_gate_and_full.sh
PYTHONPATH="${EG_REPO}:${PYTHONPATH:-}" "${COMMON_PY}" -m unittest discover \
  -s benchmarking/tests -p 'test_*.py'

echo "[gate] waiting for GPU ${GPU} to become stably idle"
"${COMMON_PY}" benchmarking/wait_for_idle_gpu.py \
  --gpu "${GPU}" --stable-checks 3 --poll-seconds 30

echo "[gate] visibility-marker timing-neutrality A/B"
set +e
env -u CFLAGS -u CPPFLAGS -u CXXFLAGS \
    -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u CPATH -u LIBRARY_PATH \
    "${COMMON_ENV[@]}" \
    "${COMMON_PY}" benchmarking/verify_visibility_marker_neutrality.py \
      --gpu "${GPU}" \
      --trials 6 \
      --calls 200 \
      --warmup 20 \
      --marker-mb 256 \
      --max-ratio 1.05 \
      --out "${GATE_OUT}/visibility_marker_neutrality.json"
MARKER_CHECK_RC=$?
set -e
if [[ "${MARKER_CHECK_RC}" -ne 0 ]]; then
  MARKER_ENABLED="FALSE"
  MARKER_MB=0
  COMMON_ENV+=(
    "EGGPU_GPU_VISIBILITY_MARKER=FALSE"
    "EGGPU_GPU_VISIBILITY_MARKER_MB=0"
    "EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB=0"
  )
  echo "[gate] visibility marker was not timing-neutral; disabling it and continuing with measurement-critical checks"
else
  echo "[gate] visibility marker passed timing-neutrality A/B"
fi
printf '%s\n' "${MARKER_ENABLED}" > "${MARKER_DECISION_FILE}"

echo "[gate] EGGPU all-function correctness"
env -u CFLAGS -u CPPFLAGS -u CXXFLAGS \
    -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u CPATH -u LIBRARY_PATH \
    "${COMMON_ENV[@]}" \
    "${COMMON_PY}" benchmarking/run_eggpu_correctness_gate.py \
      --gpu "${GPU}" \
      --out-dir "${GATE_OUT}/eggpu_correctness" \
      --datasets ca-GrQc,p2p-Gnutella04 \
      --include-synthetic \
      --functions all \
      --timeout 100 \
      --pr-alpha 0.75 \
      --pr-tol 1e-6 \
      --pr-max-iter 200 \
      --easygraph-repo "${EG_REPO}" \
      --easygraph-warmup 2 \
      --sssp-sources 2 \
      --bc-sources 2

echo "[gate] split timing/memory main smoke"
echo "[gate] smoke-only per-function timeout=${FINAL_SMOKE_TIMEOUT}s; formal timeout remains configured separately"
env -u CFLAGS -u CPPFLAGS -u CXXFLAGS \
    -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u CPATH -u LIBRARY_PATH \
    "${COMMON_ENV[@]}" \
    "${COMMON_PY}" benchmarking/run_split_full_baselines.py \
      --gpu "${GPU}" \
      --out-dir "${SMOKE_MAIN}" \
      --repeat 1 \
      --memory-repeat 1 \
      --warmup 0 \
      --easygraph-warmup 2 \
      --library-timeout "${FINAL_SMOKE_TIMEOUT}" \
      --inter-run-cooldown 0.2 \
      --pr-alpha 0.75 \
      --pr-eps 1e-6 \
      --pr-max-iter 200 \
      --easygraph-repo "${EG_REPO}" \
      --sssp-sources 2 \
      --bc-sources 2 \
      --datasets ca-GrQc,p2p-Gnutella04 \
      --functions all
"${COMMON_PY}" benchmarking/audit_full_result.py "${SMOKE_MAIN}" --expected-repeat 1
"${COMMON_PY}" benchmarking/audit_backend_separation.py "${SMOKE_MAIN}"

echo "[gate] strengthened ablation smoke"
run_ablation_smoke() {
  local experiment="$1"
  local variant="$2"
  local functions="$3"
  local output="$4"
  shift 4
  env -u CFLAGS -u CPPFLAGS -u CXXFLAGS \
      -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u CPATH -u LIBRARY_PATH \
      "${COMMON_ENV[@]}" \
      EGGPU_MEASUREMENT_MODE=timing \
      "${COMMON_PY}" benchmarking/run_eggpu_ablations.py \
        --easygraph-repo "${EG_REPO}" \
        --experiment "${experiment}" \
        --variant "${variant}" \
        --edge-path datasets/undirected/ca-GrQc.txt \
        --dataset-name ca-GrQc \
        --graph-type undirected \
        --functions "${functions}" \
        --repeat 2 \
        --warmup 2 \
        --gpu "${GPU}" \
        --sssp-sources 2 \
        --bc-sources 2 \
        --closeness-sources 2 \
        --measurement-mode timing \
        --out "${output}" \
        "$@"
}

SMOKE_FUNCTIONS="PageRank,LCC,BFS,EffectiveSize"
run_ablation_smoke workflow full "${SMOKE_FUNCTIONS}" "${SMOKE_ABL}/workflow_canonical_full.csv" --workflow-order-id canonical
run_ablation_smoke workflow no_cpp_graph_cache "${SMOKE_FUNCTIONS}" "${SMOKE_ABL}/workflow_canonical_no_cpp.csv" --workflow-order-id canonical
run_ablation_smoke workflow no_graph_context "${SMOKE_FUNCTIONS}" "${SMOKE_ABL}/workflow_canonical_no_context.csv" --workflow-order-id canonical
run_ablation_smoke workflow full "EffectiveSize,BFS,LCC,PageRank" "${SMOKE_ABL}/workflow_reverse_full.csv" --workflow-order-id reverse
run_ablation_smoke return full "PageRank,BFS,EffectiveSize" "${SMOKE_ABL}/return.csv"
run_ablation_smoke layout full all "${SMOKE_ABL}/layout.csv" --layout-pr-iters 5

"${COMMON_PY}" - <<PY
from pathlib import Path
import pandas as pd
out = Path("${SMOKE_ABL}")
frames = [pd.read_csv(path) for path in sorted(out.glob("*.csv")) if path.name != "ablation_all.csv"]
pd.concat(frames, ignore_index=True, sort=False).to_csv(out / "ablation_all.csv", index=False)
PY
"${COMMON_PY}" benchmarking/summarize_ablation_system.py \
  --ablation-dir "${SMOKE_ABL}" --out-dir "${SMOKE_ABL}"

case "${FINAL_GATE_ONLY:-FALSE}" in
  1|TRUE|true|YES|yes|ON|on)
    echo "[gate] all correctness, main, and ablation smoke checks passed; FINAL_GATE_ONLY=TRUE, stopping before the formal run"
    exit 0
    ;;
esac

echo "[gate] smoke passed; waiting for GPU ${GPU} to be idle again"
"${COMMON_PY}" benchmarking/wait_for_idle_gpu.py \
  --gpu "${GPU}" --stable-checks 3 --poll-seconds 10

echo "[gate] starting final main experiment and ablation on GPU ${GPU}"
FINAL_LOG="benchmarking/results/main_then_ablation_$(date +%Y%m%d_%H%M%S).console.log"
COMMON_PY="${COMMON_PY}" \
EGGPU_CUDA_ROOT="${CUDA_ROOT}" \
EG_GUNROCK_BIN_PATHS="${GUNROCK_PATHS}" \
MAIN_GPU="${GPU}" ABL_GPU="${GPU}" RUN_PARALLEL=0 \
LIBRARY_TIMEOUT=100 ABLATION_TIMEOUT=100 \
BENCHMARK_REPEAT="${FINAL_BENCHMARK_REPEAT}" MEMORY_REPEAT="${FINAL_MEMORY_REPEAT}" EGGPU_WARMUP=2 \
RUN_PREFLIGHT=TRUE RUN_CLOSENESS_LARGE_SUPPLEMENT=TRUE \
EGGPU_GPU_VISIBILITY_MARKER="${MARKER_ENABLED}" EGGPU_GPU_VISIBILITY_MARKER_MB="${MARKER_MB}" \
bash run_main_and_ablation.sh |& tee "${FINAL_LOG}"
