#!/usr/bin/env bash
set -uo pipefail

RUN_TS="$(date +%Y%m%d_%H%M%S)"
EVAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${EVAL_ROOT}/.." && pwd)"
MAIN_GPU="${MAIN_GPU:-0}"
ABL_GPU="${ABL_GPU:-1}"
LIBRARY_TIMEOUT="${LIBRARY_TIMEOUT:-100}"
ABLATION_TIMEOUT="${ABLATION_TIMEOUT:-300}"
RUN_PARALLEL="${RUN_PARALLEL:-${EGGPU_RUN_PARALLEL:-0}}"
RUN_ABLATION_ON_MAIN_FAILURE="${RUN_ABLATION_ON_MAIN_FAILURE:-TRUE}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-TRUE}"
RUN_CLOSENESS_LARGE_SUPPLEMENT="${RUN_CLOSENESS_LARGE_SUPPLEMENT:-TRUE}"
CLOSENESS_LARGE_SOURCES="${CLOSENESS_LARGE_SOURCES:-16}"
CLOSENESS_LARGE_TIMEOUT="${CLOSENESS_LARGE_TIMEOUT:-1800}"
EGGPU_CLOSENESS_EXACT_MAX_WORK="${EGGPU_CLOSENESS_EXACT_MAX_WORK:-50000000000}"
EGGPU_GPU_VISIBILITY_MARKER="${EGGPU_GPU_VISIBILITY_MARKER:-FALSE}"
# Default comparable runs keep the marker off.  When explicitly enabled, the
# long-lived runner reserves a small visible allocation and auto-measures the
# whole-device memory increment so the busy-GPU guard and memory tables subtract
# only that fixed marker footprint.
case "${EGGPU_GPU_VISIBILITY_MARKER}" in
  1|true|TRUE|yes|YES|on|ON)
    EGGPU_GPU_VISIBILITY_MARKER=TRUE
    EGGPU_GPU_VISIBILITY_MARKER_MB="${EGGPU_GPU_VISIBILITY_MARKER_MB:-256}"
    EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB="${EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB:-AUTO}"
    ;;
  *)
    EGGPU_GPU_VISIBILITY_MARKER=FALSE
    EGGPU_GPU_VISIBILITY_MARKER_MB=0
    EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB=0
    ;;
esac

MAIN_ID="full_eval_gpu${MAIN_GPU}_${RUN_TS}_selfloop_filtered_scc_trim16_kcore_threshold_constraint_auto"
ABL_ID="ablation_gpu${ABL_GPU}_${RUN_TS}_selfloop_filtered_scc_trim16_kcore_threshold_constraint_auto"
MAIN_OUT="benchmarking/results/${MAIN_ID}"
ABL_OUT="benchmarking/results/${ABL_ID}"
mkdir -p "${MAIN_OUT}" "${ABL_OUT}"

COMMON_PY="${COMMON_PY:-$(command -v python)}"
EG_REPO="${EG_REPO:-${WORKSPACE_ROOT}/Easy-Graph}"
CUDA_ROOT="${EGGPU_CUDA_ROOT:-${CUDA_PATH:-${CUDA_HOME:-${CONDA_PREFIX:-}}}}"
if [[ -z "${EGGPU_CUDA_ROOT:-}" && -n "${CONDA_PREFIX:-}" && "$(basename "${CONDA_PREFIX}")" == "EGGPU" ]]; then
  SIBLING_CUDA_ROOT="$(dirname "${CONDA_PREFIX}")/sglang"
  if [[ -x "${SIBLING_CUDA_ROOT}/bin/nvcc" ]]; then
    CUDA_ROOT="${SIBLING_CUDA_ROOT}"
    echo "EGGPU_CUDA_ROOT not set; using sibling local CUDA root: ${CUDA_ROOT}"
  fi
fi
CONDA_EXE_PATH="${CONDA_EXE:-${_CONDA_EXE:-$(command -v conda || true)}}"
LD_PATH="${LD_LIBRARY_PATH:-}"
if [[ -n "${CUDA_ROOT}" ]]; then
  for candidate in "${CUDA_ROOT}/lib" "${CUDA_ROOT}/targets/x86_64-linux/lib"; do
    if [[ -d "${candidate}" ]]; then
      if [[ -n "${LD_PATH}" ]]; then
        LD_PATH="${candidate}:${LD_PATH}"
      else
        LD_PATH="${candidate}"
      fi
    fi
  done
fi

gpu_idle_check_enabled() {
  case "${EGGPU_ALLOW_BUSY_GPU:-${ALLOW_BUSY_GPU:-}}" in
    1|true|TRUE|yes|YES|on|ON)
      return 1
      ;;
  esac
  return 0
}

parallel_enabled() {
  case "${RUN_PARALLEL}" in
    1|true|TRUE|yes|YES|on|ON)
      return 0
      ;;
  esac
  return 1
}

ablation_after_main_failure_enabled() {
  case "${RUN_ABLATION_ON_MAIN_FAILURE}" in
    0|false|FALSE|no|NO|off|OFF)
      return 1
      ;;
  esac
  return 0
}

preflight_enabled() {
  case "${RUN_PREFLIGHT}" in
    0|false|FALSE|no|NO|off|OFF)
      return 1
      ;;
  esac
  return 0
}

closeness_large_supplement_enabled() {
  case "${RUN_CLOSENESS_LARGE_SUPPLEMENT}" in
    0|false|FALSE|no|NO|off|OFF)
      return 1
      ;;
  esac
  return 0
}

require_gpu_idle() {
  local gpu="$1"
  local phase="$2"

  if ! gpu_idle_check_enabled; then
    echo "GPU idle check skipped for ${phase}: GPU=${gpu}"
    return 0
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU idle check failed for ${phase}: nvidia-smi not found"
    exit 2
  fi

  local max_mem_mb="${EGGPU_IDLE_MAX_MEMORY_MB:-1024}"
  local max_util="${EGGPU_IDLE_MAX_UTILIZATION:-5}"
  local gpu_rows=""
  local query_ok=0
  local attempt
  for attempt in 1 2 3 4 5; do
    if gpu_rows="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)"; then
      query_ok=1
      break
    fi
    if gpu_rows="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null)"; then
      query_ok=1
      break
    fi
    sleep 1
  done
  if [[ "${query_ok}" -ne 1 ]]; then
    echo "GPU idle check failed for ${phase}: unable to query GPU utilization with nvidia-smi after 5 attempts"
    exit 2
  fi

  local row idx mem util found=0
  while IFS= read -r row; do
    row="${row// /}"
    IFS=',' read -r idx mem util <<< "${row}"
    if [[ "${idx}" == "${gpu}" ]]; then
      mem="${mem//MiB/}"
      util="${util//%/}"
      found=1
      break
    fi
  done <<< "${gpu_rows}"

  if [[ "${found}" -ne 1 ]]; then
    echo "GPU idle check failed for ${phase}: GPU=${gpu} not found by nvidia-smi"
    echo "${gpu_rows}"
    exit 2
  fi

  if [[ "${mem}" -gt "${max_mem_mb}" || "${util}" -gt "${max_util}" ]]; then
    echo "GPU idle check failed for ${phase}: GPU=${gpu} is not idle (memory=${mem} MiB, utilization=${util}%)."
    echo "Thresholds: EGGPU_IDLE_MAX_MEMORY_MB=${max_mem_mb}, EGGPU_IDLE_MAX_UTILIZATION=${max_util}."
    echo "Current GPUs:"
    echo "${gpu_rows}"
    echo "Compute processes, if queryable:"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true
    echo "Pick an idle GPU. Override with EGGPU_ALLOW_BUSY_GPU=1 only for deliberate non-comparable/debug runs."
    exit 2
  fi

  echo "GPU idle check passed for ${phase}: GPU=${gpu} memory=${mem} MiB utilization=${util}%"
}

run_preflight() {
  env -u CFLAGS -u CPPFLAGS -u CXXFLAGS \
      -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u CPATH -u LIBRARY_PATH \
      CONDA_EXE="${CONDA_EXE_PATH}" \
      EGGPU_CHILD_PYTHON="${COMMON_PY}" \
      EGGPU_USE_CONDA_RUN=FALSE \
      CUDA_VISIBLE_DEVICES="${MAIN_GPU}" \
      EGGPU_MONITOR_GPU_INDEX="${MAIN_GPU}" \
      EGGPU_CUDA_ROOT="${CUDA_ROOT}" \
      CUDA_PATH="${CUDA_ROOT}" \
      CONDA_PREFIX="${CUDA_ROOT}" \
      EASYGRAPH_ENABLE_GPU=TRUE \
      EASYGRAPH_GPU_BACKEND=mine \
      EASYGRAPH_GPU_STRICT_ERRORS=TRUE \
      EGGPU_GPU_VISIBILITY_MARKER="${EGGPU_GPU_VISIBILITY_MARKER}" \
      EGGPU_GPU_VISIBILITY_MARKER_MB="${EGGPU_GPU_VISIBILITY_MARKER_MB}" \
      EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB="${EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB}" \
      EASYGRAPH_GPU_ADAPTIVE_POLICY=TRUE \
      EASYGRAPH_GPU_COMPONENT_DENSE_RETURN=FALSE \
      EASYGRAPH_GPU_SCC_ACTIVE_TRIM=TRUE \
      EASYGRAPH_GPU_SCC_ACTIVE_TRIM_MAX_ITERS=16 \
      EASYGRAPH_GPU_SCC_DEGREE_PIVOT=TRUE \
      EASYGRAPH_GPU_SCC_HOST_ENABLE=FALSE \
      EASYGRAPH_GPU_KCORE_HOST_ENABLE=FALSE \
      EASYGRAPH_GPU_SSSP_HOST_ENABLE=FALSE \
      EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_MIN_AVG_DEGREE=10 \
      EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_MIN_MAX_DEGREE="${EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_MIN_MAX_DEGREE:-AUTO}" \
      EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_THREADS="${EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_THREADS:-1024}" \
      EASYGRAPH_GPU_BC_WARP_SIZE="${EASYGRAPH_GPU_BC_WARP_SIZE:-AUTO}" \
      EASYGRAPH_GPU_CONSTRAINT_SMALLER_INTERSECTION=AUTO \
      EGGPU_CLOSENESS_EXACT_MAX_NODES="${EGGPU_CLOSENESS_EXACT_MAX_NODES:-1000000}" \
      EGGPU_CLOSENESS_EXACT_MAX_WORK="${EGGPU_CLOSENESS_EXACT_MAX_WORK}" \
      PYTHONPATH="${EG_REPO}:${PYTHONPATH:-}" \
      LD_LIBRARY_PATH="${LD_PATH}" \
      "${COMMON_PY}" benchmarking/preflight_full_eval_ready.py
}

run_closeness_large_supplement() {
  set -euo pipefail
  env -u CFLAGS -u CPPFLAGS -u CXXFLAGS \
      -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u CPATH -u LIBRARY_PATH \
      CONDA_EXE="${CONDA_EXE_PATH}" \
      EGGPU_CHILD_PYTHON="${COMMON_PY}" \
      EGGPU_USE_CONDA_RUN=FALSE \
      CUDA_VISIBLE_DEVICES="${MAIN_GPU}" \
      EGGPU_MONITOR_GPU_INDEX="${MAIN_GPU}" \
      EGGPU_CUDA_ROOT="${CUDA_ROOT}" \
      CUDA_PATH="${CUDA_ROOT}" \
      CONDA_PREFIX="${CUDA_ROOT}" \
      EASYGRAPH_ENABLE_GPU=TRUE \
      EASYGRAPH_GPU_BACKEND=mine \
      EASYGRAPH_GPU_STRICT_ERRORS=TRUE \
      EGGPU_GPU_VISIBILITY_MARKER="${EGGPU_GPU_VISIBILITY_MARKER}" \
      EGGPU_GPU_VISIBILITY_MARKER_MB="${EGGPU_GPU_VISIBILITY_MARKER_MB}" \
      EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB="${EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB}" \
      EASYGRAPH_GPU_RESULT_CACHE=FALSE \
      EASYGRAPH_GPU_RESULT_CACHE_RETURN_COPY=FALSE \
      EASYGRAPH_GPU_SCC_HOST_ENABLE=FALSE \
      EASYGRAPH_GPU_KCORE_HOST_ENABLE=FALSE \
      EASYGRAPH_GPU_SSSP_HOST_ENABLE=FALSE \
      PYTHONPATH="${EG_REPO}:${PYTHONPATH:-}" \
      LD_LIBRARY_PATH="${LD_PATH}" \
      "${COMMON_PY}" benchmarking/run_closeness_large_supplement.py \
        "${MAIN_OUT}" \
        --easygraph-repo "${EG_REPO}" \
        --sources "${CLOSENESS_LARGE_SOURCES}" \
        --gpu "${MAIN_GPU}" \
        --timeout "${CLOSENESS_LARGE_TIMEOUT}" \
        --python "${COMMON_PY}"
}

run_main_eval() {
  set -euo pipefail
  env -u CFLAGS -u CPPFLAGS -u CXXFLAGS \
      -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u CPATH -u LIBRARY_PATH \
      CONDA_EXE="${CONDA_EXE_PATH}" \
      EGGPU_CHILD_PYTHON="${COMMON_PY}" \
      EGGPU_USE_CONDA_RUN=FALSE \
      CUDA_VISIBLE_DEVICES="${MAIN_GPU}" \
      EGGPU_MONITOR_GPU_INDEX="${MAIN_GPU}" \
      EGGPU_CUDA_ROOT="${CUDA_ROOT}" \
      CUDA_PATH="${CUDA_ROOT}" \
      CONDA_PREFIX="${CUDA_ROOT}" \
      EASYGRAPH_ENABLE_GPU=TRUE \
      EASYGRAPH_GPU_BACKEND=mine \
      EASYGRAPH_GPU_STRICT_ERRORS=TRUE \
      EGGPU_GPU_VISIBILITY_MARKER="${EGGPU_GPU_VISIBILITY_MARKER}" \
      EGGPU_GPU_VISIBILITY_MARKER_MB="${EGGPU_GPU_VISIBILITY_MARKER_MB}" \
      EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB="${EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB}" \
      EASYGRAPH_GPU_ADAPTIVE_POLICY=TRUE \
      EASYGRAPH_GPU_COMPONENT_DENSE_RETURN=FALSE \
      EASYGRAPH_GPU_SCC_ACTIVE_TRIM=TRUE \
      EASYGRAPH_GPU_SCC_ACTIVE_TRIM_MAX_ITERS=16 \
      EASYGRAPH_GPU_SCC_DEGREE_PIVOT=TRUE \
      EASYGRAPH_GPU_SCC_HOST_ENABLE=FALSE \
      EASYGRAPH_GPU_KCORE_HOST_ENABLE=FALSE \
      EASYGRAPH_GPU_SSSP_HOST_ENABLE=FALSE \
      EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_MIN_AVG_DEGREE=10 \
      EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_MIN_MAX_DEGREE="${EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_MIN_MAX_DEGREE:-AUTO}" \
      EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_THREADS="${EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_THREADS:-1024}" \
      EASYGRAPH_GPU_BC_WARP_SIZE="${EASYGRAPH_GPU_BC_WARP_SIZE:-AUTO}" \
      EASYGRAPH_GPU_CONSTRAINT_SMALLER_INTERSECTION=AUTO \
      EGGPU_CLOSENESS_EXACT_MAX_NODES="${EGGPU_CLOSENESS_EXACT_MAX_NODES:-1000000}" \
      EGGPU_CLOSENESS_EXACT_MAX_WORK="${EGGPU_CLOSENESS_EXACT_MAX_WORK}" \
      PYTHONPATH="${EG_REPO}:${PYTHONPATH:-}" \
      LD_LIBRARY_PATH="${LD_PATH}" \
      "${COMMON_PY}" benchmarking/run_full_baselines.py \
        --gpu "${MAIN_GPU}" \
        --repeat 3 \
        --warmup 0 \
        --easygraph-warmup 2 \
        --library-timeout "${LIBRARY_TIMEOUT}" \
        --inter-run-cooldown 1.0 \
        --pr-alpha 0.75 \
        --pr-eps 1e-6 \
        --pr-max-iter 200 \
        --sssp-sources 8 \
        --bc-sources 16 \
        --datasets all \
        --functions all \
        --out-dir "${MAIN_OUT}"
}

run_ablation_one() {
  local experiment="$1"
  local variant="$2"
  local edge_path="$3"
  local dataset_name="$4"
  local graph_type="$5"
  local functions="$6"
  local repeat="$7"
  local warmup="$8"
  local out_path="$9"
  shift 9

  set +e
  timeout "${ABLATION_TIMEOUT}" \
    env -u CFLAGS -u CPPFLAGS -u CXXFLAGS \
      -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u CPATH -u LIBRARY_PATH \
      CONDA_EXE="${CONDA_EXE_PATH}" \
      CUDA_VISIBLE_DEVICES="${ABL_GPU}" \
      EGGPU_MONITOR_GPU_INDEX="${ABL_GPU}" \
      EGGPU_CUDA_ROOT="${CUDA_ROOT}" \
      CUDA_PATH="${CUDA_ROOT}" \
      CONDA_PREFIX="${CUDA_ROOT}" \
      EASYGRAPH_ENABLE_GPU=TRUE \
      EASYGRAPH_GPU_BACKEND=mine \
      EASYGRAPH_GPU_STRICT_ERRORS=TRUE \
      EGGPU_GPU_VISIBILITY_MARKER="${EGGPU_GPU_VISIBILITY_MARKER}" \
      EGGPU_GPU_VISIBILITY_MARKER_MB="${EGGPU_GPU_VISIBILITY_MARKER_MB}" \
      EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB="${EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB}" \
      EASYGRAPH_GPU_ADAPTIVE_POLICY=TRUE \
      EASYGRAPH_GPU_COMPONENT_DENSE_RETURN=FALSE \
      EASYGRAPH_GPU_SCC_ACTIVE_TRIM=TRUE \
      EASYGRAPH_GPU_SCC_ACTIVE_TRIM_MAX_ITERS=16 \
      EASYGRAPH_GPU_SCC_DEGREE_PIVOT=TRUE \
      EASYGRAPH_GPU_SCC_HOST_ENABLE=FALSE \
      EASYGRAPH_GPU_KCORE_HOST_ENABLE=FALSE \
      EASYGRAPH_GPU_SSSP_HOST_ENABLE=FALSE \
      EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_MIN_AVG_DEGREE=10 \
      EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_MIN_MAX_DEGREE="${EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_MIN_MAX_DEGREE:-AUTO}" \
      EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_THREADS="${EASYGRAPH_GPU_KCORE_SINGLE_BLOCK_THREADS:-1024}" \
      EASYGRAPH_GPU_BC_WARP_SIZE="${EASYGRAPH_GPU_BC_WARP_SIZE:-AUTO}" \
      EASYGRAPH_GPU_CONSTRAINT_SMALLER_INTERSECTION=AUTO \
      EGGPU_CLOSENESS_EXACT_MAX_NODES="${EGGPU_CLOSENESS_EXACT_MAX_NODES:-1000000}" \
      EGGPU_CLOSENESS_EXACT_MAX_WORK="${EGGPU_CLOSENESS_EXACT_MAX_WORK}" \
      PYTHONPATH="${EG_REPO}:${PYTHONPATH:-}" \
      LD_LIBRARY_PATH="${LD_PATH}" \
      "${COMMON_PY}" benchmarking/run_eggpu_ablations.py \
        --experiment "${experiment}" \
        --variant "${variant}" \
        --edge-path "${edge_path}" \
        --dataset-name "${dataset_name}" \
        --graph-type "${graph_type}" \
        --functions "${functions}" \
        --repeat "${repeat}" \
        --warmup "${warmup}" \
        --gpu "${ABL_GPU}" \
        --out "${out_path}" \
        "$@"
  local rc=$?
  set -e
  if [[ "${rc}" -eq 124 ]]; then
    {
      echo "experiment,variant,dataset,graph_type,function,metric,value,status,notes,repeat"
      echo "${experiment},${variant},${dataset_name},${graph_type},ALL,timeout_seconds,,timeout,timeout after ${ABLATION_TIMEOUT}s,"
    } > "${out_path}"
    echo "timeout: ${out_path}"
    return 0
  fi
  return "${rc}"
}

run_ablations() {
  set -euo pipefail
  local SPECS=(
    "ca-GrQc|undirected|datasets/undirected/ca-GrQc.txt" \
    "ca-HepTh|undirected|datasets/undirected/ca-HepTh.txt" \
    "LastFM|undirected|datasets/undirected/LastFM.txt" \
    "pgp|undirected|datasets/undirected/pgp.txt" \
    "ca-CondMat|undirected|datasets/undirected/ca-CondMat.txt" \
    "ca-HepPh|undirected|datasets/undirected/ca-HepPh.txt" \
    "email-Enron|undirected|datasets/undirected/email-Enron.txt" \
    "com-youtube|undirected|datasets/undirected/com-youtube.ungraph.txt" \
    "p2p-Gnutella04|directed|datasets/directed/p2p-Gnutella04.txt" \
    "p2p-Gnutella08|directed|datasets/directed/p2p-Gnutella08.txt" \
    "wiki-Vote|directed|datasets/directed/wiki-Vote.txt" \
    "soc-Epinions1|directed|datasets/directed/soc-Epinions1.txt" \
    "email-EuAll|directed|datasets/directed/email-EuAll.txt" \
    "soc-Slashdot0811|directed|datasets/directed/soc-Slashdot0811.txt" \
    "web-NotreDame|directed|datasets/directed/web-NotreDame.txt" \
    "ER-100k|directed|datasets/directed/ER-100k.txt" \
    "wiki-Talk|directed|datasets/directed/wiki-Talk.txt"
  )
  local VARIANTS=(full no_graph_context no_cpp_graph_cache no_device_csr_cache no_adaptive_policy)
  local DATASET_TOTAL="${#SPECS[@]}"
  local TASK_TOTAL=$(( DATASET_TOTAL * ( ${#VARIANTS[@]} + 2 ) ))
  local TASK_INDEX=0
  local DATASET_INDEX=0

  ablation_progress() {
    local dataset_name="$1"
    local experiment="$2"
    local variant="$3"
    TASK_INDEX=$((TASK_INDEX + 1))
    local pct=100
    if [[ "${TASK_TOTAL}" -gt 0 ]]; then
      pct=$((100 * TASK_INDEX / TASK_TOTAL))
    fi
    echo "[progress] ablation ${TASK_INDEX}/${TASK_TOTAL} (${pct}%) dataset ${DATASET_INDEX}/${DATASET_TOTAL} ${dataset_name}: ${experiment}/${variant}"
  }

  for spec in "${SPECS[@]}"
  do
    DATASET_INDEX=$((DATASET_INDEX + 1))
    IFS='|' read -r NAME GTYPE PATH_IN <<< "${spec}"

    for VARIANT in "${VARIANTS[@]}"
    do
      ablation_progress "${NAME}" workflow "${VARIANT}"
      run_ablation_one \
        workflow \
        "${VARIANT}" \
        "${PATH_IN}" \
        "${NAME}" \
        "${GTYPE}" \
        all \
        5 \
        2 \
        "${ABL_OUT}/workflow_${NAME}_${VARIANT}.csv"
    done

    ablation_progress "${NAME}" return full
    run_ablation_one \
      return \
      full \
      "${PATH_IN}" \
      "${NAME}" \
      "${GTYPE}" \
      all \
      5 \
      2 \
      "${ABL_OUT}/return_${NAME}.csv"

    ablation_progress "${NAME}" layout full
    run_ablation_one \
      layout \
      full \
      "${PATH_IN}" \
      "${NAME}" \
      "${GTYPE}" \
      all \
      5 \
      2 \
      "${ABL_OUT}/layout_${NAME}.csv" \
      --layout-pr-iters 20
  done

  "${COMMON_PY}" - <<PY
from pathlib import Path
import pandas as pd

out = Path("${ABL_OUT}")
dfs = [pd.read_csv(p) for p in sorted(out.glob("*.csv"))]
if not dfs:
    raise SystemExit(f"no ablation csv files found in {out}")
pd.concat(dfs, ignore_index=True).to_csv(out / "ablation_all.csv", index=False)
print("Ablation done:", out / "ablation_all.csv")
PY
}

run_main_audits() {
  local GROUP_RC=0

  echo "Main audit starting: OUT=${MAIN_OUT}, LOG=benchmarking/results/${MAIN_ID}.audit.log"
  "${COMMON_PY}" benchmarking/audit_full_result.py "${MAIN_OUT}" > "benchmarking/results/${MAIN_ID}.audit.log" 2>&1
  AUDIT_RC=$?

  if [[ "${AUDIT_RC}" -ne 0 ]]; then
    echo "Main audit exit code: ${AUDIT_RC}"
    echo "Main audit failed; see benchmarking/results/${MAIN_ID}.audit.log and ${MAIN_OUT}/audit/AUDIT.md"
    echo "Main result: ${MAIN_OUT}"
    GROUP_RC=1
  else
    echo "Main audit passed: OUT=${MAIN_OUT}/audit"
  fi

  echo "Backend separation audit starting: OUT=${MAIN_OUT}, LOG=benchmarking/results/${MAIN_ID}.backend_separation.log"
  "${COMMON_PY}" benchmarking/audit_backend_separation.py "${MAIN_OUT}" > "benchmarking/results/${MAIN_ID}.backend_separation.log" 2>&1
  BACKEND_AUDIT_RC=$?

  if [[ "${BACKEND_AUDIT_RC}" -ne 0 ]]; then
    echo "Backend separation audit exit code: ${BACKEND_AUDIT_RC}"
    echo "Backend separation audit failed; see benchmarking/results/${MAIN_ID}.backend_separation.log and ${MAIN_OUT}/audit/backend_separation/BACKEND_SEPARATION_AUDIT.md"
    echo "Main result: ${MAIN_OUT}"
    GROUP_RC=1
  else
    echo "Backend separation audit passed: OUT=${MAIN_OUT}/audit/backend_separation"
  fi

  echo "Pair-level SOTA summary starting: OUT=${MAIN_OUT}, LOG=benchmarking/results/${MAIN_ID}.non_sota.log"
  "${COMMON_PY}" benchmarking/summarize_non_sota_pairs.py "${MAIN_OUT}" > "benchmarking/results/${MAIN_ID}.non_sota.log" 2>&1
  NON_SOTA_RC=$?

  if [[ "${NON_SOTA_RC}" -ne 0 ]]; then
    echo "Pair-level SOTA summary exit code: ${NON_SOTA_RC}"
    echo "Pair-level SOTA summary failed; see benchmarking/results/${MAIN_ID}.non_sota.log"
    echo "Main result: ${MAIN_OUT}"
    GROUP_RC=1
  else
    echo "Pair-level SOTA summary written: OUT=${MAIN_OUT}/EGGPU_NON_SOTA_PAIRS.md"
  fi

  if closeness_large_supplement_enabled; then
    echo "Closeness large-graph supplement starting: OUT=${MAIN_OUT}/closeness_large_sampled, LOG=benchmarking/results/${MAIN_ID}.closeness_large.log"
    run_closeness_large_supplement > >(tee "benchmarking/results/${MAIN_ID}.closeness_large.log") 2>&1
    CLOSENESS_LARGE_RC=$?

    if [[ "${CLOSENESS_LARGE_RC}" -ne 0 ]]; then
      echo "Closeness large-graph supplement exit code: ${CLOSENESS_LARGE_RC}"
      echo "Closeness large-graph supplement failed; see benchmarking/results/${MAIN_ID}.closeness_large.log"
      echo "Main result: ${MAIN_OUT}"
      GROUP_RC=1
    else
      echo "Closeness large-graph supplement written: OUT=${MAIN_OUT}/closeness_large_sampled"
    fi
  else
    CLOSENESS_LARGE_RC=0
    echo "Closeness large-graph supplement skipped because RUN_CLOSENESS_LARGE_SUPPLEMENT=${RUN_CLOSENESS_LARGE_SUPPLEMENT}"
  fi

  echo "Final result summary starting: OUT=${MAIN_OUT}, LOG=benchmarking/results/${MAIN_ID}.final_summary.log"
  "${COMMON_PY}" benchmarking/summarize_final_result.py "${MAIN_OUT}" > "benchmarking/results/${MAIN_ID}.final_summary.log" 2>&1
  FINAL_SUMMARY_RC=$?

  if [[ "${FINAL_SUMMARY_RC}" -ne 0 ]]; then
    echo "Final result summary exit code: ${FINAL_SUMMARY_RC}"
    echo "Final result summary failed; see benchmarking/results/${MAIN_ID}.final_summary.log"
    echo "Main result: ${MAIN_OUT}"
    GROUP_RC=1
  else
    echo "Final result summary written: OUT=${MAIN_OUT}/EGGPU_FINAL_RESULT_SUMMARY.md"
  fi

  return "${GROUP_RC}"
}

MAIN_RC=0
AUDIT_RC=NOT_RUN
BACKEND_AUDIT_RC=NOT_RUN
NON_SOTA_RC=NOT_RUN
FINAL_SUMMARY_RC=NOT_RUN
CLOSENESS_LARGE_RC=NOT_RUN
ABL_RC=0
AUDIT_GROUP_RC=0
PREFLIGHT_RC=0

if parallel_enabled; then
  if [[ "${MAIN_GPU}" == "${ABL_GPU}" ]]; then
    echo "RUN_PARALLEL=1 requires different GPUs: MAIN_GPU=${MAIN_GPU}, ABL_GPU=${ABL_GPU}"
    exit 2
  fi

  echo "Parallel mode enabled: main GPU=${MAIN_GPU}, ablation GPU=${ABL_GPU}"
  require_gpu_idle "${MAIN_GPU}" "main eval"
  require_gpu_idle "${ABL_GPU}" "ablation"

  if preflight_enabled; then
    echo "Preflight starting: GPU=${MAIN_GPU}, LOG=benchmarking/results/${MAIN_ID}.preflight.log"
    run_preflight > "benchmarking/results/${MAIN_ID}.preflight.log" 2>&1
    PREFLIGHT_RC=$?
    if [[ "${PREFLIGHT_RC}" -ne 0 ]]; then
      echo "Preflight exit code: ${PREFLIGHT_RC}"
      echo "Preflight failed; see benchmarking/results/${MAIN_ID}.preflight.log"
      exit 1
    fi
    echo "Preflight passed"
  else
    echo "Preflight skipped because RUN_PREFLIGHT=${RUN_PREFLIGHT}"
  fi

  echo "Main eval starting: GPU=${MAIN_GPU}, OUT=${MAIN_OUT}, LOG=benchmarking/results/${MAIN_ID}.run.log"
  ( run_main_eval ) > >(tee "benchmarking/results/${MAIN_ID}.run.log") 2>&1 &
  MAIN_PID=$!

  echo "Ablation starting: GPU=${ABL_GPU}, OUT=${ABL_OUT}, LOG=benchmarking/results/${ABL_ID}.run.log"
  ( run_ablations ) > >(tee "benchmarking/results/${ABL_ID}.run.log") 2>&1 &
  ABL_PID=$!

  wait "${MAIN_PID}"
  MAIN_RC=$?
  if [[ "${MAIN_RC}" -eq 0 ]]; then
    echo "Main eval finished: OUT=${MAIN_OUT}"
    run_main_audits
    AUDIT_GROUP_RC=$?
  else
    echo "Main eval exit code: ${MAIN_RC}"
    echo "Main result: ${MAIN_OUT}"
    AUDIT_GROUP_RC=1
  fi

  wait "${ABL_PID}"
  ABL_RC=$?

  echo "Main eval exit code: ${MAIN_RC}"
  echo "Preflight exit code: ${PREFLIGHT_RC}"
  echo "Main audit exit code: ${AUDIT_RC}"
  echo "Backend separation audit exit code: ${BACKEND_AUDIT_RC}"
  echo "Pair-level SOTA summary exit code: ${NON_SOTA_RC}"
  echo "Closeness large-graph supplement exit code: ${CLOSENESS_LARGE_RC}"
  echo "Final result summary exit code: ${FINAL_SUMMARY_RC}"
  echo "Ablation exit code: ${ABL_RC}"
  echo "Main result: ${MAIN_OUT}"
  if [[ "${CLOSENESS_LARGE_RC}" == "NOT_RUN" ]]; then
    echo "Closeness supplement result: NOT_RUN"
  else
    echo "Closeness supplement result: ${MAIN_OUT}/closeness_large_sampled"
  fi
  echo "Ablation result: ${ABL_OUT}"

  if [[ "${MAIN_RC}" -ne 0 || "${AUDIT_GROUP_RC}" -ne 0 || "${ABL_RC}" -ne 0 ]]; then
    exit 1
  fi
  exit 0
fi

echo "Main eval starting: GPU=${MAIN_GPU}, OUT=${MAIN_OUT}, LOG=benchmarking/results/${MAIN_ID}.run.log"
require_gpu_idle "${MAIN_GPU}" "main eval"
if preflight_enabled; then
  echo "Preflight starting: GPU=${MAIN_GPU}, LOG=benchmarking/results/${MAIN_ID}.preflight.log"
  run_preflight > "benchmarking/results/${MAIN_ID}.preflight.log" 2>&1
  PREFLIGHT_RC=$?
  if [[ "${PREFLIGHT_RC}" -ne 0 ]]; then
    echo "Preflight exit code: ${PREFLIGHT_RC}"
    echo "Preflight failed; see benchmarking/results/${MAIN_ID}.preflight.log"
    exit 1
  fi
  echo "Preflight passed"
else
  echo "Preflight skipped because RUN_PREFLIGHT=${RUN_PREFLIGHT}"
fi
( run_main_eval ) > >(tee "benchmarking/results/${MAIN_ID}.run.log") 2>&1
MAIN_RC=$?

if [[ "${MAIN_RC}" -ne 0 ]]; then
  echo "Main eval exit code: ${MAIN_RC}"
  echo "Main result: ${MAIN_OUT}"
  AUDIT_GROUP_RC=1
  if ! ablation_after_main_failure_enabled; then
    exit 1
  fi
  echo "Main eval failed; continuing to ablation because RUN_ABLATION_ON_MAIN_FAILURE=${RUN_ABLATION_ON_MAIN_FAILURE}"
else
  echo "Main eval finished: OUT=${MAIN_OUT}"
  if run_main_audits; then
    AUDIT_GROUP_RC=0
  else
    AUDIT_GROUP_RC=1
    if ! ablation_after_main_failure_enabled; then
      exit 1
    fi
    echo "Main audit failed; continuing to ablation because RUN_ABLATION_ON_MAIN_FAILURE=${RUN_ABLATION_ON_MAIN_FAILURE}"
  fi
fi

echo "Ablation starting: GPU=${ABL_GPU}, OUT=${ABL_OUT}, LOG=benchmarking/results/${ABL_ID}.run.log"
require_gpu_idle "${ABL_GPU}" "ablation"
( run_ablations ) > >(tee "benchmarking/results/${ABL_ID}.run.log") 2>&1
ABL_RC=$?

echo "Main eval exit code: ${MAIN_RC}"
echo "Preflight exit code: ${PREFLIGHT_RC}"
echo "Main audit exit code: ${AUDIT_RC}"
echo "Backend separation audit exit code: ${BACKEND_AUDIT_RC}"
echo "Pair-level SOTA summary exit code: ${NON_SOTA_RC}"
echo "Closeness large-graph supplement exit code: ${CLOSENESS_LARGE_RC}"
echo "Final result summary exit code: ${FINAL_SUMMARY_RC}"
echo "Ablation exit code: ${ABL_RC}"
echo "Main result: ${MAIN_OUT}"
if [[ "${CLOSENESS_LARGE_RC}" == "NOT_RUN" ]]; then
  echo "Closeness supplement result: NOT_RUN"
else
  echo "Closeness supplement result: ${MAIN_OUT}/closeness_large_sampled"
fi
echo "Ablation result: ${ABL_OUT}"

if [[ "${MAIN_RC}" -ne 0 || "${AUDIT_GROUP_RC}" -ne 0 || "${ABL_RC}" -ne 0 ]]; then
  exit 1
fi
