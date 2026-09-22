#!/usr/bin/env bash
set -euo pipefail

EVAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${EVAL_ROOT}/.." && pwd)"
cd "${EVAL_ROOT}"

PAPER_GPU="${PAPER_GPU:-0}"
SECONDARY_GPU="${SECONDARY_GPU:-1}"
PAPER_REPEAT="${PAPER_REPEAT:-5}"
PAPER_MEMORY_REPEAT="${PAPER_MEMORY_REPEAT:-3}"
PAPER_TIMEOUT="${PAPER_TIMEOUT:-100}"
PAPER_RUN_TS="${PAPER_RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
COMMON_PY="${COMMON_PY:-/home/dataset-assist-0/einwang/conda_cache/conda_env/EGGPU/bin/python}"
CUDA_ROOT="${EGGPU_CUDA_ROOT:-/home/dataset-assist-0/einwang/conda_cache/conda_env/sglang}"
EG_REPO="${EG_REPO:-${WORKSPACE_ROOT}/Easy-Graph}"
LEGACY_EVAL_ROOT="${WORKSPACE_ROOT}/../EG_Evaluation"
GUNROCK_PATHS="${EG_GUNROCK_BIN_PATHS:-${LEGACY_EVAL_ROOT}/gunrock_latest/build_cuda132_a100_migrated/bin:${LEGACY_EVAL_ROOT}/gunrock_legacy_master/build_lcc_result_cuda128_clean/bin}"
CUDA_LD_PATH="${CUDA_ROOT}/lib:${CUDA_ROOT}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

MAIN_ID="full_eval_gpu${PAPER_GPU}_${PAPER_RUN_TS}_repeat${PAPER_REPEAT}_splitmem_exact_mst_strict_nxcg"
ABL_ID="ablation_gpu${PAPER_GPU}_${PAPER_RUN_TS}_repeat5_strengthened"
FIRST_ID="eggpu_first_use_gpu${PAPER_GPU}_${PAPER_RUN_TS}_repeat${PAPER_REPEAT}"
NATURAL_ID="eggpu_natural_workflow_gpu${PAPER_GPU}_${PAPER_RUN_TS}_repeat${PAPER_REPEAT}"
MAIN_OUT="benchmarking/results/${MAIN_ID}"
ABL_OUT="benchmarking/results/${ABL_ID}"
FIRST_OUT="benchmarking/results/${FIRST_ID}"
NATURAL_OUT="benchmarking/results/${NATURAL_ID}"
TOP_LOG="benchmarking/results/complete_paper_experiments_${PAPER_RUN_TS}.console.log"
MARKER_DECISION="benchmarking/results/visibility_marker_enabled_${PAPER_RUN_TS}.txt"
MARKER_ENABLED="TRUE"
MARKER_MB=256

if [[ "${PAPER_GPU}" == "${SECONDARY_GPU}" ]]; then
  echo "PAPER_GPU and SECONDARY_GPU must identify different devices." >&2
  exit 2
fi
if [[ "${PAPER_REPEAT}" -lt 2 || "${PAPER_MEMORY_REPEAT}" -lt 1 ]]; then
  echo "PAPER_REPEAT must be >=2 and PAPER_MEMORY_REPEAT must be >=1." >&2
  exit 2
fi

exec > >(tee -a "${TOP_LOG}") 2>&1

echo "[protocol] authoritative timing GPU=${PAPER_GPU}; secondary GPU=${SECONDARY_GPU} remains idle"
echo "[protocol] timing estimator=arithmetic mean; error bar=sample standard deviation; repeat=${PAPER_REPEAT}"
echo "[protocol] memory repeat=${PAPER_MEMORY_REPEAT}; timing and memory passes are isolated"
echo "[protocol] natural workflow positions 2-3 are compared with same-function isolated First-use controls"
echo "[protocol] scaling experiment is intentionally deferred"

COMMON_ENV=(
  "CUDA_VISIBLE_DEVICES=${PAPER_GPU}"
  "EGGPU_MONITOR_GPU_INDEX=${PAPER_GPU}"
  "EGGPU_CUDA_ROOT=${CUDA_ROOT}"
  "CUDA_PATH=${CUDA_ROOT}"
  "CONDA_PREFIX=${CUDA_ROOT}"
  "EASYGRAPH_ENABLE_GPU=TRUE"
  "EASYGRAPH_GPU_STRICT_ERRORS=TRUE"
  "EASYGRAPH_GPU_RESULT_CACHE=FALSE"
  "EASYGRAPH_GPU_RESULT_CACHE_RETURN_COPY=FALSE"
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
  "EGGPU_GPU_VISIBILITY_MARKER=TRUE"
  "EGGPU_GPU_VISIBILITY_MARKER_MB=256"
  "EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB=AUTO"
  "EG_GUNROCK_BIN_PATHS=${GUNROCK_PATHS}"
  "PYTHONPATH=${EG_REPO}:${PYTHONPATH:-}"
)

echo "[1/5] correctness/protocol gate"
env -u CFLAGS -u CPPFLAGS -u CXXFLAGS \
    -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u CPATH -u LIBRARY_PATH \
    -u LD_LIBRARY_PATH \
    "${COMMON_ENV[@]}" \
    FINAL_GPU="${PAPER_GPU}" \
    FINAL_GATE_ONLY=TRUE \
    FINAL_BENCHMARK_REPEAT="${PAPER_REPEAT}" \
    FINAL_MEMORY_REPEAT="${PAPER_MEMORY_REPEAT}" \
    FINAL_MARKER_DECISION_FILE="${MARKER_DECISION}" \
    COMMON_PY="${COMMON_PY}" \
    EGGPU_CUDA_ROOT="${CUDA_ROOT}" \
    EG_GUNROCK_BIN_PATHS="${GUNROCK_PATHS}" \
    bash run_final_gpu5_gate_and_full.sh

if [[ ! -f "${MARKER_DECISION}" ]]; then
  echo "Marker decision artifact was not produced by the gate." >&2
  exit 2
fi
MARKER_ENABLED="$(tr -d '[:space:]' < "${MARKER_DECISION}")"
if [[ "${MARKER_ENABLED}" != "TRUE" ]]; then
  MARKER_ENABLED="FALSE"
  MARKER_MB=0
  COMMON_ENV+=(
    "EGGPU_GPU_VISIBILITY_MARKER=FALSE"
    "EGGPU_GPU_VISIBILITY_MARKER_MB=0"
    "EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB=0"
  )
  echo "[protocol] auxiliary nvitop marker disabled by A/B gate; experiment continues without it"
else
  echo "[protocol] auxiliary nvitop marker enabled after passing A/B gate"
fi

echo "[2/5] steady-state baseline comparison and complete ablation"
set +e
env -u CFLAGS -u CPPFLAGS -u CXXFLAGS \
    -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u CPATH -u LIBRARY_PATH \
    -u LD_LIBRARY_PATH \
    "${COMMON_ENV[@]}" \
    RUN_TS_OVERRIDE="${PAPER_RUN_TS}" \
    COMMON_PY="${COMMON_PY}" \
    EGGPU_CUDA_ROOT="${CUDA_ROOT}" \
    EG_GUNROCK_BIN_PATHS="${GUNROCK_PATHS}" \
    MAIN_GPU="${PAPER_GPU}" ABL_GPU="${PAPER_GPU}" RUN_PARALLEL=0 \
    LIBRARY_TIMEOUT="${PAPER_TIMEOUT}" ABLATION_TIMEOUT="${PAPER_TIMEOUT}" \
    BENCHMARK_REPEAT="${PAPER_REPEAT}" MEMORY_REPEAT="${PAPER_MEMORY_REPEAT}" \
    EGGPU_WARMUP=2 RUN_PREFLIGHT=TRUE RUN_CLOSENESS_LARGE_SUPPLEMENT=TRUE \
    EGGPU_GPU_VISIBILITY_MARKER="${MARKER_ENABLED}" EGGPU_GPU_VISIBILITY_MARKER_MB="${MARKER_MB}" \
    bash run_main_and_ablation.sh
MAIN_ABL_RC=$?
set -e

if [[ ! -f "${MAIN_OUT}/results_long.csv" || ! -f "${ABL_OUT}/ablation_all.csv" ]]; then
  echo "Expected main/ablation outputs were not produced." >&2
  exit 2
fi

"${COMMON_PY}" - "${MAIN_OUT}/audit/audit_summary.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(f"main audit summary is missing: {path}")
data = json.loads(path.read_text())
if data.get("gate_status") != "pass":
    raise SystemExit(f"main audit did not pass: {data.get('gate_status')}")
print("[protocol] main result audit passed")
PY

if [[ "${MAIN_ABL_RC}" -ne 0 ]]; then
  echo "[protocol] main/ablation driver returned ${MAIN_ABL_RC}; validated main and available ablation rows are retained"
  echo "[protocol] timeout/right-censored ablation rows do not prevent First-use and natural-workflow supplements"
fi

echo "[3/5] wait for uncontended First-use measurement"
"${COMMON_PY}" benchmarking/wait_for_idle_gpu.py \
  --gpu "${PAPER_GPU}" --stable-checks 3 --poll-seconds 10

echo "[4/5] EGGPU First-use versus Steady-state"
env -u CFLAGS -u CPPFLAGS -u CXXFLAGS \
    -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u CPATH -u LIBRARY_PATH \
    "${COMMON_ENV[@]}" \
    LD_LIBRARY_PATH="${CUDA_LD_PATH}" \
    "${COMMON_PY}" benchmarking/run_eggpu_first_use.py \
      --gpu "${PAPER_GPU}" \
      --steady-dir "${MAIN_OUT}" \
      --out-dir "${FIRST_OUT}" \
      --repeat "${PAPER_REPEAT}" \
      --memory-repeat 1 \
      --library-timeout "${PAPER_TIMEOUT}" \
      --inter-run-cooldown 1.0 \
      --easygraph-repo "${EG_REPO}" \
      --datasets all \
      --functions all \
      --sssp-sources 8 \
      --bc-sources 16 \
      --pr-alpha 0.75 \
      --pr-eps 1e-6 \
      --pr-max-iter 200

echo "[5/5] Graphalytics-inspired natural same-graph workflow"
"${COMMON_PY}" benchmarking/wait_for_idle_gpu.py \
  --gpu "${PAPER_GPU}" --stable-checks 3 --poll-seconds 10
env -u CFLAGS -u CPPFLAGS -u CXXFLAGS \
    -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u CPATH -u LIBRARY_PATH \
    "${COMMON_ENV[@]}" \
    LD_LIBRARY_PATH="${CUDA_LD_PATH}" \
    "${COMMON_PY}" benchmarking/run_eggpu_natural_workflow.py \
      --gpu "${PAPER_GPU}" \
      --first-use-dir "${FIRST_OUT}" \
      --out-dir "${NATURAL_OUT}" \
      --datasets all \
      --repeat "${PAPER_REPEAT}" \
      --timeout "$((PAPER_TIMEOUT * 3))" \
      --cooldown 1.0 \
      --bfs-sources 8 \
      --pr-alpha 0.75 \
      --pr-eps 1e-6 \
      --pr-max-iter 200

"${COMMON_PY}" - "${PAPER_RUN_TS}" "${PAPER_GPU}" "${SECONDARY_GPU}" \
  "${MAIN_OUT}" "${ABL_OUT}" "${FIRST_OUT}" "${NATURAL_OUT}" \
  "${MARKER_ENABLED}" "${MARKER_DECISION}" <<'PY'
import json
import sys
from pathlib import Path

(
    run_ts,
    gpu,
    secondary,
    main_out,
    ablation_out,
    first_out,
    natural_out,
    marker_enabled,
    marker_decision,
) = sys.argv[1:]
manifest = {
    "schema_version": 1,
    "run_timestamp": run_ts,
    "authoritative_gpu": gpu,
    "secondary_gpu_policy": f"GPU {secondary} intentionally idle during timing-bearing experiments",
    "execution_order": ["gate", "steady_main", "ablation", "first_use", "natural_workflow"],
    "parallel_timing": False,
    "visibility_marker_enabled": marker_enabled == "TRUE",
    "visibility_marker_decision": marker_decision,
    "timing_estimator": "arithmetic_mean",
    "error_bar": "sample_standard_deviation",
    "scaling_status": "deferred",
    "artifacts": {
        "steady_main": main_out,
        "ablation": ablation_out,
        "first_use": first_out,
        "natural_workflow": natural_out,
    },
    "natural_workflow_comparisons": {
        "whole_workflow": str(Path(natural_out) / "natural_workflow_vs_isolated.csv"),
        "all_calls": str(Path(natural_out) / "natural_workflow_per_call_vs_first_use.csv"),
        "reuse_beneficiaries": str(Path(natural_out) / "natural_workflow_reuse_beneficiaries.csv"),
        "reuse_beneficiary_summary": str(Path(natural_out) / "natural_workflow_reuse_beneficiary_summary.csv"),
    },
}
path = Path("benchmarking/results") / f"paper_experiment_manifest_{run_ts}.json"
path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(f"Manifest: {path}")
PY

echo "Complete paper experiment protocol finished."
echo "Main: ${MAIN_OUT}"
echo "Ablation: ${ABL_OUT}"
echo "First-use: ${FIRST_OUT}"
echo "Natural workflow: ${NATURAL_OUT}"
