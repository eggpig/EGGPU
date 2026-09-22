#!/usr/bin/env bash
set -euo pipefail

EVAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${EVAL_ROOT}/.." && pwd)"
cd "${EVAL_ROOT}"

SUPPLEMENT_GPU="${SUPPLEMENT_GPU:-0}"
PAPER_REPEAT="${PAPER_REPEAT:-5}"
PAPER_TIMEOUT="${PAPER_TIMEOUT:-100}"
SUPPLEMENT_TS="${SUPPLEMENT_TS:-$(date +%Y%m%d_%H%M%S)}"
COMMON_PY="${COMMON_PY:-/home/dataset-assist-0/einwang/conda_cache/conda_env/EGGPU/bin/python}"
CUDA_ROOT="${EGGPU_CUDA_ROOT:-/home/dataset-assist-0/einwang/conda_cache/conda_env/sglang}"
EG_REPO="${EG_REPO:-${WORKSPACE_ROOT}/Easy-Graph}"
STEADY_DIR="${STEADY_DIR:-benchmarking/results/full_eval_gpu0_20260712_144427_repeat5_splitmem_exact_mst_strict_nxcg}"
ABLATION_DIR="${ABLATION_DIR:-benchmarking/results/ablation_gpu0_20260712_144427_repeat5_strengthened}"
RUN_CLOSENESS_SUPPLEMENT="${RUN_CLOSENESS_SUPPLEMENT:-TRUE}"
REPAIR_FIRST_USE="${REPAIR_FIRST_USE:-TRUE}"

FIRST_RAW_OUT="${FIRST_RAW_OUT:-${FIRST_OUT:-benchmarking/results/eggpu_first_use_gpu${SUPPLEMENT_GPU}_${SUPPLEMENT_TS}_repeat${PAPER_REPEAT}}}"
FIRST_REPAIRED_OUT="${FIRST_REPAIRED_OUT:-benchmarking/results/eggpu_first_use_repaired_gpu${SUPPLEMENT_GPU}_${SUPPLEMENT_TS}_repeat${PAPER_REPEAT}}"
NATURAL_OUT="${NATURAL_OUT:-benchmarking/results/eggpu_natural_workflow_gpu${SUPPLEMENT_GPU}_${SUPPLEMENT_TS}_repeat${PAPER_REPEAT}}"
CLOSENESS_OUT="${CLOSENESS_OUT:-benchmarking/results/closeness_large_gpu${SUPPLEMENT_GPU}_${SUPPLEMENT_TS}}"
PAPER_OUT="${PAPER_OUT:-benchmarking/results/eggpu_reuse_supplement_gpu${SUPPLEMENT_GPU}_${SUPPLEMENT_TS}/paper_artifacts}"
AUDIT_OUT="${AUDIT_OUT:-benchmarking/results/eggpu_reuse_supplement_gpu${SUPPLEMENT_GPU}_${SUPPLEMENT_TS}/audit}"
TOP_LOG="benchmarking/results/missing_paper_supplements_${SUPPLEMENT_TS}.console.log"
CUDA_LD_PATH="${CUDA_ROOT}/lib:${CUDA_ROOT}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
REUSE_FIRST_USE="${REUSE_FIRST_USE:-FALSE}"

exec > >(tee -a "${TOP_LOG}") 2>&1

if [[ "${PAPER_REPEAT}" -lt 2 ]]; then
  echo "PAPER_REPEAT must be at least 2." >&2
  exit 2
fi

"${COMMON_PY}" - "${STEADY_DIR}" "${ABLATION_DIR}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
ablation = Path(sys.argv[2]).resolve()
required = [root / "results_long.csv", root / "results_samples.csv", root / "run_metadata.json"]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"steady-state source is incomplete: {missing}")
audit = root / "audit" / "audit_summary.json"
if not audit.exists() or json.loads(audit.read_text()).get("gate_status") != "pass":
    raise SystemExit(f"steady-state audit is missing or did not pass: {audit}")
print(f"[supplement] validated steady-state source: {root}")

ablation_csv = ablation / "ablation_all.csv"
if not ablation_csv.exists():
    raise SystemExit(f"missing ablation source: {ablation_csv}")
with ablation_csv.open(newline="") as handle:
    failed = [row for row in csv.DictReader(handle) if row.get("status") == "failed"]
if failed:
    raise SystemExit(f"ablation source contains {len(failed)} failed rows: {ablation}")
required_patterns = [
    "workflow_*_canonical_full.csv",
    "workflow_*_canonical_no_cpp_graph_cache.csv",
    "workflow_*_canonical_no_graph_context.csv",
    "return_*.csv",
    "layout_*.csv",
]
missing_patterns = [pattern for pattern in required_patterns if not any(ablation.glob(pattern))]
if missing_patterns:
    raise SystemExit(f"ablation source lacks core modules: {missing_patterns}")
print(f"[supplement] validated ablation source: {ablation}")
PY

COMMON_ENV=(
  "CUDA_VISIBLE_DEVICES=${SUPPLEMENT_GPU}"
  "EGGPU_MONITOR_GPU_INDEX=${SUPPLEMENT_GPU}"
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
  "EGGPU_GPU_VISIBILITY_MARKER=TRUE"
  "EGGPU_GPU_VISIBILITY_MARKER_MB=256"
  "EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB=AUTO"
  "PYTHONPATH=${EG_REPO}:${PYTHONPATH:-}"
)

echo "[1/5] EGGPU First-use versus steady-state"
FIRST_USE_EXTRA_ARGS=()
if [[ "${REUSE_FIRST_USE^^}" == "TRUE" ]]; then
  FIRST_USE_EXTRA_ARGS+=(--reuse-existing)
else
  "${COMMON_PY}" benchmarking/wait_for_idle_gpu.py \
    --gpu "${SUPPLEMENT_GPU}" --stable-checks 3 --poll-seconds 10
fi
env -u CFLAGS -u CPPFLAGS -u CXXFLAGS \
    -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u CPATH -u LIBRARY_PATH \
    "${COMMON_ENV[@]}" \
    LD_LIBRARY_PATH="${CUDA_LD_PATH}" \
    "${COMMON_PY}" benchmarking/run_eggpu_first_use.py \
      --gpu "${SUPPLEMENT_GPU}" \
      --steady-dir "${STEADY_DIR}" \
      --out-dir "${FIRST_RAW_OUT}" \
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
      --pr-max-iter 200 \
      "${FIRST_USE_EXTRA_ARGS[@]}"

FIRST_EVIDENCE_OUT="${FIRST_RAW_OUT}"
if [[ "${REPAIR_FIRST_USE^^}" == "TRUE" ]]; then
  echo "[2/5] Targeted First-use completion"
  "${COMMON_PY}" benchmarking/wait_for_idle_gpu.py \
    --gpu "${SUPPLEMENT_GPU}" --stable-checks 3 --poll-seconds 10
  env -u CFLAGS -u CPPFLAGS -u CXXFLAGS \
      -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u CPATH -u LIBRARY_PATH \
      "${COMMON_ENV[@]}" \
      LD_LIBRARY_PATH="${CUDA_LD_PATH}" \
      "${COMMON_PY}" benchmarking/repair_eggpu_first_use.py \
        --base-dir "${FIRST_RAW_OUT}" \
        --steady-dir "${STEADY_DIR}" \
        --out-dir "${FIRST_REPAIRED_OUT}" \
        --gpu "${SUPPLEMENT_GPU}" \
        --repeat "${PAPER_REPEAT}" \
        --memory-repeat 1 \
        --library-timeout "${PAPER_TIMEOUT}" \
        --inter-run-cooldown 1.0 \
        --easygraph-repo "${EG_REPO}" \
        --sssp-sources 8 \
        --bc-sources 16 \
        --pr-alpha 0.75 \
        --pr-eps 1e-6 \
        --pr-max-iter 200
  FIRST_EVIDENCE_OUT="${FIRST_REPAIRED_OUT}"
else
  echo "[2/5] Targeted First-use completion disabled"
fi

echo "[3/5] Natural same-graph workflow with matched controls"
"${COMMON_PY}" benchmarking/wait_for_idle_gpu.py \
  --gpu "${SUPPLEMENT_GPU}" --stable-checks 3 --poll-seconds 10
env -u CFLAGS -u CPPFLAGS -u CXXFLAGS \
    -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u CPATH -u LIBRARY_PATH \
    "${COMMON_ENV[@]}" \
    LD_LIBRARY_PATH="${CUDA_LD_PATH}" \
    "${COMMON_PY}" benchmarking/run_eggpu_natural_workflow.py \
      --gpu "${SUPPLEMENT_GPU}" \
      --first-use-dir "${FIRST_EVIDENCE_OUT}" \
      --out-dir "${NATURAL_OUT}" \
      --datasets all \
      --repeat "${PAPER_REPEAT}" \
      --timeout "$((PAPER_TIMEOUT * 3))" \
      --cooldown 1.0 \
      --bfs-sources 8 \
      --pr-alpha 0.75 \
      --pr-eps 1e-6 \
      --pr-max-iter 200

echo "[4/5] Large-Closeness supplement"
if [[ "${RUN_CLOSENESS_SUPPLEMENT^^}" == "TRUE" ]]; then
  "${COMMON_PY}" benchmarking/wait_for_idle_gpu.py \
    --gpu "${SUPPLEMENT_GPU}" --stable-checks 3 --poll-seconds 10
  env -u CFLAGS -u CPPFLAGS -u CXXFLAGS \
      -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u CPATH -u LIBRARY_PATH \
      "${COMMON_ENV[@]}" \
      LD_LIBRARY_PATH="${CUDA_LD_PATH}" \
      "${COMMON_PY}" benchmarking/run_closeness_large_supplement.py \
        "${STEADY_DIR}" \
        --easygraph-repo "${EG_REPO}" \
        --sources 16 \
        --repeat "${PAPER_REPEAT}" \
        --gpu "${SUPPLEMENT_GPU}" \
        --timeout "${PAPER_TIMEOUT}" \
        --python "${COMMON_PY}" \
        --out-dir "${CLOSENESS_OUT}"
else
  echo "Large-Closeness supplement disabled; final supplement audit will not run."
fi

echo "[5/5] Paper artifacts and closed-loop audit"
"${COMMON_PY}" benchmarking/generate_first_use_workflow_artifacts.py \
  --first-use-dir "${FIRST_EVIDENCE_OUT}" \
  --natural-dir "${NATURAL_OUT}" \
  --out-dir "${PAPER_OUT}"

if [[ "${RUN_CLOSENESS_SUPPLEMENT^^}" == "TRUE" ]]; then
  "${COMMON_PY}" benchmarking/audit_paper_supplements.py \
    --main "${STEADY_DIR}" \
    --ablation "${ABLATION_DIR}" \
    --first-use "${FIRST_EVIDENCE_OUT}" \
    --workflow "${NATURAL_OUT}" \
    --closeness "${CLOSENESS_OUT}" \
    --out-dir "${AUDIT_OUT}"
fi

"${COMMON_PY}" - "${SUPPLEMENT_TS}" "${SUPPLEMENT_GPU}" "${STEADY_DIR}" \
  "${ABLATION_DIR}" "${FIRST_RAW_OUT}" "${FIRST_EVIDENCE_OUT}" \
  "${NATURAL_OUT}" "${CLOSENESS_OUT}" "${PAPER_OUT}" "${AUDIT_OUT}" \
  "${RUN_CLOSENESS_SUPPLEMENT}" <<'PY'
import json
import sys
from pathlib import Path

(
    run_ts,
    gpu,
    steady,
    ablation,
    first_raw,
    first_evidence,
    natural,
    closeness,
    paper,
    audit,
    run_closeness,
) = sys.argv[1:]
manifest = {
    "schema_version": 2,
    "run_timestamp": run_ts,
    "gpu": gpu,
    "steady_state_source": str(Path(steady).resolve()),
    "ablation_source": str(Path(ablation).resolve()),
    "first_use_raw": str(Path(first_raw).resolve()),
    "first_use_evidence": str(Path(first_evidence).resolve()),
    "natural_workflow": str(Path(natural).resolve()),
    "large_closeness": str(Path(closeness).resolve()) if run_closeness.upper() == "TRUE" else None,
    "paper_artifacts": str(Path(paper).resolve()),
    "supplement_audit": str(Path(audit).resolve()) if run_closeness.upper() == "TRUE" else None,
    "timing_estimator": "arithmetic_mean",
    "error_bar": "sample_standard_deviation",
    "workflow_control": "same_run_same_gpu_matched_isolated_first_use",
    "full_baselines_rerun": False,
}
path = Path("benchmarking/results") / f"missing_paper_supplements_{run_ts}.json"
path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(f"Manifest: {path}")
PY

echo "Supplement protocol complete."
echo "First-use raw: ${FIRST_RAW_OUT}"
echo "First-use evidence: ${FIRST_EVIDENCE_OUT}"
echo "Natural workflow: ${NATURAL_OUT}"
echo "Paper artifacts: ${PAPER_OUT}"
if [[ "${RUN_CLOSENESS_SUPPLEMENT^^}" == "TRUE" ]]; then
  echo "Large Closeness: ${CLOSENESS_OUT}"
  echo "Supplement audit: ${AUDIT_OUT}"
fi
