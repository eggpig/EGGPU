#!/usr/bin/env bash
set -euo pipefail

# Rerun only the 11 regular-matrix PageRank cells whose original E2E timer
# included generic Python signature/keyword adaptation.  This launcher is
# deliberately protocol-driven: every cell is rerun and later adopted
# regardless of whether the corrected timing is faster or slower.

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
evaluation_root="$(cd -- "${script_dir}/.." && pwd)"
project_root="$(cd -- "${evaluation_root}/.." && pwd)"
runtime_root="${project_root}/.codex-tmp/eggpu_v15_runtime_20260730_final"
python_bin="${EGGPU_PYTHON:-/home/dataset-assist-0/einwang/conda_cache/conda_env/EGGPU/bin/python}"
result_root="${evaluation_root}/benchmarking/results"
log_root="${result_root}/final_v15_pagerank_protocol_correction_logs_20260730"
worker_source="${script_dir}/library_baselines.py"
protocol_verifier="${script_dir}/verify_v15_pagerank_worker_protocol_20260730.py"
correction_assembler="${script_dir}/assemble_v15_pagerank_protocol_correction_20260730.py"
mkdir -p -- "${log_root}"

expected_native_sha256="d2a93a3da0ffd0c554c9dab52c9d05d8c1f5f5b4904bcfd19fd95b695bb06eaf"
expected_runtime_sha256="1be28aeb48374c65642b26f7233a7764f66fb3fb8003e6434e04744e08b643fb"

all_groups=(
  "ca_hepth"
  "lastfm_web"
  "p2p_er"
  "hepph_slashdot"
  "youtube"
  "condmat"
  "epinions"
  "enron"
)

declare -A group_datasets=(
  [ca_hepth]="ca-HepTh"
  [lastfm_web]="LastFM,web-NotreDame"
  [p2p_er]="p2p-Gnutella04,ER-100k"
  [hepph_slashdot]="ca-HepPh,soc-Slashdot0811"
  [youtube]="com-youtube"
  [condmat]="ca-CondMat"
  [epinions]="soc-Epinions1"
  [enron]="email-Enron"
)
declare -A group_default_gpu=(
  [ca_hepth]="0"
  [lastfm_web]="1"
  [p2p_er]="2"
  [hepph_slashdot]="3"
  [youtube]="4"
  [condmat]="5"
  [epinions]="6"
  [enron]="7"
)
declare -A group_gpu_env=(
  [ca_hepth]="V15_PR_GPU_CA_HEPTH"
  [lastfm_web]="V15_PR_GPU_LASTFM_WEB"
  [p2p_er]="V15_PR_GPU_P2P_ER"
  [hepph_slashdot]="V15_PR_GPU_HEPPH_SLASHDOT"
  [youtube]="V15_PR_GPU_YOUTUBE"
  [condmat]="V15_PR_GPU_CONDMAT"
  [epinions]="V15_PR_GPU_EPINIONS"
  [enron]="V15_PR_GPU_ENRON"
)

usage() {
  cat <<'EOF'
Usage:
  V15_PR_CORRECTION_GROUPS=GROUP[,GROUP...] \
    bash run_v15_pagerank_protocol_correction_20260730.sh

Groups:
  ca_hepth, lastfm_web, p2p_er, hepph_slashdot,
  youtube, condmat, epinions, enron

The default is all groups.  A group can be assigned to a different free GPU
with its matching variable:
  V15_PR_GPU_CA_HEPTH
  V15_PR_GPU_LASTFM_WEB
  V15_PR_GPU_P2P_ER
  V15_PR_GPU_HEPPH_SLASHDOT
  V15_PR_GPU_YOUTUBE
  V15_PR_GPU_CONDMAT
  V15_PR_GPU_EPINIONS
  V15_PR_GPU_ENRON

Example for the first three currently free GPUs:
  V15_PR_CORRECTION_GROUPS=lastfm_web,p2p_er,youtube \
  V15_PR_GPU_LASTFM_WEB=1 V15_PR_GPU_P2P_ER=2 V15_PR_GPU_YOUTUBE=4 \
    bash run_v15_pagerank_protocol_correction_20260730.sh

Each group has a stable, GPU-independent output directory.  Separate partial
invocations can therefore fill the complete 11-dataset correction set without
waiting for Stage 2 anchors.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if (($#)); then
  echo "Unexpected positional arguments: $*" >&2
  usage >&2
  exit 2
fi

selected_raw="${V15_PR_CORRECTION_GROUPS:-all}"
selected_groups=()
if [[ "${selected_raw}" == "all" ]]; then
  selected_groups=("${all_groups[@]}")
else
  IFS=',' read -r -a selected_groups <<<"${selected_raw}"
fi
if ((${#selected_groups[@]} == 0)); then
  echo "V15_PR_CORRECTION_GROUPS selected no groups" >&2
  exit 2
fi

declare -A selected_seen=()
declare -A selected_gpu=()
declare -A selected_tag=()
declare -A selected_log=()
declare -A gpu_seen=()
for group in "${selected_groups[@]}"; do
  if [[ -z "${group_datasets[${group}]+present}" ]]; then
    echo "Unknown PageRank correction group: ${group}" >&2
    usage >&2
    exit 2
  fi
  if [[ -n "${selected_seen[${group}]+present}" ]]; then
    echo "Duplicate PageRank correction group: ${group}" >&2
    exit 2
  fi
  selected_seen["${group}"]=1
  gpu_variable="${group_gpu_env[${group}]}"
  gpu="${!gpu_variable:-${group_default_gpu[${group}]}}"
  if [[ ! "${gpu}" =~ ^[0-7]$ ]]; then
    echo "${gpu_variable} must be an integer in [0, 7]" >&2
    exit 2
  fi
  if [[ -n "${gpu_seen[${gpu}]+present}" ]]; then
    echo "Selected correction groups may not concurrently share GPU ${gpu}" >&2
    exit 2
  fi
  gpu_seen["${gpu}"]="${group}"
  selected_gpu["${group}"]="${gpu}"
  selected_tag["${group}"]="final_v15_pagerank_protocol_correction_${group}_20260730"
  selected_log["${group}"]="${group}_gpu${gpu}.log"
done

if [[ ! -x "${python_bin}" ]]; then
  echo "EGGPU Python is not executable: ${python_bin}" >&2
  exit 1
fi
if [[ ! -d "${runtime_root}" ]]; then
  echo "Frozen V15 runtime is missing: ${runtime_root}" >&2
  exit 1
fi
for group in "${selected_groups[@]}"; do
  tag="${selected_tag[${group}]}"
  if [[ -e "${result_root}/${tag}" ]]; then
    echo "Refusing to overwrite correction result: ${result_root}/${tag}" >&2
    exit 1
  fi
done

selection_tag="${selected_raw//,/_}"
attestation_before="${log_root}/pagerank_protocol_attestation.${selection_tag}.preflight.json"
attestation_after="${log_root}/pagerank_protocol_attestation.${selection_tag}.postflight.json"
"${python_bin}" "${protocol_verifier}" \
  --source "${worker_source}" \
  --output "${attestation_before}" \
  >"${log_root}/pagerank_protocol_${selection_tag}_preflight.stdout.jsonl"

"${python_bin}" - \
  "${runtime_root}" \
  "${expected_native_sha256}" \
  "${expected_runtime_sha256}" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]).parent.parent / "EG_Evaluation" / "benchmarking"))
from easygraph_runtime_provenance import collect_runtime_repository_provenance

runtime = Path(sys.argv[1]).resolve(strict=True)
identity = collect_runtime_repository_provenance(runtime)
if identity["native_sha256"] != sys.argv[2]:
    raise SystemExit("frozen V15 native SHA-256 differs")
snapshot = identity["runtime_python_snapshot"]
if snapshot["digest"] != sys.argv[3]:
    raise SystemExit("frozen V15 Python snapshot SHA-256 differs")
if snapshot["package_is_symlink"] is not False:
    raise SystemExit("frozen V15 Python package must not be a symlink")
PY

run_page_rank() {
  local gpu="$1"
  local datasets="$2"
  local tag="$3"
  local log_name="$4"
  env \
    MPLCONFIGDIR="/tmp/eggpu_v15_pr_correction_mpl_gpu${gpu}" \
    EGGPU_CHILD_PYTHON="${python_bin}" \
    EASYGRAPH_GPU_RESULT_CACHE=FALSE \
    "${python_bin}" "${evaluation_root}/benchmarking/run_full_baselines.py" \
      --gpu "${gpu}" \
      --out-dir "${result_root}/${tag}" \
      --datasets "${datasets}" \
      --functions PageRank \
      --baselines EGGPU \
      --repeat 5 \
      --warmup 2 \
      --easygraph-warmup 2 \
      --eggpu-execution-protocol steady-state \
      --measurement-mode timing \
      --library-timeout 100 \
      --pr-alpha 0.75 \
      --pr-eps 1e-6 \
      --pr-max-iter 200 \
      --sssp-sources 8 \
      --bc-sources 16 \
      --closeness-sources 0 \
      --inter-run-cooldown 0.2 \
      --easygraph-repo "${runtime_root}" \
    >"${log_root}/${log_name}" 2>&1
}

pids=()
for group in "${selected_groups[@]}"; do
  run_page_rank \
    "${selected_gpu[${group}]}" \
    "${group_datasets[${group}]}" \
    "${selected_tag[${group}]}" \
    "${selected_log[${group}]}" &
  pids+=("$!")
  if ((${#pids[@]} < ${#selected_groups[@]})); then
    sleep 1
  fi
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "At least one PageRank correction worker failed" >&2
  exit "${status}"
fi

# The source-level contract must be identical before and after all measured
# children.  The copied attestation therefore binds every result directory to
# exactly the worker implementation checked before launch.
"${python_bin}" "${protocol_verifier}" \
  --source "${worker_source}" \
  --expect-attestation "${attestation_before}" \
  --output "${attestation_after}" \
  >"${log_root}/pagerank_protocol_${selection_tag}_postflight.stdout.jsonl"

for group in "${selected_groups[@]}"; do
  tag="${selected_tag[${group}]}"
  cp -- \
    "${attestation_after}" \
    "${result_root}/${tag}/pagerank_protocol_attestation.json"
done

batch_validation_args=()
for group in "${selected_groups[@]}"; do
  batch_validation_args+=(
    --correction-result-dir
    "${result_root}/${selected_tag[${group}]}"
  )
done
"${python_bin}" "${correction_assembler}" \
  --validate-correction-only \
  "${batch_validation_args[@]}" \
  >"${log_root}/pagerank_protocol_${selection_tag}_batch_validation.stdout.jsonl"

echo "V15 PageRank protocol correction completed for groups: ${selected_groups[*]}"
echo "Adoption policy: protocol-valid replacements are unconditional; timings are not compared for selection."
