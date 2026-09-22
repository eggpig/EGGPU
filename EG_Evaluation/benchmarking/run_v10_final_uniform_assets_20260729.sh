#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
eval_root="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${eval_root}/.." && pwd)"
writing_root="${repo_root}/writing"
python_bin="${EGGPU_PYTHON:-/home/dataset-assist-0/einwang/conda_cache/conda_env/EGGPU/bin/python}"
candidate_sha="4e59adea712b69a6a826d91029df31ce30279a47f83b13e10d582062e0c7f022"
runtime_sha="84d91c4023be9252cc6860a2129ff07ea081d4663d523a521ac726b15ecf02d9"

source_assets="${writing_root}/EGGPU_FINAL_EXPERIMENT_ASSETS_13_V10_LEDGER_20260729"
supplement_assets="${writing_root}/EGGPU_FINAL_EXPERIMENT_ASSETS_13_V9_UNIFORM_20260729"
historical_comparison="${supplement_assets}/historical_eggpu_pair_details.csv"
overlay_audit="${source_assets}/EGGPU_V10_OVERLAY_AUDIT_20260729.json"
workflow_result="${V10_WORKFLOW_RESULT:-${eval_root}/benchmarking/results/cumulative_workflow5_v10_r1_gpu4_20260729_repeat5}"
final_dir="${writing_root}/EGGPU_FINAL_EXPERIMENT_ASSETS_13_V10_UNIFORM_20260729"

while (($#)); do
  case "$1" in
    --workflow-result)
      workflow_result="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

for required in \
  "${source_assets}/final_13_cell_outcome_ledger.csv" \
  "${overlay_audit}" \
  "${overlay_audit%.json}.cells.csv" \
  "${overlay_audit%.json}.memory_cells.csv" \
  "${supplement_assets}/VLDB_UNIFORM_ASSET_MANIFEST.json" \
  "${historical_comparison}" \
  "${workflow_result}/metadata.json" \
  "${workflow_result}/cumulative_workflow_summary.csv" \
  "${workflow_result}/cumulative_workflow_samples.csv"; do
  if [[ ! -s "${required}" ]]; then
    echo "required V10_UNIFORM input is missing or empty: ${required}" >&2
    exit 2
  fi
done
if [[ -e "${final_dir}" ]]; then
  echo "refusing to overwrite existing V10_UNIFORM publication: ${final_dir}" >&2
  exit 2
fi

stage_dir="$(mktemp -d "${writing_root}/.EGGPU_V10_UNIFORM.stage.XXXXXX")"
mpl_dir="$(mktemp -d /tmp/eggpu-v10-uniform-mpl.XXXXXX)"
cleanup() {
  status=$?
  if [[ -n "${stage_dir:-}" && -d "${stage_dir}" ]]; then
    rm -rf -- "${stage_dir}"
  fi
  if [[ -n "${mpl_dir:-}" && -d "${mpl_dir}" ]]; then
    rm -rf -- "${mpl_dir}"
  fi
  exit "${status}"
}
trap cleanup EXIT INT TERM

echo "[V10_UNIFORM 1/8] regenerate V10 tables, figures, raw5 evidence, and unified memory assets"
MPLCONFIGDIR="${mpl_dir}" "${python_bin}" \
  "${script_dir}/prepare_vldb_uniform_assets.py" \
  --source-assets "${source_assets}" \
  --supplement-assets "${supplement_assets}" \
  --historical-comparison "${historical_comparison}" \
  --output-dir "${stage_dir}" \
  --candidate-sha256 "${candidate_sha}" \
  --overlay-audit "${overlay_audit}" \
  --memory-provenance-mode unified-candidate \
  --release-label V10 >/dev/null

echo "[V10_UNIFORM 2/8] audit 60 V10 workflow sample files and regenerate arithmetic-mean supplement"
MPLCONFIGDIR="${mpl_dir}" "${python_bin}" \
  "${script_dir}/prepare_v10_workflow_assets.py" \
  --workflow-result "${workflow_result}" \
  --output-dir "${stage_dir}" \
  --candidate-sha256 "${candidate_sha}" \
  --runtime-sha256 "${runtime_sha}" >/dev/null

echo "[V10_UNIFORM 3/8] generate V9-to-V10 key and cell-level metric diff"
"${python_bin}" "${script_dir}/generate_v9_v10_metric_diff.py" \
  --v9-assets "${supplement_assets}" \
  --v10-assets "${stage_dir}" \
  --output-dir "${stage_dir}" \
  --candidate-sha256 "${candidate_sha}" >/dev/null

echo "[V10_UNIFORM 4/8] initialize relocation-safe paper-sync-pending manifest"
"${python_bin}" "${script_dir}/finalize_v10_uniform_assets_stage.py" initialize \
  --stage-dir "${stage_dir}" \
  --final-dir "${final_dir}" \
  --repo-root "${repo_root}" \
  --candidate-sha256 "${candidate_sha}" \
  --runtime-sha256 "${runtime_sha}" >/dev/null

echo "[V10_UNIFORM 5/8] run strict assets-only paper-asset audit"
"${python_bin}" "${script_dir}/audit_v10_paper_claim_consistency.py" \
  --assets-only \
  --assets-dir "${stage_dir}" \
  --max-over-median-limit 5 \
  --median-over-min-limit 3 \
  --json-output "${stage_dir}/V10_ASSET_ONLY_AUDIT.json"

echo "[V10_UNIFORM 6/8] seal release manifest and complete SHA256 inventories"
"${python_bin}" "${script_dir}/finalize_v10_uniform_assets_stage.py" seal \
  --stage-dir "${stage_dir}" \
  --final-dir "${final_dir}" \
  --repo-root "${repo_root}" \
  --candidate-sha256 "${candidate_sha}" \
  --runtime-sha256 "${runtime_sha}" \
  --asset-audit "${stage_dir}/V10_ASSET_ONLY_AUDIT.json" >/dev/null

echo "[V10_UNIFORM 7/8] independently verify timing, memory, workflow, diff, manifest, and hashes"
"${python_bin}" "${script_dir}/verify_v10_uniform_assets_gate.py" \
  --assets-dir "${stage_dir}" \
  --published-dir "${final_dir}" \
  --candidate-sha256 "${candidate_sha}" \
  --runtime-sha256 "${runtime_sha}"

echo "[V10_UNIFORM 8/8] atomically publish same-filesystem staging directory"
mv -- "${stage_dir}" "${final_dir}"
stage_dir=""
rm -rf -- "${mpl_dir}"
mpl_dir=""
trap - EXIT INT TERM
echo "V10_UNIFORM publication: ${final_dir}"
