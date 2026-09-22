#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
evaluation_root="$(cd -- "${script_dir}/.." && pwd)"
project_root="$(cd -- "${evaluation_root}/.." && pwd)"
writing_root="${project_root}/writing"
python_bin="${EGGPU_PYTHON:-/home/dataset-assist-0/einwang/conda_cache/conda_env/EGGPU/bin/python}"

candidate_sha256="4e59adea712b69a6a826d91029df31ce30279a47f83b13e10d582062e0c7f022"
runtime_sha256="84d91c4023be9252cc6860a2129ff07ea081d4663d523a521ac726b15ecf02d9"
base_ledger="${writing_root}/EGGPU_FINAL_EXPERIMENT_ASSETS_13_V9_UNIFORM_20260729/final_13_cell_outcome_ledger.csv"
final_dir="${writing_root}/EGGPU_FINAL_EXPERIMENT_ASSETS_13_V10_LEDGER_20260729"

if [[ ! -x "${python_bin}" ]]; then
  echo "EGGPU Python is not executable: ${python_bin}" >&2
  exit 1
fi
if [[ ! -f "${base_ledger}" ]]; then
  echo "V9 base ledger is missing: ${base_ledger}" >&2
  exit 1
fi
if [[ -e "${final_dir}" ]]; then
  echo "Refusing to overwrite V10 release directory: ${final_dir}" >&2
  exit 1
fi

staging_dir="$(mktemp -d "${writing_root}/.EGGPU_V10_LEDGER.staging.XXXXXX")"
cleanup() {
  rm -rf -- "${staging_dir}"
}
trap cleanup EXIT

timing_main_dirs=(
  "${evaluation_root}/benchmarking/results/final_v10_r1_uniform_gpu0_20260729"
  "${evaluation_root}/benchmarking/results/final_v10_r1_uniform_gpu1_20260729"
  "${evaluation_root}/benchmarking/results/final_v10_r1_uniform_gpu2_20260729"
  "${evaluation_root}/benchmarking/results/final_v10_r1_uniform_gpu3_20260729"
  "${evaluation_root}/benchmarking/results/final_v10_r1_uniform_gpu5_20260729"
  "${evaluation_root}/benchmarking/results/final_v10_r1_uniform_gpu6_20260729"
  "${evaluation_root}/benchmarking/results/final_v10_r1_uniform_gpu7_email_20260729"
  "${evaluation_root}/benchmarking/results/final_v10_r1_uniform_closeness_ER-100k_gpu2_20260729"
  "${evaluation_root}/benchmarking/results/final_v10_r1_uniform_closeness_com-youtube_gpu7_20260729"
  "${evaluation_root}/benchmarking/results/final_v10_r1_uniform_closeness_soc-Slashdot0811_gpu2_20260729"
  "${evaluation_root}/benchmarking/results/final_v10_r1_uniform_closeness_web-NotreDame_gpu1_20260729"
)
anchor_dirs=(
  "${evaluation_root}/benchmarking/results/final_v10_r1_uniform_com_orkut_gpu4_20260729"
  "${evaluation_root}/benchmarking/results/final_v10_r1_uniform_gap_twitter_part_centrality_gpu7_20260729"
  "${evaluation_root}/benchmarking/results/final_v10_r1_uniform_gap_twitter_part_connectivity_gpu6_20260729"
  "${evaluation_root}/benchmarking/results/final_v10_r1_uniform_gap_twitter_part_paths_gpu5_20260729"
  "${evaluation_root}/benchmarking/results/final_v10_r1_uniform_gap_twitter_part_projection_structural_gpu3_20260729"
)
memory_main_dirs=(
  "${evaluation_root}/benchmarking/results/final_v10_r1_uniform_memory_gpu1_20260729"
  "${evaluation_root}/benchmarking/results/final_v10_r1_uniform_memory_gpu2_20260729"
  "${evaluation_root}/benchmarking/results/final_v10_r1_uniform_memory_gpu5_20260729"
  "${evaluation_root}/benchmarking/results/final_v10_r1_uniform_memory_gpu6_20260729"
  "${evaluation_root}/benchmarking/results/final_v10_r1_uniform_memory_gpu7_email_20260729"
  "${evaluation_root}/benchmarking/results/final_v10_r1_uniform_memory_gpu0_cah_youtube_r2_20260729"
  "${evaluation_root}/benchmarking/results/final_v10_r1_uniform_memory_gpu2_hepph_slashdot_20260729"
  "${evaluation_root}/benchmarking/results/final_v10_r1_uniform_memory_closeness4_gpu1_20260729"
)

for result_dir in \
  "${timing_main_dirs[@]}" \
  "${anchor_dirs[@]}" \
  "${memory_main_dirs[@]}"; do
  if [[ ! -d "${result_dir}" ]]; then
    echo "Required V10 result directory is missing: ${result_dir}" >&2
    exit 1
  fi
done

timing_args=()
for result_dir in "${timing_main_dirs[@]}"; do
  timing_args+=(--main-result-dir "${result_dir}")
done
for result_dir in "${anchor_dirs[@]}"; do
  timing_args+=(--anchor-result-dir "${result_dir}")
done

stability_csv="${staging_dir}/eggpu_timing_stability.csv"
stability_json="${staging_dir}/eggpu_timing_stability.json"
"${python_bin}" "${script_dir}/audit_eggpu_timing_stability.py" \
  "${timing_args[@]}" \
  --metric e2e \
  --expected-samples 5 \
  --expected-cells 208 \
  --max-over-median-limit 5.0 \
  --median-over-min-limit 3.0 \
  --output-csv "${stability_csv}" \
  --output-json "${stability_json}"

overlay_timing_args=()
for result_dir in "${timing_main_dirs[@]}"; do
  overlay_timing_args+=(--main-timing-result-dir "${result_dir}")
done
for result_dir in "${anchor_dirs[@]}"; do
  overlay_timing_args+=(--anchor-timing-result-dir "${result_dir}")
done
overlay_memory_args=()
for result_dir in "${memory_main_dirs[@]}"; do
  overlay_memory_args+=(--main-memory-result-dir "${result_dir}")
done
for result_dir in "${anchor_dirs[@]}"; do
  overlay_memory_args+=(--anchor-memory-result-dir "${result_dir}")
done

ledger="${staging_dir}/final_13_cell_outcome_ledger.csv"
overlay_json="${staging_dir}/EGGPU_V10_OVERLAY_AUDIT_20260729.json"
"${python_bin}" "${script_dir}/update_final13_eggpu_uniform_20260729.py" \
  --base-ledger "${base_ledger}" \
  --candidate-mode unified-timing-memory \
  "${overlay_timing_args[@]}" \
  "${overlay_memory_args[@]}" \
  --stability-audit "${stability_json}" \
  --stability-max-over-median-limit 5.0 \
  --stability-median-over-min-limit 3.0 \
  --candidate-sha256 "${candidate_sha256}" \
  --expected-timing-replacements 208 \
  --expected-memory-replacements 208 \
  --output-ledger "${ledger}" \
  --audit-json "${overlay_json}"

"${python_bin}" "${script_dir}/verify_v10_uniform_overlay_gate.py" \
  --stability-json "${stability_json}" \
  --overlay-json "${overlay_json}" \
  --ledger "${ledger}" \
  --candidate-sha256 "${candidate_sha256}" \
  --runtime-sha256 "${runtime_sha256}"

"${python_bin}" "${script_dir}/finalize_v10_overlay_stage.py" \
  --stage-dir "${staging_dir}" \
  --final-dir "${final_dir}"

mv -- "${staging_dir}" "${final_dir}"
trap - EXIT

echo "Published V10 ledger atomically: ${final_dir}"
echo "Candidate SHA-256: ${candidate_sha256}"
echo "Runtime Python SHA-256: ${runtime_sha256}"
