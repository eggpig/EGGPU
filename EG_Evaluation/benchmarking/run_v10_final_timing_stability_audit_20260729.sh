#!/usr/bin/env bash
set -euo pipefail

audit_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
evaluation_root="$(cd -- "${audit_script_dir}/.." && pwd)"
audit_python="${EGGPU_AUDIT_PYTHON:-python}"
cd "${evaluation_root}"

v10_input_args=(
  --main-result-dir benchmarking/results/final_v10_r1_uniform_gpu0_20260729 \
  --main-result-dir benchmarking/results/final_v10_r1_uniform_gpu1_20260729 \
  --main-result-dir benchmarking/results/final_v10_r1_uniform_gpu2_20260729 \
  --main-result-dir benchmarking/results/final_v10_r1_uniform_gpu3_20260729 \
  --main-result-dir benchmarking/results/final_v10_r1_uniform_gpu5_20260729 \
  --main-result-dir benchmarking/results/final_v10_r1_uniform_gpu6_20260729 \
  --main-result-dir benchmarking/results/final_v10_r1_uniform_gpu7_email_20260729 \
  --main-result-dir benchmarking/results/final_v10_r1_uniform_closeness_ER-100k_gpu2_20260729 \
  --main-result-dir benchmarking/results/final_v10_r1_uniform_closeness_com-youtube_gpu7_20260729 \
  --main-result-dir benchmarking/results/final_v10_r1_uniform_closeness_soc-Slashdot0811_gpu2_20260729 \
  --main-result-dir benchmarking/results/final_v10_r1_uniform_closeness_web-NotreDame_gpu1_20260729 \
  --anchor-result-dir benchmarking/results/final_v10_r1_uniform_com_orkut_gpu4_20260729 \
  --anchor-result-dir benchmarking/results/final_v10_r1_uniform_gap_twitter_part_centrality_gpu7_20260729 \
  --anchor-result-dir benchmarking/results/final_v10_r1_uniform_gap_twitter_part_connectivity_gpu6_20260729 \
  --anchor-result-dir benchmarking/results/final_v10_r1_uniform_gap_twitter_part_paths_gpu5_20260729 \
  --anchor-result-dir benchmarking/results/final_v10_r1_uniform_gap_twitter_part_projection_structural_gpu3_20260729 \
  --expected-samples 5 \
  --expected-cells 208 \
  --max-over-median-limit 5.0 \
  --median-over-min-limit 3.0
)

run_v10_metric_audit() {
  local audit_metric="$1"
  local output_stem="$2"
  "${audit_python}" benchmarking/audit_eggpu_timing_stability.py \
    "${v10_input_args[@]}" \
    --metric "${audit_metric}" \
    --output-csv \
    "benchmarking/results/final_v10_timing_stability_audit_20260729/${output_stem}.csv" \
    --output-json \
    "benchmarking/results/final_v10_timing_stability_audit_20260729/${output_stem}.json"
}

audit_exit_status=0
run_v10_metric_audit e2e eggpu_timing_stability \
  || audit_exit_status=$?
run_v10_metric_audit kernel eggpu_kernel_stability_diagnostic \
  || audit_exit_status=$?
run_v10_metric_audit build eggpu_build_stability_diagnostic \
  || audit_exit_status=$?
exit "${audit_exit_status}"
