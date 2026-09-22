#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
evaluation_root="$(cd -- "${script_dir}/.." && pwd)"
result_root="${evaluation_root}/benchmarking/results"
python_bin="${EGGPU_PYTHON:-/home/dataset-assist-0/einwang/conda_cache/conda_env/EGGPU/bin/python}"
assembler="${script_dir}/assemble_v15_pagerank_protocol_correction_20260730.py"
output_root="${result_root}/final_v15_pagerank_protocol_correction_assembly_20260730"

expected_native_sha256="d2a93a3da0ffd0c554c9dab52c9d05d8c1f5f5b4904bcfd19fd95b695bb06eaf"
expected_runtime_sha256="1be28aeb48374c65642b26f7233a7764f66fb3fb8003e6434e04744e08b643fb"

require_canonical_lists() {
  local canonical_list
  for canonical_list in main_timing_dirs.txt anchor_timing_dirs.txt; do
    if [[ ! -s "${output_root}/${canonical_list}" ]]; then
      echo "Verified correction assembly lacks ${canonical_list}" >&2
      exit 1
    fi
  done
}

if [[ "${1:-}" == "--verify" ]]; then
  if (($# != 1)); then
    echo "Usage: $0 [--verify]" >&2
    exit 2
  fi
  "${python_bin}" "${assembler}" \
    --verify-existing-root "${output_root}" \
    --expected-native-sha256 "${expected_native_sha256}" \
    --expected-runtime-sha256 "${expected_runtime_sha256}"
  require_canonical_lists
  exit 0
fi
if (($#)); then
  echo "Usage: $0 [--verify]" >&2
  exit 2
fi

original_args=(
  --original-result-dir "${result_root}/final_v15_current_recovery_ca_hepth_gpu0_20260730"
  --original-result-dir "${result_root}/final_v15_current_r2_gpu1_20260730"
  --original-result-dir "${result_root}/final_v15_current_r2_gpu2_20260730"
  --original-result-dir "${result_root}/final_v15_current_r2_gpu3_20260730"
  --original-result-dir "${result_root}/final_v15_current_recovery_com_youtube_gpu4_20260730"
  --original-result-dir "${result_root}/final_v15_current_r2_gpu5_20260730"
  --original-result-dir "${result_root}/final_v15_current_r2_gpu6_20260730"
  --original-result-dir "${result_root}/final_v15_current_r2_gpu7_email_20260730"
)

correction_args=(
  --correction-result-dir "${result_root}/final_v15_pagerank_protocol_correction_ca_hepth_20260730"
  --correction-result-dir "${result_root}/final_v15_pagerank_protocol_correction_lastfm_web_20260730"
  --correction-result-dir "${result_root}/final_v15_pagerank_protocol_correction_p2p_er_20260730"
  --correction-result-dir "${result_root}/final_v15_pagerank_protocol_correction_hepph_slashdot_20260730"
  --correction-result-dir "${result_root}/final_v15_pagerank_protocol_correction_youtube_20260730"
  --correction-result-dir "${result_root}/final_v15_pagerank_protocol_correction_condmat_20260730"
  --correction-result-dir "${result_root}/final_v15_pagerank_protocol_correction_epinions_20260730"
  --correction-result-dir "${result_root}/final_v15_pagerank_protocol_correction_enron_20260730"
)

anchor_args=(
  --anchor-result-dir "${result_root}/final_v15_current_com_orkut_gpu0_20260730"
  --anchor-result-dir "${result_root}/final_v15_current_gap_centrality_gpu3_20260730"
  --anchor-result-dir "${result_root}/final_v15_current_gap_connectivity_gpu5_20260730"
  --anchor-result-dir "${result_root}/final_v15_current_gap_paths_gpu6_20260730"
  --anchor-result-dir "${result_root}/final_v15_current_gap_projection_structural_gpu7_20260730"
)

"${python_bin}" "${assembler}" \
  "${original_args[@]}" \
  "${correction_args[@]}" \
  --supplemental-main-result-dir \
    "${result_root}/final_v15_current_recovery_closeness16_gpu5_20260730" \
  "${anchor_args[@]}" \
  --output-root "${output_root}" \
  --expected-native-sha256 "${expected_native_sha256}" \
  --expected-runtime-sha256 "${expected_runtime_sha256}"

"${python_bin}" "${assembler}" \
  --verify-existing-root "${output_root}" \
  --expected-native-sha256 "${expected_native_sha256}" \
  --expected-runtime-sha256 "${expected_runtime_sha256}"

require_canonical_lists
