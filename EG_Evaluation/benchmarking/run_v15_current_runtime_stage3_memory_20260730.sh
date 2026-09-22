#!/usr/bin/env bash
set -euo pipefail

evaluation_root="/home/dataset-assist-0/einwang/workspace/haorandu/EGGPU_Paper_Repo/EG_Evaluation"
runtime_root="/home/dataset-assist-0/einwang/workspace/haorandu/EGGPU_Paper_Repo/.codex-tmp/eggpu_v15_runtime_20260730_final"
python_bin="/home/dataset-assist-0/einwang/conda_cache/conda_env/EGGPU/bin/python"
result_root="${evaluation_root}/benchmarking/results"
log_root="${result_root}/final_v15_current_runtime_logs_20260730"
mkdir -p "${log_root}"
stage3_group="${V15_STAGE3_GROUP:-all}"

if [[ "${stage3_group}" != "all" &&
      "${stage3_group}" != "regular_124" &&
      "${stage3_group}" != "regular_rest" &&
      "${stage3_group}" != "closeness" ]]; then
  echo "V15_STAGE3_GROUP must be one of: all, regular_124, regular_rest, closeness" >&2
  exit 2
fi

selected_output_tags=()
if [[ "${stage3_group}" == "all" || "${stage3_group}" == "regular_rest" ]]; then
  selected_output_tags+=(
    "final_v15_current_memory_ca_hepth_gpu0_20260730"
    "final_v15_current_memory_gpu3_20260730"
    "final_v15_current_memory_gpu5_20260730"
    "final_v15_current_memory_gpu6_20260730"
    "final_v15_current_memory_gpu7_20260730"
  )
fi
if [[ "${stage3_group}" == "all" || "${stage3_group}" == "regular_124" ]]; then
  selected_output_tags+=(
    "final_v15_current_memory_gpu1_20260730"
    "final_v15_current_memory_gpu2_20260730"
    "final_v15_current_memory_com_youtube_gpu4_20260730"
  )
fi
if [[ "${stage3_group}" == "all" || "${stage3_group}" == "closeness" ]]; then
  selected_output_tags+=(
    "final_v15_current_memory_closeness16_gpu5_20260730"
  )
fi

# Fail before launching any child so a partial or repeated invocation cannot
# mix evidence from different runs in one nominal Stage 3 result set.
for output_tag in "${selected_output_tags[@]}"; do
  output_path="${result_root}/${output_tag}"
  if [[ -e "${output_path}" ]]; then
    echo "Refusing to reuse Stage 3 output path: ${output_path}" >&2
    exit 1
  fi
done

run_memory() {
  local gpu="$1"
  local datasets="$2"
  local functions="$3"
  local closeness_sources="$4"
  local tag="$5"
  env \
    MPLCONFIGDIR="/tmp/eggpu_v15_memory_mpl_gpu${gpu}" \
    EGGPU_CHILD_PYTHON="${python_bin}" \
    EASYGRAPH_GPU_RESULT_CACHE=FALSE \
    "${python_bin}" "${evaluation_root}/benchmarking/run_full_baselines.py" \
      --gpu "${gpu}" \
      --out-dir "${result_root}/${tag}" \
      --datasets "${datasets}" \
      --functions "${functions}" \
      --baselines EGGPU \
      --repeat 3 \
      --warmup 0 \
      --easygraph-warmup 2 \
      --eggpu-execution-protocol steady-state \
      --measurement-mode memory \
      --library-timeout 100 \
      --pr-alpha 0.75 \
      --pr-eps 1e-6 \
      --pr-max-iter 200 \
      --sssp-sources 8 \
      --bc-sources 16 \
      --closeness-sources "${closeness_sources}" \
      --inter-run-cooldown 0.2 \
      --easygraph-repo "${runtime_root}"
}

pids=()

if [[ "${stage3_group}" == "all" || "${stage3_group}" == "regular_rest" ]]; then
  run_memory 0 "ca-HepTh" all 0 "final_v15_current_memory_ca_hepth_gpu0_20260730" \
    >"${log_root}/stage3_memory_ca_hepth_gpu0.log" 2>&1 &
  pids+=("$!")
  sleep 1
  run_memory 3 "ca-HepPh,soc-Slashdot0811" all 0 "final_v15_current_memory_gpu3_20260730" \
    >"${log_root}/stage3_memory_gpu3.log" 2>&1 &
  pids+=("$!")
  sleep 1
  run_memory 5 "ca-CondMat" all 0 "final_v15_current_memory_gpu5_20260730" \
    >"${log_root}/stage3_memory_gpu5.log" 2>&1 &
  pids+=("$!")
  sleep 1
  run_memory 6 "soc-Epinions1" all 0 "final_v15_current_memory_gpu6_20260730" \
    >"${log_root}/stage3_memory_gpu6.log" 2>&1 &
  pids+=("$!")
  sleep 1
  run_memory 7 "email-Enron" all 0 "final_v15_current_memory_gpu7_20260730" \
    >"${log_root}/stage3_memory_gpu7.log" 2>&1 &
  pids+=("$!")
  sleep 1
fi

if [[ "${stage3_group}" == "all" || "${stage3_group}" == "regular_124" ]]; then
  run_memory 1 "LastFM,web-NotreDame" all 0 "final_v15_current_memory_gpu1_20260730" \
    >"${log_root}/stage3_memory_gpu1.log" 2>&1 &
  pids+=("$!")
  sleep 1
  run_memory 2 "p2p-Gnutella04,ER-100k" all 0 "final_v15_current_memory_gpu2_20260730" \
    >"${log_root}/stage3_memory_gpu2.log" 2>&1 &
  pids+=("$!")
  sleep 1
  run_memory 4 "com-youtube" all 0 "final_v15_current_memory_com_youtube_gpu4_20260730" \
    >"${log_root}/stage3_memory_com_youtube_gpu4.log" 2>&1 &
  pids+=("$!")
fi

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" -eq 0 &&
      ("${stage3_group}" == "all" || "${stage3_group}" == "closeness") ]]; then
  run_memory 5 "ER-100k,com-youtube,soc-Slashdot0811,web-NotreDame" Closeness 16 \
    "final_v15_current_memory_closeness16_gpu5_20260730" \
    >"${log_root}/stage3_memory_closeness16_gpu5.log" 2>&1
fi
exit "${status}"
