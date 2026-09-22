#!/usr/bin/env bash
set -euo pipefail

evaluation_root="/home/dataset-assist-0/einwang/workspace/haorandu/EGGPU_Paper_Repo/EG_Evaluation"
runtime_root="/home/dataset-assist-0/einwang/workspace/haorandu/EGGPU_Paper_Repo/.codex-tmp/eggpu_v15_runtime_20260730_final"
python_bin="/home/dataset-assist-0/einwang/conda_cache/conda_env/EGGPU/bin/python"
result_root="${evaluation_root}/benchmarking/results"
log_root="${result_root}/final_v15_current_runtime_logs_20260730"
mkdir -p "${log_root}"
stage1d_group="${V15_STAGE1D_GROUP:-all}"

if [[ "${stage1d_group}" != "all" &&
      "${stage1d_group}" != "gpu12" &&
      "${stage1d_group}" != "gpu4" ]]; then
  echo "V15_STAGE1D_GROUP must be one of: all, gpu12, gpu4" >&2
  exit 2
fi

selected_output_tags=()
if [[ "${stage1d_group}" == "all" || "${stage1d_group}" == "gpu12" ]]; then
  selected_output_tags+=(
    "final_v15_current_r2_gpu1_20260730"
    "final_v15_current_r2_gpu2_20260730"
  )
fi
if [[ "${stage1d_group}" == "all" || "${stage1d_group}" == "gpu4" ]]; then
  selected_output_tags+=(
    "final_v15_current_recovery_com_youtube_gpu4_20260730"
  )
fi

# Fail before launching any worker so a repeated partial invocation cannot
# combine existing evidence with a newly generated batch.
for output_tag in "${selected_output_tags[@]}"; do
  output_path="${result_root}/${output_tag}"
  if [[ -e "${output_path}" ]]; then
    echo "Refusing to reuse Stage 1d output path: ${output_path}" >&2
    exit 1
  fi
done

run_timing() {
  local gpu="$1"
  local datasets="$2"
  local tag="$3"
  local log_name="$4"
  env \
    MPLCONFIGDIR="/tmp/eggpu_v15_stage1d_mpl_gpu${gpu}" \
    EGGPU_CHILD_PYTHON="${python_bin}" \
    EASYGRAPH_GPU_RESULT_CACHE=FALSE \
    "${python_bin}" "${evaluation_root}/benchmarking/run_full_baselines.py" \
      --gpu "${gpu}" \
      --out-dir "${result_root}/${tag}" \
      --datasets "${datasets}" \
      --functions all \
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

# The original Stage 1 parent process was externally terminated while these
# three whole batches were incomplete. Their partial directories are archived
# before this script is launched; every batch below is therefore regenerated
# from sample 1 rather than resumed or spliced.
pids=()
if [[ "${stage1d_group}" == "all" || "${stage1d_group}" == "gpu12" ]]; then
  run_timing 1 "LastFM,web-NotreDame" \
    "final_v15_current_r2_gpu1_20260730" \
    "post_fix_recovery2_gpu1.log" &
  pids+=("$!")
  sleep 1
  run_timing 2 "p2p-Gnutella04,ER-100k" \
    "final_v15_current_r2_gpu2_20260730" \
    "post_fix_recovery2_gpu2.log" &
  pids+=("$!")
  sleep 1
fi
if [[ "${stage1d_group}" == "all" || "${stage1d_group}" == "gpu4" ]]; then
  run_timing 4 "com-youtube" \
    "final_v15_current_recovery_com_youtube_gpu4_20260730" \
    "post_fix_recovery2_com_youtube_gpu4.log" &
  pids+=("$!")
fi

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
