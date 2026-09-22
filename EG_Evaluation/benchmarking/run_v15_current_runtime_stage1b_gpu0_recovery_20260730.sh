#!/usr/bin/env bash
set -euo pipefail

evaluation_root="/home/dataset-assist-0/einwang/workspace/haorandu/EGGPU_Paper_Repo/EG_Evaluation"
runtime_root="/home/dataset-assist-0/einwang/workspace/haorandu/EGGPU_Paper_Repo/.codex-tmp/eggpu_v15_runtime_20260730_final"
python_bin="/home/dataset-assist-0/einwang/conda_cache/conda_env/EGGPU/bin/python"
result_root="${evaluation_root}/benchmarking/results"
log_root="${result_root}/final_v15_current_runtime_logs_20260730"
mkdir -p "${log_root}"

run_dataset() {
  local gpu="$1"
  local dataset="$2"
  local tag="$3"
  env \
    MPLCONFIGDIR="/tmp/eggpu_v15_recovery_mpl_gpu${gpu}" \
    EGGPU_CHILD_PYTHON="${python_bin}" \
    EASYGRAPH_GPU_RESULT_CACHE=FALSE \
    "${python_bin}" "${evaluation_root}/benchmarking/run_full_baselines.py" \
      --gpu "${gpu}" \
      --out-dir "${result_root}/${tag}" \
      --datasets "${dataset}" \
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
      --easygraph-repo "${runtime_root}"
}

run_dataset 0 "ca-HepTh" "final_v15_current_recovery_ca_hepth_gpu0_20260730" \
  >"${log_root}/recovery_ca_hepth_gpu0.log" 2>&1 &
pid0=$!
sleep 1
run_dataset 4 "com-youtube" "final_v15_current_recovery_com_youtube_gpu4_20260730" \
  >"${log_root}/recovery_com_youtube_gpu4.log" 2>&1 &
pid4=$!

status=0
for pid in "${pid0}" "${pid4}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
