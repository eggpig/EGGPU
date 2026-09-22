#!/usr/bin/env bash
set -euo pipefail

evaluation_root="/home/dataset-assist-0/einwang/workspace/haorandu/EGGPU_Paper_Repo/EG_Evaluation"
runtime_root="/home/dataset-assist-0/einwang/workspace/haorandu/EGGPU_Paper_Repo/.codex-tmp/eggpu_v15_runtime_20260730_final"
python_bin="/home/dataset-assist-0/einwang/conda_cache/conda_env/EGGPU/bin/python"
result_root="${evaluation_root}/benchmarking/results"
log_root="${result_root}/final_v15_current_runtime_logs_20260730"
mkdir -p "${log_root}"

run_timing() {
  local gpu="$1"
  local datasets="$2"
  local tag="$3"
  local log_name="$4"
  env \
    MPLCONFIGDIR="/tmp/eggpu_v15_post_lcc_fix_mpl_gpu${gpu}" \
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

run_timing 0 "ca-HepTh" \
  "final_v15_current_recovery_ca_hepth_gpu0_20260730" \
  "post_fix_ca_hepth_gpu0.log" &
pid0=$!
sleep 1
run_timing 1 "LastFM,web-NotreDame" \
  "final_v15_current_r2_gpu1_20260730" \
  "post_fix_gpu1.log" &
pid1=$!
sleep 1
run_timing 2 "p2p-Gnutella04,ER-100k" \
  "final_v15_current_r2_gpu2_20260730" \
  "post_fix_gpu2.log" &
pid2=$!
sleep 1
run_timing 3 "ca-HepPh,soc-Slashdot0811" \
  "final_v15_current_r2_gpu3_20260730" \
  "post_fix_gpu3.log" &
pid3=$!
sleep 1
run_timing 4 "com-youtube" \
  "final_v15_current_recovery_com_youtube_gpu4_20260730" \
  "post_fix_com_youtube_gpu4.log" &
pid4=$!
sleep 1
run_timing 5 "ca-CondMat" \
  "final_v15_current_r2_gpu5_20260730" \
  "post_fix_gpu5.log" &
pid5=$!
sleep 1
run_timing 6 "soc-Epinions1" \
  "final_v15_current_r2_gpu6_20260730" \
  "post_fix_gpu6.log" &
pid6=$!
sleep 1
run_timing 7 "email-Enron" \
  "final_v15_current_r2_gpu7_email_20260730" \
  "post_fix_gpu7.log" &
pid7=$!

status=0
for pid in \
  "${pid0}" "${pid1}" "${pid2}" "${pid3}" \
  "${pid4}" "${pid5}" "${pid6}" "${pid7}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
