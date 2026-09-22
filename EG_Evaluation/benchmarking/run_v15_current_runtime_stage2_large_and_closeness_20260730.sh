#!/usr/bin/env bash
set -euo pipefail

evaluation_root="/home/dataset-assist-0/einwang/workspace/haorandu/EGGPU_Paper_Repo/EG_Evaluation"
runtime_root="/home/dataset-assist-0/einwang/workspace/haorandu/EGGPU_Paper_Repo/.codex-tmp/eggpu_v15_runtime_20260730_final"
python_bin="/home/dataset-assist-0/einwang/conda_cache/conda_env/EGGPU/bin/python"
result_root="${evaluation_root}/benchmarking/results"
log_root="${result_root}/final_v15_current_runtime_logs_20260730"
orkut_manifest="${evaluation_root}/datasets/scaling/csr/com-Orkut/com-Orkut.json"
twitter_manifest="${evaluation_root}/datasets/scaling/csr/GAP-twitter/GAP-twitter.json"
mkdir -p "${log_root}"
stage2_group="${V15_STAGE2_GROUP:-all}"
late_orkut_gpu="${V15_STAGE2_LATE_ORKUT_GPU:-0}"
late_centrality_gpu="${V15_STAGE2_LATE_CENTRALITY_GPU:-3}"
late_connectivity_gpu="${V15_STAGE2_LATE_CONNECTIVITY_GPU:-5}"

if [[ "${stage2_group}" != "all" &&
      "${stage2_group}" != "early" &&
      "${stage2_group}" != "early_gap" &&
      "${stage2_group}" != "late" ]]; then
  echo "V15_STAGE2_GROUP must be one of: all, early, early_gap, late" >&2
  exit 2
fi
for gpu in \
  "${late_orkut_gpu}" "${late_centrality_gpu}" "${late_connectivity_gpu}"; do
  if [[ ! "${gpu}" =~ ^[0-7]$ ]]; then
    echo "V15 Stage 2 GPU indices must be integers in [0, 7]" >&2
    exit 2
  fi
done
if [[ "${late_orkut_gpu}" == "${late_centrality_gpu}" ||
      "${late_orkut_gpu}" == "${late_connectivity_gpu}" ||
      "${late_centrality_gpu}" == "${late_connectivity_gpu}" ]]; then
  echo "V15 Stage 2 late GPU indices must be distinct" >&2
  exit 2
fi

selected_output_tags=()
if [[ "${stage2_group}" == "all" || "${stage2_group}" == "late" ]]; then
  selected_output_tags+=(
    "final_v15_current_com_orkut_gpu${late_orkut_gpu}_20260730"
    "final_v15_current_gap_centrality_gpu${late_centrality_gpu}_20260730"
    "final_v15_current_gap_connectivity_gpu${late_connectivity_gpu}_20260730"
  )
fi
if [[ "${stage2_group}" == "all" || "${stage2_group}" == "early" ]]; then
  selected_output_tags+=(
    "final_v15_current_recovery_closeness16_gpu5_20260730"
  )
fi
if [[ "${stage2_group}" == "all" ||
      "${stage2_group}" == "early" ||
      "${stage2_group}" == "early_gap" ]]; then
  selected_output_tags+=(
    "final_v15_current_gap_paths_gpu6_20260730"
    "final_v15_current_gap_projection_structural_gpu7_20260730"
  )
fi

# Refuse the entire selected launch before starting any child. This prevents a
# partial rerun from mixing fresh evidence with an existing nominal Stage 2
# directory when several GPU workers are launched in parallel.
for output_tag in "${selected_output_tags[@]}"; do
  output_path="${result_root}/${output_tag}"
  if [[ -e "${output_path}" ]]; then
    echo "Refusing to reuse Stage 2 output path: ${output_path}" >&2
    exit 1
  fi
done

run_scaling() {
  local gpu="$1"
  local manifest="$2"
  local functions="$3"
  local tag="$4"
  local first_use_timeout="$5"
  env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    EGGPU_MONITOR_GPU_INDEX="${gpu}" \
    EGGPU_CHILD_PYTHON="${python_bin}" \
    EASYGRAPH_ENABLE_GPU=TRUE \
    EASYGRAPH_GPU_ADAPTIVE_HOST=FALSE \
    EASYGRAPH_GPU_KCORE_HOST_ENABLE=FALSE \
    EASYGRAPH_GPU_RESULT_CACHE=FALSE \
    EASYGRAPH_GPU_SCC_HOST_ENABLE=FALSE \
    EASYGRAPH_GPU_SSSP_HOST_ENABLE=FALSE \
    EASYGRAPH_GPU_STRICT_ERRORS=TRUE \
    EGGPU_ALLOW_CUDA_SYNC=TRUE \
    MPLCONFIGDIR="/tmp/eggpu_v15_stage2_mpl_gpu${gpu}" \
    PYTHONPATH="${runtime_root}" \
    "${python_bin}" "${evaluation_root}/benchmarking/run_eggpu_scaling.py" \
      --easygraph-repo "${runtime_root}" \
      --manifests "${manifest}" \
      --functions "${functions}" \
      --output-dir "${result_root}/${tag}" \
      --repeat 5 \
      --warmup 1 \
      --timing-processes 0 \
      --memory-repeat 3 \
      --source-count 8 \
      --bc-source-count 16 \
      --closeness-source-count 16 \
      --structural-node-count 0 \
      --memory-poll-ms 2 \
      --timeout 100 \
      --first-use-call-timeout "${first_use_timeout}" \
      --load-timeout 900 \
      --validate \
      --no-resume
}

run_closeness() {
  local gpu="5"
  env \
    MPLCONFIGDIR="/tmp/eggpu_v15_stage2_mpl_gpu${gpu}" \
    EGGPU_CHILD_PYTHON="${python_bin}" \
    EASYGRAPH_GPU_RESULT_CACHE=FALSE \
    "${python_bin}" "${evaluation_root}/benchmarking/run_full_baselines.py" \
      --gpu "${gpu}" \
      --out-dir "${result_root}/final_v15_current_recovery_closeness16_gpu5_20260730" \
      --datasets "ER-100k,com-youtube,soc-Slashdot0811,web-NotreDame" \
      --functions Closeness \
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
      --closeness-sources 16 \
      --inter-run-cooldown 0.2 \
      --easygraph-repo "${runtime_root}"
}

pids=()

if [[ "${stage2_group}" == "all" || "${stage2_group}" == "late" ]]; then
  run_scaling "${late_orkut_gpu}" "${orkut_manifest}" all \
    "final_v15_current_com_orkut_gpu${late_orkut_gpu}_20260730" 100 \
    >"${log_root}/stage2_com_orkut_gpu${late_orkut_gpu}.log" 2>&1 &
  pids+=("$!")
  sleep 1
  run_scaling "${late_centrality_gpu}" "${twitter_manifest}" "PageRank,BC,Closeness" \
    "final_v15_current_gap_centrality_gpu${late_centrality_gpu}_20260730" 220 \
    >"${log_root}/stage2_gap_centrality_gpu${late_centrality_gpu}.log" 2>&1 &
  pids+=("$!")
  sleep 1
  run_scaling "${late_connectivity_gpu}" "${twitter_manifest}" "WCC,SCC,KCore" \
    "final_v15_current_gap_connectivity_gpu${late_connectivity_gpu}_20260730" 220 \
    >"${log_root}/stage2_gap_connectivity_gpu${late_connectivity_gpu}.log" 2>&1 &
  pids+=("$!")
fi

if [[ "${stage2_group}" == "all" || "${stage2_group}" == "early" ]]; then
  run_closeness \
    >"${log_root}/stage2_closeness16_gpu5.log" 2>&1 &
  pids+=("$!")
  sleep 1
fi

if [[ "${stage2_group}" == "all" ||
      "${stage2_group}" == "early" ||
      "${stage2_group}" == "early_gap" ]]; then
  run_scaling 6 "${twitter_manifest}" "BFS,Dijkstra,BellmanFord,SSSP" \
    "final_v15_current_gap_paths_gpu6_20260730" 220 \
    >"${log_root}/stage2_gap_paths_gpu6.log" 2>&1 &
  pids+=("$!")
  sleep 1
  run_scaling 7 "${twitter_manifest}" \
    "MST,LCC,EffectiveSize,Efficiency,Constraint,Hierarchy" \
    "final_v15_current_gap_projection_structural_gpu7_20260730" 220 \
    >"${log_root}/stage2_gap_projection_structural_gpu7.log" 2>&1 &
  pids+=("$!")
fi

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
