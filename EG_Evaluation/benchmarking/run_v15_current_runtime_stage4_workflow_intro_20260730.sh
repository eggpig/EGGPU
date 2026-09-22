#!/usr/bin/env bash
set -euo pipefail

evaluation_root="/home/dataset-assist-0/einwang/workspace/haorandu/EGGPU_Paper_Repo/EG_Evaluation"
runtime_root="/home/dataset-assist-0/einwang/workspace/haorandu/EGGPU_Paper_Repo/.codex-tmp/eggpu_v15_runtime_20260730_final"
python_bin="/home/dataset-assist-0/einwang/conda_cache/conda_env/EGGPU/bin/python"
result_root="${evaluation_root}/benchmarking/results"
log_root="${result_root}/final_v15_current_runtime_logs_20260730"
mkdir -p "${log_root}"
stage4_group="${V15_STAGE4_GROUP:-all}"
intro_gpu="${V15_STAGE4_INTRO_GPU:-4}"

if [[ "${stage4_group}" != "all" && "${stage4_group}" != "workflow" && "${stage4_group}" != "intro" ]]; then
  echo "V15_STAGE4_GROUP must be one of: all, workflow, intro" >&2
  exit 2
fi
if [[ ! "${intro_gpu}" =~ ^[0-7]$ ]]; then
  echo "V15_STAGE4_INTRO_GPU must be an integer from 0 through 7" >&2
  exit 2
fi

pids=()

if [[ "${stage4_group}" == "all" || "${stage4_group}" == "workflow" ]]; then
  env \
    CUDA_VISIBLE_DEVICES=0 \
    EGGPU_MONITOR_GPU_INDEX=0 \
    EGGPU_CUDA_ROOT="/home/dataset-assist-0/einwang/conda_cache/conda_env/sglang" \
    EASYGRAPH_GPU_RESULT_CACHE=FALSE \
    MPLCONFIGDIR="/tmp/eggpu_v15_workflow_mpl_gpu0" \
    "${python_bin}" "${evaluation_root}/benchmarking/run_cumulative_workflow_comparison.py" \
      --easygraph-repo "${runtime_root}" \
      --datasets ca-HepTh LastFM p2p-Gnutella04 ca-HepPh email-Enron ca-CondMat \
      --baselines EGGPU EGGPU-isolated \
      --repeat 5 \
      --sources 8 \
      --workflow WCC PageRank BFS SSSP Closeness \
      --timeout 100 \
      --load-timeout 900 \
      --gpu 0 \
      --output-dir "${result_root}/cumulative_workflow5_v15_current_gpu0_20260730_repeat5" \
      --no-resume \
    >"${log_root}/stage4_workflow_gpu0.log" 2>&1 &
  pids+=("$!")
fi

if [[ "${stage4_group}" == "all" || "${stage4_group}" == "intro" ]]; then
  sleep 1
  env \
    BLIS_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES="${intro_gpu}" \
    EASYGRAPH_ENABLE_GPU=TRUE \
    EASYGRAPH_GPU_ADAPTIVE_HOST=FALSE \
    EASYGRAPH_GPU_RESULT_CACHE=FALSE \
    EASYGRAPH_GPU_STRICT_ERRORS=TRUE \
    EGGPU_ALLOW_CUDA_SYNC=TRUE \
    EGGPU_MONITOR_GPU_INDEX="${intro_gpu}" \
    EGGPU_STABLE_TIMING_PROTOCOL=TRUE \
    MALLOC_ARENA_MAX=2 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    OMP_DYNAMIC=FALSE \
    OMP_NUM_THREADS=1 \
    OMP_PROC_BIND=TRUE \
    OPENBLAS_NUM_THREADS=1 \
    PYTHONHASHSEED=0 \
    PYTHONPATH="${runtime_root}" \
    VECLIB_MAXIMUM_THREADS=1 \
    MPLCONFIGDIR="/tmp/eggpu_v15_intro_mpl_gpu${intro_gpu}" \
    "${python_bin}" "${evaluation_root}/benchmarking/run_eggpu_scaling.py" \
      --easygraph-repo "${runtime_root}" \
      --manifests \
        "${evaluation_root}/datasets/scaling/csr/R-MAT-S20-EF16/R-MAT-S20-EF16.json" \
        "${evaluation_root}/datasets/scaling/csr/R-MAT-S22-EF16/R-MAT-S22-EF16.json" \
        "${evaluation_root}/datasets/scaling/csr/R-MAT-S24-EF16/R-MAT-S24-EF16.json" \
        "${evaluation_root}/datasets/scaling/csr/R-MAT-S26-EF16/R-MAT-S26-EF16.json" \
      --functions PageRank \
      --output-dir "${result_root}/fig1_eggpu_v15_current_gpu${intro_gpu}_20260730_call4" \
      --repeat 5 \
      --warmup 2 \
      --timing-processes 0 \
      --memory-repeat 0 \
      --timeout 300 \
      --load-timeout 300 \
      --first-use-call-timeout 300 \
      --validate \
      --eggpu-stable-timing-protocol \
      --no-resume \
    >"${log_root}/stage4_intro_gpu${intro_gpu}.log" 2>&1 &
  pids+=("$!")
fi

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
