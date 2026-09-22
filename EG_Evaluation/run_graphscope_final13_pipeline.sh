#!/usr/bin/env bash
set -euo pipefail

EVAL_ROOT="/home/dataset-assist-0/einwang/workspace/haorandu/EGGPU_Paper_Repo/EG_Evaluation"
PAPER_ROOT="/home/dataset-assist-0/einwang/workspace/haorandu/EGGPU_Paper_Repo/writing"
PYTHON="/home/dataset-assist-0/einwang/conda_cache/conda_env/EGGPU/bin/python"
GRAPHSCOPE_PYTHON="/home/dataset-assist-0/einwang/workspace/haorandu/.envs/graphscope-0.29.0/bin/python"
MANIFEST_ROOT="${EVAL_ROOT}/datasets/graphscope_prepared_13_v1"
RUN_ROOT="${EVAL_ROOT}/benchmarking/results/graphscope_13_20260727_repeat5_clean"
TIMING_DIR="${RUN_ROOT}/timing"
MEMORY_DIR="${RUN_ROOT}/memory"
AUDIT_DIR="${RUN_ROOT}/audit"
BASE_ASSETS="${PAPER_ROOT}/EGGPU_FINAL_EXPERIMENT_ASSETS_13_V5_20260726"
FINAL_ASSETS="${PAPER_ROOT}/EGGPU_FINAL_EXPERIMENT_ASSETS_13_V6_GRAPHSCOPE_20260727"
LOG="${EVAL_ROOT}/benchmarking/results/graphscope_13_20260727_repeat5_clean.pipeline.log"

exec > >(tee -a "${LOG}") 2>&1
echo "[pipeline] started $(date --iso-8601=seconds)"
echo "[pipeline] waiting for the GAP-twitter prepared manifest"
while [[ ! -f "${MANIFEST_ROOT}/GAP-twitter/manifest.json" ]]; do
    if ! pgrep -f "prepare_graphscope_dataset.*GAP-twitter" >/dev/null; then
        echo "[pipeline] preparation process exited without a GAP-twitter manifest"
        exit 1
    fi
    sleep 60
done
while pgrep -f "prepare_graphscope_dataset.*GAP-twitter" >/dev/null; do
    sleep 10
done

MANIFEST_COUNT="$(find "${MANIFEST_ROOT}" -mindepth 2 -maxdepth 2 -name manifest.json | wc -l)"
if [[ "${MANIFEST_COUNT}" != "13" ]]; then
    echo "[pipeline] expected 13 dataset manifests, found ${MANIFEST_COUNT}"
    exit 1
fi
echo "[pipeline] all 13 prepared inputs are complete"

cd "${EVAL_ROOT}"
"${PYTHON}" benchmarking/run_graphscope_baseline.py \
    --graphscope-python "${GRAPHSCOPE_PYTHON}" \
    --manifest-root "${MANIFEST_ROOT}" \
    --output "${TIMING_DIR}" \
    --phase timing \
    --repeat 5 \
    --timeout 100

"${PYTHON}" benchmarking/validate_graphscope_results.py \
    --timing-dir "${TIMING_DIR}" \
    --legacy-validation \
      "${EVAL_ROOT}/benchmarking/results/latest_v5_main11_with_baselines_20260726/correctness_validation.csv" \
    --large-reference-dir \
      "${EVAL_ROOT}/benchmarking/results/latest_v5_large_core_20260726/eggpu_large_matrix/raw" \
    --output "${AUDIT_DIR}"

"${PYTHON}" benchmarking/run_graphscope_baseline.py \
    --graphscope-python "${GRAPHSCOPE_PYTHON}" \
    --manifest-root "${MANIFEST_ROOT}" \
    --output "${MEMORY_DIR}" \
    --phase memory \
    --memory-repeat 3 \
    --timeout 100 \
    --eligibility-results "${TIMING_DIR}/results_long.csv"

mkdir -p "${FINAL_ASSETS}"
cp -a "${BASE_ASSETS}/." "${FINAL_ASSETS}/"

"${PYTHON}" benchmarking/merge_graphscope_into_final_13.py \
    --base-ledger "${BASE_ASSETS}/final_13_cell_outcome_ledger.csv" \
    --timing-dir "${TIMING_DIR}" \
    --memory-dir "${MEMORY_DIR}" \
    --validation-csv "${AUDIT_DIR}/graphscope_correctness_validation.csv" \
    --output-dir "${FINAL_ASSETS}"

"${PYTHON}" benchmarking/generate_final_13_complete_assets_graphscope.py \
    --ledger "${FINAL_ASSETS}/final_13_cell_outcome_ledger.csv" \
    --output-dir "${FINAL_ASSETS}"

"${PYTHON}" benchmarking/generate_chapter4_evaluation_assets_graphscope.py \
    --source-assets "${FINAL_ASSETS}" \
    --workflow-result \
      "${EVAL_ROOT}/benchmarking/results/cumulative_workflow5_latest_v4_gpu2_20260726_repeat5" \
    --output-dir "${FINAL_ASSETS}/chapter4" \
    --timing-ledger "${FINAL_ASSETS}/final_13_cell_outcome_ledger.csv" \
    --historical-eggpu-summary \
      "/home/dataset-assist-0/einwang/workspace/haorandu/EGGPU_Legacy_Baseline_2024/results/legacy_eggpu_2024_gpu7_20260721_141222/legacy_eggpu_2024_summary.csv" \
    --historical-eggpu-comparison \
      "${BASE_ASSETS}/historical_eggpu_2024/legacy_vs_current_e2e.csv" \
    --device-registry-acceptance \
      "${EVAL_ROOT}/benchmarking/results/device_graph_view_registry_20260721/dense_balanced2_web_notredame_acceptance.json" \
    --device-registry-acceptance \
      "${EVAL_ROOT}/benchmarking/results/device_graph_view_registry_20260721/dense_balanced_com_youtube_acceptance.json"

echo "[pipeline] completed $(date --iso-8601=seconds)"
echo "[pipeline] final assets: ${FINAL_ASSETS}"
