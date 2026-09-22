#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${COMMON_PY:-/home/dataset-assist-0/einwang/conda_cache/conda_env/EGGPU/bin/python}"
CORE_ROOT="${FINAL13_RESULT_ROOT:-${ROOT}/benchmarking/results/final13_missing_gpu7_20260716_212844}"
FOLLOWUP_ROOT="${FINAL13_FOLLOWUP_ROOT:?Set FINAL13_FOLLOWUP_ROOT to the completed recovery follow-up directory}"
OUTPUT_ROOT="${FINAL13_ASSET_ROOT:-${ROOT}/../writing/EGGPU_FINAL_EXPERIMENT_ASSETS_13_COMPLETE_20260717}"

ORIGINAL_MAIN_ROOT="${ROOT}/benchmarking/results/full_eval_gpu0_20260712_144427_repeat5_splitmem_exact_mst_strict_nxcg"
TARGETED_RETEST_ROOT="${ROOT}/benchmarking/results/final13_targeted_eggpu_gpu7_20260717/timing"
MAIN_ROOT="${ROOT}/benchmarking/results/final13_authoritative_main_20260717"
OLD_CLOSENESS_ROOT="${ROOT}/benchmarking/results/closeness_large_gpu0_20260715_121333_repeat5"
CURRENT_CLOSENESS_ROOT="${ROOT}/benchmarking/results/final13_targeted_closeness_split_gpu5_20260717"
CLOSENESS_REFERENCE_ROOT="${ROOT}/benchmarking/results/final13_targeted_closeness_cpu_reference_20260717"
CLOSENESS_ROOT="${ROOT}/benchmarking/results/final13_authoritative_closeness_20260717"
FIRST_USE_ROOT="${ROOT}/benchmarking/results/eggpu_first_use_repaired_gpu0_20260715_121333_repeat5"
NATURAL_ROOT="${ROOT}/benchmarking/results/eggpu_natural_workflow_gpu0_20260715_121333_matched_repeat5"
ABLATION_ROOT="${ROOT}/benchmarking/results/ablation_gpu0_20260712_144427_repeat5_strengthened"
OLD_SCALE="${ROOT}/benchmarking/results/second_supplement_gpu7_20260716_final12_scaling_semantic_reaudit/scaling/scaling_all.csv"
OLD_RMAT="${ROOT}/benchmarking/results/second_supplement_gpu7_20260716_final12_scaling_semantic_reaudit/controlled_rmat/scaling_all.csv"
OLD_NXCG="${ROOT}/benchmarking/results/nxcugraph_scale_qualification_gpu7_20260716_orkut_wcc_fix/nxcugraph_scale_qualification.csv"
SYGRAPH_ROOT="${SYGRAPH_RESULT_ROOT:-${CORE_ROOT}/sygraph_bfs}"
SYGRAPH_QUALIFICATION_ROOT="${ROOT}/benchmarking/results/sygraph_qualification_gpu0_20260717"

SYGRAPH_LEDGER_ARGS=()
SYGRAPH_VISUAL_ARGS=()
if [[ -s "${SYGRAPH_ROOT}/sygraph_bfs.csv" ]] && \
  "${PYTHON_BIN}" -c '
import pandas as pd, sys
data = pd.read_csv(sys.argv[1])
raise SystemExit(0 if ((data["status"] == "ok") & (data["validation"] == "pass")).any() else 1)
' "${SYGRAPH_ROOT}/sygraph_bfs.csv"; then
  SYGRAPH_LEDGER_ARGS=(--sygraph-result "${SYGRAPH_ROOT}")
  SYGRAPH_VISUAL_ARGS=(--sygraph-result "${SYGRAPH_ROOT}")
fi

for required in \
  "${CORE_ROOT}/phase_status.json" \
  "${CORE_ROOT}/eggpu_large_matrix/scaling_all.csv" \
  "${CORE_ROOT}/nxcugraph_large_matrix/nxcugraph_large_matrix.csv" \
  "${CORE_ROOT}/gunrock_large_matrix/gunrock_large_matrix.csv" \
  "${CORE_ROOT}/cpu_large_matrix/cpu_large_matrix.csv" \
  "${FOLLOWUP_ROOT}/phase_status.json" \
  "${ORIGINAL_MAIN_ROOT}/results_long.csv" \
  "${TARGETED_RETEST_ROOT}/results_samples.csv" \
  "${OLD_CLOSENESS_ROOT}/closeness_large_sampled_validation.csv" \
  "${CURRENT_CLOSENESS_ROOT}/results_long.csv" \
  "${CLOSENESS_REFERENCE_ROOT}/results_samples.csv" \
  "${SYGRAPH_QUALIFICATION_ROOT}/SYGRAPH_QUALIFICATION.json" \
  "${SYGRAPH_QUALIFICATION_ROOT}/SYGRAPH_QUALIFICATION.md" \
  "${FIRST_USE_ROOT}/first_use_vs_steady.csv" \
  "${ABLATION_ROOT}/ablation_workflow_dataset_totals.csv"; do
  if [[ ! -s "${required}" ]]; then
    echo "required completed artifact is missing: ${required}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_ROOT}"

echo "[finalize 1/10] provenance-preserving authoritative timing views"
"${PYTHON_BIN}" "${ROOT}/benchmarking/prepare_final_13_authoritative_inputs.py" \
  --main "${ORIGINAL_MAIN_ROOT}" \
  --targeted-retest "${TARGETED_RETEST_ROOT}" \
  --main-output "${MAIN_ROOT}" \
  --old-closeness "${OLD_CLOSENESS_ROOT}" \
  --current-closeness "${CURRENT_CLOSENESS_ROOT}" \
  --closeness-reference "${CLOSENESS_REFERENCE_ROOT}" \
  --closeness-output "${CLOSENESS_ROOT}"
cp "${MAIN_ROOT}/TARGETED_EGGPU_SELECTION_DECISIONS.csv" "${OUTPUT_ROOT}/"
cp "${CLOSENESS_ROOT}/CURRENT_CLOSENESS_SELECTION_DECISIONS.csv" "${OUTPUT_ROOT}/"
cp "${CLOSENESS_ROOT}/CURRENT_CLOSENESS_EXTERNAL_VALIDATION.csv" "${OUTPUT_ROOT}/"
cp "${SYGRAPH_QUALIFICATION_ROOT}/SYGRAPH_QUALIFICATION.json" "${OUTPUT_ROOT}/"
cp "${SYGRAPH_QUALIFICATION_ROOT}/SYGRAPH_QUALIFICATION.md" "${OUTPUT_ROOT}/"

echo "[finalize 2/10] complete 13 x 16 outcome ledger across all qualified systems"
"${PYTHON_BIN}" "${ROOT}/benchmarking/generate_final_13_failure_ledger.py" \
  --main-result "${MAIN_ROOT}" \
  --closeness-result "${CLOSENESS_ROOT}" \
  --closeness-memory-result "${CURRENT_CLOSENESS_ROOT}" \
  --core-result "${CORE_ROOT}" \
  "${SYGRAPH_LEDGER_ARGS[@]}" \
  --output-dir "${OUTPUT_ROOT}"

"${PYTHON_BIN}" -c '
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
summary = json.loads((root / "final_13_cell_outcome_summary.json").read_text())
if summary["missing_experiment_cells"]:
    raise SystemExit(
        f"final main ledger still contains {summary['"'"'missing_experiment_cells'"'"']} "
        "unclassified cells; inspect FINAL_13_FAILURE_AND_COMPLETENESS_REPORT.md"
    )
print("main-ledger completeness gate: pass")
' "${OUTPUT_ROOT}"

echo "[finalize 3/10] concise baseline-version provenance"
"${PYTHON_BIN}" "${ROOT}/benchmarking/generate_final_baseline_versions.py" \
  --main-version-json "${MAIN_ROOT}/baseline_versions.json" \
  --sygraph-result "${SYGRAPH_ROOT}" \
  --output-dir "${OUTPUT_ROOT}"

echo "[finalize 4/10] exact numerical summaries and TeX tables"
"${PYTHON_BIN}" "${ROOT}/benchmarking/generate_final_13_complete_assets.py" \
  --ledger "${OUTPUT_ROOT}/final_13_cell_outcome_ledger.csv" \
  --output-dir "${OUTPUT_ROOT}"

echo "[finalize 5/10] requested paper figures"
"${PYTHON_BIN}" "${ROOT}/benchmarking/generate_final_experiment_visuals_v2.py" \
  --main-result "${MAIN_ROOT}" \
  --first-use "${FIRST_USE_ROOT}" \
  --natural-workflow "${NATURAL_ROOT}" \
  --ablation "${ABLATION_ROOT}" \
  --old-scale-csv "${OLD_SCALE}" \
  --old-rmat-csv "${OLD_RMAT}" \
  --old-nxcugraph-csv "${OLD_NXCG}" \
  --core-result "${CORE_ROOT}" \
  --followup-result "${FOLLOWUP_ROOT}" \
  --cumulative-result "${FOLLOWUP_ROOT}/cumulative_workflow" \
  "${SYGRAPH_VISUAL_ARGS[@]}" \
  --output-dir "${OUTPUT_ROOT}"

echo "[finalize 6/10] all-experiment non-success ledger"
"${PYTHON_BIN}" "${ROOT}/benchmarking/generate_all_experiment_outcome_report.py" \
  --final-ledger "${OUTPUT_ROOT}/final_13_cell_outcome_ledger.csv" \
  --first-use "${FIRST_USE_ROOT}" \
  --ablation "${ABLATION_ROOT}" \
  --cumulative "${FOLLOWUP_ROOT}/cumulative_workflow" \
  --core-result "${CORE_ROOT}" \
  --followup-result "${FOLLOWUP_ROOT}" \
  "${SYGRAPH_VISUAL_ARGS[@]}" \
  --output-dir "${OUTPUT_ROOT}"

echo "[finalize 7/10] authoritative Chinese experiment guide"
"${PYTHON_BIN}" "${ROOT}/benchmarking/generate_final_13_authoritative_guide.py" \
  --asset-dir "${OUTPUT_ROOT}" \
  --core-result "${CORE_ROOT}" \
  --followup-result "${FOLLOWUP_ROOT}" \
  --output "${ROOT}/../writing/EGGPU_FINAL_13_EXPERIMENT_RESULTS_AND_EVIDENCE_20260717_CN.md"

echo "[finalize 8/10] standalone TeX table review PDF"
"${PYTHON_BIN}" "${ROOT}/benchmarking/generate_final_13_table_review.py" \
  --asset-dir "${OUTPUT_ROOT}"
(
  cd "${OUTPUT_ROOT}"
  latexmk -pdf -interaction=nonstopmode -halt-on-error \
    EGGPU_FINAL_TABLES_REVIEW_20260717.tex
  latexmk -c EGGPU_FINAL_TABLES_REVIEW_20260717.tex
)

echo "[finalize 9/10] raw-evidence and paper-asset audit"
"${PYTHON_BIN}" "${ROOT}/benchmarking/audit_final_13_paper_assets.py" \
  --assets "${OUTPUT_ROOT}" \
  --main-result "${MAIN_ROOT}" \
  --closeness-result "${CLOSENESS_ROOT}" \
  --core-result "${CORE_ROOT}" \
  --followup-result "${FOLLOWUP_ROOT}" \
  --anchor-manifest "${ROOT}/datasets/scaling/csr/com-Orkut/com-Orkut.json" \
  --anchor-manifest "${ROOT}/datasets/scaling/csr/GAP-twitter/GAP-twitter.json"

echo "[finalize 10/10] bundle manifest"
"${PYTHON_BIN}" -c '
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
files = []
for path in sorted(item for item in root.iterdir() if item.is_file()):
    if path.name == "FINAL_13_COMPLETE_ASSET_MANIFEST.json":
        continue
    files.append({
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })
manifest = {
    "status": "completed",
    "asset_root": str(root),
    "files": files,
    "authoritative_numeric_summary": "final_13_numeric_summary.json",
    "main_failure_report": "FINAL_13_FAILURE_AND_COMPLETENESS_REPORT.md",
    "all_experiment_failure_report": "ALL_EXPERIMENT_NON_SUCCESS_REPORT.md",
}
(root / "FINAL_13_COMPLETE_ASSET_MANIFEST.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
print(root)
' "${OUTPUT_ROOT}"

echo "Final 13-dataset paper bundle: ${OUTPUT_ROOT}"
