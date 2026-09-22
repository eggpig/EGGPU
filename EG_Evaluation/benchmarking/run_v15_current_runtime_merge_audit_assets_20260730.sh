#!/usr/bin/env bash
set -euo pipefail

# Merge the complete V15 timing/memory matrix, audit it against the frozen
# runtime, rebuild the mixed-estimator ledger (EGGPU: minimum of five;
# external baselines: arithmetic mean of five), and regenerate the
# paper-facing numerical assets.  This script is intentionally fail-closed
# and refuses to overwrite an existing release directory.

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
evaluation_root="$(cd -- "${script_dir}/.." && pwd)"
project_root="$(cd -- "${evaluation_root}/.." && pwd)"
writing_root="${project_root}/writing"
result_root="${evaluation_root}/benchmarking/results"

python_bin="${EGGPU_PYTHON:-/home/dataset-assist-0/einwang/conda_cache/conda_env/EGGPU/bin/python}"
runtime_root="${project_root}/.codex-tmp/eggpu_v15_runtime_20260730_final"
native_binary="${runtime_root}/cpp_easygraph.cpython-310-x86_64-linux-gnu.so"

expected_candidate_sha256="d2a93a3da0ffd0c554c9dab52c9d05d8c1f5f5b4904bcfd19fd95b695bb06eaf"
expected_runtime_sha256="1be28aeb48374c65642b26f7233a7764f66fb3fb8003e6434e04744e08b643fb"
expected_base_ledger_sha256="307722ed0ae82e5b7f43500d27e9847df215b4d85fc2093105f75ab675a293a1"

base_assets="${writing_root}/EGGPU_FINAL_EXPERIMENT_ASSETS_14_UNIFORM_MIN_20260730"
base_ledger="${base_assets}/final_13_cell_outcome_ledger.csv"
output_dir="${writing_root}/EGGPU_FINAL_EXPERIMENT_ASSETS_15_CURRENT_RUNTIME_MIXED_ESTIMATOR_20260730"

main_samples="${result_root}/final13_authoritative_main_20260717/results_samples.csv"
closeness_evidence="${result_root}/closeness_large_gpu0_20260715_121333_repeat5"
graphscope_samples="${result_root}/graphscope_13_20260727_repeat5_reconciled/timing/results_samples.csv"
cpu_large_matrix="${result_root}/latest_v5_large_core_20260726/cpu_large_matrix/cpu_large_matrix.csv"
nxcugraph_samples="${result_root}/nxcugraph_strict_main99_20260730/audit_final/nxcugraph_strict_main99_samples.csv"
gunrock_samples="${result_root}/gunrock_strict_formal_audit_20260730/gunrock_strict_overlay_samples.csv"

main_timing_dirs=()
main_memory_dirs=()
anchor_dirs=()
default_pagerank_correction_root="${result_root}/final_v15_pagerank_protocol_correction_assembly_20260730"
pagerank_correction_root="${default_pagerank_correction_root}"

usage() {
  cat <<'EOF'
Usage:
  run_v15_current_runtime_merge_audit_assets_20260730.sh [options]

Options:
  --main-memory-dir PATH  Add a stage3 regular-matrix memory directory.
                          Repeat to replace the complete built-in memory list.
  --pagerank-correction-root PATH
                          Verify a completed protocol-correction assembly and
                          use its hash-bound main_timing_dirs.txt and
                          anchor_timing_dirs.txt.
  --output-dir PATH       Release directory (must not already exist).
  -h, --help              Show this help.

The canonical release never accepts ad hoc main-timing or anchor directories.
With no directory options, the canonical 2026-07-30 protocol-corrected
PageRank assembly and V15 stage3 memory paths are used.
EOF
}

while (($#)); do
  case "$1" in
    --main-timing-dir)
      echo "--main-timing-dir is disabled: use a verified PageRank correction assembly" >&2
      exit 2
      ;;
    --main-memory-dir)
      if (($# < 2)); then
        echo "--main-memory-dir requires a path" >&2
        exit 2
      fi
      main_memory_dirs+=("$2")
      shift 2
      ;;
    --anchor-dir)
      echo "--anchor-dir is disabled: anchors must come from the verified PageRank correction assembly" >&2
      exit 2
      ;;
    --pagerank-correction-root)
      if (($# < 2)); then
        echo "--pagerank-correction-root requires a path" >&2
        exit 2
      fi
      pagerank_correction_root="$2"
      shift 2
      ;;
    --output-dir)
      if (($# < 2)); then
        echo "--output-dir requires a path" >&2
        exit 2
      fi
      output_dir="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "${python_bin}" ]]; then
  echo "EGGPU Python is not executable: ${python_bin}" >&2
  exit 1
fi

if [[ -z "${pagerank_correction_root}" ||
      ! -d "${pagerank_correction_root}" ]]; then
  echo "PageRank correction assembly is missing: ${pagerank_correction_root}" >&2
  exit 1
fi
pagerank_correction_root="$(
  cd -- "${pagerank_correction_root}" && pwd
)"
"${python_bin}" \
  "${script_dir}/assemble_v15_pagerank_protocol_correction_20260730.py" \
  --verify-existing-root "${pagerank_correction_root}" \
  --expected-native-sha256 "${expected_candidate_sha256}" \
  --expected-runtime-sha256 "${expected_runtime_sha256}" \
  >/dev/null
mapfile -t main_timing_dirs \
  <"${pagerank_correction_root}/main_timing_dirs.txt"
mapfile -t anchor_dirs \
  <"${pagerank_correction_root}/anchor_timing_dirs.txt"
if ((${#main_timing_dirs[@]} == 0)); then
  echo "PageRank correction main-timing directory list is empty" >&2
  exit 1
fi
if ((${#anchor_dirs[@]} == 0)); then
  echo "PageRank correction anchor-timing directory list is empty" >&2
  exit 1
fi

if ((${#main_memory_dirs[@]} == 0)); then
  main_memory_dirs=(
    "${result_root}/final_v15_current_memory_ca_hepth_gpu0_20260730"
    "${result_root}/final_v15_current_memory_gpu1_20260730"
    "${result_root}/final_v15_current_memory_gpu2_20260730"
    "${result_root}/final_v15_current_memory_gpu3_20260730"
    "${result_root}/final_v15_current_memory_com_youtube_gpu4_20260730"
    "${result_root}/final_v15_current_memory_gpu5_20260730"
    "${result_root}/final_v15_current_memory_gpu6_20260730"
    "${result_root}/final_v15_current_memory_gpu7_20260730"
    "${result_root}/final_v15_current_memory_closeness16_gpu5_20260730"
  )
fi

if [[ ! -d "${runtime_root}" ]]; then
  echo "Frozen V15 runtime is missing: ${runtime_root}" >&2
  exit 1
fi
if [[ ! -f "${native_binary}" ]]; then
  echo "Frozen V15 native extension is missing: ${native_binary}" >&2
  exit 1
fi
if [[ ! -f "${base_ledger}" ]]; then
  echo "V14 base ledger is missing: ${base_ledger}" >&2
  exit 1
fi
if [[ -e "${output_dir}" ]]; then
  echo "Refusing to overwrite V15 release directory: ${output_dir}" >&2
  exit 1
fi

required_files=(
  "${base_assets}/paper_table_baseline_versions.csv"
  "${base_assets}/paper_baseline_versions.json"
  "${main_samples}"
  "${graphscope_samples}"
  "${cpu_large_matrix}"
  "${nxcugraph_samples}"
  "${gunrock_samples}"
)
for path in "${required_files[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required V15 input file is missing: ${path}" >&2
    exit 1
  fi
done
if [[ ! -d "${closeness_evidence}" ]]; then
  echo "CPU Closeness evidence directory is missing: ${closeness_evidence}" >&2
  exit 1
fi
for result_dir in \
  "${main_timing_dirs[@]}" \
  "${main_memory_dirs[@]}" \
  "${anchor_dirs[@]}"; do
  if [[ ! -d "${result_dir}" ]]; then
    echo "Required V15 result directory is missing: ${result_dir}" >&2
    exit 1
  fi
done

sha256_file() {
  "${python_bin}" -c \
    'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' \
    "$1"
}

actual_base_sha256="$(sha256_file "${base_ledger}")"
if [[ "${actual_base_sha256}" != "${expected_base_ledger_sha256}" ]]; then
  echo "V14 base ledger SHA-256 differs: ${actual_base_sha256}" >&2
  exit 1
fi
actual_candidate_sha256="$(sha256_file "${native_binary}")"
if [[ "${actual_candidate_sha256}" != "${expected_candidate_sha256}" ]]; then
  echo "Frozen V15 native extension SHA-256 differs: ${actual_candidate_sha256}" >&2
  exit 1
fi

mkdir -p -- "${output_dir}"
audit_dir="${output_dir}/v15_unified_overlay_audit"
mplconfig_dir="${output_dir}/.matplotlib"
mkdir -p -- "${audit_dir}" "${mplconfig_dir}"
if [[ -n "${pagerank_correction_root}" ]]; then
  cp -- \
    "${pagerank_correction_root}/V15_PAGERANK_PROTOCOL_CORRECTION_AUDIT.json" \
    "${pagerank_correction_root}/pagerank_protocol_action_ledger.csv" \
    "${pagerank_correction_root}/non_pagerank_cell_hashes.csv" \
    "${pagerank_correction_root}/main_timing_dirs.txt" \
    "${pagerank_correction_root}/anchor_timing_dirs.txt" \
    "${audit_dir}/"
fi
incomplete_marker="${output_dir}/V15_BUILD_INCOMPLETE"
touch -- "${incomplete_marker}"

completed=0
report_incomplete_build() {
  local status=$?
  if [[ "${completed}" -ne 1 ]]; then
    echo "V15 build failed; partial evidence is retained at ${output_dir}" >&2
  fi
  return "${status}"
}
trap report_incomplete_build EXIT

PYTHONPATH="${script_dir}" "${python_bin}" - \
  "${runtime_root}" \
  "${expected_candidate_sha256}" \
  "${expected_runtime_sha256}" \
  "${audit_dir}/V15_RUNTIME_IDENTITY.json" <<'PY'
import json
import sys
from pathlib import Path

from easygraph_runtime_provenance import collect_runtime_repository_provenance

runtime_root = Path(sys.argv[1]).resolve(strict=True)
expected_native = sys.argv[2]
expected_python = sys.argv[3]
output = Path(sys.argv[4])
identity = collect_runtime_repository_provenance(runtime_root)
snapshot = identity["runtime_python_snapshot"]
if identity["native_sha256"] != expected_native:
    raise SystemExit(
        f"native SHA differs: {identity['native_sha256']} != {expected_native}"
    )
if snapshot["digest"] != expected_python:
    raise SystemExit(
        f"Python snapshot SHA differs: {snapshot['digest']} != {expected_python}"
    )
if snapshot["package_is_symlink"] is not False:
    raise SystemExit("frozen Python package is a symlink")
output.write_text(
    json.dumps(
        {
            "status": "pass",
            "candidate_sha256": identity["native_sha256"],
            "runtime_python_snapshot_sha256": snapshot["digest"],
            "runtime_package_is_symlink": snapshot["package_is_symlink"],
            "runtime_root": str(runtime_root),
            "runtime_provenance": identity,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY

stability_args=()
for result_dir in "${main_timing_dirs[@]}"; do
  stability_args+=(--main-result-dir "${result_dir}")
done
for result_dir in "${anchor_dirs[@]}"; do
  stability_args+=(--anchor-result-dir "${result_dir}")
done

stability_csv="${audit_dir}/eggpu_timing_stability.csv"
stability_json="${audit_dir}/eggpu_timing_stability.json"
"${python_bin}" "${script_dir}/audit_eggpu_timing_stability.py" \
  "${stability_args[@]}" \
  --metric e2e \
  --expected-samples 5 \
  --expected-cells 208 \
  --max-over-median-limit 5.0 \
  --median-over-min-limit 3.0 \
  --output-csv "${stability_csv}" \
  --output-json "${stability_json}"

overlay_timing_args=()
for result_dir in "${main_timing_dirs[@]}"; do
  overlay_timing_args+=(--main-timing-result-dir "${result_dir}")
done
for result_dir in "${anchor_dirs[@]}"; do
  overlay_timing_args+=(--anchor-timing-result-dir "${result_dir}")
done

overlay_memory_args=()
for result_dir in "${main_memory_dirs[@]}"; do
  overlay_memory_args+=(--main-memory-result-dir "${result_dir}")
done
for result_dir in "${anchor_dirs[@]}"; do
  overlay_memory_args+=(--anchor-memory-result-dir "${result_dir}")
done

overlay_ledger="${audit_dir}/eggpu_v15_unified_overlay_ledger.csv"
overlay_json="${audit_dir}/EGGPU_V15_UNIFIED_OVERLAY_AUDIT.json"
overlay_common_args=(
  --base-ledger "${base_ledger}"
  --candidate-mode unified-timing-memory
  "${overlay_timing_args[@]}"
  "${overlay_memory_args[@]}"
  --stability-audit "${stability_json}"
  --stability-max-over-median-limit 5.0
  --stability-median-over-min-limit 3.0
  --candidate-sha256 "${expected_candidate_sha256}"
  --expected-timing-replacements 208
  --expected-memory-replacements 208
  --output-ledger "${overlay_ledger}"
  --audit-json "${overlay_json}"
)

"${python_bin}" "${script_dir}/update_final13_eggpu_uniform_20260729.py" \
  "${overlay_common_args[@]}" \
  --dry-run \
  | tee "${audit_dir}/overlay_dry_run.stdout.jsonl"

"${python_bin}" "${script_dir}/update_final13_eggpu_uniform_20260729.py" \
  "${overlay_common_args[@]}"

"${python_bin}" "${script_dir}/verify_v10_uniform_overlay_gate.py" \
  --stability-json "${stability_json}" \
  --overlay-json "${overlay_json}" \
  --ledger "${overlay_ledger}" \
  --candidate-sha256 "${expected_candidate_sha256}" \
  --runtime-sha256 "${expected_runtime_sha256}" \
  | tee "${audit_dir}/overlay_gate.stdout.jsonl"

cp -- \
  "${base_assets}/paper_table_baseline_versions.csv" \
  "${output_dir}/paper_table_baseline_versions.csv"
cp -- \
  "${base_assets}/paper_baseline_versions.json" \
  "${output_dir}/paper_baseline_versions.json"

"${python_bin}" "${script_dir}/prepare_uniform_minimum_ledger_20260730.py" \
  --base-ledger "${overlay_ledger}" \
  --main-samples "${main_samples}" \
  --closeness-evidence "${closeness_evidence}" \
  --graphscope-samples "${graphscope_samples}" \
  --cpu-large-matrix "${cpu_large_matrix}" \
  --nxcugraph-samples "${nxcugraph_samples}" \
  --gunrock-samples "${gunrock_samples}" \
  --estimator-policy eggpu-minimum-baseline-mean \
  --output-dir "${output_dir}"

final_ledger="${output_dir}/final_13_cell_outcome_ledger.csv"
MPLCONFIGDIR="${mplconfig_dir}" \
  "${python_bin}" "${script_dir}/generate_final_13_complete_assets_graphscope.py" \
    --ledger "${final_ledger}" \
    --output-dir "${output_dir}"

MPLCONFIGDIR="${mplconfig_dir}" PYTHONPATH="${script_dir}" \
  "${python_bin}" - \
    "${output_dir}" \
    "${final_ledger}" <<'PY'
import sys
from pathlib import Path

import generate_chapter4_evaluation_assets_graphscope as chapter4

output = Path(sys.argv[1]).resolve(strict=True)
ledger = Path(sys.argv[2]).resolve(strict=True)
chapter4.setup_style()
chapter4.category_overview_figure(
    source_dir=output,
    output_dir=output,
    timing_ledger=ledger,
    historical_summary=None,
    raw_samples=output / "five_run_raw_samples.csv",
)
PY

"${python_bin}" "${script_dir}/generate_headline_results_20260730.py" \
  --asset-dir "${output_dir}" \
  --output-dir "${output_dir}" \
  --estimator-policy eggpu-minimum-baseline-mean \
  | tee "${audit_dir}/headline_results.stdout.jsonl"

"${python_bin}" "${script_dir}/generate_estimator_sensitivity_20260730.py" \
  --ledger "${final_ledger}" \
  --output-dir "${output_dir}" \
  | tee "${audit_dir}/estimator_sensitivity.stdout.jsonl"

"${python_bin}" - \
  "${base_ledger}" \
  "${final_ledger}" \
  "${output_dir}/five_run_raw_samples.csv" \
  "${output_dir}/MIXED_ESTIMATOR_FIVE_RUN_MANIFEST.json" \
  "${output_dir}/mixed_estimator_audit.csv" \
  "${stability_json}" \
  "${overlay_json}" \
  "${audit_dir}/V15_PAGERANK_PROTOCOL_CORRECTION_AUDIT.json" \
  "${audit_dir}/pagerank_protocol_action_ledger.csv" \
  "${audit_dir}/non_pagerank_cell_hashes.csv" \
  "${audit_dir}/main_timing_dirs.txt" \
  "${audit_dir}/anchor_timing_dirs.txt" \
  "${expected_candidate_sha256}" \
  "${expected_runtime_sha256}" \
  "${output_dir}/V15_FINAL_VERIFICATION.json" <<'PY'
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import pandas as pd

(
    base_path,
    final_path,
    raw_path,
    manifest_path,
    mixed_audit_path,
    stability_path,
    overlay_path,
    correction_audit_path,
    correction_action_path,
    correction_non_pr_path,
    correction_main_list_path,
    correction_anchor_list_path,
    expected_candidate,
    expected_runtime,
    verification_path,
) = sys.argv[1:]
base_path = Path(base_path).resolve(strict=True)
final_path = Path(final_path).resolve(strict=True)
raw_path = Path(raw_path).resolve(strict=True)
manifest_path = Path(manifest_path).resolve(strict=True)
mixed_audit_path = Path(mixed_audit_path).resolve(strict=True)
stability_path = Path(stability_path).resolve(strict=True)
overlay_path = Path(overlay_path).resolve(strict=True)
correction_audit_path = Path(correction_audit_path).resolve(strict=True)
correction_action_path = Path(correction_action_path).resolve(strict=True)
correction_non_pr_path = Path(correction_non_pr_path).resolve(strict=True)
correction_main_list_path = Path(correction_main_list_path).resolve(strict=True)
correction_anchor_list_path = Path(correction_anchor_list_path).resolve(strict=True)
verification_path = Path(verification_path)

EXPECTED_LEDGER_ROWS = 1664
EXPECTED_SUCCESSFUL_CELLS = 1040
EXPECTED_DISPLAYED_METRICS = 3120
EXPECTED_RAW_ROWS = 15600
EXPECTED_EGGPU_CELLS = 208
EXPECTED_PAGERANK_REPLACEMENTS = 11
EXPECTED_PAGERANK_ANCHORS = 2
EXPECTED_NON_PAGERANK_CELLS = 195
METRICS = ("build", "kernel", "e2e")
KEY = ("dataset", "function", "baseline")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(left, right):
    return math.isclose(
        float(left),
        float(right),
        rel_tol=2.0e-8,
        abs_tol=2.0e-11,
    )


def numeric_equivalent(left, right):
    try:
        lhs = float(left)
        rhs = float(right)
    except (TypeError, ValueError):
        return False
    return math.isfinite(lhs) and math.isfinite(rhs) and math.isclose(
        lhs,
        rhs,
        rel_tol=1.0e-12,
        abs_tol=1.0e-15,
    )


final = pd.read_csv(final_path, low_memory=False)
raw = pd.read_csv(raw_path, low_memory=False)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
mixed_audit = pd.read_csv(mixed_audit_path, low_memory=False)
stability = json.loads(stability_path.read_text(encoding="utf-8"))
overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
correction = json.loads(correction_audit_path.read_text(encoding="utf-8"))

require(len(final) == EXPECTED_LEDGER_ROWS, f"ledger rows={len(final)}")
require(
    not final.duplicated(list(KEY)).any(),
    "final ledger contains duplicate dataset/function/baseline keys",
)
successful = final[final["execution_status"].eq("ok")].copy()
require(
    len(successful) == EXPECTED_SUCCESSFUL_CELLS,
    f"successful cells={len(successful)}",
)
require(len(raw) == EXPECTED_RAW_ROWS, f"portable raw rows={len(raw)}")

manifest_expected = {
    "schema_version": "eggpu_minimum_baseline_mean_five_run_v1",
    "status": "pass",
    "display_estimator": "mixed_by_baseline_class",
    "display_estimator_by_baseline_class": {
        "EGGPU": "minimum_of_five",
        "external_baselines": "arithmetic_mean_of_five",
    },
    "reported_error": "sample_standard_deviation_ddof1",
    "sample_count_per_displayed_metric": 5,
    "successful_cells": EXPECTED_SUCCESSFUL_CELLS,
    "displayed_metrics": EXPECTED_DISPLAYED_METRICS,
    "portable_raw_rows": EXPECTED_RAW_ROWS,
}
for field, expected in manifest_expected.items():
    require(
        manifest.get(field) == expected,
        f"mixed manifest {field}={manifest.get(field)!r}, expected {expected!r}",
    )
require(
    manifest.get("output_ledger_sha256") == sha256(final_path),
    "mixed manifest ledger hash differs",
)
require(
    manifest.get("portable_raw_samples_sha256") == sha256(raw_path),
    "mixed manifest raw-sample hash differs",
)
require(
    manifest.get("overlay_audit_sha256") == sha256(mixed_audit_path),
    "mixed manifest estimator-audit hash differs",
)

group_columns = ["dataset", "function", "baseline", "metric"]
groups = raw.groupby(group_columns, sort=False, dropna=False)
require(len(groups) == EXPECTED_DISPLAYED_METRICS, f"raw groups={len(groups)}")
expected_keys = {
    (str(row.dataset), str(row.function), str(row.baseline), metric)
    for row in successful.itertuples(index=False)
    for metric in METRICS
}
observed_keys = {
    tuple(map(str, key))
    for key in groups.groups
}
require(observed_keys == expected_keys, "portable raw keys differ from successful cells")

for key, samples in groups:
    require(len(samples) == 5, f"{key}: raw sample count={len(samples)}")
    indices = sorted(int(value) for value in samples["sample_index"])
    require(indices == [1, 2, 3, 4, 5], f"{key}: sample indices={indices}")
    values = pd.to_numeric(samples["seconds"], errors="coerce")
    require(values.notna().all(), f"{key}: nonnumeric raw timing")
    require((values > 0).all(), f"{key}: nonpositive raw timing")

for row in successful.itertuples(index=False):
    for metric in METRICS:
        key = (str(row.dataset), str(row.function), str(row.baseline), metric)
        samples = groups.get_group(key)
        values = pd.to_numeric(samples["seconds"], errors="raise")
        observed_minimum = float(values.min())
        observed_mean = float(values.mean())
        observed_sample_sd = float(values.std(ddof=1))
        displayed = getattr(row, f"{metric}_paper_seconds")
        estimator = getattr(row, f"{metric}_estimator")
        reported_error = getattr(row, f"{metric}_std_seconds")
        expected_estimator = (
            "minimum_of_five"
            if str(row.baseline) == "EGGPU"
            else "arithmetic_mean_of_five"
        )
        expected_displayed = (
            observed_minimum
            if expected_estimator == "minimum_of_five"
            else observed_mean
        )
        require(
            estimator == expected_estimator,
            f"{key}: estimator={estimator!r}, expected={expected_estimator!r}",
        )
        require(
            close(displayed, expected_displayed),
            f"{key}: displayed={displayed}, raw estimate={expected_displayed}",
        )
        require(
            close(reported_error, observed_sample_sd),
            f"{key}: reported error={reported_error}, "
            f"sample SD={observed_sample_sd}",
        )
        require(
            int(row.sample_count) == 5,
            f"{key}: ledger sample_count={row.sample_count!r}",
        )

audit_key_columns = ["dataset", "function", "baseline", "metric"]
audit_required_columns = {
    *audit_key_columns,
    "new_paper_seconds",
    "new_estimator",
    "raw_mean_seconds",
    "raw_min_seconds",
    "sample_sd_seconds",
    "sample_count",
    "action",
}
require(
    audit_required_columns.issubset(mixed_audit.columns),
    "mixed estimator audit lacks required columns",
)
require(
    len(mixed_audit) == EXPECTED_DISPLAYED_METRICS,
    f"mixed estimator audit rows={len(mixed_audit)}",
)
require(
    not mixed_audit.duplicated(audit_key_columns).any(),
    "mixed estimator audit contains duplicate metric keys",
)
audit_by_key = mixed_audit.set_index(audit_key_columns, drop=False)
require(
    {tuple(map(str, key)) for key in audit_by_key.index} == expected_keys,
    "mixed estimator audit keys differ from successful cells",
)
successful_by_key = successful.set_index(list(KEY), drop=False)
for key, audit_row in audit_by_key.iterrows():
    dataset, function, baseline, metric = tuple(map(str, key))
    ledger_row = successful_by_key.loc[(dataset, function, baseline)]
    raw_values = pd.to_numeric(
        groups.get_group((dataset, function, baseline, metric))["seconds"],
        errors="raise",
    )
    expected_estimator = (
        "minimum_of_five"
        if baseline == "EGGPU"
        else "arithmetic_mean_of_five"
    )
    require(
        str(audit_row["new_estimator"]) == expected_estimator,
        f"{key}: mixed audit estimator differs",
    )
    require(
        close(
            audit_row["new_paper_seconds"],
            ledger_row[f"{metric}_paper_seconds"],
        ),
        f"{key}: mixed audit displayed value differs from ledger",
    )
    require(
        close(
            audit_row["sample_sd_seconds"],
            ledger_row[f"{metric}_std_seconds"],
        ),
        f"{key}: mixed audit sample SD differs from ledger",
    )
    require(
        close(audit_row["raw_mean_seconds"], raw_values.mean())
        and close(audit_row["raw_min_seconds"], raw_values.min())
        and close(audit_row["sample_sd_seconds"], raw_values.std(ddof=1)),
        f"{key}: mixed audit descriptive statistics differ from raw samples",
    )
    require(
        int(audit_row["sample_count"]) == 5,
        f"{key}: mixed audit sample count differs",
    )
    require(
        str(audit_row["action"]).endswith(expected_estimator),
        f"{key}: mixed audit action differs",
    )

eggpu = final[final["baseline"].eq("EGGPU")]
require(len(eggpu) == EXPECTED_EGGPU_CELLS, f"EGGPU cells={len(eggpu)}")
require(
    eggpu["execution_status"].eq("ok").all(),
    "an EGGPU cell is not successful",
)
for column, expected in (
    ("candidate_sha256", expected_candidate),
    ("runtime_python_snapshot_sha256", expected_runtime),
    ("memory_candidate_sha256", expected_candidate),
    ("memory_runtime_python_snapshot_sha256", expected_runtime),
):
    require(
        set(eggpu[column].astype(str)) == {expected},
        f"EGGPU {column} differs",
    )

require(
    stability.get("status") == "pass"
    and stability.get("audited_cells") == EXPECTED_EGGPU_CELLS
    and stability.get("unique_workload_keys") == EXPECTED_EGGPU_CELLS
    and stability.get("expected_samples_per_cell") == 5
    and stability.get("candidate_sha256") == expected_candidate
    and stability.get("runtime_python_snapshot_sha256") == expected_runtime,
    "stability identity/count gate differs",
)
require(
    overlay.get("status") == "pass"
    and overlay.get("candidate_mode") == "unified-timing-memory"
    and overlay.get("timing_replacement_count") == EXPECTED_EGGPU_CELLS
    and overlay.get("memory_replacement_count") == EXPECTED_EGGPU_CELLS
    and overlay.get("candidate_sha256") == [expected_candidate]
    and overlay.get("runtime_python_snapshot_sha256") == [expected_runtime],
    "unified overlay identity/count gate differs",
)

correction_actions = correction.get("actions")
correction_non_pr = correction.get("non_pagerank_identity")
require(
    correction.get("status") == "pass"
    and correction.get("protocol")
    == "eggpu_pagerank_direct_public_call_fail_closed_v1"
    and correction.get("candidate_sha256") == expected_candidate
    and correction.get("runtime_python_snapshot_sha256") == expected_runtime
    and correction.get("adoption_policy")
    == "protocol_validity_only_unconditional_no_relative_timing_selection"
    and correction.get("relative_performance_considered_for_adoption") is False
    and correction.get("relative_speed_comparison_performed") is False,
    "PageRank correction identity/protocol gate differs",
)
require(
    isinstance(correction_actions, dict)
    and correction_actions.get("replace_protocol_invalid")
    == EXPECTED_PAGERANK_REPLACEMENTS
    and correction_actions.get("retain_equivalent_direct_protocol")
    == EXPECTED_PAGERANK_ANCHORS
    and correction_actions.get("total_pagerank_cells")
    == EXPECTED_PAGERANK_REPLACEMENTS + EXPECTED_PAGERANK_ANCHORS,
    "PageRank correction action counts differ",
)
require(
    isinstance(correction_non_pr, dict)
    and correction_non_pr.get("status") == "pass"
    and correction_non_pr.get("normalized_cell_count")
    == EXPECTED_NON_PAGERANK_CELLS
    and correction_non_pr.get("before_sha256")
    == correction_non_pr.get("after_sha256")
    and correction_non_pr.get("all_cell_hashes_equal") is True,
    "PageRank correction non-PageRank identity gate differs",
)
with correction_action_path.open(newline="", encoding="utf-8") as handle:
    correction_action_rows = list(csv.DictReader(handle))
require(
    len(correction_action_rows)
    == EXPECTED_PAGERANK_REPLACEMENTS + EXPECTED_PAGERANK_ANCHORS,
    f"PageRank action ledger rows={len(correction_action_rows)}",
)
require(
    len({row.get("dataset") for row in correction_action_rows})
    == len(correction_action_rows)
    and {row.get("function") for row in correction_action_rows} == {"PageRank"},
    "PageRank action ledger keys differ",
)
require(
    sum(
        row.get("action") == "replace_protocol_invalid"
        for row in correction_action_rows
    )
    == EXPECTED_PAGERANK_REPLACEMENTS
    and sum(
        row.get("action") == "retain_equivalent_direct_protocol"
        for row in correction_action_rows
    )
    == EXPECTED_PAGERANK_ANCHORS
    and all(
        str(row.get("relative_performance_considered_for_adoption", "")).lower()
        == "false"
        for row in correction_action_rows
    ),
    "PageRank action ledger semantics differ",
)
with correction_non_pr_path.open(newline="", encoding="utf-8") as handle:
    correction_non_pr_rows = list(csv.DictReader(handle))
require(
    len(correction_non_pr_rows) == EXPECTED_NON_PAGERANK_CELLS
    and all(
        row.get("function") != "PageRank"
        and row.get("before_sha256") == row.get("after_sha256")
        for row in correction_non_pr_rows
    ),
    "PageRank correction 195-cell identity ledger differs",
)
main_timing_lines = correction_main_list_path.read_text(
    encoding="utf-8"
).splitlines()
anchor_timing_lines = correction_anchor_list_path.read_text(
    encoding="utf-8"
).splitlines()
expected_main_lines = [
    *correction.get("combined_main_timing_dirs", []),
    *correction.get("supplemental_main_timing_dirs", []),
]
expected_anchor_lines = correction.get("anchor_timing_dirs", [])
require(
    main_timing_lines == expected_main_lines and bool(main_timing_lines),
    "copied PageRank main timing list differs from correction audit",
)
require(
    anchor_timing_lines == expected_anchor_lines
    and len(anchor_timing_lines) > 0,
    "copied PageRank anchor timing list differs from correction audit",
)
correction_output_hashes = correction.get("output_file_sha256")
require(
    isinstance(correction_output_hashes, dict),
    "PageRank correction output hash manifest is absent",
)
for name, path in (
    ("pagerank_protocol_action_ledger.csv", correction_action_path),
    ("non_pagerank_cell_hashes.csv", correction_non_pr_path),
    ("main_timing_dirs.txt", correction_main_list_path),
    ("anchor_timing_dirs.txt", correction_anchor_list_path),
):
    require(
        correction_output_hashes.get(name) == sha256(path),
        f"PageRank correction hash differs for {name}",
    )

with base_path.open(newline="", encoding="utf-8") as handle:
    base_reader = csv.DictReader(handle)
    base_fields = list(base_reader.fieldnames or [])
    base_rows = [row for row in base_reader if row["baseline"] != "EGGPU"]
with final_path.open(newline="", encoding="utf-8") as handle:
    final_reader = csv.DictReader(handle)
    final_fields = list(final_reader.fieldnames or [])
    final_rows = [row for row in final_reader if row["baseline"] != "EGGPU"]
require(base_fields == final_fields, "non-EGGPU ledger columns changed")
require(
    len(base_rows) == len(final_rows) == EXPECTED_LEDGER_ROWS - EXPECTED_EGGPU_CELLS,
    "non-EGGPU row count changed",
)
display_derived_fields = {
    f"{metric}_{suffix}"
    for metric in METRICS
    for suffix in (
        "paper_seconds",
        "raw_mean_seconds",
        "std_seconds",
        "estimator",
        "raw_min_seconds",
        "raw_median_seconds",
        "raw_max_seconds",
        "coefficient_of_variation",
        "max_over_median",
        "median_over_minimum",
        "variance_policy",
        "stability_status",
    )
}
display_derived_fields.add("sample_count")
base_rows.sort(key=lambda row: tuple(row[field] for field in KEY))
final_rows.sort(key=lambda row: tuple(row[field] for field in KEY))
differences = []
display_field_changes = 0
provenance_text_changes = 0
for before, after in zip(base_rows, final_rows):
    before_key = tuple(before[field] for field in KEY)
    after_key = tuple(after[field] for field in KEY)
    if before_key != after_key:
        differences.append({"key_before": before_key, "key_after": after_key})
        continue
    for field in base_fields:
        if before[field] == after[field]:
            continue
        if numeric_equivalent(before[field], after[field]):
            continue
        if (
            before.get("execution_status") == "ok"
            and after.get("execution_status") == "ok"
            and field in display_derived_fields
        ):
            display_field_changes += 1
            continue
        if (
            before.get("baseline") == "nx-cugraph"
            and field == "result_source"
            and before[field]
            == "strict nx-cugraph 99-cell rerun; minimum of five"
            and after[field]
            == (
                "strict nx-cugraph 99-cell rerun; arithmetic mean of five "
                "fresh processes"
            )
        ):
            provenance_text_changes += 1
            continue
        if (
            before.get("baseline") == "Gunrock"
            and field == "reason"
            and before[field]
            == (
                "strict native Gunrock three-phase timing; minimum of five "
                "fresh processes; full-result validation passed; external "
                "CLI wall substitution forbidden"
            )
            and after[field]
            == (
                "strict native Gunrock three-phase timing; arithmetic mean "
                "of five fresh processes; full-result validation passed; "
                "external CLI wall substitution forbidden"
            )
        ):
            provenance_text_changes += 1
            continue
        differences.append(
            {
                "key": before_key,
                "field": field,
                "before": before[field],
                "after": after[field],
            }
        )
        if len(differences) >= 20:
            break
    if len(differences) >= 20:
        break
require(
    not differences,
    f"non-EGGPU source/provenance fields changed: {differences}",
)
require(
    display_field_changes > 0,
    "mixed estimator did not recompute any non-EGGPU display field",
)
require(
    provenance_text_changes == 189,
    f"normalized external provenance rows={provenance_text_changes}, expected 189",
)

required_assets = (
    "FINAL_13_NUMERICAL_RESULTS.md",
    "final_13_numeric_summary.json",
    "final_13_pairwise_sota_details.csv",
    "pairwise_baseline_13_exact_summary.csv",
    "paper_table_main_compact_best_competitor.tex",
    "paper_table_main_compact_pairwise_speedup.tex",
    "paper_table_headline_results.csv",
    "paper_table_headline_results.tex",
    "paper_table_headline_results_README.md",
    "paper_table_headline_results_manifest.json",
    "paper_table_baseline_versions.csv",
    "paper_table_baseline_versions.tex",
    "paper_baseline_versions.json",
    "equal_mean_estimator_sensitivity.csv",
    "EQUAL_MEAN_ESTIMATOR_SENSITIVITY.json",
    "category_time_by_baseline_3panel.csv",
    "category_time_by_baseline_3panel.metadata.json",
    "category_time_by_baseline_3panel.pdf",
    "category_time_by_baseline_3panel.png",
)
asset_hashes = {}
for name in required_assets:
    path = final_path.parent / name
    require(path.is_file() and path.stat().st_size > 0, f"missing asset: {name}")
    asset_hashes[name] = sha256(path)

category_path = final_path.parent / "category_time_by_baseline_3panel.csv"
category = pd.read_csv(category_path, low_memory=False)
require(
    set(category["metric"].astype(str))
    == {"Graph construction", "Processing time", "End-to-end"},
    "Figure 3 metric titles differ",
)
category_errors = pd.to_numeric(
    category["sample_std_seconds"], errors="coerce"
)
category_sample_counts = pd.to_numeric(
    category["aggregate_sample_count"], errors="coerce"
)
require(
    len(category) > 0
    and category_errors.notna().all()
    and category_errors.ge(0).all()
    and category_sample_counts.eq(5).all(),
    "Figure 3 does not bind every error bar to five-run sample SD",
)
category_metadata = json.loads(
    (
        final_path.parent / "category_time_by_baseline_3panel.metadata.json"
    ).read_text(encoding="utf-8")
)
category_error_contract = category_metadata.get("error_bars")
require(
    isinstance(category_error_contract, dict)
    and category_error_contract.get("samples") == 5
    and category_error_contract.get("statistic")
    == "sample_standard_deviation_ddof1"
    and bool(category_error_contract.get("aggregation"))
    and category_metadata.get("point_estimator", {}).get("current_EGGPU")
    == "minimum_of_five"
    and category_metadata.get("point_estimator", {}).get(
        "external_baselines"
    )
    == "arithmetic_mean_of_five",
    "Figure 3 mixed-estimator/error-bar metadata differs",
)
gunrock_e2e = category[
    category["baseline"].astype(str).eq("Gunrock")
    & category["metric"].astype(str).eq("End-to-end")
]
require(
    len(gunrock_e2e) > 0
    and gunrock_e2e["timing_boundary"]
    .astype(str)
    .eq("standalone load-to-complete-host-result")
    .all(),
    "Figure 3 Gunrock E2E boundary is not the standalone boundary",
)

baseline_versions_path = final_path.parent / "paper_baseline_versions.json"
baseline_versions = json.loads(
    baseline_versions_path.read_text(encoding="utf-8")
)
require(
    baseline_versions.get("eggpu_candidate_binary_sha256")
    == expected_candidate
    and baseline_versions.get("eggpu_runtime_python_snapshot_sha256")
    == expected_runtime
    and (baseline_versions.get("eggpu_provenance") or {}).get(
        "release_label"
    )
    == "V15",
    "baseline-version JSON does not bind the V15 runtime",
)
version_csv = pd.read_csv(
    final_path.parent / "paper_table_baseline_versions.csv",
    keep_default_na=False,
)
version_eggpu = version_csv[
    version_csv["system"].astype(str).eq("EGGPU")
]
require(
    len(version_eggpu) == 1
    and str(version_eggpu.iloc[0]["commit"]) == expected_candidate
    and str(
        version_eggpu.iloc[0]["timing_runtime_python_snapshot_sha256"]
    )
    == expected_runtime
    and "V15" in str(version_eggpu.iloc[0]["version"])
    and "V10" not in str(version_eggpu.iloc[0]["artifact_scope"]),
    "baseline-version CSV does not bind the V15 runtime",
)
version_tex = (
    final_path.parent / "paper_table_baseline_versions.tex"
).read_text(encoding="utf-8")
require(
    "V15 current-runtime candidate" in version_tex
    and "V10" not in version_tex,
    "baseline-version TeX still identifies an earlier candidate",
)

external = final[~final["baseline"].astype(str).eq("EGGPU")]
legacy_external_provenance = external.apply(
    lambda column: column.astype(str).str.contains(
        "minimum of five", case=False, regex=False
    )
).any(axis=1)
require(
    not bool(legacy_external_provenance.any()),
    "external-baseline provenance still claims minimum-of-five timing",
)

sensitivity_path = (
    final_path.parent / "EQUAL_MEAN_ESTIMATOR_SENSITIVITY.json"
)
sensitivity = json.loads(sensitivity_path.read_text(encoding="utf-8"))
sensitivity_expected = {
    "library_public_call_e2e": (167, 161, 0, 6, 8.36),
    "native_gpu_processing": (105, 99, 0, 6, 7.49),
    "strict_nxcugraph_public_call_e2e": (99, 99, 0, 0, 42.05),
}
sensitivity_rows = {
    row["comparison_id"]: row
    for row in sensitivity.get("results", [])
}
require(
    sensitivity.get("status") == "pass"
    and sensitivity.get("source_ledger_sha256") == sha256(final_path)
    and set(sensitivity_rows) == set(sensitivity_expected),
    "equal-mean sensitivity manifest identity differs",
)
for comparison_id, expected in sensitivity_expected.items():
    row = sensitivity_rows[comparison_id]
    observed = (
        int(row["common_pairs"]),
        int(row["eggpu_strict_wins"]),
        int(row["eggpu_ties"]),
        int(row["eggpu_losses"]),
        round(float(row["geomean_speedup"]), 2),
    )
    require(
        observed == expected,
        f"equal-mean sensitivity differs for {comparison_id}",
    )

headline_manifest_path = (
    final_path.parent / "paper_table_headline_results_manifest.json"
)
headline = json.loads(headline_manifest_path.read_text(encoding="utf-8"))
require(
    headline.get("status") == "pass"
    and headline.get("schema_version")
    == "eggpu_headline_results_v2_mixed_estimator",
    "headline-results manifest status/schema differs",
)
headline_inputs = {
    "final_13_cell_outcome_ledger.csv": final_path,
    "final_13_pairwise_sota_details.csv": (
        final_path.parent / "final_13_pairwise_sota_details.csv"
    ),
    "pairwise_baseline_13_exact_summary.csv": (
        final_path.parent / "pairwise_baseline_13_exact_summary.csv"
    ),
    "final_13_numeric_summary.json": (
        final_path.parent / "final_13_numeric_summary.json"
    ),
    "MIXED_ESTIMATOR_FIVE_RUN_MANIFEST.json": manifest_path,
}
require(
    headline.get("input_sha256")
    == {name: sha256(path) for name, path in sorted(headline_inputs.items())},
    "headline-results input hashes differ",
)
headline_outputs = {
    "paper_table_headline_results.csv": (
        final_path.parent / "paper_table_headline_results.csv"
    ),
    "paper_table_headline_results.tex": (
        final_path.parent / "paper_table_headline_results.tex"
    ),
    "paper_table_headline_results_README.md": (
        final_path.parent / "paper_table_headline_results_README.md"
    ),
}
require(
    headline.get("output_sha256")
    == {name: sha256(path) for name, path in sorted(headline_outputs.items())},
    "headline-results output hashes differ",
)
headline_gates = headline.get("gates")
require(
    isinstance(headline_gates, dict)
    and headline_gates
    and all(value is True for value in headline_gates.values()),
    "a headline-results semantic gate is not true",
)
headline_matrix = headline.get("matrix_contract")
headline_estimator = headline.get("estimator_contract")
require(
    isinstance(headline_matrix, dict)
    and headline_matrix.get("ledger_rows") == EXPECTED_LEDGER_ROWS
    and headline_matrix.get("datasets") == 13
    and headline_matrix.get("functions") == 16
    and headline_matrix.get("workloads") == EXPECTED_EGGPU_CELLS,
    "headline-results matrix contract differs",
)
require(
    isinstance(headline_estimator, dict)
    and headline_estimator.get("display_estimator")
    == "mixed_by_baseline_class"
    and headline_estimator.get("policy")
    == {
        "EGGPU": "minimum_of_five",
        "baselines": "arithmetic_mean_of_five",
        "error_bar": "sample_standard_deviation_ddof1",
        "samples": 5,
    }
    and headline_estimator.get("retained_samples_per_displayed_metric") == 5
    and headline_estimator.get("successful_cells")
    == EXPECTED_SUCCESSFUL_CELLS
    and headline_estimator.get("displayed_metrics")
    == EXPECTED_DISPLAYED_METRICS
    and headline_estimator.get("portable_raw_rows") == EXPECTED_RAW_ROWS,
    "headline-results estimator contract differs",
)

verification = {
    "status": "pass",
    "candidate_sha256": expected_candidate,
    "runtime_python_snapshot_sha256": expected_runtime,
    "ledger_rows": len(final),
    "successful_cells": len(successful),
    "displayed_metrics": len(groups),
    "portable_raw_rows": len(raw),
    "samples_per_displayed_metric": 5,
    "estimator_policy": "eggpu-minimum-baseline-mean",
    "display_estimator": "mixed_by_baseline_class",
    "display_estimator_by_baseline_class": {
        "EGGPU": "minimum_of_five",
        "external_baselines": "arithmetic_mean_of_five",
    },
    "reported_error": "sample_standard_deviation_ddof1",
    "eggpu_cells": len(eggpu),
    "non_eggpu_rows": len(final_rows),
    "non_eggpu_evidence_fields_unchanged": True,
    "non_eggpu_display_fields_recomputed": True,
    "non_eggpu_display_field_changes": display_field_changes,
    "external_estimator_provenance_text_normalized": True,
    "external_estimator_provenance_text_changes": provenance_text_changes,
    "non_eggpu_comparison": (
        "Measurement evidence fields are exact strings, allowing only "
        "numerically equivalent CSV renderings at rtol=1e-12/atol=1e-15. "
        "Five-run display/statistic fields are recomputed, and 189 legacy "
        "estimator-description strings are normalized from minimum to "
        "arithmetic mean without changing their underlying samples."
    ),
    "ledger_sha256": sha256(final_path),
    "raw_samples_sha256": sha256(raw_path),
    "mixed_manifest_sha256": sha256(manifest_path),
    "mixed_estimator_audit_sha256": sha256(mixed_audit_path),
    "stability_audit_sha256": sha256(stability_path),
    "overlay_audit_sha256": sha256(overlay_path),
    "pagerank_protocol_corrected": True,
    "pagerank_protocol_correction": {
        "protocol": correction.get("protocol"),
        "replace_protocol_invalid": EXPECTED_PAGERANK_REPLACEMENTS,
        "retain_equivalent_direct_protocol": EXPECTED_PAGERANK_ANCHORS,
        "non_pagerank_cells_unchanged": EXPECTED_NON_PAGERANK_CELLS,
        "correction_audit_sha256": sha256(correction_audit_path),
        "action_ledger_sha256": sha256(correction_action_path),
        "non_pagerank_cell_hashes_sha256": sha256(correction_non_pr_path),
        "main_timing_dirs_sha256": sha256(correction_main_list_path),
        "anchor_timing_dirs_sha256": sha256(correction_anchor_list_path),
    },
    "headline_results": {
        "status": headline.get("status"),
        "schema_version": headline.get("schema_version"),
        "manifest_sha256": sha256(headline_manifest_path),
        "input_sha256": headline.get("input_sha256"),
        "output_sha256": headline.get("output_sha256"),
        "all_semantic_gates_pass": True,
    },
    "asset_sha256": asset_hashes,
}
verification_path.write_text(
    json.dumps(verification, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(verification, sort_keys=True))
PY

rm -f -- "${incomplete_marker}"
completed=1
trap - EXIT

echo "Published verified V15 current-runtime assets: ${output_dir}"
echo "Candidate SHA-256: ${expected_candidate_sha256}"
echo "Runtime Python SHA-256: ${expected_runtime_sha256}"
