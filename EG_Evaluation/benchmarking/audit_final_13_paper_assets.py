#!/usr/bin/env python3
"""Audit the final 13-dataset paper bundle against its raw evidence.

All thirteen datasets enter the 16-function main matrix.  A cell may contain a
validated measurement or an explicit outcome such as timeout, representation
limit, semantic mismatch, or unsupported API.  No attempted cell may silently
disappear, and only validated successful cells enter performance comparisons.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


CROSS_LIBRARY = [
    "ca-HepTh",
    "LastFM",
    "p2p-Gnutella04",
    "ca-HepPh",
    "email-Enron",
    "ca-CondMat",
    "soc-Epinions1",
    "soc-Slashdot0811",
    "ER-100k",
    "web-NotreDame",
    "com-youtube",
]
SCALE_ANCHORS = ["com-Orkut", "GAP-twitter"]
FINAL_13 = CROSS_LIBRARY + SCALE_ANCHORS
SAMPLED_CLOSENESS = {
    "ER-100k",
    "soc-Slashdot0811",
    "web-NotreDame",
    "com-youtube",
}
FUNCTION_COUNT = 16
BASELINES_REQUIRED = {
    "networkx",
    "easygraph-cpu",
    "easygraph-cpp",
    "igraph",
    "nx-cugraph",
    "Gunrock",
    "EGGPU",
}


def add_check(checks: list[dict], name: str, passed: bool, detail: str) -> None:
    checks.append({"check": name, "status": "pass" if passed else "fail", "detail": detail})


def require_columns(data: pd.DataFrame, columns: set[str]) -> set[str]:
    return columns - set(data.columns)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit_dataset_values(
    datasets: pd.DataFrame,
    main_result: Path,
    anchor_manifests: list[Path],
) -> tuple[bool, str]:
    """Cross-check paper dataset counts against raw run/manifests."""
    expected: dict[str, dict] = {}
    for row in load_json(main_result / "dataset_stats.json"):
        name = str(row["name"])
        if name not in CROSS_LIBRARY:
            continue
        directed = str(row["graph_type"]) == "directed"
        simple_edges = (
            int(row["edges_directed_unique"])
            if directed
            else int(row["edges_undirected_unique"])
        )
        expected[name] = {
            "nodes": int(row["nodes_raw"]),
            "simple_edges": simple_edges,
            "csr_entries": (
                simple_edges if directed else 2 * simple_edges
            ),
            "self_loops": int(row["selfloops"]),
            "directed": directed,
        }
    for path in anchor_manifests:
        manifest = load_json(path)
        expected[str(manifest["name"])] = {
            "nodes": int(manifest["num_nodes"]),
            "simple_edges": int(manifest["num_edges"]),
            "csr_entries": int(manifest["num_entries"]),
            "self_loops": int(manifest["self_loops_removed"]),
            "directed": bool(manifest["directed"]),
        }

    mismatches = []
    lookup = datasets.set_index("dataset")
    for name in FINAL_13:
        if name not in expected or name not in lookup.index:
            mismatches.append(f"{name}:missing")
            continue
        observed = lookup.loc[name]
        for field, value in expected[name].items():
            got = observed[field]
            if field == "directed":
                if bool(got) != value:
                    mismatches.append(f"{name}.{field}={got}!={value}")
            elif int(got) != value:
                mismatches.append(f"{name}.{field}={got}!={value}")
    return not mismatches, "mismatches=" + repr(mismatches)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--main-result", required=True, type=Path)
    parser.add_argument("--closeness-result", required=True, type=Path)
    parser.add_argument("--core-result", required=True, type=Path)
    parser.add_argument("--followup-result", required=True, type=Path)
    parser.add_argument("--anchor-manifest", action="append", required=True, type=Path)
    args = parser.parse_args()

    assets = args.assets.resolve()
    checks: list[dict] = []

    datasets = pd.read_csv(assets / "paper_table_datasets_13.csv")
    add_check(
        checks,
        "dataset_scope",
        datasets["dataset"].tolist() == FINAL_13,
        f"rows={len(datasets)}, full_main=13, historical=11, scale_anchors=2",
    )
    add_check(
        checks,
        "dataset_structure_fields",
        not require_columns(
            datasets,
            {
                "nodes",
                "simple_edges",
                "csr_entries",
                "avg_degree",
                "max_degree",
                "density",
                "self_loops",
                "directed",
                "role",
            },
        )
        and (datasets[["nodes", "simple_edges", "csr_entries"]].min().min() > 0),
        "node, edge, CSR, degree, density, loop, direction, and role fields present",
    )
    values_ok, values_detail = audit_dataset_values(
        datasets,
        args.main_result.resolve(),
        [path.resolve() for path in args.anchor_manifest],
    )
    add_check(checks, "dataset_values_match_raw_evidence", values_ok, values_detail)

    ledger = pd.read_csv(assets / "final_13_cell_outcome_ledger.csv", low_memory=False)
    baselines = set(ledger["baseline"].astype(str))
    expected_cells = len(FINAL_13) * FUNCTION_COUNT * len(baselines)
    add_check(
        checks,
        "full_13_by_16_outcome_matrix",
        len(ledger) == expected_cells
        and ledger["dataset"].nunique() == len(FINAL_13)
        and ledger["function"].nunique() == FUNCTION_COUNT
        and BASELINES_REQUIRED.issubset(baselines),
        f"cells={len(ledger)}/{expected_cells}, datasets={ledger['dataset'].nunique()}, functions={ledger['function'].nunique()}, baselines={sorted(baselines)}",
    )
    missing = ledger["execution_status"].eq("missing_experiment")
    non_ok = ~ledger["execution_status"].eq("ok")
    reasons = ledger.get("reason", pd.Series(index=ledger.index, dtype=object)).fillna("").astype(str).str.strip()
    add_check(
        checks,
        "all_main_cells_classified",
        not bool(missing.any()) and bool(reasons[non_ok].ne("").all()),
        f"missing_experiment={int(missing.sum())}, non_ok_without_reason={int((non_ok & reasons.eq('')).sum())}",
    )
    forbidden_numeric = pd.Series(False, index=ledger.index)
    for column in (
        "e2e_paper_seconds",
        "e2e_raw_mean_seconds",
        "kernel_paper_seconds",
        "kernel_raw_mean_seconds",
    ):
        if column in ledger.columns:
            forbidden_numeric |= pd.to_numeric(
                ledger[column], errors="coerce"
            ).notna()
    f_rows = ledger["support_class"].eq("F")
    add_check(
        checks,
        "support_contract_excludes_f_results",
        not bool((f_rows & forbidden_numeric).any())
        and not bool((f_rows & ledger["execution_status"].eq("ok")).any()),
        f"F_with_numeric={int((f_rows & forbidden_numeric).sum())}, "
        f"F_with_ok_status={int((f_rows & ledger['execution_status'].eq('ok')).sum())}",
    )
    eggpu = ledger[ledger["baseline"].eq("EGGPU")]
    add_check(
        checks,
        "eggpu_all_function_attempts",
        len(eggpu) == len(FINAL_13) * FUNCTION_COUNT
        and not eggpu["execution_status"].eq("missing_experiment").any(),
        f"classified EGGPU workloads={len(eggpu)}/{len(FINAL_13) * FUNCTION_COUNT}, ok={int(eggpu['execution_status'].eq('ok').sum())}",
    )

    pairwise = pd.read_csv(assets / "final_13_pairwise_sota_details.csv")
    for metric in ("e2e", "kernel"):
        part = pairwise[pairwise["metric"].eq(metric)]
        add_check(
            checks,
            f"{metric}_pairwise_shape",
            len(part) == len(FINAL_13) * FUNCTION_COUNT
            and part["dataset"].nunique() == len(FINAL_13)
            and part["function"].nunique() == FUNCTION_COUNT,
            f"rows={len(part)}/{len(FINAL_13) * FUNCTION_COUNT}, successful_EGGPU={int(part['eggpu_status'].eq('ok').sum())}",
        )

    closeness_e2e = pd.read_csv(
        args.closeness_result.resolve() / "closeness_large_sampled_e2e.csv"
    )
    closeness_eggpu = closeness_e2e[
        closeness_e2e["dataset"].isin(SAMPLED_CLOSENESS)
        & closeness_e2e["baseline"].eq("EGGPU")
        & closeness_e2e["status"].eq("ok")
    ]
    closeness_counts = closeness_eggpu.groupby("dataset").size().to_dict()
    add_check(
        checks,
        "closeness_merged_samples",
        set(closeness_counts) == SAMPLED_CLOSENESS
        and set(closeness_counts.values()) == {5},
        f"EGGPU E2E sample counts={closeness_counts}; semantic=exact_selected_vertices",
    )

    core = args.core_result.resolve()
    followup = args.followup_result.resolve()
    scale = pd.read_csv(core / "eggpu_large_matrix" / "scaling_all.csv")
    scale_timing_all = scale[
        scale["dataset"].isin(SCALE_ANCHORS)
        & scale["measurement"].eq("timing")
    ]
    scale_timing = scale_timing_all[scale_timing_all["status"].eq("ok")]
    add_check(
        checks,
        "scale_anchor_all_function_timing_attempts",
        len(scale_timing_all) == len(SCALE_ANCHORS) * FUNCTION_COUNT
        and scale_timing_all[["dataset", "function"]].drop_duplicates().shape[0]
        == len(SCALE_ANCHORS) * FUNCTION_COUNT
        and not scale_timing_all["status"].isin(["", "missing_experiment", "unknown"]).any(),
        f"classified timing rows={len(scale_timing_all)}/{len(SCALE_ANCHORS) * FUNCTION_COUNT}, ok={len(scale_timing)}",
    )
    add_check(
        checks,
        "scale_anchor_successful_timing_contract",
        len(scale_timing) > 0
        and scale_timing["timing_process_samples"].eq(5).all()
        and scale_timing["result_validation"].eq("pass").all(),
        f"validated timing rows={len(scale_timing)}, timing samples={sorted(scale_timing['timing_process_samples'].dropna().unique().tolist())}",
    )
    scale_memory = scale[
        scale["dataset"].isin(SCALE_ANCHORS)
        & scale["measurement"].eq("memory")
        & scale["status"].eq("ok")
    ]
    memory_counts = scale_memory.groupby(["dataset", "function"]).size()
    successful_pairs = set(map(tuple, scale_timing[["dataset", "function"]].to_numpy()))
    memory_pairs = set(memory_counts.index.tolist())
    add_check(
        checks,
        "scale_anchor_memory",
        memory_pairs == successful_pairs and memory_counts.eq(3).all(),
        f"memory groups={len(memory_counts)}, samples_per_group={sorted(memory_counts.unique().tolist())}",
    )

    rmat = pd.read_csv(followup / "eggpu_rmat_four_functions" / "scaling_all.csv")
    rmat_timing = rmat[rmat["measurement"].eq("timing") & rmat["status"].eq("ok")]
    rmat_memory = rmat[rmat["measurement"].eq("memory") & rmat["status"].eq("ok")]
    rmat_memory_counts = rmat_memory.groupby(["dataset", "function"]).size()
    add_check(
        checks,
        "controlled_rmat_timing",
        len(rmat_timing) == 16
        and rmat_timing["timing_process_samples"].eq(5).all()
        and rmat_timing["result_validation"].eq("pass").all(),
        f"validated timing rows={len(rmat_timing)}, expected=4 scales x 4 functions",
    )
    add_check(
        checks,
        "controlled_rmat_memory",
        len(rmat_memory_counts) == 16 and rmat_memory_counts.eq(3).all(),
        f"memory groups={len(rmat_memory_counts)}, samples_per_group={sorted(rmat_memory_counts.unique().tolist())}",
    )

    nx_scale = pd.read_csv(
        core / "nxcugraph_large_matrix" / "nxcugraph_large_matrix.csv"
    )
    nx_ok = nx_scale[nx_scale["status"].eq("ok")]
    add_check(
        checks,
        "nxcugraph_scale_qualification",
        len(nx_scale) == len(SCALE_ANCHORS) * FUNCTION_COUNT
        and not nx_scale["status"].isin(["", "missing_experiment", "unknown"]).any()
        and len(nx_ok) > 0
        and nx_ok["timing_samples"].eq(5).all()
        and nx_ok["validation"].eq("pass").all(),
        f"records={len(nx_scale)}, completed={len(nx_ok)}, statuses={nx_scale['status'].value_counts().to_dict()}",
    )

    gunrock = pd.read_csv(core / "gunrock_large_matrix" / "gunrock_large_matrix.csv")
    add_check(
        checks,
        "gunrock_scale_anchor_classification",
        len(gunrock) == len(SCALE_ANCHORS) * FUNCTION_COUNT
        and not gunrock["status"].isin(["", "missing_experiment", "unknown"]).any(),
        f"records={len(gunrock)}, statuses={gunrock['status'].value_counts().to_dict()}",
    )

    cpu = pd.read_csv(core / "cpu_large_matrix" / "cpu_large_matrix.csv")
    add_check(
        checks,
        "cpu_scale_anchor_classification",
        len(cpu) == len(SCALE_ANCHORS) * 4 * FUNCTION_COUNT
        and not cpu["status"].isin(["", "missing_experiment", "unknown"]).any(),
        f"records={len(cpu)}/{len(SCALE_ANCHORS) * 4 * FUNCTION_COUNT}, statuses={cpu['status'].value_counts().to_dict()}",
    )

    workflow = pd.read_csv(followup / "cumulative_workflow" / "cumulative_workflow_summary.csv")
    workflow_failure_path = (
        followup / "cumulative_workflow" / "cumulative_workflow_failures.csv"
    )
    workflow_failures = (
        pd.read_csv(workflow_failure_path)
        if workflow_failure_path.stat().st_size > 2
        else pd.DataFrame()
    )
    add_check(
        checks,
        "cumulative_workflow_samples",
        workflow["sample_count"].eq(5).all()
        and workflow.groupby(["dataset", "baseline"])["call_position"].nunique().eq(3).all()
        and workflow_failures.empty,
        f"rows={len(workflow)}, datasets={workflow['dataset'].nunique()}, baselines={workflow['baseline'].nunique()}, failures={len(workflow_failures)}",
    )

    first_use = pd.read_csv(assets / "first_use_steady_actual_times.csv")
    add_check(
        checks,
        "first_use_summary",
        len(first_use) == 8
        and first_use["first_use_over_steady"].map(math.isfinite).all()
        and first_use["first_use_over_steady"].gt(0).all(),
        f"rows={len(first_use)}, four families x E2E/kernel",
    )

    ablation = pd.read_csv(assets / "ablation_actual_values.csv")
    add_check(
        checks,
        "ablation_core_modules",
        len(ablation) == 5
        and ablation["ratio"].map(math.isfinite).all()
        and ablation["ratio"].gt(0).all(),
        f"rows={len(ablation)}, modules={ablation['module'].tolist()}",
    )

    scope = load_json(assets / "final_13_numeric_summary.json")
    add_check(
        checks,
        "scope_summary",
        scope.get("attempted_workloads") == len(FINAL_13) * FUNCTION_COUNT
        and scope.get("datasets") == len(FINAL_13)
        and scope.get("functions") == FUNCTION_COUNT,
        f"workloads={scope.get('attempted_workloads')}, datasets={scope.get('datasets')}, functions={scope.get('functions')}",
    )

    required_assets = [
        "paper_table_baseline_versions.tex",
        "paper_table_baseline_versions.csv",
        "paper_baseline_versions.json",
        "paper_table_datasets_13.tex",
        "paper_table_main_compact_pairwise_speedup.tex",
        "paper_table_main_compact_best_competitor.tex",
        "paper_table_build_13_by_dataset.tex",
        "paper_table_category_13_summary.tex",
        "paper_table_e2e_13_full.csv",
        "paper_table_kernel_13_full.csv",
        "final_13_cell_outcome_ledger.csv",
        "all_experiment_non_success_ledger.csv",
        "category_time_by_baseline_3panel.pdf",
        "first_use_vs_steady_actual_time.pdf",
        "cumulative_cold_start_workflow.pdf",
        "ablation_actual_time_memory_composite.pdf",
        "scaling_four_functions.pdf",
        "z_case_study_same_graph_workflow.pdf",
        "EGGPU_FINAL_TABLES_REVIEW_20260717.tex",
        "EGGPU_FINAL_TABLES_REVIEW_20260717.pdf",
        "TARGETED_EGGPU_SELECTION_DECISIONS.csv",
        "CURRENT_CLOSENESS_SELECTION_DECISIONS.csv",
        "CURRENT_CLOSENESS_EXTERNAL_VALIDATION.csv",
        "SYGRAPH_QUALIFICATION.json",
        "SYGRAPH_QUALIFICATION.md",
    ]
    missing_or_empty = [
        name
        for name in required_assets
        if not (assets / name).exists() or (assets / name).stat().st_size < 256
    ]
    add_check(
        checks,
        "paper_assets_present",
        not missing_or_empty,
        f"missing_or_empty={missing_or_empty}",
    )

    failures = [item for item in checks if item["status"] != "pass"]
    report = {
        "status": "pass" if not failures else "fail",
        "checks": checks,
        "failure_count": len(failures),
    }
    (assets / "FINAL_13_ASSET_AUDIT.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Final 13-Dataset Asset Audit",
        "",
        f"Status: **{report['status'].upper()}**",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for item in checks:
        detail = str(item["detail"]).replace("|", "\\|")
        lines.append(f"| {item['check']} | {item['status']} | {detail} |")
    lines.extend(
        [
            "",
            "The audited scope is a 13 x 16 main-workload matrix. Every cell is attempted or explicitly classified, while performance aggregates include only correctness-validated successful cells.",
            "",
        ]
    )
    (assets / "FINAL_13_ASSET_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
