#!/usr/bin/env python3
"""Audit completeness, correctness, statistics, and memory of supplement 2."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


DATASETS = (
    "ca-HepTh",
    "LastFM",
    "p2p-Gnutella04",
    "ca-HepPh",
    "email-Enron",
    "ca-CondMat",
    "soc-Epinions1",
    "com-youtube",
    "ER-100k",
    "soc-Slashdot0811",
)
COLD_FUNCTIONS = ("PageRank", "LCC", "WCC", "BFS", "SSSP", "KCore")
COLD_BASELINES = ("EGGPU", "igraph", "nx-cugraph")
SCALING_FUNCTIONS = ("PageRank", "WCC", "BFS", "KCore")
RMAT_DATASETS = (
    "R-MAT-S20-EF16",
    "R-MAT-S22-EF16",
    "R-MAT-S24-EF16",
    "R-MAT-S26-EF16",
)
RMAT_FUNCTIONS = ("PageRank", "WCC", "BFS")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.result_root.resolve()
    audit_dir = root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    hard = []
    warnings = []
    evidence = {}

    gate_path = root / "bulk_csr_performance_gate.json"
    if not gate_path.exists():
        hard.append("missing bulk_csr_performance_gate.json")
    else:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        evidence["native_csr_gate_status"] = gate.get("status")
        evidence["native_csr_median_geomean_e2e_speedup"] = gate.get(
            "function_median_geomean_e2e_speedup"
        )
        evidence["native_csr_median_geomean_kernel_speedup"] = gate.get(
            "function_median_geomean_kernel_speedup"
        )
        if gate.get("status") != "pass":
            hard.append(f"native CSR performance gate failed: {gate.get('failures')}")
        for function in SCALING_FUNCTIONS:
            record = gate.get("functions", {}).get(function)
            if not record or not record.get("correct"):
                hard.append(f"native CSR gate lacks a correct {function} result")

    completeness_path = root / "paper_artifacts" / "cold_start_completeness.csv"
    if not completeness_path.exists():
        hard.append("missing cold_start_completeness.csv")
    else:
        completeness = read_csv(completeness_path)
        incomplete = [row for row in completeness if row.get("status") != "complete"]
        evidence["cold_timing_groups"] = len(completeness)
        evidence["cold_timing_incomplete_groups"] = len(incomplete)
        expected_groups = len(DATASETS) * len(COLD_FUNCTIONS) * len(COLD_BASELINES)
        if len(completeness) != expected_groups:
            hard.append(
                f"cold-start group count is {len(completeness)}, expected {expected_groups}"
            )
        if incomplete:
            examples = [
                f"{row['dataset']}/{row['function']}/{row['baseline']}="
                f"{row['observed_paired_samples']}/{row['expected_samples']}"
                for row in incomplete[:10]
            ]
            hard.append("incomplete cold-start timing groups: " + "; ".join(examples))

    samples_path = root / "cold_start" / "results_samples.csv"
    if not samples_path.exists():
        hard.append("missing cold-start results_samples.csv")
    else:
        samples = read_csv(samples_path)
        timing_samples = set()
        for row in samples:
            if row.get("measurement_phase") != "timing" or row.get("status") != "ok":
                continue
            if row.get("dataset") not in DATASETS:
                continue
            if row.get("function") not in COLD_FUNCTIONS:
                continue
            if row.get("baseline") not in COLD_BASELINES:
                continue
            if row.get("metric") not in {"build", "e2e", "kernel"}:
                continue
            timing_samples.add(
                (
                    row.get("dataset"),
                    row.get("function"),
                    row.get("baseline"),
                    row.get("metric"),
                    row.get("sample_index"),
                )
            )
        expected_timing = len(DATASETS) * len(COLD_FUNCTIONS) * len(COLD_BASELINES) * 3 * 5
        evidence["cold_timing_unique_metric_samples"] = len(timing_samples)
        if len(timing_samples) != expected_timing:
            hard.append(
                f"cold-start timing has {len(timing_samples)} unique metric samples, "
                f"expected {expected_timing}"
            )
        unexpected_memory = [
            row for row in samples if row.get("measurement_phase") == "memory"
        ]
        if unexpected_memory:
            hard.append(
                "cold-start assembly unexpectedly contains memory rows; the protocol "
                "uses existing timing samples and leaves memory to isolated main/scaling passes"
            )

    reuse_metadata_path = root / "cold_start" / "cold_start_reuse_metadata.json"
    if not reuse_metadata_path.exists():
        hard.append("missing cold-start reuse metadata")
    else:
        reuse_metadata = json.loads(reuse_metadata_path.read_text(encoding="utf-8"))
        evidence["cold_start_protocol"] = reuse_metadata.get("protocol")
        evidence["cold_start_memory_policy"] = reuse_metadata.get("memory_policy")
        if reuse_metadata.get("protocol") != "audited_existing_fresh_process_cold_start_v1":
            hard.append("cold-start rows were not assembled from the audited fresh-process runs")

    correctness_path = root / "cold_start" / "correctness_validation.csv"
    if not correctness_path.exists():
        hard.append("missing cold-start correctness_validation.csv")
    else:
        correctness = read_csv(correctness_path)
        scoped = [
            row
            for row in correctness
            if row.get("dataset") in DATASETS
            and row.get("function") in COLD_FUNCTIONS
            and row.get("baseline") in COLD_BASELINES
        ]
        invalid = [
            row
            for row in scoped
            if row.get("validation_status") not in {"pass", "reference"}
        ]
        evidence["cold_correctness_rows"] = len(scoped)
        evidence["cold_correctness_invalid"] = len(invalid)
        expected_correctness = len(DATASETS) * len(COLD_FUNCTIONS) * len(COLD_BASELINES)
        if len(scoped) != expected_correctness:
            hard.append(
                f"cold-start correctness has {len(scoped)} rows, "
                f"expected {expected_correctness}"
            )
        if invalid:
            hard.append(
                "cold-start correctness failures: "
                + "; ".join(
                    f"{row['dataset']}/{row['function']}/{row['baseline']}="
                    f"{row['validation_status']}"
                    for row in invalid[:10]
                )
            )

    scaling_path = root / "scaling" / "scaling_all.json"
    if not scaling_path.exists():
        hard.append("missing scaling_all.json")
    else:
        scaling = json.loads(scaling_path.read_text(encoding="utf-8"))
        counts = Counter(
            f"{row.get('measurement')}:{row.get('status')}" for row in scaling
        )
        evidence["scaling_status_counts"] = dict(counts)
        expected = {
            "timing:ok": 11,
            "timing:skipped": 1,
            "memory:ok": 33,
            "memory:skipped": 1,
        }
        for key, count in expected.items():
            if counts.get(key, 0) != count:
                hard.append(
                    f"scaling status {key} has {counts.get(key, 0)}, expected {count}"
                )
        unexpected = {
            key: count for key, count in counts.items() if key.endswith(":failed")
        }
        if unexpected:
            hard.append(f"scaling contains failures: {unexpected}")

        timing = [
            row
            for row in scaling
            if row.get("measurement") == "timing" and row.get("status") == "ok"
        ]
        for row in timing:
            if int(row.get("timing_process_samples", 0)) != 5:
                hard.append(
                    f"{row.get('dataset')}/{row.get('function')} has "
                    f"{row.get('timing_process_samples')} timing processes"
                )
            if row.get("result_validation", {}).get("status") != "pass":
                hard.append(
                    f"{row.get('dataset')}/{row.get('function')} failed result validation"
                )
            cv = row.get("steady_e2e", {}).get("cv")
            if cv is not None and float(cv) > 0.20:
                warnings.append(
                    f"high steady E2E CV: {row.get('dataset')}/{row.get('function')}="
                    f"{float(cv) * 100:.1f}%"
                )

        memory = [
            row
            for row in scaling
            if row.get("measurement") == "memory" and row.get("status") == "ok"
        ]
        for row in memory:
            if row.get("result_validation", {}).get("status") != "pass":
                hard.append(
                    f"memory result validation failed: {row.get('dataset')}/"
                    f"{row.get('function')}"
                )
            monitor = row.get("memory", {})
            if int(monitor.get("monitor_rss_samples", 0)) <= 0:
                hard.append(
                    f"RSS monitor captured no samples: {row.get('dataset')}/"
                    f"{row.get('function')}"
                )
            if int(monitor.get("monitor_gpu_proc_samples", 0)) <= 0:
                hard.append(
                    f"GPU process monitor captured no samples: {row.get('dataset')}/"
                    f"{row.get('function')}"
                )

    rmat_reference_path = root / "rmat_reference_gate.json"
    if not rmat_reference_path.exists():
        hard.append("missing rmat_reference_gate.json")
    else:
        reference = json.loads(rmat_reference_path.read_text(encoding="utf-8"))
        evidence["rmat_reference_gate_status"] = reference.get("status")
        evidence["rmat_reference_checks"] = reference.get("checks")
        if reference.get("status") != "pass":
            hard.append(f"controlled R-MAT CPU reference gate failed: {reference}")

    rmat_path = root / "controlled_rmat" / "scaling_all.json"
    if not rmat_path.exists():
        hard.append("missing controlled R-MAT scaling_all.json")
    else:
        rmat_records = json.loads(rmat_path.read_text(encoding="utf-8"))
        counts = Counter(
            f"{row.get('measurement')}:{row.get('status')}" for row in rmat_records
        )
        evidence["controlled_rmat_status_counts"] = dict(counts)
        if counts.get("timing:ok", 0) != 12:
            hard.append(
                f"controlled R-MAT timing:ok has {counts.get('timing:ok', 0)}, expected 12"
            )
        if counts.get("memory:ok", 0) != 36:
            hard.append(
                f"controlled R-MAT memory:ok has {counts.get('memory:ok', 0)}, expected 36"
            )
        unexpected = {
            key: value
            for key, value in counts.items()
            if key.endswith(":failed") or key.endswith(":skipped")
        }
        if unexpected:
            hard.append(f"controlled R-MAT contains non-pass rows: {unexpected}")
        identities = {
            (row.get("dataset"), row.get("function"))
            for row in rmat_records
            if row.get("measurement") == "timing" and row.get("status") == "ok"
        }
        expected_identities = {
            (dataset, function)
            for dataset in RMAT_DATASETS
            for function in RMAT_FUNCTIONS
        }
        if identities != expected_identities:
            hard.append(
                "controlled R-MAT timing identities differ from protocol: "
                f"missing={sorted(expected_identities - identities)}, "
                f"extra={sorted(identities - expected_identities)}"
            )
        expected_raw = {20: 1 << 24, 22: 1 << 26, 24: 1 << 28, 26: 1 << 30}
        for row in rmat_records:
            if row.get("status") != "ok":
                continue
            if row.get("input_family") != "controlled_rmat":
                hard.append(
                    f"R-MAT provenance missing: {row.get('dataset')}/{row.get('function')}"
                )
            scale = int(row.get("rmat_scale") or 0)
            if int(row.get("raw_edge_records") or 0) != expected_raw.get(scale):
                hard.append(
                    f"R-MAT raw edge count mismatch: {row.get('dataset')}/"
                    f"{row.get('function')}"
                )
            if int(row.get("num_entries") or 0) >= int(row.get("raw_edge_records") or 0):
                hard.append(
                    f"R-MAT normalization was not recorded: {row.get('dataset')}/"
                    f"{row.get('function')}"
                )
            if row.get("result_validation", {}).get("status") != "pass":
                hard.append(
                    f"R-MAT result validation failed: {row.get('dataset')}/"
                    f"{row.get('function')}/{row.get('measurement')}"
                )
            if row.get("measurement") == "timing" and int(
                row.get("timing_process_samples", 0)
            ) != 5:
                hard.append(
                    f"R-MAT timing process count differs from five: "
                    f"{row.get('dataset')}/{row.get('function')}"
                )
            if row.get("measurement") == "memory":
                monitor = row.get("memory", {})
                if int(monitor.get("monitor_rss_samples", 0)) <= 0:
                    hard.append(
                        f"R-MAT RSS monitor captured no samples: "
                        f"{row.get('dataset')}/{row.get('function')}"
                    )
                if int(monitor.get("monitor_gpu_proc_samples", 0)) <= 0:
                    hard.append(
                        f"R-MAT GPU monitor captured no samples: "
                        f"{row.get('dataset')}/{row.get('function')}"
                    )

    summary_path = root / "paper_artifacts" / "second_supplement_summary.json"
    if not summary_path.exists():
        hard.append("missing second_supplement_summary.json")
    else:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        evidence["paper_summary"] = summary
        dataset_table = summary.get("main_dataset_table", {})
        if dataset_table.get("status") != "ok" or int(
            dataset_table.get("rows", 0)
        ) != 12:
            hard.append(f"final 12-dataset table is incomplete: {dataset_table}")
        controlled_table = root / "paper_artifacts" / "paper_table_rmat_scaling_datasets.csv"
        if not controlled_table.exists():
            hard.append("missing controlled R-MAT dataset table")
        else:
            controlled_rows = read_csv(controlled_table)
            if len(controlled_rows) != 4:
                hard.append(
                    f"controlled R-MAT dataset table has {len(controlled_rows)} rows, expected 4"
                )

    main_manifest_path = root / "main_10_paper_artifacts" / "artifact_manifest.json"
    if not main_manifest_path.exists():
        hard.append("missing selected ten-graph main artifact manifest")
    else:
        main_manifest = json.loads(main_manifest_path.read_text(encoding="utf-8"))
        evidence["main_10_artifact_manifest"] = {
            key: main_manifest.get(key)
            for key in (
                "dataset_filter",
                "dataset_filter_is_explicit",
                "eggpu_primary_estimator",
                "competitor_primary_estimator",
                "timing_repeat",
            )
        }
        if main_manifest.get("dataset_filter") != list(DATASETS):
            hard.append(
                "main artifact dataset filter differs from the declared balanced ten: "
                f"{main_manifest.get('dataset_filter')}"
            )
        if not main_manifest.get("dataset_filter_is_explicit"):
            hard.append("main artifact bundle did not record an explicit dataset filter")
        if main_manifest.get("eggpu_primary_estimator") != "best_observed_of_five":
            hard.append("main artifact EGGPU estimator is not best-observed-of-five")
        if main_manifest.get("competitor_primary_estimator") != "arithmetic_mean":
            hard.append("main artifact competitor estimator is not arithmetic mean")

    status = "pass" if not hard else "fail"
    report = {
        "status": status,
        "result_root": str(root),
        "hard_issues": hard,
        "warnings": warnings,
        "evidence": evidence,
    }
    (audit_dir / "SECOND_SUPPLEMENT_AUDIT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Second Supplement Audit",
        "",
        f"Status: **{status.upper()}**",
        "",
        "## Hard issues",
        "",
    ]
    lines.extend(f"- {item}" for item in hard or ["None."])
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {item}" for item in warnings or ["None."])
    lines.extend(["", "## Evidence", "", "```json"])
    lines.append(json.dumps(evidence, indent=2, sort_keys=True))
    lines.extend(["```", ""])
    (audit_dir / "SECOND_SUPPLEMENT_AUDIT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if hard:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
