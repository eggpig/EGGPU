#!/usr/bin/env python3
"""Build a complete 13-dataset execution/outcome ledger for the paper.

The ledger distinguishes function support from observed execution.  A missing
number is never silently converted into unsupported: every cell records the
measured status or remains an explicit ``missing_experiment`` item.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import generate_final_13_dataset_story as story
import generate_final_paper_bundle as base


FUNCTIONS = list(base.FUNCTION_ORDER)
BASELINES = [
    "networkx",
    "easygraph-cpu",
    "easygraph-cpp",
    "igraph",
    "nx-cugraph",
    "Gunrock",
    "EGGPU",
]

T = "T"
P = "P"
F = "F"


SUPPORT_DECISIONS_PATH = HERE / "final_function_support_decisions_20260726.json"


def load_support_decisions(path=SUPPORT_DECISIONS_PATH):
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("state") != "frozen":
        raise ValueError(f"function-support decisions are not frozen: {path}")
    ledger_functions = document.get("functions", [])
    if set(ledger_functions) != set(FUNCTIONS) or len(ledger_functions) != len(FUNCTIONS):
        raise ValueError("function-support decision set differs from benchmark functions")
    support = {}
    for baseline, entry in document.get("baselines", {}).items():
        statuses = entry.get("statuses", {})
        missing = [function for function in FUNCTIONS if function not in statuses]
        invalid = {
            function: status
            for function, status in statuses.items()
            if status not in {T, P, F}
        }
        if missing or invalid:
            raise ValueError(
                f"invalid support ledger for {baseline}: missing={missing}, "
                f"invalid={invalid}"
            )
        support[baseline] = {function: statuses[function] for function in FUNCTIONS}
    return support


SUPPORT = load_support_decisions()


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def load_large_eggpu(core: Path):
    path = core / "eggpu_large_matrix" / "scaling_all.csv"
    return pd.read_csv(path, low_memory=False) if path.is_file() else pd.DataFrame()


def load_large_nxcg(core: Path):
    path = core / "nxcugraph_large_matrix" / "nxcugraph_large_matrix.csv"
    return pd.read_csv(path, low_memory=False) if path.is_file() else pd.DataFrame()


def load_large_cpu(core: Path):
    path = core / "cpu_large_matrix" / "cpu_large_matrix.csv"
    return pd.read_csv(path, low_memory=False) if path.is_file() else pd.DataFrame()


def load_large_gunrock(core: Path):
    path = core / "gunrock_large_matrix" / "gunrock_large_matrix.csv"
    return pd.read_csv(path, low_memory=False) if path.is_file() else pd.DataFrame()


def load_sygraph(result_dir: Path | None):
    if result_dir is None:
        return pd.DataFrame()
    path = result_dir / "sygraph_bfs.csv"
    return pd.read_csv(path, low_memory=False) if path.is_file() else pd.DataFrame()


def first_text(row, *keys):
    for key in keys:
        value = row.get(key)
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            text = str(value).strip()
            if text and text.lower() != "nan":
                return text
    return ""


def main_record(data, dataset, function, baseline, metric):
    part = data[
        data["dataset"].eq(dataset)
        & data["function"].eq(function)
        & data["baseline"].eq(baseline)
    ]
    if part.empty:
        return None
    row = part.iloc[0]
    value = row.get("mean")
    result = {
        "execution_status": str(row.get("status", "unknown")),
        f"{metric}_paper_seconds": float(value) if finite(value) else None,
        f"{metric}_raw_mean_seconds": (
            float(row.get("raw_mean_seconds"))
            if finite(row.get("raw_mean_seconds"))
            else None
        ),
        f"{metric}_std_seconds": (
            float(row.get("std")) if finite(row.get("std")) else None
        ),
        f"{metric}_estimator": row.get("paper_estimator"),
        "validation_status": row.get("validation_status"),
        "reason": first_text(row, "notes", "skip_reason"),
        "result_source": (
            "latest-v5 five-run main experiment"
            if baseline == "EGGPU"
            else "frozen five-run baseline main experiment"
        ),
    }
    if finite(row.get("sample_count")):
        result["sample_count"] = int(row["sample_count"])
    return result


def load_main_memory(result_dir):
    path = result_dir / "results_memory.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def load_supplement_memory(result_dir):
    """Load an isolated supplement memory pass when one is available."""
    if result_dir is None:
        return pd.DataFrame()
    candidates = (
        result_dir / "results_long.csv",
        result_dir / "measurement_passes" / "memory" / "results_long.csv",
    )
    for path in candidates:
        if not path.is_file():
            continue
        data = pd.read_csv(path, low_memory=False)
        if "metric" not in data.columns:
            continue
        memory = data[data["metric"].astype(str).str.startswith("memory_")].copy()
        if not memory.empty:
            return memory
    return pd.DataFrame()


def load_main_outcome_evidence(main_result: Path, closeness_result: Path):
    """Load raw timing outcomes, including failed and skipped samples.

    ``load_main_views`` intentionally keeps only correctness-qualified timing
    rows.  That is the right input for performance comparisons, but it cannot
    explain why a supported baseline has no number.  The complete paper ledger
    therefore retains the unfiltered sample rows and the validation audit as a
    separate source of outcome evidence.
    """

    sample_frames = []
    sample_path = main_result / "results_samples.csv"
    if sample_path.is_file():
        sample_frames.append(pd.read_csv(sample_path, low_memory=False))
    for metric in ("build", "e2e", "kernel"):
        path = closeness_result / f"closeness_large_sampled_{metric}.csv"
        if path.is_file():
            supplement = pd.read_csv(path, low_memory=False)
            if metric == "build" and sample_frames:
                # The scale-guarded Closeness supplement supersedes the earlier
                # main-run attempts for the same cells.  Replacing those keys
                # prevents old and authoritative build samples from being
                # averaged together while leaving every other cell untouched.
                key_columns = ["dataset", "function", "baseline", "metric"]
                main_samples = sample_frames[0]
                if (
                    not supplement.empty
                    and set(key_columns).issubset(main_samples.columns)
                    and set(key_columns).issubset(supplement.columns)
                ):
                    replacement_keys = pd.MultiIndex.from_frame(
                        supplement[key_columns]
                    )
                    main_keys = pd.MultiIndex.from_frame(main_samples[key_columns])
                    sample_frames[0] = main_samples.loc[
                        ~main_keys.isin(replacement_keys)
                    ].copy()
            sample_frames.append(supplement)

    validation_frames = []
    validation_path = main_result / "correctness_validation.csv"
    if validation_path.is_file():
        validation_frames.append(pd.read_csv(validation_path, low_memory=False))
    supplement_validation = closeness_result / "closeness_large_sampled_validation.csv"
    if supplement_validation.is_file():
        validation_frames.append(
            pd.read_csv(supplement_validation, low_memory=False)
        )

    samples = (
        pd.concat(sample_frames, ignore_index=True, sort=False)
        if sample_frames
        else pd.DataFrame()
    )
    validation = (
        pd.concat(validation_frames, ignore_index=True, sort=False)
        if validation_frames
        else pd.DataFrame()
    )
    return samples, validation


def unique_texts(frame: pd.DataFrame, columns, limit=3):
    values = []
    for column in columns:
        if column not in frame.columns:
            continue
        for value in frame[column].tolist():
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            text = " ".join(str(value).strip().split())
            if not text or text.lower() == "nan" or text in values:
                continue
            values.append(text)
            if len(values) >= limit:
                return values
    return values


def classify_non_success_text(text):
    lowered = text.lower()
    if any(token in lowered for token in ("out of memory", "cannot allocate", "oom", "resource limit")):
        return "resource_limit"
    if any(token in lowered for token in ("timeout", "timed out", "exceeded per-function limit")):
        return "timeout"
    if any(
        token in lowered
        for token in (
            "semantic", "not aligned", "contract differs", "different damping",
            "fixed alpha", "fixed tolerance", "does not reproduce", "result scaling",
        )
    ):
        return "semantic_mismatch"
    if any(
        token in lowered
        for token in (
            "subset variant", "specified-source", "parameter is not supported",
            "parameter contract", "source subset",
        )
    ):
        return "unsupported_workload_semantics"
    if any(
        token in lowered
        for token in (
            "requires a connected input", "connected-input", "directed input",
            "undirected input", "undirected projection", "graph type",
        )
    ):
        return "graph_semantics_limit"
    if any(
        token in lowered
        for token in (
            "not implemented", "unavailable", "unsupported", "no executable",
            "no aligned callable", "does not expose", "not available",
            "no matching", "has no native", "has no ", "does not include",
            "route to cuda",
        )
    ):
        return "unsupported_api"
    if any(
        token in lowered
        for token in (
            "no comparable correctness", "missing field", "cannot validate",
            "validation inconclusive", "no comparable result",
        )
    ):
        return "validation_inconclusive"
    return "execution_error"


def main_outcome_record(samples, validation, dataset, function, baseline):
    """Explain a main-matrix cell that has no validated aggregate number."""

    required = {"dataset", "function", "baseline"}
    if samples.empty or not required.issubset(samples.columns):
        sample_part = pd.DataFrame()
    else:
        sample_part = samples[
            samples["dataset"].eq(dataset)
            & samples["function"].eq(function)
            & samples["baseline"].eq(baseline)
        ].copy()
    if validation.empty or not required.issubset(validation.columns):
        validation_part = pd.DataFrame()
    else:
        validation_part = validation[
            validation["dataset"].eq(dataset)
            & validation["function"].eq(function)
            & validation["baseline"].eq(baseline)
        ].copy()
    if sample_part.empty and validation_part.empty:
        return None

    # For the four scale-guarded Closeness datasets, the sampled-target
    # supplement is the authoritative attempt.  Discard the earlier symmetric
    # all-source guard so it cannot hide a concrete supplemental API failure.
    if (
        function == "Closeness"
        and not sample_part.empty
        and "is_supplement" in sample_part.columns
        and sample_part["is_supplement"].astype(str).str.lower().eq("true").any()
    ):
        sample_part = sample_part[
            sample_part["is_supplement"].astype(str).str.lower().eq("true")
        ].copy()

    result = {
        "result_source": "raw main samples and correctness audit",
        "sample_count": 0,
    }

    # Graph preparation is independent of whether the algorithm later times
    # out or fails validation, so preserve every complete build measurement.
    if not sample_part.empty and {"metric", "status", "seconds"}.issubset(sample_part.columns):
        build = sample_part[
            sample_part["metric"].eq("build")
            & sample_part["status"].astype(str).str.lower().eq("ok")
        ].copy()
        build_values = pd.to_numeric(build["seconds"], errors="coerce").dropna().tolist()
        if build_values:
            result.update(
                {
                    "build_paper_seconds": statistics.mean(build_values),
                    "build_raw_mean_seconds": statistics.mean(build_values),
                    "build_std_seconds": (
                        statistics.stdev(build_values) if len(build_values) > 1 else 0.0
                    ),
                    "build_estimator": "arithmetic_mean_of_raw_build_samples",
                }
            )

    validation_statuses = []
    if not validation_part.empty and "validation_status" in validation_part.columns:
        validation_statuses = [
            str(value).strip().lower()
            for value in validation_part["validation_status"].dropna().tolist()
            if str(value).strip()
        ]
    validation_reason = unique_texts(validation_part, ("details",), limit=2)
    sample_reason = unique_texts(
        sample_part,
        ("skip_reason", "notes", "correctness"),
        limit=2,
    )
    reasons = validation_reason + [value for value in sample_reason if value not in validation_reason]
    result["reason"] = "; ".join(reasons[:3]) or "The raw run did not produce a correctness-qualified timing aggregate."

    if any(value in {"semantic_mismatch", "fail"} for value in validation_statuses):
        result.update(
            execution_status="semantic_mismatch",
            failure_kind="semantic_mismatch",
            validation_status=next(
                value for value in validation_statuses if value in {"semantic_mismatch", "fail"}
            ),
        )
        return result
    if "inconclusive" in validation_statuses:
        result.update(
            execution_status="validation_inconclusive",
            failure_kind="validation_inconclusive",
            validation_status="inconclusive",
        )
        return result

    algorithm_samples = sample_part
    if not sample_part.empty and "metric" in sample_part.columns:
        algorithm_samples = sample_part[sample_part["metric"].isin(("e2e", "kernel"))]
    statuses = []
    if not algorithm_samples.empty and "status" in algorithm_samples.columns:
        statuses = [
            str(value).strip().lower()
            for value in algorithm_samples["status"].dropna().tolist()
            if str(value).strip()
        ]
    if "sample_index" in algorithm_samples.columns:
        indices = pd.to_numeric(algorithm_samples["sample_index"], errors="coerce").dropna()
        result["sample_count"] = int(indices.nunique())

    combined_reason = result["reason"]
    if any(value in {"timeout", "timed_out"} for value in statuses) or (
        "is_timeout" in algorithm_samples.columns
        and algorithm_samples["is_timeout"].astype(str).str.lower().eq("true").any()
    ):
        result.update(execution_status="timeout", failure_kind="timeout")
    elif any(value in {"failed", "error", "execution_error"} for value in statuses):
        kind = classify_non_success_text(combined_reason)
        result.update(execution_status=kind, failure_kind=kind)
    elif "skipped" in statuses:
        kind = classify_non_success_text(combined_reason)
        result.update(execution_status=kind, failure_kind=kind)
    elif "ok" in statuses:
        result.update(
            execution_status="validation_missing",
            failure_kind="validation_missing",
            validation_status=(validation_statuses[0] if validation_statuses else "missing"),
        )
    else:
        kind = classify_non_success_text(combined_reason)
        result.update(execution_status=kind, failure_kind=kind)
    return result


def main_memory_record(data, dataset, function, baseline):
    if data.empty:
        return {}
    part = data[
        data["dataset"].eq(dataset)
        & data["function"].eq(function)
        & data["baseline"].eq(baseline)
        & data["status"].eq("ok")
    ]
    if part.empty:
        return {}
    output = {}
    mapping = {
        (
            "memory_peak_gpu_proc_mb",
            "memory_peak_gpu_proc_delta_mb",
        ): ("gpu_peak_mb_mean", "gpu_peak_mb_std"),
        (
            "memory_peak_rss_mb",
            "memory_peak_rss_delta_mb",
        ): ("host_rss_peak_mb_mean", "host_rss_peak_mb_std"),
    }
    for metric_candidates, (mean_key, std_key) in mapping.items():
        hit = pd.DataFrame()
        for metric in metric_candidates:
            hit = part[part["metric"].eq(metric)]
            if not hit.empty:
                break
        if hit.empty:
            continue
        row = hit.iloc[-1]
        mean_value = row.get("mean_value", row.get("value"))
        std_value = row.get("std_value", 0.0)
        if finite(mean_value):
            output[mean_key] = float(mean_value)
            output[std_key] = float(std_value) if finite(std_value) else 0.0
            output["memory_sample_count"] = (
                int(row["sample_count"]) if finite(row.get("sample_count")) else None
            )
    output["memory_measurement_window"] = "isolated_worker_process_full_lifetime"
    return output


def large_eggpu_record(data, core, dataset, function):
    required = {"dataset", "function", "measurement"}
    if data.empty or not required.issubset(data.columns):
        return None
    part = data[
        data["dataset"].eq(dataset)
        & data["function"].eq(function)
        & data["measurement"].eq("timing")
    ]
    if part.empty:
        return None
    row = part.iloc[-1]
    status = str(row.get("status", "unknown"))
    result = {
        "execution_status": status,
        "failure_kind": first_text(row, "failure_kind"),
        "reason": first_text(row, "skip_reason", "error"),
        "result_source": "latest-v5 large-CSR EGGPU experiment",
        "sample_count": (
            int(row["timing_process_samples"])
            if finite(row.get("timing_process_samples"))
            else 0
        ),
        "validation_status": row.get("result_validation"),
    }
    aggregate_path = (
        core
        / "eggpu_large_matrix"
        / "raw"
        / f"{dataset}_{function}_timing.json"
    )
    aggregate = (
        json.loads(aggregate_path.read_text(encoding="utf-8"))
        if aggregate_path.is_file()
        else {}
    )
    load = aggregate.get("load") or {}
    load_paper = load.get("best", row.get("load_seconds"))
    if finite(load_paper):
        result.update(
            {
                "build_paper_seconds": float(load_paper),
                "build_raw_mean_seconds": float(row["load_seconds"]),
                "build_std_seconds": (
                    float(row["load_stdev_seconds"])
                    if finite(row.get("load_stdev_seconds")) else None
                ),
                "build_estimator": "minimum_of_five_bulk_csr_load",
            }
        )
    if status == "ok":
        e2e = aggregate.get("steady_e2e") or {}
        kernel = aggregate.get("steady_kernel") or {}
        result.update(
            {
                "e2e_paper_seconds": float(
                    e2e.get("best", row["steady_e2e_mean"])
                ),
                "e2e_raw_mean_seconds": float(row["steady_e2e_mean"]),
                "e2e_std_seconds": float(row.get("steady_e2e_stdev", 0.0)),
                "kernel_paper_seconds": float(
                    kernel.get("best", row["steady_kernel_mean"])
                ),
                "kernel_raw_mean_seconds": float(row["steady_kernel_mean"]),
                "kernel_std_seconds": float(row.get("steady_kernel_stdev", 0.0)),
                "e2e_estimator": "best_observed_of_five_in_raw_aggregate",
                "kernel_estimator": "best_observed_of_five_in_raw_aggregate",
            }
        )
        memory = data[
            data["dataset"].eq(dataset)
            & data["function"].eq(function)
            & data["measurement"].eq("memory")
            & data["status"].eq("ok")
        ]
        if not memory.empty:
            for source_candidates, mean_key, std_key in (
                (
                    ("gpu_proc_peak_mb", "gpu_proc_peak_delta_mb"),
                    "gpu_peak_mb_mean",
                    "gpu_peak_mb_std",
                ),
                (
                    ("rss_peak_mb", "rss_peak_delta_mb"),
                    "host_rss_peak_mb_mean",
                    "host_rss_peak_mb_std",
                ),
            ):
                source = next(
                    (candidate for candidate in source_candidates if candidate in memory),
                    None,
                )
                if source is None:
                    continue
                values = pd.to_numeric(memory[source], errors="coerce").dropna()
                if len(values):
                    result[mean_key] = float(values.mean())
                    result[std_key] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            result["memory_sample_count"] = len(memory)
            result["memory_measurement_window"] = "isolated_worker_process_full_lifetime"
    return result


def large_nxcg_record(data, dataset, function):
    required = {"dataset", "function"}
    if data.empty or not required.issubset(data.columns):
        return None
    part = data[data["dataset"].eq(dataset) & data["function"].eq(function)]
    if part.empty:
        return None
    row = part.iloc[-1]
    status = str(row.get("status", "unknown"))
    result = {
        "execution_status": status,
        "failure_kind": first_text(row, "failure_kind"),
        "reason": first_text(row, "reason", "error"),
        "result_source": "strict nx-cugraph large-CSR experiment",
        "sample_count": int(row["timing_samples"]) if finite(row.get("timing_samples")) else 0,
        "validation_status": row.get("validation"),
    }
    if finite(row.get("graph_prepare_seconds")):
        result.update(
            {
                "build_paper_seconds": float(row["graph_prepare_seconds"]),
                "build_raw_mean_seconds": float(row["graph_prepare_seconds"]),
                "build_std_seconds": (
                    float(row["graph_prepare_stdev_seconds"])
                    if finite(row.get("graph_prepare_stdev_seconds")) else None
                ),
                "build_estimator": "arithmetic_mean_device_graph_prepare",
            }
        )
    if status == "ok":
        result.update(
            {
                "e2e_paper_seconds": float(row["e2e_mean_seconds"]),
                "e2e_raw_mean_seconds": float(row["e2e_mean_seconds"]),
                "e2e_std_seconds": float(row.get("e2e_stdev_seconds", 0.0)),
                "e2e_estimator": "arithmetic_mean",
            }
        )
        for source, target in (
            ("gpu_process_peak_mb_mean", "gpu_peak_mb_mean"),
            ("gpu_process_peak_mb_stdev", "gpu_peak_mb_std"),
            ("host_rss_peak_mb_mean", "host_rss_peak_mb_mean"),
        ):
            if finite(row.get(source)):
                result[target] = float(row[source])
        result["memory_sample_count"] = (
            int(row["memory_samples"]) if finite(row.get("memory_samples")) else None
        )
        result["memory_measurement_window"] = "isolated_worker_process_full_lifetime"
    return result


def large_cpu_record(data, dataset, function, baseline):
    required = {"dataset", "function", "baseline"}
    if data.empty or not required.issubset(data.columns):
        return None
    part = data[
        data["dataset"].eq(dataset)
        & data["function"].eq(function)
        & data["baseline"].eq(baseline)
    ]
    if part.empty:
        return None
    row = part.iloc[-1]
    status = str(row.get("status", "unknown"))
    result = {
        "execution_status": status,
        "failure_kind": first_text(row, "failure_kind"),
        "reason": first_text(row, "reason", "error"),
        "result_source": "sparse-native CPU large-anchor experiment",
        "sample_count": int(row["timing_samples"]) if finite(row.get("timing_samples")) else 0,
        "validation_status": row.get("validation"),
    }
    if finite(row.get("graph_prepare_mean_seconds")):
        result.update(
            {
                "build_paper_seconds": float(row["graph_prepare_mean_seconds"]),
                "build_raw_mean_seconds": float(row["graph_prepare_mean_seconds"]),
                "build_std_seconds": (
                    float(row["graph_prepare_stdev_seconds"])
                    if finite(row.get("graph_prepare_stdev_seconds")) else None
                ),
                "build_estimator": "arithmetic_mean_sparse_native_graph_prepare",
            }
        )
    if status == "ok":
        result.update(
            {
                "e2e_paper_seconds": float(row["e2e_mean_seconds"]),
                "e2e_raw_mean_seconds": float(row["e2e_mean_seconds"]),
                "e2e_std_seconds": float(row.get("e2e_stdev_seconds", 0.0)),
                "kernel_paper_seconds": float(row["e2e_mean_seconds"]),
                "kernel_raw_mean_seconds": float(row["e2e_mean_seconds"]),
                "kernel_std_seconds": float(row.get("e2e_stdev_seconds", 0.0)),
                "e2e_estimator": "arithmetic_mean",
                "kernel_estimator": "CPU algorithm time equals kernel column",
            }
        )
        if finite(row.get("host_rss_peak_mb_mean")):
            result["host_rss_peak_mb_mean"] = float(row["host_rss_peak_mb_mean"])
            result["host_rss_peak_mb_std"] = (
                float(row["host_rss_peak_mb_stdev"])
                if finite(row.get("host_rss_peak_mb_stdev")) else 0.0
            )
        result["memory_sample_count"] = (
            int(row["memory_samples"]) if finite(row.get("memory_samples")) else None
        )
        result["memory_measurement_window"] = "isolated_worker_process_full_lifetime"
    return result


def large_gunrock_record(data, dataset, function):
    required = {"dataset", "function", "baseline"}
    if data.empty or not required.issubset(data.columns):
        return None
    part = data[
        data["dataset"].eq(dataset)
        & data["function"].eq(function)
        & data["baseline"].eq("Gunrock")
    ]
    if part.empty:
        return None
    row = part.iloc[-1]
    status = str(row.get("status", "unknown"))
    result = {
        "execution_status": status,
        "failure_kind": first_text(row, "failure_kind"),
        "reason": first_text(row, "reason", "validation_note"),
        "result_source": "pinned native Gunrock scale-anchor experiment",
        "sample_count": (
            int(row["timing_process_samples"])
            if finite(row.get("timing_process_samples")) else 0
        ),
        "validation_status": row.get("validation"),
    }
    if finite(row.get("e2e_mean_seconds")):
        result.update(
            {
                "e2e_paper_seconds": float(row["e2e_mean_seconds"]),
                "e2e_raw_mean_seconds": float(row["e2e_mean_seconds"]),
                "e2e_std_seconds": float(row.get("e2e_stdev_seconds", 0.0)),
                "e2e_estimator": "arithmetic_mean",
            }
        )
    if finite(row.get("kernel_mean_seconds")):
        result.update(
            {
                "kernel_paper_seconds": float(row["kernel_mean_seconds"]),
                "kernel_raw_mean_seconds": float(row["kernel_mean_seconds"]),
                "kernel_std_seconds": float(row.get("kernel_stdev_seconds", 0.0)),
                "kernel_estimator": "arithmetic_mean",
            }
        )
    for source, target in (
        ("gpu_process_peak_mb_mean", "gpu_peak_mb_mean"),
        ("gpu_process_peak_mb_stdev", "gpu_peak_mb_std"),
        ("host_rss_peak_mb_mean", "host_rss_peak_mb_mean"),
        ("host_rss_peak_mb_stdev", "host_rss_peak_mb_std"),
    ):
        if finite(row.get(source)):
            result[target] = float(row[source])
    result["memory_sample_count"] = (
        int(row["memory_samples"]) if finite(row.get("memory_samples")) else None
    )
    result["memory_measurement_window"] = "isolated_worker_process_full_lifetime"
    return result


def sygraph_record(data, dataset, function):
    if function != "BFS" or data.empty:
        return None
    required = {"dataset", "function", "baseline", "status"}
    if not required.issubset(data.columns):
        return None
    part = data[
        data["dataset"].eq(dataset)
        & data["function"].eq(function)
        & data["baseline"].eq("SYgraph")
    ]
    if part.empty:
        return None
    row = part.iloc[-1]
    status = str(row.get("status", "unknown"))
    result = {
        "execution_status": status,
        "failure_kind": first_text(row, "failure_kind"),
        "reason": first_text(row, "reason"),
        "result_source": "native upstream SYgraph BFS qualification",
        "sample_count": int(row["timing_samples"]) if finite(row.get("timing_samples")) else 0,
        "validation_status": row.get("validation"),
    }
    if status != "ok" or str(row.get("validation")) != "pass":
        return result
    result.update(
        {
            "build_paper_seconds": float(row["build_mean"]),
            "build_raw_mean_seconds": float(row["build_mean"]),
            "build_std_seconds": float(row.get("build_stdev", 0.0)),
            "build_estimator": "arithmetic_mean_native_SYgraph_graph_prepare",
            "e2e_paper_seconds": float(row["e2e_mean"]),
            "e2e_raw_mean_seconds": float(row["e2e_mean"]),
            "e2e_std_seconds": float(row.get("e2e_stdev", 0.0)),
            "e2e_estimator": "arithmetic_mean",
            "kernel_paper_seconds": float(row["kernel_mean"]),
            "kernel_raw_mean_seconds": float(row["kernel_mean"]),
            "kernel_std_seconds": float(row.get("kernel_stdev", 0.0)),
            "kernel_estimator": "arithmetic_mean",
        }
    )
    for source, target in (
        ("gpu_peak_mb_mean", "gpu_peak_mb_mean"),
        ("gpu_peak_mb_stdev", "gpu_peak_mb_std"),
        ("host_rss_peak_mb_mean", "host_rss_peak_mb_mean"),
        ("host_rss_peak_mb_stdev", "host_rss_peak_mb_std"),
    ):
        if finite(row.get(source)):
            result[target] = float(row[source])
    result["memory_sample_count"] = (
        int(row["memory_samples"]) if finite(row.get("memory_samples")) else None
    )
    result["memory_measurement_window"] = "isolated_worker_process_full_lifetime"
    return result


def default_large_outcome(dataset, function, baseline):
    support = SUPPORT[baseline][function]
    if support == F:
        return {
            "execution_status": "unsupported_api",
            "failure_kind": "unsupported_api",
            "reason": "No aligned callable implementation under the audited function-support contract.",
            "result_source": "function-support audit",
        }
    return {
        "execution_status": "missing_experiment",
        "failure_kind": "missing_experiment",
        "reason": (
            "The function is supported or conditionally supported, but this system has "
            "no measured large-anchor cell in the supplied result artifacts."
        ),
        "result_source": "completeness audit",
    }


def enforce_support_contract(row):
    """Exclude numeric results that violate the audited support contract.

    Historical runners occasionally timed a runner-side derivation or a
    backend fallback even though no aligned native function was available.
    Those observations remain useful provenance, but they are not eligible
    performance results under the paper's T/P/F definition.
    """

    if row.get("support_class") != F:
        return row
    has_algorithm_number = any(
        finite(row.get(key))
        for key in (
            "e2e_paper_seconds",
            "e2e_raw_mean_seconds",
            "kernel_paper_seconds",
            "kernel_raw_mean_seconds",
        )
    )
    if not has_algorithm_number and row.get("execution_status") != "ok":
        return row

    prior_status = str(row.get("execution_status") or "unknown")
    prior_reason = str(row.get("reason") or "").strip()
    row["excluded_observed_status"] = prior_status
    row["execution_status"] = "unsupported_api"
    row["failure_kind"] = "unsupported_api"
    row["validation_status"] = "excluded_by_support_contract"
    row["reason"] = (
        "No aligned native callable API exists under the audited support "
        "contract; a historical runner-side derivation or fallback result "
        "is retained only as provenance and excluded from comparison."
        + (f" Original outcome: {prior_reason}" if prior_reason else "")
    )
    row["result_source"] = "function-support audit over historical runner output"
    for key in list(row):
        if key.startswith(("e2e_", "kernel_", "gpu_peak_", "host_rss_peak_")):
            row[key] = None
    row["memory_sample_count"] = None
    row["memory_measurement_window"] = None
    return row


def markdown_report(ledger: pd.DataFrame, output: Path):
    missing = ledger[ledger["execution_status"].eq("missing_experiment")]
    unsuccessful = ledger[
        ledger["dataset"].isin(story.SCALE_ANCHORS)
        & ~ledger["execution_status"].eq("ok")
    ]
    lines = [
        "# EGGPU Final 13-Dataset Completeness and Failure Ledger",
        "",
        "This report distinguishes API/function support from observed execution. "
        "Unsupported, timeout, OOM, representation limits, semantic mismatches, "
        "and experiments that were never run are separate outcomes.",
        "",
        "## Coverage summary",
        "",
        "| Scope | Cells | OK | Missing experiment | Other explicit outcomes |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, part in (
        ("All 13 datasets", ledger),
        ("11 cross-library datasets", ledger[ledger["dataset"].isin(story.CROSS_LIBRARY_11)]),
        ("2 real scale anchors", ledger[ledger["dataset"].isin(story.SCALE_ANCHORS)]),
    ):
        lines.append(
            f"| {label} | {len(part)} | {int(part['execution_status'].eq('ok').sum())} | "
            f"{int(part['execution_status'].eq('missing_experiment').sum())} | "
            f"{int((~part['execution_status'].isin(['ok', 'missing_experiment'])).sum())} |"
        )
    lines.extend(["", "## Missing experiments", ""])
    if missing.empty:
        lines.append("None.")
    else:
        for (dataset, baseline), part in missing.groupby(["dataset", "baseline"], sort=True):
            lines.append(
                f"- **{dataset} / {baseline}:** {', '.join(part['function'].tolist())}."
            )
    lines.extend(["", "## Explicit non-OK outcomes on scale anchors", ""])
    for _, row in unsuccessful.sort_values(["dataset", "baseline", "function"]).iterrows():
        reason = str(row.get("reason") or "").replace("\n", " ")
        lines.append(
            f"- **{row['dataset']} / {row['baseline']} / {row['function']}**: "
            f"`{row['execution_status']}` (`{row.get('failure_kind', '')}`). {reason}"
        )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "A `missing_experiment` row is a hard completeness gap and must not be "
            "presented as unsupported, timeout, OOM, or a measured result. The paper "
            "may report a narrower comparison scope only if that scope is stated "
            "explicitly.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-result", required=True, type=Path)
    parser.add_argument("--closeness-result", required=True, type=Path)
    parser.add_argument("--closeness-memory-result", type=Path)
    parser.add_argument("--core-result", required=True, type=Path)
    parser.add_argument("--sygraph-result", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    _validation, raw, _strict, paper, _samples, _policy = story.load_main_views(
        args.main_result.resolve()
    )
    # Four scale-guarded datasets use the separately validated exact
    # 16-target Closeness contract.  Merge those samples before constructing
    # the cell ledger so a completed supplement is never misclassified as a
    # missing main-table workload.
    story.merge_closeness(raw, paper, args.closeness_result.resolve())
    main_memory = load_main_memory(args.main_result.resolve())
    supplement_memory = load_supplement_memory(
        args.closeness_memory_result.resolve()
        if args.closeness_memory_result
        else None
    )
    if not supplement_memory.empty:
        # Put the externally validated, isolated Closeness memory pass last so
        # main_memory_record selects it for the four supplemented datasets.
        main_memory = pd.concat(
            [main_memory, supplement_memory], ignore_index=True, sort=False
        )
    main_outcomes, main_validation = load_main_outcome_evidence(
        args.main_result.resolve(), args.closeness_result.resolve()
    )
    eggpu_large = load_large_eggpu(args.core_result.resolve())
    nxcg_large = load_large_nxcg(args.core_result.resolve())
    cpu_large = load_large_cpu(args.core_result.resolve())
    gunrock_large = load_large_gunrock(args.core_result.resolve())
    sygraph = load_sygraph(args.sygraph_result.resolve() if args.sygraph_result else None)
    baselines = list(BASELINES)
    if not sygraph.empty:
        baselines.insert(-1, "SYgraph")

    rows = []
    for dataset in story.FINAL_13:
        for function in FUNCTIONS:
            for baseline in baselines:
                row = {
                    "dataset": dataset,
                    "function": function,
                    "category": base.FUNCTION_CATEGORY[function],
                    "baseline": baseline,
                    "support_class": SUPPORT[baseline][function],
                    "execution_status": "unknown",
                    "failure_kind": "",
                    "reason": "",
                }
                if baseline == "SYgraph":
                    row.update(
                        sygraph_record(sygraph, dataset, function)
                        or default_large_outcome(dataset, function, baseline)
                    )
                elif dataset in story.CROSS_LIBRARY_11:
                    build = main_record(
                        paper["build"], dataset, function, baseline, "build"
                    )
                    e2e = main_record(paper["e2e"], dataset, function, baseline, "e2e")
                    kernel = main_record(
                        paper["kernel"], dataset, function, baseline, "kernel"
                    )
                    raw_outcome = None
                    if build is None or (e2e is None and kernel is None):
                        raw_outcome = main_outcome_record(
                            main_outcomes,
                            main_validation,
                            dataset,
                            function,
                            baseline,
                        )
                    observed = e2e or kernel
                    if observed is None:
                        observed = (
                            raw_outcome
                            or default_large_outcome(dataset, function, baseline)
                        )
                    row.update(observed)
                    if build:
                        for key, value in build.items():
                            if key.startswith("build_"):
                                row[key] = value
                    elif raw_outcome:
                        for key, value in raw_outcome.items():
                            if key.startswith("build_"):
                                row[key] = value
                    if e2e:
                        row.update(e2e)
                    if kernel:
                        for key, value in kernel.items():
                            if key.startswith("kernel_"):
                                row[key] = value
                    row.update(
                        main_memory_record(
                            main_memory, dataset, function, baseline
                        )
                    )
                elif baseline == "EGGPU":
                    row.update(
                        large_eggpu_record(
                            eggpu_large, args.core_result.resolve(), dataset, function
                        )
                        or default_large_outcome(dataset, function, baseline)
                    )
                elif baseline == "nx-cugraph":
                    row.update(
                        large_nxcg_record(nxcg_large, dataset, function)
                        or default_large_outcome(dataset, function, baseline)
                    )
                elif baseline in {"igraph", "networkx", "easygraph-cpu", "easygraph-cpp"}:
                    row.update(
                        large_cpu_record(cpu_large, dataset, function, baseline)
                        or default_large_outcome(dataset, function, baseline)
                    )
                elif baseline == "Gunrock":
                    row.update(
                        large_gunrock_record(gunrock_large, dataset, function)
                        or default_large_outcome(dataset, function, baseline)
                    )
                else:
                    row.update(default_large_outcome(dataset, function, baseline))
                rows.append(enforce_support_contract(row))

    ledger = pd.DataFrame(rows)
    ledger.to_csv(output / "final_13_cell_outcome_ledger.csv", index=False)
    markdown_report(ledger, output / "FINAL_13_FAILURE_AND_COMPLETENESS_REPORT.md")
    summary = {
        "cells": len(ledger),
        "datasets": ledger["dataset"].nunique(),
        "functions": ledger["function"].nunique(),
        "baselines": ledger["baseline"].nunique(),
        "status_counts": ledger["execution_status"].value_counts().to_dict(),
        "missing_experiment_cells": int(
            ledger["execution_status"].eq("missing_experiment").sum()
        ),
    }
    (output / "final_13_cell_outcome_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
