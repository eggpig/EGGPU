#!/usr/bin/env python3
"""Self-describing metric schema shared by EGGPU benchmark runners."""

from __future__ import annotations

import json
from pathlib import Path


SCHEMA_VERSION = 6


def _kernel_timer_kind(baseline):
    if baseline == "EGGPU":
        return "cuda_event"
    if baseline == "nx-cugraph":
        return "cuda_event_at_pylibcugraph_backend"
    if baseline == "Gunrock":
        return "external_reported_device_timer"
    return "algorithm_wall_time_surrogate"


def _measurement_window(metric, baseline):
    if baseline == "Gunrock":
        if metric == "kernel":
            return "device_execution_reported_by_external_cli"
        return "external_cli_process"
    if metric == "build":
        return "graph_construction"
    if metric == "e2e":
        return "python_function_call"
    if metric == "kernel":
        return (
            "device_execution"
            if baseline in {"EGGPU", "nx-cugraph"}
            else "algorithm_call"
        )
    if str(metric).startswith("memory_"):
        return "algorithm_call"
    return "ablation_component"


def describe_metric(metric, baseline=""):
    """Return unit and provenance fields for a numeric benchmark metric."""

    metric = str(metric or "")
    baseline = str(baseline or "")
    if metric in {"build", "e2e", "kernel"}:
        scope = {
            "build": "baseline_graph_construction",
            "e2e": "user_visible_function_call",
            "kernel": "algorithm_compute",
        }[metric]
        if metric == "e2e" and baseline == "Gunrock":
            scope = "external_cli_invocation"
        timer_kind = _kernel_timer_kind(baseline) if metric == "kernel" else "perf_counter_wall"
        return {
            "unit": "s",
            "metric_family": "time",
            "measurement_scope": scope,
            "timer_kind": timer_kind,
            "measurement_window": _measurement_window(metric, baseline),
        }

    if metric.startswith("memory_"):
        if metric == "memory_probe_calls":
            return {
                "unit": "count",
                "metric_family": "measurement_quality",
                "measurement_scope": "memory_monitor",
                "timer_kind": "counter",
                "measurement_window": _measurement_window(metric, baseline),
            }
        if metric == "memory_probe_window_seconds":
            return {
                "unit": "s",
                "metric_family": "measurement_quality",
                "measurement_scope": "memory_monitor",
                "timer_kind": "perf_counter_wall",
                "measurement_window": _measurement_window(metric, baseline),
            }
        if "monitor" in metric and "samples" in metric:
            return {
                "unit": "count",
                "metric_family": "measurement_quality",
                "measurement_scope": "memory_monitor",
                "timer_kind": "counter",
                "measurement_window": _measurement_window(metric, baseline),
            }
        if "monitor" in metric and "poll" in metric:
            return {
                "unit": "ms",
                "metric_family": "measurement_quality",
                "measurement_scope": "memory_monitor",
                "timer_kind": "configured_interval",
                "measurement_window": _measurement_window(metric, baseline),
            }
        if "monitor" in metric and "window" in metric:
            return {
                "unit": "s",
                "metric_family": "measurement_quality",
                "measurement_scope": "memory_monitor",
                "timer_kind": "perf_counter_wall",
                "measurement_window": _measurement_window(metric, baseline),
            }
        if "rss" in metric:
            timer_kind = "psutil_process_tree_sampling"
            scope = "benchmark_process_tree_cpu_memory"
        elif "gpu_proc" in metric:
            timer_kind = "nvml_process_tree_sampling"
            scope = "benchmark_process_tree_gpu_memory"
        elif "gpu_non_process" in metric:
            timer_kind = "nvml_derived_sampling"
            scope = "non_benchmark_gpu_memory_at_window_start"
        elif "gpu" in metric:
            timer_kind = "nvml_whole_device_sampling"
            scope = "whole_physical_gpu_memory_diagnostic"
        else:
            timer_kind = "memory_sampling"
            scope = "benchmark_memory"
        return {
            "unit": "MiB",
            "metric_family": "memory",
            "measurement_scope": scope,
            "timer_kind": timer_kind,
            "measurement_window": _measurement_window(metric, baseline),
        }

    if metric.endswith("_seconds") or metric in {
        "build_graph_bundle",
        "call_return_seconds",
        "forced_result_traversal_seconds",
        "call_plus_forced_traversal_seconds",
    }:
        timer_kind = "cuda_event" if "kernel" in metric else "perf_counter_wall"
        return {
            "unit": "s",
            "metric_family": "time",
            "measurement_scope": "ablation_component",
            "timer_kind": timer_kind,
            "measurement_window": "ablation_component",
        }

    if metric.endswith("_mb") or metric == "host_storage_mb":
        return {
            "unit": "MiB",
            "metric_family": "memory",
            "measurement_scope": "ablation_component",
            "timer_kind": "derived_bytes",
            "measurement_window": "ablation_component",
        }

    if metric.endswith("_relative_error"):
        return {
            "unit": "ratio",
            "metric_family": "correctness",
            "measurement_scope": "ablation_equivalence_check",
            "timer_kind": "derived_numeric_check",
            "measurement_window": "ablation_component",
        }

    return {
        "unit": "",
        "metric_family": "other",
        "measurement_scope": "",
        "timer_kind": "",
        "measurement_window": _measurement_window(metric, baseline),
    }


def schema_document():
    return {
        "schema_version": SCHEMA_VERSION,
        "value_columns": {
            "value": "Canonical numeric value in the declared unit.",
            "seconds": "Legacy compatibility alias. It can contain MiB/count values for historical readers; use value+unit in new analysis.",
            "external_cli_wall_seconds": (
                "Diagnostic whole-process wall time for external executables. "
                "It includes process startup and file/graph loading and is not "
                "used as Gunrock E2E when the aligned host-result timer exists."
            ),
        },
        "primary_metrics": {
            "time": "e2e",
            "gpu_memory": "memory_peak_gpu_proc_mb",
            "cpu_memory": "memory_peak_rss_mb",
        },
        "memory_semantics": {
            "process_tree_absolute_peak": "Primary comparable GPU footprint. In split runs it is sampled by the parent over the complete isolated memory subprocess, including graph build, untimed warmup, algorithm execution, and result emission.",
            "process_tree_delta": "Incremental allocation above the process-tree value at the start of the timed window; it can be zero when warmup retained the allocation.",
            "whole_device": "Diagnostic only. It can include unrelated processes and must not be used for cross-library paper comparisons.",
            "sampling": "NVML/psutil sampling can miss a transient shorter than the poll interval; sample counts, poll interval, and monitor window are recorded.",
        },
        "split_measurement_protocol": {
            "timing_pass": "Independent subprocess samples with psutil/NVML memory sampling disabled. Only this pass supplies build/e2e/kernel values to performance tables.",
            "memory_pass": "Independent subprocess samples with psutil/NVML sampling enabled. Its timing values are discarded and only memory/monitor-quality metrics are merged.",
            "phase_column": "measurement_phase identifies timing, memory, or legacy combined provenance for every raw and aggregate row.",
            "repeat_contract": "Timing and memory metrics may have different repeat counts; each aggregate records its own sample_count.",
        },
        "eggpu_execution_protocols": {
            "steady-state": (
                "After Python graph construction, the benchmark prepares GraphContext/C++ graph state, "
                "performs the configured untimed EGGPU function warmups, and then measures one user-visible call."
            ),
            "first-use": (
                "A fresh subprocess builds only the Python graph and then measures the first supported "
                "EGGPU function call with no GraphContext/C++ prewarm and no function warmup."
            ),
            "natural-workflow": (
                "A fresh subprocess builds one Python graph and invokes WCC, PageRank, and BFS in sequence "
                "without artificial prewarm; later calls may reuse state created by earlier calls."
            ),
        },
        "timer_semantics": {
            "cuda_event": "Device elapsed time reported by EGGPU CUDA events.",
            "cuda_event_at_pylibcugraph_backend": (
                "CUDA-event interval recorded at the actual pylibcugraph "
                "algorithm boundary on an explicit RAFT-bound stream."
            ),
            "external_reported_device_timer": "Device elapsed time printed by the external Gunrock executable.",
            "steady_clock_wall": (
                "Host steady-clock interval. For instrumented Gunrock binaries, "
                "it starts on a prepared graph and ends after the complete "
                "requested result has been copied to host memory."
            ),
            "algorithm_wall_time_surrogate": "Wall-clock algorithm time used when a baseline does not expose device-only timing.",
            "perf_counter_wall": "Host monotonic wall-clock interval.",
        },
        "measurement_windows": {
            "graph_construction": (
                "Baseline-native graph preparation after raw edge-list parsing. For EGGPU steady-state this "
                "includes Python graph construction plus GraphContext/C++ graph-container prebuild; for EGGPU "
                "first-use it contains Python graph construction only."
            ),
            "python_function_call": "The user-visible Python function invocation; the graph object already exists.",
            "device_execution": "CUDA device work delimited by EGGPU events.",
            "algorithm_call": "The in-process algorithm call used by CPU or Python GPU baselines.",
            "external_cli_process": "The complete external Gunrock process, excluding MatrixMarket conversion performed by the runner.",
            "device_execution_reported_by_external_cli": "The device timer reported by the Gunrock executable.",
            "isolated_memory_subprocess": "An independent memory-only subprocess monitored externally by its parent; no value from this pass is used as a performance timing.",
            "ablation_component": "The explicitly named component of an EGGPU ablation.",
        },
        "aggregation": {
            "estimator": "arithmetic_mean",
            "dispersion": ["sample_standard_deviation", "sample_variance", "relative_standard_deviation_percent"],
            "interval": "two-sided_95_percent_student_t",
            "single_sample_memory": "When memory_repeat=1, peak memory is descriptive and dispersion/CI fields are intentionally empty.",
        },
    }


def write_measurement_schema(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema_document(), indent=2, sort_keys=True) + "\n")
    return path
