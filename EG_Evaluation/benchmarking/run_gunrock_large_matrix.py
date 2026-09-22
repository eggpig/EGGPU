#!/usr/bin/env python3
"""Run pinned Gunrock executables with strict three-phase native timing.

The runner never substitutes another backend.  Every function receives either
five fresh-process timing samples plus three isolated memory samples, or an
explicit unsupported/representation/timeout/validation outcome.  MatrixMarket
file generation is performed before this program.  Each native timing sample
starts immediately before loading that file, records construction after device
graph materialization, retains Gunrock's native device processing interval, and
ends after the complete host result is available.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

from child_process_memory_monitor import ChildProcessMemoryMonitor
from gunrock_timing_protocol import (
    GunrockTimingProtocolError,
    parse_strict_gunrock_timing,
)
from pagerank_residual_validation import validate_pagerank_fixed_point
import run_full_baselines as shared


FUNCTIONS = (
    "PageRank", "MST", "LCC", "WCC", "SCC", "BFS", "Dijkstra",
    "BellmanFord", "SSSP", "KCore", "BC", "Closeness", "EffectiveSize",
    "Efficiency", "Constraint", "Hierarchy",
)
SUPPORT = {
    "PageRank": "T", "MST": "P", "LCC": "P", "WCC": "F", "SCC": "F",
    "BFS": "T", "Dijkstra": "T", "BellmanFord": "P", "SSSP": "T",
    "KCore": "T", "BC": "P", "Closeness": "F", "EffectiveSize": "F",
    "Efficiency": "F", "Constraint": "F", "Hierarchy": "F",
}
APP = {
    "PageRank": "pr", "MST": "mst", "LCC": "lcc", "BFS": "bfs",
    "Dijkstra": "sssp", "BellmanFord": "sssp", "SSSP": "sssp",
    "KCore": "kcore", "BC": "bc",
}
PROJECTED = {"MST", "LCC", "KCore"}
MULTISOURCE = {"BFS", "BellmanFord", "SSSP", "BC"}
UNSUPPORTED = {
    "WCC": "maintained Gunrock has no aligned WCC executable",
    "SCC": "maintained Gunrock has no aligned SCC executable",
    "Closeness": "no aligned Closeness executable",
    "EffectiveSize": "no Burt effective-size executable",
    "Efficiency": "no Burt efficiency executable",
    "Constraint": "no Burt constraint executable",
    "Hierarchy": "no Burt hierarchy executable",
}
INT32_MAX = (1 << 31) - 1


def stats(values):
    values = [float(value) for value in values]
    return {
        "mean": statistics.mean(values),
        "best": min(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "samples": values,
        "count": len(values),
    }


def flatten(record):
    return {
        key: json.dumps(value, ensure_ascii=False, sort_keys=True)
        if isinstance(value, (dict, list)) else value
        for key, value in record.items()
    }


def select_sources(metadata, count):
    recorded = metadata.get("benchmark_sources_zero_based") or []
    selected = []
    seen = set()
    for value in recorded:
        node = int(value)
        if 0 <= node < int(metadata["num_nodes"]) and node not in seen:
            selected.append(node)
            seen.add(node)
        if len(selected) == count:
            return selected
    for node in shared.deterministic_sources(int(metadata["num_nodes"]), count):
        if node not in seen:
            selected.append(node)
            seen.add(node)
        if len(selected) == count:
            break
    return selected


def applicability(metadata, function, matrix_path):
    if SUPPORT[function] == "F":
        return False, "unsupported_api", UNSUPPORTED[function]
    if not matrix_path.is_file():
        return False, "missing_input_artifact", f"aligned MatrixMarket input is absent: {matrix_path}"
    if function in PROJECTED and bool(metadata.get("directed")):
        projected = 2 * int(metadata.get("num_entries", 0))
        if projected > INT32_MAX:
            return False, "representation_limit", (
                f"{function} requires the common undirected projection with up to "
                f"{projected:,} entries, beyond the signed-int32 Gunrock/CSR ABI"
            )
        return False, "missing_projection_artifact", (
            f"{function} requires an explicit undirected projection; no directed-graph "
            "fallback or largest-component substitution is permitted"
        )
    return True, "", ""


def command_for(
    executable,
    function,
    matrix_path,
    source=None,
    pagerank_alpha=0.75,
    pagerank_tolerance=1.0e-6,
    result_file=None,
    validate=False,
):
    exe = str(executable)
    matrix = str(matrix_path)
    if function == "PageRank":
        command = [
            exe, "-m", matrix, "-n", "1",
            "--alpha", f"{float(pagerank_alpha):.12g}",
            "--tol", f"{float(pagerank_tolerance):.12g}",
        ]
        if result_file is not None:
            command.extend(["--result_file", str(result_file)])
        return command
    if function == "MST":
        return [exe, "-m", matrix]
    if function == "LCC":
        return [
            exe, "market", matrix, "--undirected=true", "--sort-csr=true",
            "--validation=none", "--quick=true",
        ]
    if function in {"BFS", "Dijkstra", "BellmanFord", "SSSP"}:
        if isinstance(source, (list, tuple)):
            source_arg = ",".join(str(int(item)) for item in source)
        else:
            source_arg = str(int(source))
        command = [exe, "-m", matrix, "-s", source_arg]
        if validate:
            command.append("--validate")
        if result_file is not None:
            command.extend(["--result_file", str(result_file)])
        return command
    if function == "KCore":
        command = [exe, matrix]
        if not validate:
            command.append("--no-validate")
        if result_file is not None:
            command.extend(["--result_file", str(result_file)])
        return command
    if function == "BC":
        if isinstance(source, (list, tuple)):
            source_arg = ",".join(str(int(item)) for item in source)
        else:
            source_arg = str(int(source))
        command = [exe, "-m", matrix, "-s", source_arg]
        if result_file is not None:
            command.extend(["--result_file", str(result_file)])
        return command
    raise ValueError(function)


def run_process(command, log_path, env, timeout, monitor_gpu=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=shared.ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    monitor = None
    if monitor_gpu is not None:
        monitor = ChildProcessMemoryMonitor(
            process.pid, physical_gpu=monitor_gpu, interval_seconds=0.002
        ).start()
    timed_out = False
    try:
        output, _ = process.communicate(timeout=max(0.001, timeout))
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        output, _ = process.communicate()
    finally:
        memory = monitor.stop() if monitor is not None else None
    wall = time.perf_counter() - started
    log_path.write_text(
        "$ " + " ".join(command) + "\n\n" + (output or "")
        + (f"\nTIMEOUT after {timeout:.6g}s\n" if timed_out else ""),
        encoding="utf-8",
        errors="replace",
    )
    return {
        "returncode": 124 if timed_out else process.returncode,
        "wall_seconds": wall,
        "output": output or "",
        "memory": memory,
        "timed_out": timed_out,
        "log": str(log_path),
    }


def combine_memory(items):
    items = [item for item in items if item]
    if not items:
        return {}
    result = {}
    for key in ("rss_mb", "rss_peak_delta_mb", "gpu_proc_peak_mb", "gpu_proc_peak_delta_mb"):
        values = [float(item[key]) for item in items if item.get(key) is not None]
        result[key] = max(values) if values else None
    for key in ("monitor_rss_samples", "monitor_gpu_proc_samples"):
        result[key] = sum(int(item.get(key, 0)) for item in items)
    result["monitor_window_seconds"] = sum(float(item.get("monitor_window_seconds", 0.0)) for item in items)
    result["memory_monitor_origin"] = "coordinator_process_child_tree"
    return result


def expected_mst_weight(eggpu_dir, dataset):
    path = eggpu_dir / "raw" / f"{dataset}_MST_timing.json"
    if not path.is_file():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8")).get("result", {})
        return float(result["total_weight"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def classify(function, invocations, expected_weight):
    if any(item["timed_out"] for item in invocations):
        return "timeout", "timeout", "one or more native executable invocations exceeded the shared function budget"
    if any(item["returncode"] != 0 for item in invocations):
        return "failed", "execution_error", "one or more native executable invocations exited non-zero"
    for item in invocations:
        try:
            item["strict_timing"] = parse_strict_gunrock_timing(item["output"])
        except GunrockTimingProtocolError as exc:
            return "failed", "invalid_three_phase_timing", (
                f"native output violates the strict construction/processing/E2E "
                f"contract: {exc}; external CLI wall time is diagnostic only"
            )
    if function in {"BFS", "Dijkstra", "BellmanFord", "SSSP", "KCore"}:
        note = "CPU validation is excluded from timed invocations and runs separately"
        if function == "Dijkstra":
            note += "; maintained SSSP executable is the named single-source Dijkstra alias"
        elif function == "BellmanFord":
            note += (
                "; maintained SSSP executable is a conditional implementation alias "
                "for this nonnegative-weight benchmark and does not establish "
                "negative-edge support"
            )
        return "ok", "", note
    if function == "MST":
        actual = shared.parse_gunrock_mst_weight(invocations[0]["output"])
        if actual is None:
            return "semantic_mismatch", "validation_missing", "Gunrock did not report a total MST weight"
        if expected_weight is None:
            return "semantic_mismatch", "validation_inconclusive", (
                f"Gunrock reported weight={actual:g}, but the aligned EGGPU reference weight is unavailable"
            )
        tolerance = max(1e-5, 1e-6 * abs(expected_weight))
        if not math.isclose(actual, expected_weight, rel_tol=1e-6, abs_tol=tolerance):
            return "semantic_mismatch", "validation_failed", (
                f"Gunrock MST weight={actual:g} differs from aligned reference={expected_weight:g}"
            )
        return "ok", "", f"total MST weight matches aligned reference ({expected_weight:g})"
    if function == "PageRank":
        return "ok", "", "aligned parameters; full-vector validation runs outside timing"
    if function == "LCC":
        return "ok", "", "full-vector validation runs outside timing"
    if function == "BC":
        return "ok", "", (
            "one process invokes the maintained single-source kernel for the "
            "complete specified source list; full-vector validation runs outside timing"
        )
    raise ValueError(function)


def run_sample(args, metadata, function, matrix_path, executable, sample_index, measurement):
    count = args.bc_source_count if function == "BC" else args.source_count
    if function == "Dijkstra":
        sources = select_sources(metadata, 1)
    else:
        sources = select_sources(metadata, count) if function in MULTISOURCE else [None]
    deadline = time.monotonic() + args.timeout
    invocations = []
    invocation_sources = [sources] if function in MULTISOURCE else sources
    for source_index, source in enumerate(invocation_sources, start=1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            invocations.append({
                "returncode": 124, "wall_seconds": 0.0, "output": "",
                "memory": None, "timed_out": True, "log": "",
            })
            break
        if isinstance(source, (list, tuple)):
            suffix = f"_sources_{len(source)}"
        else:
            suffix = f"_source_{source}" if source is not None else ""
        log = args.out_dir / "logs" / metadata["name"] / (
            f"{function}_{measurement}_{sample_index}{suffix}.log"
        )
        env = shared.gunrock_runtime_env(executable, os.environ.copy())
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        invocation = run_process(
            command_for(
                executable,
                function,
                matrix_path,
                source,
                pagerank_alpha=args.pagerank_alpha,
                pagerank_tolerance=args.pagerank_tolerance,
            ),
            log,
            env,
            remaining,
            monitor_gpu=args.gpu if measurement == "memory" else None,
        )
        invocation["source"] = source
        invocation["source_index"] = source_index
        invocations.append(invocation)
        if invocation["returncode"] != 0:
            break
    expected = expected_mst_weight(args.eggpu_result_dir, metadata["name"])
    status, failure_kind, validation_note = classify(function, invocations, expected)
    strict_timings = [
        item["strict_timing"]
        for item in invocations
        if item.get("strict_timing") is not None
    ]
    construction = sum(
        float(item["construction_seconds"]) for item in strict_timings
    )
    kernel = sum(
        float(item["processing_seconds"]) for item in strict_timings
    )
    aligned_e2e = sum(float(item["e2e_seconds"]) for item in strict_timings)
    external_wall = sum(float(item["wall_seconds"]) for item in invocations)
    return {
        "status": status,
        "failure_kind": failure_kind,
        "validation": "pass" if status == "ok" else status,
        "validation_note": validation_note,
        "build_seconds": construction if construction > 0 else None,
        "kernel_seconds": kernel if kernel > 0 else None,
        "e2e_seconds": aligned_e2e if aligned_e2e > 0 else None,
        "external_cli_wall_seconds": external_wall,
        "measurement_scope": "standalone_gpu_function",
        "build_measurement_window": "matrix_market_load_to_device_graph",
        "kernel_measurement_window": "native_device_interval",
        "measurement_window": "matrix_market_load_to_complete_host_result",
        "timing_protocol": (
            strict_timings[0] if len(strict_timings) == 1 else strict_timings
        ),
        "external_cli_wall_used": False,
        "sources": sources,
        "measurement": measurement,
        "sample_index": sample_index,
        "invocations": [
            {key: value for key, value in item.items() if key != "output"}
            for item in invocations
        ],
        "memory": combine_memory([item.get("memory") for item in invocations]),
    }


def validate_vector_output(args, metadata, function, matrix_path, executable):
    """Validate a full output vector in a separate, unmeasured invocation."""

    validation_dir = args.out_dir / "validation" / metadata["name"]
    validation_dir.mkdir(parents=True, exist_ok=True)
    raw_path = validation_dir / f"{function}.raw"
    log_path = validation_dir / f"{function}.log"
    env = shared.gunrock_runtime_env(executable, os.environ.copy())
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    source_array = None
    if function in {"BFS", "Dijkstra", "BellmanFord", "SSSP"}:
        source_count = 1 if function == "Dijkstra" else args.source_count
        source_array = select_sources(metadata, source_count)
        rows = []
        invocation_logs = []
        for source in source_array:
            source_raw = validation_dir / f"{function}_source_{int(source)}.raw"
            source_log = validation_dir / f"{function}_source_{int(source)}.log"
            command = command_for(
                executable,
                function,
                matrix_path,
                source=source,
                result_file=source_raw,
            )
            invocation = run_process(command, source_log, env, args.timeout)
            invocation_logs.append(str(source_log))
            if invocation["returncode"] != 0 or not source_raw.is_file():
                source_raw.unlink(missing_ok=True)
                return {
                    "status": "semantic_mismatch",
                    "failure_kind": "validation_missing",
                    "note": f"full-result export failed for source={int(source)}",
                    "log": str(source_log),
                }
            raw_dtype = np.int32 if function == "BFS" else np.float32
            row = np.fromfile(source_raw, dtype=raw_dtype)
            source_raw.unlink(missing_ok=True)
            if row.size != int(metadata["num_nodes"]):
                return {
                    "status": "semantic_mismatch",
                    "failure_kind": "validation_failed",
                    "note": (
                        f"source={int(source)} returned {row.size} values for "
                        f"{int(metadata['num_nodes'])} vertices"
                    ),
                    "log": str(source_log),
                }
            comparable = row.astype(np.float64)
            if function == "BFS":
                comparable[row == np.iinfo(np.int32).max] = np.inf
            else:
                comparable[np.abs(comparable) >= 1.0e30] = np.inf
            rows.append(comparable)
        comparable_values = np.ascontiguousarray(np.stack(rows, axis=0))
        values = comparable_values
        log_path.write_text("\n".join(invocation_logs) + "\n", encoding="utf-8")
    elif function == "KCore":
        dtype = np.int32
        command = command_for(
            executable,
            function,
            matrix_path,
            result_file=raw_path,
        )
        invocation = run_process(command, log_path, env, args.timeout)
        if invocation["returncode"] != 0 or not raw_path.is_file():
            return {
                "status": "semantic_mismatch",
                "failure_kind": "validation_missing",
                "note": "separate full-vector KCore export failed",
                "log": str(log_path),
            }
        values = np.fromfile(raw_path, dtype=dtype)
        comparable_values = np.asarray(values)
    elif function == "PageRank":
        dtype = np.float32
        command = command_for(
            executable,
            function,
            matrix_path,
            pagerank_alpha=args.pagerank_alpha,
            pagerank_tolerance=args.pagerank_tolerance,
            result_file=raw_path,
        )
    elif function == "LCC":
        dtype = np.float64
        command = command_for(executable, function, matrix_path)
        env["EG_GUNROCK_LCC_RESULT_FILE"] = str(raw_path)
    elif function == "BC":
        dtype = np.float32
        sources = select_sources(metadata, args.bc_source_count)
        command = command_for(
            executable,
            function,
            matrix_path,
            source=sources,
            result_file=raw_path,
        )
    else:
        raise ValueError(function)

    if function not in {"BFS", "Dijkstra", "BellmanFord", "SSSP", "KCore"}:
        invocation = run_process(command, log_path, env, args.timeout)
        if invocation["returncode"] != 0 or not raw_path.is_file():
            return {
                "status": "semantic_mismatch",
                "failure_kind": "validation_missing",
                "note": "separate full-vector validation invocation failed",
                "log": str(log_path),
            }

        values = np.memmap(raw_path, dtype=dtype, mode="r")
        comparable_values = np.asarray(values, dtype=np.float64)
        if function == "BC" and bool(metadata["directed"]):
            # Gunrock's maintained BC kernel applies a 1/2 undirected-path factor
            # internally. Restore the directed NetworkX subset-result convention.
            comparable_values = comparable_values * 2.0
    expected_nodes = int(metadata["num_nodes"])
    failures = []
    expected_shape = (
        (len(source_array), expected_nodes)
        if source_array is not None
        else (expected_nodes,)
    )
    if comparable_values.shape != expected_shape:
        failures.append(
            f"returned shape {comparable_values.shape}, expected {expected_shape}"
        )
    elif not np.isfinite(comparable_values).all():
        failures.append("result contains non-finite values")
    elif function == "PageRank" and float(values.min(initial=0.0)) < -1.0e-12:
        failures.append("PageRank contains negative values")
    elif function == "LCC" and (
        float(values.min(initial=0.0)) < -1.0e-12
        or float(values.max(initial=0.0)) > 1.0 + 1.0e-9
    ):
        failures.append("LCC values fall outside [0, 1]")

    reference_path = args.eggpu_result_dir / "raw" / (
        f"{metadata['name']}_{function}_timing.json"
    )
    max_abs_error = None
    residual_evidence = None
    if not failures and function == "PageRank":
        manifest_path = Path(metadata.get("_manifest_path", ""))
        if not manifest_path.is_file():
            failures.append("CSR manifest is unavailable for PageRank residual validation")
        else:
            residual_evidence = validate_pagerank_fixed_point(
                raw_path,
                manifest_path,
                alpha=args.pagerank_alpha,
            )
            residual_path = validation_dir / "PageRank.fixed_point.json"
            residual_path.write_text(
                json.dumps(residual_evidence, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            residual_evidence["evidence_path"] = str(residual_path)
            if residual_evidence["status"] != "pass":
                failures.append(
                    "PageRank fixed-point validation failed: "
                    f"mean residual={residual_evidence['mean_residual']:.6g}, "
                    f"threshold={residual_evidence['mean_residual_tolerance']:.6g}"
                )
        if reference_path.is_file():
            reference = json.loads(reference_path.read_text(encoding="utf-8"))
            reference_sample = np.asarray(
                reference.get("result", {}).get("validation_sample", []),
                dtype=np.float64,
            )
            if reference_sample.size:
                indices = np.linspace(
                    0, expected_nodes - 1, num=reference_sample.size, dtype=np.int64
                )
                observed_sample = np.asarray(comparable_values[indices], dtype=np.float64)
                max_abs_error = float(np.max(np.abs(observed_sample - reference_sample)))
    elif not failures and reference_path.is_file():
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        reference_digest = reference.get("result", {})
        reference_sample = np.asarray(
            reference_digest.get("validation_sample", []),
            dtype=np.float64,
        )
        if reference_sample.size:
            indices = np.linspace(
                0, expected_nodes - 1, num=reference_sample.size, dtype=np.int64
            )
            observed_sample = np.asarray(comparable_values[indices], dtype=np.float64)
            max_abs_error = float(np.max(np.abs(observed_sample - reference_sample)))
            rtol, atol = (1.0e-6, 1.0e-8)
            if not np.allclose(observed_sample, reference_sample, rtol=rtol, atol=atol):
                failures.append(
                    f"sampled output differs from EGGPU reference; max_abs_error={max_abs_error:.6g}"
                )
        elif function in {"BFS", "Dijkstra", "BellmanFord", "SSSP", "KCore"}:
            observed = np.ascontiguousarray(comparable_values)
            observed_finite = np.isfinite(observed) if observed.dtype.kind == "f" else None
            observed_digest = {
                "shape": list(observed.shape),
                "dtype": str(observed.dtype),
                "finite_count": int(observed_finite.sum()) if observed_finite is not None else None,
                "inf_count": int(np.isinf(observed).sum()) if observed_finite is not None else None,
                "zero_count": int(np.count_nonzero(observed == 0)),
                "sum": float(np.nansum(observed, dtype=np.float64)),
                "minimum": float(observed[observed_finite].min()) if observed_finite is not None and observed_finite.any() else int(observed.min()),
                "maximum": float(observed[observed_finite].max()) if observed_finite is not None and observed_finite.any() else int(observed.max()),
            }
            for key in ("shape", "finite_count", "inf_count", "zero_count"):
                if reference_digest.get(key) != observed_digest.get(key):
                    failures.append(
                        f"result summary differs for {key}: "
                        f"observed={observed_digest.get(key)}, "
                        f"reference={reference_digest.get(key)}"
                    )
            for key in ("sum", "minimum", "maximum"):
                lhs = float(observed_digest[key])
                rhs = float(reference_digest[key])
                if not (
                    (math.isinf(lhs) and math.isinf(rhs) and (lhs > 0) == (rhs > 0))
                    or math.isclose(lhs, rhs, rel_tol=1.0e-5, abs_tol=1.0e-6)
                ):
                    failures.append(
                        f"result summary differs for {key}: observed={lhs}, reference={rhs}"
                    )
            if function == "KCore":
                observed_sha = __import__("hashlib").sha256(observed.view(np.uint8)).hexdigest()
                if observed_sha != reference_digest.get("sha256"):
                    failures.append("KCore full-vector SHA-256 differs from EGGPU reference")
        else:
            failures.append("EGGPU reference has no validation sample")
    elif not failures:
        failures.append(f"EGGPU reference is absent: {reference_path}")

    del values
    raw_path.unlink(missing_ok=True)
    return {
        "status": "pass" if not failures else "semantic_mismatch",
        "failure_kind": "" if not failures else "validation_failed",
        "note": (
            "full PageRank vector satisfies the fixed-point equation"
            if function == "PageRank" and not failures
            else "full output shape and evenly spaced samples match EGGPU"
            if not failures else "; ".join(failures)
        ),
        "max_abs_error": max_abs_error,
        "fixed_point_evidence": residual_evidence,
        "log": str(log_path),
        "measurement_window": "separate_unmeasured_validation_process",
    }


def aggregate(args, metadata, function, matrix_path, executable):
    timing = []
    terminal = {"timeout", "failed"}
    for index in range(1, args.repeat + 1):
        sample = run_sample(args, metadata, function, matrix_path, executable, index, "timing")
        timing.append(sample)
        print(f"[gunrock-large] {metadata['name']}/{function} timing {index}/{args.repeat}: {sample['status']}", flush=True)
        if sample["status"] in terminal:
            break
    observed = [
        row
        for row in timing
        if row.get("build_seconds") is not None
        and row.get("kernel_seconds") is not None
        and row.get("e2e_seconds") is not None
    ]
    statuses = {row["status"] for row in timing}
    status = timing[-1]["status"] if len(statuses) == 1 else (
        "failed" if "failed" in statuses else "timeout" if "timeout" in statuses else "semantic_mismatch"
    )
    record = {
        "dataset": metadata["name"], "function": function, "baseline": "Gunrock",
        "support_class": SUPPORT[function], "status": status,
        "failure_kind": timing[-1].get("failure_kind", ""),
        "validation": timing[-1].get("validation"),
        "validation_note": timing[-1].get("validation_note"),
        "num_nodes": int(metadata["num_nodes"]), "num_entries": int(metadata["num_entries"]),
        "directed": bool(metadata["directed"]), "timing_process_samples": len(observed),
        "timing_process_records": timing, "matrix_market_input": str(matrix_path),
    }
    if function == "Dijkstra":
        record["implementation_alias"] = "Gunrock maintained SSSP executable; one source"
    elif function == "BellmanFord":
        record["implementation_alias"] = (
            "Gunrock maintained SSSP executable; benchmark weights are nonnegative; "
            "negative-edge Bellman-Ford semantics are not claimed"
        )
    if observed:
        build = stats([row["build_seconds"] for row in observed])
        e2e = stats([row["e2e_seconds"] for row in observed])
        kernel = stats([row["kernel_seconds"] for row in observed])
        record.update({
            "build": build, "e2e": e2e, "kernel": kernel,
            "build_mean_seconds": build["mean"],
            "build_stdev_seconds": build["stdev"],
            "build_best_seconds": build["best"],
            "e2e_mean_seconds": e2e["mean"], "e2e_stdev_seconds": e2e["stdev"],
            "e2e_best_seconds": e2e["best"], "kernel_mean_seconds": kernel["mean"],
            "kernel_stdev_seconds": kernel["stdev"], "kernel_best_seconds": kernel["best"],
        })
    if observed and status == "ok" and function in {
        "PageRank", "LCC", "BC", "BFS", "Dijkstra", "BellmanFord", "SSSP", "KCore"
    }:
        validation_probe = validate_vector_output(
            args, metadata, function, matrix_path, executable
        )
        record["external_validation_probe"] = validation_probe
        if validation_probe["status"] != "pass":
            status = "semantic_mismatch"
            record["status"] = status
            record["failure_kind"] = validation_probe["failure_kind"]
            record["validation"] = "semantic_mismatch"
            record["validation_note"] = validation_probe["note"]
        else:
            record["validation"] = "pass"
            record["validation_note"] = validation_probe["note"]
    if observed and status in {"ok", "semantic_mismatch"}:
        memory_rows = []
        for index in range(1, args.memory_repeat + 1):
            sample = run_sample(args, metadata, function, matrix_path, executable, index, "memory")
            memory_rows.append(sample)
            print(f"[gunrock-large] {metadata['name']}/{function} memory {index}/{args.memory_repeat}: {sample['status']}", flush=True)
            if sample["status"] in terminal:
                break
        good = [row["memory"] for row in memory_rows if row.get("memory")]
        record["memory_process_records"] = memory_rows
        record["memory_samples"] = len(good)
        record["memory_status"] = "ok" if len(good) == args.memory_repeat else "incomplete"
        if good:
            for source, target in (
                ("rss_mb", "host_rss_peak_mb"),
                ("gpu_proc_peak_mb", "gpu_process_peak_mb"),
            ):
                values = [float(item[source]) for item in good if item.get(source) is not None]
                if values:
                    record[f"{target}_mean"] = statistics.mean(values)
                    record[f"{target}_stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return record


def skipped(metadata, function, kind, reason):
    return {
        "dataset": metadata["name"], "function": function, "baseline": "Gunrock",
        "support_class": SUPPORT[function], "status": "skipped",
        "failure_kind": kind, "reason": reason, "validation": "not_run",
        "num_nodes": int(metadata["num_nodes"]), "num_entries": int(metadata["num_entries"]),
        "directed": bool(metadata["directed"]), "timing_process_samples": 0,
    }


def driver(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = shared.gunrock_executable_artifacts(tuple(sorted(set(APP.values()))))
    records = []
    total = len(args.manifests) * len(args.functions)
    progress = 0
    for manifest in args.manifests:
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        metadata["_manifest_path"] = str(manifest.resolve())
        matrix = args.input_dir / f"{metadata['name']}.aligned-weighted.mtx"
        for function in args.functions:
            progress += 1
            final_path = args.out_dir / f"{metadata['name']}_{function}.json"
            if args.resume and final_path.is_file():
                record = json.loads(final_path.read_text(encoding="utf-8"))
                records.append(record)
                print(f"[gunrock-large {progress}/{total}] reused {metadata['name']}/{function}", flush=True)
                continue
            allowed, kind, reason = applicability(metadata, function, matrix)
            executable = shared.find_gunrock_exe(APP[function]) if function in APP else None
            if not allowed:
                record = skipped(metadata, function, kind, reason)
            elif executable is None:
                record = skipped(metadata, function, "missing_executable", f"pinned Gunrock executable {APP[function]} is absent")
            else:
                record = aggregate(args, metadata, function, matrix, executable)
                record["executable"] = artifacts.get(APP[function], {})
            final_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            records.append(record)
            print(f"[gunrock-large {progress}/{total}] {metadata['name']}/{function}: {record['status']} {record.get('failure_kind', '')}", flush=True)
    (args.out_dir / "gunrock_large_matrix.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    rows = [flatten(record) for record in records]
    fields = sorted({key for row in rows for key in row})
    with (args.out_dir / "gunrock_large_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "gunrock_executable_artifacts.json").write_text(
        json.dumps(artifacts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--functions", nargs="+", choices=FUNCTIONS, default=list(FUNCTIONS))
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--eggpu-result-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--memory-repeat", type=int, default=3)
    parser.add_argument("--source-count", type=int, default=8)
    parser.add_argument("--bc-source-count", type=int, default=16)
    parser.add_argument("--pagerank-alpha", type=float, default=0.75)
    parser.add_argument("--pagerank-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--timeout", type=float, default=100.0)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(driver(parse_args()))
