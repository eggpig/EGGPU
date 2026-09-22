#!/usr/bin/env python3
"""Strictly qualify the native SYgraph BFS implementation on normalized CSR inputs."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path

import pandas as pd

from child_process_memory_monitor import ChildProcessMemoryMonitor


def sample_stats(values):
    values = [float(value) for value in values]
    return {
        "mean": statistics.mean(values),
        "best": min(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "samples": values,
        "count": len(values),
    }


def sources(metadata, count):
    recorded = metadata.get("benchmark_sources_zero_based")
    if isinstance(recorded, list) and len(recorded) >= count:
        return [int(value) for value in recorded[:count]]
    n = int(metadata["num_nodes"])
    return sorted({min(n - 1, index * n // max(1, count)) for index in range(count)})


def parse_result(stdout):
    records = []
    for line in stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            records.append(json.loads(line[len("RESULT_JSON ") :]))
    if len(records) != 1:
        raise RuntimeError(f"expected one RESULT_JSON record, found {len(records)}")
    return records[0]


def run_once(args, manifest_path, metadata, measurement, sample_index, expected):
    root = manifest_path.parent
    selected_sources = sources(metadata, args.source_count)
    command = [
        str(args.binary),
        "--offsets", str(root / metadata["offsets_path"]),
        "--indices", str(root / metadata["indices_path"]),
        "--nodes", str(metadata["num_nodes"]),
        "--entries", str(metadata["num_entries"]),
        "--directed", "1" if metadata.get("directed") else "0",
        "--sources", ",".join(str(value) for value in selected_sources),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    monitor = None
    if measurement == "memory":
        monitor = ChildProcessMemoryMonitor(
            process.pid, args.gpu, interval_seconds=args.memory_poll_ms / 1000.0
        ).start()
    try:
        stdout, stderr = process.communicate(timeout=args.load_timeout + args.timeout + 60)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        memory = monitor.stop() if monitor is not None else {}
        return {
            "status": "failed",
            "failure_kind": "timeout",
            "reason": "SYgraph process exceeded graph-load plus function timeout",
            "measurement": measurement,
            "sample_index": sample_index,
            "wall_seconds": time.perf_counter() - started,
            "stdout": stdout,
            "stderr": stderr,
            "memory": memory,
        }
    memory = monitor.stop() if monitor is not None else {}
    wall_seconds = time.perf_counter() - started
    try:
        record = parse_result(stdout)
    except Exception as error:
        return {
            "status": "failed",
            "failure_kind": "execution_error",
            "reason": str(error),
            "measurement": measurement,
            "sample_index": sample_index,
            "returncode": process.returncode,
            "wall_seconds": wall_seconds,
            "stdout": stdout,
            "stderr": stderr,
            "memory": memory,
        }
    record.update(
        {
            "measurement": measurement,
            "sample_index": sample_index,
            "returncode": process.returncode,
            "wall_seconds": wall_seconds,
            "stderr": stderr,
            "memory": memory,
        }
    )
    if process.returncode != 0 or record.get("status") != "ok":
        record.update(
            status="failed",
            failure_kind="execution_error",
            reason=record.get("error") or stderr.strip() or "SYgraph runner failed",
        )
        return record
    if float(record["e2e_seconds"]) > args.timeout:
        record.update(
            status="failed",
            failure_kind="timeout",
            reason=f"BFS E2E exceeded the {args.timeout:g}s per-function limit",
        )
        return record
    expected_sources = expected.get("sources") or selected_sources
    failures = []
    if not expected.get("sha256"):
        failures.append("no aligned reference SHA-256 was supplied")
    if [int(value) for value in record.get("sources", [])] != [int(value) for value in expected_sources]:
        failures.append("source sequence differs from the aligned workload")
    if expected.get("sha256") and record.get("result_sha256") != expected["sha256"]:
        failures.append("distance matrix SHA-256 differs from the aligned EGGPU result")
    expected_shape = expected.get("shape")
    if expected_shape and [int(value) for value in record.get("result_shape", [])] != [int(value) for value in expected_shape]:
        failures.append("distance matrix shape differs from the aligned workload")
    if failures:
        record.update(
            status="failed",
            failure_kind="semantic_mismatch",
            reason="; ".join(failures),
            validation="fail",
        )
    else:
        record.update(validation="pass", failure_kind="", reason="")
    return record


def aggregate_dataset(args, manifest_path, references, raw_dir):
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    name = str(metadata["name"])
    expected = references.get(name, {})
    timing = []
    memory = []
    for measurement, count, target in (
        ("timing", args.repeat, timing),
        ("memory", args.memory_repeat, memory),
    ):
        for sample_index in range(1, count + 1):
            record = run_once(
                args, manifest_path, metadata, measurement, sample_index, expected
            )
            target.append(record)
            (raw_dir / f"{name}_BFS_{measurement}_{sample_index}.json").write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(
                f"[SYgraph] {name}/BFS {measurement} {sample_index}/{count}: "
                f"{record.get('status')} {record.get('failure_kind', '')}",
                flush=True,
            )
            if record.get("status") != "ok":
                break
    timing_failures = [record for record in timing if record.get("status") != "ok"]
    memory_failures = [record for record in memory if record.get("status") != "ok"]
    row = {
        "dataset": name,
        "function": "BFS",
        "baseline": "SYgraph",
        "num_nodes": int(metadata["num_nodes"]),
        "num_entries": int(metadata["num_entries"]),
        "directed": bool(metadata.get("directed")),
        "status": "failed" if timing_failures else "ok",
        "failure_kind": timing_failures[0].get("failure_kind", "") if timing_failures else "",
        "reason": timing_failures[0].get("reason", "") if timing_failures else "",
        "validation": "fail" if timing_failures else "pass",
        "timing_samples": sum(record.get("status") == "ok" for record in timing),
        "memory_samples": sum(record.get("status") == "ok" for record in memory),
        "memory_status": "failed" if memory_failures else "ok",
        "memory_failure_kind": memory_failures[0].get("failure_kind", "") if memory_failures else "",
        "memory_reason": memory_failures[0].get("reason", "") if memory_failures else "",
        "source_count": args.source_count,
        "source_commit": args.source_commit,
        "source_version": "SYgraph upstream HEAD qualification",
    }
    successful_timing = [record for record in timing if record.get("status") == "ok"]
    if len(successful_timing) == args.repeat and not timing_failures:
        for source, prefix in (
            ("build_seconds", "build"),
            ("host_csr_load_seconds", "host_csr_load"),
            ("device_graph_build_seconds", "device_graph_build"),
            ("e2e_seconds", "e2e"),
            ("kernel_seconds", "kernel"),
            ("wall_seconds", "process_wall"),
        ):
            values = [record[source] for record in successful_timing]
            for key, value in sample_stats(values).items():
                if key != "samples":
                    row[f"{prefix}_{key}"] = value
            row[f"{prefix}_samples"] = json.dumps(values)
        row["result_sha256"] = successful_timing[-1]["result_sha256"]
    successful_memory = [record for record in memory if record.get("status") == "ok"]
    if len(successful_memory) == args.memory_repeat and not memory_failures:
        for source, output in (
            ("gpu_proc_peak_delta_mb", "gpu_peak_mb"),
            ("rss_peak_delta_mb", "host_rss_peak_mb"),
        ):
            values = [record.get("memory", {}).get(source) for record in successful_memory]
            values = [value for value in values if value is not None and math.isfinite(float(value))]
            if values:
                stats = sample_stats(values)
                row[f"{output}_mean"] = stats["mean"]
                row[f"{output}_stdev"] = stats["stdev"]
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--reference-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--memory-repeat", type=int, default=3)
    parser.add_argument("--source-count", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=100.0)
    parser.add_argument("--load-timeout", type=float, default=900.0)
    parser.add_argument("--memory-poll-ms", type=float, default=2.0)
    parser.add_argument("--source-commit", default="")
    args = parser.parse_args()

    args.binary = args.binary.resolve()
    if not args.binary.is_file():
        raise SystemExit(f"missing SYgraph runner: {args.binary}")
    references = json.loads(args.reference_json.read_text(encoding="utf-8"))
    output = args.output_dir.resolve()
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    rows = [
        aggregate_dataset(args, manifest.resolve(), references, raw)
        for manifest in args.manifest
    ]
    pd.DataFrame(rows).to_csv(output / "sygraph_bfs.csv", index=False)
    metadata = {
        "baseline": "SYgraph",
        "scope": "native upstream BFS with exact dense distance-matrix validation",
        "binary": str(args.binary),
        "source_commit": args.source_commit,
        "repeat": args.repeat,
        "memory_repeat": args.memory_repeat,
        "timeout_seconds": args.timeout,
        "load_timeout_seconds": args.load_timeout,
        "gpu": args.gpu,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output / 'sygraph_bfs.csv'}")


if __name__ == "__main__":
    main()
