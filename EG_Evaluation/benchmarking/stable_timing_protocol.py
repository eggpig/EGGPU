#!/usr/bin/env python3
"""Shared controls for auditable five-sample EGGPU timing batches.

The submission estimator is the minimum of one complete five-sample batch.
Acceptance is independent of sample standard deviation: SD and CV are always
reported, while the hard guard rejects only catastrophic high or low outliers.
"""

from __future__ import annotations

import copy
import csv
import ctypes
import ctypes.util
import hashlib
import math
import os
import shutil
import statistics
import subprocess
from collections.abc import Iterable, Mapping, MutableMapping
from pathlib import Path


EXPECTED_TIMING_SAMPLES = 5
DEFAULT_MAX_OVER_MEDIAN_LIMIT = 5.0
DEFAULT_MEDIAN_OVER_MIN_LIMIT = 3.0
VARIANCE_POLICY = "catastrophic_outlier_guard"
PAPER_ESTIMATOR = "minimum_of_five"
FORMAL_BATCH_SELECTION_POLICY = (
    "unique_complete_batch_per_key_no_cross_batch_selection"
)
GATE_CALIBRATION = {
    "version": "2026-07-29-youtube-constraint-dual-batch",
    "max_over_median_limit": DEFAULT_MAX_OVER_MEDIAN_LIMIT,
    "median_over_min_limit": DEFAULT_MEDIAN_OVER_MIN_LIMIT,
    "calibration_cell": "com-youtube/Constraint/e2e",
    "independent_batch_median_over_min": [
        2.2574233350473074,
        2.261533125067979,
    ],
    "calibration_kernel_range_seconds": [
        0.119654716,
        0.129456665,
    ],
    "calibration_evidence": [
        {
            "batch": "formal_original",
            "result_source": (
                "benchmarking/results/"
                "final_v10_r1_uniform_gpu0_20260729"
            ),
            "results_samples_sha256": (
                "e268318a32a5ad974c769252b8e08f8254c12e3d4ad5bacb85f87cacca4a1994"
            ),
            "e2e_raw5_seconds": [
                0.141710006,
                0.127117831,
                0.286958758,
                0.5971683,
                0.338744991,
            ],
            "median_over_min": 2.2574233350473074,
            "result_selection_usage": (
                "unique_original_complete_batch"
            ),
        },
        {
            "batch": "independent_calibration_rerun",
            "result_source": (
                "benchmarking/results/"
                "final_v10_r1_stability_rerun_youtube_constraint_gpu1_20260729"
            ),
            "results_samples_sha256": (
                "b4edba77123c99d88770335c87cd9ee86357c0d117f32d5f53a91458540d6eda"
            ),
            "e2e_raw5_seconds": [
                0.431060884,
                0.31946284,
                0.132862233,
                0.134928804,
                0.300472341,
            ],
            "kernel_raw5_seconds": [
                0.122204544,
                0.129456665,
                0.119654716,
                0.121487137,
                0.119908127,
            ],
            "median_over_min": 2.261533125067979,
            "result_selection_usage": (
                "threshold_calibration_only_excluded_from_formal_results"
            ),
        },
    ],
    "interpretation": (
        "Two independent complete batches reproduce an approximately 2.26x "
        "E2E median/min split while the device kernel remains in a narrow "
        "approximately 120--129 ms band. This is a repeatable host-return/"
        "materialization bimodality, not a pathological timing outlier."
    ),
    "selection_usage": "threshold_calibration_only_not_result_selection",
}

CONTROLLED_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "OMP_PROC_BIND": "TRUE",
    "PYTHONHASHSEED": "0",
    "MALLOC_ARENA_MAX": "2",
}


class StableTimingProtocolError(ValueError):
    """The requested controlled timing contract is incomplete or inconsistent."""


def gate_calibration_payload() -> dict[str, object]:
    """Return an isolated copy of the one publication-gate calibration."""

    return copy.deepcopy(GATE_CALIBRATION)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _calibration_raw5(
    samples_path: Path,
    *,
    dataset: str,
    function: str,
    metric: str,
) -> list[float]:
    by_index: dict[int, float] = {}
    with samples_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row.get("dataset") != dataset
                or row.get("function") != function
                or row.get("baseline") != "EGGPU"
                or row.get("metric") != metric
                or row.get("status") != "ok"
            ):
                continue
            try:
                sample_index = int(row["sample_index"])
                sample_count = int(row["sample_count"])
                seconds = float(row["seconds"])
            except (KeyError, TypeError, ValueError) as exc:
                raise StableTimingProtocolError(
                    f"invalid calibration row in {samples_path}: {exc}"
                ) from exc
            if sample_count != EXPECTED_TIMING_SAMPLES:
                raise StableTimingProtocolError(
                    f"calibration row in {samples_path} declares "
                    f"sample_count={sample_count}, expected "
                    f"{EXPECTED_TIMING_SAMPLES}"
                )
            if (
                sample_index < 1
                or sample_index > EXPECTED_TIMING_SAMPLES
                or sample_index in by_index
                or not math.isfinite(seconds)
                or seconds < 0
            ):
                raise StableTimingProtocolError(
                    f"invalid or duplicate calibration sample_index="
                    f"{sample_index} for {dataset}/{function}/{metric} "
                    f"in {samples_path}"
                )
            by_index[sample_index] = seconds
    expected_indices = set(range(1, EXPECTED_TIMING_SAMPLES + 1))
    if set(by_index) != expected_indices:
        raise StableTimingProtocolError(
            f"calibration evidence {samples_path} has sample indices "
            f"{sorted(by_index)}, expected {sorted(expected_indices)} for "
            f"{dataset}/{function}/{metric}"
        )
    return [by_index[index] for index in sorted(by_index)]


def _require_same_numbers(
    observed: object,
    expected: object,
    *,
    label: str,
) -> None:
    if not isinstance(observed, (list, tuple)) or not isinstance(
        expected, (list, tuple)
    ):
        raise StableTimingProtocolError(
            f"{label} must be a numerical sequence"
        )
    if len(observed) != len(expected):
        raise StableTimingProtocolError(
            f"{label} length {len(observed)} != expected {len(expected)}"
        )
    try:
        pairs = zip(
            (float(value) for value in observed),
            (float(value) for value in expected),
            strict=True,
        )
        matches = all(
            math.isfinite(left)
            and math.isfinite(right)
            and math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
            for left, right in pairs
        )
    except (TypeError, ValueError):
        matches = False
    if not matches:
        raise StableTimingProtocolError(
            f"{label} does not match the hash-bound calibration evidence"
        )


def validate_gate_calibration(
    value: Mapping[str, object],
    *,
    repo_root: Path | None = None,
) -> dict[str, object]:
    """Fail closed on calibration drift and verify every evidence binding.

    ``repo_root`` is the ``EG_Evaluation`` directory.  When omitted it is
    resolved from this module.  The returned dictionary is a deep copy, so
    consumers cannot mutate the shared definition through nested values.
    """

    if not isinstance(value, Mapping):
        raise StableTimingProtocolError(
            "gate_calibration must be a mapping"
        )
    candidate = copy.deepcopy(dict(value))
    canonical = gate_calibration_payload()
    if candidate != canonical:
        differing = sorted(
            key
            for key in set(candidate) | set(canonical)
            if candidate.get(key) != canonical.get(key)
        )
        raise StableTimingProtocolError(
            "gate_calibration does not match the shared canonical "
            f"definition; differing keys={differing}"
        )

    if (
        float(canonical["max_over_median_limit"])
        != DEFAULT_MAX_OVER_MEDIAN_LIMIT
        or float(canonical["median_over_min_limit"])
        != DEFAULT_MEDIAN_OVER_MIN_LIMIT
    ):
        raise StableTimingProtocolError(
            "canonical calibration limits drift from protocol defaults"
        )

    cell_parts = str(canonical["calibration_cell"]).split("/")
    if len(cell_parts) != 3:
        raise StableTimingProtocolError(
            "calibration_cell must be dataset/function/metric"
        )
    dataset, function, e2e_metric = cell_parts
    if e2e_metric != "e2e":
        raise StableTimingProtocolError(
            "publication-gate calibration must bind E2E samples"
        )

    root = (
        Path(repo_root).expanduser().resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[1]
    )
    evidence = canonical.get("calibration_evidence")
    if not isinstance(evidence, list) or len(evidence) != 2:
        raise StableTimingProtocolError(
            "calibration_evidence must contain exactly two independent batches"
        )

    observed_ratios: list[float] = []
    observed_kernel_values: list[float] = []
    observed_batches: set[str] = set()
    for index, item in enumerate(evidence):
        if not isinstance(item, Mapping):
            raise StableTimingProtocolError(
                f"calibration_evidence[{index}] must be a mapping"
            )
        batch = str(item.get("batch", ""))
        if not batch or batch in observed_batches:
            raise StableTimingProtocolError(
                "calibration evidence batch labels must be nonempty and unique"
            )
        observed_batches.add(batch)

        result_source = Path(str(item.get("result_source", "")))
        if result_source.is_absolute() or ".." in result_source.parts:
            raise StableTimingProtocolError(
                f"unsafe calibration result_source={result_source}"
            )
        samples_path = (root / result_source / "results_samples.csv").resolve()
        try:
            samples_path.relative_to(root)
        except ValueError as exc:
            raise StableTimingProtocolError(
                f"calibration evidence escapes repo_root: {samples_path}"
            ) from exc
        if not samples_path.is_file():
            raise StableTimingProtocolError(
                f"calibration evidence is missing: {samples_path}"
            )

        expected_sha = str(item.get("results_samples_sha256", ""))
        actual_sha = _sha256_file(samples_path)
        if actual_sha != expected_sha:
            raise StableTimingProtocolError(
                f"calibration evidence SHA mismatch for {samples_path}: "
                f"{actual_sha} != {expected_sha}"
            )

        actual_e2e = _calibration_raw5(
            samples_path,
            dataset=dataset,
            function=function,
            metric=e2e_metric,
        )
        _require_same_numbers(
            actual_e2e,
            item.get("e2e_raw5_seconds"),
            label=f"calibration_evidence[{index}].e2e_raw5_seconds",
        )
        minimum = min(actual_e2e)
        if minimum <= 0:
            raise StableTimingProtocolError(
                f"calibration evidence has nonpositive E2E minimum: {minimum}"
            )
        ratio = statistics.median(actual_e2e) / minimum
        if not math.isclose(
            ratio,
            float(item.get("median_over_min", math.nan)),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise StableTimingProtocolError(
                f"calibration_evidence[{index}].median_over_min is not "
                "reproducible from raw5"
            )
        observed_ratios.append(ratio)

        expected_kernel = item.get("kernel_raw5_seconds")
        if expected_kernel is not None:
            actual_kernel = _calibration_raw5(
                samples_path,
                dataset=dataset,
                function=function,
                metric="kernel",
            )
            _require_same_numbers(
                actual_kernel,
                expected_kernel,
                label=f"calibration_evidence[{index}].kernel_raw5_seconds",
            )
            observed_kernel_values.extend(actual_kernel)

    _require_same_numbers(
        observed_ratios,
        canonical["independent_batch_median_over_min"],
        label="independent_batch_median_over_min",
    )
    if not observed_kernel_values:
        raise StableTimingProtocolError(
            "calibration evidence must include a kernel raw5 diagnostic"
        )
    _require_same_numbers(
        [min(observed_kernel_values), max(observed_kernel_values)],
        canonical["calibration_kernel_range_seconds"],
        label="calibration_kernel_range_seconds",
    )
    return canonical


class _NumaBitmask(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_ulong),
        ("maskp", ctypes.POINTER(ctypes.c_ulong)),
    ]


def _load_libnuma():
    library = ctypes.util.find_library("numa")
    if not library:
        raise StableTimingProtocolError("libnuma is not installed")
    try:
        libnuma = ctypes.CDLL(library, use_errno=True)
    except OSError as exc:
        raise StableTimingProtocolError(
            f"cannot load libnuma: {exc}"
        ) from exc
    mask_pointer = ctypes.POINTER(_NumaBitmask)
    libnuma.numa_available.restype = ctypes.c_int
    libnuma.numa_max_node.restype = ctypes.c_int
    libnuma.numa_allocate_nodemask.restype = mask_pointer
    libnuma.numa_get_membind.restype = mask_pointer
    libnuma.numa_bitmask_clearall.argtypes = [mask_pointer]
    libnuma.numa_bitmask_setbit.argtypes = [mask_pointer, ctypes.c_uint]
    libnuma.numa_bitmask_isbitset.argtypes = [mask_pointer, ctypes.c_uint]
    libnuma.numa_bitmask_isbitset.restype = ctypes.c_int
    libnuma.numa_set_membind.argtypes = [mask_pointer]
    libnuma.numa_set_strict.argtypes = [ctypes.c_int]
    libnuma.numa_bitmask_free.argtypes = [mask_pointer]
    if libnuma.numa_available() < 0:
        raise StableTimingProtocolError("libnuma reports no NUMA support")
    return libnuma


def collect_libnuma_membind() -> dict[str, object]:
    """Read the current process memory-node policy without the numactl CLI."""

    try:
        libnuma = _load_libnuma()
        mask = libnuma.numa_get_membind()
        if not mask:
            raise StableTimingProtocolError(
                "numa_get_membind returned a null mask"
            )
        try:
            maximum = int(libnuma.numa_max_node())
            nodes = [
                node
                for node in range(maximum + 1)
                if libnuma.numa_bitmask_isbitset(mask, node)
            ]
        finally:
            libnuma.numa_bitmask_free(mask)
        return {
            "available": True,
            "membind_ids": nodes,
            "membind_list": format_linux_list(nodes),
            "error": "",
        }
    except (OSError, StableTimingProtocolError) as exc:
        return {
            "available": False,
            "membind_ids": [],
            "membind_list": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def bind_memory_nodes(nodes: str | Iterable[int]) -> dict[str, object]:
    """Bind future allocations to an exact NUMA-node set through libnuma."""

    requested = parse_linux_list(nodes)
    if not requested:
        raise StableTimingProtocolError(
            "at least one NUMA node is required"
        )
    libnuma = _load_libnuma()
    maximum = int(libnuma.numa_max_node())
    if requested[-1] > maximum:
        raise StableTimingProtocolError(
            f"NUMA node {requested[-1]} exceeds host maximum {maximum}"
        )
    mask = libnuma.numa_allocate_nodemask()
    if not mask:
        raise StableTimingProtocolError(
            "numa_allocate_nodemask returned a null mask"
        )
    try:
        libnuma.numa_bitmask_clearall(mask)
        for node in requested:
            libnuma.numa_bitmask_setbit(mask, node)
        libnuma.numa_set_strict(1)
        libnuma.numa_set_membind(mask)
    finally:
        libnuma.numa_bitmask_free(mask)
    observed = collect_libnuma_membind()
    if not observed["available"] or observed["membind_ids"] != requested:
        raise StableTimingProtocolError(
            "libnuma memory binding verification failed: "
            f"requested={format_linux_list(requested)} observed={observed}"
        )
    return observed


def parse_linux_list(value: str | Iterable[int]) -> list[int]:
    """Parse Linux CPU/node list syntax such as ``0-3,8,10-11``."""

    if not isinstance(value, str):
        result = sorted({int(item) for item in value})
        if any(item < 0 for item in result):
            raise StableTimingProtocolError("CPU/node ids must be nonnegative")
        return result
    result: set[int] = set()
    text = value.strip()
    if not text:
        return []
    for token in text.split(","):
        token = token.strip()
        if not token:
            raise StableTimingProtocolError(f"invalid Linux list: {value!r}")
        if "-" not in token:
            try:
                item = int(token)
            except ValueError as exc:
                raise StableTimingProtocolError(
                    f"invalid Linux list token: {token!r}"
                ) from exc
            if item < 0:
                raise StableTimingProtocolError("CPU/node ids must be nonnegative")
            result.add(item)
            continue
        bounds = token.split("-", 1)
        try:
            start, end = (int(item) for item in bounds)
        except ValueError as exc:
            raise StableTimingProtocolError(
                f"invalid Linux list range: {token!r}"
            ) from exc
        if start < 0 or end < start:
            raise StableTimingProtocolError(
                f"invalid Linux list range: {token!r}"
            )
        result.update(range(start, end + 1))
    return sorted(result)


def format_linux_list(values: Iterable[int]) -> str:
    """Return a canonical compact Linux list."""

    ordered = parse_linux_list(values)
    if not ordered:
        return ""
    ranges: list[str] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def apply_controlled_thread_environment(
    environment: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Force the thread/hash/allocator controls inherited by worker processes."""

    target = os.environ if environment is None else environment
    for key, value in CONTROLLED_THREAD_ENVIRONMENT.items():
        target[key] = value
    return {
        key: str(target.get(key, ""))
        for key in CONTROLLED_THREAD_ENVIRONMENT
    }


def _proc_status_lists() -> dict[str, str]:
    fields = {"Cpus_allowed_list": "", "Mems_allowed_list": ""}
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                name, separator, value = line.partition(":")
                if separator and name in fields:
                    fields[name] = value.strip()
    except OSError:
        pass
    return fields


def _parse_numactl_show(output: str) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for line in output.splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue
        normalized = key.strip().lower().replace(" ", "_")
        text = value.strip()
        parsed[normalized] = text
        if normalized in {"physcpubind", "cpubind", "nodebind", "membind"}:
            try:
                parsed[f"{normalized}_ids"] = parse_linux_list(
                    text.replace(" ", ",")
                )
            except StableTimingProtocolError:
                parsed[f"{normalized}_ids"] = []
    return parsed


def collect_numactl_policy(
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Capture the inherited NUMA policy outside any measured call."""

    executable = shutil.which("numactl")
    if not executable:
        return {
            "available": False,
            "command": [],
            "returncode": None,
            "stdout": "",
            "stderr": "numactl not found",
            "parsed": {},
        }
    command = [executable, "--show"]
    try:
        completed = subprocess.run(
            command,
            env=dict(os.environ if environment is None else environment),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "available": True,
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "parsed": {},
        }
    return {
        "available": True,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "parsed": (
            _parse_numactl_show(completed.stdout)
            if completed.returncode == 0
            else {}
        ),
    }


def collect_execution_placement(
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Record taskset-visible CPU affinity and numactl-visible memory policy."""

    source = os.environ if environment is None else environment
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = []
    proc_lists = _proc_status_lists()
    return {
        "pid": os.getpid(),
        "cpu_affinity_ids": affinity,
        "cpu_affinity_list": format_linux_list(affinity),
        "cpu_affinity_count": len(affinity),
        "proc_cpus_allowed_list": proc_lists["Cpus_allowed_list"],
        "proc_mems_allowed_list": proc_lists["Mems_allowed_list"],
        "numactl": collect_numactl_policy(source),
        "libnuma": collect_libnuma_membind(),
        "controlled_thread_environment": {
            key: str(source.get(key, ""))
            for key in CONTROLLED_THREAD_ENVIRONMENT
        },
    }


def validate_controlled_execution(
    placement: Mapping[str, object],
    *,
    expected_cpu_affinity: str = "",
    expected_numa_nodes: str = "",
) -> None:
    """Reject a launcher whose observed placement differs from its declaration."""

    observed_threads = placement.get("controlled_thread_environment") or {}
    mismatched_threads = {
        key: {
            "expected": value,
            "observed": str(observed_threads.get(key, "")),
        }
        for key, value in CONTROLLED_THREAD_ENVIRONMENT.items()
        if str(observed_threads.get(key, "")) != value
    }
    if mismatched_threads:
        raise StableTimingProtocolError(
            f"controlled thread environment mismatch: {mismatched_threads}"
        )

    if expected_cpu_affinity:
        expected = parse_linux_list(expected_cpu_affinity)
        observed = [int(item) for item in placement.get("cpu_affinity_ids", [])]
        if observed != expected:
            raise StableTimingProtocolError(
                "CPU affinity does not match external taskset declaration: "
                f"expected={format_linux_list(expected)} "
                f"observed={format_linux_list(observed)}"
            )

    if expected_numa_nodes:
        expected = parse_linux_list(expected_numa_nodes)
        numactl = placement.get("numactl") or {}
        parsed = numactl.get("parsed") or {}
        libnuma = placement.get("libnuma") or {}
        if numactl.get("available") and numactl.get("returncode") == 0:
            observed = [
                int(item) for item in parsed.get("membind_ids", [])
            ]
            observation_source = "numactl --show"
        elif libnuma.get("available"):
            observed = [
                int(item) for item in libnuma.get("membind_ids", [])
            ]
            observation_source = "libnuma numa_get_membind"
        else:
            raise StableTimingProtocolError(
                "cannot audit the requested NUMA binding because "
                "neither `numactl --show` nor libnuma policy inspection "
                "succeeded"
            )
        if observed != expected:
            raise StableTimingProtocolError(
                "NUMA memory binding does not match the launcher declaration "
                f"({observation_source}): "
                f"expected={format_linux_list(expected)} "
                f"observed={format_linux_list(observed)}"
            )


def summarize_five_samples(
    values: Iterable[float],
    *,
    max_over_median_limit: float = DEFAULT_MAX_OVER_MEDIAN_LIMIT,
    median_over_min_limit: float = DEFAULT_MEDIAN_OVER_MIN_LIMIT,
) -> dict[str, object]:
    """Summarize and accept/reject exactly five nonnegative timing samples."""

    samples = [float(value) for value in values]
    if len(samples) != EXPECTED_TIMING_SAMPLES:
        raise StableTimingProtocolError(
            f"expected exactly {EXPECTED_TIMING_SAMPLES} timing samples, "
            f"observed {len(samples)}"
        )
    if any(not math.isfinite(value) or value < 0 for value in samples):
        raise StableTimingProtocolError(
            "timing samples must be finite and nonnegative"
        )
    if max_over_median_limit < 1:
        raise StableTimingProtocolError("max/median limit must be at least one")
    if median_over_min_limit < 1:
        raise StableTimingProtocolError("median/min limit must be at least one")

    minimum = min(samples)
    maximum = max(samples)
    median = statistics.median(samples)
    mean = statistics.mean(samples)
    sample_std = statistics.stdev(samples)
    cv = sample_std / mean if mean > 0 else 0.0
    all_zero = maximum == 0
    nonzero_with_zero_minimum = minimum == 0 and not all_zero
    if all_zero:
        max_over_median = 1.0
        median_over_min = 1.0
    elif nonzero_with_zero_minimum:
        max_over_median = (
            maximum / median if median > 0 else math.inf
        )
        median_over_min = math.inf
    else:
        max_over_median = maximum / median
        median_over_min = median / minimum

    failures = []
    if nonzero_with_zero_minimum:
        failures.append("zero_minimum_in_nonzero_batch")
    if max_over_median > max_over_median_limit:
        failures.append("max_over_median_exceeds_limit")
    if median_over_min > median_over_min_limit:
        failures.append("median_over_min_exceeds_limit")
    accepted = not failures
    return {
        "variance_policy": VARIANCE_POLICY,
        "paper_estimator": PAPER_ESTIMATOR,
        "expected_sample_count": EXPECTED_TIMING_SAMPLES,
        "sample_count": len(samples),
        "raw5_seconds": samples,
        "minimum_seconds": minimum,
        "arithmetic_mean_seconds": mean,
        "median_seconds": median,
        "maximum_seconds": maximum,
        "sample_std_seconds": sample_std,
        "coefficient_of_variation": cv,
        "max_over_min": (
            maximum / minimum
            if minimum > 0
            else (1.0 if all_zero else math.inf)
        ),
        "max_over_median": max_over_median,
        "median_over_min": median_over_min,
        "max_over_median_limit": float(max_over_median_limit),
        "median_over_min_limit": float(median_over_min_limit),
        "stability_status": "pass" if accepted else "fail",
        "batch_acceptance_status": (
            "accepted_complete_batch"
            if accepted
            else "rejected_entire_batch"
        ),
        "submission_seconds": minimum if accepted else None,
        "failure_reasons": failures,
    }


def controlled_protocol_metadata(
    *,
    placement: Mapping[str, object],
    expected_cpu_affinity: str = "",
    expected_numa_nodes: str = "",
    max_over_median_limit: float = DEFAULT_MAX_OVER_MEDIAN_LIMIT,
    median_over_min_limit: float = DEFAULT_MEDIAN_OVER_MIN_LIMIT,
) -> dict[str, object]:
    """Build the immutable protocol block written beside a benchmark batch."""

    return {
        "enabled": True,
        "variance_policy": VARIANCE_POLICY,
        "paper_estimator": PAPER_ESTIMATOR,
        "expected_sample_count": EXPECTED_TIMING_SAMPLES,
        "complete_batch_required": True,
        "failed_batch_policy": "reject_entire_batch_do_not_select_across_batches",
        "formal_batch_selection_policy": FORMAL_BATCH_SELECTION_POLICY,
        "sample_std_and_cv_role": "reported_diagnostics_not_acceptance_gate",
        "max_over_median_limit": float(max_over_median_limit),
        "median_over_min_limit": float(median_over_min_limit),
        "gate_calibration": gate_calibration_payload(),
        "expected_cpu_affinity": (
            format_linux_list(parse_linux_list(expected_cpu_affinity))
            if expected_cpu_affinity
            else ""
        ),
        "expected_numa_nodes": (
            format_linux_list(parse_linux_list(expected_numa_nodes))
            if expected_numa_nodes
            else ""
        ),
        "execution_placement": dict(placement),
    }
