#!/usr/bin/env python3
"""Validate a PageRank vector against the defining fixed-point equation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np


DEFAULT_MEAN_RESIDUAL_TOLERANCE = 1.0e-6


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_path(manifest_path: Path, manifest: dict, key: str) -> Path:
    value = manifest.get(key)
    if not value:
        value = manifest.get("csr_artifacts", {}).get(key.removesuffix("_path"), {}).get("path")
    if not value:
        raise ValueError(f"CSR manifest has no {key}")
    return (manifest_path.parent / value).resolve()


def validate_pagerank_fixed_point(
    result_path: Path,
    manifest_path: Path,
    *,
    alpha: float,
    mean_residual_tolerance: float = DEFAULT_MEAN_RESIDUAL_TOLERANCE,
    chunk_entries: int = 8_000_000,
) -> dict:
    """Return correctness evidence without comparing two iterative solvers.

    The CSR is processed in bounded chunks.  This keeps validation outside the
    measured process and avoids materializing one source id per adjacency entry.
    """

    result_path = Path(result_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    num_nodes = int(manifest["num_nodes"])
    num_entries = int(manifest["num_entries"])
    offsets_path = _artifact_path(manifest_path, manifest, "offsets_path")
    indices_path = _artifact_path(manifest_path, manifest, "indices_path")

    offsets = np.memmap(offsets_path, dtype=np.int32, mode="r")
    indices = np.memmap(indices_path, dtype=np.int32, mode="r")
    raw_values = np.memmap(result_path, dtype=np.float32, mode="r")
    if offsets.size != num_nodes + 1:
        raise ValueError(f"offset count {offsets.size} != {num_nodes + 1}")
    if indices.size != num_entries:
        raise ValueError(f"index count {indices.size} != {num_entries}")
    if raw_values.size != num_nodes:
        raise ValueError(f"PageRank result count {raw_values.size} != {num_nodes}")

    values = np.asarray(raw_values, dtype=np.float64)
    finite = bool(np.isfinite(values).all())
    minimum = float(values.min()) if values.size else float("nan")
    score_sum = float(values.sum(dtype=np.float64))
    degrees = np.diff(offsets).astype(np.int64, copy=False)
    contribution = np.zeros(num_nodes, dtype=np.float64)

    row_start = 0
    chunks = 0
    while row_start < num_nodes:
        edge_start = int(offsets[row_start])
        edge_target = min(num_entries, edge_start + max(1, int(chunk_entries)))
        row_end = int(np.searchsorted(offsets, edge_target, side="right") - 1)
        row_end = min(num_nodes, max(row_start + 1, row_end))
        edge_end = int(offsets[row_end])
        source_degrees = degrees[row_start:row_end]
        nonzero = source_degrees > 0
        if nonzero.any():
            source_weights = values[row_start:row_end][nonzero] / source_degrees[nonzero]
            edge_weights = np.repeat(source_weights, source_degrees[nonzero])
            destinations = np.asarray(indices[edge_start:edge_end], dtype=np.int64)
            if edge_weights.size != destinations.size:
                raise ValueError(
                    f"CSR chunk mismatch: {edge_weights.size} weights vs "
                    f"{destinations.size} destinations"
                )
            contribution += np.bincount(
                destinations,
                weights=edge_weights,
                minlength=num_nodes,
            )
        row_start = row_end
        chunks += 1

    dangling_mass = float(values[degrees == 0].sum(dtype=np.float64))
    expected = (
        float(alpha) * contribution
        + (float(alpha) * dangling_mass + (1.0 - float(alpha))) / num_nodes
    )
    difference = np.abs(values - expected)
    mean_residual = float(difference.mean(dtype=np.float64))
    max_residual = float(difference.max(initial=0.0))
    sum_ok = math.isclose(score_sum, 1.0, rel_tol=1.0e-5, abs_tol=1.0e-6)
    passed = bool(
        finite
        and minimum >= -1.0e-12
        and sum_ok
        and mean_residual <= float(mean_residual_tolerance)
    )
    return {
        "status": "pass" if passed else "fail",
        "criterion": "PageRank fixed-point equation",
        "alpha": float(alpha),
        "mean_residual": mean_residual,
        "max_residual": max_residual,
        "mean_residual_tolerance": float(mean_residual_tolerance),
        "score_sum": score_sum,
        "score_sum_valid": sum_ok,
        "minimum_score": minimum,
        "all_finite": finite,
        "num_nodes": num_nodes,
        "num_entries": num_entries,
        "chunks": chunks,
        "result_path": str(result_path),
        "result_bytes": result_path.stat().st_size,
        "result_sha256": _sha256_file(result_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "measurement_window": "separate_unmeasured_validation_process",
    }


def _alpha(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("alpha must be a number") from error
    if not math.isfinite(parsed) or not 0.0 <= parsed < 1.0:
        raise argparse.ArgumentTypeError(
            "alpha must be finite and satisfy 0 <= alpha < 1"
        )
    return parsed


def _positive_tolerance(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("tolerance must be a number") from error
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError(
            "tolerance must be finite and positive"
        )
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Qualify a float32 PageRank vector with the fixed-point residual "
            "criterion outside the benchmark timing window."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Normalized CSR manifest JSON.",
    )
    parser.add_argument(
        "--result",
        type=Path,
        required=True,
        help="Raw float32 PageRank vector in contiguous node order.",
    )
    parser.add_argument(
        "--alpha",
        type=_alpha,
        required=True,
        help="PageRank damping factor; Figure 1 uses 0.75.",
    )
    parser.add_argument(
        "--tolerance",
        type=_positive_tolerance,
        required=True,
        help=(
            "Maximum accepted mean absolute fixed-point residual "
            f"(the study default is {DEFAULT_MEAN_RESIDUAL_TOLERANCE:g})."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSON evidence file, written atomically.",
    )
    return parser.parse_args()


def _atomic_write_json(path: Path, payload: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(
                payload,
                handle,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    result_path = args.result.resolve()
    output_path = args.output.resolve()
    if output_path in {manifest_path, result_path}:
        raise ValueError("--output must not overwrite --manifest or --result")

    evidence = validate_pagerank_fixed_point(
        result_path,
        manifest_path,
        alpha=args.alpha,
        mean_residual_tolerance=args.tolerance,
    )
    evidence["evidence_path"] = str(output_path)
    _atomic_write_json(output_path, evidence)
    print(json.dumps(evidence, ensure_ascii=False, sort_keys=True))
    return 0 if evidence["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
