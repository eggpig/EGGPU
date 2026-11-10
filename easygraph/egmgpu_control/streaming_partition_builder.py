"""Project-local, two-pass builders for range-partitioned CSR manifests.

The builders never construct a global int32 CSR.  They count degrees and halo
candidates in project-local staging files, external-sort halo IDs, and fill
each partition's compact CSR during a second source scan.  The complete output
directory is atomically renamed into place only after manifest validation.
The optional incoming_adjacency policy adds an indices scan and O(n) uint64
degree counters to choose owner ranges for destination-owned pull algorithms.
"""

from __future__ import annotations

import bisect
import functools
import hashlib
import heapq
import itertools
import json
import math
import mmap
import os
import shutil
import struct
import sys
import uuid

from array import array

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Iterator
from typing import Optional
from typing import Sequence
from typing import Tuple

from .partition_manifest import ArrayFile
from .partition_manifest import INT32_MAX
from .partition_manifest import ManifestPartitionMetadata
from .partition_manifest import ManifestPathError
from .partition_manifest import PartitionedCSRManifest
from .partition_manifest import UINT64_MAX
from .partition_manifest import _hash_file
from .partition_manifest import _manifest_document
from .partition_manifest import _topology_signature
from .partition_manifest import _view_signature
from .partition_manifest import _weight_signature
from .partition_manifest import load_partition_manifest
from .partition_manifest import write_array_stream
from .state import MAX_LOCAL_ADJACENCY_ENTRIES


_U64 = struct.Struct("<Q")
_I32 = struct.Struct("<i")
_F32 = struct.Struct("<f")
_F64 = struct.Struct("<d")
_HALO_SLOT = struct.Struct("<Qi")
_DEFAULT_SORT_CHUNK_VALUES = 1 << 20
_DEFAULT_MERGE_FAN_IN = 64


class PartitionBuildError(RuntimeError):
    """A source graph cannot be converted under manifest-v1 invariants."""


@dataclass(frozen=True)
class PartitionBuildReport:
    manifest_path: Path
    source_sha256: str
    vertex_count: int
    edge_count: int
    adjacency_count: int
    partition_count: int
    weighted: bool


@dataclass(frozen=True)
class _ScanResult:
    edge_count: int
    adjacency_count: int
    source_sha256: str
    offsets_sha256: Optional[str] = None


@dataclass(frozen=True)
class _BalancedRangePlan:
    ranges: Tuple["_Range", ...]
    offsets_sha256: Optional[str]


@dataclass(frozen=True)
class _Range:
    partition_id: int
    device_id: int
    begin: int
    end: int

    @property
    def count(self) -> int:
        return self.end - self.begin


class _Source:
    weighted: bool
    weight_dtype: Optional[str]

    def scan(self, consume: Callable[[int, int, Optional[float]], None]) -> _ScanResult:
        raise NotImplementedError


class _EdgeListSource(_Source):
    def __init__(
        self,
        path: Path,
        *,
        vertex_count: int,
        directed: bool,
        weighted: bool,
        weight_dtype: str,
        delimiter: Optional[str],
        comment_prefix: str,
    ) -> None:
        self.path = path.resolve(strict=True)
        if not self.path.is_file():
            raise PartitionBuildError(f"edge-list source is not a file: {self.path}")
        self.vertex_count = vertex_count
        self.directed = directed
        self.weighted = weighted
        self.weight_dtype = weight_dtype if weighted else None
        self.delimiter = delimiter
        self.comment_prefix = comment_prefix

    def scan(self, consume: Callable[[int, int, Optional[float]], None]) -> _ScanResult:
        digest = hashlib.sha256()
        edge_count = 0
        adjacency_count = 0
        with self.path.open("rb") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                digest.update(raw_line)
                try:
                    line = raw_line.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise PartitionBuildError(
                        f"edge-list line {line_number} is not UTF-8"
                    ) from exc
                stripped = line.strip()
                if not stripped or stripped.startswith(self.comment_prefix):
                    continue
                fields = (
                    stripped.split()
                    if self.delimiter is None
                    else [field.strip() for field in stripped.split(self.delimiter)]
                )
                expected = 3 if self.weighted else 2
                if len(fields) != expected or any(field == "" for field in fields):
                    raise PartitionBuildError(
                        f"edge-list line {line_number} must contain exactly "
                        f"{expected} fields"
                    )
                try:
                    source = int(fields[0], 10)
                    target = int(fields[1], 10)
                except ValueError as exc:
                    raise PartitionBuildError(
                        f"edge-list line {line_number} has a non-integer vertex ID"
                    ) from exc
                if not (0 <= source < self.vertex_count):
                    raise PartitionBuildError(
                        f"edge-list line {line_number} source {source} is outside "
                        f"[0, {self.vertex_count})"
                    )
                if not (0 <= target < self.vertex_count):
                    raise PartitionBuildError(
                        f"edge-list line {line_number} target {target} is outside "
                        f"[0, {self.vertex_count})"
                    )
                if self.weighted:
                    try:
                        weight = float(fields[2])
                    except ValueError as exc:
                        raise PartitionBuildError(
                            f"edge-list line {line_number} has an invalid weight"
                        ) from exc
                    if not math.isfinite(weight):
                        raise PartitionBuildError(
                            f"edge-list line {line_number} weight must be finite"
                        )
                else:
                    weight = None
                edge_count = _checked_u64_add(edge_count, 1, "logical edge count")
                consume(source, target, weight)
                adjacency_count = _checked_u64_add(
                    adjacency_count, 1, "adjacency count"
                )
                if not self.directed and source != target:
                    consume(target, source, weight)
                    adjacency_count = _checked_u64_add(
                        adjacency_count, 1, "adjacency count"
                    )
        return _ScanResult(edge_count, adjacency_count, digest.hexdigest())


class _HostCSRSource(_Source):
    def __init__(
        self,
        offsets_path: Path,
        indices_path: Path,
        *,
        vertex_count: int,
        edge_count: int,
        weights_path: Optional[Path],
        weight_dtype: str,
        integer_dtype: str = "uint64",
    ) -> None:
        if integer_dtype == "uint64":
            self._integer_struct = _U64
        elif integer_dtype == "int32":
            self._integer_struct = _I32
        else:
            raise ValueError("host CSR integer_dtype must be uint64 or int32")
        self.integer_dtype = integer_dtype
        self.offsets_path = offsets_path.resolve(strict=True)
        self.indices_path = indices_path.resolve(strict=True)
        self.weights_path = (
            None if weights_path is None else weights_path.resolve(strict=True)
        )
        for name, path in (
            ("offsets", self.offsets_path),
            ("indices", self.indices_path),
        ):
            if not path.is_file():
                raise PartitionBuildError(f"host CSR {name} is not a file: {path}")
        if self.offsets_path.stat().st_size != (
            vertex_count + 1
        ) * self._integer_struct.size:
            raise PartitionBuildError(
                "host CSR offsets must contain vertex_count+1 {} values".format(
                    integer_dtype
                )
            )
        if self.indices_path.stat().st_size % self._integer_struct.size:
            raise PartitionBuildError(
                "host CSR indices are not {}-aligned".format(integer_dtype)
            )
        self.adjacency_count = (
            self.indices_path.stat().st_size // self._integer_struct.size
        )
        if edge_count > self.adjacency_count:
            raise PartitionBuildError("logical edge count cannot exceed adjacency count")
        self.vertex_count = vertex_count
        self.edge_count = edge_count
        self.weighted = self.weights_path is not None
        self.weight_dtype = weight_dtype if self.weighted else None
        self._weight_struct = _weight_struct(weight_dtype) if self.weighted else None
        if self.weights_path is not None:
            if not self.weights_path.is_file():
                raise PartitionBuildError(
                    f"host CSR weights are not a file: {self.weights_path}"
                )
            assert self._weight_struct is not None
            expected = self.adjacency_count * self._weight_struct.size
            if self.weights_path.stat().st_size != expected:
                raise PartitionBuildError(
                    f"host CSR weights size must be {expected} bytes"
                )

    def scan(self, consume: Callable[[int, int, Optional[float]], None]) -> _ScanResult:
        digest = hashlib.sha256()
        offsets_digest = hashlib.sha256()
        observed = 0
        with self.offsets_path.open("rb") as offsets_stream:
            with self.indices_path.open("rb") as indices_stream:
                weights_stream = (
                    None if self.weights_path is None else self.weights_path.open("rb")
                )
                try:
                    previous = _read_value(
                        offsets_stream,
                        self._integer_struct,
                        digest,
                        "offsets",
                        secondary_digest=offsets_digest,
                    )
                    if previous != 0:
                        raise PartitionBuildError("host CSR offsets must start at zero")
                    for source in range(self.vertex_count):
                        current = _read_value(
                            offsets_stream,
                            self._integer_struct,
                            digest,
                            "offsets",
                            secondary_digest=offsets_digest,
                        )
                        if (
                            current < 0
                            or current < previous
                            or current > self.adjacency_count
                        ):
                            raise PartitionBuildError(
                                "host CSR offsets must be monotone and bounded by "
                                "the indices count"
                            )
                        for _ in range(previous, current):
                            target = _read_value(
                                indices_stream, self._integer_struct, digest, "indices"
                            )
                            if target < 0 or target >= self.vertex_count:
                                raise PartitionBuildError(
                                    f"host CSR target {target} is outside "
                                    f"[0, {self.vertex_count})"
                                )
                            if weights_stream is None:
                                weight = None
                            else:
                                assert self._weight_struct is not None
                                weight = _read_value(
                                    weights_stream,
                                    self._weight_struct,
                                    digest,
                                    "weights",
                                )
                                if not math.isfinite(weight):
                                    raise PartitionBuildError(
                                        "host CSR weights must be finite"
                                    )
                            consume(source, target, weight)
                            observed += 1
                        previous = current
                    if previous != self.adjacency_count:
                        raise PartitionBuildError(
                            "host CSR final offset does not equal indices count"
                        )
                    if offsets_stream.read(1) or indices_stream.read(1):
                        raise PartitionBuildError("host CSR contains trailing data")
                    if weights_stream is not None and weights_stream.read(1):
                        raise PartitionBuildError("host CSR weights contain trailing data")
                finally:
                    if weights_stream is not None:
                        weights_stream.close()
        if observed != self.adjacency_count:
            raise PartitionBuildError("host CSR scan did not consume every adjacency")
        return _ScanResult(
            self.edge_count,
            observed,
            digest.hexdigest(),
            offsets_digest.hexdigest(),
        )


@dataclass
class _PartitionScratch:
    ownership: _Range
    degree_path: Path
    halo_candidates_path: Path
    adjacency_count: int = 0
    cut_edge_count: int = 0
    degree_file: Any = None
    degree_map: Optional[mmap.mmap] = None
    halo_stream: Any = None
    halo: Optional[ArrayFile] = None
    offsets: Optional[ArrayFile] = None
    local_to_global: Optional[ArrayFile] = None
    owners: Optional[ArrayFile] = None
    indices: Optional[ArrayFile] = None
    weights: Optional[ArrayFile] = None
    cursor_path: Optional[Path] = None
    cursor_file: Any = None
    cursor_map: Optional[mmap.mmap] = None
    indices_file: Any = None
    indices_map: Optional[mmap.mmap] = None
    weights_file: Any = None
    weights_map: Optional[mmap.mmap] = None
    halo_lookup_path: Optional[Path] = None
    halo_lookup_file: Any = None
    halo_lookup_map: Optional[mmap.mmap] = None
    halo_lookup_capacity: int = 0

    def open_first_pass(self) -> None:
        self.degree_path.parent.mkdir(parents=True, exist_ok=True)
        self.degree_file = self.degree_path.open("x+b")
        self.degree_file.truncate(self.ownership.count * _U64.size)
        if self.ownership.count:
            self.degree_map = _map_file(
                self.degree_file, "disk-backed degree counters"
            )
        self.halo_stream = self.halo_candidates_path.open("xb")

    def close_first_pass(self) -> None:
        failures = []
        if self.degree_map is not None:
            try:
                self.degree_map.close()
            except BaseException as exc:
                failures.append(exc)
            self.degree_map = None
        if self.degree_file is not None:
            try:
                self.degree_file.close()
            except BaseException as exc:
                failures.append(exc)
            self.degree_file = None
        if self.halo_stream is not None:
            try:
                self.halo_stream.close()
            except BaseException as exc:
                failures.append(exc)
            self.halo_stream = None
        if failures:
            raise failures[0]

    def close_second_pass(self) -> None:
        failures = []
        for mapped, durable in (
            (self.cursor_map, False),
            (self.indices_map, True),
            (self.weights_map, True),
            (self.halo_lookup_map, False),
        ):
            if mapped is None:
                continue
            try:
                if durable:
                    mapped.flush()
                mapped.close()
            except BaseException as exc:
                failures.append(exc)
        self.cursor_map = None
        self.indices_map = None
        self.weights_map = None
        self.halo_lookup_map = None
        for stream, durable in (
            (self.cursor_file, False),
            (self.indices_file, True),
            (self.weights_file, True),
            (self.halo_lookup_file, False),
        ):
            if stream is None:
                continue
            try:
                if durable:
                    stream.flush()
                    os.fsync(stream.fileno())
                stream.close()
            except BaseException as exc:
                failures.append(exc)
        self.cursor_file = None
        self.indices_file = None
        self.weights_file = None
        self.halo_lookup_file = None
        self.halo_lookup_capacity = 0
        if failures:
            raise failures[0]

    def close_all(self) -> None:
        failures = []
        for close in (self.close_first_pass, self.close_second_pass):
            try:
                close()
            except BaseException as exc:
                failures.append(exc)
        if failures:
            raise failures[0]


class _RangeOwner:
    def __init__(self, ranges: Sequence[_Range]) -> None:
        self.ranges = tuple(ranges)
        self.boundaries = tuple(part.begin for part in self.ranges) + (
            self.ranges[-1].end,
        )

    def partition(self, global_id: int) -> _Range:
        index = bisect.bisect_right(self.boundaries, global_id) - 1
        if index < 0 or index >= len(self.ranges):
            raise PartitionBuildError(f"global vertex {global_id} has no owner")
        result = self.ranges[index]
        if not result.begin <= global_id < result.end:
            raise PartitionBuildError(f"global vertex {global_id} has no owner")
        return result


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _atomic_destination(
    output_root: os.PathLike[str] | str,
) -> Tuple[Path, Path]:
    final = Path(output_root)
    if not final.name:
        raise ManifestPathError("output directory must have a final path component")
    parent = final.parent.resolve(strict=True)
    project = _project_root()
    try:
        parent.relative_to(project)
    except ValueError as exc:
        raise ManifestPathError(
            f"partition output must remain under the EGMGPU project root {project}"
        ) from exc
    final = parent / final.name
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"refusing to overwrite output directory {final}")
    stage = parent / f".{final.name}.building-{uuid.uuid4().hex}"
    stage.mkdir()
    return final, stage


def _range_partitions(vertex_count: int, devices: Sequence[int]) -> Tuple[_Range, ...]:
    if isinstance(vertex_count, bool) or not isinstance(vertex_count, int):
        raise TypeError("vertex_count must be an integer")
    if vertex_count <= 0 or vertex_count > UINT64_MAX:
        raise ValueError(f"vertex_count must be in [1, {UINT64_MAX}]")
    normalized = tuple(devices)
    if not normalized:
        raise ValueError("at least one device is required")
    if any(isinstance(device, bool) or not isinstance(device, int) for device in normalized):
        raise TypeError("device identifiers must be integers")
    if len(set(normalized)) != len(normalized) or any(device < 0 for device in normalized):
        raise ValueError("device identifiers must be unique and non-negative")
    if any(device > INT32_MAX for device in normalized):
        raise ValueError("device identifiers must fit signed int32")
    count = len(normalized)
    return tuple(
        _Range(
            partition_id=index,
            device_id=device,
            begin=(index * vertex_count) // count,
            end=((index + 1) * vertex_count) // count,
        )
        for index, device in enumerate(normalized)
    )


def _incoming_adjacency_balanced_ranges(
    indices_path: Path,
    *,
    vertex_count: int,
    devices: Sequence[int],
    integer_struct: struct.Struct,
) -> _BalancedRangePlan:
    """Choose owner ranges by incoming work while retaining outgoing CSR.

    This research policy adds one indices scan and O(vertex_count) uint64
    counters. No edges are transposed or reordered. The normal builder still
    checks the capacity of the stored outgoing partitions.
    """
    base = _range_partitions(vertex_count, devices)
    prefix = array("Q", [0]) * (vertex_count + 1)
    with indices_path.open("rb") as stream:
        while True:
            data = stream.read((1 << 17) * integer_struct.size)
            if not data:
                break
            for (target,) in integer_struct.iter_unpack(data):
                if target < 0 or target >= vertex_count:
                    raise PartitionBuildError("incoming balance target is outside the graph")
                prefix[target + 1] += 1
    for vertex in range(vertex_count):
        prefix[vertex + 1] += prefix[vertex]
    total = prefix[-1]
    if total == 0:
        return _BalancedRangePlan(tuple(base), None)
    boundaries = [0]
    for rank in range(1, len(base)):
        target = (rank * total + len(base) - 1) // len(base)
        upper = bisect.bisect_left(prefix, target, lo=boundaries[-1])
        lower = max(boundaries[-1], upper - 1)
        boundaries.append(min((lower, upper), key=lambda item: (abs(prefix[item] - target), item)))
    boundaries.append(vertex_count)
    for begin, end in zip(boundaries, boundaries[1:]):
        if prefix[end] - prefix[begin] > MAX_LOCAL_ADJACENCY_ENTRIES:
            raise PartitionBuildError("incoming-balanced pull partition exceeds int32 capacity")
    return _BalancedRangePlan(
        tuple(_Range(rank, base[rank].device_id, boundaries[rank], boundaries[rank + 1])
              for rank in range(len(base))),
        None,
    )


def _adjacency_balanced_ranges(
    offsets_path: Path,
    *,
    vertex_count: int,
    adjacency_count: int,
    devices: Sequence[int],
    integer_struct: struct.Struct,
) -> _BalancedRangePlan:
    """Choose capacity-safe contiguous ranges near equal adjacency targets.

    A row is never split.  Before selecting a boundary, the builder reserves
    enough suffix capacity for every remaining partition.  This avoids the
    unsafe equal-distance tie in which a nearest-target policy can put more
    than ``INT32_MAX`` adjacency entries in the current partition even though
    a feasible contiguous segmentation exists.  Repeated boundaries remain
    legal and encode strict empty partitions.

    The returned SHA-256 is computed both before and after the boundary pass.
    The first full source scan must observe the same offsets hash, binding the
    partition plan to the source on which it was derived.
    """

    base = _range_partitions(vertex_count, devices)
    path = offsets_path.resolve(strict=True)
    expected = (vertex_count + 1) * integer_struct.size
    if path.stat().st_size != expected:
        raise PartitionBuildError(
            "host CSR offsets do not contain vertex_count+1 values"
        )
    before_sha256 = _hash_file(path)
    with path.open("rb") as stream:
        mapped = _map_file(
            stream,
            "adjacency-balanced host CSR offsets",
            access=mmap.ACCESS_READ,
        )
        try:
            def offset(index: int) -> int:
                value = integer_struct.unpack_from(
                    mapped, index * integer_struct.size
                )[0]
                if value < 0:
                    raise PartitionBuildError("host CSR offset is negative")
                return int(value)

            if offset(0) != 0 or offset(vertex_count) != adjacency_count:
                raise PartitionBuildError(
                    "host CSR offsets do not span the complete adjacency array"
                )

            # Earliest suffix start that can be covered by r capacity-limited
            # partitions.  Backward greedy is optimal for nonnegative CSR row
            # sizes and needs only O(number of devices) resident memory.
            suffix_start = [vertex_count] * (len(base) + 1)
            suffix_start[0] = vertex_count
            end = vertex_count
            for remaining in range(1, len(base) + 1):
                threshold = max(0, offset(end) - MAX_LOCAL_ADJACENCY_ENTRIES)
                low, high = 0, end
                while low < high:
                    middle = (low + high) // 2
                    if offset(middle) < threshold:
                        low = middle + 1
                    else:
                        high = middle
                end = low
                suffix_start[remaining] = end
            if suffix_start[len(base)] != 0:
                raise PartitionBuildError(
                    "no capacity-safe contiguous segmentation exists for the "
                    "requested device count"
                )

            boundaries = [0]
            begin = 0
            for rank in range(1, len(base)):
                remaining = len(base) - rank
                minimum = max(begin, suffix_start[remaining])
                capacity_target = (
                    offset(begin) + MAX_LOCAL_ADJACENCY_ENTRIES
                )
                low, high = begin, vertex_count + 1
                while low < high:
                    middle = (low + high) // 2
                    if middle <= vertex_count and offset(middle) <= capacity_target:
                        low = middle + 1
                    else:
                        high = middle
                maximum = low - 1
                if minimum > maximum:
                    raise PartitionBuildError(
                        "no capacity-safe contiguous segmentation exists for the "
                        "requested device count"
                    )

                target = (
                    rank * adjacency_count + len(base) - 1
                ) // len(base)
                low, high = minimum, maximum + 1
                while low < high:
                    middle = (low + high) // 2
                    if offset(middle) < target:
                        low = middle + 1
                    else:
                        high = middle
                candidates = {minimum, maximum}
                if low <= maximum:
                    candidates.add(low)
                if low - 1 >= minimum:
                    candidates.add(low - 1)
                # Prefer the lower boundary on an exact distance tie.  Besides
                # determinism, this leaves capacity available to the suffix.
                boundary = min(
                    candidates,
                    key=lambda candidate: (
                        abs(offset(candidate) - target),
                        candidate,
                    ),
                )
                boundaries.append(boundary)
                begin = boundary
            boundaries.append(vertex_count)

            for begin, end in zip(boundaries, boundaries[1:]):
                if offset(end) - offset(begin) > MAX_LOCAL_ADJACENCY_ENTRIES:
                    raise PartitionBuildError(
                        "adjacency-balanced range exceeds the signed int32 "
                        "partition capacity"
                    )
        finally:
            active_failure = sys.exc_info()[0] is not None
            try:
                mapped.close()
            except BaseException as exc:
                if not active_failure:
                    raise PartitionBuildError(
                        f"cannot close adjacency-balanced offsets mapping: {exc}"
                    ) from exc
    after_sha256 = _hash_file(path)
    if after_sha256 != before_sha256:
        raise PartitionBuildError(
            "host CSR offsets changed while adjacency-balanced boundaries "
            "were computed"
        )
    return _BalancedRangePlan(
        ranges=tuple(
            _Range(
                index,
                base[index].device_id,
                boundaries[index],
                boundaries[index + 1],
            )
            for index in range(len(base))
        ),
        offsets_sha256=before_sha256,
    )


def _checked_u64_add(value: int, increment: int, context: str) -> int:
    result = value + increment
    if result > UINT64_MAX:
        raise PartitionBuildError(f"{context} exceeds the uint64 domain")
    return result


def _map_file(
    stream: Any,
    context: str,
    *,
    access: Optional[int] = None,
) -> mmap.mmap:
    """Map a nonempty file and normalize address-space failures."""

    try:
        if access is None:
            return mmap.mmap(stream.fileno(), 0)
        return mmap.mmap(stream.fileno(), 0, access=access)
    except (OSError, ValueError, OverflowError, MemoryError) as exc:
        raise PartitionBuildError(f"cannot map {context}: {exc}") from exc


def _translate_resource_errors(function: Callable[..., Any]) -> Callable[..., Any]:
    """Convert storage/allocation failures at the public builder boundary."""

    @functools.wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except (PartitionBuildError, FileExistsError):
            raise
        except (OSError, OverflowError, MemoryError) as exc:
            raise PartitionBuildError(
                f"partition build resource failure: {exc}"
            ) from exc

    return wrapped


def _weight_struct(dtype: str) -> struct.Struct:
    if dtype == "float32":
        return _F32
    if dtype == "float64":
        return _F64
    raise ValueError("weight_dtype must be 'float32' or 'float64'")


def _normalize_view(
    directed: bool,
    weighted: bool,
    orientation: Optional[str],
    weight_key: Optional[str],
) -> Tuple[str, Optional[str]]:
    if orientation is not None and not isinstance(orientation, str):
        raise TypeError("orientation must be a string or None")
    normalized = (
        ("outgoing" if directed else "undirected")
        if orientation is None
        else orientation
    )
    if normalized not in {"outgoing", "incoming", "undirected"}:
        raise ValueError(
            "orientation must be 'outgoing', 'incoming', or 'undirected'"
        )
    if directed != (normalized != "undirected"):
        raise ValueError("orientation is inconsistent with directed")
    if weight_key is not None and (not isinstance(weight_key, str) or not weight_key):
        raise ValueError("weight_key must be a non-empty string or None")
    if not weighted and weight_key is not None:
        raise ValueError("an unweighted view cannot declare a weight key")
    return normalized, weight_key


def _read_value(
    stream: Any,
    unpacker: struct.Struct,
    digest: Any,
    name: str,
    *,
    secondary_digest: Any = None,
) -> Any:
    raw = stream.read(unpacker.size)
    if len(raw) != unpacker.size:
        raise PartitionBuildError(f"unexpected end of host CSR {name}")
    digest.update(raw)
    if secondary_digest is not None:
        secondary_digest.update(raw)
    return unpacker.unpack(raw)[0]


def _increment_degree(partition: _PartitionScratch, local_source: int) -> None:
    assert partition.degree_map is not None
    offset = local_source * _U64.size
    current = _U64.unpack_from(partition.degree_map, offset)[0]
    if current == UINT64_MAX:
        raise PartitionBuildError("vertex degree exceeds the uint64 domain")
    _U64.pack_into(partition.degree_map, offset, current + 1)


def _read_u64_file(path: Path) -> Iterator[int]:
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(_DEFAULT_SORT_CHUNK_VALUES * _U64.size)
            if not chunk:
                break
            if len(chunk) % _U64.size:
                raise PartitionBuildError(f"unaligned uint64 scratch file {path}")
            for (value,) in struct.iter_unpack("<Q", chunk):
                yield value


def _write_u64_run(path: Path, values: Iterable[int]) -> None:
    with path.open("xb") as stream:
        buffer = bytearray()
        for value in values:
            buffer.extend(_U64.pack(value))
            if len(buffer) >= (1 << 20):
                stream.write(buffer)
                buffer.clear()
        if buffer:
            stream.write(buffer)


def _unique(values: Iterable[int]) -> Iterator[int]:
    marker: Optional[int] = None
    initialized = False
    for value in values:
        if not initialized or value != marker:
            yield value
            marker = value
            initialized = True


def _external_sort_unique_u64(
    candidates: Path,
    output: Path,
    run_root: Path,
    *,
    chunk_values: int,
    merge_fan_in: int,
) -> None:
    if chunk_values < 1 or merge_fan_in < 2:
        raise ValueError("sort chunk must be positive and merge fan-in at least two")
    run_root.mkdir()
    runs = []
    with candidates.open("rb") as stream:
        run_index = 0
        while True:
            chunk = stream.read(chunk_values * _U64.size)
            if not chunk:
                break
            if len(chunk) % _U64.size:
                raise PartitionBuildError("halo candidate scratch is not uint64-aligned")
            values = sorted(
                set(value for (value,) in struct.iter_unpack("<Q", chunk))
            )
            run = run_root / f"run-{run_index:08d}.u64"
            _write_u64_run(run, values)
            runs.append(run)
            run_index += 1
    if not runs:
        _write_u64_run(output, ())
        run_root.rmdir()
        return

    pass_index = 0
    while len(runs) > 1:
        merged_runs = []
        for group_index in range(0, len(runs), merge_fan_in):
            group = runs[group_index : group_index + merge_fan_in]
            merged = run_root / (
                f"merge-{pass_index:04d}-{group_index // merge_fan_in:08d}.u64"
            )
            streams = [_read_u64_file(path) for path in group]
            _write_u64_run(merged, _unique(heapq.merge(*streams)))
            for path in group:
                path.unlink()
            merged_runs.append(merged)
        runs = merged_runs
        pass_index += 1
    runs[0].replace(output)
    run_root.rmdir()


def _undirected_validation_record(
    source: int,
    target: int,
    weight: Optional[float],
    weight_packer: Optional[struct.Struct],
) -> bytes:
    low = min(source, target)
    high = max(source, target)
    direction = 2 if source == target else (0 if source == low else 1)
    record = _U64.pack(low) + _U64.pack(high)
    if weight_packer is not None:
        if weight is None:
            raise PartitionBuildError("weighted undirected CSR omitted an edge weight")
        record += weight_packer.pack(weight)
    return record + bytes((direction,))


def _read_fixed_records(path: Path, record_size: int) -> Iterator[bytes]:
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(record_size * 65536)
            if not chunk:
                break
            if len(chunk) % record_size:
                raise PartitionBuildError(
                    f"unaligned undirected-validation scratch file {path}"
                )
            for offset in range(0, len(chunk), record_size):
                yield chunk[offset : offset + record_size]


def _write_fixed_records(path: Path, records: Iterable[bytes]) -> None:
    with path.open("xb") as stream:
        buffer = bytearray()
        for record in records:
            buffer.extend(record)
            if len(buffer) >= (1 << 20):
                stream.write(buffer)
                buffer.clear()
        if buffer:
            stream.write(buffer)


def _external_sort_fixed_records(
    records_path: Path,
    output: Path,
    run_root: Path,
    *,
    record_size: int,
    chunk_records: int,
    merge_fan_in: int,
) -> None:
    """External-sort fixed records while preserving duplicate multiplicity."""

    run_root.mkdir()
    runs = []
    with records_path.open("rb") as stream:
        run_index = 0
        while True:
            chunk = stream.read(chunk_records * record_size)
            if not chunk:
                break
            if len(chunk) % record_size:
                raise PartitionBuildError(
                    "undirected-validation scratch is not record-aligned"
                )
            values = [
                chunk[offset : offset + record_size]
                for offset in range(0, len(chunk), record_size)
            ]
            values.sort()
            run = run_root / f"run-{run_index:08d}.bin"
            _write_fixed_records(run, values)
            runs.append(run)
            run_index += 1
    if not runs:
        _write_fixed_records(output, ())
        run_root.rmdir()
        return
    pass_index = 0
    while len(runs) > 1:
        merged_runs = []
        for group_index in range(0, len(runs), merge_fan_in):
            group = runs[group_index : group_index + merge_fan_in]
            merged = run_root / (
                f"merge-{pass_index:04d}-{group_index // merge_fan_in:08d}.bin"
            )
            _write_fixed_records(
                merged,
                heapq.merge(
                    *(_read_fixed_records(path, record_size) for path in group)
                ),
            )
            for path in group:
                path.unlink()
            merged_runs.append(merged)
        runs = merged_runs
        pass_index += 1
    runs[0].replace(output)
    run_root.rmdir()


def _validate_undirected_records(
    sorted_records: Path,
    *,
    record_size: int,
    expected_edge_count: int,
) -> None:
    current_key: Optional[bytes] = None
    forward = 0
    reverse = 0
    loops = 0
    logical_edges = 0

    def finish_group() -> None:
        nonlocal logical_edges
        if current_key is None:
            return
        low = _U64.unpack(current_key[: _U64.size])[0]
        high = _U64.unpack(current_key[_U64.size : 2 * _U64.size])[0]
        if low == high:
            if forward or reverse:
                raise PartitionBuildError(
                    "undirected host CSR contains an invalid self-loop encoding"
                )
            logical_edges += loops
        else:
            if loops or forward != reverse:
                raise PartitionBuildError(
                    "undirected host CSR is not reciprocal with symmetric weights"
                )
            logical_edges += forward

    for record in _read_fixed_records(sorted_records, record_size):
        key = record[:-1]
        direction = record[-1]
        if key != current_key:
            finish_group()
            current_key = key
            forward = 0
            reverse = 0
            loops = 0
        if direction == 0:
            forward += 1
        elif direction == 1:
            reverse += 1
        elif direction == 2:
            loops += 1
        else:
            raise PartitionBuildError(
                "undirected-validation record has an invalid direction"
            )
    finish_group()
    if logical_edges != expected_edge_count:
        raise PartitionBuildError(
            "undirected host CSR logical edge_count does not match its "
            "reciprocal adjacency"
        )


def _describe_array(
    stage: Path, relative_path: str, dtype: str, count: int
) -> ArrayFile:
    path = stage / relative_path
    sizes = {"int32": 4, "uint64": 8, "float32": 4, "float64": 8}
    expected = count * sizes[dtype]
    actual = path.stat().st_size
    if actual != expected:
        raise PartitionBuildError(
            f"{relative_path}: generated size {actual} does not equal {expected}"
        )
    return ArrayFile(relative_path, dtype, count, actual, _hash_file(path))


def _find_halo_local_id(partition: _PartitionScratch, global_id: int) -> int:
    if (
        partition.halo_lookup_map is None
        or partition.halo_lookup_capacity == 0
    ):
        raise PartitionBuildError("partition halo lookup is not open")
    encoded = global_id + 1
    mask = partition.halo_lookup_capacity - 1
    slot = _splitmix64(global_id) & mask
    for _ in range(partition.halo_lookup_capacity):
        offset = slot * _HALO_SLOT.size
        stored, local_id = _HALO_SLOT.unpack_from(
            partition.halo_lookup_map, offset
        )
        if stored == encoded:
            if local_id < partition.ownership.count:
                raise PartitionBuildError(
                    f"remote neighbor {global_id} has an invalid halo local ID"
                )
            return local_id
        if stored == 0:
            break
        slot = (slot + 1) & mask
    raise PartitionBuildError(
        f"remote neighbor {global_id} is absent from partition halo"
    )


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & UINT64_MAX
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & UINT64_MAX
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & UINT64_MAX
    return value ^ (value >> 31)


def _halo_table_capacity(halo_count: int) -> int:
    if halo_count == 0:
        return 0
    capacity = 1
    required = halo_count * 2
    while capacity < required:
        capacity <<= 1
    return capacity


def _open_halo_lookup(
    partition: _PartitionScratch,
    *,
    stage: Path,
    work: Path,
) -> None:
    """Build a deterministic O(H) disk-backed open-address halo table."""

    if partition.halo is None:
        raise PartitionBuildError("partition halo metadata is missing")
    capacity = _halo_table_capacity(partition.halo.count)
    if capacity == 0:
        # A zero-halo partition has no lookup file or mapping at all.
        return
    partition.halo_lookup_path = work / (
        f"p{partition.ownership.partition_id}.halo_lookup.bin"
    )
    partition.halo_lookup_file = partition.halo_lookup_path.open("x+b")
    byte_count = capacity * _HALO_SLOT.size
    partition.halo_lookup_file.truncate(byte_count)
    partition.halo_lookup_map = _map_file(
        partition.halo_lookup_file,
        "disk-backed halo hash table",
    )
    partition.halo_lookup_capacity = capacity
    mask = capacity - 1
    local_id = partition.ownership.count
    for global_id in partition.halo.values(stage):
        if local_id > INT32_MAX:
            raise PartitionBuildError("halo local ID exceeds signed int32")
        encoded = global_id + 1
        slot = _splitmix64(global_id) & mask
        for _ in range(capacity):
            offset = slot * _HALO_SLOT.size
            stored = _U64.unpack_from(partition.halo_lookup_map, offset)[0]
            if stored == 0:
                _HALO_SLOT.pack_into(
                    partition.halo_lookup_map,
                    offset,
                    encoded,
                    local_id,
                )
                break
            if stored == encoded:
                raise PartitionBuildError(
                    f"duplicate global vertex {global_id} in sorted halo"
                )
            slot = (slot + 1) & mask
        else:
            raise PartitionBuildError("halo lookup table is unexpectedly full")
        local_id += 1


def _ensure_partition_adjacency_limit(counts: Iterable[int]) -> None:
    for partition_id, count in enumerate(counts):
        if count > MAX_LOCAL_ADJACENCY_ENTRIES:
            raise PartitionBuildError(
                f"partition {partition_id} has {count} adjacency entries; "
                "manifest v1 requires fewer than 2^31"
            )


def _verify_row_cursors(partition: _PartitionScratch) -> None:
    if partition.cursor_path is None:
        raise PartitionBuildError("partition cursor path is missing")
    cumulative = 0
    with partition.degree_path.open("rb") as degree_stream:
        with partition.cursor_path.open("rb") as cursor_stream:
            for _ in range(partition.ownership.count):
                degree_raw = degree_stream.read(_U64.size)
                cursor_raw = cursor_stream.read(_U64.size)
                if len(degree_raw) != _U64.size or len(cursor_raw) != _U64.size:
                    raise PartitionBuildError("generated row cursor file is truncated")
                cumulative += _U64.unpack(degree_raw)[0]
                cursor = _U64.unpack(cursor_raw)[0]
                if cursor != cumulative:
                    raise PartitionBuildError(
                        f"partition {partition.ownership.partition_id} CSR row fill "
                        "does not match its first-pass degree"
                    )
            if degree_stream.read(1) or cursor_stream.read(1):
                raise PartitionBuildError("generated row cursor file has trailing data")


def _initial_row_cursors(offsets: ArrayFile, root: Path, row_count: int) -> Iterator[int]:
    """Read exactly the row-start offsets and close the source deterministically."""

    path = offsets.resolve(root)
    with path.open("rb") as stream:
        for _ in range(row_count):
            raw = stream.read(_I32.size)
            if len(raw) != _I32.size:
                raise PartitionBuildError("generated offsets file is truncated")
            value = _I32.unpack(raw)[0]
            if value < 0:
                raise PartitionBuildError("generated offsets contain a negative value")
            yield value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _build_range_partitioned_csr(
    output_root: os.PathLike[str] | str,
    source: _Source,
    *,
    vertex_count: int,
    devices: Sequence[int],
    directed: bool,
    view_orientation: str,
    weight_key: Optional[str],
    graph_id: str,
    generation: int,
    sort_chunk_values: int,
    merge_fan_in: int,
    ranges: Optional[Sequence[_Range]] = None,
    boundary_offsets_sha256: Optional[str] = None,
    validate_undirected_host: bool = False,
) -> PartitionBuildReport:
    if not isinstance(graph_id, str) or not graph_id:
        raise ValueError("graph_id must be a non-empty string")
    if isinstance(generation, bool) or not isinstance(generation, int):
        raise TypeError("generation must be an integer")
    if generation < 0 or generation > UINT64_MAX:
        raise ValueError("generation must fit uint64")
    if isinstance(sort_chunk_values, bool) or not isinstance(sort_chunk_values, int):
        raise TypeError("sort_chunk_values must be an integer")
    if isinstance(merge_fan_in, bool) or not isinstance(merge_fan_in, int):
        raise TypeError("merge_fan_in must be an integer")
    if sort_chunk_values < 1 or merge_fan_in < 2:
        raise ValueError("sort chunk must be positive and merge fan-in at least two")
    default_ranges = _range_partitions(vertex_count, devices)
    ranges = tuple(default_ranges if ranges is None else ranges)
    if tuple(part.device_id for part in ranges) != tuple(devices):
        raise ValueError("partition ranges must preserve the requested device order")
    if (
        not ranges
        or ranges[0].begin != 0
        or ranges[-1].end != vertex_count
        or any(part.partition_id != index for index, part in enumerate(ranges))
        or any(part.begin > part.end for part in ranges)
        or any(ranges[index].end != ranges[index + 1].begin for index in range(len(ranges) - 1))
    ):
        raise ValueError("partition ranges must be an ordered exact vertex cover")
    owner = _RangeOwner(ranges)
    final, stage = _atomic_destination(output_root)
    work = stage / ".work"
    parts_dir = stage / "parts"
    try:
        work.mkdir()
        parts_dir.mkdir()
        scratch = [
            _PartitionScratch(
                ownership=partition,
                degree_path=work / f"p{partition.partition_id}.degrees.u64",
                halo_candidates_path=(
                    work / f"p{partition.partition_id}.halo-candidates.u64"
                ),
            )
            for partition in ranges
        ]
    except BaseException:
        try:
            if stage.exists():
                resolved_stage = stage.resolve(strict=True)
                resolved_stage.relative_to(_project_root())
                shutil.rmtree(resolved_stage)
        except BaseException:
            pass
        raise
    committed = False
    try:
        for partition in scratch:
            partition.open_first_pass()

        validation_path = (
            work / "undirected-validation.records"
            if validate_undirected_host
            else None
        )
        validation_stream = (
            None if validation_path is None else validation_path.open("xb")
        )
        validation_packer = (
            _weight_struct(str(source.weight_dtype))
            if validate_undirected_host and source.weighted
            else None
        )

        def count_arc(source_id: int, target_id: int, weight: Optional[float]) -> None:
            source_owner = owner.partition(source_id)
            target_owner = owner.partition(target_id)
            partition = scratch[source_owner.partition_id]
            local_source = source_id - source_owner.begin
            _increment_degree(partition, local_source)
            partition.adjacency_count = _checked_u64_add(
                partition.adjacency_count, 1, "partition adjacency count"
            )
            if source_owner.partition_id != target_owner.partition_id:
                partition.cut_edge_count = _checked_u64_add(
                    partition.cut_edge_count, 1, "partition cut-edge count"
                )
                assert partition.halo_stream is not None
                partition.halo_stream.write(_U64.pack(target_id))
            if validation_stream is not None:
                validation_stream.write(
                    _undirected_validation_record(
                        source_id,
                        target_id,
                        weight,
                        validation_packer,
                    )
                )

        try:
            first = source.scan(count_arc)
        except BaseException:
            if validation_stream is not None:
                try:
                    validation_stream.close()
                except BaseException:
                    pass
            raise
        else:
            if validation_stream is not None:
                validation_stream.close()
        for partition in scratch:
            partition.close_first_pass()
        if (
            boundary_offsets_sha256 is not None
            and first.offsets_sha256 != boundary_offsets_sha256
        ):
            raise PartitionBuildError(
                "host CSR offsets changed after adjacency-balanced boundaries "
                "were computed"
            )
        if sum(part.adjacency_count for part in scratch) != first.adjacency_count:
            raise PartitionBuildError("first-pass partition counts are inconsistent")
        _ensure_partition_adjacency_limit(
            part.adjacency_count for part in scratch
        )
        if validation_path is not None:
            record_size = 2 * _U64.size + (
                0 if validation_packer is None else validation_packer.size
            ) + 1
            sorted_validation = work / "undirected-validation.sorted"
            _external_sort_fixed_records(
                validation_path,
                sorted_validation,
                work / "undirected-validation-runs",
                record_size=record_size,
                chunk_records=sort_chunk_values,
                merge_fan_in=merge_fan_in,
            )
            _validate_undirected_records(
                sorted_validation,
                record_size=record_size,
                expected_edge_count=first.edge_count,
            )
            validation_path.unlink()
            sorted_validation.unlink()

        for partition in scratch:
            partition_id = partition.ownership.partition_id
            halo_path = work / f"p{partition_id}.halo.u64"
            _external_sort_unique_u64(
                partition.halo_candidates_path,
                halo_path,
                work / f"p{partition_id}.runs",
                chunk_values=sort_chunk_values,
                merge_fan_in=merge_fan_in,
            )
            partition.halo_candidates_path.unlink()
            halo_count = halo_path.stat().st_size // _U64.size
            partition.halo = ArrayFile(
                relative_path=str(halo_path.relative_to(stage)),
                dtype="uint64",
                count=halo_count,
                byte_size=halo_path.stat().st_size,
                sha256=_hash_file(halo_path),
            )
            if partition.ownership.count + halo_count > (1 << 31):
                raise PartitionBuildError(
                    f"partition {partition_id} exceeds the signed int32 local-ID domain"
                )

            def offsets() -> Iterator[int]:
                cumulative = 0
                yield 0
                for degree in _read_u64_file(partition.degree_path):
                    cumulative += degree
                    if cumulative > MAX_LOCAL_ADJACENCY_ENTRIES:
                        raise PartitionBuildError(
                            f"partition {partition_id} offsets exceed signed int32"
                        )
                    yield cumulative
                if cumulative != partition.adjacency_count:
                    raise PartitionBuildError(
                        f"partition {partition_id} degree sum is inconsistent"
                    )

            prefix = f"parts/p{partition_id}"
            partition.offsets = write_array_stream(
                stage, f"{prefix}.offsets.i32", "int32", offsets()
            )
            assert partition.halo is not None
            partition.local_to_global = write_array_stream(
                stage,
                f"{prefix}.local_to_global.u64",
                "uint64",
                itertools.chain(
                    range(partition.ownership.begin, partition.ownership.end),
                    partition.halo.values(stage),
                ),
            )
            partition.owners = write_array_stream(
                stage,
                f"{prefix}.owners.i32",
                "int32",
                itertools.chain(
                    itertools.repeat(partition_id, partition.ownership.count),
                    (
                        owner.partition(global_id).partition_id
                        for global_id in partition.halo.values(stage)
                    ),
                ),
            )

            partition.cursor_path = work / f"p{partition_id}.cursors.u64"
            _write_u64_run(
                partition.cursor_path,
                _initial_row_cursors(
                    partition.offsets, stage, partition.ownership.count
                ),
            )
            partition.cursor_file = partition.cursor_path.open("r+b")
            if partition.ownership.count:
                partition.cursor_map = _map_file(
                    partition.cursor_file, "disk-backed CSR row cursors"
                )

            indices_relative = f"{prefix}.indices.i32"
            indices_path = stage / indices_relative
            partition.indices_file = indices_path.open("x+b")
            partition.indices_file.truncate(partition.adjacency_count * _I32.size)
            partition.indices_file.flush()
            if partition.adjacency_count:
                partition.indices_map = _map_file(
                    partition.indices_file, "generated partition indices"
                )
            if source.weighted:
                assert source.weight_dtype is not None
                weights_relative = f"{prefix}.weights.{source.weight_dtype}"
                weights_path = stage / weights_relative
                partition.weights_file = weights_path.open("x+b")
                weight_size = _weight_struct(source.weight_dtype).size
                partition.weights_file.truncate(partition.adjacency_count * weight_size)
                partition.weights_file.flush()
                if partition.adjacency_count:
                    partition.weights_map = _map_file(
                        partition.weights_file,
                        "generated partition weights",
                    )
            _open_halo_lookup(
                partition,
                stage=stage,
                work=work,
            )

        def fill_arc(source_id: int, target_id: int, weight: Optional[float]) -> None:
            source_owner = owner.partition(source_id)
            target_owner = owner.partition(target_id)
            partition = scratch[source_owner.partition_id]
            local_source = source_id - source_owner.begin
            assert partition.degree_map is None
            if partition.cursor_map is None:
                raise PartitionBuildError(
                    "an edge was assigned to a partition with no owned vertices"
                )
            cursor_offset = local_source * _U64.size
            position = _U64.unpack_from(partition.cursor_map, cursor_offset)[0]
            if position >= partition.adjacency_count:
                raise PartitionBuildError("second pass exceeds partition adjacency count")
            if source_owner.partition_id == target_owner.partition_id:
                local_target = target_id - source_owner.begin
            else:
                local_target = _find_halo_local_id(partition, target_id)
            if local_target > INT32_MAX:
                raise PartitionBuildError("local target does not fit signed int32")
            if partition.indices_map is None:
                raise PartitionBuildError("nonempty partition indices are not mapped")
            _I32.pack_into(partition.indices_map, position * _I32.size, local_target)
            if source.weighted:
                if weight is None or partition.weights_map is None:
                    raise PartitionBuildError("weighted source omitted an edge weight")
                packer = _weight_struct(str(source.weight_dtype))
                packer.pack_into(
                    partition.weights_map, position * packer.size, weight
                )
            _U64.pack_into(partition.cursor_map, cursor_offset, position + 1)

        second = source.scan(fill_arc)
        if second != first:
            raise PartitionBuildError(
                "graph source changed between the two streaming passes"
            )

        for partition in scratch:
            partition.close_second_pass()
            _verify_row_cursors(partition)
            prefix = f"parts/p{partition.ownership.partition_id}"
            partition.indices = _describe_array(
                stage,
                f"{prefix}.indices.i32",
                "int32",
                partition.adjacency_count,
            )
            if source.weighted:
                assert source.weight_dtype is not None
                partition.weights = _describe_array(
                    stage,
                    f"{prefix}.weights.{source.weight_dtype}",
                    source.weight_dtype,
                    partition.adjacency_count,
                )

        manifest_partitions = []
        for partition in scratch:
            assert partition.offsets is not None
            assert partition.indices is not None
            assert partition.local_to_global is not None
            assert partition.owners is not None
            manifest_partitions.append(
                ManifestPartitionMetadata(
                    partition_id=partition.ownership.partition_id,
                    device_id=partition.ownership.device_id,
                    ownership_encoding="range",
                    owned_vertex_count=partition.ownership.count,
                    halo_vertex_count=(
                        partition.local_to_global.count - partition.ownership.count
                    ),
                    adjacency_count=partition.adjacency_count,
                    cut_edge_count=partition.cut_edge_count,
                    offsets=partition.offsets,
                    indices=partition.indices,
                    local_to_global=partition.local_to_global,
                    owners=partition.owners,
                    weights=partition.weights,
                    vertex_begin=partition.ownership.begin,
                    vertex_end=partition.ownership.end,
                )
            )

        provisional = PartitionedCSRManifest(
            manifest_path=stage / "manifest.json",
            graph_id=graph_id,
            generation=generation,
            directed=directed,
            weighted=source.weighted,
            view_orientation=view_orientation,
            weight_key=weight_key,
            view_signature="0" * 64,
            vertex_count=vertex_count,
            edge_count=first.edge_count,
            adjacency_count=first.adjacency_count,
            topology_sha256="0" * 64,
            weight_sha256="0" * 64 if source.weighted else None,
            ownership_encoding="range",
            partitions=tuple(manifest_partitions),
        )
        topology_ready = PartitionedCSRManifest(
            manifest_path=provisional.manifest_path,
            graph_id=provisional.graph_id,
            generation=provisional.generation,
            directed=provisional.directed,
            weighted=provisional.weighted,
            view_orientation=provisional.view_orientation,
            weight_key=provisional.weight_key,
            view_signature=provisional.view_signature,
            vertex_count=provisional.vertex_count,
            edge_count=provisional.edge_count,
            adjacency_count=provisional.adjacency_count,
            topology_sha256=_topology_signature(provisional),
            weight_sha256=(
                _weight_signature(provisional) if provisional.weighted else None
            ),
            ownership_encoding=provisional.ownership_encoding,
            partitions=provisional.partitions,
        )
        manifest = PartitionedCSRManifest(
            manifest_path=topology_ready.manifest_path,
            graph_id=topology_ready.graph_id,
            generation=topology_ready.generation,
            directed=topology_ready.directed,
            weighted=topology_ready.weighted,
            view_orientation=topology_ready.view_orientation,
            weight_key=topology_ready.weight_key,
            view_signature=_view_signature(topology_ready),
            vertex_count=topology_ready.vertex_count,
            edge_count=topology_ready.edge_count,
            adjacency_count=topology_ready.adjacency_count,
            topology_sha256=topology_ready.topology_sha256,
            weight_sha256=topology_ready.weight_sha256,
            ownership_encoding=topology_ready.ownership_encoding,
            partitions=topology_ready.partitions,
        )

        for partition in scratch:
            partition.close_all()
        shutil.rmtree(work)
        with manifest.manifest_path.open("x", encoding="utf-8") as stream:
            json.dump(_manifest_document(manifest), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        load_partition_manifest(manifest.manifest_path)
        _fsync_directory(stage / "parts")
        _fsync_directory(stage)
        stage.rename(final)
        _fsync_directory(final.parent)
        committed = True
        final_manifest = final / "manifest.json"
        load_partition_manifest(final_manifest)
        return PartitionBuildReport(
            manifest_path=final_manifest,
            source_sha256=first.source_sha256,
            vertex_count=vertex_count,
            edge_count=first.edge_count,
            adjacency_count=first.adjacency_count,
            partition_count=len(ranges),
            weighted=source.weighted,
        )
    finally:
        active_failure = sys.exc_info()[0] is not None
        cleanup_failures = []
        for partition in scratch:
            try:
                partition.close_all()
            except BaseException as exc:
                cleanup_failures.append(exc)
        if not committed:
            try:
                if stage.exists():
                    resolved_stage = stage.resolve(strict=True)
                    resolved_stage.relative_to(_project_root())
                    shutil.rmtree(resolved_stage)
            except BaseException as exc:
                cleanup_failures.append(exc)
        if cleanup_failures and not active_failure:
            raise PartitionBuildError(
                f"partition staging cleanup failed: {cleanup_failures[0]}"
            ) from cleanup_failures[0]


@_translate_resource_errors
def build_partitioned_csr_from_edge_list(
    output_root: os.PathLike[str] | str,
    edge_list_path: os.PathLike[str] | str,
    *,
    vertex_count: int,
    devices: Sequence[int],
    directed: bool = True,
    weighted: bool = False,
    weight_dtype: str = "float32",
    orientation: Optional[str] = None,
    weight_key: Optional[str] = None,
    delimiter: Optional[str] = None,
    comment_prefix: str = "#",
    graph_id: str = "edge-list",
    generation: int = 0,
    sort_chunk_values: int = _DEFAULT_SORT_CHUNK_VALUES,
    merge_fan_in: int = _DEFAULT_MERGE_FAN_IN,
) -> PartitionBuildReport:
    """Build a range-partitioned CSR using exactly two edge-list scans."""

    if not isinstance(directed, bool) or not isinstance(weighted, bool):
        raise TypeError("directed and weighted must be booleans")
    _range_partitions(vertex_count, devices)
    if not isinstance(comment_prefix, str) or not comment_prefix:
        raise ValueError("comment_prefix must be a non-empty string")
    if delimiter is not None and not isinstance(delimiter, str):
        raise TypeError("delimiter must be a string or None")
    if delimiter == "":
        raise ValueError("delimiter must be None or a non-empty string")
    if weighted:
        _weight_struct(weight_dtype)
    view_orientation, normalized_weight_key = _normalize_view(
        directed, weighted, orientation, weight_key
    )
    source = _EdgeListSource(
        Path(edge_list_path),
        vertex_count=vertex_count,
        directed=directed,
        weighted=weighted,
        weight_dtype=weight_dtype,
        delimiter=delimiter,
        comment_prefix=comment_prefix,
    )
    return _build_range_partitioned_csr(
        output_root,
        source,
        vertex_count=vertex_count,
        devices=devices,
        directed=directed,
        view_orientation=view_orientation,
        weight_key=normalized_weight_key,
        graph_id=graph_id,
        generation=generation,
        sort_chunk_values=sort_chunk_values,
        merge_fan_in=merge_fan_in,
    )


@_translate_resource_errors
def build_partitioned_csr_from_host_csr(
    output_root: os.PathLike[str] | str,
    offsets_path: os.PathLike[str] | str,
    indices_path: os.PathLike[str] | str,
    *,
    vertex_count: int,
    devices: Sequence[int],
    directed: bool = True,
    edge_count: Optional[int] = None,
    weights_path: Optional[os.PathLike[str] | str] = None,
    weight_dtype: str = "float32",
    orientation: Optional[str] = None,
    weight_key: Optional[str] = None,
    graph_id: str = "host-csr",
    generation: int = 0,
    sort_chunk_values: int = _DEFAULT_SORT_CHUNK_VALUES,
    merge_fan_in: int = _DEFAULT_MERGE_FAN_IN,
    balance: str = "vertices",
    trust_undirected_input: bool = False,
) -> PartitionBuildReport:
    """Build from little-endian uint64 host offsets and global indices."""

    if not isinstance(directed, bool):
        raise TypeError("directed must be a boolean")
    if not isinstance(trust_undirected_input, bool):
        raise TypeError("trust_undirected_input must be a boolean")
    _range_partitions(vertex_count, devices)
    indices = Path(indices_path).resolve(strict=True)
    if indices.stat().st_size % _U64.size:
        raise PartitionBuildError("host CSR indices are not uint64-aligned")
    adjacency_count = indices.stat().st_size // _U64.size
    if edge_count is None:
        if not directed:
            raise ValueError(
                "edge_count is required for an undirected materialized host CSR"
            )
        logical_edge_count = adjacency_count
    else:
        if isinstance(edge_count, bool) or not isinstance(edge_count, int):
            raise TypeError("edge_count must be an integer")
        if edge_count < 0 or edge_count > UINT64_MAX:
            raise ValueError("edge_count must fit uint64")
        logical_edge_count = edge_count
    if weights_path is not None:
        _weight_struct(weight_dtype)
    view_orientation, normalized_weight_key = _normalize_view(
        directed, weights_path is not None, orientation, weight_key
    )
    source = _HostCSRSource(
        Path(offsets_path),
        indices,
        vertex_count=vertex_count,
        edge_count=logical_edge_count,
        weights_path=None if weights_path is None else Path(weights_path),
        weight_dtype=weight_dtype,
        integer_dtype="uint64",
    )
    if balance not in {"vertices", "adjacency", "incoming_adjacency"}:
        raise ValueError("balance must be 'vertices', 'adjacency', or 'incoming_adjacency'")
    range_plan = (
        _adjacency_balanced_ranges(
            Path(offsets_path),
            vertex_count=vertex_count,
            adjacency_count=adjacency_count,
            devices=devices,
            integer_struct=_U64,
        )
        if balance == "adjacency"
        else None
    )
    if balance == "incoming_adjacency":
        range_plan = _incoming_adjacency_balanced_ranges(
            indices, vertex_count=vertex_count, devices=devices, integer_struct=_U64
        )
    return _build_range_partitioned_csr(
        output_root,
        source,
        vertex_count=vertex_count,
        devices=devices,
        directed=directed,
        view_orientation=view_orientation,
        weight_key=normalized_weight_key,
        graph_id=graph_id,
        generation=generation,
        sort_chunk_values=sort_chunk_values,
        merge_fan_in=merge_fan_in,
        ranges=None if range_plan is None else range_plan.ranges,
        boundary_offsets_sha256=(
            None if range_plan is None else range_plan.offsets_sha256
        ),
        validate_undirected_host=(
            not directed and not trust_undirected_input
        ),
    )


@_translate_resource_errors
def build_partitioned_csr_from_int32_csr(
    output_root: os.PathLike[str] | str,
    offsets_path: os.PathLike[str] | str,
    indices_path: os.PathLike[str] | str,
    *,
    vertex_count: int,
    devices: Sequence[int],
    directed: bool = True,
    edge_count: Optional[int] = None,
    weights_path: Optional[os.PathLike[str] | str] = None,
    weight_dtype: str = "float64",
    orientation: Optional[str] = None,
    weight_key: Optional[str] = None,
    graph_id: str = "legacy-int32-csr",
    generation: int = 0,
    sort_chunk_values: int = _DEFAULT_SORT_CHUNK_VALUES,
    merge_fan_in: int = _DEFAULT_MERGE_FAN_IN,
    balance: str = "vertices",
    trust_undirected_input: bool = False,
) -> PartitionBuildReport:
    """Stream an existing EGGPU signed-int32 CSR into multi-card shards.

    This compatibility path avoids expanding a valid single-card CSR to an
    in-memory uint64 representation.  Global counters and output manifest
    identities are still uint64; only the input offsets/indices use the
    legacy signed-int32 ABI.
    """

    if not isinstance(directed, bool):
        raise TypeError("directed must be a boolean")
    if not isinstance(trust_undirected_input, bool):
        raise TypeError("trust_undirected_input must be a boolean")
    _range_partitions(vertex_count, devices)
    indices = Path(indices_path).resolve(strict=True)
    if indices.stat().st_size % _I32.size:
        raise PartitionBuildError("host CSR indices are not int32-aligned")
    adjacency_count = indices.stat().st_size // _I32.size
    if edge_count is None:
        if not directed:
            raise ValueError(
                "edge_count is required for an undirected materialized host CSR"
            )
        logical_edge_count = adjacency_count
    else:
        if isinstance(edge_count, bool) or not isinstance(edge_count, int):
            raise TypeError("edge_count must be an integer")
        if edge_count < 0 or edge_count > UINT64_MAX:
            raise ValueError("edge_count must fit uint64")
        logical_edge_count = edge_count
    if weights_path is not None:
        _weight_struct(weight_dtype)
    view_orientation, normalized_weight_key = _normalize_view(
        directed, weights_path is not None, orientation, weight_key
    )
    source = _HostCSRSource(
        Path(offsets_path),
        indices,
        vertex_count=vertex_count,
        edge_count=logical_edge_count,
        weights_path=None if weights_path is None else Path(weights_path),
        weight_dtype=weight_dtype,
        integer_dtype="int32",
    )
    if balance not in {"vertices", "adjacency", "incoming_adjacency"}:
        raise ValueError("balance must be 'vertices', 'adjacency', or 'incoming_adjacency'")
    range_plan = (
        _adjacency_balanced_ranges(
            Path(offsets_path),
            vertex_count=vertex_count,
            adjacency_count=adjacency_count,
            devices=devices,
            integer_struct=_I32,
        )
        if balance == "adjacency"
        else None
    )
    if balance == "incoming_adjacency":
        range_plan = _incoming_adjacency_balanced_ranges(
            indices, vertex_count=vertex_count, devices=devices, integer_struct=_I32
        )
    return _build_range_partitioned_csr(
        output_root,
        source,
        vertex_count=vertex_count,
        devices=devices,
        directed=directed,
        view_orientation=view_orientation,
        weight_key=normalized_weight_key,
        graph_id=graph_id,
        generation=generation,
        sort_chunk_values=sort_chunk_values,
        merge_fan_in=merge_fan_in,
        ranges=None if range_plan is None else range_plan.ranges,
        boundary_offsets_sha256=(
            None if range_plan is None else range_plan.offsets_sha256
        ),
        validate_undirected_host=(
            not directed and not trust_undirected_input
        ),
    )


__all__ = [
    "PartitionBuildError",
    "PartitionBuildReport",
    "build_partitioned_csr_from_edge_list",
    "build_partitioned_csr_from_host_csr",
    "build_partitioned_csr_from_int32_csr",
]
