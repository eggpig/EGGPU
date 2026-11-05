"""Strict, stream-oriented EGMGPU partitioned-CSR manifest v1.

The manifest describes immutable binary CSR shards.  Loading validates JSON
shape, integer domains, path containment, file sizes, SHA-256 checksums, CSR
contents, ownership coverage, halo mappings, and partition owners without
materializing the graph's edge arrays in Python memory.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import stat
import struct

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple

from .state import GraphSnapshot
from .state import MAX_LOCAL_ADJACENCY_ENTRIES
from .state import PartitionMetadata
from .state import ViewKey
from .state import validate_partition_set


MANIFEST_FORMAT = "egmgpu-partitioned-csr"
MANIFEST_VERSION = 1
UINT64_MAX = (1 << 64) - 1
INT32_MAX = (1 << 31) - 1
SHA256_HEX_LENGTH = 64
_CHECKSUM_CHUNK_BYTES = 1 << 20
_ARRAY_CHUNK_VALUES = 1 << 15
_DTYPES = {
    "int32": (4, "i"),
    "int64": (8, "q"),
    "uint64": (8, "Q"),
    "float32": (4, "f"),
    "float64": (8, "d"),
}
_VIEW_ORIENTATIONS = {"outgoing", "incoming", "undirected"}


class ManifestError(ValueError):
    """Base class for malformed or inconsistent manifests."""


class ManifestPathError(ManifestError):
    """A referenced file is not contained by the manifest root."""


class ManifestFileError(ManifestError):
    """A referenced binary file fails size or checksum validation."""


class ManifestContentError(ManifestError):
    """Binary array contents violate the partitioned-CSR contract."""


def _require_object(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{context} must be a JSON object")
    return value


def _require_keys(
    value: Mapping[str, Any],
    required: Iterable[str],
    optional: Iterable[str],
    context: str,
) -> None:
    required_set = set(required)
    optional_set = set(optional)
    missing = required_set - set(value)
    unknown = set(value) - required_set - optional_set
    if missing:
        raise ManifestError(f"{context} is missing fields {sorted(missing)}")
    if unknown:
        raise ManifestError(f"{context} has unknown fields {sorted(unknown)}")


def _integer(
    value: Any, context: str, *, maximum: int = UINT64_MAX
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"{context} must be an integer")
    if value < 0 or value > maximum:
        raise ManifestError(f"{context} must be in [0, {maximum}]")
    return value


def _boolean(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise ManifestError(f"{context} must be a boolean")
    return value


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{context} must be a non-empty string")
    return value


def _sha256(value: Any, context: str) -> str:
    digest = _string(value, context)
    if len(digest) != SHA256_HEX_LENGTH:
        raise ManifestError(f"{context} must contain 64 lowercase hex characters")
    if digest.lower() != digest or any(ch not in "0123456789abcdef" for ch in digest):
        raise ManifestError(f"{context} must contain 64 lowercase hex characters")
    return digest


@dataclass(frozen=True)
class ArrayFile:
    """One checksummed, little-endian binary array referenced by a manifest."""

    relative_path: str
    dtype: str
    count: int
    byte_size: int
    sha256: str

    def __post_init__(self) -> None:
        if self.dtype not in _DTYPES:
            raise ManifestError(f"unsupported array dtype {self.dtype!r}")
        _integer(self.count, "array count")
        _integer(self.byte_size, "array byte_size")
        _sha256(self.sha256, "array sha256")
        expected = self.count * self.item_size
        if self.byte_size != expected:
            raise ManifestFileError(
                f"{self.relative_path}: byte_size {self.byte_size} does not equal "
                f"count*dtype_size ({expected})"
            )

    @property
    def item_size(self) -> int:
        return _DTYPES[self.dtype][0]

    @property
    def struct_code(self) -> str:
        return _DTYPES[self.dtype][1]

    def resolve(self, manifest_root: Path) -> Path:
        raw = Path(self.relative_path)
        if raw.is_absolute() or not raw.parts:
            raise ManifestPathError(
                f"array path must be relative to the manifest root: {self.relative_path!r}"
            )
        if "\\" in self.relative_path:
            raise ManifestPathError(
                "manifest v1 array paths use forward-slash relative components"
            )
        if any(part in {"", ".", ".."} for part in raw.parts):
            raise ManifestPathError(
                f"array path contains a forbidden component: {self.relative_path!r}"
            )
        root = manifest_root.resolve(strict=True)
        try:
            resolved = (root / raw).resolve(strict=True)
        except FileNotFoundError as exc:
            raise ManifestFileError(
                f"referenced array does not exist: {self.relative_path}"
            ) from exc
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ManifestPathError(
                f"array path escapes the manifest root: {self.relative_path!r}"
            ) from exc
        if not stat.S_ISREG(resolved.stat().st_mode):
            raise ManifestFileError(
                f"referenced array is not a regular file: {self.relative_path}"
            )
        return resolved

    def validate_file(self, manifest_root: Path, verify_checksum: bool = True) -> Path:
        path = self.resolve(manifest_root)
        actual_size = path.stat().st_size
        if actual_size != self.byte_size:
            raise ManifestFileError(
                f"{self.relative_path}: file size {actual_size} does not match "
                f"declared byte_size {self.byte_size}"
            )
        if verify_checksum:
            actual_digest = _hash_file(path)
            if actual_digest != self.sha256:
                raise ManifestFileError(
                    f"{self.relative_path}: SHA-256 mismatch "
                    f"(declared {self.sha256}, observed {actual_digest})"
                )
        return path

    def values(self, manifest_root: Path) -> Iterator[Any]:
        path = self.resolve(manifest_root)
        item_size = self.item_size
        chunk_bytes = _ARRAY_CHUNK_VALUES * item_size
        unpacker = struct.Struct("<" + self.struct_code)
        observed = 0
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(chunk_bytes)
                if not chunk:
                    break
                if len(chunk) % item_size:
                    raise ManifestFileError(
                        f"{self.relative_path}: file is not aligned to {self.dtype}"
                    )
                for (value,) in struct.iter_unpack(unpacker.format, chunk):
                    observed += 1
                    yield value
        if observed != self.count:
            raise ManifestFileError(
                f"{self.relative_path}: read {observed} values, expected {self.count}"
            )


@dataclass(frozen=True)
class ManifestPartitionMetadata:
    """Partition metadata supporting range or explicit sorted ownership."""

    partition_id: int
    device_id: int
    ownership_encoding: str
    owned_vertex_count: int
    halo_vertex_count: int
    adjacency_count: int
    cut_edge_count: int
    offsets: ArrayFile
    indices: ArrayFile
    local_to_global: ArrayFile
    owners: ArrayFile
    weights: Optional[ArrayFile] = None
    owned_global_ids: Optional[ArrayFile] = None
    vertex_begin: Optional[int] = None
    vertex_end: Optional[int] = None

    def __post_init__(self) -> None:
        _integer(self.partition_id, "partition_id", maximum=INT32_MAX)
        _integer(self.device_id, "device_id", maximum=INT32_MAX)
        _integer(self.owned_vertex_count, "owned_vertex_count")
        _integer(self.halo_vertex_count, "halo_vertex_count")
        _integer(self.adjacency_count, "adjacency_count")
        _integer(self.cut_edge_count, "cut_edge_count")
        if self.owned_vertex_count == 0 and any(
            (
                self.halo_vertex_count,
                self.adjacency_count,
                self.cut_edge_count,
            )
        ):
            raise ManifestError(
                "an empty partition must have zero halo, adjacency, and cut-edge counts"
            )
        if self.offsets.dtype not in {"int32", "int64"}:
            raise ManifestError("offsets must have dtype int32 or int64")
        max_entries = MAX_LOCAL_ADJACENCY_ENTRIES if self.offsets.dtype == "int32" else (1 << 63) - 1
        if self.adjacency_count > max_entries:
            raise ManifestError(
                "partition adjacency count exceeds its signed offset dtype"
            )
        if self.cut_edge_count > self.adjacency_count:
            raise ManifestError("cut_edge_count cannot exceed adjacency_count")
        if self.local_vertex_count > (1 << 31):
            raise ManifestError(
                "owned plus halo vertices exceed the signed int32 local-ID domain"
            )
        if self.ownership_encoding == "range":
            if self.vertex_begin is None or self.vertex_end is None:
                raise ManifestError("range ownership requires vertex_begin and vertex_end")
            if self.vertex_end < self.vertex_begin:
                raise ManifestError("range ownership has an invalid half-open interval")
            if self.vertex_end - self.vertex_begin != self.owned_vertex_count:
                raise ManifestError(
                    "range length does not equal owned_vertex_count"
                )
            if self.owned_global_ids is not None:
                raise ManifestError("range ownership must not provide owned_global_ids")
        elif self.ownership_encoding == "explicit":
            if self.vertex_begin is not None or self.vertex_end is not None:
                raise ManifestError(
                    "explicit ownership must not provide a contiguous range"
                )
            if self.owned_global_ids is None:
                raise ManifestError("explicit ownership requires owned_global_ids")
        else:
            raise ManifestError(
                f"unknown ownership encoding {self.ownership_encoding!r}"
            )

        expected = {
            "offsets": (self.offsets, self.offsets.dtype, self.owned_vertex_count + 1),
            "indices": (self.indices, "int32", self.adjacency_count),
            "local_to_global": (
                self.local_to_global,
                "uint64",
                self.local_vertex_count,
            ),
            "owners": (self.owners, "int32", self.local_vertex_count),
        }
        if self.owned_global_ids is not None:
            expected["owned_global_ids"] = (
                self.owned_global_ids,
                "uint64",
                self.owned_vertex_count,
            )
        if self.weights is not None:
            if self.weights.dtype not in {"float32", "float64"}:
                raise ManifestError("weights must use float32 or float64")
            if self.weights.count != self.adjacency_count:
                raise ManifestError("weights count must equal adjacency_count")
        for name, (array, dtype, count) in expected.items():
            if array.dtype != dtype or array.count != count:
                raise ManifestError(
                    f"{name} must have dtype={dtype} and count={count}"
                )

    @property
    def local_vertex_count(self) -> int:
        return self.owned_vertex_count + self.halo_vertex_count

    def to_partition_metadata(self) -> PartitionMetadata:
        """Convert only the contiguous-owner representation used by kernels."""

        if self.ownership_encoding != "range":
            raise ManifestError(
                "explicit ownership has no lossless contiguous PartitionMetadata; "
                "the current planner may use its size metadata, but a native owner "
                "kernel must consume ManifestPartitionMetadata directly"
            )
        assert self.vertex_begin is not None and self.vertex_end is not None
        return PartitionMetadata(
            partition_id=self.partition_id,
            device_id=self.device_id,
            vertex_begin=self.vertex_begin,
            vertex_end=self.vertex_end,
            adjacency_entries=self.adjacency_count,
            halo_vertices=self.halo_vertex_count,
            cut_edges=self.cut_edge_count,
        )

    def arrays(self) -> Mapping[str, ArrayFile]:
        result: Dict[str, ArrayFile] = {
            "offsets": self.offsets,
            "indices": self.indices,
            "local_to_global": self.local_to_global,
            "owners": self.owners,
        }
        if self.weights is not None:
            result["weights"] = self.weights
        if self.owned_global_ids is not None:
            result["owned_global_ids"] = self.owned_global_ids
        return result


@dataclass(frozen=True)
class ManifestValidationReport:
    partition_count: int
    file_count: int
    checked_bytes: int
    vertex_count: int
    edge_count: int
    adjacency_count: int
    ownership_encoding: str
    weighted: bool


@dataclass(frozen=True)
class PartitionedCSRManifest:
    """Loaded and structurally validated manifest v1."""

    manifest_path: Path
    graph_id: str
    generation: int
    directed: bool
    weighted: bool
    view_orientation: str
    weight_key: Optional[str]
    view_signature: str
    vertex_count: int
    edge_count: int
    adjacency_count: int
    topology_sha256: str
    weight_sha256: Optional[str]
    ownership_encoding: str
    partitions: Tuple[ManifestPartitionMetadata, ...]

    def __post_init__(self) -> None:
        if self.view_orientation not in _VIEW_ORIENTATIONS:
            raise ManifestError(
                "view orientation must be 'outgoing', 'incoming', or 'undirected'"
            )
        if self.directed != (self.view_orientation != "undirected"):
            raise ManifestError(
                "directed graph views must be outgoing or incoming; "
                "undirected graph views must use undirected orientation"
            )
        if self.weight_key is not None:
            _string(self.weight_key, "view.weight_key")
        if not self.weighted and self.weight_key is not None:
            raise ManifestError("an unweighted view cannot declare a weight key")
        _sha256(self.view_signature, "view.signature")

    @property
    def root(self) -> Path:
        return self.manifest_path.parent.resolve(strict=True)

    def graph_snapshot(self) -> GraphSnapshot:
        return GraphSnapshot(
            graph_id=self.graph_id,
            generation=self.generation,
            vertex_count=self.vertex_count,
            adjacency_entries=self.adjacency_count,
            topology_signature=self.topology_sha256,
            weight_signature=self.weight_sha256,
        )

    def view_key(self) -> ViewKey:
        """Return the exact semantic view identity used by reusable state."""

        return ViewKey.for_snapshot(self.graph_snapshot(), self.view_signature)

    def range_partition_metadata(self) -> Tuple[PartitionMetadata, ...]:
        if self.ownership_encoding != "range":
            raise ManifestError(
                "explicit ownership cannot be converted to contiguous "
                "PartitionMetadata"
            )
        converted = tuple(part.to_partition_metadata() for part in self.partitions)
        validate_partition_set(
            self.graph_snapshot(),
            converted,
            tuple(part.device_id for part in self.partitions),
        )
        return converted

    def validate(
        self, *, verify_checksums: bool = True, validate_arrays: bool = True
    ) -> ManifestValidationReport:
        """Validate all referenced arrays without loading full CSR arrays."""

        seen_paths: Dict[Path, str] = {}
        checked_bytes = 0
        file_count = 0
        for partition in self.partitions:
            for name, array in partition.arrays().items():
                resolved = array.validate_file(self.root, verify_checksums)
                previous = seen_paths.get(resolved)
                if previous is not None:
                    raise ManifestFileError(
                        f"array file {array.relative_path!r} is reused by "
                        f"{previous} and partition {partition.partition_id}/{name}"
                    )
                seen_paths[resolved] = f"partition {partition.partition_id}/{name}"
                checked_bytes += array.byte_size
                file_count += 1
        if validate_arrays:
            self._validate_array_contents()
        return ManifestValidationReport(
            partition_count=len(self.partitions),
            file_count=file_count,
            checked_bytes=checked_bytes,
            vertex_count=self.vertex_count,
            edge_count=self.edge_count,
            adjacency_count=self.adjacency_count,
            ownership_encoding=self.ownership_encoding,
            weighted=self.weighted,
        )

    def _validate_array_contents(self) -> None:
        self._validate_owned_coverage()
        owner_lookup = _OwnerLookup(self)
        for partition in self.partitions:
            _validate_offsets(partition, self.root)
            _validate_indices(partition, self.root)
            _validate_vertex_map(partition, self, owner_lookup)
            if partition.weights is not None:
                _validate_weights(partition, self.root)

    def _validate_owned_coverage(self) -> None:
        if self.ownership_encoding == "range":
            expected = 0
            for partition in self.partitions:
                assert partition.vertex_begin is not None
                assert partition.vertex_end is not None
                if partition.vertex_begin != expected:
                    raise ManifestContentError(
                        "range ownership must cover [0, vertex_count) exactly "
                        "without gaps or overlap"
                    )
                expected = partition.vertex_end
            if expected != self.vertex_count:
                raise ManifestContentError(
                    "range ownership does not cover the global vertex domain"
                )
            return

        streams = []
        for partition in self.partitions:
            assert partition.owned_global_ids is not None
            stream = _strictly_increasing_ids(
                partition.owned_global_ids,
                self.root,
                self.vertex_count,
                f"partition {partition.partition_id} owned_global_ids",
            )
            streams.append(stream)
        merged = heapq.merge(*streams)
        expected = 0
        for observed in merged:
            if observed != expected:
                if observed < expected:
                    raise ManifestContentError(
                        f"explicit ownership duplicates global vertex {observed}"
                    )
                raise ManifestContentError(
                    f"explicit ownership is missing global vertex {expected}"
                )
            expected += 1
        if expected != self.vertex_count:
            raise ManifestContentError(
                f"explicit ownership is missing global vertex {expected}"
            )


class _OwnerLookup:
    """Memory-bounded global vertex owner lookup."""

    def __init__(self, manifest: PartitionedCSRManifest) -> None:
        self._manifest = manifest
        self._ranges = tuple(
            (
                int(part.vertex_begin),
                int(part.vertex_end),
                part.partition_id,
            )
            for part in manifest.partitions
            if part.ownership_encoding == "range"
        )

    def owner(self, global_id: int) -> int:
        if self._manifest.ownership_encoding == "range":
            low = 0
            high = len(self._ranges)
            while low < high:
                middle = (low + high) // 2
                begin, end, owner = self._ranges[middle]
                if global_id < begin:
                    high = middle
                elif global_id >= end:
                    low = middle + 1
                else:
                    return owner
            raise ManifestContentError(f"global vertex {global_id} has no owner")

        for partition in self._manifest.partitions:
            assert partition.owned_global_ids is not None
            if _binary_search_uint64(
                partition.owned_global_ids.resolve(self._manifest.root),
                partition.owned_global_ids.count,
                global_id,
            ):
                return partition.partition_id
        raise ManifestContentError(f"global vertex {global_id} has no owner")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(_CHECKSUM_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _strictly_increasing_ids(
    array: ArrayFile,
    root: Path,
    vertex_count: int,
    context: str,
) -> Iterator[int]:
    previous = -1
    for value in array.values(root):
        if value >= vertex_count:
            raise ManifestContentError(
                f"{context} contains out-of-range global vertex {value}"
            )
        if value <= previous:
            raise ManifestContentError(f"{context} must be strictly increasing")
        previous = value
        yield value


def _validate_offsets(partition: ManifestPartitionMetadata, root: Path) -> None:
    previous = None
    last = None
    for index, value in enumerate(partition.offsets.values(root)):
        if value < 0:
            raise ManifestContentError(
                f"partition {partition.partition_id} offsets contain a negative value"
            )
        if index == 0 and value != 0:
            raise ManifestContentError(
                f"partition {partition.partition_id} offsets must start at zero"
            )
        if previous is not None and value < previous:
            raise ManifestContentError(
                f"partition {partition.partition_id} offsets are not monotone"
            )
        previous = value
        last = value
    if last != partition.adjacency_count:
        raise ManifestContentError(
            f"partition {partition.partition_id} final offset {last} does not "
            f"equal adjacency_count {partition.adjacency_count}"
        )


def _validate_indices(partition: ManifestPartitionMetadata, root: Path) -> None:
    for value in partition.indices.values(root):
        if value < 0 or value >= partition.local_vertex_count:
            raise ManifestContentError(
                f"partition {partition.partition_id} contains local index {value} "
                f"outside [0, {partition.local_vertex_count})"
            )


def _validate_vertex_map(
    partition: ManifestPartitionMetadata,
    manifest: PartitionedCSRManifest,
    owner_lookup: _OwnerLookup,
) -> None:
    globals_iter = partition.local_to_global.values(manifest.root)
    owners_iter = partition.owners.values(manifest.root)
    explicit_iter: Optional[Iterator[int]] = None
    if partition.owned_global_ids is not None:
        explicit_iter = partition.owned_global_ids.values(manifest.root)

    halo_previous = -1
    for local_id, (global_id, owner) in enumerate(zip(globals_iter, owners_iter)):
        if global_id >= manifest.vertex_count:
            raise ManifestContentError(
                f"partition {partition.partition_id} local_to_global contains "
                f"out-of-range vertex {global_id}"
            )
        if local_id < partition.owned_vertex_count:
            if partition.ownership_encoding == "range":
                assert partition.vertex_begin is not None
                expected = partition.vertex_begin + local_id
            else:
                assert explicit_iter is not None
                expected = next(explicit_iter)
            if global_id != expected:
                raise ManifestContentError(
                    f"partition {partition.partition_id} owned prefix disagrees "
                    "with its ownership descriptor"
                )
            if owner != partition.partition_id:
                raise ManifestContentError(
                    f"partition {partition.partition_id} owned vertex {global_id} "
                    f"declares owner {owner}"
                )
            continue

        if partition.ownership_encoding == "range":
            assert partition.vertex_begin is not None
            assert partition.vertex_end is not None
            repeats_owned = partition.vertex_begin <= global_id < partition.vertex_end
        else:
            assert partition.owned_global_ids is not None
            repeats_owned = _binary_search_uint64(
                partition.owned_global_ids.resolve(manifest.root),
                partition.owned_global_ids.count,
                global_id,
            )
        if repeats_owned:
            raise ManifestContentError(
                f"partition {partition.partition_id} halo repeats owned vertex {global_id}"
            )
        if global_id <= halo_previous:
            raise ManifestContentError(
                f"partition {partition.partition_id} halo global IDs must be "
                "strictly increasing"
            )
        halo_previous = global_id
        actual_owner = owner_lookup.owner(global_id)
        if owner != actual_owner:
            raise ManifestContentError(
                f"partition {partition.partition_id} halo vertex {global_id} "
                f"declares owner {owner}, expected {actual_owner}"
            )


def _validate_weights(partition: ManifestPartitionMetadata, root: Path) -> None:
    assert partition.weights is not None
    for value in partition.weights.values(root):
        if not math.isfinite(value):
            raise ManifestContentError(
                f"partition {partition.partition_id} weights must be finite"
            )


def _binary_search_uint64(path: Path, count: int, target: int) -> bool:
    item = struct.Struct("<Q")
    low = 0
    high = count
    with path.open("rb") as stream:
        while low < high:
            middle = (low + high) // 2
            stream.seek(middle * item.size)
            raw = stream.read(item.size)
            if len(raw) != item.size:
                raise ManifestFileError(f"short read while searching {path.name}")
            value = item.unpack(raw)[0]
            if value < target:
                low = middle + 1
            elif value > target:
                high = middle
            else:
                return True
    return False


def _parse_array(value: Any, context: str) -> ArrayFile:
    obj = _require_object(value, context)
    _require_keys(
        obj,
        {"path", "dtype", "count", "byte_size", "sha256"},
        set(),
        context,
    )
    return ArrayFile(
        relative_path=_string(obj["path"], f"{context}.path"),
        dtype=_string(obj["dtype"], f"{context}.dtype"),
        count=_integer(obj["count"], f"{context}.count"),
        byte_size=_integer(obj["byte_size"], f"{context}.byte_size"),
        sha256=_sha256(obj["sha256"], f"{context}.sha256"),
    )


def _parse_partition(
    value: Any, index: int, ownership_encoding: str
) -> ManifestPartitionMetadata:
    context = f"partitions[{index}]"
    obj = _require_object(value, context)
    _require_keys(
        obj,
        {
            "partition_id",
            "device_id",
            "owned_vertex_count",
            "halo_vertex_count",
            "adjacency_count",
            "cut_edge_count",
            "ownership",
            "files",
        },
        set(),
        context,
    )
    ownership = _require_object(obj["ownership"], f"{context}.ownership")
    if ownership_encoding == "range":
        _require_keys(
            ownership,
            {"encoding", "vertex_begin", "vertex_end"},
            set(),
            f"{context}.ownership",
        )
        vertex_begin = _integer(
            ownership["vertex_begin"], f"{context}.ownership.vertex_begin"
        )
        vertex_end = _integer(
            ownership["vertex_end"], f"{context}.ownership.vertex_end"
        )
    else:
        _require_keys(
            ownership, {"encoding"}, set(), f"{context}.ownership"
        )
        vertex_begin = None
        vertex_end = None
    encoding = _string(ownership["encoding"], f"{context}.ownership.encoding")
    if encoding != ownership_encoding:
        raise ManifestError(
            f"{context} ownership encoding {encoding!r} disagrees with partitioning"
        )

    files = _require_object(obj["files"], f"{context}.files")
    required_files = {"offsets", "indices", "local_to_global", "owners"}
    if ownership_encoding == "explicit":
        required_files.add("owned_global_ids")
    _require_keys(
        files,
        required_files,
        {"weights"},
        f"{context}.files",
    )
    parsed_files = {
        name: _parse_array(file_value, f"{context}.files.{name}")
        for name, file_value in files.items()
    }
    return ManifestPartitionMetadata(
        partition_id=_integer(
            obj["partition_id"], f"{context}.partition_id", maximum=INT32_MAX
        ),
        device_id=_integer(
            obj["device_id"], f"{context}.device_id", maximum=INT32_MAX
        ),
        ownership_encoding=encoding,
        owned_vertex_count=_integer(
            obj["owned_vertex_count"], f"{context}.owned_vertex_count"
        ),
        halo_vertex_count=_integer(
            obj["halo_vertex_count"], f"{context}.halo_vertex_count"
        ),
        adjacency_count=_integer(
            obj["adjacency_count"], f"{context}.adjacency_count"
        ),
        cut_edge_count=_integer(
            obj["cut_edge_count"], f"{context}.cut_edge_count"
        ),
        offsets=parsed_files["offsets"],
        indices=parsed_files["indices"],
        local_to_global=parsed_files["local_to_global"],
        owners=parsed_files["owners"],
        weights=parsed_files.get("weights"),
        owned_global_ids=parsed_files.get("owned_global_ids"),
        vertex_begin=vertex_begin,
        vertex_end=vertex_end,
    )


def load_partition_manifest(
    manifest_path: os.PathLike[str] | str,
    *,
    verify_checksums: bool = True,
    validate_arrays: bool = True,
) -> PartitionedCSRManifest:
    """Load and validate an EGMGPU partitioned-CSR manifest v1."""

    path = Path(manifest_path).resolve(strict=True)
    if not path.is_file():
        raise ManifestFileError(f"manifest is not a regular file: {path}")
    try:
        with path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"manifest is not valid UTF-8 JSON: {path}") from exc
    root = _require_object(document, "manifest")
    _require_keys(
        root,
        {
            "format",
            "version",
            "graph",
            "view",
            "indexing",
            "partitioning",
            "partitions",
        },
        set(),
        "manifest",
    )
    if not isinstance(root["format"], str) or root["format"] != MANIFEST_FORMAT:
        raise ManifestError(f"unsupported manifest format {root['format']!r}")
    if (
        isinstance(root["version"], bool)
        or not isinstance(root["version"], int)
        or root["version"] != MANIFEST_VERSION
    ):
        raise ManifestError(f"unsupported manifest version {root['version']!r}")

    graph = _require_object(root["graph"], "graph")
    _require_keys(
        graph,
        {
            "id",
            "generation",
            "directed",
            "weighted",
            "vertex_count",
            "edge_count",
            "adjacency_count",
            "topology_sha256",
        },
        {"weight_sha256"},
        "graph",
    )
    view = _require_object(root["view"], "view")
    _require_keys(view, {"orientation", "signature"}, {"weight_key"}, "view")
    view_orientation = _string(view["orientation"], "view.orientation")
    if view_orientation not in _VIEW_ORIENTATIONS:
        raise ManifestError(
            "view.orientation must be 'outgoing', 'incoming', or 'undirected'"
        )
    raw_weight_key = view.get("weight_key")
    weight_key = (
        None if raw_weight_key is None else _string(raw_weight_key, "view.weight_key")
    )
    parsed_view_signature = _sha256(view["signature"], "view.signature")
    indexing = _require_object(root["indexing"], "indexing")
    _require_keys(
        indexing,
        {"offset_dtype", "index_dtype", "global_vertex_dtype", "endianness"},
        set(),
        "indexing",
    )
    if indexing["offset_dtype"] not in {"int32", "int64"}:
        raise ManifestError("offset_dtype must be int32 or int64")
    expected_indexing = {
        "offset_dtype": indexing["offset_dtype"],
        "index_dtype": "int32",
        "global_vertex_dtype": "uint64",
        "endianness": "little",
    }
    if dict(indexing) != expected_indexing:
        raise ManifestError(f"manifest v1 indexing must equal {expected_indexing}")

    partitioning = _require_object(root["partitioning"], "partitioning")
    _require_keys(
        partitioning, {"ownership_encoding"}, set(), "partitioning"
    )
    ownership_encoding = _string(
        partitioning["ownership_encoding"], "partitioning.ownership_encoding"
    )
    if ownership_encoding not in {"range", "explicit"}:
        raise ManifestError("ownership_encoding must be 'range' or 'explicit'")
    raw_partitions = root["partitions"]
    if not isinstance(raw_partitions, list) or not raw_partitions:
        raise ManifestError("partitions must be a non-empty JSON array")
    partitions = tuple(
        _parse_partition(value, index, ownership_encoding)
        for index, value in enumerate(raw_partitions)
    )

    if any(part.offsets.dtype != indexing["offset_dtype"] for part in partitions):
        raise ManifestError("partition offsets dtype differs from indexing.offset_dtype")

    directed = _boolean(graph["directed"], "graph.directed")
    weighted = _boolean(graph["weighted"], "graph.weighted")
    if directed != (view_orientation != "undirected"):
        raise ManifestError(
            "graph.directed is inconsistent with view.orientation"
        )
    if not weighted and weight_key is not None:
        raise ManifestError("an unweighted view cannot declare view.weight_key")
    weight_sha = graph.get("weight_sha256")
    if weighted:
        if weight_sha is None:
            raise ManifestError("weighted graph requires graph.weight_sha256")
        parsed_weight_sha = _sha256(weight_sha, "graph.weight_sha256")
        if any(part.weights is None for part in partitions):
            raise ManifestError("weighted graph requires weights in every partition")
    else:
        if weight_sha is not None:
            raise ManifestError("unweighted graph must not declare weight_sha256")
        parsed_weight_sha = None
        if any(part.weights is not None for part in partitions):
            raise ManifestError("unweighted graph must not contain partition weights")

    partition_ids = tuple(part.partition_id for part in partitions)
    if partition_ids != tuple(range(len(partitions))):
        raise ManifestError("partition IDs must be ordered and contiguous from zero")
    device_ids = tuple(part.device_id for part in partitions)
    if len(set(device_ids)) != len(device_ids):
        raise ManifestError("each manifest partition must target a distinct device")
    vertex_count = _integer(graph["vertex_count"], "graph.vertex_count")
    edge_count = _integer(graph["edge_count"], "graph.edge_count")
    adjacency_count = _integer(
        graph["adjacency_count"], "graph.adjacency_count"
    )
    if edge_count > adjacency_count:
        raise ManifestError("graph.edge_count cannot exceed graph.adjacency_count")
    if sum(part.owned_vertex_count for part in partitions) != vertex_count:
        raise ManifestError("partition owned_vertex_count values do not sum to graph total")
    if sum(part.adjacency_count for part in partitions) != adjacency_count:
        raise ManifestError("partition adjacency_count values do not sum to graph total")
    if weighted and len({part.weights.dtype for part in partitions}) != 1:
        raise ManifestError("all partition weight arrays must use the same dtype")

    manifest = PartitionedCSRManifest(
        manifest_path=path,
        graph_id=_string(graph["id"], "graph.id"),
        generation=_integer(graph["generation"], "graph.generation"),
        directed=directed,
        weighted=weighted,
        view_orientation=view_orientation,
        weight_key=weight_key,
        view_signature=parsed_view_signature,
        vertex_count=vertex_count,
        edge_count=edge_count,
        adjacency_count=adjacency_count,
        topology_sha256=_sha256(
            graph["topology_sha256"], "graph.topology_sha256"
        ),
        weight_sha256=parsed_weight_sha,
        ownership_encoding=ownership_encoding,
        partitions=partitions,
    )
    topology_digest = _topology_signature(manifest)
    if manifest.topology_sha256 != topology_digest:
        raise ManifestError(
            "graph.topology_sha256 does not match manifest topology metadata"
        )
    view_digest = _view_signature(manifest)
    if manifest.view_signature != view_digest:
        raise ManifestError(
            "view.signature does not match graph and view metadata"
        )
    if manifest.weighted:
        weight_digest = _weight_signature(manifest)
        if manifest.weight_sha256 != weight_digest:
            raise ManifestError(
                "graph.weight_sha256 does not match partition weight metadata"
            )
    manifest.validate(
        verify_checksums=verify_checksums, validate_arrays=validate_arrays
    )
    return manifest


def _array_signature(array: ArrayFile) -> Mapping[str, Any]:
    return {
        "dtype": array.dtype,
        "count": array.count,
        "sha256": array.sha256,
    }


def _topology_payload(manifest: PartitionedCSRManifest) -> Mapping[str, Any]:
    partitions = []
    for partition in manifest.partitions:
        ownership: Dict[str, Any] = {"encoding": partition.ownership_encoding}
        if partition.ownership_encoding == "range":
            ownership.update(
                {
                    "vertex_begin": partition.vertex_begin,
                    "vertex_end": partition.vertex_end,
                }
            )
        files = {
            name: _array_signature(array)
            for name, array in partition.arrays().items()
            if name != "weights"
        }
        partitions.append(
            {
                "partition_id": partition.partition_id,
                "owned_vertex_count": partition.owned_vertex_count,
                "halo_vertex_count": partition.halo_vertex_count,
                "adjacency_count": partition.adjacency_count,
                "cut_edge_count": partition.cut_edge_count,
                "ownership": ownership,
                "files": files,
            }
        )
    return {
        "format": MANIFEST_FORMAT,
        "version": MANIFEST_VERSION,
        "directed": manifest.directed,
        "view": {
            "orientation": manifest.view_orientation,
            "weight_key": manifest.weight_key,
        },
        "vertex_count": manifest.vertex_count,
        "edge_count": manifest.edge_count,
        "adjacency_count": manifest.adjacency_count,
        "ownership_encoding": manifest.ownership_encoding,
        "partitions": partitions,
    }


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _topology_signature(manifest: PartitionedCSRManifest) -> str:
    return _canonical_digest(_topology_payload(manifest))


def _weight_signature(manifest: PartitionedCSRManifest) -> str:
    return _canonical_digest(
        {
            "weights": [
                {
                    "partition_id": partition.partition_id,
                    "array": _array_signature(partition.weights),
                }
                for partition in manifest.partitions
                if partition.weights is not None
            ]
        }
    )


def _view_signature(manifest: PartitionedCSRManifest) -> str:
    return _canonical_digest(
        {
            "orientation": manifest.view_orientation,
            "weight_key": manifest.weight_key,
            "topology_sha256": manifest.topology_sha256,
            "weight_sha256": manifest.weight_sha256,
        }
    )


def _file_document(array: ArrayFile) -> Mapping[str, Any]:
    return {
        "path": array.relative_path,
        "dtype": array.dtype,
        "count": array.count,
        "byte_size": array.byte_size,
        "sha256": array.sha256,
    }


def _manifest_document(manifest: PartitionedCSRManifest) -> Mapping[str, Any]:
    graph: Dict[str, Any] = {
        "id": manifest.graph_id,
        "generation": manifest.generation,
        "directed": manifest.directed,
        "weighted": manifest.weighted,
        "vertex_count": manifest.vertex_count,
        "edge_count": manifest.edge_count,
        "adjacency_count": manifest.adjacency_count,
        "topology_sha256": manifest.topology_sha256,
    }
    if manifest.weight_sha256 is not None:
        graph["weight_sha256"] = manifest.weight_sha256
    partitions = []
    for partition in manifest.partitions:
        ownership: Dict[str, Any] = {"encoding": partition.ownership_encoding}
        if partition.ownership_encoding == "range":
            ownership.update(
                {
                    "vertex_begin": partition.vertex_begin,
                    "vertex_end": partition.vertex_end,
                }
            )
        partitions.append(
            {
                "partition_id": partition.partition_id,
                "device_id": partition.device_id,
                "owned_vertex_count": partition.owned_vertex_count,
                "halo_vertex_count": partition.halo_vertex_count,
                "adjacency_count": partition.adjacency_count,
                "cut_edge_count": partition.cut_edge_count,
                "ownership": ownership,
                "files": {
                    name: _file_document(array)
                    for name, array in partition.arrays().items()
                },
            }
        )
    return {
        "format": MANIFEST_FORMAT,
        "version": MANIFEST_VERSION,
        "graph": graph,
        "view": {
            "orientation": manifest.view_orientation,
            "weight_key": manifest.weight_key,
            "signature": manifest.view_signature,
        },
        "indexing": {
            "offset_dtype": manifest.partitions[0].offsets.dtype,
            "index_dtype": "int32",
            "global_vertex_dtype": "uint64",
            "endianness": "little",
        },
        "partitioning": {"ownership_encoding": manifest.ownership_encoding},
        "partitions": partitions,
    }


def _safe_output_path(root: Path, relative_path: str) -> Path:
    root = root.resolve(strict=True)
    raw = Path(relative_path)
    if raw.is_absolute() or not raw.parts or "\\" in relative_path:
        raise ManifestPathError("output array path must be a relative POSIX path")
    if any(part in {"", ".", ".."} for part in raw.parts):
        raise ManifestPathError("output array path contains a forbidden component")
    current = root
    for component in raw.parts[:-1]:
        candidate = current / component
        if not candidate.exists():
            candidate.mkdir()
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ManifestPathError("output array path escapes its root") from exc
        if not resolved.is_dir():
            raise ManifestPathError("output array parent is not a directory")
        current = resolved
    path = current / raw.name
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite array file {relative_path!r}")
    return path


def write_array_stream(
    root: os.PathLike[str] | str,
    relative_path: str,
    dtype: str,
    values: Iterable[Any],
) -> ArrayFile:
    """Write a binary array incrementally and return its manifest descriptor."""

    if dtype not in _DTYPES:
        raise ManifestError(f"unsupported array dtype {dtype!r}")
    root_path = Path(root)
    if not root_path.exists() or not root_path.is_dir():
        raise ManifestPathError("array output root must be an existing directory")
    path = _safe_output_path(root_path, relative_path)
    packer = struct.Struct("<" + _DTYPES[dtype][1])
    digest = hashlib.sha256()
    count = 0
    try:
        with path.open("xb") as stream:
            buffer = bytearray()
            for value in values:
                encoded = packer.pack(value)
                buffer.extend(encoded)
                digest.update(encoded)
                count += 1
                if count > UINT64_MAX:
                    raise ManifestError("array count exceeds the uint64 domain")
                if len(buffer) >= _CHECKSUM_CHUNK_BYTES:
                    stream.write(buffer)
                    buffer.clear()
            if buffer:
                stream.write(buffer)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if path.exists():
            path.unlink()
        raise
    return ArrayFile(
        relative_path=relative_path,
        dtype=dtype,
        count=count,
        byte_size=path.stat().st_size,
        sha256=digest.hexdigest(),
    )


def _write_array(
    root: Path, relative_path: str, dtype: str, values: Sequence[Any]
) -> ArrayFile:
    return write_array_stream(root, relative_path, dtype, values)


def build_tiny_partition_fixture(
    root: os.PathLike[str] | str,
    *,
    ownership_encoding: str = "range",
    weighted: bool = True,
    orientation: str = "outgoing",
    weight_key: Optional[str] = None,
) -> Path:
    """Create a two-partition, six-vertex fixture for tests and examples.

    The destination must be absent or empty. Existing files are never
    overwritten. The returned manifest is fully validated before return.
    """

    if ownership_encoding not in {"range", "explicit"}:
        raise ValueError("ownership_encoding must be 'range' or 'explicit'")
    if orientation not in _VIEW_ORIENTATIONS:
        raise ValueError(
            "orientation must be 'outgoing', 'incoming', or 'undirected'"
        )
    if weight_key is not None and (not isinstance(weight_key, str) or not weight_key):
        raise ValueError("weight_key must be a non-empty string or None")
    if not weighted and weight_key is not None:
        raise ValueError("an unweighted fixture cannot declare a weight key")
    target = Path(root)
    if target.exists():
        if not target.is_dir() or any(target.iterdir()):
            raise FileExistsError("tiny fixture destination must be absent or empty")
    else:
        target.mkdir(parents=True)
    target = target.resolve(strict=True)

    adjacency = {
        0: ((1, 1.0), (3, 2.0)),
        1: ((2, 3.0),),
        2: ((4, 4.0),),
        3: ((4, 5.0),),
        4: ((5, 6.0),),
        5: ((0, 7.0),),
    }
    ownership = (
        ((0, 1, 2), (3, 4, 5))
        if ownership_encoding == "range"
        else ((0, 2, 4), (1, 3, 5))
    )
    owner_by_global = {
        global_id: partition_id
        for partition_id, owned in enumerate(ownership)
        for global_id in owned
    }
    partitions = []
    for partition_id, owned in enumerate(ownership):
        halo = tuple(
            sorted(
                {
                    neighbor
                    for source in owned
                    for neighbor, _weight in adjacency[source]
                    if owner_by_global[neighbor] != partition_id
                }
            )
        )
        local_to_global = tuple(owned) + halo
        local_by_global = {
            global_id: local_id
            for local_id, global_id in enumerate(local_to_global)
        }
        offsets = [0]
        indices = []
        weights = []
        cut_edges = 0
        for source in owned:
            for neighbor, weight in adjacency[source]:
                indices.append(local_by_global[neighbor])
                weights.append(weight)
                if owner_by_global[neighbor] != partition_id:
                    cut_edges += 1
            offsets.append(len(indices))

        prefix = f"parts/p{partition_id}"
        files: Dict[str, ArrayFile] = {
            "offsets": _write_array(
                target, f"{prefix}.offsets.i32", "int32", offsets
            ),
            "indices": _write_array(
                target, f"{prefix}.indices.i32", "int32", indices
            ),
            "local_to_global": _write_array(
                target,
                f"{prefix}.local_to_global.u64",
                "uint64",
                local_to_global,
            ),
            "owners": _write_array(
                target,
                f"{prefix}.owners.i32",
                "int32",
                [owner_by_global[global_id] for global_id in local_to_global],
            ),
        }
        if weighted:
            files["weights"] = _write_array(
                target, f"{prefix}.weights.f32", "float32", weights
            )
        if ownership_encoding == "explicit":
            files["owned_global_ids"] = _write_array(
                target,
                f"{prefix}.owned_global_ids.u64",
                "uint64",
                owned,
            )

        if ownership_encoding == "range":
            vertex_begin = owned[0]
            vertex_end = owned[-1] + 1
        else:
            vertex_begin = None
            vertex_end = None
        partitions.append(
            ManifestPartitionMetadata(
                partition_id=partition_id,
                device_id=partition_id,
                ownership_encoding=ownership_encoding,
                owned_vertex_count=len(owned),
                halo_vertex_count=len(halo),
                adjacency_count=len(indices),
                cut_edge_count=cut_edges,
                offsets=files["offsets"],
                indices=files["indices"],
                local_to_global=files["local_to_global"],
                owners=files["owners"],
                weights=files.get("weights"),
                owned_global_ids=files.get("owned_global_ids"),
                vertex_begin=vertex_begin,
                vertex_end=vertex_end,
            )
        )

    provisional = PartitionedCSRManifest(
        manifest_path=target / "manifest.json",
        graph_id=f"tiny-{ownership_encoding}",
        generation=0,
        directed=orientation != "undirected",
        weighted=weighted,
        view_orientation=orientation,
        weight_key=weight_key,
        view_signature="0" * 64,
        vertex_count=6,
        edge_count=7,
        adjacency_count=7,
        topology_sha256="0" * 64,
        weight_sha256="0" * 64 if weighted else None,
        ownership_encoding=ownership_encoding,
        partitions=tuple(partitions),
    )
    topology_sha = _topology_signature(provisional)
    weight_sha = _weight_signature(provisional) if weighted else None
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
        topology_sha256=topology_sha,
        weight_sha256=weight_sha,
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
    manifest_path = manifest.manifest_path
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(_manifest_document(manifest), stream, indent=2, sort_keys=True)
        stream.write("\n")
    load_partition_manifest(manifest_path)
    return manifest_path


__all__ = [
    "ArrayFile",
    "MANIFEST_FORMAT",
    "MANIFEST_VERSION",
    "ManifestContentError",
    "ManifestError",
    "ManifestFileError",
    "ManifestPartitionMetadata",
    "ManifestPathError",
    "ManifestValidationReport",
    "PartitionedCSRManifest",
    "build_tiny_partition_fixture",
    "load_partition_manifest",
    "write_array_stream",
]
