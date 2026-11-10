"""Immutable CSR-backed graph handle for large EGGPU inputs.

The regular EasyGraph classes intentionally keep Python node and adjacency
objects.  Those objects are useful for interactive graph mutation, but their
per-node and per-edge overhead is prohibitive for hundred-million-edge input.
This handle keeps only metadata in Python and loads a validated 32-bit CSR
directly into the native EGGPU graph container.
"""

from __future__ import annotations

import json
import hashlib
import os
import time
from pathlib import Path

import numpy as np


_INT32_LIMIT = (1 << 31) - 1
_FORMAT = "eggpu-csr-v1"
_PROJECTION_FORMAT = "eggpu-logical-undirected-projection-v1"


def _resolved_path(metadata_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = metadata_path.parent / path
    return path.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(16 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_undirected_projection(
    source_metadata_path: Path,
    projection_metadata_path,
    num_nodes: int,
):
    if projection_metadata_path is None:
        return None, None, None, 0
    projection_metadata_path = (
        Path(projection_metadata_path).expanduser().resolve()
    )
    with projection_metadata_path.open("r", encoding="utf-8") as handle:
        projection = json.load(handle)
    if projection.get("format") != _PROJECTION_FORMAT:
        raise ValueError(
            "unsupported EGGPU undirected projection format "
            f"{projection.get('format')!r}; expected {_PROJECTION_FORMAT!r}"
        )
    if int(projection["num_nodes"]) != num_nodes:
        raise ValueError(
            "undirected projection node count differs from the source CSR"
        )
    unique_edge_count = int(projection["unique_edge_count"])
    if unique_edge_count < 0 or unique_edge_count >= _INT32_LIMIT:
        raise ValueError(
            "undirected projection unique_edge_count exceeds int32"
        )
    if (
        projection.get("offset_dtype") != "int32"
        or projection.get("index_dtype") != "int32"
        or projection.get("degree_dtype") != "int32"
    ):
        raise ValueError(
            "undirected projection offsets, indices, and degree must use int32"
        )

    with source_metadata_path.open("r", encoding="utf-8") as handle:
        source_metadata = json.load(handle)
    source_record = projection.get("source_graph")
    if isinstance(source_record, dict):
        for role, field in (("offsets", "offsets_path"), ("indices", "indices_path")):
            expected = source_record.get(role, {}).get("sha256")
            current = (
                source_metadata.get("csr_artifacts", {})
                .get(role, {})
                .get("sha256")
            )
            if expected and current and expected != current:
                raise ValueError(
                    "undirected projection was built from different source "
                    f"CSR {role}"
                )
        expected_manifest_sha = source_record.get("manifest_sha256")
        topology_checksums_available = all(
            source_record.get(role, {}).get("sha256")
            and source_metadata.get("csr_artifacts", {})
            .get(role, {})
            .get("sha256")
            for role in ("offsets", "indices")
        )
        if (
            expected_manifest_sha
            and not topology_checksums_available
            and _sha256(source_metadata_path) != expected_manifest_sha
        ):
            raise ValueError(
                "undirected projection was built from a different source "
                "CSR manifest"
            )

    path_fields = (
        "lower_V_path",
        "lower_E_path",
        "upper_V_path",
        "upper_E_path",
        "degree_path",
        "forward_V_path",
        "forward_E_path",
    )
    paths = {
        field: _resolved_path(projection_metadata_path, projection[field])
        for field in path_fields
    }
    expected_sizes = {
        "lower_V_path": (num_nodes + 1) * 4,
        "lower_E_path": unique_edge_count * 4,
        "upper_V_path": (num_nodes + 1) * 4,
        "upper_E_path": unique_edge_count * 4,
        "degree_path": num_nodes * 4,
        "forward_V_path": (num_nodes + 1) * 4,
        "forward_E_path": unique_edge_count * 4,
    }
    projection_weights_value = projection.get("weights_path")
    source_weight_key = str(source_metadata.get("weight_key", "weight"))
    include_projection_weights = bool(
        projection_weights_value
        and source_metadata.get("weights_path")
        and source_weight_key
        == str(projection.get("weight_key", "weight"))
    )
    if include_projection_weights:
        if projection.get("weight_dtype") != "float64":
            raise ValueError(
                "undirected projection weights must use float64"
            )
        paths["weights_path"] = _resolved_path(
            projection_metadata_path, projection_weights_value
        )
        expected_sizes["weights_path"] = unique_edge_count * 8
    for field, expected_bytes in expected_sizes.items():
        found_bytes = paths[field].stat().st_size
        if found_bytes != expected_bytes:
            raise ValueError(
                f"undirected projection {field} size mismatch: expected "
                f"{expected_bytes}, found {found_bytes}"
            )

    spec = {
        "unique_edge_count": unique_edge_count,
        **{field: str(path) for field, path in paths.items()},
    }
    if include_projection_weights:
        spec["weight_key"] = source_weight_key
    projection_bytes = sum(expected_sizes.values())
    return projection_metadata_path, projection, spec, projection_bytes


class EGGPUBulkGraph:
    """Read-only graph with contiguous integer node labels ``0..N-1``.

    Parameters
    ----------
    metadata_path : path-like
        JSON manifest produced by the EGGPU CSR conversion tools.
    validate : bool, optional
        Scan every offset and endpoint after loading.  If omitted, the value
        recorded in ``EASYGRAPH_GPU_BULK_VALIDATE`` is used (default: true).

    Notes
    -----
    The current native CSR ABI uses signed 32-bit offsets and indices.  A
    graph may therefore contain at most ``2^31-1`` stored CSR entries.  For an
    undirected graph, both directions count toward this limit.
    """

    cflag = 0
    _eggpu_bulk_csr = True

    def __init__(
        self,
        metadata_path,
        validate=None,
        undirected_projection_path=None,
    ):
        metadata_path = Path(metadata_path).expanduser().resolve()
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        if metadata.get("format") != _FORMAT:
            raise ValueError(
                f"unsupported EGGPU CSR format {metadata.get('format')!r}; "
                f"expected {_FORMAT!r}"
            )
        if metadata.get("offset_dtype") != "int32" or metadata.get("index_dtype") != "int32":
            raise ValueError("EGGPU bulk CSR requires int32 offsets and int32 indices")
        if metadata.get("node_labels") != "zero_based_contiguous":
            raise ValueError("EGGPU bulk CSR requires zero-based contiguous node labels")

        num_nodes = int(metadata["num_nodes"])
        num_entries = int(metadata["num_entries"])
        if num_nodes < 0 or num_nodes >= _INT32_LIMIT:
            raise ValueError("num_nodes exceeds the signed 32-bit EGGPU CSR limit")
        if num_entries < 0 or num_entries >= _INT32_LIMIT:
            raise ValueError("num_entries exceeds the signed 32-bit EGGPU CSR limit")

        offsets_path = _resolved_path(metadata_path, metadata["offsets_path"])
        indices_path = _resolved_path(metadata_path, metadata["indices_path"])
        weights_value = metadata.get("weights_path")
        weights_path = (
            _resolved_path(metadata_path, weights_value) if weights_value else None
        )
        weight_key = str(metadata.get("weight_key", "weight"))
        expected_offsets = (num_nodes + 1) * 4
        expected_indices = num_entries * 4
        if offsets_path.stat().st_size != expected_offsets:
            raise ValueError(
                f"offset file size mismatch: expected {expected_offsets}, "
                f"found {offsets_path.stat().st_size}"
            )
        if indices_path.stat().st_size != expected_indices:
            raise ValueError(
                f"index file size mismatch: expected {expected_indices}, "
                f"found {indices_path.stat().st_size}"
            )
        if weights_path is not None:
            if metadata.get("weight_dtype") != "float64":
                raise ValueError("EGGPU bulk CSR explicit weights require float64")
            expected_weights = num_entries * 8
            if weights_path.stat().st_size != expected_weights:
                raise ValueError(
                    f"weight file size mismatch: expected {expected_weights}, "
                    f"found {weights_path.stat().st_size}"
                )
            if not weight_key:
                raise ValueError("weight_key must be non-empty")

        if validate is None:
            validate = os.environ.get("EASYGRAPH_GPU_BULK_VALIDATE", "TRUE").strip().upper() in {
                "1",
                "TRUE",
                "ON",
                "YES",
            }

        if undirected_projection_path is None:
            undirected_projection_path = metadata.get(
                "undirected_projection_manifest"
            )
            if undirected_projection_path is not None:
                undirected_projection_path = _resolved_path(
                    metadata_path, undirected_projection_path
                )
        (
            projection_metadata_path,
            projection_metadata,
            projection_spec,
            projection_bytes,
        ) = _load_undirected_projection(
            metadata_path,
            undirected_projection_path,
            num_nodes,
        )

        try:
            import cpp_easygraph
        except ImportError as exc:
            raise RuntimeError("EGGPUBulkGraph requires the compiled EasyGraph extension") from exc
        loader = getattr(cpp_easygraph, "cpp_graph_from_csr_files", None)
        if loader is None:
            raise RuntimeError(
                "the compiled EasyGraph extension does not include the EGGPU bulk CSR loader; rebuild it"
            )

        started = time.perf_counter()
        loader_args = [
            str(offsets_path),
            str(indices_path),
            num_nodes,
            num_entries,
            bool(metadata["directed"]),
            bool(validate),
            str(weights_path) if weights_path is not None else "",
            weight_key,
            projection_spec,
        ]
        self._cpp_graph = loader(*loader_args)
        self.load_seconds = time.perf_counter() - started
        self.metadata_path = metadata_path
        self.metadata = metadata
        self.offsets_path = offsets_path
        self.indices_path = indices_path
        self.num_nodes = num_nodes
        self.num_entries = num_entries
        self.num_edges = int(metadata.get("num_edges", num_entries))
        self._directed = bool(metadata["directed"])
        self.weights_path = weights_path
        self.weight_key = weight_key if weights_path is not None else None
        self.undirected_projection_path = projection_metadata_path
        self.undirected_projection = projection_metadata
        self.undirected_projection_bytes = projection_bytes
        self.nodes = range(num_nodes)
        self.cache = {}
        self.graph = {
            "name": metadata.get("name", metadata_path.stem),
            "source": metadata.get("source", ""),
            "eggpu_bulk_csr": True,
        }
        self.name = self.graph["name"]
        self._eggpu_bulk_signature = (
            str(metadata_path),
            int(metadata.get("generation", 0)),
            num_nodes,
            num_entries,
            self._directed,
            str(projection_metadata_path)
            if projection_metadata_path is not None
            else None,
            int(projection_metadata.get("generation", 0))
            if projection_metadata is not None
            else 0,
        )
        self._degree_cache = {}

    def cpp(self):
        return self._cpp_graph

    def undirected_projection_info(self):
        """Return native projection sizes without exposing the large arrays."""

        try:
            import cpp_easygraph
        except ImportError as exc:
            raise RuntimeError(
                "EGGPUBulkGraph requires the compiled EasyGraph extension"
            ) from exc
        info = cpp_easygraph.cpp_graph_undirected_projection_info(
            self._cpp_graph
        )
        return dict(info)

    def is_directed(self):
        return self._directed

    def is_multigraph(self):
        return False

    def __len__(self):
        return self.num_nodes

    def __iter__(self):
        return iter(self.nodes)

    def __contains__(self, node):
        return isinstance(node, int) and 0 <= node < self.num_nodes

    def number_of_nodes(self):
        return self.num_nodes

    def number_of_edges(self, u=None, v=None):
        if u is not None or v is not None:
            raise NotImplementedError("per-pair edge lookup is not available on EGGPUBulkGraph")
        return self.num_edges

    def has_node(self, node):
        return node in self

    def copy(self):
        return self

    def degree_values(self, weight=None):
        """Return dense total-degree values without Python adjacency objects.

        For directed graphs this follows EasyGraph's degree convention and
        sums outgoing and incoming contributions.  The result is cached on
        this immutable graph handle, so structural-hole postprocessing can
        reuse it across calls.
        """

        if weight not in {None, self.weight_key}:
            raise KeyError(
                f"bulk graph only stores weight attribute {self.weight_key!r}, "
                f"not {weight!r}"
            )
        cache_key = "unit" if weight is None else f"weight:{weight}"
        cached = self._degree_cache.get(cache_key)
        if cached is not None:
            return cached

        if self.num_entries == 0:
            values = np.zeros(self.num_nodes, dtype=np.float64)
            values.setflags(write=False)
            self._degree_cache[cache_key] = values
            return values

        offsets = np.memmap(
            self.offsets_path,
            mode="r",
            dtype=np.int32,
            shape=(self.num_nodes + 1,),
        )
        if weight is None:
            outgoing = np.diff(offsets).astype(np.float64, copy=False)
        else:
            if self.weights_path is None:
                raise ValueError("bulk graph does not contain explicit edge weights")
            weights = np.memmap(
                self.weights_path,
                mode="r",
                dtype=np.float64,
                shape=(self.num_entries,),
            )
            outgoing = np.add.reduceat(
                weights,
                np.minimum(
                    np.asarray(offsets[:-1], dtype=np.int64),
                    max(0, self.num_entries - 1),
                ),
            )
            empty = np.diff(offsets) == 0
            outgoing[empty] = 0.0

        if self._directed:
            indices = np.memmap(
                self.indices_path,
                mode="r",
                dtype=np.int32,
                shape=(self.num_entries,),
            )
            if weight is None:
                incoming = np.bincount(indices, minlength=self.num_nodes)
            else:
                weights = np.memmap(
                    self.weights_path,
                    mode="r",
                    dtype=np.float64,
                    shape=(self.num_entries,),
                )
                incoming = np.bincount(
                    indices,
                    weights=weights,
                    minlength=self.num_nodes,
                )
            values = outgoing + incoming
        else:
            values = outgoing

        values = np.asarray(values, dtype=np.float64)
        values.setflags(write=False)
        self._degree_cache[cache_key] = values
        return values

    def nonisolated_mask(self):
        """Return a cached dense mask for nodes incident to at least one edge."""

        cached = self._degree_cache.get("nonisolated")
        if cached is not None:
            return cached
        offsets = np.memmap(
            self.offsets_path,
            mode="r",
            dtype=np.int32,
            shape=(self.num_nodes + 1,),
        )
        mask = np.diff(offsets) != 0
        if self._directed and self.num_entries:
            indices = np.memmap(
                self.indices_path,
                mode="r",
                dtype=np.int32,
                shape=(self.num_entries,),
            )
            chunk = 32_000_000
            for start in range(0, self.num_entries, chunk):
                end = min(self.num_entries, start + chunk)
                mask[np.asarray(indices[start:end], dtype=np.int32)] = True
        mask = np.asarray(mask, dtype=np.bool_)
        mask.setflags(write=False)
        self._degree_cache["nonisolated"] = mask
        return mask

    @property
    def adj(self):
        raise NotImplementedError(
            "EGGPUBulkGraph intentionally does not materialize Python adjacency dictionaries"
        )

    @property
    def pred(self):
        if not self._directed:
            raise AttributeError("undirected EGGPUBulkGraph has no predecessor view")
        raise NotImplementedError(
            "EGGPUBulkGraph intentionally does not materialize Python predecessor dictionaries"
        )

    def _immutable(self, *args, **kwargs):
        raise TypeError("EGGPUBulkGraph is immutable; regenerate its CSR artifact to change it")

    add_node = _immutable
    add_nodes = _immutable
    add_nodes_from = _immutable
    add_edge = _immutable
    add_edges = _immutable
    add_edges_from = _immutable
    remove_node = _immutable
    remove_nodes = _immutable
    remove_edge = _immutable
    remove_edges = _immutable

    def memory_estimate(self):
        """Return the persistent host CSR footprint recorded by the format."""

        base_total = (
            (self.num_nodes + 1 + self.num_entries) * 4
            + (self.num_entries * 8 if self.weights_path else 0)
        )
        return {
            "host_offsets_bytes": (self.num_nodes + 1) * 4,
            "host_indices_bytes": self.num_entries * 4,
            "host_weights_bytes": self.num_entries * 8 if self.weights_path else 0,
            "host_total_bytes": base_total,
            "host_undirected_projection_bytes": self.undirected_projection_bytes,
            "host_total_with_projection_bytes": (
                base_total + self.undirected_projection_bytes
            ),
            "implicit_unit_weights": self.weights_path is None,
            "weight_key": self.weight_key,
        }

    def __repr__(self):
        kind = "DiGraph" if self._directed else "Graph"
        return (
            f"EGGPUBulk{kind}(name={self.name!r}, nodes={self.num_nodes}, "
            f"edges={self.num_edges}, csr_entries={self.num_entries}, "
            "undirected_projection="
            f"{self.undirected_projection is not None})"
        )


def read_eggpu_csr(
    metadata_path,
    validate=None,
    undirected_projection_path=None,
):
    """Load an immutable native CSR graph for EGGPU analysis."""

    return EGGPUBulkGraph(
        metadata_path,
        validate=validate,
        undirected_projection_path=undirected_projection_path,
    )


__all__ = ["EGGPUBulkGraph", "read_eggpu_csr"]
