import os
import re
import shutil
import subprocess
import tempfile
import time
import weakref
import math
from collections.abc import Mapping
from pathlib import Path

from easygraph.utils import gpu_adaptive_policy as adaptive_policy
from easygraph.utils.gpu_runtime import gpu_backend_name
from easygraph.utils.gpu_runtime import gpu_runtime_enabled


_LAST_KERNEL_SECONDS = {}
_PREPARED_CACHE_KEY = "__gpu_mine_prepared_graph__"
_CACHE_ROOT = Path(
    os.environ.get(
        "EASYGRAPH_GPU_CACHE_DIR",
        str(Path(tempfile.gettempdir()) / "easygraph_gpu_cache"),
    )
).expanduser()
_MAX_CACHE_DIRS = int(os.environ.get("EASYGRAPH_GPU_MAX_CACHE_DIRS", "32"))
_HASH_MASK = (1 << 64) - 1
_RESULT_CACHE_ENABLED = os.environ.get("EASYGRAPH_GPU_RESULT_CACHE", "TRUE").strip().upper() in {
    "1",
    "TRUE",
    "ON",
    "YES",
}
_RESULT_CACHE_RETURN_COPY = os.environ.get(
    "EASYGRAPH_GPU_RESULT_CACHE_RETURN_COPY", "FALSE"
).strip().upper() in {
    "1",
    "TRUE",
    "ON",
    "YES",
}
_RESULT_CACHE_MAX_ITEMS = max(1, int(os.environ.get("EASYGRAPH_GPU_RESULT_CACHE_MAX_ITEMS", "8")))
_RESULT_CACHE_MAX_NODES = max(0, int(os.environ.get("EASYGRAPH_GPU_RESULT_CACHE_MAX_NODES", "2000000")))
_FALLBACK_GRAPH_CACHE = weakref.WeakKeyDictionary()
_CPP_GRAPH_CACHE_KEY = "__gpu_mine_cpp_graph__"
_CPP_GRAPH_CACHE_CTX_KEY = "__gpu_mine_cpp_graph_ctx__"
_CPP_DIRECT_DISABLED = os.environ.get("EASYGRAPH_GPU_DISABLE_CPP_DIRECT", "").strip().upper() in {
    "1",
    "TRUE",
    "ON",
    "YES",
}
_GRAPH_CONTEXT_CACHE_DISABLED = os.environ.get(
    "EASYGRAPH_GPU_DISABLE_GRAPH_CONTEXT_CACHE", ""
).strip().upper() in {
    "1",
    "TRUE",
    "ON",
    "YES",
}
_CPP_GRAPH_CACHE_DISABLED = os.environ.get(
    "EASYGRAPH_GPU_DISABLE_CPP_GRAPH_CACHE", ""
).strip().upper() in {
    "1",
    "TRUE",
    "ON",
    "YES",
}
_GRAPH_CTX_VALIDATE_EVERY = max(
    1,
    int(os.environ.get("EASYGRAPH_GPU_CTX_VALIDATE_EVERY", "32")),
)


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().upper() in {"1", "TRUE", "ON", "YES"}


_CPP_PREFER_DICT_DEFAULT = _env_bool("EASYGRAPH_GPU_PREFER_CPP_DICT", False)
_CPP_PREFER_DICT_GLOBAL_SET = os.environ.get("EASYGRAPH_GPU_PREFER_CPP_DICT") is not None
_CPP_PR_PREFER_DICT = _env_bool(
    "EASYGRAPH_GPU_PR_PREFER_CPP_DICT",
    _CPP_PREFER_DICT_DEFAULT if _CPP_PREFER_DICT_GLOBAL_SET else False,
)
_CPP_LCC_PREFER_DICT = _env_bool(
    "EASYGRAPH_GPU_LCC_PREFER_CPP_DICT",
    _CPP_PREFER_DICT_DEFAULT if _CPP_PREFER_DICT_GLOBAL_SET else False,
)
_CPP_CC_PREFER_DICT = _env_bool(
    "EASYGRAPH_GPU_CC_PREFER_CPP_DICT",
    _CPP_PREFER_DICT_DEFAULT if _CPP_PREFER_DICT_GLOBAL_SET else False,
)


def mine_backend_enabled():
    if not gpu_runtime_enabled():
        return False
    return gpu_backend_name() in {
        "mine",
        "mine-bin",
        "eggpu",
        "native-mine",
        "auto",
        "default",
    }


def set_last_kernel_time(key, seconds):
    if seconds is None:
        _LAST_KERNEL_SECONDS.pop(key, None)
    else:
        _LAST_KERNEL_SECONDS[key] = float(seconds)


def get_last_kernel_time(key):
    return _LAST_KERNEL_SECONDS.get(key)


def _temporarily_disable_gpu_env(callable_obj):
    old = os.environ.get("EASYGRAPH_ENABLE_GPU")
    os.environ["EASYGRAPH_ENABLE_GPU"] = "FALSE"
    try:
        return callable_obj()
    finally:
        if old is None:
            os.environ.pop("EASYGRAPH_ENABLE_GPU", None)
        else:
            os.environ["EASYGRAPH_ENABLE_GPU"] = old


def _edge_slots(prepared):
    return adaptive_policy.edge_slots(prepared)


class _DenseValueDict(dict):
    """Dict-compatible view over a dense result vector.

    The EasyGraph public API historically returns dict-like objects for
    PageRank/LCC.  Materializing millions of Python key/value pairs dominates
    EGGPU e2e time, so this view keeps dense storage and only creates Python
    pairs when a caller explicitly iterates over items.
    """

    def __init__(self, nodes, values, node_to_idx=None, dtype=float):
        import numpy as np

        self._nodes = list(nodes)
        self._values = np.asarray(values, dtype=dtype).reshape(-1)
        if len(self._values) < len(self._nodes):
            padded = np.zeros(len(self._nodes), dtype=self._values.dtype)
            padded[: len(self._values)] = self._values
            self._values = padded
        elif len(self._values) > len(self._nodes):
            self._values = self._values[: len(self._nodes)]
        self._node_to_idx = node_to_idx if isinstance(node_to_idx, dict) else None
        self._index = None

    def _idx(self, key):
        if self._node_to_idx is not None:
            return self._node_to_idx.get(key)
        if self._index is None:
            self._index = {node: i for i, node in enumerate(self._nodes)}
        return self._index.get(key)

    def __len__(self):
        return len(self._nodes)

    def __iter__(self):
        return iter(self._nodes)

    def __contains__(self, key):
        return self._idx(key) is not None

    def __getitem__(self, key):
        idx = self._idx(key)
        if idx is None:
            raise KeyError(key)
        return float(self._values[int(idx)])

    def get(self, key, default=None):
        idx = self._idx(key)
        if idx is None:
            return default
        return float(self._values[int(idx)])

    def keys(self):
        return list(self._nodes)

    def values(self):
        return self._values

    def items(self):
        for i, node in enumerate(self._nodes):
            yield node, float(self._values[i])

    def copy(self):
        return dict(self.items())

    def to_dict(self):
        return dict(self.items())

    def to_numpy(self, copy=False):
        return self._values.copy() if copy else self._values

    def tolist(self):
        return self._values.tolist()

    def __array__(self, dtype=None):
        import numpy as np

        return np.asarray(self._values, dtype=dtype)

    def __repr__(self):
        return repr(self.to_dict())


class _DenseDistanceDict(dict):
    def __init__(self, nodes, values, node_to_idx=None):
        import numpy as np

        self._nodes = list(nodes)
        self._values = np.asarray(values, dtype=np.float64).reshape(-1)
        if len(self._values) < len(self._nodes):
            padded = np.full(len(self._nodes), math.inf, dtype=np.float64)
            padded[: len(self._values)] = self._values
            self._values = padded
        elif len(self._values) > len(self._nodes):
            self._values = self._values[: len(self._nodes)]
        self._node_to_idx = node_to_idx if isinstance(node_to_idx, dict) else None
        self._index = None
        self._finite_mask = None
        self._finite_idxs = None

    def _idx(self, key):
        if self._node_to_idx is not None:
            return self._node_to_idx.get(key)
        if self._index is None:
            self._index = {node: i for i, node in enumerate(self._nodes)}
        return self._index.get(key)

    def _finite_indices(self):
        if self._finite_idxs is None:
            import numpy as np

            vals = self._values
            self._finite_mask = np.isfinite(vals) & (np.abs(vals) < 1.0e30)
            self._finite_idxs = np.flatnonzero(self._finite_mask)
        return self._finite_idxs

    def __len__(self):
        return int(len(self._finite_indices()))

    def __iter__(self):
        for idx in self._finite_indices():
            yield self._nodes[int(idx)]

    def __contains__(self, key):
        idx = self._idx(key)
        if idx is None:
            return False
        val = float(self._values[int(idx)])
        return math.isfinite(val) and abs(val) < 1.0e30

    def __getitem__(self, key):
        idx = self._idx(key)
        if idx is None:
            raise KeyError(key)
        val = float(self._values[int(idx)])
        if not math.isfinite(val) or abs(val) >= 1.0e30:
            raise KeyError(key)
        return val

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def values(self):
        return self._values[self._finite_indices()]

    def items(self):
        for idx in self._finite_indices():
            i = int(idx)
            yield self._nodes[i], float(self._values[i])

    def keys(self):
        return list(iter(self))

    def copy(self):
        return dict(self.items())

    def to_numpy(self, copy=False):
        return self._values.copy() if copy else self._values

    def __repr__(self):
        return repr(self.copy())


class _DenseMultiSourceSSSPDict(dict):
    def __init__(self, sources, nodes, values, node_to_idx=None):
        import numpy as np

        self._sources = list(sources)
        self._nodes = list(nodes)
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[0] < len(self._sources):
            padded = np.full((len(self._sources), arr.shape[1]), math.inf, dtype=np.float64)
            padded[: arr.shape[0], : arr.shape[1]] = arr
            arr = padded
        if arr.shape[1] == len(self._nodes) + 1:
            arr = arr[:, 1:]
        if arr.shape[1] < len(self._nodes):
            padded = np.full((arr.shape[0], len(self._nodes)), math.inf, dtype=np.float64)
            padded[:, : arr.shape[1]] = arr
            arr = padded
        elif arr.shape[1] > len(self._nodes):
            arr = arr[:, : len(self._nodes)]
        self._values = arr[: len(self._sources)]
        self._node_to_idx = node_to_idx if isinstance(node_to_idx, dict) else None
        self._rows = {}

    def __len__(self):
        return len(self._sources)

    def __iter__(self):
        return iter(self._sources)

    def __contains__(self, key):
        return key in self._sources

    def __getitem__(self, key):
        try:
            row = self._sources.index(key)
        except ValueError:
            raise KeyError(key) from None
        cached = self._rows.get(row)
        if cached is None:
            cached = _DenseDistanceDict(
                self._nodes,
                self._values[row],
                node_to_idx=self._node_to_idx,
            )
            self._rows[row] = cached
        return cached

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def values(self):
        return [self[source] for source in self._sources]

    def items(self):
        for source in self._sources:
            yield source, self[source]

    def keys(self):
        return list(self._sources)

    def copy(self):
        return {source: self[source].copy() for source in self._sources}

    def to_numpy(self, copy=False):
        return self._values.copy() if copy else self._values

    def __repr__(self):
        return repr(self.copy())


def _candidate_bin_dirs():
    dirs = []
    env_dir = os.environ.get("EASYGRAPH_GPU_BIN_DIR")
    if env_dir:
        dirs.append(Path(env_dir).expanduser())
    # Prefer repo-relative discovery so the backend does not depend on a
    # machine-specific absolute path.
    try:
        easygraph_repo = Path(__file__).resolve().parents[2]
        sibling_eval_dir = easygraph_repo.parent / "EG_Evaluation" / "basic_functions"
        if sibling_eval_dir.exists():
            dirs.append(sibling_eval_dir)
    except Exception:
        pass

    # Optional local fallback under current workspace.
    local_eval_dir = Path.cwd() / "EG_Evaluation" / "basic_functions"
    if local_eval_dir.exists():
        dirs.append(local_eval_dir)

    # Deduplicate while preserving search order.
    uniq = []
    seen = set()
    for d in dirs:
        k = str(d.resolve()) if d.exists() else str(d)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(d)
    dirs = uniq
    return dirs


def _resolve_bin(env_key, names):
    env_path = os.environ.get(env_key)
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists() and os.access(p, os.X_OK):
            return p
    for d in _candidate_bin_dirs():
        for name in names:
            p = d / name
            if p.exists() and os.access(p, os.X_OK):
                return p
    return None


def _parse_first_float(pattern, text):
    m = re.search(pattern, text, re.MULTILINE)
    return float(m.group(1)) if m else None


def _run_cmd(cmd, cwd=None):
    out = subprocess.run(
        [str(x) for x in cmd],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        raise RuntimeError(
            f"command failed ({out.returncode}): {' '.join(str(x) for x in cmd)}\n{out.stdout}\n{out.stderr}"
        )
    return out.stdout


def _graph_cache_dict(G):
    cache = getattr(G, "cache", None)
    if isinstance(cache, dict):
        return cache
    try:
        cache = _FALLBACK_GRAPH_CACHE.get(G)
        if cache is None:
            cache = {}
            _FALLBACK_GRAPH_CACHE[G] = cache
        return cache
    except TypeError:
        # Some extension types are not weak-referenceable.
        return None


def _cpp_graph_cache_keys(directed_mode, host_mode=False):
    suffix = "directed" if bool(directed_mode) else "undirected"
    if host_mode:
        suffix = f"host:{suffix}"
    return f"{_CPP_GRAPH_CACHE_KEY}:{suffix}", f"{_CPP_GRAPH_CACHE_CTX_KEY}:{suffix}"


def _build_cpp_graph(G, directed_mode):
    cpp_graph = None
    try:
        import cpp_easygraph

        fast_builder = getattr(cpp_easygraph, "cpp_graph_from_easygraph", None)
        if fast_builder is not None:
            cpp_graph = fast_builder(G, directed_mode)
    except Exception:
        cpp_graph = None
    if cpp_graph is None:
        cpp_graph = G.cpp()
    return cpp_graph


def _get_cached_cpp_graph(G, prepared=None, directed_override=None, host_mode=False):
    if _CPP_DIRECT_DISABLED:
        return None
    if not hasattr(G, "cpp"):
        return None
    if prepared is None or not isinstance(prepared, dict):
        prepared = _prepare_graph_cache(G)
    ctx_id = prepared.get("ctx_id")
    directed_mode = (
        bool(prepared.get("directed", G.is_directed()))
        if directed_override is None
        else bool(directed_override)
    )
    cache = None if _CPP_GRAPH_CACHE_DISABLED else _graph_cache_dict(G)
    cache_key, cache_ctx_key = _cpp_graph_cache_keys(
        directed_mode,
        host_mode=bool(host_mode),
    )
    if cache is not None:
        cached = cache.get(cache_key)
        cached_ctx = cache.get(cache_ctx_key)
        if cached is not None and cached_ctx == ctx_id:
            return cached
    if host_mode:
        cpp_graph = _temporarily_disable_gpu_env(lambda: _build_cpp_graph(G, directed_mode))
    else:
        cpp_graph = _build_cpp_graph(G, directed_mode)
    if cache is not None:
        cache[cache_key] = cpp_graph
        cache[cache_ctx_key] = ctx_id
    return cpp_graph


def _cpp_gpu_pagerank(prepared, G, alpha, max_iter, eps, weight):
    cpp_graph = _get_cached_cpp_graph(G, prepared=prepared)
    if cpp_graph is None:
        return None
    import cpp_easygraph

    nodes = prepared.get("nodes", [])
    target_len = len(nodes)
    dict_attempted = False
    if _CPP_PR_PREFER_DICT and hasattr(cpp_easygraph, "cpp_gpu_pagerank"):
        dict_attempted = True
        out = cpp_easygraph.cpp_gpu_pagerank(
            cpp_graph,
            alpha=float(alpha),
            max_iterator=int(max_iter),
            threshold=float(eps),
            weight=weight,
        )
        if isinstance(out, dict):
            values = out.get("values")
            kernel_s = out.get("kernel_seconds")
            if isinstance(values, dict):
                if target_len > 0 and len(values) == 0 and (kernel_s is None or float(kernel_s) == 0.0):
                    return None
                return values, (float(kernel_s) if kernel_s is not None else None)

    if hasattr(cpp_easygraph, "cpp_gpu_pagerank_dense"):
        out = cpp_easygraph.cpp_gpu_pagerank_dense(
            cpp_graph,
            alpha=float(alpha),
            max_iterator=int(max_iter),
            threshold=float(eps),
            weight=weight,
        )
        if isinstance(out, dict):
            dense = out.get("values_dense")
            if dense is not None:
                kernel_s = out.get("kernel_seconds")
                try:
                    raw_len = len(dense)
                except Exception:
                    raw_len = None
                if target_len > 0 and raw_len == 0 and (kernel_s is None or float(kernel_s) == 0.0):
                    return None
                vals = _normalize_dense_array(dense, target_len, dtype="float64")
                values = _DenseValueDict(
                    nodes,
                    vals,
                    node_to_idx=prepared.get("node_to_idx"),
                    dtype=float,
                )
                return values, (float(kernel_s) if kernel_s is not None else None)

    if dict_attempted or not hasattr(cpp_easygraph, "cpp_gpu_pagerank"):
        return None
    out = cpp_easygraph.cpp_gpu_pagerank(
        cpp_graph,
        alpha=float(alpha),
        max_iterator=int(max_iter),
        threshold=float(eps),
        weight=weight,
    )
    if not isinstance(out, dict):
        return None
    values = out.get("values")
    if not isinstance(values, dict):
        return None
    kernel_s = out.get("kernel_seconds")
    return values, (float(kernel_s) if kernel_s is not None else None)


def _cpp_gpu_mst_edges(prepared, G, weight):
    cpp_graph = _get_cached_cpp_graph(G, prepared=prepared)
    if cpp_graph is None:
        return None
    import cpp_easygraph

    nodes = prepared.get("nodes", [])
    edge_slots = prepared.get("stamp", (False, 0, 0, 0))[2]
    if hasattr(cpp_easygraph, "cpp_gpu_mst_edges"):
        out = cpp_easygraph.cpp_gpu_mst_edges(cpp_graph, weight=weight)
        if isinstance(out, dict):
            edges = out.get("edges")
            if isinstance(edges, list):
                kernel_s = out.get("kernel_seconds")
                if edge_slots and not edges and (kernel_s is None or float(kernel_s) == 0.0):
                    return None
                normalized = []
                for e in edges:
                    if not isinstance(e, (list, tuple)) or len(e) < 3:
                        continue
                    normalized.append((e[0], e[1], float(e[2])))
                return normalized, (float(kernel_s) if kernel_s is not None else None)

    if hasattr(cpp_easygraph, "cpp_gpu_mst_index_edges"):
        out = cpp_easygraph.cpp_gpu_mst_index_edges(cpp_graph, weight=weight)
        if isinstance(out, dict):
            idx_edges = out.get("index_edges")
            if isinstance(idx_edges, list):
                kernel_s = out.get("kernel_seconds")
                if edge_slots and not idx_edges and (kernel_s is None or float(kernel_s) == 0.0):
                    return None
                normalized = []
                n = len(nodes)
                for e in idx_edges:
                    if not isinstance(e, (list, tuple)) or len(e) < 3:
                        continue
                    try:
                        iu = int(e[0])
                        iv = int(e[1])
                        w = float(e[2])
                    except Exception:
                        continue
                    if 0 <= iu < n and 0 <= iv < n:
                        normalized.append((nodes[iu], nodes[iv], w))
                return normalized, (float(kernel_s) if kernel_s is not None else None)

    return None


def _cpp_gpu_mst_tree(prepared, G, weight):
    cpp_graph = _get_cached_cpp_graph(G, prepared=prepared, directed_override=False)
    if cpp_graph is None:
        return None
    import cpp_easygraph

    if not hasattr(cpp_easygraph, "cpp_gpu_mst_tree"):
        return None
    edge_slots = prepared.get("stamp", (False, 0, 0, 0))[2]
    out = cpp_easygraph.cpp_gpu_mst_tree(cpp_graph, weight=weight)
    if not isinstance(out, dict):
        return None
    tree = out.get("tree")
    kernel_s = out.get("kernel_seconds")
    if edge_slots and tree is None and (kernel_s is None or float(kernel_s) == 0.0):
        return None
    return tree, (float(kernel_s) if kernel_s is not None else None)


def _cpp_gpu_clustering(prepared, G):
    cpp_graph = _get_cached_cpp_graph(G, prepared=prepared)
    if cpp_graph is None:
        return None
    import cpp_easygraph

    nodes = prepared.get("nodes", [])
    target_len = len(nodes)
    dict_attempted = False
    if _CPP_LCC_PREFER_DICT and hasattr(cpp_easygraph, "cpp_gpu_clustering"):
        dict_attempted = True
        out = cpp_easygraph.cpp_gpu_clustering(cpp_graph)
        if isinstance(out, dict):
            values = out.get("values")
            kernel_s = out.get("kernel_seconds")
            if isinstance(values, dict):
                if target_len > 0 and len(values) == 0 and (kernel_s is None or float(kernel_s) == 0.0):
                    return None
                return values, (float(kernel_s) if kernel_s is not None else None)

    if hasattr(cpp_easygraph, "cpp_gpu_clustering_dense"):
        out = cpp_easygraph.cpp_gpu_clustering_dense(cpp_graph)
        if isinstance(out, dict):
            dense = out.get("values_dense")
            if dense is not None:
                kernel_s = out.get("kernel_seconds")
                try:
                    raw_len = len(dense)
                except Exception:
                    raw_len = None
                if target_len > 0 and raw_len == 0 and (kernel_s is None or float(kernel_s) == 0.0):
                    return None
                vals = _normalize_dense_array(dense, target_len, dtype="float64")
                values = _DenseValueDict(
                    nodes,
                    vals,
                    node_to_idx=prepared.get("node_to_idx"),
                    dtype=float,
                )
                return values, (float(kernel_s) if kernel_s is not None else None)

    if dict_attempted or not hasattr(cpp_easygraph, "cpp_gpu_clustering"):
        return None
    out = cpp_easygraph.cpp_gpu_clustering(cpp_graph)
    if not isinstance(out, dict):
        return None
    values = out.get("values")
    if not isinstance(values, dict):
        return None
    kernel_s = out.get("kernel_seconds")
    return values, (float(kernel_s) if kernel_s is not None else None)


def _structural_nodes(prepared, nodes):
    all_nodes = prepared.get("nodes", [])
    if nodes is None:
        return None, all_nodes, prepared.get("node_to_idx")
    node_to_idx = prepared.get("node_to_idx", {})
    try:
        if nodes in node_to_idx:
            selected = [nodes]
            return selected, selected, {nodes: 0}
    except TypeError:
        pass
    selected = list(nodes)
    missing = [node for node in selected if node not in node_to_idx]
    if missing:
        raise KeyError(f"nodes are not in graph: {missing[:5]}")
    return selected, selected, {node: i for i, node in enumerate(selected)}


def _edge_attr_weight(data, weight):
    if weight is None:
        return 1.0
    if isinstance(data, Mapping):
        try:
            return float(data.get(weight, 1))
        except Exception:
            return 1.0
    return 1.0


def _weighted_degree_values(G, target_nodes, weight):
    adj = _adj_dict(G)
    directed = bool(G.is_directed())
    pred = getattr(G, "_pred", None)
    values = []
    for node in target_nodes:
        total = 0.0
        for data in adj.get(node, {}).values():
            total += _edge_attr_weight(data, weight)
        if directed:
            if isinstance(pred, dict):
                for data in pred.get(node, {}).values():
                    total += _edge_attr_weight(data, weight)
            else:
                for _, nbrs in adj.items():
                    data = nbrs.get(node)
                    if data is not None:
                        total += _edge_attr_weight(data, weight)
        values.append(total)
    return values


def _neighbor_count_values(G, target_nodes):
    adj = _adj_dict(G)
    directed = bool(G.is_directed())
    pred = getattr(G, "_pred", None)
    counts = []
    for node in target_nodes:
        if directed and isinstance(pred, dict):
            nbrs = set(adj.get(node, {}).keys())
            nbrs.update(pred.get(node, {}).keys())
            counts.append(len(nbrs))
        elif directed:
            nbrs = set(adj.get(node, {}).keys())
            for src, out_nbrs in adj.items():
                if node in out_nbrs:
                    nbrs.add(src)
            counts.append(len(nbrs))
        else:
            counts.append(len(adj.get(node, {})))
    return counts


def _slice_dense_for_nodes(prepared, dense, selected_nodes, selected_index):
    import numpy as np

    all_nodes = prepared.get("nodes", [])
    full = np.asarray(dense, dtype=np.float64).reshape(-1)
    if selected_nodes is all_nodes:
        return _DenseValueDict(
            all_nodes,
            full,
            node_to_idx=prepared.get("node_to_idx"),
            dtype=float,
        )
    if len(selected_nodes) == len(all_nodes) and all(a == b for a, b in zip(selected_nodes, all_nodes)):
        return _DenseValueDict(
            all_nodes,
            full,
            node_to_idx=prepared.get("node_to_idx"),
            dtype=float,
        )
    node_to_idx = prepared.get("node_to_idx", {})
    values = np.empty(len(selected_nodes), dtype=np.float64)
    for i, node in enumerate(selected_nodes):
        values[i] = full[int(node_to_idx[node])]
    return _DenseValueDict(selected_nodes, values, node_to_idx=selected_index, dtype=float)


def _cpp_gpu_structural_dense(prepared, G, metric, nodes, weight):
    cpp_graph = _get_cached_cpp_graph(G, prepared=prepared)
    if cpp_graph is None:
        return None
    import cpp_easygraph
    import numpy as np

    fn_name = f"cpp_gpu_{metric}_dense"
    fn = getattr(cpp_easygraph, fn_name, None)
    if fn is None:
        return None

    cpp_nodes_arg, selected_nodes, selected_index = _structural_nodes(prepared, nodes)
    out = fn(cpp_graph, nodes=cpp_nodes_arg, weight=weight)
    if not isinstance(out, dict):
        return None
    dense = out.get("values_dense")
    if dense is None:
        return None
    all_nodes = prepared.get("nodes", [])
    values = _normalize_dense_array(dense, len(all_nodes), dtype="float64", fill_value=0)
    counts = np.asarray(_neighbor_count_values(G, all_nodes), dtype=np.int64)
    if metric in {"effective_size", "constraint"}:
        values[counts == 0] = np.nan
    elif metric == "hierarchy":
        values[counts == 0] = 0.0
    result = _slice_dense_for_nodes(prepared, values, selected_nodes, selected_index)
    kernel_s = out.get("kernel_seconds")
    return result, (float(kernel_s) if kernel_s is not None else None)


def _cpp_gpu_connected_component_labels(prepared, G, directed):
    cpp_graph = _get_cached_cpp_graph(G, prepared=prepared, directed_override=bool(directed))
    if cpp_graph is None:
        return None
    import cpp_easygraph

    nodes = prepared.get("nodes", [])
    target_len = len(nodes)
    dict_attempted = False
    if _CPP_CC_PREFER_DICT and hasattr(cpp_easygraph, "cpp_gpu_connected_component_labels"):
        dict_attempted = True
        out = cpp_easygraph.cpp_gpu_connected_component_labels(cpp_graph, directed=bool(directed))
        if isinstance(out, dict):
            labels = out.get("labels")
            kernel_s = out.get("kernel_seconds")
            if isinstance(labels, dict):
                if target_len > 0 and len(labels) == 0 and (kernel_s is None or float(kernel_s) == 0.0):
                    return None
                return labels, (float(kernel_s) if kernel_s is not None else None)

    if hasattr(cpp_easygraph, "cpp_gpu_connected_component_labels_dense"):
        out = cpp_easygraph.cpp_gpu_connected_component_labels_dense(
            cpp_graph, directed=bool(directed)
        )
        if isinstance(out, dict):
            dense = out.get("labels_dense")
            if dense is not None:
                kernel_s = out.get("kernel_seconds")
                try:
                    raw_len = len(dense)
                except Exception:
                    raw_len = None
                if target_len > 0 and raw_len == 0 and (kernel_s is None or float(kernel_s) == 0.0):
                    return None
                raw = _normalize_dense_vector(dense, target_len)
                labels = dict(zip(nodes, (int(v) for v in raw)))
                return labels, (float(kernel_s) if kernel_s is not None else None)

    if dict_attempted or not hasattr(cpp_easygraph, "cpp_gpu_connected_component_labels"):
        return None
    out = cpp_easygraph.cpp_gpu_connected_component_labels(cpp_graph, directed=bool(directed))
    if not isinstance(out, dict):
        return None
    labels = out.get("labels")
    if not isinstance(labels, dict):
        return None
    kernel_s = out.get("kernel_seconds")
    return labels, (float(kernel_s) if kernel_s is not None else None)


def _index_to_node_list(prepared, G):
    nodes = list(prepared.get("nodes", []))
    idx_map = getattr(G, "index2node", None)
    if isinstance(idx_map, dict) and idx_map:
        out = []
        i = 0
        while i in idx_map:
            out.append(idx_map[i])
            i += 1
        if len(out) == len(nodes):
            return out
    return nodes


def _normalize_dense_vector(raw_values, target_len):
    try:
        n = len(raw_values)
    except Exception:
        n = None

    start = 0
    if n is not None and n == target_len + 1:
        # Some cpp paths use 1-based IDs and return a sentinel at 0.
        start = 1
        n -= 1

    if n is not None and n >= target_len:
        return [raw_values[start + i] for i in range(target_len)]

    values = list(raw_values[start:]) if start else list(raw_values)
    if len(values) < target_len:
        values.extend([0.0] * (target_len - len(values)))
    elif len(values) > target_len:
        values = values[:target_len]
    return values


def _normalize_dense_array(raw_values, target_len, dtype="float64", fill_value=0):
    import numpy as np

    arr = np.asarray(raw_values, dtype=dtype).reshape(-1)
    if len(arr) == target_len + 1:
        arr = arr[1:]
    if len(arr) < target_len:
        padded = np.full(target_len, fill_value, dtype=arr.dtype)
        padded[: len(arr)] = arr
        arr = padded
    elif len(arr) > target_len:
        arr = arr[:target_len]
    return arr


def _normalize_dense_labels(raw_values, target_len):
    values = _normalize_dense_array(raw_values, target_len, dtype="int64", fill_value=-1)
    return values


def _components_from_dense_labels(prepared, dense_labels):
    import numpy as np

    nodes = prepared.get("nodes", [])
    labels = np.asarray(dense_labels, dtype=np.int64).reshape(-1)
    if len(labels) == len(nodes) + 1:
        labels = labels[1:]
    if len(labels) < len(nodes):
        padded = np.full(len(nodes), -1, dtype=np.int64)
        padded[: len(labels)] = labels
        labels = padded
    elif len(labels) > len(nodes):
        labels = labels[: len(nodes)]

    valid = labels >= 0
    if not bool(valid.any()):
        return [{node} for node in nodes]

    max_label = int(labels[valid].max())
    if max_label <= max(1, len(nodes) * 2):
        counts = np.bincount(labels[valid], minlength=max_label + 1)
    else:
        uniq, cnt = np.unique(labels[valid], return_counts=True)
        counts = dict(zip((int(x) for x in uniq), (int(x) for x in cnt)))

    components = []
    grouped = {}
    next_singleton = max_label + 1
    for idx, node in enumerate(nodes):
        lab = int(labels[idx]) if idx < len(labels) else -1
        if lab < 0:
            lab = next_singleton
            next_singleton += 1
            components.append({node})
            continue
        count = int(counts[lab]) if not isinstance(counts, dict) else int(counts.get(lab, 0))
        if count <= 1:
            components.append({node})
        else:
            grouped.setdefault(lab, []).append(node)
    for members in grouped.values():
        components.append(set(members))
    return components


def _cpp_gpu_connected_component_labels_dense(prepared, G, directed):
    cpp_graph = _get_cached_cpp_graph(G, prepared=prepared, directed_override=bool(directed))
    if cpp_graph is None:
        return None
    import cpp_easygraph

    if not hasattr(cpp_easygraph, "cpp_gpu_connected_component_labels_dense"):
        return None
    out = cpp_easygraph.cpp_gpu_connected_component_labels_dense(
        cpp_graph, directed=bool(directed)
    )
    if not isinstance(out, dict):
        return None
    dense = out.get("labels_dense")
    if dense is None:
        return None
    nodes = prepared.get("nodes", [])
    kernel_s = out.get("kernel_seconds")
    try:
        raw_len = len(dense)
    except Exception:
        raw_len = None
    if len(nodes) > 0 and raw_len == 0 and (kernel_s is None or float(kernel_s) == 0.0):
        return None
    labels = _normalize_dense_labels(dense, len(nodes))
    return labels, (float(kernel_s) if kernel_s is not None else None)


def _cpp_gpu_connected_components_sets(prepared, G, directed):
    cpp_graph = _get_cached_cpp_graph(G, prepared=prepared, directed_override=bool(directed))
    if cpp_graph is None:
        return None
    import cpp_easygraph

    if not hasattr(cpp_easygraph, "cpp_gpu_connected_components_sets"):
        return None
    out = cpp_easygraph.cpp_gpu_connected_components_sets(
        cpp_graph, directed=bool(directed)
    )
    if not isinstance(out, dict):
        return None
    components = out.get("components")
    if components is None:
        return None
    kernel_s = out.get("kernel_seconds")
    return components, (float(kernel_s) if kernel_s is not None else None)


def _cpp_host_connected_components_sets(prepared, G, directed):
    cpp_graph = _get_cached_cpp_graph(
        G,
        prepared=prepared,
        directed_override=bool(directed),
        host_mode=True,
    )
    if cpp_graph is None:
        return None
    import cpp_easygraph

    if directed:
        if not hasattr(cpp_easygraph, "cpp_strongly_connected_components"):
            return None
        fn = lambda: cpp_easygraph.cpp_strongly_connected_components(cpp_graph)
    else:
        if not hasattr(cpp_easygraph, "cpp_connected_components_undirected"):
            return None
        fn = lambda: cpp_easygraph.cpp_connected_components_undirected(cpp_graph)
    t0 = time.perf_counter()
    out = _temporarily_disable_gpu_env(fn)
    elapsed = time.perf_counter() - t0
    if out is None:
        return None
    return list(out), elapsed


def _cpp_gpu_multi_source_dijkstra(prepared, G, sources, weight, target):
    cpp_graph = _get_cached_cpp_graph(G, prepared=prepared)
    if cpp_graph is None:
        return None
    import cpp_easygraph

    source_list = list(sources)
    nodes = _index_to_node_list(prepared, G)

    if hasattr(cpp_easygraph, "cpp_gpu_dijkstra_multisource_dense"):
        out = cpp_easygraph.cpp_gpu_dijkstra_multisource_dense(
            cpp_graph,
            source_list,
            weight=weight,
            target=target,
        )
        if isinstance(out, dict):
            dense = out.get("values_dense")
            kernel_s = out.get("kernel_seconds")
            if dense is not None:
                result = _DenseMultiSourceSSSPDict(
                    source_list,
                    nodes,
                    dense,
                    node_to_idx=prepared.get("node_to_idx"),
                )
                return result, (float(kernel_s) if kernel_s is not None else None)

    if not hasattr(cpp_easygraph, "cpp_dijkstra_multisource"):
        return None
    t0 = time.perf_counter()
    out = cpp_easygraph.cpp_dijkstra_multisource(
        cpp_graph,
        source_list,
        weight=weight,
        target=target,
    )
    if out is None:
        return None
    if isinstance(out, dict):
        kernel_s = out.get("kernel_seconds")
        out = out.get("values")
    else:
        kernel_s = time.perf_counter() - t0
    if kernel_s is not None:
        kernel_s = float(kernel_s)
    if isinstance(out, dict):
        return out, kernel_s
    try:
        import numpy as np

        arr = np.asarray(out, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[0] != len(source_list):
            return None
        if arr.shape[1] == len(nodes) + 1:
            arr = arr[:, 1:]
        result = _DenseMultiSourceSSSPDict(
            source_list,
            nodes,
            arr,
            node_to_idx=prepared.get("node_to_idx"),
        )
        return result, kernel_s
    except Exception:
        rows = out.tolist() if hasattr(out, "tolist") else list(out)
        if len(rows) != len(source_list):
            return None
        result = {}
        for source, row in zip(source_list, rows):
            dist = {}
            values = row.tolist() if hasattr(row, "tolist") else list(row)
            if len(values) == len(nodes) + 1:
                values = values[1:]
            if len(values) < len(nodes):
                values = values + [math.inf] * (len(nodes) - len(values))
            for idx, v in enumerate(values[: len(nodes)]):
                try:
                    dv = float(v)
                except Exception:
                    continue
                if math.isfinite(dv) and abs(dv) < 1.0e30:
                    dist[nodes[idx]] = dv
            result[source] = dist
        return result, kernel_s


def _cpp_gpu_multi_source_bellman_ford(prepared, G, sources, weight, target):
    cpp_graph = _get_cached_cpp_graph(G, prepared=prepared)
    if cpp_graph is None:
        return None
    import cpp_easygraph

    if not hasattr(cpp_easygraph, "cpp_gpu_bellman_ford_multisource_dense"):
        return None

    source_list = list(sources)
    nodes = _index_to_node_list(prepared, G)
    out = cpp_easygraph.cpp_gpu_bellman_ford_multisource_dense(
        cpp_graph,
        source_list,
        weight=weight,
        target=target,
    )
    if not isinstance(out, dict):
        return None
    dense = out.get("values_dense")
    kernel_s = out.get("kernel_seconds")
    if dense is None:
        return None
    result = _DenseMultiSourceSSSPDict(
        source_list,
        nodes,
        dense,
        node_to_idx=prepared.get("node_to_idx"),
    )
    return result, (float(kernel_s) if kernel_s is not None else None)


def _cpp_host_multi_source_dijkstra(prepared, G, sources, weight, target):
    cpp_graph = _get_cached_cpp_graph(G, prepared=prepared, host_mode=True)
    if cpp_graph is None:
        return None
    import cpp_easygraph

    if not hasattr(cpp_easygraph, "cpp_dijkstra_multisource"):
        return None

    source_list = list(sources)
    nodes = _index_to_node_list(prepared, G)
    t0 = time.perf_counter()
    out = _temporarily_disable_gpu_env(
        lambda: cpp_easygraph.cpp_dijkstra_multisource(
            cpp_graph,
            source_list,
            weight=weight,
            target=target,
        )
    )
    kernel_s = time.perf_counter() - t0
    if out is None:
        return None
    if isinstance(out, dict):
        reported_kernel = out.get("kernel_seconds")
        out = out.get("values")
        if reported_kernel is not None:
            kernel_s = float(reported_kernel)
    if isinstance(out, dict):
        return out, kernel_s

    try:
        import numpy as np

        arr = np.asarray(out, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[0] != len(source_list):
            return None
        if arr.shape[1] == len(nodes) + 1:
            arr = arr[:, 1:]
        result = _DenseMultiSourceSSSPDict(
            source_list,
            nodes,
            arr,
            node_to_idx=prepared.get("node_to_idx"),
        )
        return result, kernel_s
    except Exception:
        rows = out.tolist() if hasattr(out, "tolist") else list(out)
        if len(rows) != len(source_list):
            return None
        result = {}
        for source, row in zip(source_list, rows):
            dist = {}
            values = row.tolist() if hasattr(row, "tolist") else list(row)
            if len(values) == len(nodes) + 1:
                values = values[1:]
            if len(values) < len(nodes):
                values = values + [math.inf] * (len(nodes) - len(values))
            for idx, v in enumerate(values[: len(nodes)]):
                try:
                    dv = float(v)
                except Exception:
                    continue
                if math.isfinite(dv) and abs(dv) < 1.0e30:
                    dist[nodes[idx]] = dv
            result[source] = dist
        return result, kernel_s


def _cpp_gpu_k_core(prepared, G):
    cpp_graph = _get_cached_cpp_graph(G, prepared=prepared)
    if cpp_graph is None:
        return None
    import cpp_easygraph

    if hasattr(cpp_easygraph, "cpp_gpu_k_core"):
        out = cpp_easygraph.cpp_gpu_k_core(cpp_graph)
        if isinstance(out, dict):
            dense = out.get("values_dense")
            if dense is not None:
                nodes = prepared.get("nodes", [])
                try:
                    raw_len = len(dense)
                except Exception:
                    raw_len = None
                arr = (
                    dense
                    if raw_len == len(nodes)
                    else _normalize_dense_array(dense, len(nodes), dtype="int32", fill_value=0)
                )
                kernel_s = out.get("kernel_seconds")
                return arr, (float(kernel_s) if kernel_s is not None else None)

    if not hasattr(cpp_easygraph, "cpp_k_core"):
        return None
    t0 = time.perf_counter()
    out = cpp_easygraph.cpp_k_core(cpp_graph)
    kernel_s = time.perf_counter() - t0
    if out is None:
        return None
    nodes = prepared.get("nodes", [])
    values = _normalize_dense_vector(out, len(nodes))
    normalized = [int(v) for v in values]
    return normalized, kernel_s


def _cpp_host_k_core(prepared, G):
    cpp_graph = _get_cached_cpp_graph(G, prepared=prepared, host_mode=True)
    if cpp_graph is None:
        return None
    import cpp_easygraph

    fn = getattr(cpp_easygraph, "cpp_cpu_k_core", None)
    if fn is None:
        fn = getattr(cpp_easygraph, "cpp_k_core", None)
    if fn is None:
        return None
    t0 = time.perf_counter()
    if getattr(cpp_easygraph, "cpp_cpu_k_core", None) is not None:
        out = fn(cpp_graph)
    else:
        out = _temporarily_disable_gpu_env(lambda: fn(cpp_graph))
    elapsed = time.perf_counter() - t0
    if out is None:
        return None
    nodes = prepared.get("nodes", [])
    values = _normalize_dense_array(out, len(nodes), dtype="int64", fill_value=0)
    return values, elapsed


def _cpp_gpu_betweenness(prepared, G, weight, sources, normalized, endpoints):
    cpp_graph = _get_cached_cpp_graph(G, prepared=prepared)
    if cpp_graph is None:
        return None
    import cpp_easygraph

    if not hasattr(cpp_easygraph, "cpp_betweenness_centrality"):
        return None
    t0 = time.perf_counter()
    out = cpp_easygraph.cpp_betweenness_centrality(
        cpp_graph,
        weight=weight,
        cutoff=None,
        sources=sources,
        normalized=bool(normalized),
        endpoints=bool(endpoints),
    )
    kernel_s = time.perf_counter() - t0
    if out is None:
        return None
    if isinstance(out, dict):
        kernel_s = float(out.get("kernel_seconds", kernel_s))
        out = out.get("values_dense")
        if out is None:
            return None
    nodes = prepared.get("nodes", [])
    values = _normalize_dense_array(out, len(nodes), dtype="float64", fill_value=0.0)
    return values, kernel_s


def _cpp_gpu_closeness(prepared, G, weight, sources):
    cpp_graph = _get_cached_cpp_graph(G, prepared=prepared)
    if cpp_graph is None:
        return None
    import cpp_easygraph

    if not hasattr(cpp_easygraph, "cpp_closeness_centrality"):
        return None
    t0 = time.perf_counter()
    out = cpp_easygraph.cpp_closeness_centrality(
        cpp_graph,
        weight=weight,
        cutoff=None,
        sources=sources,
    )
    kernel_s = time.perf_counter() - t0
    if out is None:
        return None
    if isinstance(out, dict):
        kernel_s = float(out.get("kernel_seconds", kernel_s))
        out = out.get("values_dense")
        if out is None:
            return None
    if sources is None:
        target_len = len(prepared.get("nodes", []))
    else:
        try:
            target_len = len(sources)
        except TypeError:
            target_len = len(tuple(sources))
    values = _normalize_dense_array(out, target_len, dtype="float64", fill_value=0.0)
    return values, kernel_s


def _adj_dict(G):
    if hasattr(G, "_adj"):
        return G._adj
    return G.adj


def _mix64(h, x):
    h ^= (int(x) + 0x9E3779B97F4A7C15 + ((h << 6) & _HASH_MASK) + (h >> 2)) & _HASH_MASK
    return h & _HASH_MASK


def _graph_stamp(G):
    """Fast graph fingerprint for cache validation.

    This is O(|V|) and intentionally avoids scanning all edges.
    """
    adj = _adj_dict(G)
    directed = bool(G.is_directed())
    node_count = len(adj)
    edge_slots = 0
    h = 1469598103934665603

    for i, (u, nbrs) in enumerate(adj.items()):
        deg = len(nbrs)
        edge_slots += deg
        h = _mix64(h, hash(u))
        h = _mix64(h, deg)
        if i < 32:
            j = 0
            for v in nbrs:
                h = _mix64(h, hash(v))
                j += 1
                if j >= 4:
                    break
    return (directed, node_count, edge_slots, h)


def _ensure_cache_root():
    _CACHE_ROOT.mkdir(parents=True, exist_ok=True)


def _cleanup_cache_dirs():
    try:
        dirs = [p for p in _CACHE_ROOT.iterdir() if p.is_dir()]
    except FileNotFoundError:
        return
    if len(dirs) <= _MAX_CACHE_DIRS:
        return
    dirs.sort(key=lambda p: p.stat().st_mtime)
    for p in dirs[: max(0, len(dirs) - _MAX_CACHE_DIRS)]:
        try:
            shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass


def _build_graph_dir_name(G, stamp):
    directed, node_count, edge_slots, sig = stamp
    tag = "d" if directed else "u"
    return f"g{id(G):x}_{tag}_n{node_count}_e{edge_slots}_{sig:016x}"


def _iter_directed_edges_with_data(G):
    for u, nbrs in _adj_dict(G).items():
        for v, data in nbrs.items():
            yield u, v, data


def _prepared_graph_usable(G, prepared):
    if not isinstance(prepared, dict):
        return False
    if bool(G.is_directed()) != bool(prepared.get("directed")):
        return False
    adj = _adj_dict(G)
    if len(adj) != int(prepared.get("node_count", -1)):
        return False
    if not _validate_sentinels(G, prepared):
        return False
    reuse_hits = int(prepared.get("reuse_hits", 0)) + 1
    prepared["reuse_hits"] = reuse_hits
    if reuse_hits % _GRAPH_CTX_VALIDATE_EVERY != 0:
        return True
    return prepared.get("stamp") == _graph_stamp(G)


def _prepare_graph_cache(G):
    cache = None if _GRAPH_CONTEXT_CACHE_DISABLED else _graph_cache_dict(G)
    prepared = cache.get(_PREPARED_CACHE_KEY) if cache is not None else None
    if prepared is not None and _prepared_graph_usable(G, prepared):
        return prepared

    stamp = _graph_stamp(G)
    nodes = list(G.nodes.keys()) if hasattr(G.nodes, "keys") else list(G.nodes)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    directed = bool(G.is_directed())
    graph_stats = (
        adaptive_policy.build_graph_stats(G, _adj_dict, stamp)
        if adaptive_policy.policy_enabled()
        else None
    )

    _ensure_cache_root()
    graph_dir = _CACHE_ROOT / _build_graph_dir_name(G, stamp)
    graph_dir.mkdir(parents=True, exist_ok=True)

    directed_path = graph_dir / "edges_directed.txt"
    undirected_path = graph_dir / "edges_undirected.txt"

    prepared = {
        "ctx_id": time.time_ns(),
        "stamp": stamp,
        "node_count": len(nodes),
        "graph_stats": graph_stats,
        "graph_dir": graph_dir,
        "directed": directed,
        "nodes": nodes,
        "node_to_idx": node_to_idx,
        "directed_path": directed_path,
        "undirected_path": undirected_path,
        "directed_ready": False,
        "undirected_ready": False,
        "weighted_paths": {},
        "result_cache": {},
        "result_cache_order": [],
        "sentinel_directed": [],
        "sentinel_undirected": [],
        "reuse_hits": 0,
    }
    if not directed:
        # Undirected graph can reuse a single unique-edge file for both modes.
        prepared["directed_path"] = undirected_path
    if cache is not None:
        cache[_PREPARED_CACHE_KEY] = prepared
    _cleanup_cache_dirs()
    return prepared


def _graph_context(G, prewarm_cpp=False):
    prepared = _prepare_graph_cache(G)
    if prewarm_cpp:
        try:
            _get_cached_cpp_graph(G, prepared=prepared)
        except Exception:
            pass
    return prepared


def _prepared_files_ready(prepared):
    directed = bool(prepared.get("directed"))
    dpath = prepared.get("directed_path")
    upath = prepared.get("undirected_path")
    if directed:
        return bool(prepared.get("directed_ready")) and dpath is not None and Path(dpath).exists()
    return bool(prepared.get("undirected_ready")) and upath is not None and Path(upath).exists()


def _validate_sentinels(G, prepared):
    adj = _adj_dict(G)
    if bool(G.is_directed()) != bool(prepared.get("directed")):
        return False

    directed = bool(prepared.get("directed"))
    sentinels = prepared.get("sentinel_directed" if directed else "sentinel_undirected", [])
    for u, v in sentinels:
        if u not in adj:
            return False
        if directed:
            if v not in adj[u]:
                return False
        else:
            ok = v in adj[u]
            if not ok and v in adj and u in adj[v]:
                ok = True
            if not ok:
                return False
    return True


def _ensure_directed_edge_file(prepared, G):
    if not bool(prepared.get("directed")):
        return _ensure_undirected_edge_file(prepared, G)
    p = Path(prepared["directed_path"])
    if prepared.get("directed_ready") and p.exists():
        return p
    sent = []
    node_to_idx = prepared["node_to_idx"]
    with p.open("w") as f:
        for u, v, _ in _iter_directed_edges_with_data(G):
            iu = node_to_idx[u]
            iv = node_to_idx[v]
            if iu == iv:
                continue
            f.write(f"{iu} {iv}\n")
            if len(sent) < 16:
                sent.append((u, v))
    prepared["sentinel_directed"] = sent
    prepared["directed_ready"] = True
    return p


def _ensure_undirected_edge_file(prepared, G):
    p = Path(prepared["undirected_path"])
    if prepared.get("undirected_ready") and p.exists():
        return p
    node_to_idx = prepared["node_to_idx"]
    sent = []
    if not bool(prepared.get("directed")):
        # EasyGraph undirected graph stores each edge twice; keep iu < iv once.
        with p.open("w") as f:
            for u, v, _ in _iter_directed_edges_with_data(G):
                iu = node_to_idx[u]
                iv = node_to_idx[v]
                if iu == iv:
                    continue
                if iu < iv:
                    f.write(f"{iu} {iv}\n")
                    if len(sent) < 16:
                        sent.append((u, v))
        prepared["sentinel_directed"] = sent
        prepared["directed_ready"] = True
    else:
        # Directed graph fallback: lazily materialize an undirected projection.
        seen = set()
        with p.open("w") as f:
            for u, v, _ in _iter_directed_edges_with_data(G):
                iu = node_to_idx[u]
                iv = node_to_idx[v]
                if iu == iv:
                    continue
                if iu > iv:
                    iu, iv = iv, iu
                key = (iu, iv)
                if key in seen:
                    continue
                seen.add(key)
                f.write(f"{iu} {iv}\n")
                if len(sent) < 16:
                    sent.append((u, v))
    prepared["sentinel_undirected"] = sent
    prepared["undirected_ready"] = True
    return p


def _edge_path_for_mode(prepared, G, directed_mode):
    if directed_mode:
        return _ensure_directed_edge_file(prepared, G)
    return _ensure_undirected_edge_file(prepared, G)


def _ensure_undirected_projection_file(prepared, G):
    return _ensure_undirected_edge_file(prepared, G)


def _weight_from_graph(G, u, v, weight_key):
    data = None
    adj = _adj_dict(G)
    if u in adj:
        data = adj[u].get(v)
    if data is None and v in adj:
        data = adj[v].get(u)
    if isinstance(data, dict):
        return float(data.get(weight_key, 1))
    return 1.0


def _ensure_weighted_file(prepared, G, weight_key):
    weighted_paths = prepared.get("weighted_paths", {})
    if weight_key in weighted_paths and Path(weighted_paths[weight_key]).exists():
        return Path(weighted_paths[weight_key])

    node_to_idx = prepared["node_to_idx"]
    graph_dir = Path(prepared["graph_dir"])
    out = graph_dir / f"mst_weighted_{weight_key}.txt"

    if not bool(prepared["directed"]):
        with out.open("w") as f:
            for u, v, data in _iter_directed_edges_with_data(G):
                iu = node_to_idx[u]
                iv = node_to_idx[v]
                if iu == iv:
                    continue
                if iu < iv:
                    w = data.get(weight_key, 1) if isinstance(data, dict) else 1
                    w = float(w)
                    if float(w).is_integer():
                        f.write(f"{iu} {iv} {int(w)}\n")
                    else:
                        f.write(f"{iu} {iv} {w:.12g}\n")
    else:
        # Conservative projection path for directed inputs: keep minimal weight.
        weights = {}
        for u, v, data in _iter_directed_edges_with_data(G):
            iu = node_to_idx[u]
            iv = node_to_idx[v]
            if iu == iv:
                continue
            if iu > iv:
                iu, iv = iv, iu
            w = data.get(weight_key, 1) if isinstance(data, dict) else 1
            w = float(w)
            old = weights.get((iu, iv))
            if old is None or w < old:
                weights[(iu, iv)] = w
        with out.open("w") as f:
            for (iu, iv), w in weights.items():
                if float(w).is_integer():
                    f.write(f"{iu} {iv} {int(w)}\n")
                else:
                    f.write(f"{iu} {iv} {w:.12g}\n")

    weighted_paths[weight_key] = out
    prepared["weighted_paths"] = weighted_paths
    return out


def _safe_tmpdir(prefix):
    return tempfile.TemporaryDirectory(prefix=prefix)


def _cache_key(name, *parts):
    out = [name]
    for p in parts:
        if isinstance(p, float):
            out.append(round(float(p), 12))
        else:
            out.append(p)
    return tuple(out)


def _allow_result_cache(prepared):
    if not _RESULT_CACHE_ENABLED:
        return False
    nodes = prepared.get("nodes")
    if nodes is None:
        return False
    if _RESULT_CACHE_MAX_NODES > 0 and len(nodes) > _RESULT_CACHE_MAX_NODES:
        return False
    return True


def _copy_for_return(obj):
    if not _RESULT_CACHE_RETURN_COPY:
        return obj
    if isinstance(obj, dict):
        return dict(obj)
    if isinstance(obj, list):
        return list(obj)
    return obj


def _result_cache_get(prepared, key):
    cache = prepared.get("result_cache")
    if not isinstance(cache, dict):
        return None
    entry = cache.get(key)
    if entry is None:
        return None
    return _copy_for_return(entry.get("result")), entry.get("kernel")


def _result_cache_put(prepared, key, result, kernel):
    if not _allow_result_cache(prepared):
        return
    cache = prepared.get("result_cache")
    if not isinstance(cache, dict):
        cache = {}
        prepared["result_cache"] = cache
    order = prepared.get("result_cache_order")
    if not isinstance(order, list):
        order = []
        prepared["result_cache_order"] = order
    cache[key] = {"result": result, "kernel": kernel}
    if key in order:
        order.remove(key)
    order.append(key)
    while len(order) > _RESULT_CACHE_MAX_ITEMS:
        old = order.pop(0)
        cache.pop(old, None)


def pagerank(G, alpha=0.85, max_iter=200, eps=1e-6, weight=None):
    prepared = _graph_context(G)
    cache_key = _cache_key(
        "pagerank",
        bool(prepared["directed"]),
        alpha,
        int(max_iter),
        eps,
        str(weight),
    )
    cached = _result_cache_get(prepared, cache_key)
    if cached is not None:
        result, kernel_s = cached
        set_last_kernel_time("pagerank", kernel_s)
        return result

    nodes = prepared["nodes"]
    if not nodes:
        set_last_kernel_time("pagerank", 0.0)
        return {}

    try:
        cpp_hit = _cpp_gpu_pagerank(prepared, G, alpha=alpha, max_iter=max_iter, eps=eps, weight=weight)
        if cpp_hit is not None:
            result, kernel_s = cpp_hit
            set_last_kernel_time("pagerank", kernel_s)
            _result_cache_put(prepared, cache_key, result, kernel_s)
            return result
    except Exception:
        pass

    if weight is not None:
        raise RuntimeError("weighted PageRank fallback binary unavailable; cpp gpu path failed")

    exe = _resolve_bin("EASYGRAPH_GPU_PR_BIN", ["pagerank_v3_local", "pagerank_v3"])
    if exe is None:
        raise RuntimeError("pagerank backend unavailable: cpp gpu path failed and pagerank binary not found")

    edge_path = _edge_path_for_mode(prepared, G, directed_mode=bool(prepared["directed"]))
    if edge_path.stat().st_size == 0:
        uniform = 1.0 / len(nodes)
        result = {node: uniform for node in nodes}
        set_last_kernel_time("pagerank", 0.0)
        return result

    with _safe_tmpdir("eg_pr_") as td:
        dump_path = Path(td) / f"pagerank_{time.time_ns()}.tsv"
        cmd = [
            exe,
            edge_path,
            "--alpha",
            str(alpha),
            "--eps",
            str(eps),
            "--max-iter",
            str(max_iter),
            "--warmup",
            "0",
            "--dump",
            dump_path,
        ]
        if not prepared["directed"]:
            cmd.append("--undirected")
        stdout = _run_cmd(cmd)
        kernel_s = _parse_first_float(r"Iter time:\s*([0-9.eE+-]+)\s*s", stdout)
        set_last_kernel_time("pagerank", kernel_s)

        result = {node: 0.0 for node in nodes}
        with dump_path.open("r") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                p = s.split()
                if len(p) < 2:
                    continue
                idx = int(p[0])
                if 0 <= idx < len(nodes):
                    result[nodes[idx]] = float(p[1])
        total = sum(result.values())
        if total > 0:
            inv = 1.0 / total
            for k in result:
                result[k] *= inv
        _result_cache_put(prepared, cache_key, result, kernel_s)
        return result


def minimum_spanning_tree_edges(G, weight="weight"):
    prepared = _graph_context(G)
    cache_key = _cache_key("mst_edges", bool(prepared["directed"]), str(weight))
    cached = _result_cache_get(prepared, cache_key)
    if cached is not None:
        result, kernel_s = cached
        set_last_kernel_time("mst", kernel_s)
        return result

    nodes = prepared["nodes"]
    if not nodes:
        set_last_kernel_time("mst", 0.0)
        return []

    try:
        cpp_hit = _cpp_gpu_mst_edges(prepared, G, weight=weight)
        if cpp_hit is not None:
            edges, kernel_s = cpp_hit
            set_last_kernel_time("mst", kernel_s)
            _result_cache_put(prepared, cache_key, edges, kernel_s)
            return edges
    except Exception:
        pass

    exe = _resolve_bin("EASYGRAPH_GPU_MST_BIN", ["mst_v2_local", "mst_v2"])
    if exe is None:
        raise RuntimeError("mst backend unavailable: cpp gpu path failed and mst binary not found")

    edge_file = _ensure_weighted_file(prepared, G, weight)
    if edge_file.stat().st_size == 0:
        set_last_kernel_time("mst", 0.0)
        return []

    with _safe_tmpdir("eg_mst_") as td:
        out_file = Path(td) / f"result_{edge_file.name}"
        cmd = [exe, edge_file, "--skip-cpu"]
        stdout = _run_cmd(cmd, cwd=td)
        kernel_s = _parse_first_float(r"Device GPU runtime:\s*([0-9.eE+-]+)\s*s", stdout)
        set_last_kernel_time("mst", kernel_s)
        if not out_file.exists():
            return []

        edge_list = []
        with out_file.open("r") as f:
            for line in f:
                s = line.strip().strip("()")
                if not s:
                    continue
                p = [x.strip() for x in s.split(",")]
                if len(p) < 2:
                    continue
                iu = int(p[0])
                iv = int(p[1])
                if iu > iv:
                    iu, iv = iv, iu
                edge_list.append((iu, iv))

    out = []
    for iu, iv in edge_list:
        if iu < 0 or iv < 0 or iu >= len(nodes) or iv >= len(nodes):
            continue
        u = nodes[iu]
        v = nodes[iv]
        w = _weight_from_graph(G, u, v, weight)
        out.append((u, v, float(w)))
    _result_cache_put(prepared, cache_key, out, kernel_s)
    return out


def minimum_spanning_tree(G, weight="weight"):
    prepared = _graph_context(G)
    cache_key = _cache_key("mst_tree", bool(prepared["directed"]), str(weight))
    cached = _result_cache_get(prepared, cache_key)
    if cached is not None:
        result, kernel_s = cached
        set_last_kernel_time("mst", kernel_s)
        return result

    try:
        cpp_hit = _cpp_gpu_mst_tree(prepared, G, weight=weight)
        if cpp_hit is not None:
            tree, kernel_s = cpp_hit
            if tree is not None:
                set_last_kernel_time("mst", kernel_s)
                _result_cache_put(prepared, cache_key, tree, kernel_s)
                return tree
    except Exception:
        pass

    edges = minimum_spanning_tree_edges(G, weight=weight)
    kernel_s = get_last_kernel_time("mst")
    try:
        import easygraph as eg

        T = eg.Graph()
        nodes = prepared.get("nodes", [])
        T._adj = {n: {} for n in nodes}
        T._node = {n: {} for n in nodes}
        T._node_index = {n: i for i, n in enumerate(nodes)}
        T._id = len(nodes)
        adj = T._adj
        for u, v, w in edges:
            attr = {weight: w}
            adj[u][v] = attr
            adj[v][u] = attr
        T.cache = {}
    except Exception:
        T = G.__class__()
        if edges:
            T.add_weighted_edges_from(edges, weight=weight)
        if len(T) < len(G):
            T.add_nodes_from(node for node in G.nodes if node not in T)
    _result_cache_put(prepared, cache_key, T, kernel_s)
    return T


def clustering(G):
    prepared = _graph_context(G)
    cache_key = _cache_key("lcc", bool(prepared["directed"]))
    cached = _result_cache_get(prepared, cache_key)
    if cached is not None:
        values, kernel_s = cached
        set_last_kernel_time("lcc", kernel_s)
        return values

    nodes = prepared["nodes"]
    values = {node: 0.0 for node in nodes}
    if not nodes:
        set_last_kernel_time("lcc", 0.0)
        return values

    try:
        cpp_hit = _cpp_gpu_clustering(prepared, G)
        if cpp_hit is not None:
            values, kernel_s = cpp_hit
            set_last_kernel_time("lcc", kernel_s)
            _result_cache_put(prepared, cache_key, values, kernel_s)
            return values
    except Exception:
        pass

    exe = _resolve_bin("EASYGRAPH_GPU_LCC_BIN", ["cc_gpu_local", "cc_gpu"])
    if exe is None:
        raise RuntimeError(
            "lcc backend unavailable: cpp gpu path failed and lcc binary not found "
            "(set EASYGRAPH_GPU_LCC_BIN or EASYGRAPH_GPU_BIN_DIR)"
        )

    directed_mode = bool(prepared["directed"])
    edge_file = _edge_path_for_mode(prepared, G, directed_mode=directed_mode)
    if edge_file.stat().st_size == 0:
        set_last_kernel_time("lcc", 0.0)
        return values

    with _safe_tmpdir("eg_lcc_") as td:
        dump_file = Path(td) / f"lcc_{time.time_ns()}.tsv"
        gtype = "directed" if directed_mode else "undirected"
        cmd = [exe, edge_file, gtype, "--trust-cleaned", "--dump", dump_file]
        stdout = _run_cmd(cmd, cwd=td)
        kernel_s = _parse_first_float(r"GPU compute time:\s*([0-9.eE+-]+)\s*s", stdout)
        set_last_kernel_time("lcc", kernel_s)
        with dump_file.open("r") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                p = s.split()
                if len(p) < 2:
                    continue
                idx = int(p[0])
                if 0 <= idx < len(nodes):
                    values[nodes[idx]] = float(p[1])
    _result_cache_put(prepared, cache_key, values, kernel_s)
    return values


def connected_component_labels(G, directed):
    prepared = _graph_context(G)
    cache_key = _cache_key("components", bool(directed))
    cached = _result_cache_get(prepared, cache_key)
    if cached is not None:
        labels, kernel_s = cached
        set_last_kernel_time("scc" if directed else "cc", kernel_s)
        return labels

    nodes = prepared["nodes"]
    labels = {node: -1 for node in nodes}
    if not nodes:
        set_last_kernel_time("scc" if directed else "cc", 0.0)
        return labels

    try:
        cpp_hit = _cpp_gpu_connected_component_labels(prepared, G, directed=bool(directed))
        if cpp_hit is not None:
            labels, kernel_s = cpp_hit
            set_last_kernel_time("scc" if directed else "cc", kernel_s)
            _result_cache_put(prepared, cache_key, labels, kernel_s)
            return labels
    except Exception:
        pass

    exe = _resolve_bin("EASYGRAPH_GPU_CC_BIN", ["gpu_graph_scc_v2_local", "gpu_graph_scc_v2"])
    if exe is None:
        raise RuntimeError(
            "cc/scc backend unavailable: cpp gpu path failed and cc/scc binary not found "
            "(set EASYGRAPH_GPU_CC_BIN or EASYGRAPH_GPU_BIN_DIR)"
        )

    if directed:
        edge_file = _edge_path_for_mode(prepared, G, directed_mode=True)
        gtype = "directed"
    else:
        edge_file = _ensure_undirected_projection_file(prepared, G)
        gtype = "undirected"

    if edge_file.stat().st_size == 0:
        for i, node in enumerate(nodes):
            labels[node] = i
        set_last_kernel_time("scc" if directed else "cc", 0.0)
        return labels

    with _safe_tmpdir("eg_cc_") as td:
        dump_file = Path(td) / f"gpu_comp_{time.time_ns()}.tsv"
        cmd = [
            exe,
            edge_file,
            gtype,
            "--skip-cpu",
            "--warmup",
            "0",
            "--repeat",
            "1",
            "--dump-gpu",
            dump_file,
            "--no-small-write",
            "--quiet",
        ]
        stdout = _run_cmd(cmd, cwd=td)
        kernel_s = _parse_first_float(r"GPU Time:\s*([0-9.eE+-]+)\s*s", stdout)
        set_last_kernel_time("scc" if directed else "cc", kernel_s)
        with dump_file.open("r") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                p = s.split()
                if len(p) < 2:
                    continue
                idx = int(p[0])
                if 0 <= idx < len(nodes):
                    labels[nodes[idx]] = int(p[1])

    # Assign singleton labels for nodes absent from edge rows.
    next_label = max(labels.values()) + 1 if labels else 0
    for node in nodes:
        if labels[node] < 0:
            labels[node] = next_label
            next_label += 1
    _result_cache_put(prepared, cache_key, labels, kernel_s)
    return labels


def connected_components(G, directed=False):
    prepared = _graph_context(G)
    cache_key = _cache_key("component_sets", bool(directed))
    cached = _result_cache_get(prepared, cache_key)
    kernel_key = "scc" if directed else "cc"
    if cached is not None:
        groups, kernel_s = cached
        set_last_kernel_time(kernel_key, kernel_s)
        return groups

    if (
        directed
        and adaptive_policy.prefer_scc_host(prepared)
    ):
        try:
            host_hit = _cpp_host_connected_components_sets(
                prepared,
                G,
                directed=True,
            )
        except Exception:
            host_hit = None
        if host_hit is not None:
            out, kernel_s = host_hit
            set_last_kernel_time(kernel_key, kernel_s)
            _result_cache_put(prepared, cache_key, out, kernel_s)
            return out

    dense_kernel_s = None
    if adaptive_policy.prefer_dense_component_return(prepared, directed):
        try:
            dense_hit = _cpp_gpu_connected_component_labels_dense(
                prepared,
                G,
                directed=bool(directed),
            )
        except Exception:
            dense_hit = None
        if dense_hit is not None:
            dense_labels, dense_kernel_s = dense_hit
            out = _components_from_dense_labels(prepared, dense_labels)
            set_last_kernel_time(kernel_key, dense_kernel_s)
            _result_cache_put(prepared, cache_key, out, dense_kernel_s)
            return out

    try:
        set_hit = _cpp_gpu_connected_components_sets(
            prepared,
            G,
            directed=bool(directed),
        )
    except Exception:
        set_hit = None
    if set_hit is not None:
        out, kernel_s = set_hit
        set_last_kernel_time(kernel_key, kernel_s)
        _result_cache_put(prepared, cache_key, out, kernel_s)
        return out

    try:
        dense_hit = _cpp_gpu_connected_component_labels_dense(
            prepared,
            G,
            directed=bool(directed),
        )
    except Exception:
        dense_hit = None
    if dense_hit is not None:
        dense_labels, dense_kernel_s = dense_hit
        out = _components_from_dense_labels(prepared, dense_labels)
        set_last_kernel_time(kernel_key, dense_kernel_s)
        _result_cache_put(prepared, cache_key, out, dense_kernel_s)
        return out

    labels = connected_component_labels(G, directed=bool(directed))
    groups = {}
    for node, lab in labels.items():
        groups.setdefault(int(lab), set()).add(node)
    out = list(groups.values())
    kernel_s = dense_kernel_s if dense_kernel_s is not None else get_last_kernel_time(kernel_key)
    _result_cache_put(prepared, cache_key, out, kernel_s)
    return out


def multi_source_dijkstra(G, sources, weight="weight", target=None):
    prepared = _graph_context(G)
    source_list = list(sources)
    cache_key = _cache_key(
        "sssp",
        bool(prepared["directed"]),
        tuple(source_list),
        str(weight),
        target,
    )
    cached = _result_cache_get(prepared, cache_key)
    if cached is not None:
        result, kernel_s = cached
        set_last_kernel_time("sssp", kernel_s)
        set_last_kernel_time("bfs" if weight is None else "dijkstra", kernel_s)
        return result

    if not source_list:
        set_last_kernel_time("sssp", 0.0)
        set_last_kernel_time("bfs" if weight is None else "dijkstra", 0.0)
        return {}

    # The host Dijkstra fallback is only a weighted shortest-path fast path.
    # Reusing it for BFS (weight=None) is unsafe in long-lived workflow runs
    # after GPU component kernels have initialized cached graph contexts.
    if weight is not None and adaptive_policy.prefer_sssp_host(prepared):
        try:
            host_hit = _cpp_host_multi_source_dijkstra(
                prepared,
                G,
                sources=source_list,
                weight=weight,
                target=target,
            )
        except Exception:
            host_hit = None
        if host_hit is not None:
            result, kernel_s = host_hit
            set_last_kernel_time("sssp", kernel_s)
            set_last_kernel_time("bfs" if weight is None else "dijkstra", kernel_s)
            _result_cache_put(prepared, cache_key, result, kernel_s)
            return result

    cpp_hit = _cpp_gpu_multi_source_dijkstra(
        prepared,
        G,
        sources=source_list,
        weight=weight,
        target=target,
    )
    if cpp_hit is None:
        raise RuntimeError("sssp backend unavailable: cpp_dijkstra_multisource not found")
    result, kernel_s = cpp_hit
    set_last_kernel_time("sssp", kernel_s)
    set_last_kernel_time("bfs" if weight is None else "dijkstra", kernel_s)
    _result_cache_put(prepared, cache_key, result, kernel_s)
    return result


def single_source_dijkstra(G, source, weight="weight", target=None):
    result = multi_source_dijkstra(G, [source], weight=weight, target=target)[source]
    set_last_kernel_time("dijkstra", get_last_kernel_time("sssp"))
    return result


def multi_source_bfs(G, sources, target=None):
    result = multi_source_dijkstra(G, sources, weight=None, target=target)
    set_last_kernel_time("bfs", get_last_kernel_time("sssp"))
    return result


def single_source_bfs(G, source, target=None):
    result = multi_source_bfs(G, [source], target=target)[source]
    set_last_kernel_time("bfs", get_last_kernel_time("sssp"))
    return result


def multi_source_bellman_ford(G, sources, weight="weight", target=None):
    prepared = _graph_context(G)
    source_list = list(sources)
    cache_key = _cache_key(
        "bellman_ford",
        bool(prepared["directed"]),
        tuple(source_list),
        str(weight),
        target,
    )
    cached = _result_cache_get(prepared, cache_key)
    if cached is not None:
        result, kernel_s = cached
        set_last_kernel_time("bellman_ford", kernel_s)
        return result

    if not source_list:
        set_last_kernel_time("bellman_ford", 0.0)
        return {}

    cpp_hit = _cpp_gpu_multi_source_bellman_ford(
        prepared,
        G,
        sources=source_list,
        weight=weight,
        target=target,
    )
    if cpp_hit is None:
        raise RuntimeError("bellman_ford backend unavailable: cpp_gpu_bellman_ford_multisource_dense not found")
    result, kernel_s = cpp_hit
    set_last_kernel_time("bellman_ford", kernel_s)
    _result_cache_put(prepared, cache_key, result, kernel_s)
    return result


def single_source_bellman_ford(G, source, weight="weight", target=None):
    result = multi_source_bellman_ford(
        G,
        [source],
        weight=weight,
        target=target,
    )[source]
    set_last_kernel_time("bellman_ford", get_last_kernel_time("bellman_ford"))
    return result


def k_core(G):
    prepared = _graph_context(G)
    cache_key = _cache_key("kcore", bool(prepared["directed"]))
    cached = _result_cache_get(prepared, cache_key)
    if cached is not None:
        values, kernel_s = cached
        set_last_kernel_time("kcore", kernel_s)
        return values

    if (
        adaptive_policy.prefer_kcore_host(prepared)
    ):
        try:
            host_hit = _cpp_host_k_core(prepared, G)
        except Exception:
            host_hit = None
        if host_hit is not None:
            values, kernel_s = host_hit
            set_last_kernel_time("kcore", kernel_s)
            _result_cache_put(prepared, cache_key, values, kernel_s)
            return values

    cpp_hit = _cpp_gpu_k_core(prepared, G)
    if cpp_hit is None:
        raise RuntimeError("k-core backend unavailable: cpp_k_core not found")
    values, kernel_s = cpp_hit
    set_last_kernel_time("kcore", kernel_s)
    _result_cache_put(prepared, cache_key, values, kernel_s)
    return values


def betweenness_centrality(
    G,
    weight=None,
    sources=None,
    normalized=True,
    endpoints=False,
):
    prepared = _graph_context(G)
    source_tuple = None if sources is None else tuple(sources)
    cache_key = _cache_key(
        "bc",
        bool(prepared["directed"]),
        str(weight),
        source_tuple,
        bool(normalized),
        bool(endpoints),
    )
    cached = _result_cache_get(prepared, cache_key)
    if cached is not None:
        values, kernel_s = cached
        set_last_kernel_time("bc", kernel_s)
        return values

    cpp_hit = _cpp_gpu_betweenness(
        prepared,
        G,
        weight=weight,
        sources=sources,
        normalized=normalized,
        endpoints=endpoints,
    )
    if cpp_hit is None:
        raise RuntimeError("bc backend unavailable: cpp_betweenness_centrality not found")
    values, kernel_s = cpp_hit
    set_last_kernel_time("bc", kernel_s)
    _result_cache_put(prepared, cache_key, values, kernel_s)
    return values


def closeness_centrality(G, weight=None, sources=None):
    prepared = _graph_context(G)
    source_key = None if sources is None else tuple(sources)
    source_arg = None if source_key is None else source_key
    cache_key = _cache_key(
        "closeness",
        bool(prepared["directed"]),
        str(weight),
        source_key,
    )
    cached = _result_cache_get(prepared, cache_key)
    if cached is not None:
        values, kernel_s = cached
        set_last_kernel_time("closeness", kernel_s)
        return values

    nodes = prepared.get("nodes", [])
    if not nodes or (source_key is not None and not source_key):
        set_last_kernel_time("closeness", 0.0)
        return []
    cpp_hit = _cpp_gpu_closeness(prepared, G, weight=weight, sources=source_arg)
    if cpp_hit is None:
        raise RuntimeError("closeness backend unavailable: cpp_closeness_centrality not found")
    values, kernel_s = cpp_hit
    set_last_kernel_time("closeness", kernel_s)
    _result_cache_put(prepared, cache_key, values, kernel_s)
    return values


def effective_size(G, nodes=None, weight=None):
    prepared = _graph_context(G)
    cpp_nodes_arg, selected_nodes, _ = _structural_nodes(prepared, nodes)
    cache_key = _cache_key(
        "structural_effective_size",
        bool(prepared["directed"]),
        None if cpp_nodes_arg is None else tuple(selected_nodes),
        str(weight),
    )
    cached = _result_cache_get(prepared, cache_key)
    if cached is not None:
        values, kernel_s = cached
        set_last_kernel_time("effective_size", kernel_s)
        return values

    cpp_hit = _cpp_gpu_structural_dense(prepared, G, "effective_size", nodes, weight)
    if cpp_hit is None:
        raise RuntimeError("effective_size backend unavailable: cpp_gpu_effective_size_dense not found")
    values, kernel_s = cpp_hit
    set_last_kernel_time("effective_size", kernel_s)
    _result_cache_put(prepared, cache_key, values, kernel_s)
    return values


def constraint(G, nodes=None, weight=None):
    prepared = _graph_context(G)
    cpp_nodes_arg, selected_nodes, _ = _structural_nodes(prepared, nodes)
    cache_key = _cache_key(
        "structural_constraint",
        bool(prepared["directed"]),
        None if cpp_nodes_arg is None else tuple(selected_nodes),
        str(weight),
    )
    cached = _result_cache_get(prepared, cache_key)
    if cached is not None:
        values, kernel_s = cached
        set_last_kernel_time("constraint", kernel_s)
        return values

    cpp_hit = _cpp_gpu_structural_dense(prepared, G, "constraint", nodes, weight)
    if cpp_hit is None:
        raise RuntimeError("constraint backend unavailable: cpp_gpu_constraint_dense not found")
    values, kernel_s = cpp_hit
    set_last_kernel_time("constraint", kernel_s)
    _result_cache_put(prepared, cache_key, values, kernel_s)
    return values


def hierarchy(G, nodes=None, weight=None):
    prepared = _graph_context(G)
    cpp_nodes_arg, selected_nodes, _ = _structural_nodes(prepared, nodes)
    cache_key = _cache_key(
        "structural_hierarchy",
        bool(prepared["directed"]),
        None if cpp_nodes_arg is None else tuple(selected_nodes),
        str(weight),
    )
    cached = _result_cache_get(prepared, cache_key)
    if cached is not None:
        values, kernel_s = cached
        set_last_kernel_time("hierarchy", kernel_s)
        return values

    cpp_hit = _cpp_gpu_structural_dense(prepared, G, "hierarchy", nodes, weight)
    if cpp_hit is None:
        raise RuntimeError("hierarchy backend unavailable: cpp_gpu_hierarchy_dense not found")
    values, kernel_s = cpp_hit
    set_last_kernel_time("hierarchy", kernel_s)
    _result_cache_put(prepared, cache_key, values, kernel_s)
    return values


def efficiency(G, nodes=None, weight=None):
    prepared = _graph_context(G)
    _, selected_nodes, selected_index = _structural_nodes(prepared, nodes)
    cache_key = _cache_key(
        "structural_efficiency",
        bool(prepared["directed"]),
        None if nodes is None else tuple(selected_nodes),
        str(weight),
    )
    cached = _result_cache_get(prepared, cache_key)
    if cached is not None:
        values, kernel_s = cached
        set_last_kernel_time("efficiency", kernel_s)
        return values

    e_size = effective_size(G, nodes=nodes, weight=weight)
    kernel_s = get_last_kernel_time("effective_size")
    import numpy as np

    es_values = np.asarray(e_size.values(), dtype=np.float64).reshape(-1)
    degree = np.asarray(_weighted_degree_values(G, selected_nodes, weight), dtype=np.float64)
    out = np.full(len(selected_nodes), np.nan, dtype=np.float64)
    valid = degree != 0
    out[valid] = es_values[valid] / degree[valid]
    values = _DenseValueDict(selected_nodes, out, node_to_idx=selected_index, dtype=float)
    set_last_kernel_time("efficiency", kernel_s)
    _result_cache_put(prepared, cache_key, values, kernel_s)
    return values
